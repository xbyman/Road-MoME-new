"""
[Step 0] 几何预处理与特征构建 (v9.5 终极加固版 - 性能优化)

核心重构：
1. 环境修复：自动设置 OMP_NUM_THREADS，消除 libgomp 警告。
2. 极速 I/O 优化：将 centers 和 neighborhood 占位符改为标量，解决 np.array 转换卡顿。
3. 空间扶平 (Spatial Leveling): 彻底消除坡度对物理特征的干扰。
4. 增量扫描：支持断点续传。
"""

import os

# [v9.5 核心注入]: 修复 AutoDL 环境变量警告
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import pickle
import uuid
import numpy as np
import open3d as o3d
import yaml
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_calib_params(calib_dir):
    calib_files = list(Path(calib_dir).glob("*.pkl"))
    if not calib_files:
        raise FileNotFoundError(f"在 {calib_dir} 找不到 .pkl 标定文件！")
    with open(calib_files[0], "rb") as f:
        return pickle.load(f)

def project_3d_corners_to_pixel(corners_3d, calib_params):
    """原生矩阵投影：无黑盒，100% 还原真实透视"""
    R, T, K = calib_params["R"], calib_params["T"], calib_params["K"]
    uv_list = []
    for i in range(corners_3d.shape[0]):
        pt_lidar = corners_3d[i:i+1, :].T
        pt_cam = np.matmul(R, pt_lidar) + T
        pt_pixel = np.matmul(K, pt_cam)
        depth = float(pt_pixel[2, 0])
        if depth <= 0.1: return None 
        u = float(pt_pixel[0, 0] / depth)
        v = float(pt_pixel[1, 0] / depth)
        uv_list.append([u, v])
    return np.array(uv_list, dtype=np.float32)

def compute_phys_features(pts):
    """提取 8 维物理统计特征"""
    if len(pts) < 50: return np.zeros(8, dtype=np.float32)
    z = pts[:, 2]
    return np.array([
        np.max(z) - np.min(z), np.std(z),
        np.percentile(z, 95) - np.percentile(z, 5),
        len(pts) / 1000.0, np.mean(z), np.median(z),
        np.min(z), np.max(z)
    ], dtype=np.float32)

def main():
    cfg = load_config()
    pcd_dir = Path(cfg["paths"]["raw_pcd_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_params = read_calib_params(project_root / "RSRD_dev_toolkit" / "calibration_files")
    IMG_W, IMG_H = calib_params["Width"], calib_params["Height"]

    GEO = cfg["geometry"]
    NUM_COLS, NUM_ROWS = 7, 9
    roi_x, roi_y = GEO["roi_x"], GEO["roi_y"]
    
    psize_x = (roi_x[1] - roi_x[0]) / NUM_COLS
    psize_y = (roi_y[1] - roi_y[0]) / NUM_ROWS
    
    x_bins = np.linspace(roi_x[0], roi_x[1] - psize_x, NUM_COLS)
    y_bins = np.linspace(roi_y[0], roi_y[1] - psize_y, NUM_ROWS)

    all_pcd_files = sorted(list(pcd_dir.rglob("*.pcd")))
    
    processed_ids = {f.stem.replace("pkg_", "") for f in out_dir.glob("pkg_*.npz")}
    pending_files = [p for p in all_pcd_files if p.stem not in processed_ids]

    print("=" * 60)
    print(f"🚀 Road-MoME v9.5 预处理引擎启动 | 待处理: {len(pending_files)} 帧")
    print(f"📊 物理网格锁定: 7(列) x 9(行) = 63 Patch")
    print("=" * 60)

    for pcd_path in tqdm(pending_files, desc="Processing"):
        frame_id = pcd_path.stem
        final_out_path = out_dir / f"pkg_{frame_id}.npz"

        try:
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            pts_raw = np.asarray(pcd.points)

            # RANSAC 地形拟合
            calc_mask = (pts_raw[:, 0] >= roi_x[0]) & (pts_raw[:, 0] <= roi_x[1]) & \
                        (pts_raw[:, 1] >= roi_y[0]) & (pts_raw[:, 1] <= roi_y[1])
            plane_pts = pts_raw[calc_mask]

            if len(plane_pts) >= 150:
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(plane_pts)
                plane_model, _ = temp_pcd.segment_plane(0.04, 3, 1000)
                A, B, C, D = plane_model
            else:
                A, B, C, D = 0.0, 0.0, 1.0, 1.04 

            pts_feat = pts_raw.copy()
            safe_C = C if abs(C) > 1e-6 else 1e-6
            
            # 【核心空间扶平】
            z_plane = -(A * pts_feat[:, 0] + B * pts_feat[:, 1] + D) / safe_C
            pts_feat[:, 2] = pts_feat[:, 2] - z_plane

            phys_list, uv_list, corners_list, meta_list, pts_list = [], [], [], [], []

            for xi in x_bins:
                for yi in y_bins:
                    mask = (pts_feat[:, 0] >= xi) & (pts_feat[:, 0] < xi + psize_x) & \
                           (pts_feat[:, 1] >= yi) & (pts_feat[:, 1] < yi + psize_y)
                    p_pts = pts_feat[mask].copy()

                    def get_true_z(x, y): return -(A * x + B * y + D) / safe_C
                    corners_3d = np.array([
                        [xi, yi, get_true_z(xi, yi)],
                        [xi + psize_x, yi, get_true_z(xi + psize_x, yi)],
                        [xi + psize_x, yi + psize_y, get_true_z(xi + psize_x, yi + psize_y)],
                        [xi, yi + psize_y, get_true_z(xi, yi + psize_y)],
                    ])
                    
                    c_uv = project_3d_corners_to_pixel(corners_3d, calib_params)
                    
                    is_in_view = True
                    if c_uv is None: is_in_view = False
                    else:
                        if np.min(c_uv[:, 1]) > IMG_H or np.max(c_uv[:, 1]) < 0: is_in_view = False

                    if len(p_pts) < 50 or not is_in_view:
                        pts_padded = np.zeros((8192, 3), dtype=np.float32)
                        phys_feat = np.zeros(8, dtype=np.float32)
                        meta_info = [0.0, 0.0, 0.0] 
                    else:
                        phys_feat = compute_phys_features(p_pts)
                        p_pts[:, 2] -= np.mean(p_pts[:, 2]) # 局部中心化
                        if len(p_pts) >= 8192:
                            idx = np.random.choice(len(p_pts), 8192, replace=False)
                            pts_padded = p_pts[idx].astype(np.float32)
                        else:
                            pts_padded = np.pad(p_pts, ((0, 8192 - len(p_pts)), (0, 0)), mode='constant').astype(np.float32)
                        
                        p_label = 1.0 if phys_feat[2] > GEO["th_anomaly"] else 0.0
                        meta_info = [p_label, 0.5, 1.0]

                    pts_list.append(pts_padded)
                    phys_list.append(phys_feat)
                    meta_list.append(meta_info)
                    uv_list.append(np.nanmean(c_uv, axis=0) if c_uv is not None else np.array([0,0]))
                    corners_list.append(c_uv if c_uv is not None else np.zeros((4,2)))

            # 原子化保存
            unique_id = uuid.uuid4().hex[:6]
            tmp_path = out_dir / f"tmp_0_{unique_id}_{frame_id}.npz"
            
            # [v9.5 性能优化]: 仅保存极小的标量占位符，防止 Step 2 的加载报错，同时规避内存转换卡顿
            np.savez_compressed(
                tmp_path, 
                phys_8d=np.array(phys_list), 
                sampled_pts=np.array(pts_list),
                patch_uv=np.array(uv_list), 
                patch_corners_uv=np.array(corners_list),
                meta=np.array(meta_list),
                centers=np.array([0.0]),      # 轻量级占位符
                neighborhood=np.array([0.0])  # 轻量级占位符
            )
            os.replace(tmp_path, final_out_path)
            
        except Exception as e:
            print(f"\n❌ 处理帧 {frame_id} 失败: {e}")

    print(f"\n✅ 预处理任务圆满完成。")

if __name__ == "__main__":
    main()