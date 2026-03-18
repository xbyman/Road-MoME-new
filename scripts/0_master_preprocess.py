"""
[Step 0] 几何预处理与特征构建 (v7.8 空间扶平版)
核心重构：
1. 空间扶平手术 (Spatial Leveling): 彻底解决“坡度误判坑洼”的致命 Bug。利用 RANSAC 平面方程扣除原生点云的全局倾角，使所有路面归零至绝对水平面，还原真实的表面粗糙度。
2. 继承前置特性: 断点续传、50点密度底线、分位数抗噪指标。
"""

import os
import sys
import pickle
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
    calib_file = calib_files[0]
    with open(calib_file, "rb") as f:
        return pickle.load(f)


def project_3d_corners_to_pixel(corners_3d, calib_params):
    R = calib_params["R"]
    T = calib_params["T"]
    K = calib_params["K"]

    uv_list = []
    for i in range(corners_3d.shape[0]):
        pt_lidar = corners_3d[i : i + 1, :].T
        pt_cam = np.matmul(R, pt_lidar) + T
        pt_pixel = np.matmul(K, pt_cam)

        depth = float(pt_pixel[2, 0])
        if depth <= 0.1:
            return None

        u = float(pt_pixel[0, 0] / depth)
        v = float(pt_pixel[1, 0] / depth)
        uv_list.append([u, v])

    return np.array(uv_list, dtype=np.float32)


def compute_phys_features(pts):
    if len(pts) < 50:
        return np.zeros(8, dtype=np.float32)
    z = pts[:, 2]
    return np.array(
        [
            np.max(z) - np.min(z),
            np.std(z),
            np.percentile(z, 95) - np.percentile(z, 5),
            len(pts) / 1000.0,
            np.mean(z),
            np.median(z),
            np.min(z),
            np.max(z),
        ],
        dtype=np.float32,
    )


def main():
    cfg = load_config()
    pcd_dir = Path(cfg["paths"]["raw_pcd_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_params = read_calib_params(
        project_root / "RSRD_dev_toolkit" / "calibration_files"
    )
    IMG_W, IMG_H = calib_params["Width"], calib_params["Height"]

    GEO = cfg["geometry"]
    NUM_COLS, NUM_ROWS = 7, 9

    roi_x, roi_y = GEO["roi_x"], GEO["roi_y"]

    psize_x = (roi_x[1] - roi_x[0]) / NUM_COLS
    psize_y = (roi_y[1] - roi_y[0]) / NUM_ROWS

    x_bins = np.linspace(roi_x[0], roi_x[1] - psize_x, NUM_COLS)
    y_bins = np.linspace(roi_y[0], roi_y[1] - psize_y, NUM_ROWS)

    print("🔍 正在扫描原始目录与已完成的任务...")
    all_pcd_files = list(pcd_dir.rglob("*.pcd"))

    existing_npz_files = list(out_dir.glob("pkg_*.npz"))
    processed_frame_ids = {f.stem.replace("pkg_", "") for f in existing_npz_files}

    pending_pcd_files = []
    for pcd in all_pcd_files:
        if pcd.stem not in processed_frame_ids:
            pending_pcd_files.append(pcd)

    skipped_count = len(all_pcd_files) - len(pending_pcd_files)

    print("=" * 60)
    print(f"🚀 启动 v7.8 预处理流程 (空间扶平版) | ROI_Y: {roi_y}")
    if skipped_count > 0:
        print(f"⏭️  检测到已完成的特征包，自动跳过: {skipped_count} 帧")
    print(f"⏳ 本次实际待处理: {len(pending_pcd_files)} 帧")
    print("=" * 60)

    if not pending_pcd_files:
        print("🎉 所有特征包均已处理完毕，无需重复计算！")
        return

    processed_count = 0

    for pcd_path in tqdm(pending_pcd_files, desc="Preprocess"):
        frame_id = pcd_path.stem
        final_out_path = out_dir / f"pkg_{frame_id}.npz"
        tmp_out_path = out_dir / f"tmp_0_pkg_{frame_id}.npz"

        try:
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            pts_raw = np.asarray(pcd.points)

            # RANSAC 拟合地形
            calc_mask = (
                (pts_raw[:, 0] >= roi_x[0])
                & (pts_raw[:, 0] <= roi_x[1])
                & (pts_raw[:, 1] >= roi_y[0])
                & (pts_raw[:, 1] <= roi_y[1])
                & (pts_raw[:, 2] >= -1.5)
                & (pts_raw[:, 2] <= -0.5)
            )
            plane_pts = pts_raw[calc_mask]

            if len(plane_pts) >= 100:
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(plane_pts)
                plane_model, inliers = temp_pcd.segment_plane(0.03, 3, 1000)
                A, B, C, D = plane_model
            else:
                A, B, C, D = 0.0, 0.0, 1.0, 1.04

            pts_feat = pts_raw.copy()
            safe_C = C if abs(C) > 1e-6 else 1e-6

            # ========================================================
            # 【核心修复：空间全局扶平】
            # 将每个点在斜面上的基准高度扣除，强制将马路熨平成 Z=0 的绝对水平面
            # 这使得无论马路怎么倾斜，物理特征提取的只有“纯粹的粗糙度”
            # ========================================================
            z_plane = -(A * pts_feat[:, 0] + B * pts_feat[:, 1] + D) / safe_C
            pts_feat[:, 2] = pts_feat[:, 2] - z_plane

            phys_list, uv_list, corners_list, meta_list, pts_list = [], [], [], [], []

            for xi in x_bins:
                for yi in y_bins:
                    mask = (
                        (pts_feat[:, 0] >= xi)
                        & (pts_feat[:, 0] < xi + psize_x)
                        & (pts_feat[:, 1] >= yi)
                        & (pts_feat[:, 1] < yi + psize_y)
                    )
                    p_pts_feat = pts_feat[mask].copy()

                    # 投影用的 2D 坐标必须使用【原始的、未被扶平的物理斜面方程】，这样透视关系才吻合照片
                    def get_true_z(x, y):
                        return -(A * x + B * y + D) / safe_C

                    corners_3d = np.array(
                        [
                            [xi, yi, get_true_z(xi, yi)],
                            [xi + psize_x, yi, get_true_z(xi + psize_x, yi)],
                            [
                                xi + psize_x,
                                yi + psize_y,
                                get_true_z(xi + psize_x, yi + psize_y),
                            ],
                            [xi, yi + psize_y, get_true_z(xi, yi + psize_y)],
                        ]
                    )

                    c_uv = project_3d_corners_to_pixel(corners_3d, calib_params)

                    is_in_view = True
                    if c_uv is None:
                        is_in_view = False
                    else:
                        min_u, max_u = np.min(c_uv[:, 0]), np.max(c_uv[:, 0])
                        min_v, max_v = np.min(c_uv[:, 1]), np.max(c_uv[:, 1])
                        box_area = (max_u - min_u) * (max_v - min_v)

                        if box_area <= 0:
                            is_in_view = False
                        else:
                            inter_min_u = max(0, min_u)
                            inter_max_u = min(IMG_W, max_u)
                            inter_min_v = max(0, min_v)
                            inter_max_v = min(IMG_H, max_v)
                            inter_area = max(0, inter_max_u - inter_min_u) * max(
                                0, inter_max_v - inter_min_v
                            )
                            if (inter_area / box_area) < 0.50:
                                is_in_view = False

                    if len(p_pts_feat) < 50 or not is_in_view:
                        pts_padded = np.zeros((8192, 3), dtype=np.float32)
                        phys_feat = np.zeros(8, dtype=np.float32)
                        meta_info = [0.0, 0.0, 0.0]
                    else:
                        phys_feat = compute_phys_features(p_pts_feat)
                        p_pts_feat[:, 2] -= np.mean(
                            p_pts_feat[:, 2]
                        )  # 仅做微小局部中心化

                        if len(p_pts_feat) >= 8192:
                            idx = np.random.choice(len(p_pts_feat), 8192, replace=False)
                            pts_padded = p_pts_feat[idx].astype(np.float32)
                        else:
                            pts_padded = np.pad(
                                p_pts_feat,
                                ((0, 8192 - len(p_pts_feat)), (0, 0)),
                                mode="constant",
                            ).astype(np.float32)

                        # 由于现在点云已被扶平为绝对水平面，超过 3.5cm 甚至 4cm 的起伏就是实打实的坑洼了
                        p_label = 1.0 if phys_feat[2] > GEO["th_anomaly"] else 0.0
                        meta_info = [p_label, 0.5, 1.0]

                    pts_list.append(pts_padded)
                    phys_list.append(phys_feat)
                    meta_list.append(meta_info)
                    uv_list.append(
                        np.nanmean(c_uv, axis=0)
                        if c_uv is not None
                        else np.array([0, 0])
                    )
                    corners_list.append(c_uv if c_uv is not None else np.zeros((4, 2)))

            np.savez_compressed(
                tmp_out_path,
                phys_8d=np.array(phys_list),
                sampled_pts=np.array(pts_list),
                patch_uv=np.array(uv_list),
                patch_corners_uv=np.array(corners_list),
                meta=np.array(meta_list),
            )
            os.replace(tmp_out_path, final_out_path)
            processed_count += 1

        except Exception as e:
            print(f"\n❌ 处理帧 {frame_id} 失败: {e}")

    print(f"\n✅ 本次预处理完成！")


if __name__ == "__main__":
    main()
