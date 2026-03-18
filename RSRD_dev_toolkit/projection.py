import numpy as np
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import yaml
import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# GPU 加速导入
try:
    import cupy as cp

    HAS_GPU = True
    GPU_AVAILABLE = True
except ImportError:
    HAS_GPU = False
    GPU_AVAILABLE = False
    cp = np  # 降级到numpy


def get_array_module(use_gpu=True):
    """
    根据GPU可用性选择计算库

    Parameters:
    -----------
    use_gpu : bool
        是否尝试使用GPU

    Returns:
    --------
    module : numpy 或 cupy
        计算模块
    """
    if use_gpu and HAS_GPU:
        return cp
    return np


def read_calib_params(calib_file):
    """
    intrinsics (after rectification): calib_params["K"]
    stereo baseline(in mm): calib_params["B"]
    lidar -> left camera extrinsics: calib_params["R"], calib_params["T"]
    """
    with open(calib_file, "rb") as f:
        calib_params = pickle.load(f)

    return calib_params


def load_config(config_path):
    """
    从YAML配置文件加载配置

    Parameters:
    -----------
    config_path : str
        配置文件路径

    Returns:
    --------
    dict : 配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def scan_data_pairs(raw_pcd_dir, raw_img_dir):
    """
    扫描raw_pcd_dir和raw_img_dir目录，匹配对应的点云和图像文件对

    Parameters:
    -----------
    raw_pcd_dir : str
        点云数据根目录
    raw_img_dir : str
        图像数据根目录

    Returns:
    --------
    list : 文件对列表，每项为 (pcd_path, img_path, timestamp)
    """
    data_pairs = []

    # 遍历所有日期-时间目录
    raw_pcd_dir = Path(raw_pcd_dir)
    date_dirs = sorted([d for d in raw_pcd_dir.iterdir() if d.is_dir()])

    for date_dir in date_dirs:
        pcd_subdir = date_dir / "pcd"
        img_subdir = date_dir / "left"

        if not pcd_subdir.exists() or not img_subdir.exists():
            continue

        # 获取点云文件
        pcd_files = sorted([f for f in pcd_subdir.glob("*.pcd")])

        for pcd_file in pcd_files:
            # 提取时间戳（不含扩展名）
            timestamp = pcd_file.stem
            # 查找对应的图像文件
            img_file = img_subdir / f"{timestamp}.jpg"

            if img_file.exists():
                data_pairs.append((str(pcd_file), str(img_file), timestamp))

    return data_pairs


def get_calibration_file(calib_dir):
    """
    自动查找最新的校准文件

    Parameters:
    -----------
    calib_dir : str
        校准文件目录

    Returns:
    --------
    str : 最新校准文件的路径
    """
    calib_dir = Path(calib_dir)
    calib_files = sorted(calib_dir.glob("*.pkl"))
    if calib_files:
        return str(calib_files[-1])
    else:
        raise FileNotFoundError(f"未找到校准文件在目录: {calib_dir}")


def project_point2camera(calib_params, cloud, use_gpu=True):
    """
    将LiDAR点云投影到相机图像平面（支持GPU加速）

    Parameters:
    -----------
    calib_params : dict
        校准参数
    cloud : ndarray
        点云坐标 (N, 3)
    use_gpu : bool
        是否使用GPU加速

    Returns:
    --------
    tuple : (uv坐标, 深度值, 有效点索引)
    """
    xp = get_array_module(use_gpu)

    # 将数据移至GPU/CPU
    if use_gpu and HAS_GPU:
        cloud_gpu = cp.asarray(cloud)
        R = cp.asarray(calib_params["R"])
        T = cp.asarray(calib_params["T"])
        K = cp.asarray(calib_params["K"])
    else:
        cloud_gpu = cloud
        R = calib_params["R"]
        T = calib_params["T"]
        K = calib_params["K"]

    # 矢量化投影计算（一次性处理所有点而不是逐个循环）
    # cloud: (N, 3) -> (3, N)
    points_lidar = cloud_gpu.T  # (3, N)

    # 从LiDAR坐标系转换到相机坐标系
    points_camera = xp.matmul(R, points_lidar) + T  # (3, N)

    # 提取深度（Z坐标）
    point_camera_depth = points_camera[2, :]  # (N,)

    # 投影到图像平面
    uv_pixel = xp.matmul(K, points_camera)  # (3, N)

    # 归一化投影坐标
    point_uv = xp.zeros((cloud_gpu.shape[0], 2), dtype=xp.float32)
    point_uv[:, 0] = uv_pixel[0, :] / uv_pixel[2, :]
    point_uv[:, 1] = uv_pixel[1, :] / uv_pixel[2, :]

    # 筛查有效点（在图像范围内）
    img_width = calib_params["Width"]
    img_height = calib_params["Height"]

    # 使用GPU加速的布尔索引
    valid_mask = (
        (point_uv[:, 0] >= 0)
        & (point_uv[:, 0] <= img_width)
        & (point_uv[:, 1] >= 0)
        & (point_uv[:, 1] <= img_height)
        & (point_camera_depth > 0)  # 深度值必须为正
    )

    valid_indices = xp.where(valid_mask)[0]

    # 提取有效点
    uv_valid = point_uv[valid_mask]
    depth_valid = point_camera_depth[valid_mask]

    # 转回numpy（如果使用了GPU）
    if use_gpu and HAS_GPU:
        uv_valid = cp.asnumpy(uv_valid)
        depth_valid = cp.asnumpy(depth_valid)
        valid_indices = cp.asnumpy(valid_indices)

    return uv_valid, depth_valid.astype(np.float32), valid_indices.astype(np.int32)


def show_image_with_points(uv, depth_uv, image, cloud=None, calib_params=None):
    # show the image with projected lidar points and their 3D coordinates
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)

    # show the background image
    ax.imshow(image)

    # scatter the projected points with depth coloring
    scatter = ax.scatter(
        uv[:, 0],
        uv[:, 1],
        c=depth_uv,
        cmap="brg",
        s=20,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Depth (m)", fontsize=10)

    # if point cloud coordinates provided, display them
    if cloud is not None:
        # get valid point indices (those within image bounds)
        point_uv_int = uv.astype(np.int16)
        valid_indices = []
        for i, (u, v) in enumerate(point_uv_int):
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                valid_indices.append(i)

        # display 3D coordinates for a subset of points (to avoid cluttering)
        sample_step = max(1, len(valid_indices) // 20)  # show ~20 labels
        for idx_pos, i in enumerate(valid_indices[::sample_step]):
            x3d, y3d, z3d = cloud[i]
            u_pos, v_pos = uv[i]

            # format 3D coordinate label
            coord_text = f"({x3d:.2f}, {y3d:.2f}, {z3d:.2f})"

            # add text annotation
            ax.text(
                u_pos,
                v_pos - 10,
                coord_text,
                fontsize=7,
                color="yellow",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
                ha="center",
                va="top",
            )

    # set coordinate axes
    if calib_params is not None:
        img_height = calib_params["Height"]
        img_width = calib_params["Width"]
    else:
        img_height, img_width = image.shape[:2]

    ax.set_xlabel("X (pixels)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y (pixels)", fontsize=11, fontweight="bold")

    # set ticks
    x_ticks = np.linspace(0, img_width, 11)
    y_ticks = np.linspace(0, img_height, 11)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{int(x)}" for x in x_ticks], fontsize=8)
    ax.set_yticklabels([f"{int(y)}" for y in y_ticks], fontsize=8)

    # add grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # set axis limits
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)

    if cloud is not None:
        ax.set_title(
            f"Point Cloud Projection - 3D Coordinates (x, y, z) in LiDAR frame",
            fontsize=12,
            fontweight="bold",
        )
    else:
        ax.set_title("Point Cloud Projection", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()


def get_dis_depth_map(uv, depth_uv, calib_params, use_gpu=True):
    """
    生成视差图和深度图（支持GPU加速）

    Parameters:
    -----------
    uv : ndarray
        投影坐标 (N, 2)
    depth_uv : ndarray
        深度值 (N,)
    calib_params : dict
        校准参数
    use_gpu : bool
        是否使用GPU加速

    Returns:
    --------
    tuple : (视差图, 深度图)
    """
    xp = get_array_module(use_gpu)

    height = calib_params["Height"]
    width = calib_params["Width"]

    # 初始化地图
    dis_map = xp.zeros((height, width), dtype=xp.float32)
    depth_map = xp.zeros((height, width), dtype=xp.float32)

    # 转换坐标和深度到适当的设备
    if use_gpu and HAS_GPU:
        uv_gpu = cp.asarray(uv.astype(np.int16))
        depth_gpu = cp.asarray(depth_uv)
        B = calib_params["B"] / 1000
        fx = calib_params["K"][0, 0]
    else:
        uv_gpu = uv.astype(np.int16)
        depth_gpu = depth_uv
        B = calib_params["B"] / 1000
        fx = calib_params["K"][0, 0]

    # 使用向量化操作而不是循环
    x_coords = uv_gpu[:, 0]
    y_coords = uv_gpu[:, 1]

    # 视差 = baseline * fx / depth
    disparities = (B * fx) / depth_gpu

    # 使用高级索引赋值
    dis_map[y_coords, x_coords] = disparities
    depth_map[y_coords, x_coords] = depth_gpu

    # 转回numpy（如果使用了GPU）
    if use_gpu and HAS_GPU:
        dis_map = cp.asnumpy(dis_map)
        depth_map = cp.asnumpy(depth_map)

    return dis_map, depth_map


def get_projection_statistics(
    uv, depth_uv, cloud=None, calib_params=None, valid_indices=None
):
    """
    统计投影点云的相关信息，并收集分布数据

    Parameters:
    -----------
    uv : ndarray
        投影后的像素坐标 (N, 2)
    depth_uv : ndarray
        投影后的深度值 (N,)
    cloud : ndarray, optional
        原始3D点云坐标 (M, 3)，用于统计3D范围
    calib_params : dict, optional
        相机参数
    valid_indices : ndarray, optional
        投影到图片内的有效点索引

    Returns:
    --------
    dict : 包含投影统计信息和分布数据
    """
    stats = {}

    # 投影点数统计
    stats["projected_count"] = len(uv)
    stats["depth_mean"] = float(np.mean(depth_uv))
    stats["depth_std"] = float(np.std(depth_uv))
    stats["depth_min"] = float(np.min(depth_uv))
    stats["depth_max"] = float(np.max(depth_uv))

    if cloud is not None:
        stats["total_cloud_count"] = cloud.shape[0]
        stats["projection_ratio"] = len(uv) / cloud.shape[0]

        # 统计全部3D坐标范围
        stats["x_range"] = (float(cloud[:, 0].min()), float(cloud[:, 0].max()))
        stats["y_range"] = (float(cloud[:, 1].min()), float(cloud[:, 1].max()))
        stats["z_range"] = (float(cloud[:, 2].min()), float(cloud[:, 2].max()))

        # 统计3D坐标均值和标准差
        stats["x_mean"] = float(np.mean(cloud[:, 0]))
        stats["y_mean"] = float(np.mean(cloud[:, 1]))
        stats["z_mean"] = float(np.mean(cloud[:, 2]))
        stats["x_std"] = float(np.std(cloud[:, 0]))
        stats["y_std"] = float(np.std(cloud[:, 1]))
        stats["z_std"] = float(np.std(cloud[:, 2]))

        # 统计投影到图片上的点的3D坐标范围和分布
        if valid_indices is not None and len(valid_indices) > 0:
            projected_cloud = cloud[valid_indices]
            stats["projected_count_valid"] = len(valid_indices)
            stats["projected_x_range"] = (
                float(projected_cloud[:, 0].min()),
                float(projected_cloud[:, 0].max()),
            )
            stats["projected_y_range"] = (
                float(projected_cloud[:, 1].min()),
                float(projected_cloud[:, 1].max()),
            )
            stats["projected_z_range"] = (
                float(projected_cloud[:, 2].min()),
                float(projected_cloud[:, 2].max()),
            )

            # 投影点的3D统计
            stats["projected_x_mean"] = float(np.mean(projected_cloud[:, 0]))
            stats["projected_y_mean"] = float(np.mean(projected_cloud[:, 1]))
            stats["projected_z_mean"] = float(np.mean(projected_cloud[:, 2]))
            stats["projected_x_std"] = float(np.std(projected_cloud[:, 0]))
            stats["projected_y_std"] = float(np.std(projected_cloud[:, 1]))
            stats["projected_z_std"] = float(np.std(projected_cloud[:, 2]))

            # 分布信息（分成10个bin）
            stats["projected_x_distribution"] = np.histogram(
                projected_cloud[:, 0], bins=10
            )[0].tolist()
            stats["projected_y_distribution"] = np.histogram(
                projected_cloud[:, 1], bins=10
            )[0].tolist()
            stats["projected_z_distribution"] = np.histogram(
                projected_cloud[:, 2], bins=10
            )[0].tolist()

    # 统计投影后的像素范围
    stats["uv_x_range"] = (float(uv[:, 0].min()), float(uv[:, 0].max()))
    stats["uv_y_range"] = (float(uv[:, 1].min()), float(uv[:, 1].max()))
    stats["uv_x_mean"] = float(np.mean(uv[:, 0]))
    stats["uv_y_mean"] = float(np.mean(uv[:, 1]))

    # 获取图像范围
    if calib_params is not None:
        stats["image_width"] = calib_params["Width"]
        stats["image_height"] = calib_params["Height"]

    return stats


def print_projection_statistics(stats):
    """
    打印投影统计信息

    Parameters:
    -----------
    stats : dict
        由 get_projection_statistics() 返回的统计字典
    """
    print("\n" + "=" * 70)
    print("点云投影统计信息详情")
    print("=" * 70)

    print(f"\n【投影点数统计】")
    print(f"  投影到图片上的点数: {stats['projected_count']}")
    if "total_cloud_count" in stats:
        print(f"  总点云数量: {stats['total_cloud_count']}")
        print(f"  投影比例: {stats['projection_ratio']:.2%}")

    print(f"\n【深度统计】")
    print(f"  深度均值: {stats['depth_mean']:.3f} m")
    print(f"  深度标准差: {stats['depth_std']:.3f} m")
    print(f"  深度范围: [{stats['depth_min']:.3f}, {stats['depth_max']:.3f}] m")

    print(f"\n【全部点云3D坐标范围（LiDAR坐标系）】")
    if "x_range" in stats:
        print(f"  X坐标范围: [{stats['x_range'][0]:.3f}, {stats['x_range'][1]:.3f}] m")
        print(f"  X坐标均值±std: {stats['x_mean']:.3f} ± {stats['x_std']:.3f} m")
        print(f"  Y坐标范围: [{stats['y_range'][0]:.3f}, {stats['y_range'][1]:.3f}] m")
        print(f"  Y坐标均值±std: {stats['y_mean']:.3f} ± {stats['y_std']:.3f} m")
        print(f"  Z坐标范围: [{stats['z_range'][0]:.3f}, {stats['z_range'][1]:.3f}] m")
        print(f"  Z坐标均值±std: {stats['z_mean']:.3f} ± {stats['z_std']:.3f} m")

    print(f"\n【投影到图片上的点云3D坐标范围（LiDAR坐标系）】")
    if "projected_x_range" in stats:
        print(f"  有效投影点数: {stats['projected_count_valid']}")
        print(
            f"  投影X坐标范围: [{stats['projected_x_range'][0]:.3f}, {stats['projected_x_range'][1]:.3f}] m"
        )
        print(
            f"  投影X均值±std: {stats['projected_x_mean']:.3f} ± {stats['projected_x_std']:.3f} m"
        )
        print(
            f"  投影Y坐标范围: [{stats['projected_y_range'][0]:.3f}, {stats['projected_y_range'][1]:.3f}] m"
        )
        print(
            f"  投影Y均值±std: {stats['projected_y_mean']:.3f} ± {stats['projected_y_std']:.3f} m"
        )
        print(
            f"  投影Z坐标范围: [{stats['projected_z_range'][0]:.3f}, {stats['projected_z_range'][1]:.3f}] m"
        )
        print(
            f"  投影Z均值±std: {stats['projected_z_mean']:.3f} ± {stats['projected_z_std']:.3f} m"
        )

    print(f"\n【2D投影范围（像素坐标）】")
    print(f"  像素X范围: [{stats['uv_x_range'][0]:.1f}, {stats['uv_x_range'][1]:.1f}]")
    print(f"  像素X均值: {stats['uv_x_mean']:.1f}")
    print(f"  像素Y范围: [{stats['uv_y_range'][0]:.1f}, {stats['uv_y_range'][1]:.1f}]")
    print(f"  像素Y均值: {stats['uv_y_mean']:.1f}")

    if "image_width" in stats:
        print(f"\n【图像分辨率】")
        print(f"  宽度: {stats['image_width']} 像素")
        print(f"  高度: {stats['image_height']} 像素")
        # 计算覆盖率
        x_coverage = (
            (stats["uv_x_range"][1] - stats["uv_x_range"][0])
            / stats["image_width"]
            * 100
        )
        y_coverage = (
            (stats["uv_y_range"][1] - stats["uv_y_range"][0])
            / stats["image_height"]
            * 100
        )
        print(f"  X方向覆盖率: {x_coverage:.1f}%")
        print(f"  Y方向覆盖率: {y_coverage:.1f}%")

    print("=" * 70 + "\n")


def show_clouds_with_color(image, depth_map, calib_params):
    # Recovering the colorized point cloud using Open3D.
    image_o3d = o3d.geometry.Image(image)
    depth_o3d = o3d.geometry.Image(depth_map)

    rgbd_image_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d,
        depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
        depth_trunc=20,
    )

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.intrinsic_matrix = calib_params["K"]
    camera_intrinsic.height = calib_params["Height"]
    camera_intrinsic.width = calib_params["Width"]
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_o3d, camera_intrinsic
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()


def save_statistics_report(stats_list, output_path, timestamp_list=None):
    """
    将统计信息保存为JSON文件和CSV摘要

    Parameters:
    -----------
    stats_list : list
        统计字典列表
    output_path : str
        输出文件路径（不含扩展名）
    timestamp_list : list, optional
        时间戳列表，用于关联统计数据
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细统计为JSON
    json_file = f"{output_path}_statistics_detail.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(stats_list, f, indent=2, ensure_ascii=False)
    print(f"✓ 详细统计信息已保存到: {json_file}")

    # 保存摘要统计为CSV
    csv_file = f"{output_path}_statistics_summary.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        if len(stats_list) > 0:
            # 写入表头
            keys = list(stats_list[0].keys())
            f.write(",".join(keys) + "\n")

            # 写入数据行
            for idx, stats in enumerate(stats_list):
                values = []
                for key in keys:
                    val = stats.get(key, "")
                    if isinstance(val, (list, tuple)):
                        values.append(f'"{val}"')
                    else:
                        values.append(str(val))
                f.write(",".join(values) + "\n")

    print(f"✓ 摘要统计信息已保存到: {csv_file}")

    # 生成统计报告文本
    report_file = f"{output_path}_statistics_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("点云投影统计报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理文件数: {len(stats_list)}\n")
        f.write("=" * 70 + "\n\n")

        for idx, stats in enumerate(stats_list):
            f.write(f"\n【样本 {idx + 1}】")
            if timestamp_list and idx < len(timestamp_list):
                f.write(f" - {timestamp_list[idx]}\n")
            else:
                f.write("\n")
            f.write("-" * 70 + "\n")

            f.write(f"投影点数: {stats.get('projected_count', 'N/A')}\n")
            if "projection_ratio" in stats:
                f.write(f"投影比例: {stats['projection_ratio']:.2%}\n")

            if "projected_z_range" in stats:
                f.write(
                    f"投影Z范围: [{stats['projected_z_range'][0]:.3f}, {stats['projected_z_range'][1]:.3f}] m\n"
                )
                f.write(
                    f"投影X范围: [{stats['projected_x_range'][0]:.3f}, {stats['projected_x_range'][1]:.3f}] m\n"
                )
                f.write(
                    f"投影Y范围: [{stats['projected_y_range'][0]:.3f}, {stats['projected_y_range'][1]:.3f}] m\n"
                )

            f.write("\n")

    print(f"✓ 统计报告已保存到: {report_file}")


def calculate_global_statistics(stats_list, output_path):
    """
    计算所有样本的全局统计和分布

    Parameters:
    -----------
    stats_list : list
        统计字典列表
    output_path : str
        输出文件路径（不含扩展名）
    """
    if len(stats_list) == 0:
        print("[WARNING] 统计列表为空，无法计算全局统计")
        return {}

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集全局数据
    global_stats = {}

    # ========== 投影点数统计 ==========
    projected_counts = [s["projected_count"] for s in stats_list]
    global_stats["total_samples"] = len(stats_list)
    global_stats["projected_count_mean"] = float(np.mean(projected_counts))
    global_stats["projected_count_std"] = float(np.std(projected_counts))
    global_stats["projected_count_min"] = int(np.min(projected_counts))
    global_stats["projected_count_max"] = int(np.max(projected_counts))
    global_stats["projected_count_total"] = int(np.sum(projected_counts))

    # ========== 深度统计 ==========
    depth_means = [s["depth_mean"] for s in stats_list if "depth_mean" in s]
    depth_stds = [s["depth_std"] for s in stats_list if "depth_std" in s]
    depth_mins = [s["depth_min"] for s in stats_list if "depth_min" in s]
    depth_maxs = [s["depth_max"] for s in stats_list if "depth_max" in s]

    if depth_means:
        global_stats["depth_mean_of_means"] = float(np.mean(depth_means))
        global_stats["depth_mean_of_stds"] = float(np.mean(depth_stds))
        global_stats["depth_global_min"] = float(np.min(depth_mins))
        global_stats["depth_global_max"] = float(np.max(depth_maxs))

    # ========== 投影比例 ==========
    projection_ratios = [
        s["projection_ratio"] for s in stats_list if "projection_ratio" in s
    ]
    if projection_ratios:
        global_stats["projection_ratio_mean"] = float(np.mean(projection_ratios))
        global_stats["projection_ratio_std"] = float(np.std(projection_ratios))
        global_stats["projection_ratio_min"] = float(np.min(projection_ratios))
        global_stats["projection_ratio_max"] = float(np.max(projection_ratios))

    # ========== 全部点云3D范围统计 ==========
    x_mins = [s["x_range"][0] for s in stats_list if "x_range" in s]
    x_maxs = [s["x_range"][1] for s in stats_list if "x_range" in s]
    y_mins = [s["y_range"][0] for s in stats_list if "y_range" in s]
    y_maxs = [s["y_range"][1] for s in stats_list if "y_range" in s]
    z_mins = [s["z_range"][0] for s in stats_list if "z_range" in s]
    z_maxs = [s["z_range"][1] for s in stats_list if "z_range" in s]

    if x_mins:
        global_stats["full_cloud_x_global_min"] = float(np.min(x_mins))
        global_stats["full_cloud_x_global_max"] = float(np.max(x_maxs))
        global_stats["full_cloud_y_global_min"] = float(np.min(y_mins))
        global_stats["full_cloud_y_global_max"] = float(np.max(y_maxs))
        global_stats["full_cloud_z_global_min"] = float(np.min(z_mins))
        global_stats["full_cloud_z_global_max"] = float(np.max(z_maxs))

    # ========== 投影到图片上的3D范围统计 ==========
    proj_x_mins = [
        s["projected_x_range"][0] for s in stats_list if "projected_x_range" in s
    ]
    proj_x_maxs = [
        s["projected_x_range"][1] for s in stats_list if "projected_x_range" in s
    ]
    proj_y_mins = [
        s["projected_y_range"][0] for s in stats_list if "projected_y_range" in s
    ]
    proj_y_maxs = [
        s["projected_y_range"][1] for s in stats_list if "projected_y_range" in s
    ]
    proj_z_mins = [
        s["projected_z_range"][0] for s in stats_list if "projected_z_range" in s
    ]
    proj_z_maxs = [
        s["projected_z_range"][1] for s in stats_list if "projected_z_range" in s
    ]

    if proj_x_mins:
        global_stats["projected_cloud_x_global_min"] = float(np.min(proj_x_mins))
        global_stats["projected_cloud_x_global_max"] = float(np.max(proj_x_maxs))
        global_stats["projected_cloud_y_global_min"] = float(np.min(proj_y_mins))
        global_stats["projected_cloud_y_global_max"] = float(np.max(proj_y_maxs))
        global_stats["projected_cloud_z_global_min"] = float(np.min(proj_z_mins))
        global_stats["projected_cloud_z_global_max"] = float(np.max(proj_z_maxs))

        # 坐标范围的均值统计（而非坐标均值）
        proj_x_range_min_mean = float(np.mean(proj_x_mins))
        proj_x_range_max_mean = float(np.mean(proj_x_maxs))
        proj_y_range_min_mean = float(np.mean(proj_y_mins))
        proj_y_range_max_mean = float(np.mean(proj_y_maxs))
        proj_z_range_min_mean = float(np.mean(proj_z_mins))
        proj_z_range_max_mean = float(np.mean(proj_z_maxs))

        global_stats["projected_x_range_mean"] = [
            proj_x_range_min_mean,
            proj_x_range_max_mean,
        ]
        global_stats["projected_y_range_mean"] = [
            proj_y_range_min_mean,
            proj_y_range_max_mean,
        ]
        global_stats["projected_z_range_mean"] = [
            proj_z_range_min_mean,
            proj_z_range_max_mean,
        ]

    # 保存全局统计为JSON
    global_json_file = f"{output_path}_global_statistics.json"
    with open(global_json_file, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2, ensure_ascii=False)
    print(f"✓ 全局统计信息已保存到: {global_json_file}")

    # 生成全局统计报告
    global_report_file = f"{output_path}_global_report.txt"
    with open(global_report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("点云投影全局统计报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"【样本统计】\n")
        f.write(f"  总样本数: {global_stats.get('total_samples', 'N/A')}\n")
        f.write(f"  总投影点数: {global_stats.get('projected_count_total', 'N/A')}\n")

        f.write(f"\n【投影点数分布】\n")
        f.write(
            f"  平均投影点数: {global_stats.get('projected_count_mean', 'N/A'):.0f}\n"
        )
        f.write(f"  标准差: {global_stats.get('projected_count_std', 'N/A'):.0f}\n")
        f.write(f"  最小值: {global_stats.get('projected_count_min', 'N/A')}\n")
        f.write(f"  最大值: {global_stats.get('projected_count_max', 'N/A')}\n")

        f.write(f"\n【投影比例分布】\n")
        if "projection_ratio_mean" in global_stats:
            f.write(f"  平均比例: {global_stats['projection_ratio_mean']:.2%}\n")
            f.write(f"  标准差: {global_stats['projection_ratio_std']:.2%}\n")
            f.write(f"  最小比例: {global_stats['projection_ratio_min']:.2%}\n")
            f.write(f"  最大比例: {global_stats['projection_ratio_max']:.2%}\n")

        f.write(f"\n【深度统计（全部样本）】\n")
        if "depth_global_min" in global_stats:
            f.write(f"  全局最小深度: {global_stats['depth_global_min']:.3f} m\n")
            f.write(f"  全局最大深度: {global_stats['depth_global_max']:.3f} m\n")
            f.write(f"  综合平均深度: {global_stats['depth_mean_of_means']:.3f} m\n")
            f.write(f"  综合标准差: {global_stats['depth_mean_of_stds']:.3f} m\n")

        f.write(f"\n【全部点云3D坐标全局范围（LiDAR坐标系）】\n")
        if "full_cloud_x_global_min" in global_stats:
            f.write(
                f"  X坐标全局范围: [{global_stats['full_cloud_x_global_min']:.3f}, {global_stats['full_cloud_x_global_max']:.3f}] m\n"
            )
            f.write(
                f"  Y坐标全局范围: [{global_stats['full_cloud_y_global_min']:.3f}, {global_stats['full_cloud_y_global_max']:.3f}] m\n"
            )
            f.write(
                f"  Z坐标全局范围: [{global_stats['full_cloud_z_global_min']:.3f}, {global_stats['full_cloud_z_global_max']:.3f}] m\n"
            )

        f.write(f"\n【投影到图片上的点云3D坐标全局范围（LiDAR坐标系）】\n")
        if "projected_cloud_x_global_min" in global_stats:
            f.write(
                f"  投影X坐标全局范围: [{global_stats['projected_cloud_x_global_min']:.3f}, {global_stats['projected_cloud_x_global_max']:.3f}] m\n"
            )
            if "projected_x_range_mean" in global_stats:
                x_range_mean = global_stats["projected_x_range_mean"]
                f.write(
                    f"  投影X坐标范围均值: [{x_range_mean[0]:.3f}, {x_range_mean[1]:.3f}] m\n"
                )

            f.write(
                f"  投影Y坐标全局范围: [{global_stats['projected_cloud_y_global_min']:.3f}, {global_stats['projected_cloud_y_global_max']:.3f}] m\n"
            )
            if "projected_y_range_mean" in global_stats:
                y_range_mean = global_stats["projected_y_range_mean"]
                f.write(
                    f"  投影Y坐标范围均值: [{y_range_mean[0]:.3f}, {y_range_mean[1]:.3f}] m\n"
                )

            f.write(
                f"  投影Z坐标全局范围: [{global_stats['projected_cloud_z_global_min']:.3f}, {global_stats['projected_cloud_z_global_max']:.3f}] m\n"
            )
            if "projected_z_range_mean" in global_stats:
                z_range_mean = global_stats["projected_z_range_mean"]
                f.write(
                    f"  投影Z坐标范围均值: [{z_range_mean[0]:.3f}, {z_range_mean[1]:.3f}] m\n"
                )

        f.write(f"\n【推荐的数据范围设置】\n")
        if "projected_cloud_x_global_min" in global_stats:
            # 加入一些边界优化（扩展10%作为安全余地）
            x_min = global_stats["projected_cloud_x_global_min"]
            x_max = global_stats["projected_cloud_x_global_max"]
            y_min = global_stats["projected_cloud_y_global_min"]
            y_max = global_stats["projected_cloud_y_global_max"]
            z_min = global_stats["projected_cloud_z_global_min"]
            z_max = global_stats["projected_cloud_z_global_max"]

            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            z_margin = (z_max - z_min) * 0.1

            f.write(
                f"  建议X范围: [{x_min - x_margin:.3f}, {x_max + x_margin:.3f}] m\n"
            )
            f.write(
                f"  建议Y范围: [{y_min - y_margin:.3f}, {y_max + y_margin:.3f}] m\n"
            )
            f.write(
                f"  建议Z范围: [{z_min - z_margin:.3f}, {z_max + z_margin:.3f}] m\n"
            )

        f.write("\n" + "=" * 80 + "\n")

    print(f"✓ 全局报告已保存到: {global_report_file}")

    return global_stats


if __name__ == "__main__":
    # =============== 配置区 ===============
    CONFIG_PATH = r"C:\Users\31078\Desktop\ROAD\config\config.yaml"
    CALIB_DIR = r"C:\Users\31078\Desktop\ROAD\RSRD_dev_toolkit\calibration_files"

    # 可视化开关
    SHOW_PLOTS = False  # 设置为 True 显示可视化，False 不显示
    SHOW_COLORIZED_CLOUD = False  # 是否显示着色点云

    # GPU加速开关
    USE_GPU = True  # 设置为 True 使用GPU加速，False 使用CPU

    # 处理限制（设为None则处理全部）
    MAX_SAMPLES = None  # 只处理前N个文件对，None表示全部

    # =====================================
    print("\n" + "=" * 70)
    print("点云投影统计系统")
    print("=" * 70)

    # 显示GPU状态
    if USE_GPU and HAS_GPU:
        print("\n✓ GPU加速模式启用")
        try:
            gpu_name = cp.cuda.Device().get_device_id()
            print(f"  使用GPU设备: {gpu_name}")
        except:
            print("  GPU设备信息不可用")
    else:
        print("\n✗ CPU模式：未检测到GPU或GPU加速未启用")
        if not HAS_GPU:
            print("  建议安装CuPy以启用GPU加速: pip install cupy-cuda11x")

    print("\n[1/4] 加载配置...")
    config = load_config(CONFIG_PATH)

    raw_pcd_dir = config["paths"]["raw_pcd_dir"]
    raw_img_dir = config["paths"]["raw_img_dir"]

    print("[2/4] 扫描数据文件对...")
    data_pairs = scan_data_pairs(raw_pcd_dir, raw_img_dir)

    if len(data_pairs) == 0:
        print("[ERROR] 未找到任何数据对！")
        exit(1)

    # 限制处理数量
    if MAX_SAMPLES is not None:
        data_pairs = data_pairs[:MAX_SAMPLES]

    print(f"      找到 {len(data_pairs)} 个有效数据对")

    print("[3/4] 加载校准文件...")
    calib_file = get_calibration_file(CALIB_DIR)
    calib_params = read_calib_params(calib_file)

    # 统计结果列表
    all_stats = []
    all_timestamps = []

    print("[4/4] 处理数据...")
    print()
    # 处理每个数据对（使用进度条）
    pbar = tqdm(data_pairs, desc="处理数据对", unit=" 样本")

    for pcd_path, img_path, timestamp in pbar:
        try:
            # 更新进度条描述
            pbar.set_description(f"处理: {timestamp}")

            # 读取数据
            cloud = o3d.io.read_point_cloud(pcd_path)
            cloud = np.asarray(cloud.points)

            image = cv2.imread(img_path)
            if image is None:
                pbar.write(f"[WARNING] 无法读取图像: {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 投影
            pbar.set_postfix({"状态": "投影中..."}, refresh=True)
            uv, depth_uv, valid_indices = project_point2camera(
                calib_params, cloud, use_gpu=USE_GPU
            )
            dis_map, depth_map = get_dis_depth_map(
                uv, depth_uv, calib_params, use_gpu=USE_GPU
            )

            # 统计
            pbar.set_postfix({"状态": "统计中..."}, refresh=True)
            stats = get_projection_statistics(
                uv, depth_uv, cloud, calib_params, valid_indices
            )

            # 保存统计
            all_stats.append(stats)
            all_timestamps.append(timestamp)

            # 可视化（如果开启）
            if SHOW_PLOTS:
                show_image_with_points(uv, depth_uv, image, cloud, calib_params)

            if SHOW_COLORIZED_CLOUD:
                show_clouds_with_color(image, depth_map, calib_params)

            pbar.set_postfix({"状态": "✓完成"}, refresh=True)

        except Exception as e:
            pbar.write(f"[ERROR] 处理 {timestamp} 失败: {str(e)}")
            continue

    pbar.close()

    # 保存统计报告
    with tqdm(total=3, desc="保存报告", unit=" 步") as pbar:
        pbar.set_description("保存单样本报告")
        output_base = r"C:\Users\31078\Desktop\ROAD\data\projection_statistics"
        save_statistics_report(all_stats, output_base, all_timestamps)
        pbar.update(1)

        # 计算全局统计
        pbar.set_description("计算全局统计")
        global_stats = calculate_global_statistics(all_stats, output_base)
        pbar.update(1)

        pbar.set_description("完成")
        pbar.update(1)

    # 最终统计
    print(f"\n{'='*70}")
    print(f"✓ 所有处理完成！")
    print(f"  处理样本数: {len(all_stats)}")
    print(f"  总投影点数: {sum(s.get('projected_count', 0) for s in all_stats)}")
    print(f"  输出位置: {output_base}")
    print(f"{'='*70}")
