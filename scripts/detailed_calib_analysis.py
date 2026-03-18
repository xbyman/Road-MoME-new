import pickle
import numpy as np
from pathlib import Path


def detailed_projection_analysis(pkl_path):
    print(f'\n{"="*70}')
    print(f"文件: {Path(pkl_path).name}")
    print(f'{"="*70}')

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    K = data["K"]
    R = data["R"]
    T = data["T"].flatten()

    # 制造测试点：模拟物理框的四个角点
    # 假设在LiDAR坐标系中，一个patch的四个顶点
    patch_x_min, patch_x_max = 0.0, 0.5
    patch_y_min, patch_y_max = 0.0, 0.5
    z_height = 0.0

    test_points_lidar = np.array(
        [
            [patch_x_min, patch_y_min, z_height],
            [patch_x_max, patch_y_min, z_height],
            [patch_x_max, patch_y_max, z_height],
            [patch_x_min, patch_y_max, z_height],
        ]
    )

    print(f"\n[测试1] 投影一个物理patch的四个顶点")
    print(f"LiDAR坐标系中的测试点:")
    for i, pt in enumerate(test_points_lidar):
        print(f"  点{i+1}: {pt}")

    # 投影到相机坐标系
    P = K @ (R @ test_points_lidar.T + T.reshape(-1, 1))

    print(f"\nP矩阵 (N x 3中间过程):")
    print(f"P = K @ (R @ points_lidar.T + T)")
    print(f"P shape: {P.shape}")
    print(f"P:\n{P}")

    # 计算像素坐标
    cam_xyz = R @ test_points_lidar.T + T.reshape(-1, 1)
    print(f"\n相机坐标系中的点 (3D):")
    print(f"cam_xyz:\n{cam_xyz}")

    # 归一化投影
    uv = P[:2, :] / P[2, :]  # 像素坐标

    print(f"\n像素坐标 (u, v):")
    for i, (u, v) in enumerate(uv.T):
        print(f"  点{i+1}: ({u:.2f}, {v:.2f})")

    # 检查Z坐标是否都很小（这会导致投影坍塌）
    print(f"\n相机Z坐标分析:")
    z_cam = cam_xyz[2, :]
    print(f"Z坐标: {z_cam}")
    print(f"Z坐标范围: [{np.min(z_cam):.6f}, {np.max(z_cam):.6f}]")
    print(f"Z坐标标准差: {np.std(z_cam):.6f}")

    if np.max(np.abs(z_cam)) < 0.1:
        print(f"⚠️  **严重警告**: Z坐标极小，所有点都几乎在同一深度！")
        print(f"这会导致投影坍塌 - 四个顶点的像素坐标会几乎相同或形成水平线!")

    # 分析像素点的分布
    print(f"\n像素坐标分布分析:")
    u_coords = uv[0, :]
    v_coords = uv[1, :]
    print(
        f"U坐标范围: [{np.min(u_coords):.2f}, {np.max(u_coords):.2f}], 跨度: {np.max(u_coords)-np.min(u_coords):.2f}"
    )
    print(
        f"V坐标范围: [{np.min(v_coords):.2f}, {np.max(v_coords):.2f}], 跨度: {np.max(v_coords)-np.min(v_coords):.2f}"
    )

    if (np.max(u_coords) - np.min(u_coords)) < 5 or (
        np.max(v_coords) - np.min(v_coords)
    ) < 5:
        print(f"⚠️  **问题**: 投影点群非常紧凑，形成水平或竖直的细线!")

    # [关键诊断] 分析T向量的物理意义
    print(f"\n[测试2] T向量物理意义分析")
    print(f"T向量 (相机相对于LiDAR的位移): {T}")
    print(f"T的模长: {np.linalg.norm(T):.6f} m")

    T_expected_height = 1.8  # 预期高度
    if np.linalg.norm(T) < 0.1:
        print(f"⚠️  **严重问题**: T向量模长仅 {np.linalg.norm(T):.6f} m!")
        print(f"这远小于预期的相机高度 {T_expected_height} m")
        print(f"可能的原因：")
        print(f'  1. 标定文件中的T是"局部"位移而非绝对高度')
        print(f"  2. 坐标系定义错误（轴交换）")
        print(f"  3. 标定文件的单位不一致")

    # [关键诊断2] R矩阵的分析
    print(f"\n[测试3] R矩阵的坐标轴映射分析")
    print(f"R矩阵代表LiDAR坐标系到相机坐标系的旋转:")
    print(f"R第1行（对应相机X轴）: {R[0, :]} ← LiDAR哪个轴投射到相机X？")
    print(f"R第2行（对应相机Y轴）: {R[1, :]} ← LiDAR哪个轴投射到相机Y？")
    print(f"R第3行（对应相机Z轴/深度）: {R[2, :]} ← **关键：LiDAR哪个轴成为相机深度**")

    # 识别主导轴
    lidar_axes = ["X", "Y", "Z"]
    for cam_idx, row in enumerate(R):
        abs_vals = np.abs(row)
        max_idx = np.argmax(abs_vals)
        max_val = row[max_idx]
        print(
            f'  相机{["X", "Y", "Z"][cam_idx]}轴 主要来自 LiDAR-{lidar_axes[max_idx]}轴 (系数: {max_val:.6f})'
        )

    # [关键诊断3] 投影矩阵 P = K[R|T] 的秩
    print(f"\n[测试4] 投影矩阵秩分析")
    P_matrix = K @ np.hstack([R, T.reshape(-1, 1)])
    rank = np.linalg.matrix_rank(P_matrix)
    print(f"投影矩阵P (K[R|T]) 的秩: {rank}/3")
    if rank < 3:
        print(f"⚠️  **警告**: 投影矩阵秩不足！这会导致退化投影。")
    else:
        print(f"✓ 投影矩阵秩正常")

    # 条件数
    cond = np.linalg.cond(K @ R)
    print(f"K@R的条件数: {cond:.2f} (越小越稳定)")


# 分析两个文件
for pkl_file in ["calib_20230408.pkl", "calib_20230317.pkl"]:
    detailed_projection_analysis(f"RSRD_dev_toolkit/calibration_files/{pkl_file}")

print(f'\n\n{"="*70}')
print(f"[综合诊断结论]")
print(f'{"="*70}')
print(
    f"""
投影坍塌的根本原因通常是：
1. T向量过小 → 所有物理点在相机坐标系中Z值都非常小
   → 导致透视投影时所有点都集中在一起
   → 表现为在图像上形成细线

2. R矩阵映射错误 → LiDAR的Z轴没有正确映射到相机的Z轴
   → 导致物理高度差被忽视

3. 坐标系混乱 → 当LiDAR的Y轴被强制映射到相机Z轴时
   → 如果LiDAR Y轴变化小，相机Z也变化小

根据本次分析的T向量极小值（~0.07m），最可能的原因是：
T向量不代表实际相机高度，而是局部偏移量！
"""
)
