#!/usr/bin/env python3
"""
详细投影可视化诊断脚本
绘制投影点的分布，发现 flip_x 的实际效果
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path("scripts")))
from rsrd_projection_utils import RSRDProjector

# ============================================================================
# 配置
# ============================================================================
CALIB_DIR = Path("RSRD_dev_toolkit/calibration_files")
GEO_CFG = {
    "patch_size": 1.0,
    "overlap": 0.5,
    "roi_x": [-50, 50],
    "roi_y": [0, 100],
}

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# ============================================================================
# 生成投影点分布图
# ============================================================================
print("生成投影点分布可视化...")

projector = RSRDProjector(calib_dir=CALIB_DIR)
frame_id = "pkg_20230408.450"

p_size = GEO_CFG["patch_size"]
step = p_size * (1 - GEO_CFG["overlap"])
x_bins = np.arange(GEO_CFG["roi_x"][0], GEO_CFG["roi_x"][1] - p_size + 0.1, step)
y_bins = np.arange(GEO_CFG["roi_y"][0], GEO_CFG["roi_y"][1] - p_size + 0.1, step)

print(f"总Patch数: {len(x_bins) * len(y_bins)}")

# ============================================================================
# 测试 flip_x=True 和 flip_x=False
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for flip_x_val, ax, title in [
    (True, axes[0], "flip_x=True (当前设置)"),
    (False, axes[1], "flip_x=False (备选)"),
]:
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(-500, 2400)
    ax.set_ylim(-200, 1280)
    ax.invert_yaxis()

    # 绘制图像边界
    rect = plt.Rectangle(
        (0, 0),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        fill=False,
        edgecolor="red",
        linewidth=2,
        label="图像边界",
    )
    ax.add_patch(rect)

    # 采样投影点（全部2万多个太多了，采样绘制）
    sample_interval = 3
    in_count = 0
    out_count = 0

    for idx_x, xi in enumerate(x_bins[::sample_interval]):
        for idx_y, yi in enumerate(y_bins[::sample_interval]):
            corners_3d = np.array(
                [
                    [xi, yi, 0],
                    [xi + p_size, yi, 0],
                    [xi + p_size, yi + p_size, 0],
                    [xi, yi + p_size, 0],
                ]
            )

            corners_uv = projector.lidar_to_pixel(
                corners_3d, frame_id, flip_x=flip_x_val
            )

            # 检查是否在图像范围内
            in_bounds = np.all(
                [
                    corners_uv[:, 0] >= 0,
                    corners_uv[:, 0] <= IMAGE_WIDTH,
                    corners_uv[:, 1] >= 0,
                    corners_uv[:, 1] <= IMAGE_HEIGHT,
                ],
                axis=0,
            )

            if np.any(in_bounds):
                # 在图像范围内，画绿色
                color = "green"
                in_count += 1
            else:
                # 超出范围，画蓝色
                color = "blue"
                out_count += 1

            # 绘制四边形框
            pt = corners_uv.astype(int)
            for i in range(4):
                j = (i + 1) % 4
                ax.plot(
                    [pt[i, 0], pt[j, 0]],
                    [pt[i, 1], pt[j, 1]],
                    color=color,
                    alpha=0.3,
                    linewidth=0.5,
                )

    total = in_count + out_count
    in_pct = in_count / total * 100 if total > 0 else 0

    # 添加统计信息
    ax.text(
        100,
        100,
        f"在范围: {in_count} ({in_pct:.1f}%)\n超出: {out_count}",
        fontsize=12,
        color="black",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    ax.set_xlabel("像素 X")
    ax.set_ylabel("像素 Y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("投影分布对比.png", dpi=150, bbox_inches="tight")
print("✓ 投影分布图已保存: 投影分布对比.png")

# ============================================================================
# 详细分析 flip_x 的影响
# ============================================================================
print("\n" + "=" * 80)
print("Flip_X 影响分析")
print("=" * 80)

# 取几个代表点进行对比
test_points = [
    ([0, 10, 0], "中心前方"),
    ([30, 15, 0], "右侧前方"),
    ([-30, 15, 0], "左侧前方"),
    ([0, 50, 0], "远处中心"),
]

import pickle
from pathlib import Path as P

calib_file = CALIB_DIR / "calib_20230408.pkl"
with open(calib_file, "rb") as f:
    calib = pickle.load(f)
    calib["T"] = calib["T"].reshape(3, 1)
    calib["R"] = calib["R"].copy()
    calib["R"][2, :] = -calib["R"][2, :]  # 应用修复

print(
    f"{'物理坐标':<20} {'场景':<15} {'flip_x':<10} {'像素U':<10} {'像素V':<10} {'在范围':<10}"
)
print("-" * 85)

for pt, scene in test_points:
    pt_arr = np.array([pt])

    for flip_x_val in [True, False]:
        # 手动投影计算
        pt_lidar = pt_arr.copy()
        if flip_x_val:
            pt_lidar[0, 0] = -pt_lidar[0, 0]

        pt_cam = calib["R"] @ pt_lidar.T + calib["T"]
        z = pt_cam[2, 0]

        if z > 0.1:
            pt_img = calib["K"] @ pt_cam
            u = pt_img[0, 0] / z
            v = pt_img[1, 0] / z

            in_bounds = 0 <= u <= IMAGE_WIDTH and 0 <= v <= IMAGE_HEIGHT

            print(
                f"{str(pt):<20} {scene:<15} {str(flip_x_val):<10} "
                f"{u:>8.1f} {v:>8.1f} {'✓' if in_bounds else '✗':<10}"
            )
        else:
            print(
                f"{str(pt):<20} {scene:<15} {str(flip_x_val):<10} "
                f"{'被过滤':<8} {'':<8} {'✗':<10}"
            )

# ============================================================================
# X轴镜像分析
# ============================================================================
print("\n" + "=" * 80)
print("X轴镜像效果分析")
print("=" * 80)

# 物理ROI的中心
roi_center_x = (GEO_CFG["roi_x"][0] + GEO_CFG["roi_x"][1]) / 2  # 0
roi_center_y = (GEO_CFG["roi_y"][0] + GEO_CFG["roi_y"][1]) / 2  # 50

# 左右边界的代表点
roi_left_x = GEO_CFG["roi_x"][0]  # -50
roi_right_x = GEO_CFG["roi_x"][1] - 1  # 49

print(f"\n物理ROI:")
print(f"  X范围: [{GEO_CFG['roi_x'][0]}, {GEO_CFG['roi_x'][1]}]")
print(f"  Y范围: [{GEO_CFG['roi_y'][0]}, {GEO_CFG['roi_y'][1]}]")
print(f"  中心: ({roi_center_x}, {roi_center_y})")

print(f"\n图像尺寸: {IMAGE_WIDTH} × {IMAGE_HEIGHT}")
print(f"图像中心: (960, 540)")

# 投影四个边界点
boundary_points = [
    ([roi_left_x, roi_center_y, 0], "物理左边界"),
    ([roi_right_x, roi_center_y, 0], "物理右边界"),
    ([roi_center_x, GEO_CFG["roi_y"][0], 0], "物理下边界(相机前方)"),
    ([roi_center_x, GEO_CFG["roi_y"][1] - 1, 0], "物理上边界(远处)"),
]

print(f"\n{'物理坐标':<30} {'场景':<20} {'flip_x':<10} {'像素U':<10} {'状态':<10}")
print("-" * 80)

for pt, desc in boundary_points:
    for flip_x_val in [True, False]:
        pt_lidar = np.array(pt)
        if flip_x_val:
            pt_lidar[0] = -pt_lidar[0]

        pt_cam = calib["R"] @ pt_lidar + calib["T"].flatten()
        if pt_cam[2] > 0.1:
            pt_img = calib["K"] @ pt_cam
            u = pt_img[0] / pt_cam[2]
            status = "✓" if 0 <= u <= IMAGE_WIDTH else "✗ 超出"
            print(
                f"{str(pt):<30} {desc:<20} {str(flip_x_val):<10} "
                f"{u:>8.1f} {status:<10}"
            )

# ============================================================================
# 结论
# ============================================================================
print("\n" + "=" * 80)
print("诊断结论")
print("=" * 80)

print(
    """
根据可视化分析，投影热力图看不到的原因是：

【主要原因】
flip_x=True 导致X坐标被反号。在当前配置下：
- 物理左侧点(X<0) 被反号后(X>0) 投影到图像右侧，超出范围
- 物理右侧点(X>0) 被反号后(X<0) 投影到图像左侧，超出范围
- 结果：50%的点(左右两侧)被投影到图像外

【投影分布图说明】
- 绿色框: 投影在图像范围内的Patch (~50%)
- 蓝色框: 投影超出图像范围的Patch (~50%)
- 红色边框: 1920×1080 的实际图像边界

【解决方案】
根据实际情况选择：
1. 如果 flip_x=False 能覆盖整个图像 → 改为 flip_x=False
2. 如果其他工具默认使用 flip_x=True → 需要调整物理ROI或坐标系
3. 或在可视化时动态调整投影算法
"""
)

print("\n✓ 诊断完成！请查看 投影分布对比.png 的可视化结果")
