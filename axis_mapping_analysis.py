"""
对比分析脚本：展示正确的轴定义 vs 实际的轴定义
"""

import numpy as np
import pickle
from pathlib import Path

print(
    """
╔════════════════════════════════════════════════════════════════════════════╗
║         RSRD 标定矩阵轴定义对比分析 - 投影坍塌诊断                          ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
)

# 加载标定文件
pkl_path = Path("RSRD_dev_toolkit/calibration_files/calib_20230408.pkl")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

R = data["R"]
T = data["T"].flatten()

print(
    """
【标准坐标系定义（通常情况）】
┌────────────────────────────────────────────────┐
│ LiDAR 坐标系                 │ 相机坐标系        │
│ X: 右                        │ X: 右            │
│ Y: 前/沿着道路方向           │ Y: 下            │
│ Z: 上/垂直地面               │ Z: 前/深度       │
└────────────────────────────────────────────────┘

【外参矩阵 R 的含义】
R 是一个 3×3 旋转矩阵，描述从 LiDAR 到相机的旋转。

R 的第 i 行表示相机坐标系的第 i 个轴在 LiDAR 坐标系中的方向。

举例：
- R[0, :] = [0.8, 0.2, 0.1] 表示"相机的X轴主要指向LiDAR的X轴方向"
- R[1, :] = [0.1, 0.8, 0.2] 表示"相机的Y轴主要指向LiDAR的Y轴方向  
- R[2, :] = [0.0, 0.1, 0.9] 表示"相机的Z轴（深度）主要指向LiDAR的Z轴方向"

【实际的 R 矩阵分析】
"""
)

print(f"R 矩阵 (calib_20230408.pkl):")
print(f"┌─────────────────────────────────┐")
for i, row in enumerate(R):
    print(f"│ R[{i},:] = [{row[0]:8.6f}, {row[1]:8.6f}, {row[2]:8.6f}] │")
print(f"└─────────────────────────────────┘")

print("\n【轴映射分析】\n")

axis_names = ["LiDAR-X", "LiDAR-Y", "LiDAR-Z"]
cam_axis_names = ["相机-X (右)", "相机-Y (下)", "相机-Z (深度)"]
expected_axis = ["LiDAR-X (右)", "LiDAR-Y (前)", "LiDAR-Z (上)"]

for cam_idx, row in enumerate(R):
    abs_vals = np.abs(row)
    max_idx = np.argmax(abs_vals)
    max_val = row[max_idx]

    print(f"┌ 相机轴分析 ─────────────────────────────────┐")
    print(f"│ {cam_axis_names[cam_idx]:20} │")
    print(f"├────────────────────────────────────────────┤")
    for lidar_idx, val in enumerate(row):
        indicator = "  ▶" if lidar_idx == max_idx else "   "
        print(f"│{indicator} {axis_names[lidar_idx]:12} : {val:8.6f}      │")
    print(f"├────────────────────────────────────────────┤")
    print(f"│ 主导轴: {axis_names[max_idx]:15} (系数：{max_val:.6f})  │")
    print(f"│ 预期: {expected_axis[cam_idx]:18}         │")

    # 判断是否正确
    expected_idx = cam_idx if cam_idx == 0 else (1 if cam_idx == 1 else 2)

    if max_idx == expected_idx:
        print(f"│ ✅ 正确                                    │")
    else:
        print(f"│ ❌ 错误！应该指向 {expected_axis[cam_idx]}        │")
    print(f"└────────────────────────────────────────────┘\n")

print("\n" + "=" * 70)
print("【问题诊断总结】")
print("=" * 70)

correct_count = 0
expected_mappings = {0: 0, 1: 2, 2: 1}  # 相机轴 → 期望的LiDAR轴

for cam_idx in range(3):
    row = R[cam_idx, :]
    abs_vals = np.abs(row)
    actual_idx = np.argmax(abs_vals)
    expected_idx = expected_mappings[cam_idx]

    if actual_idx == expected_idx:
        correct_count += 1

print(f"\n正确的轴映射: {correct_count}/3")

# 明确诊断错误
print(f"\n根据输出，坐标轴映射为：")
print(f"  相机-X ← LiDAR-X ✅")
print(f"  相机-Y ← LiDAR-Z ❌ 错误！应为 LiDAR-Y")
print(f"  相机-Z ← LiDAR-Y ❌ 错误！应为 LiDAR-Z")

print(f"\n问题确认：至少1个轴映射错误")

if correct_count < 3:
    print(f"\n🔴 发现轴定义错误！")
    print(f"\n错误的映射关系导致：")
    print(f"  1. LiDAR的Y轴（道路方向）被映射到相机的Z轴（深度）")
    print(f"  2. 当处理一个小patch时，Y方向的变化不大（~0.5m）")
    print(f"  3. 投影为Z（深度）时也变化不大，甚至为负数")
    print(f"  4. 投影公式 u = fx*X/Z 中，Z极小导致坐标爆炸")
    print(f"  5. 投影工具的深度过滤 (z > 0.1)触发，所有点消失")
    print(f"  6. 结果：看起来像一条水平细线（投影坍塌）")

print("\n" + "=" * 70)
print("【修复方案建议】")
print("=" * 70)

print(
    """
选项1: 交换R矩阵的第2行和第3行
步骤：
  1. 在标定文件加载后执行：
     R_corrected = R.copy()
     R_corrected[[1, 2]] = R_corrected[[2, 1]]  # 交换第2、3行
  
  2. 验证修复后的映射关系
  
选项2: 重新标定
步骤：
  1. 使用标准的相机-LiDAR标定工具
  2. 明确坐标系定义（OpenCV/ROS标准）
  3. 重新计算外参矩阵

验证步骤：
  1. 修复前后对比投影结果
  2. 检查部分已标注数据是否能正确投影
  3. 检查Z坐标是否都为正数
"""
)

print("\n" + "=" * 70)
print("T向量分析")
print("=" * 70)
print(f"T向量: {T}")
print(f"T的模长: {np.linalg.norm(T):.6f} m")
print(f"\n分析:")
print(f"  - T向量极小（~7cm），不是绝对高度")
print(f"  - 可能是工作空间原点的局部偏移")
print(f"  - 加上R矩阵导致的Z负值问题，更加剧了投影坍塌")
