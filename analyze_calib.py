import pickle
import numpy as np
from pathlib import Path


def analyze_calib(pkl_path):
    print(f'\n{"="*70}')
    print(f"文件: {Path(pkl_path).name}")
    print(f'{"="*70}')

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 1. 结构审计
    print(f"\n[步骤1] 结构审计")
    print(f"Keys: {list(data.keys())}")

    # 提取关键矩阵
    if "K" in data:
        K = data["K"]
        print(f"\nK矩阵维度: {K.shape}")
        print(f"K矩阵:\n{K}")

    if "R" in data:
        R = data["R"]
        print(f"\nR矩阵维度: {R.shape}")
        print(f"R矩阵:\n{R}")

    if "T" in data:
        T = data["T"]
        print(f"\nT向量维度: {T.shape}")
        print(f"T向量: {T}")

    # 2. 内参K矩阵检查
    print(f"\n[步骤2] 内参K矩阵检查")
    if "K" in data:
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        status_fx = "✓正常" if 1000 < fx < 3000 else "✗异常"
        status_fy = "✓正常" if 1000 < fy < 3000 else "✗异常"
        print(f"fx (K[0,0]): {fx:.2f} (正常范围: 1000-3000) → {status_fx}")
        print(f"fy (K[1,1]): {fy:.2f} (正常范围: 1000-3000) → {status_fy}")
        print(f"cx (K[0,2]): {cx:.2f} (预期近似值: ~960)")
        print(f"cy (K[1,2]): {cy:.2f} (预期近似值: ~540)")

    # 3. 外参R矩阵深度逻辑审计（最关键）
    print(f"\n[步骤3] 外参R矩阵深度逻辑审计")
    if "R" in data:
        print(f"R矩阵第三行 R[2,:]: {R[2, :]}")
        print(f"R[2,0]: {R[2,0]:.6f}")
        print(f"R[2,1]: {R[2,1]:.6f}")
        print(f"R[2,2]: {R[2,2]:.6f}")

        # 分析第三行的主导元素
        abs_vals = np.abs(R[2, :])
        max_idx = np.argmax(abs_vals)
        max_val = abs_vals[max_idx]

        axis_names = ["X轴(LiDAR)", "Y轴(LiDAR)", "Z轴(LiDAR)"]
        print(f"\n深度映射分析:")
        print(f"主导元素: R[2,{max_idx}] = {R[2,max_idx]:.6f} ({axis_names[max_idx]})")
        print(f"主导元素幅值: {max_val:.6f}")

        if max_val < 0.1:
            print(f"⚠️  警告: 第三行全为极小值，容易导致投影坍塌！")
        else:
            print(f"✓ 第三行有明显的主导元素")

    # 4. 外参T向量检查
    print(f"\n[步骤4] 外参T向量检查")
    if "T" in data:
        T_flat = T.flatten()
        print(f"T向量: [{T_flat[0]:.4f}, {T_flat[1]:.4f}, {T_flat[2]:.4f}]")
        print(f"T[0] (X方向位移): {T_flat[0]:.4f} m")
        print(f"T[1] (Y方向位移): {T_flat[1]:.4f} m")
        print(f"T[2] (Z方向位移): {T_flat[2]:.4f} m")
        print(f"(预期高度轴位移: 1.5-2.2m)")


# 分析两个文件
analyze_calib("RSRD_dev_toolkit/calibration_files/calib_20230408.pkl")
analyze_calib("RSRD_dev_toolkit/calibration_files/calib_20230317.pkl")
