"""
Road-MoME 数据分布诊断探针
功能：扫描已生成的 .npz 文件，统计伪标签阳性率，并分析真实路面厚度分布，从而推导出最科学的 th_anomaly 阈值。
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    npz_dir = Path(cfg["paths"]["output_dir"])
    npz_files = list(npz_dir.glob("pkg_*.npz"))

    if not npz_files:
        print(f"❌ 在 {npz_dir} 中未找到任何 .npz 文件。")
        return

    print(f"🔍 正在扫描 {len(npz_files)} 个底层物理特征包...\n")

    total_valid_patches = 0
    total_positive_labels = 0
    height_diffs = []  # 记录 phys_8d[:, 2] (95% - 5% 分位数差)
    max_min_diffs = []  # 记录 phys_8d[:, 0] (绝对极差)

    # 抽取最多 500 帧进行统计，足以反映总体分布
    sample_files = npz_files[:500]

    for npz_path in tqdm(sample_files, desc="数据解剖中"):
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                meta = data["meta"]
                phys_8d = data["phys_8d"]

            valid_mask = meta[:, 2] == 1.0
            pseudo_label = meta[:, 0]

            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue

            total_valid_patches += len(valid_indices)
            total_positive_labels += np.sum(pseudo_label[valid_indices])

            # 记录有效的厚度特征
            height_diffs.extend(phys_8d[valid_indices, 2])
            max_min_diffs.extend(phys_8d[valid_indices, 0])

        except Exception as e:
            pass

    # ================= 诊断报告 =================
    positive_rate = total_positive_labels / total_valid_patches
    height_diffs = np.array(height_diffs)
    max_min_diffs = np.array(max_min_diffs)

    print("\n" + "=" * 60)
    print(" 🏥 物理数据底层诊断报告")
    print("=" * 60)

    print(f"📌 [标签分布]")
    print(f"有效 Patch 总数: {total_valid_patches}")
    print(f"标记为病害 (Label=1.0) 的数量: {int(total_positive_labels)}")
    print(f"🔴 当前伪标签阳性率: {positive_rate*100:.2f}%")

    if positive_rate > 0.8:
        print("  ⚠️ 致命警告: 阳性率极度异常！你的数据集正处于『全阳性坍塌』状态。")
        print("  说明：目前的阈值配置将所有健康路面都判定为了坑洼。")
    elif positive_rate < 0.01:
        print("  ⚠️ 警告: 阳性率过低 (<1%)。模型可能什么都学不到。")
    else:
        print("  ✅ 阳性率正常 (通常在 5% ~ 20% 之间)。")

    print(f"\n📌 [路面原生厚度分析 (单位: 米)]")
    print(f"--- 绝对极差 (Max-Min, 旧版指标) ---")
    print(f"  平均值: {np.mean(max_min_diffs):.4f}m")
    print(
        f"  90%分位数: {np.percentile(max_min_diffs, 90):.4f}m (90%的路面极差小于此值)"
    )

    print(f"\n--- 95%-5% 分位数高差 (抗噪指标) ---")
    print(f"  平均值: {np.mean(height_diffs):.4f}m")
    print(f"  50%分位数(中位数): {np.percentile(height_diffs, 50):.4f}m")
    print(f"  85%分位数: {np.percentile(height_diffs, 85):.4f}m")
    print(f"  95%分位数: {np.percentile(height_diffs, 95):.4f}m")

    # 科学推导推荐阈值
    # 假设一条道路上大约有 10% 的面积是真实的异常病害，那么 90% 分位数的厚度就是健康路面的极限。
    recommended_th = np.percentile(height_diffs, 90)

    print("\n" + "=" * 60)
    print(" 💡 修正行动建议")
    print("=" * 60)
    print(f"基于当前点云的真实物理分布，为了让约 10% 的严重区域被判定为病害，")
    print(f"请打开 config.yaml，将 geometry 模块中的 th_anomaly 修改为：")
    print(f"👉 th_anomaly: {recommended_th:.3f}")
    print("\n修改完成后，你必须：")
    print("1. 删除 frame_packages_densefull 文件夹下的所有 .npz 文件！")
    print("2. 重新运行 0_master_preprocess.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
