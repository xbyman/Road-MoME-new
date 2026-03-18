"""
Road-MoME 稀疏点云可用性体检脚本 (对齐 master_preprocess)

核心物理逻辑：
1. 绝对对齐：使用与 0_master_preprocess 相同的 Open3D 读取逻辑。
2. 空间密度透视：不仅统计合格率，更叠加全局空间热力图，诊断激光雷达的物理扫描盲区。
"""

import sys
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

# ==========================================
# 1. 动态读取 Config (确保与训练网格绝对对齐)
# ==========================================
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
GEO_CFG = cfg["geometry"]
ROI_X = GEO_CFG["roi_x"]
ROI_Y = GEO_CFG["roi_y"]
NUM_COLS = 7  # X 轴切分
NUM_ROWS = 9  # Y 轴切分
TOTAL_PATCHES = NUM_COLS * NUM_ROWS

# 判定阈值 (工程师需根据业务容忍度微调)
MIN_POINTS_PER_PATCH = 15  # 一个网格至少需要 15 个点才能计算可靠的 Z 轴高度方差
MIN_VALID_PATCH_RATIO = (
    0.70  # 一帧画面中，必须有 70% 的网格是“亮”的（点数达标），否则 3D 报废
)


# ==========================================
# 2. 核心对接区: 严格对齐 0_master_preprocess 的点云加载
# ==========================================
def load_point_cloud(pcd_path):
    """
    使用与 0_master_preprocess 相同的 Open3D 底层逻辑。
    """
    try:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts_raw = np.asarray(pcd.points)
        return pts_raw
    except Exception as e:
        print(f"\n❌ 读取点云失败 {pcd_path.name}: {e}")
        return np.empty((0, 3))


# ==========================================
# 3. 物理体检核心引擎
# ==========================================
def evaluate_frame(points):
    """
    对单帧点云进行 ROI 过滤与网格密度体检
    """
    total_raw_points = len(points)
    if total_raw_points == 0:
        return 0, 0, np.zeros((NUM_ROWS, NUM_COLS)), 0, 0.0

    # 1. ROI 空间过滤 (只看车头前方的感兴趣区域)
    mask = (
        (points[:, 0] >= ROI_X[0])
        & (points[:, 0] <= ROI_X[1])
        & (points[:, 1] >= ROI_Y[0])
        & (points[:, 1] <= ROI_Y[1])
    )
    roi_points = points[mask]
    roi_count = len(roi_points)

    if roi_count == 0:
        return total_raw_points, roi_count, np.zeros((NUM_ROWS, NUM_COLS)), 0, 0.0

    # 2. 网格化统计 (利用 2D 直方图瞬间完成 63 个格子的点数统计)
    x_edges = np.linspace(ROI_X[0], ROI_X[1], NUM_COLS + 1)
    y_edges = np.linspace(ROI_Y[0], ROI_Y[1], NUM_ROWS + 1)

    # hist shape: (9, 7)
    hist, _, _ = np.histogram2d(
        roi_points[:, 1], roi_points[:, 0], bins=[y_edges, x_edges]
    )

    # 3. 计算体检指标
    valid_patches_count = np.sum(hist >= MIN_POINTS_PER_PATCH)
    valid_patch_ratio = valid_patches_count / TOTAL_PATCHES

    return total_raw_points, roi_count, hist, valid_patches_count, valid_patch_ratio


def main():
    # ⚠️ 请将此处替换为你那 7000 帧稀疏点云的根目录
    # 如果放在 cfg 配置的目录下，可直接运行
    pcd_dir = Path(cfg["paths"]["raw_pcd_dir"])

    print(f"🔍 正在扫描点云目录: {pcd_dir}")
    pcd_files = sorted(list(pcd_dir.rglob("*.pcd")))

    if not pcd_files:
        print("❌ 未找到点云文件。请检查路径是否正确。")
        return

    print("=" * 60)
    print(f"🔬 开始对 {len(pcd_files)} 帧点云进行物理密度体检...")
    print(
        f"📐 规则: 网格至少 {MIN_POINTS_PER_PATCH} 点合格，单帧合格率 >= {MIN_VALID_PATCH_RATIO*100:.0f}% 为可用。"
    )
    print("=" * 60)

    stats = {"gold": [], "degraded": [], "trash": []}

    # 【新增统计维度】
    roi_counts_log = []
    ratios_log = []
    global_hist_sum = np.zeros((NUM_ROWS, NUM_COLS))  # 用于统计全数据集的热力分布

    for p in tqdm(pcd_files, desc="Evaluating Sparsity"):
        points = load_point_cloud(p)
        raw_cnt, roi_cnt, hist, valid_p, valid_ratio = evaluate_frame(points)

        roi_counts_log.append(roi_cnt)
        ratios_log.append(valid_ratio)
        global_hist_sum += hist

        # 核心分级逻辑
        if roi_cnt < 300:
            stats["trash"].append(p.name)
        elif valid_ratio >= MIN_VALID_PATCH_RATIO:
            stats["gold"].append(p.name)
        else:
            stats["degraded"].append(p.name)

    # ==========================================
    # 4. 生成诊断报告与全局热力图
    # ==========================================
    print("\n📊 =============== 体检报告 =============== 📊")
    print(f"总样本数: {len(pcd_files)}")
    print(
        f"🟢 Gold (多模态优质帧)   : {len(stats['gold'])} 帧 ({len(stats['gold'])/len(pcd_files)*100:.1f}%)"
    )
    print(
        f"🟡 Degraded (3D残缺帧)   : {len(stats['degraded'])} 帧 ({len(stats['degraded'])/len(pcd_files)*100:.1f}%)"
    )
    print(
        f"🔴 Trash (雷达瞎子帧)     : {len(stats['trash'])} 帧 ({len(stats['trash'])/len(pcd_files)*100:.1f}%)"
    )

    # 统计更严谨的中位数与分位数 (揭露均值的欺骗性)
    print(f"\n📌 空间落点统计 (ROI 内):")
    print(f"  平均点数: {np.mean(roi_counts_log):.0f} 点")
    print(f"  中位数  : {np.median(roi_counts_log):.0f} 点 (50% 的帧少于此点数)")
    print(f"  均合格率: {np.mean(ratios_log)*100:.1f}%")

    # 全局平均每个格子的点云数量
    avg_patch_density = global_hist_sum / max(1, len(pcd_files))

    # 绘制可视化诊断图表
    vis_output = Path(cfg["paths"]["vis_output_dir"])
    vis_output.mkdir(parents=True, exist_ok=True)
    report_path = vis_output / "sparsity_report.png"

    # 新增为 1x3 布局，包含热力图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图 1: ROI 总点数分布
    axes[0].hist(roi_counts_log, bins=50, color="skyblue", edgecolor="black")
    axes[0].axvline(x=300, color="r", linestyle="--", label="Trash Threshold (300)")
    axes[0].set_title("ROI Point Count Distribution")
    axes[0].set_xlabel("Total Points in 6x7m ROI")
    axes[0].set_ylabel("Frame Count")
    axes[0].legend()

    # 图 2: 合格网格率分布
    axes[1].hist(ratios_log, bins=50, color="lightgreen", edgecolor="black")
    axes[1].axvline(
        x=MIN_VALID_PATCH_RATIO,
        color="r",
        linestyle="--",
        label=f"Gold Threshold ({MIN_VALID_PATCH_RATIO:.2f})",
    )
    axes[1].set_title("Valid Patch Ratio Distribution")
    axes[1].set_xlabel("Ratio of Valid Patches (>= 15 pts)")
    axes[1].set_ylabel("Frame Count")
    axes[1].legend()

    # 图 3: [新增] 全局空间密度热力图
    # x 轴对应车辆左右，y 轴对应车辆纵深
    im = axes[2].imshow(
        avg_patch_density,
        cmap="viridis",
        origin="lower",
        extent=[ROI_X[0], ROI_X[1], ROI_Y[0], ROI_Y[1]],
    )
    axes[2].set_title("Average Spatial Density (Pts/Patch)")
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Y (m)")
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Average Points")

    plt.tight_layout()
    plt.savefig(report_path, dpi=150)
    print(f"\n📈 深度体检报告与【空间热力分布图】已保存至: {report_path}")
    print(
        "💡 架构师提示: 请重点观察热力图。如果远处 (Y接近-8) 的网格呈现深蓝色 (点数极低)，说明低线束雷达存在物理盲区，必须在训练中引入掩码！"
    )


if __name__ == "__main__":
    main()
