"""
[Step 4] MoME 深度诊断可视化 (v8.7 局部空间增强版)

核心适配：
1. 架构同步：完全对齐 v8.7 的 LocalWindowAttention 结构及三专家 (Phys, Geom, Tex) 正交架构。
2. 权重锁定：默认优先加载由 EarlyStopping 产生的 'road_mome_v8_best.pth'。
3. 诊断增强：通过 Visual Quality Prior (q_2d) 热力图，透视局部空间增强后的门控决策逻辑。
"""

import os
import sys
import torch
import numpy as np
import yaml
import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import random

# 环境设置
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.mome_model import MoMEEngine

try:
    from scripts.rsrd_projection_utils import RSRDProjector

    HAS_PROJECTOR = True
except ImportError:
    HAS_PROJECTOR = False


def load_config():
    cfg_path = project_root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
GEO_CFG = cfg["geometry"]
PATH_CFG = cfg["paths"]
INF_CFG = cfg["inference"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_valid_projections(corners_uv, probs, valid_mask, img_w=1920, img_h=1080):
    valid_mask_proj = []
    for i, corners in enumerate(corners_uv):
        if valid_mask[i] == 0:
            valid_mask_proj.append(False)
            continue
        if np.any(np.isnan(corners)):
            valid_mask_proj.append(False)
            continue
        in_bounds = np.sum(
            [
                (0 <= corners[j, 0] <= img_w) and (0 <= corners[j, 1] <= img_h)
                for j in range(len(corners))
            ]
        )
        valid_mask_proj.append(in_bounds > 0)
    valid_mask_proj = np.array(valid_mask_proj)
    return corners_uv[valid_mask_proj], probs[valid_mask_proj]


def reconstruct_grid(values, num_cols=7, num_rows=9, bg_value=np.nan):
    """将 1D 数组 (63,) 还原为 2D 矩阵 (9, 7) 以供渲染"""
    grid = np.full((num_rows, num_cols), bg_value, dtype=np.float32)
    idx = 0
    for i in range(num_cols):
        for j in range(num_rows):
            if idx < len(values):
                grid[j, i] = values[idx]
            idx += 1
    return grid


def generate_2d_projection_overlay(
    img_path, patch_corners_uv, probs, frame_id, threshold=0.2
):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img_h, img_w = img.shape[:2]
    overlay = img.copy()

    for i, corners in enumerate(patch_corners_uv):
        prob = probs[i]
        if prob < threshold:
            continue
        valid_points = np.sum(
            [
                (0 <= corners[j, 0] <= img_w) and (0 <= corners[j, 1] <= img_h)
                for j in range(len(corners))
            ]
        )
        if valid_points < 3:
            continue

        pts_clipped = corners.copy()
        pts_clipped[:, 0] = np.clip(pts_clipped[:, 0], 0, img_w)
        pts_clipped[:, 1] = np.clip(pts_clipped[:, 1], 0, img_h)

        # 橙色色调表示检测结果
        color = (0, int(255 * (1 - prob)), 255)
        pts = pts_clipped.astype(np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)

        if prob > 0.5:
            center = np.mean(pts_clipped, axis=0).astype(int)
            text_x = max(0, min(center[0] - 15, img_w - 50))
            text_y = max(15, min(center[1], img_h - 5))
            cv2.putText(
                img,
                f"{prob:.2f}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    combined = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)


def run_diagnostic():
    feat_cfg = cfg.get("features", {})
    dim_phys = feat_cfg.get("phys", {}).get("input_dim", 8)
    dim_3d = feat_cfg.get("3d", {}).get("input_dim", 384)
    dim_2d = feat_cfg.get("2d", {}).get("input_dim", 12288)

    # 1. 实例化包含局部注意力的 v8.7 模型
    model = MoMEEngine(dim_f3_stats=dim_phys, dim_f3_mae=dim_3d, dim_f2_dino=dim_2d).to(
        DEVICE
    )

    # 2. 权重加载策略：优先加载 best，其次加载最新 epoch
    ckpt_dir = project_root / "checkpoints"
    best_path = ckpt_dir / "road_mome_v8_best.pth"

    if best_path.exists():
        model_path = best_path
    else:
        ckpt_files = sorted(list(ckpt_dir.glob("road_mome_v7_1_ep*.pth")))
        if not ckpt_files:
            print(f"❌ 找不到任何权重文件于: {ckpt_dir}")
            return
        model_path = ckpt_files[-1]

    print(f"📦 加载权重: {model_path.name} (v8.7 局部增强架构)")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. 索引与真值加载
    img_map = {}
    index_path = Path(PATH_CFG["output_dir"]).parent / "dataset_index.yaml"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            img_map = {item["id"]: item["img"] for item in yaml.safe_load(f)}

    manual_gt_dict = {}
    json_path = Path(
        PATH_CFG.get(
            "manual_label_path",
            project_root / "data" / "manual_visual_gt_Annotatordense.json",
        )
    )
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            manual_gt_dict = json.load(f)
            print(f"📂 已加载全量标注记录: {len(manual_gt_dict)} 帧。")

    all_npz = sorted(list(Path(PATH_CFG["output_dir"]).glob("pkg_*.npz")))
    if INF_CFG["mode"] == "select" and INF_CFG.get("select_files"):
        target_ids = INF_CFG["select_files"]
        npz_files = [p for p in all_npz if any(tid in p.name for tid in target_ids)]
    elif INF_CFG["mode"] == "random":
        npz_files = random.sample(all_npz, min(len(all_npz), INF_CFG["batch_limit"]))
    else:
        npz_files = all_npz[: INF_CFG.get("batch_limit", 15)]

    extent = [
        GEO_CFG["roi_x"][0],
        GEO_CFG["roi_x"][1],
        GEO_CFG["roi_y"][0],
        GEO_CFG["roi_y"][1],
    ]
    vis_dir = Path(PATH_CFG["vis_output_dir"])
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 诊断引擎启动 | 样本数: {len(npz_files)}")

    for p in tqdm(npz_files, desc="Inference"):
        try:
            data = np.load(p, allow_pickle=True)
            frame_id = p.stem.replace("pkg_", "")

            patch_corners_uv = data.get("patch_corners_uv", None)
            phys = torch.from_numpy(data["phys_8d"]).float().unsqueeze(0).to(DEVICE)
            geom = torch.from_numpy(data["deep_512d"]).float().unsqueeze(0).to(DEVICE)
            tex = torch.from_numpy(data["deep_2d_768d"]).float().unsqueeze(0).to(DEVICE)
            q_2d = torch.from_numpy(data["quality_2d"]).float().unsqueeze(0).to(DEVICE)

            meta = data["meta"]
            valid_mask = meta[:, 2]

            with torch.no_grad():
                final_logit, internals = model(phys, geom, tex, q_2d)
                probs_raw = (
                    torch.sigmoid(final_logit).squeeze(-1).squeeze(0).cpu().numpy()
                )
                probs_masked = probs_raw * valid_mask

                # 获取三个专家的权重分布 (Phys, Geom, Tex)
                w_all = internals["weights"].squeeze(0).cpu().numpy()

            # 判定主导专家索引
            dominant_idx = np.argmax(w_all, axis=1).astype(float)
            dominant_idx[valid_mask == 0] = -1.0

            overlay_img = None
            if frame_id in img_map and patch_corners_uv is not None:
                pc_filtered, pr_filtered = filter_valid_projections(
                    patch_corners_uv, probs_masked, valid_mask, img_w=1920, img_h=1080
                )
                overlay_img = generate_2d_projection_overlay(
                    img_map[frame_id], pc_filtered, pr_filtered, frame_id
                )

            # --- 绘图布局 ---
            fig, axes = plt.subplots(2, 4, figsize=(24, 11), dpi=100)
            q2_avg = q_2d.mean().item()
            fig.suptitle(
                f"MI-MoE Diagnostic (v8.7 Local Space Enhanced) | {p.stem}\nVisual Quality Avg: {q2_avg:.3f}",
                fontsize=18,
                fontweight="bold",
            )

            # [0, 0] 最终融合概率
            probs_disp = probs_raw.copy()
            probs_disp[valid_mask == 0] = np.nan
            im1 = axes[0, 0].imshow(
                reconstruct_grid(probs_disp),
                extent=extent,
                cmap="jet",
                origin="lower",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            axes[0, 0].set_title("1. Detection Prob (Fusion)", fontsize=12)
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

            # [0, 1] 标注真值
            frame_img_name = f"{frame_id}.jpg"
            if frame_img_name in manual_gt_dict:
                target_array = np.array(
                    manual_gt_dict[frame_img_name], dtype=np.float32
                )
                target_title = "2. Target (Manual GT)"
            else:
                target_array = meta[:, 0]
                target_title = "2. Target (Pseudo-3D)"

            target_disp = target_array.copy()
            target_disp[valid_mask == 0] = np.nan
            im2 = axes[0, 1].imshow(
                reconstruct_grid(target_disp),
                extent=extent,
                cmap="Reds",
                origin="lower",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            axes[0, 1].set_title(
                target_title,
                fontsize=12,
                fontweight="bold" if frame_img_name in manual_gt_dict else "normal",
            )
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

            # [0, 2] 主导专家
            expert_cmap = ListedColormap(["#2c3e50", "#bdc3c7", "#3498db", "#f1c40f"])
            im3 = axes[0, 2].imshow(
                reconstruct_grid(dominant_idx, bg_value=-1),
                extent=extent,
                cmap=expert_cmap,
                origin="lower",
                vmin=-1.5,
                vmax=2.5,
                aspect="auto",
            )
            axes[0, 2].set_title("3. Dominant Expert", fontsize=12)
            cbar3 = plt.colorbar(
                im3, ax=axes[0, 2], fraction=0.046, pad=0.04, ticks=[-1, 0, 1, 2]
            )
            cbar3.ax.set_yticklabels(["Blind", "Phys", "3D-Geom", "2D-Tex"])

            # [0, 3] 活跃权重柱状图
            if valid_mask.sum() > 0:
                avg_w = (w_all * valid_mask[:, None]).sum(axis=0) / valid_mask.sum()
            else:
                avg_w = np.zeros(3)
            axes[0, 3].bar(
                ["Phys", "Geom", "Tex"], avg_w, color=["#bdc3c7", "#3498db", "#f1c40f"]
            )
            axes[0, 3].set_ylim(0, 1.0)
            axes[0, 3].set_title("4. Active Weight Dist", fontsize=12)
            for i, v in enumerate(avg_w):
                axes[0, 3].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

            # [1, 0] 物理专家权重 (Phys)
            w_phys_disp = w_all[:, 0].copy()
            w_phys_disp[valid_mask == 0] = np.nan
            im4 = axes[1, 0].imshow(
                reconstruct_grid(w_phys_disp),
                extent=extent,
                cmap="Greens",
                origin="lower",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            axes[1, 0].set_title("5. Phys Weight", fontsize=12)
            plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

            # [1, 1] 几何专家权重 (Geom)
            w_geom_disp = w_all[:, 1].copy()
            w_geom_disp[valid_mask == 0] = np.nan
            im5 = axes[1, 1].imshow(
                reconstruct_grid(w_geom_disp),
                extent=extent,
                cmap="Blues",
                origin="lower",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            axes[1, 1].set_title("6. 3D-Geom Weight", fontsize=12)
            plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

            # [1, 2] 纹理专家权重 (Tex)
            w_tex_disp = w_all[:, 2].copy()
            w_tex_disp[valid_mask == 0] = np.nan
            im6 = axes[1, 2].imshow(
                reconstruct_grid(w_tex_disp),
                extent=extent,
                cmap="YlOrBr",
                origin="lower",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            axes[1, 2].set_title("7. 2D-Tex Weight", fontsize=12)
            plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

            # [1, 3] 视觉质量先验评分 (q_2d)
            q2_disp = q_2d.squeeze(-1).squeeze(0).cpu().numpy().copy()
            q2_disp[valid_mask == 0] = np.nan
            im7 = axes[1, 3].imshow(
                reconstruct_grid(q2_disp),
                extent=extent,
                cmap="magma",
                origin="lower",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            axes[1, 3].set_title("8. Visual Quality Prior (q_2d)", fontsize=12)
            plt.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04)

            # 坐标轴标签美化
            for ax in axes.flat:
                if not isinstance(ax, plt.Axes) or "Weight Dist" in ax.get_title():
                    continue
                ax.set_xlabel("X (m)", fontsize=8)
                ax.set_ylabel("Y (m)", fontsize=8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(vis_dir / f"full_diag_{frame_id}.png", dpi=120)
            plt.close()

            # 保存 2D 投影叠加图
            if overlay_img is not None:
                plt.figure(figsize=(16, 9))
                plt.imshow(overlay_img)
                plt.title(
                    f"2D Spatial Consistency Projection | Frame: {frame_id}",
                    fontsize=14,
                )
                plt.axis("off")
                plt.savefig(
                    vis_dir / f"projected_overlay_{frame_id}.jpg",
                    bbox_inches="tight",
                    dpi=150,
                )
                plt.close()

        except Exception as e:
            print(f"❌ 帧 {p.name} 处理失败: {str(e)}")
            import traceback

            traceback.print_exc()

    print(f"✨ v8.7 诊断报告生成完毕，位置: {vis_dir}")


if __name__ == "__main__":
    run_diagnostic()
