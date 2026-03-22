"""
Road-MoME 推理与可视化引擎 (v11.0 答辩展示版)

布局：2×2 等大四格 + 顶部信息栏
  ┌──────────────────┬──────────────────┐
  │  Frame ID | Phys:x.xx Geom:x.xx Tex:x.xx | Valid:xx/63  │  顶部信息栏
  ├──────────────────┬──────────────────┤
  │  预测概率热力图   │  Phys 权重热力图  │
  │  INFERNO colormap│  门控权重 w[:,0]  │
  ├──────────────────┼──────────────────┤
  │  Geom 权重热力图  │  Tex  权重热力图  │
  │  门控权重 w[:,1]  │  门控权重 w[:,2]  │
  └──────────────────┴──────────────────┘

热力图值：
  - 预测图：sigmoid(final_logit) ∈ [0,1]
  - 专家图：weights[:,:,i] ∈ [0,1]（门控网络 Softmax 输出，三图之和=1）

透明度：colormap 叠加 alpha=0.55（原图纹理清晰透出），轮廓线 1px
"""

import sys
import json
import yaml
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.mome_model import MoMEEngine


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def generate_rois(patch_corners_uv, valid_mask, orig_w, orig_h,
                  target_w=1008, target_h=560):
    rois = []
    scale_x, scale_y = target_w / orig_w, target_h / orig_h
    for i in range(63):
        if valid_mask[i] == 0:
            rois.append([0.0, 0.0, 1.0, 1.0])
        else:
            corners = patch_corners_uv[i]
            x_min = max(0, np.min(corners[:, 0]) * scale_x)
            x_max = min(target_w, np.max(corners[:, 0]) * scale_x)
            y_min = max(0, np.min(corners[:, 1]) * scale_y)
            y_max = min(target_h, np.max(corners[:, 1]) * scale_y)
            if x_max <= x_min or y_max <= y_min:
                rois.append([0.0, 0.0, 1.0, 1.0])
            else:
                rois.append([float(x_min), float(y_min), float(x_max), float(y_max)])
    rois_tensor = torch.tensor(rois, dtype=torch.float32)
    batch_idx   = torch.zeros((63, 1), dtype=torch.float32)
    return torch.cat([batch_idx, rois_tensor], dim=1)


def val_to_inferno_bgr(v: float):
    """
    将 [0,1] 标量映射到 INFERNO colormap，返回 BGR tuple（与 OpenCV 兼容）。
    手工实现，无需 matplotlib，保证无外部依赖。
    控制点来自 matplotlib INFERNO 官方色表。
    """
    stops_rgb = np.array([
        [  0,   0,   4],
        [ 40,  11,  84],
        [101,  21, 110],
        [159,  42,  99],
        [212,  72,  66],
        [245, 125,  21],
        [252, 193,  33],
        [252, 255, 164],
    ], dtype=np.float32)

    v = float(np.clip(v, 0.0, 1.0))
    idx = v * (len(stops_rgb) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(stops_rgb) - 1)
    frac = idx - lo
    rgb = stops_rgb[lo] * (1 - frac) + stops_rgb[hi] * frac
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return (b, g, r)   # OpenCV BGR


def draw_heatmap_panel(base_img_bgr, patch_corners_uv, valid_mask,
                       values, title, alpha=0.55):
    """
    在 base_img_bgr 上叠加 INFERNO 热力图，返回新图像。

    参数：
      base_img_bgr   : 原始图像（H, W, 3），BGR
      patch_corners_uv : (63, 4, 2) patch 角点，原始图像坐标
      valid_mask     : (63,) float，1=有效
      values         : (63,) float，热力值 ∈ [0,1]
      title          : 左上角标题文字
      alpha          : colormap 叠加透明度（越小越透明，原图越清晰）
    """
    img   = base_img_bgr.copy()
    overlay = base_img_bgr.copy()
    h, w  = img.shape[:2]

    for i in range(63):
        if valid_mask[i] < 0.5:
            continue
        corners = patch_corners_uv[i].astype(np.int32)
        color   = val_to_inferno_bgr(values[i])

        # 填充半透明色块
        cv2.fillPoly(overlay, [corners], color)
        # 轮廓线（1px，同色但略亮）
        cv2.polylines(img, [corners], isClosed=True,
                      color=color, thickness=1, lineType=cv2.LINE_AA)

    # 半透明叠加：alpha 控制 colormap 的不透明度
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 左上角标题（带黑色描边保证可读性）
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    pos        = (14, 28)
    cv2.putText(img, title, pos, font, font_scale, (0, 0, 0),    thickness + 2, cv2.LINE_AA)
    cv2.putText(img, title, pos, font, font_scale, (255, 255, 255), thickness,  cv2.LINE_AA)

    return img


def make_colorbar(width, height=20):
    """
    生成一条 INFERNO colorbar，宽度=width，高度=height，BGR 格式。
    附带 0.0 / 0.5 / 1.0 刻度文字。
    """
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        bar[:, x] = val_to_inferno_bgr(x / max(width - 1, 1))

    # 刻度文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    for val, label in [(0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")]:
        x = int(val * (width - 1))
        cv2.putText(bar, label, (max(0, x - 10), height - 4),
                    font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    return bar


# ─────────────────────────────────────────────
# 核心渲染函数
# ─────────────────────────────────────────────

def render_frame(img_bgr, patch_corners_uv, valid_mask,
                 probs, expert_weights, frame_id, out_dir):
    """
    生成 2×2 四格热力图拼接图，带顶部信息栏和 colorbar。

    四格布局（等大）：
      [预测概率]  [Phys 权重]
      [Geom 权重] [Tex  权重]
    """
    orig_h, orig_w = img_bgr.shape[:2]

    # 将图像缩放到统一面板尺寸（保持原始宽高比）
    PANEL_W, PANEL_H = 840, 468
    img_resized = cv2.resize(img_bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)

    # patch 角点同步缩放到面板坐标
    scale_x = PANEL_W / orig_w
    scale_y = PANEL_H / orig_h
    corners_scaled = patch_corners_uv.copy().astype(np.float32)
    corners_scaled[:, :, 0] *= scale_x
    corners_scaled[:, :, 1] *= scale_y
    corners_scaled = corners_scaled.astype(np.int32)

    # ── 计算有效 patch 平均权重（用于顶栏显示）──
    valid_idx = valid_mask > 0.5
    n_valid   = int(valid_idx.sum())
    if n_valid > 0:
        w_phys = float(expert_weights[valid_idx, 0].mean())
        w_geom = float(expert_weights[valid_idx, 1].mean())
        w_tex  = float(expert_weights[valid_idx, 2].mean())
        avg_prob = float(probs[valid_idx].mean())
    else:
        w_phys = w_geom = w_tex = avg_prob = 0.0

    # ── 绘制四个面板 ──
    # 1. 预测概率热力图
    panel_pred = draw_heatmap_panel(
        img_resized, corners_scaled, valid_mask,
        values=probs,
        title=f"Prediction  (avg={avg_prob:.2f})",
        alpha=0.55,
    )

    # 2. Phys 权重热力图（门控权重 w[:,0]）
    panel_phys = draw_heatmap_panel(
        img_resized, corners_scaled, valid_mask,
        values=expert_weights[:, 0],
        title=f"Phys expert  (w={w_phys:.2f})",
        alpha=0.55,
    )

    # 3. Geom 权重热力图（门控权重 w[:,1]）
    panel_geom = draw_heatmap_panel(
        img_resized, corners_scaled, valid_mask,
        values=expert_weights[:, 1],
        title=f"Geom expert  (w={w_geom:.2f})",
        alpha=0.55,
    )

    # 4. Tex 权重热力图（门控权重 w[:,2]）
    panel_tex = draw_heatmap_panel(
        img_resized, corners_scaled, valid_mask,
        values=expert_weights[:, 2],
        title=f"Tex expert   (w={w_tex:.2f})",
        alpha=0.55,
    )

    # ── 2×2 拼接 ──
    row_top    = np.hstack([panel_pred, panel_phys])
    row_bottom = np.hstack([panel_geom, panel_tex])
    grid       = np.vstack([row_top, row_bottom])

    # ── colorbar（贴在 grid 底部）──
    cb_h  = 24
    cb    = make_colorbar(grid.shape[1], height=cb_h)
    cb_bg = np.zeros((cb_h + 4, grid.shape[1], 3), dtype=np.uint8)
    cb_bg[2:2 + cb_h] = cb
    grid  = np.vstack([grid, cb_bg])

    # ── 顶部信息栏 ──
    total_w   = grid.shape[1]
    header_h  = 52
    header    = np.zeros((header_h, total_w, 3), dtype=np.uint8)
    header[:] = (30, 30, 30)

    info_text = (
        f"Frame: {frame_id}    "
        f"Valid patches: {n_valid}/63    "
        f"Phys: {w_phys:.2f}  Geom: {w_geom:.2f}  Tex: {w_tex:.2f}    "
        f"Avg pred prob: {avg_prob:.3f}"
    )
    cv2.putText(header, info_text, (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (220, 220, 220), 1, cv2.LINE_AA)

    # ── 最终画布 ──
    canvas = np.vstack([header, grid])

    # 等比例限宽（防止文件过大）
    MAX_W = 2560
    if canvas.shape[1] > MAX_W:
        ratio  = MAX_W / canvas.shape[1]
        canvas = cv2.resize(canvas, (MAX_W, int(canvas.shape[0] * ratio)),
                            interpolation=cv2.INTER_AREA)

    save_path = Path(out_dir) / f"vis_{frame_id}.jpg"
    cv2.imwrite(str(save_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_abs(p):
        p = str(p)
        if p.startswith("./"): p = p[2:]
        return project_root / p if not Path(p).is_absolute() else Path(p)

    npz_dir     = get_abs(cfg["paths"]["output_dir"])
    img_dir     = get_abs(cfg["paths"]["raw_img_dir"])
    vis_out_dir = get_abs(cfg["paths"].get("vis_output_dir", "./data/inference_vis_v11"))
    vis_out_dir.mkdir(parents=True, exist_ok=True)

    # ── 模型加载 ──
    print("⏳ 初始化 MoME 模型与加载权重...")
    dim_phys = cfg["features"]["phys"]["input_dim"]
    dim_3d   = cfg["features"]["3d"]["input_dim"]
    model    = MoMEEngine(dim_f3_stats=dim_phys, dim_f3_mae=dim_3d).to(device)

    weight_path = get_abs(cfg["paths"]["weights"]["mome_model"])
    if not weight_path.exists():
        print(f"❌ 找不到权重文件: {weight_path}")
        return

    checkpoint       = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict       = checkpoint.get("model_state_dict", checkpoint)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()
    print(f"✅ 权重加载完成: {weight_path.name}  "
          f"(best val F1={checkpoint.get('best_val_f1', 'N/A')})")

    # ── 数据扫描 ──
    img_map    = {p.stem: p for p in img_dir.rglob("*.jpg")}
    npz_files  = sorted(npz_dir.glob("pkg_*.npz"))
    valid_files = [f for f in npz_files if f.stem.replace("pkg_", "") in img_map]

    limit = cfg.get("inference", {}).get("batch_limit", 20)
    if len(valid_files) > limit:
        np.random.seed(42)
        test_files = list(np.random.choice(valid_files, limit, replace=False))
    else:
        test_files = valid_files

    print(f"🚀 开始推理渲染 | 帧数: {len(test_files)} | 输出目录: {vis_out_dir}")

    with torch.no_grad():
        for npz_path in tqdm(test_files, desc="Inference & Render"):
            frame_id = npz_path.stem.replace("pkg_", "")

            with np.load(npz_path, allow_pickle=True) as data:
                f3_stats         = torch.from_numpy(data["phys_8d"]).float().unsqueeze(0).to(device)
                f3_mae           = torch.from_numpy(data["deep_512d"]).float().unsqueeze(0).to(device)
                q_2d             = torch.from_numpy(
                    data.get("quality_2d", np.ones((63, 1), dtype=np.float32))
                ).float().unsqueeze(0).to(device)
                patch_corners_uv = data["patch_corners_uv"]   # (63, 4, 2)，原始图像坐标
                meta             = data["meta"]

            valid_mask = meta[:, 2]   # (63,)

            img_path = img_map[frame_id]
            img_bgr  = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"⚠️ 图像读取失败: {img_path}")
                continue

            orig_h, orig_w = img_bgr.shape[:2]

            # 推理用缩放图
            img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (1008, 560))
            img_tensor  = img_transform(img_resized).unsqueeze(0).to(device)

            rois_tensor        = generate_rois(patch_corners_uv, valid_mask, orig_w, orig_h).to(device)
            valid_mask_tensor  = torch.from_numpy(valid_mask).float().unsqueeze(0).to(device)

            final_logit, internals = model(
                img_tensor, rois_tensor, f3_stats, f3_mae, q_2d, valid_mask_tensor
            )

            # 提取结果
            probs           = torch.sigmoid(final_logit).squeeze(0).cpu().numpy()   # (63,)
            # expert_weights: (63, 3)，三个专家的门控权重
            expert_weights  = internals["weights"].squeeze(0).cpu().numpy()

            render_frame(
                img_bgr          = img_bgr,
                patch_corners_uv = patch_corners_uv,
                valid_mask       = valid_mask,
                probs            = probs,
                expert_weights   = expert_weights,
                frame_id         = frame_id,
                out_dir          = vis_out_dir,
            )

    print(f"\n✨ 渲染完成！共输出 {len(test_files)} 张，保存至: {vis_out_dir}")


if __name__ == "__main__":
    main()