"""
Road-MoME 推理与可视化引擎 (v10.6 精准索引与绝对路径锚定版)

核心重构：
1. 索引制导加载：彻底废弃 rglob 盲搜，优先读取 dataset_index 标准索引，确保图像路径 100% 精准。
2. 绝对路径锚定：引入 get_abs_path 机制，免疫终端 CWD (当前工作目录) 错位导致的文件找不到问题。
3. 动态路径挂载: 修复 JSON 真值路径硬编码，完美接入 config.yaml，渲染对比图更严谨。
"""

import os
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

def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_rois(patch_corners_uv, valid_mask, orig_w, orig_h, target_w=1008, target_h=560):
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
    batch_idx = torch.zeros((63, 1), dtype=torch.float32)
    return torch.cat([batch_idx, rois_tensor], dim=1)

def draw_inference_result(img_bgr, patch_corners_uv, valid_mask, preds, gt_labels, weights, frame_id, out_dir):
    """
    像素级复刻标注工具的视觉效果 (0.1 Alpha, 1px线宽)
    同时渲染 Ground Truth (左) 和 Prediction (右) 以便排查对比
    """
    h, w = img_bgr.shape[:2]
    
    img_gt = img_bgr.copy()
    img_pred = img_bgr.copy()
    overlay_gt = img_bgr.copy()
    overlay_pred = img_bgr.copy()
    
    for i in range(63):
        if valid_mask[i] == 0: continue
        
        corners = patch_corners_uv[i].astype(np.int32)
        
        # --- 绘制 Ground Truth ---
        is_gt_anomaly = gt_labels[i] > 0.5
        color_gt = (0, 0, 255) if is_gt_anomaly else (0, 255, 0)
        cv2.fillPoly(overlay_gt, [corners], color_gt)
        cv2.polylines(img_gt, [corners], isClosed=True, color=color_gt, thickness=1, lineType=cv2.LINE_AA)
        
        # --- 绘制 Prediction ---
        is_pred_anomaly = preds[i] > 0.5
        color_pred = (0, 0, 255) if is_pred_anomaly else (0, 255, 0)
        cv2.fillPoly(overlay_pred, [corners], color_pred)
        cv2.polylines(img_pred, [corners], isClosed=True, color=color_pred, thickness=2, lineType=cv2.LINE_AA)

    # 弱透明度混合 (10%)
    cv2.addWeighted(overlay_gt, 0.1, img_gt, 0.9, 0, img_gt)
    cv2.addWeighted(overlay_pred, 0.1, img_pred, 0.9, 0, img_pred)

    valid_idx = valid_mask > 0.5
    if valid_idx.sum() > 0:
        w_p = weights[0, valid_idx, 0].mean().item()
        w_g = weights[0, valid_idx, 1].mean().item()
        w_t = weights[0, valid_idx, 2].mean().item()
    else:
        w_p, w_g, w_t = 0.0, 0.0, 0.0
        
    text_pred = f"Prediction | Phys:{w_p:.2f} Geom:{w_g:.2f} Tex(2D):{w_t:.2f}"
    
    header_h = 60
    header_gt = np.zeros((header_h, w, 3), dtype=np.uint8)
    header_gt[:] = (45, 45, 45)
    header_pred = np.zeros((header_h, w, 3), dtype=np.uint8)
    header_pred[:] = (45, 45, 45)
    
    cv2.putText(header_gt, "Ground Truth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header_pred, text_pred, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    panel_gt = np.vstack([header_gt, img_gt])
    panel_pred = np.vstack([header_pred, img_pred])
    
    final_canvas = np.hstack([panel_gt, panel_pred])
    
    MAX_OUT_WIDTH = 2560 
    final_h, final_w = final_canvas.shape[:2]
    if final_w > MAX_OUT_WIDTH:
        scale_ratio = MAX_OUT_WIDTH / final_w
        new_h = int(final_h * scale_ratio)
        final_canvas = cv2.resize(final_canvas, (MAX_OUT_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    save_path = Path(out_dir) / f"vis_{frame_id}_pred.jpg"
    cv2.imwrite(str(save_path), final_canvas)

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ====================================================
    # 【核心修复区】：绝对路径锚定机制 (与训练脚本保持一致)
    # ====================================================
    def get_abs_path(p_str):
        p_str = str(p_str)
        if p_str.startswith("./"):
            return project_root / p_str[2:]
        if Path(p_str).is_absolute():
            return Path(p_str)
        return project_root / p_str

    npz_dir = get_abs_path(cfg["paths"]["output_dir"])
    img_dir = get_abs_path(cfg["paths"]["raw_img_dir"])
    vis_out_dir = get_abs_path(cfg["paths"].get("vis_output_dir", "./data/inference_vis_v9.8"))
    json_gt_path = get_abs_path(cfg["paths"].get("manual_label_path", "./data/merged_visual_gt.json"))
    
    vis_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 挂载人工真值库
    manual_gt_dict = {}
    if json_gt_path.exists():
        with open(json_gt_path, "r", encoding="utf-8") as f:
            manual_gt_dict = json.load(f)
        print(f"✅ 成功加载人工真值标签库: {json_gt_path.name} (共 {len(manual_gt_dict)} 帧)")
    else:
        print(f"⚠️ 警告: 未找到人工标签库 {json_gt_path}，将全部使用 3D 伪标签！")
    # ====================================================

    print("⏳ 正在初始化 MoME E2E 架构与加载权重...")
    dim_phys = cfg["features"]["phys"]["input_dim"]
    dim_3d = cfg["features"]["3d"]["input_dim"]
    model = MoMEEngine(dim_f3_stats=dim_phys, dim_f3_mae=dim_3d).to(device)
    
    weight_path = get_abs_path(cfg["paths"]["weights"]["mome_model"])
    if not weight_path.exists():
        print(f"❌ 找不到最优权重文件: {weight_path}")
        return
        
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    # ==========================================================
    # 【核心重构：精准索引制导加载机制】
    # ==========================================================
    img_map = {}
    
    # 扩大索引文件的搜索半径，兼容根目录与 data 目录
    possible_index_paths = [
        project_root / "data" / "dataset_index.yaml",
        project_root / "data" / "dataset_index.json",
        project_root / "dataset_index.yaml",
        project_root / "dataset_index.json"
    ]
    
    index_loaded = False
    for path in possible_index_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix == ".yaml":
                    index_data = yaml.safe_load(f)
                else:
                    index_data = json.load(f)
                # 强制转 str 防止 YAML 解析错误
                img_map = {str(item["id"]): Path(item["img"]) for item in index_data}
            print(f"📖 成功挂载标准数据集索引: {path} (共映射 {len(img_map)} 帧)")
            index_loaded = True
            break
            
    if not index_loaded:
        print("⚠️ 未检测到 dataset_index.yaml/json，回退到 rglob 盲搜模式...")
        all_imgs = list(img_dir.rglob("*.jpg"))
        img_map = {p.stem: p for p in all_imgs}
    # ==========================================================
    
    npz_files = sorted(list(npz_dir.glob("pkg_*.npz")))
    valid_files = [f for f in npz_files if f.stem.replace("pkg_", "") in img_map]
    
    limit = cfg.get("inference", {}).get("batch_limit", 20)
    if len(valid_files) > limit:
        np.random.seed(42) 
        test_files = np.random.choice(valid_files, limit, replace=False)
    else:
        test_files = valid_files

    print(f"🚀 启动【精准索引版】推理可视化 | 测试帧数: {len(test_files)}")

    with torch.no_grad():
        for npz_path in tqdm(test_files, desc="Inference & Render"):
            frame_id = npz_path.stem.replace("pkg_", "")
            
            with np.load(npz_path, allow_pickle=True) as data:
                f3_stats = torch.from_numpy(data["phys_8d"]).float().unsqueeze(0).to(device)
                f3_mae = torch.from_numpy(data["deep_512d"]).float().unsqueeze(0).to(device)
                q_2d = torch.from_numpy(data.get("quality_2d", np.ones((63, 1), dtype=np.float32))).float().unsqueeze(0).to(device)
                patch_corners_uv = data["patch_corners_uv"]
                meta = data["meta"]
                
            pseudo_label = meta[:, 0]
            valid_mask = meta[:, 2]
            
            img_path = img_map[frame_id]
            try:
                # 兼容 Windows/Linux 中文或复杂路径的绝对安全读取法
                img_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                img_bgr = None
                
            if img_bgr is None:
                print(f"\n⚠️ 无法读取图像: {img_path}")
                continue
            
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (1008, 560))
            img_tensor = img_transform(img_resized).unsqueeze(0).to(device)
            
            rois_tensor = generate_rois(patch_corners_uv, valid_mask, orig_w, orig_h).to(device)
            valid_mask_tensor = torch.from_numpy(valid_mask).float().unsqueeze(0).to(device)
            
            final_logit, internals = model(img_tensor, rois_tensor, f3_stats, f3_mae, q_2d, valid_mask_tensor)
            
            probs = torch.sigmoid(final_logit).squeeze(0).cpu().numpy()
            preds = (probs > 0.5).astype(np.float32)
            weights = internals["weights"].cpu().numpy() 
            
            img_filename = f"{frame_id}.jpg"
            if img_filename in manual_gt_dict:
                gt_labels = np.array(manual_gt_dict[img_filename], dtype=np.float32)
            else:
                gt_labels = pseudo_label
                
            draw_inference_result(
                img_bgr=img_bgr,
                patch_corners_uv=patch_corners_uv,
                valid_mask=valid_mask,
                preds=preds,
                gt_labels=gt_labels,
                weights=weights,
                frame_id=frame_id,
                out_dir=vis_out_dir
            )

    print(f"\n✨ 渲染完成! 结果已保存至: {vis_out_dir}")

if __name__ == "__main__":
    main()