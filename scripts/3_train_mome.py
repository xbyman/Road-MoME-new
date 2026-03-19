"""
Road-MoME 端到端终极训练引擎 (v10.1 精准索引对齐版)

核心重构：
1. 索引制导加载：优先读取 dataset_indexer 生成的标准索引，废弃 rglob 盲搜，确保数据读取 100% 严谨。
2. 数据大盘盘点：在数据集初始化阶段实时输出 NPZ、索引和对齐后的精确数量，直接拦截 num_samples=0 报错。
3. 致命级防错：强制挂载 manual_label_path (JSON)，若未读取到有效标注数据，直接触发 RuntimeError 中断训练。
"""

import os
import sys
import json
import yaml
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.mome_model import MoMEEngine

def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_dynamic_blindness_prob(current_epoch, max_epochs, start_prob, end_prob):
    progress = current_epoch / max_epochs
    return start_prob + progress * (end_prob - start_prob)

class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class E2EDataset(Dataset):
    def __init__(self, npz_dir, img_dir, json_gt_path):
        # ==========================================================
        # 【核心重构：精准索引制导加载机制】
        # ==========================================================
        self.img_map = {}
        
        # 尝试寻找之前生成的标准索引文件 (支持 yaml 或 json)
        index_yaml = project_root / "data" / "dataset_index.yaml"
        index_json = project_root / "data" / "dataset_index.json"
        
        index_loaded = False
        if index_yaml.exists():
            with open(index_yaml, "r", encoding="utf-8") as f:
                index_data = yaml.safe_load(f)
                # 强制转换为 str，防止 YAML 将 '20230317074850.000' 误解析为 float 导致匹配失败
                self.img_map = {str(item["id"]): Path(item["img"]) for item in index_data}
            print(f"📖 成功挂载标准数据集索引: {index_yaml.name} (共映射 {len(self.img_map)} 帧)")
            index_loaded = True
        elif index_json.exists():
            with open(index_json, "r", encoding="utf-8") as f:
                index_data = json.load(f)
                self.img_map = {str(item["id"]): Path(item["img"]) for item in index_data}
            print(f"📖 成功挂载标准数据集索引: {index_json.name} (共映射 {len(self.img_map)} 帧)")
            index_loaded = True
            
        # 如果确实没有索引，才回退到 rglob 盲搜兜底
        if not index_loaded:
            print("⚠️ 未检测到 dataset_index.yaml/json，回退到 rglob 盲搜模式...")
            all_imgs = list(Path(img_dir).rglob("*.jpg"))
            self.img_map = {p.stem: p for p in all_imgs}
        # ==========================================================

        raw_npz_files = sorted(list(Path(npz_dir).glob("pkg_*.npz")))
        
        self.npz_files = []
        for f in raw_npz_files:
            fid = f.stem.replace("pkg_", "")
            if fid in self.img_map:
                self.npz_files.append(f)

        # 【核心防爆：训练数据大盘盘点】
        print("\n" + "="*50)
        print("📊 [训练数据大盘盘点]")
        print(f"   - 预处理特征包 (.npz): {len(raw_npz_files)} 个 (寻找路径: {npz_dir})")
        print(f"   - 原始图像映射 (img):  {len(self.img_map)} 条记录")
        print(f"   - 最终对齐可训练:      {len(self.npz_files)} 帧")
        print("="*50 + "\n")

        if len(self.npz_files) == 0:
            raise RuntimeError(
                f"❌ 致命错误: 训练集样本数为 0！\n"
                f"这直接导致了 DataLoader 发生 'num_samples=0' 的报错。\n"
                f"原因排查：\n"
                f"1. 如果上面的 '.npz' 数量为 0，说明你没有在 {npz_dir} 里生成特征包。\n"
                f"2. 如果 '.npz' 数量 > 0 但最终可训练为 0，说明索引文件里的 ID 和 npz 文件的名字完全匹配不上！"
            )

        # 【核心防错防毒机制】
        if not Path(json_gt_path).exists():
            raise RuntimeError(f"❌ 致命错误: 找不到人工真值库 {json_gt_path}！为防止模型被伪标签毒害，训练强制中止。")
            
        with open(json_gt_path, "r", encoding="utf-8") as f:
            self.manual_gt_dict = json.load(f)
            
        if len(self.manual_gt_dict) == 0:
            raise RuntimeError(f"❌ 致命错误: JSON 真值库为空！请检查 {json_gt_path}")
            
        print(f"✅ 成功挂载纯净人工真值数据: {Path(json_gt_path).name} (共包含 {len(self.manual_gt_dict)} 帧完美标注)")

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        frame_id = npz_path.stem.replace("pkg_", "")
        
        with np.load(npz_path, allow_pickle=True) as data:
            f3_stats = data["phys_8d"]
            f3_mae = data["deep_512d"]
            q_2d = data.get("quality_2d", np.ones((63, 1), dtype=np.float32))
            patch_corners_uv = data["patch_corners_uv"] 
            meta = data["meta"]

        pseudo_label = meta[:, 0]
        valid_mask = meta[:, 2]

        # 直接从索引映射中获取绝对路径，安全可靠
        img_path = self.img_map[frame_id]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"❌ 图像读取失败，请检查索引路径是否有效: {img_path}")
            
        orig_h, orig_w = img_bgr.shape[:2]
        
        target_w, target_h = 1008, 560 
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (target_w, target_h))
        img_tensor = img_transform(img_resized) 
        
        rois = []
        scale_x, scale_y = target_w / orig_w, target_h / orig_h
        for i in range(63):
            if valid_mask[i] == 0:
                rois.append([0.0, 0.0, 1.0, 1.0])
            else:
                corners = patch_corners_uv[i]
                x_min, x_max = max(0, np.min(corners[:, 0]) * scale_x), min(target_w, np.max(corners[:, 0]) * scale_x)
                y_min, y_max = max(0, np.min(corners[:, 1]) * scale_y), min(target_h, np.max(corners[:, 1]) * scale_y)
                if x_max <= x_min or y_max <= y_min:
                    rois.append([0.0, 0.0, 1.0, 1.0])
                else:
                    rois.append([float(x_min), float(y_min), float(x_max), float(y_max)])
        rois_array = np.array(rois, dtype=np.float32)

        # 优先读取真值，缺失则降级为伪标签 (并记录 has_gt 状态)
        img_filename = f"{frame_id}.jpg"
        if img_filename in self.manual_gt_dict:
            target_gt = np.array(self.manual_gt_dict[img_filename], dtype=np.float32)
            has_gt = 1.0
        else:
            target_gt = np.zeros_like(pseudo_label)
            has_gt = 0.0

        return {
            "img_tensor": img_tensor,                   
            "rois": torch.from_numpy(rois_array),       
            "f3_stats": torch.from_numpy(f3_stats).float(),
            "f3_mae": torch.from_numpy(f3_mae).float(),
            "q_2d": torch.from_numpy(q_2d).float(),
            "pseudo_label": torch.from_numpy(pseudo_label).float(),
            "target_gt": torch.from_numpy(target_gt).float(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
            "has_gt": torch.tensor(has_gt).float(),
        }

def mome_loss_v9_5(
    final_logit, internals, f3_stats, target_pseudo, target_gt,
    valid_mask, has_gt, pos_weight_val, is_blind, cfg
):
    device = final_logit.device
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    gamma = cfg.get("training", {}).get("focal_loss_gamma", 2.0)
    loss_func = FocalLossWithLogits(gamma=gamma, pos_weight=pos_weight_tensor)

    loss_phys = loss_func(internals["pred_phys"], target_pseudo)
    loss_geom = loss_func(internals["pred_geom"], target_pseudo)

    delta_z_idx = 2 
    delta_z = f3_stats[:, :, delta_z_idx]
    soft_range = cfg.get("training", {}).get("soft_confidence_range", [0.025, 0.035])
    soft_weight_val = cfg.get("training", {}).get("soft_confidence_weight", 0.1)
    
    fuzzy_mask = (delta_z >= soft_range[0]) & (delta_z <= soft_range[1])
    soft_weight_tensor = torch.ones_like(target_pseudo)
    soft_weight_tensor[fuzzy_mask] = soft_weight_val
    
    loss_phys = loss_phys * soft_weight_tensor
    loss_geom = loss_geom * soft_weight_tensor

    # 【重要】如果当前帧有 GT，强迫视觉专家向真实的 GT 学习！
    has_gt_expanded = has_gt.unsqueeze(1).expand_as(target_gt)
    active_target_2d = torch.where(has_gt_expanded > 0.5, target_gt, target_pseudo)
    loss_tex = loss_func(internals["pred_tex"], active_target_2d)

    if is_blind:
        loss_tex = loss_tex * 0.0
        loss_phys = loss_phys * 2.0
        loss_geom = loss_geom * 2.0

    target_fusion = torch.max(target_pseudo, active_target_2d)
    loss_fusion = loss_func(final_logit, target_fusion)

    total_raw_loss = loss_fusion + 0.4 * loss_phys + 0.4 * loss_geom + 0.4 * loss_tex
    masked_total_loss = total_raw_loss * valid_mask
    num_valid = valid_mask.sum()

    if num_valid > 0:
        final_loss = masked_total_loss.sum() / num_valid
        weights = internals["weights"]
        avg_weights = (weights * valid_mask.unsqueeze(-1)).sum(dim=(0, 1)) / num_valid
        loss_dict = {
            "Total": final_loss.item(),
            "w_phys": avg_weights[0].item(),
            "w_geom": avg_weights[1].item(),
            "w_tex": avg_weights[2].item(),
        }
    else:
        final_loss = masked_total_loss.sum() * 0.0
        loss_dict = {"Total": 0.0, "w_phys": 0.0, "w_geom": 0.0, "w_tex": 0.0}

    return final_loss, loss_dict

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    epochs = cfg.get("training", {}).get("epochs", 60)
    batch_size = cfg.get("training", {}).get("batch_size", 32) 
    pos_weight = cfg.get("training", {}).get("pos_weight", 3.0)
    lr_base = cfg.get("training", {}).get("learning_rate_base", 1e-4)
    lr_2d = cfg.get("training", {}).get("learning_rate_2d", 3e-5)
    blind_start = cfg.get("training", {}).get("blind_prob_start", 0.2)
    blind_end = cfg.get("training", {}).get("blind_prob_end", 0.4)

    npz_dir = Path(cfg["paths"]["output_dir"])
    img_dir = Path(cfg["paths"]["raw_img_dir"])
    
    # 解析 JSON 绝对路径
    raw_json_path = cfg["paths"].get("manual_label_path", "./data/merged_visual_gt.json")
    if raw_json_path.startswith("./"):
        raw_json_path = raw_json_path[2:]
    json_gt_path = project_root / raw_json_path
    
    ckpt_dir = project_root / "pretrained_models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / "road_mome_v9_e2e_best.pth"

    writer = None
    if cfg.get("train", {}).get("use_tensorboard", True):
        log_dir = project_root / "logs" / "mome_e2e_training"
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"📈 TensorBoard 已启动，可用命令查看: tensorboard --logdir={log_dir}")

    # 严密的数据集挂载 (会自动触发异常检测)
    dataset = E2EDataset(npz_dir, img_dir, json_gt_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=8, pin_memory=True, persistent_workers=True      
    )

    dim_phys = cfg["features"]["phys"]["input_dim"]
    dim_3d = cfg["features"]["3d"]["input_dim"]

    model = MoMEEngine(dim_f3_stats=dim_phys, dim_f3_mae=dim_3d).to(device)
    
    print("⚡ 正在启用 torch.compile 编译计算图 (请耐心等待首次加载)...")
    model = torch.compile(model)
    
    base_params = []
    live_2d_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "live_2d" in name:
            live_2d_params.append(param)
        else:
            base_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': lr_base},
        {'params': live_2d_params, 'lr': lr_2d}
    ])
    
    scaler = torch.amp.GradScaler(device='cuda', enabled=cfg.get("training", {}).get("amp_enabled", True))

    best_f1 = 0.0

    print(f"🚀 强制对齐训练马拉松启动 | BS: {batch_size} | Pos Weight: {pos_weight}")

    for epoch in range(1, epochs + 1):
        model.train()
        blind_prob = get_dynamic_blindness_prob(epoch, epochs, blind_start, blind_end)
        epoch_losses = []
        epoch_tp, epoch_fp, epoch_tn, epoch_fn = 0, 0, 0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            img_tensor = batch["img_tensor"].to(device, non_blocking=True) 
            raw_rois = batch["rois"].to(device, non_blocking=True)         
            f3_stats = batch["f3_stats"].to(device, non_blocking=True)
            f3_mae = batch["f3_mae"].to(device, non_blocking=True)
            q_2d = batch["q_2d"].to(device, non_blocking=True)
            pseudo_label = batch["pseudo_label"].to(device, non_blocking=True)
            target_gt = batch["target_gt"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            has_gt = batch["has_gt"].to(device, non_blocking=True)
            
            B = img_tensor.size(0)
            batch_rois_list = []
            for b_idx in range(B):
                b_idx_tensor = torch.full((63, 1), b_idx, dtype=torch.float32, device=device)
                batch_rois_list.append(torch.cat([b_idx_tensor, raw_rois[b_idx]], dim=1))
            rois_tensor = torch.cat(batch_rois_list, dim=0)

            optimizer.zero_grad(set_to_none=True)

            is_blind = False
            if random.random() < blind_prob:
                is_blind = True
                img_tensor = torch.zeros_like(img_tensor)
                q_2d = torch.zeros_like(q_2d)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=scaler.is_enabled()):
                final_logit, internals = model(img_tensor, rois_tensor, f3_stats, f3_mae, q_2d, valid_mask)
                loss, loss_dict = mome_loss_v9_5(
                    final_logit, internals, f3_stats, pseudo_label, target_gt,
                    valid_mask, has_gt, pos_weight, is_blind, cfg
                )

            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

            epoch_losses.append(loss_dict)
            
            with torch.no_grad():
                probs = torch.sigmoid(final_logit)
                preds = (probs > 0.5).float()
                
                has_gt_expanded = has_gt.unsqueeze(1).expand_as(target_gt)
                active_target_2d = torch.where(has_gt_expanded > 0.5, target_gt, pseudo_label)
                target_fusion = torch.max(pseudo_label, active_target_2d)
                
                valid_idx = valid_mask > 0.5
                p_v = preds[valid_idx]
                t_v = target_fusion[valid_idx]
                
                epoch_tp += ((p_v == 1) & (t_v == 1)).sum().item()
                epoch_fp += ((p_v == 1) & (t_v == 0)).sum().item()
                epoch_tn += ((p_v == 0) & (t_v == 0)).sum().item()
                epoch_fn += ((p_v == 0) & (t_v == 1)).sum().item()

            pbar.set_postfix({"Loss": f"{loss_dict['Total']:.3f}"})

        avg_loss = np.mean([d["Total"] for d in epoch_losses])
        avg_w_phys = np.mean([d["w_phys"] for d in epoch_losses])
        avg_w_geom = np.mean([d["w_geom"] for d in epoch_losses])
        avg_w_tex = np.mean([d["w_tex"] for d in epoch_losses])
        
        acc = (epoch_tp + epoch_tn) / (epoch_tp + epoch_tn + epoch_fp + epoch_fn + 1e-6)
        precision = epoch_tp / (epoch_tp + epoch_fp + 1e-6)
        recall = epoch_tp / (epoch_tp + epoch_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        print(f"\n📊 [诊断报告] Epoch {epoch}")
        print(f"   ┣ 专家决策权重 -> Phys: {avg_w_phys:.2f} | Geom(3D): {avg_w_geom:.2f} | Tex(2D): {avg_w_tex:.2f}")
        print(f"   ┗ 病害识别指标 -> Accuracy: {acc*100:.1f}% | Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}% | 🏆 F1-Score: {f1:.4f}")

        if writer:
            writer.add_scalar("Train/Loss", avg_loss, epoch)
            writer.add_scalar("Metrics/F1-Score", f1, epoch)
            writer.add_scalar("Metrics/Precision", precision, epoch)
            writer.add_scalar("Metrics/Recall", recall, epoch)
            writer.add_scalar("ExpertWeight/Phys", avg_w_phys, epoch)
            writer.add_scalar("ExpertWeight/Geom_3D", avg_w_geom, epoch)
            writer.add_scalar("ExpertWeight/Tex_2D", avg_w_tex, epoch)

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, save_path)
            print(f"   💾 [Weight Saved] 突破历史最高 F1 ({best_f1:.4f})! 权重已保存在: {save_path.name}\n")
        else:
            print(f"   - (未突破最佳 F1: {best_f1:.4f})\n")

    if writer:
        writer.close()

if __name__ == "__main__":
    main()