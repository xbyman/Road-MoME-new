"""
Road-MoME 端到端训练引擎 (v14.0)

本版本在 v12.0 基础上叠加三项确定性改进：

③ CrossModalAttention padding mask（mome_model.py 已修复）：
   盲区 patch 不再作为 Key/Value 参与注意力计算，有效 patch 注意力集中。

④ 噪声帧重加权：
   has_gt=0（纯伪标签帧）的损失整体乘以 pseudo_frame_weight（默认 0.5），
   让模型把更多学习资源集中在人工标注的干净帧上，
   改善训练集 F1 < 验证集 F1 的倒置异常。

⑤ CosineAnnealingLR 学习率调度：
   替换全程固定 lr，后期 epoch 精细收敛，
   缓解 v12 Epoch 69 达峰后下滑的震荡问题。

损失函数逻辑回归 v12：
   loss_tex     ← target_gt（人工 GT，保持 Precision 优势）
   loss_phys/geom ← target_pseudo（伪标签，保持原有平衡）
   loss_fusion  ← target_fusion = max(pseudo, gt)
"""

import os
import sys
import json
import yaml
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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

def mome_loss_v14(
    final_logit, internals, f3_stats, target_pseudo, target_gt,
    valid_mask, has_gt, pos_weight_val, is_blind, cfg
):
    """
    v14.0 损失函数

    监督目标（回归 v12 逻辑，保留 Precision 优势）：
      loss_fusion  ← target_fusion = max(pseudo, gt)   完整异常集合
      loss_phys    ← target_pseudo                     伪标签（3D 障碍物）
      loss_geom    ← target_pseudo                     伪标签（3D 障碍物）
      loss_tex     ← target_gt                         人工 GT（裂缝/坑洞）

    新增 ④ 噪声帧重加权：
      has_gt=0 的纯伪标签帧损失乘以 pseudo_frame_weight（默认 0.5），
      减少噪声标注对梯度的污染，提升学习效率。
      has_gt=1 的帧权重保持 1.0（满权重训练）。
    """
    device = final_logit.device
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    gamma    = cfg.get("training", {}).get("focal_loss_gamma", 2.0)
    expert_w = cfg.get("training", {}).get("expert_loss_weight", 0.4)
    # ④ 噪声帧重加权系数（可在 config 中覆盖）
    pseudo_frame_w = cfg.get("training", {}).get("pseudo_frame_weight", 0.5)
    loss_func = FocalLossWithLogits(gamma=gamma, pos_weight=pos_weight_tensor)

    # 完整异常集合
    target_fusion = torch.max(target_pseudo, target_gt)   # [B, 63]

    # --------------------------------------------------
    # 各路损失
    # --------------------------------------------------
    loss_fusion = loss_func(final_logit,                   target_fusion)
    loss_phys   = loss_func(internals["pred_phys"],        target_pseudo)
    loss_geom   = loss_func(internals["pred_geom"],        target_pseudo)
    loss_tex    = loss_func(internals["pred_tex"],         target_gt)

    # 软置信度降权：点云模糊区域降权（仅 Phys/Geom）
    delta_z_idx     = 2
    delta_z         = f3_stats[:, :, delta_z_idx]
    soft_range      = cfg.get("training", {}).get("soft_confidence_range", [0.025, 0.035])
    soft_weight_val = cfg.get("training", {}).get("soft_confidence_weight", 0.1)
    fuzzy_mask      = (delta_z >= soft_range[0]) & (delta_z <= soft_range[1])
    soft_w          = torch.ones_like(target_pseudo)
    soft_w[fuzzy_mask] = soft_weight_val
    loss_phys = loss_phys * soft_w
    loss_geom = loss_geom * soft_w

    # 致盲训练
    if is_blind:
        loss_tex    = loss_tex    * 0.0
        loss_phys   = loss_phys   * 2.0
        loss_geom   = loss_geom   * 2.0

    # ④ 噪声帧重加权：has_gt=0 的帧整体降权
    # has_gt: [B]，扩展到 [B, 63] 与 loss 形状对齐
    frame_w = torch.where(
        has_gt.unsqueeze(1).expand_as(target_gt) > 0.5,
        torch.ones_like(target_gt),
        torch.full_like(target_gt, pseudo_frame_w)
    )   # [B, 63]

    total_raw_loss = (
        loss_fusion
        + expert_w * (loss_phys + loss_geom + loss_tex)
    ) * frame_w   # 按帧加权

    masked_total_loss = total_raw_loss * valid_mask
    num_valid         = valid_mask.sum()

    if num_valid > 0:
        final_loss  = masked_total_loss.sum() / num_valid
        weights     = internals["weights"]
        avg_weights = (weights * valid_mask.unsqueeze(-1)).sum(dim=(0, 1)) / num_valid
        loss_dict   = {
            "Total":  final_loss.item(),
            "w_phys": avg_weights[0].item(),
            "w_geom": avg_weights[1].item(),
            "w_tex":  avg_weights[2].item(),
        }
    else:
        final_loss = masked_total_loss.sum() * 0.0
        loss_dict  = {"Total": 0.0, "w_phys": 0.0, "w_geom": 0.0, "w_tex": 0.0}

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

    seed         = cfg.get("training", {}).get("seed", 42)
    val_split    = cfg.get("training", {}).get("val_split", 0.15)
    val_workers  = cfg.get("training", {}).get("val_num_workers", 4)

    # ====================================================
    # 【核心修复区】：绝对路径锚定机制
    # 彻底免疫终端 CWD (Current Working Directory) 错误
    # ====================================================
    def get_abs_path(p_str):
        p_str = str(p_str)
        if p_str.startswith("./"):
            return project_root / p_str[2:]
        if Path(p_str).is_absolute():
            return Path(p_str)
        return project_root / p_str

    npz_dir      = get_abs_path(cfg["paths"]["output_dir"])
    img_dir      = get_abs_path(cfg["paths"]["raw_img_dir"])
    json_gt_path = get_abs_path(cfg["paths"].get("manual_label_path", "./data/merged_visual_gt.json"))
    # ====================================================

    ckpt_dir  = project_root / "pretrained_models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / "road_mome_v14_best.pth"
    writer = None
    if cfg.get("train", {}).get("use_tensorboard", True):
        log_dir = project_root / "logs" / "mome_e2e_training"
        writer  = SummaryWriter(log_dir=str(log_dir))
        print(f"📈 TensorBoard 已启动，可用命令查看: tensorboard --logdir={log_dir}")

    # --------------------------------------------------
    # 【v12.0 核心修复】训练 / 验证集正式分离
    # val_split 配置项从"死配置"变为真正生效的划分逻辑
    # 使用固定 seed 保证每次运行划分结果一致
    # --------------------------------------------------
    full_dataset = E2EDataset(npz_dir, img_dir, json_gt_path)
    n_total  = len(full_dataset)
    n_val    = int(n_total * val_split)
    n_train  = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )
    print(f"📂 数据集划分 → 训练: {n_train} 帧 | 验证: {n_val} 帧 (val_split={val_split})")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=True, persistent_workers=True
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

    # ⑤ CosineAnnealingLR：全程余弦退火，eta_min 为初始 lr 的 1/20
    # 让模型在后期 epoch 精细收敛，缓解 v12 Epoch 69 后的震荡下滑
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr_base / 20
    )

    scaler = torch.amp.GradScaler(device='cuda', enabled=cfg.get("training", {}).get("amp_enabled", True))

    best_f1 = 0.0

    print(f"🚀 Road-MoME v14.0 训练启动 | BS: {batch_size} | Pos Weight: {pos_weight} | Train: {n_train} | Val: {n_val}")

    for epoch in range(1, epochs + 1):
        # ================================================
        # 训练阶段
        # ================================================
        model.train()
        blind_prob = get_dynamic_blindness_prob(epoch, epochs, blind_start, blind_end)
        epoch_losses = []
        epoch_tp, epoch_fp, epoch_tn, epoch_fn = 0, 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
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
                loss, loss_dict = mome_loss_v14(
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

                # 综合指标口径：两类目标取 OR，与推理行为一致
                target_fusion = torch.max(pseudo_label, target_gt)

                valid_idx = valid_mask > 0.5
                p_v = preds[valid_idx]
                t_v = target_fusion[valid_idx]

                epoch_tp += ((p_v == 1) & (t_v == 1)).sum().item()
                epoch_fp += ((p_v == 1) & (t_v == 0)).sum().item()
                epoch_tn += ((p_v == 0) & (t_v == 0)).sum().item()
                epoch_fn += ((p_v == 0) & (t_v == 1)).sum().item()

            pbar.set_postfix({"Loss": f"{loss_dict['Total']:.3f}"})

        avg_loss   = np.mean([d["Total"]  for d in epoch_losses])
        avg_w_phys = np.mean([d["w_phys"] for d in epoch_losses])
        avg_w_geom = np.mean([d["w_geom"] for d in epoch_losses])
        avg_w_tex  = np.mean([d["w_tex"]  for d in epoch_losses])

        train_acc       = (epoch_tp + epoch_tn) / (epoch_tp + epoch_tn + epoch_fp + epoch_fn + 1e-6)
        train_precision = epoch_tp / (epoch_tp + epoch_fp + 1e-6)
        train_recall    = epoch_tp / (epoch_tp + epoch_fn + 1e-6)
        train_f1        = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-6)

        print(f"\n📊 [诊断报告] Epoch {epoch}")
        print(f"   ┣ 专家决策权重  -> Phys: {avg_w_phys:.2f} | Geom(3D): {avg_w_geom:.2f} | Tex(2D): {avg_w_tex:.2f}")
        print(f"   ┣ [训练集] Acc: {train_acc*100:.1f}% | Prec: {train_precision*100:.1f}% | Rec: {train_recall*100:.1f}% | F1: {train_f1:.4f}")

        # ================================================
        # 【v12.0 核心修复】验证阶段
        # model.eval() + no_grad()，在验证集上独立评估
        # ================================================
        model.eval()
        val_tp, val_fp, val_tn, val_fn = 0, 0, 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                img_tensor   = batch["img_tensor"].to(device, non_blocking=True)
                raw_rois     = batch["rois"].to(device, non_blocking=True)
                f3_stats     = batch["f3_stats"].to(device, non_blocking=True)
                f3_mae       = batch["f3_mae"].to(device, non_blocking=True)
                q_2d         = batch["q_2d"].to(device, non_blocking=True)
                pseudo_label = batch["pseudo_label"].to(device, non_blocking=True)
                target_gt    = batch["target_gt"].to(device, non_blocking=True)
                valid_mask   = batch["valid_mask"].to(device, non_blocking=True)
                has_gt       = batch["has_gt"].to(device, non_blocking=True)

                B = img_tensor.size(0)
                batch_rois_list = []
                for b_idx in range(B):
                    b_idx_tensor = torch.full((63, 1), b_idx, dtype=torch.float32, device=device)
                    batch_rois_list.append(torch.cat([b_idx_tensor, raw_rois[b_idx]], dim=1))
                rois_tensor = torch.cat(batch_rois_list, dim=0)

                final_logit, _ = model(
                    img_tensor, rois_tensor, f3_stats, f3_mae, q_2d, valid_mask
                )

                probs = torch.sigmoid(final_logit)
                preds = (probs > 0.5).float()

                # 综合指标口径：两类目标取 OR，与推理行为一致
                target_fusion = torch.max(pseudo_label, target_gt)

                valid_idx = valid_mask > 0.5
                p_v = preds[valid_idx]
                t_v = target_fusion[valid_idx]

                val_tp += ((p_v == 1) & (t_v == 1)).sum().item()
                val_fp += ((p_v == 1) & (t_v == 0)).sum().item()
                val_tn += ((p_v == 0) & (t_v == 0)).sum().item()
                val_fn += ((p_v == 0) & (t_v == 1)).sum().item()

        val_precision = val_tp / (val_tp + val_fp + 1e-6)
        val_recall    = val_tp / (val_tp + val_fn + 1e-6)
        val_f1        = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-6)
        val_acc       = (val_tp + val_tn) / (val_tp + val_tn + val_fp + val_fn + 1e-6)

        print(f"   ┗ [验证集] Acc: {val_acc*100:.1f}% | Prec: {val_precision*100:.1f}% | Rec: {val_recall*100:.1f}% | 🏆 F1: {val_f1:.4f}")

        if writer:
            writer.add_scalar("Train/Loss",           avg_loss,        epoch)
            writer.add_scalar("Train/F1",             train_f1,        epoch)
            writer.add_scalar("Train/Precision",      train_precision, epoch)
            writer.add_scalar("Train/Recall",         train_recall,    epoch)
            writer.add_scalar("Val/F1",               val_f1,          epoch)
            writer.add_scalar("Val/Precision",        val_precision,   epoch)
            writer.add_scalar("Val/Recall",           val_recall,      epoch)
            writer.add_scalar("Val/Accuracy",         val_acc,         epoch)
            writer.add_scalar("ExpertWeight/Phys",    avg_w_phys,      epoch)
            writer.add_scalar("ExpertWeight/Geom_3D", avg_w_geom,      epoch)
            writer.add_scalar("ExpertWeight/Tex_2D",  avg_w_tex,       epoch)
            writer.add_scalar("Train/LR_base",
                              scheduler.get_last_lr()[0], epoch)

        # ⑤ LR Scheduler step（验证之后执行）
        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1':          best_f1,
                'train_f1':             train_f1,
                'version':              'v14.0',
            }, save_path)
            print(f"   💾 [Weight Saved] 验证集 F1 突破历史最高 ({best_f1:.4f})! 权重已保存: {save_path.name}\n")
        else:
            print(f"   - (未突破最佳验证 F1: {best_f1:.4f})\n")

    if writer:
        writer.close()

if __name__ == "__main__":
    main()