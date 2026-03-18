"""
Road-MoME 训练引擎 (v9.5 终极云端版 - RTX 5090 专供)

核心升级 (v9.5):
1. AMP 混合精度训练: 激活 Blackwell Tensor Core，显存减半，速度翻倍。
2. Focal Loss: 替换原生 BCE，gamma=2.0，死磕模棱两可的困难病害样本。
3. 3D 软置信度加权 (Soft Confidence): 缓解雷达噪点在硬阈值边缘造成的梯度撕裂。
4. 动态生存演练: 维持 v9.0 的致盲与 3D 暴击机制。
"""

import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.mome_model import MoMEEngine


def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dynamic_blindness_prob(current_epoch, max_epochs, start_prob=0.3, end_prob=0.7):
    """计算当前 Epoch 的动态致盲概率 (线性递增)"""
    progress = current_epoch / max_epochs
    return start_prob + progress * (end_prob - start_prob)


# ==========================================
# [v9.5 核心注入 1]: Focal Loss 引擎
# ==========================================
class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        # pt 是模型预测正确的概率。对于 BCE，pt = exp(-bce_loss)
        pt = torch.exp(-bce_loss)
        # Focal Loss 核心调制：(1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss


class NPZDataset(Dataset):
    def __init__(self, npz_dir, json_gt_path=None):
        self.npz_files = sorted(list(Path(npz_dir).glob("pkg_*.npz")))
        self.manual_gt_dict = {}
        if json_gt_path and Path(json_gt_path).exists():
            with open(json_gt_path, "r", encoding="utf-8") as f:
                self.manual_gt_dict = json.load(f)

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        frame_id = npz_path.stem.replace("pkg_", "")
        with np.load(npz_path, allow_pickle=True) as data:
            f3_stats = data["phys_8d"]
            f3_mae = data["deep_512d"]
            f2_conv = data["deep_2d_768d"]
            q_2d = data.get("quality_2d", np.ones((63, 1), dtype=np.float32))
            meta = data["meta"]

        pseudo_label = meta[:, 0]
        valid_mask = meta[:, 2]

        img_filename = f"{frame_id}.jpg"
        if img_filename in self.manual_gt_dict:
            target_gt = np.array(self.manual_gt_dict[img_filename], dtype=np.float32)
            has_gt = 1.0
        else:
            target_gt = np.zeros_like(pseudo_label)
            has_gt = 0.0

        return {
            "f3_stats": torch.from_numpy(f3_stats).float(),
            "f3_mae": torch.from_numpy(f3_mae).float(),
            "f2_conv": torch.from_numpy(f2_conv).float(),
            "q_2d": torch.from_numpy(q_2d).float(),
            "pseudo_label": torch.from_numpy(pseudo_label).float(),
            "target_gt": torch.from_numpy(target_gt).float(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
            "has_gt": torch.tensor(has_gt).float(),
        }


def mome_loss_v9_5(
    final_logit,
    internals,
    f3_stats,
    target_pseudo,
    target_gt,
    valid_mask,
    has_gt,
    pos_weight_val,
    is_blind,
    cfg,
):
    """v9.5 终极版 Loss：引入 Focal Loss 与 3D 软置信度"""
    device = final_logit.device
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)

    # 替换原生 BCE 为 Focal Loss
    gamma = cfg.get("training", {}).get("focal_loss_gamma", 2.0)
    loss_func = FocalLossWithLogits(gamma=gamma, pos_weight=pos_weight_tensor)

    loss_phys = loss_func(internals["pred_phys"], target_pseudo)
    loss_geom = loss_func(internals["pred_geom"], target_pseudo)

    # ==========================================
    # [v9.5 核心注入 2]: 3D 软置信度加权 (Soft Confidence)
    # ==========================================
    # 假设 f3_stats 的第 2 个维度 (index 2) 是绝对高度差 delta_Z
    # (如果你的特征预处理中 delta_Z 在别的 index，请在此处修改)
    delta_z_idx = 2
    delta_z = f3_stats[:, :, delta_z_idx]

    soft_range = cfg.get("training", {}).get("soft_confidence_range", [0.025, 0.035])
    soft_weight_val = cfg.get("training", {}).get("soft_confidence_weight", 0.1)

    # 生成软置信度 Mask：落在模糊区间的雷达判定，Loss 惩罚大幅衰减
    fuzzy_mask = (delta_z >= soft_range[0]) & (delta_z <= soft_range[1])
    soft_weight_tensor = torch.ones_like(target_pseudo)
    soft_weight_tensor[fuzzy_mask] = soft_weight_val

    # 将软置信度应用于 3D 专家
    loss_phys = loss_phys * soft_weight_tensor
    loss_geom = loss_geom * soft_weight_tensor

    has_gt_expanded = has_gt.unsqueeze(1).expand_as(target_gt)
    active_target_2d = torch.where(has_gt_expanded > 0.5, target_gt, target_pseudo)
    loss_tex = loss_func(internals["pred_tex"], active_target_2d)

    # 动态致盲惩罚 (维持 v9.0 逻辑)
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
    epochs = cfg.get("training", {}).get("epochs", 50)
    batch_size = cfg.get("training", {}).get("batch_size", 64)
    pos_weight = cfg.get("training", {}).get("pos_weight", 5.0)

    npz_dir = Path(cfg["paths"]["output_dir"])
    json_gt_path = project_root / "data" / "manual_visual_gt_Annotatordense.json"

    dataset = NPZDataset(npz_dir, json_gt_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dim_phys = cfg["features"]["phys"]["input_dim"]
    dim_3d = cfg["features"]["3d"]["input_dim"]
    dim_2d = cfg["features"]["2d"]["input_dim"]

    model = MoMEEngine(dim_f3_stats=dim_phys, dim_f3_mae=dim_3d, dim_f2_dino=dim_2d).to(
        device
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # ==========================================
    # [v9.5 核心注入 3]: AMP 梯度缩放器
    # ==========================================
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.get("training", {}).get("amp_enabled", True)
    )

    print(
        f"🚀 云端 v9.5 启动: AMP [{scaler.is_enabled()}] | FocalLoss | SoftConfidence"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        blind_prob = get_dynamic_blindness_prob(epoch, epochs)
        epoch_losses = []

        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch}/{epochs} (Blind: {blind_prob:.2f})"
        )
        for batch in pbar:
            f3_stats = batch["f3_stats"].to(device)
            f3_mae = batch["f3_mae"].to(device)
            f2_conv = batch["f2_conv"].to(device)
            q_2d = batch["q_2d"].to(device)
            pseudo_label = batch["pseudo_label"].to(device)
            target_gt = batch["target_gt"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            has_gt = batch["has_gt"].to(device)

            optimizer.zero_grad()

            is_blind = False
            if random.random() < blind_prob:
                is_blind = True
                f2_conv = torch.zeros_like(f2_conv)
                q_2d = torch.zeros_like(q_2d)

            # ==========================================
            # [v9.5 核心注入 3]: 混合精度前向与反向传播
            # ==========================================
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=scaler.is_enabled()
            ):
                final_logit, internals = model(f3_stats, f3_mae, f2_conv, q_2d)

                loss, loss_dict = mome_loss_v9_5(
                    final_logit,
                    internals,
                    f3_stats,
                    pseudo_label,
                    target_gt,
                    valid_mask,
                    has_gt,
                    pos_weight,
                    is_blind,
                    cfg,
                )

            if loss.requires_grad:
                # 使用 Scaler 缩放 Loss 并反向传播
                scaler.scale(loss).backward()
                # 梯度裁剪前需要先 unscale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                # 更新权重
                scaler.step(optimizer)
                scaler.update()

            epoch_losses.append(loss_dict)
            pbar.set_postfix({"Loss": f"{loss_dict['Total']:.3f}"})

        avg_w_phys = np.mean([d["w_phys"] for d in epoch_losses])
        avg_w_tex = np.mean([d["w_tex"] for d in epoch_losses])
        print(
            f"⚖️ Epoch {epoch} 门控分布 -> Phys: {avg_w_phys:.2f} | Tex(2D): {avg_w_tex:.2f}"
        )


if __name__ == "__main__":
    main()
