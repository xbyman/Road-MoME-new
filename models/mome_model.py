"""
Road-MoME 端到端多模态混合专家模型 (v9.5 E2E Cloud 版)

核心重构：
1. 活体视觉接入 (Live 2D Extractor)：内嵌 ConvNeXt-Base，废弃外部离线 .npz 视觉特征输入。
2. 动态 ROI Align：在 GPU 显存中实时将全局特征图切割为 63 个 Patch。
3. 梯度精细管控：冻结 ConvNeXt 的前 3 个 Stage (防过拟合与加速)，仅放开 Stage 2 (features.3) 参与微调。
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import roi_align


# ==========================================
# 1. 局部窗口注意力层 (保留物理空间连续性)
# ==========================================
class LocalWindowAttention(nn.Module):
    def __init__(self, dim=1024, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.register_buffer("attn_mask", self._create_local_mask())

    def _create_local_mask(self):
        mask = torch.full((63, 63), float("-inf"))
        for i in range(63):
            r1, c1 = divmod(i, 7)
            for j in range(63):
                r2, c2 = divmod(j, 7)
                if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                    mask[i, j] = 0.0
        return mask

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, attn_mask=self.attn_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ==========================================
# 2. 活体 2D 特征提取器 (E2E 核心组件)
# ==========================================
class Live2DExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 依托 RTX 5090 算力，直接使用 ConvNeXt-Base
        base_model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        # 截取 features.3 (即 Stage 2), 具有 Stride=8 的高分辨率，输出通道 256
        self.backbone = create_feature_extractor(
            base_model, return_nodes={"features.3": "feat"}
        )

        # 【梯度管控策略】
        # 冻结浅层特征，仅让 features.3 参与反向传播，避免小数据集破坏 ImageNet 预训练泛化能力
        for name, param in self.backbone.named_parameters():
            if not name.startswith("features.3"):
                param.requires_grad = False

    def forward(self, img_tensor, rois):
        """
        img_tensor: [B, 3, 560, 1008]
        rois: [B*63, 5] 格式为 (batch_idx, x_min, y_min, x_max, y_max)
        """
        # [B, 256, H/8, W/8]
        spatial_feat = self.backbone(img_tensor)["feat"]

        # 动态 ROI Align (对齐 Stride=8)
        # 输出尺寸设为 2x2，保持空间纹理结构
        pooled_feat = roi_align(
            spatial_feat, rois, output_size=(2, 2), spatial_scale=1 / 8.0
        )

        # 展平: [B*63, 256, 2, 2] -> [B*63, 1024]
        flat_feat = pooled_feat.view(pooled_feat.size(0), -1)

        # 恢复 [B, 63, 1024] 的形状以对接后续时序/空间网络
        B = img_tensor.shape[0]
        return flat_feat.view(B, 63, 1024)


# ==========================================
# 3. 质量感知动态门控 (MoE 决策枢纽)
# ==========================================
class MoMEGatingNetwork(nn.Module):
    def __init__(self, dim_f3_stats=8, dim_f3_mae=384, dim_f2_conv=1024):
        super().__init__()
        self.quality_proj = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 32)
        )
        concat_dim = dim_f3_stats + dim_f3_mae + dim_f2_conv + 32

        self.gate = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),  # 对应 Phys, Geom, Tex 三个专家
            nn.Softmax(dim=-1),
        )

    def forward(self, f3_stats, f3_mae, f2_conv, quality_2d):
        q_feat = self.quality_proj(quality_2d)
        combined_feat = torch.cat([f3_stats, f3_mae, f2_conv, q_feat], dim=-1)
        return self.gate(combined_feat), combined_feat


# ==========================================
# 4. Road-MoME 主引擎 (Engine)
# ==========================================
class MoMEEngine(nn.Module):
    def __init__(self, dim_f3_stats=8, dim_f3_mae=384):
        super().__init__()

        # 活体 2D 骨干网络 (输出 1024 维)
        self.live_2d = Live2DExtractor()
        dim_f2_conv = 1024

        self.f2_local_attn = LocalWindowAttention(dim=dim_f2_conv)

        # 专家网络池
        self.exp_phys = nn.Sequential(
            nn.Linear(dim_f3_stats, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.exp_geom = nn.Sequential(
            nn.Linear(dim_f3_mae, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.exp_tex = nn.Sequential(
            nn.Linear(dim_f2_conv, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        self.gating = MoMEGatingNetwork(dim_f3_stats, dim_f3_mae, dim_f2_conv)

    def forward(self, img_tensor, rois, f3_stats, f3_mae, quality_2d, valid_mask):
        # 1. 活体视觉流提取 (E2E)
        f2_live = self.live_2d(img_tensor, rois)  # [B, 63, 1024]

        # 物理盲区强制清零，防止无关区域干扰视觉特征
        f2_live = f2_live * valid_mask.unsqueeze(-1)

        # 空间注意力强化
        f2_conv_enhanced = self.f2_local_attn(f2_live)

        # 2. 三专家独立前向传播
        pred_phys = self.exp_phys(f3_stats)
        pred_geom = self.exp_geom(f3_mae)
        pred_tex = self.exp_tex(f2_conv_enhanced)

        # 3. 门控网络分配投票权
        weights, _ = self.gating(f3_stats, f3_mae, f2_conv_enhanced, quality_2d)

        # 4. 融合决策
        preds_stacked = torch.cat([pred_phys, pred_geom, pred_tex], dim=-1)
        final_logit = torch.sum(preds_stacked * weights, dim=-1, keepdim=True)

        internals = {
            "pred_phys": pred_phys.squeeze(-1),
            "pred_geom": pred_geom.squeeze(-1),
            "pred_tex": pred_tex.squeeze(-1),
            "weights": weights,
        }

        return final_logit.squeeze(-1), internals
