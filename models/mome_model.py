"""
Road-MoME 核心模型架构 (v14.0)

差异化专家（v12.0）保持不变：
- PhysExpert   : 窄 MLP + LayerNorm + 软单调性约束
- GeomExpert   : 残差 MLP + 邻域深度对比特征
- TexExpert    : Patch 间 Self-Attention + valid_mask key_padding_mask

v14.0 新增 — CrossModalAttention padding mask（修复已知 bug）：
    盲区 patch 的 f3_mae 为无效点云，原来仍作为 Key/Value 参与 softmax，
    零向量的 attention score 平均分摊了概率质量，稀释有效 patch 的注意力能量。
    修复：将 valid_mask==0 的位置设为 key_padding_mask=True，
    使盲区 Key 被 MultiheadAttention 完全忽略，有效 patch 注意力集中。

专家升级策略：
- Level 1：结构差异化
    * PhysExpert   : 窄 MLP + LayerNorm
    * GeomExpert   : 残差 MLP + LayerNorm
    * TexExpert    : Patch 间 Self-Attention + MLP

- Level 2：任务感知差异化
    * PhysExpert   : 非负权重单调性约束
    * GeomExpert   : 邻域深度对比特征
    * TexExpert    : valid_mask 注入 Attention key_padding_mask

v13.1 修复（保留）— 统一监督单头：
    回退 v13.0 双头设计，回归单输出头。

v13.2 修复（保留）— 分层监督：
    loss_tex 回归 target_gt，Tex 专家保留干净视觉监督信号。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import roi_align


# ==========================================
# 1. 跨模态交叉注意力模块（保持不变）
# ==========================================
class CrossModalAttention(nn.Module):
    def __init__(self, dim_2d=1024, dim_3d=384, d_model=256, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(dim_2d, d_model)
        self.k_proj = nn.Linear(dim_3d, d_model)
        self.v_proj = nn.Linear(dim_3d, d_model)

        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, dim_2d)

    def forward(self, f2_tex, f3_mae, valid_mask=None):
        """
        f2_tex     : [B, 63, dim_2d]
        f3_mae     : [B, 63, dim_3d]
        valid_mask : [B, 63]  1=有效 0=盲区（可选）

        ③ Attention padding mask（v14.0 修复）：
        盲区 patch 的 f3_mae 为无效点云，不应作为 Key/Value 参与 softmax。
        将盲区位置设为 True 传入 key_padding_mask，使其被完全忽略，
        有效 patch 的注意力能量不再被盲区稀释。
        """
        q = self.q_proj(f2_tex)
        k = self.k_proj(f3_mae)
        v = self.v_proj(f3_mae)

        # 构造 key_padding_mask：盲区位置为 True（被 MHA 忽略）
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = (valid_mask == 0)   # [B, 63]

        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))

        # 残差防火墙：3D 情报以增量方式叠加，不覆盖原始 2D 特征
        f2_enhanced = f2_tex + self.out_proj(x)
        return f2_enhanced


# ==========================================
# 2. 差异化专家模块（v12.0 核心升级）
# ==========================================

class PhysExpert(nn.Module):
    """
    物理统计专家（Level 1 + Level 2）

    Level 1 结构差异化：
      - 窄 MLP (8→64→32→1)，容量与 8 维输入匹配
      - LayerNorm 替换 BatchNorm，消除小 batch 下统计量不稳定问题

    Level 2 任务感知差异化：
      - 使用 weight 绝对值 + Softplus 激活实现软单调性约束
      - 物理统计量（粗糙度/坡度/反射率方差）与病害概率理论单调相关
      - 软约束：梯度更新中非负方向受到鼓励，但不做硬截断
    """

    def __init__(self, input_dim: int = 8, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(32)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, 8]  (N = B * 63，展平格式)"""
        # 非负权重：abs() 实现软单调约束
        h = F.linear(x, torch.abs(self.fc1.weight), self.fc1.bias)
        h = self.drop(F.softplus(self.norm1(h)))          # Softplus 是单调激活

        h = F.linear(h, torch.abs(self.fc2.weight), self.fc2.bias)
        h = F.softplus(self.norm2(h))

        return self.fc3(h)                                 # [N, 1]


class GeomExpert(nn.Module):
    """
    几何深度专家（Level 1 + Level 2）

    Level 1 结构差异化：
      - 残差 MLP：主路径 fc1→fc2，捷径 shortcut 直接从输入跨两层
      - LayerNorm 替换 BatchNorm

    Level 2 任务感知差异化：
      - 邻域深度对比特征：计算每个 patch 与本帧全局均值的差分
      - 坑洞 = 局部深度突变，差分特征显式捕捉这种异常
      - forward 接收序列格式 [B, 63, 384]，内部展平后处理
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        # 拼接原始特征 + 对比特征后的输入维度
        contrast_dim = 64
        aug_dim = input_dim + contrast_dim

        # 对比特征编码器
        self.contrast_proj = nn.Sequential(
            nn.Linear(input_dim, contrast_dim),
            nn.LayerNorm(contrast_dim),
            nn.GELU(),
        )

        # 主路径
        self.fc1 = nn.Linear(aug_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.drop = nn.Dropout(dropout)

        # 残差捷径：跨越 fc1+fc2，梯度可以直接从 fc3 回传到输入
        self.shortcut = nn.Linear(aug_dim, hidden_dim // 2)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B, 63, 384]  ← 注意：序列格式，不是展平格式
        返回:  [B*63, 1]     ← 与旧接口一致，Engine 中 view 还原即可
        """
        B, N, D = x_seq.shape

        # Level 2：计算邻域对比特征
        # 每个 patch 与本帧所有 patch 均值的差 → 捕捉局部深度异常
        mean_feat = x_seq.mean(dim=1, keepdim=True)        # [B, 1, 384]
        contrast = x_seq - mean_feat                        # [B, 63, 384]
        contrast_enc = self.contrast_proj(contrast)         # [B, 63, 64]

        # 拼接原始特征 + 对比特征
        x_aug = torch.cat([x_seq, contrast_enc], dim=-1)   # [B, 63, 448]
        x_flat = x_aug.view(B * N, -1)                      # [B*63, 448]

        # 主路径
        h = self.drop(F.gelu(self.norm1(self.fc1(x_flat))))
        h = self.norm2(self.fc2(h))

        # 残差捷径 + 激活
        h = F.gelu(h + self.shortcut(x_flat))

        return self.fc3(h)                                  # [B*63, 1]


class TexExpert(nn.Module):
    """
    视觉纹理专家（Level 1 + Level 2）

    Level 1 结构差异化：
      - 先降维投影 (1024→hidden_dim)，减少 Attention 计算量
      - Patch 间 Self-Attention：63 个 patch 作为序列，互相参考视觉状态
      - 残差连接保留原始特征
      - LayerNorm 替换 BatchNorm

    Level 2 任务感知差异化：
      - valid_mask 注入 MultiheadAttention 的 key_padding_mask
      - 盲区 patch 不作为 key/value 参与其他 patch 的注意力计算
      - 裂缝（跨 patch 连续结构）和坑洞（局部凹陷）均可从空间关系中获益
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Step 1：降维，1024 → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Step 2：Patch 间 Self-Attention（序列长度 = 63）
        self.patch_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Step 3：逐 patch 分类头
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x          : [B, 63, 1024]  ← 序列格式
        valid_mask : [B, 63]        ← 1=有效，0=盲区（可选）
        返回       : [B*63, 1]      ← 与旧接口一致
        """
        B, N, _ = x.shape

        # Step 1：降维投影
        h = self.input_proj(x)                             # [B, 63, hidden_dim]

        # Step 2：构造 key_padding_mask（Level 2 核心）
        # MultiheadAttention 约定：True = 该位置被忽略（屏蔽）
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = (valid_mask == 0)           # [B, 63]，盲区为 True

        # Self-Attention + 残差
        attn_out, _ = self.patch_attn(
            h, h, h, key_padding_mask=key_padding_mask
        )
        h = self.attn_norm(h + attn_out)                   # [B, 63, hidden_dim]

        # Step 3：逐 patch 输出
        out = self.head(h.view(B * N, -1))                 # [B*63, 1]
        return out


# ==========================================
# 3. 质量感知动态门控网络（保持不变）
# ==========================================
class MoMEGatingNetwork(nn.Module):
    def __init__(self, dim_f3_stats=8, dim_f3_mae=384, dim_f2=1024):
        super().__init__()
        self.quality_proj = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 32)
        )
        concat_dim = dim_f3_stats + dim_f3_mae + dim_f2 + 32

        self.gate = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, f3_stats, f3_mae, f2, quality_2d):
        q_feat = self.quality_proj(quality_2d)
        combined_feat = torch.cat([f3_stats, f3_mae, f2, q_feat], dim=-1)
        return self.gate(combined_feat)


# ==========================================
# 4. Road-MoME 主引擎（v13.1 统一监督单头）
# ==========================================
class MoMEEngine(nn.Module):
    def __init__(self, dim_f3_stats=8, dim_f3_mae=384, hidden_dim=256):
        super().__init__()

        # --- 1. 活体 2D 提取器（不变） ---
        base_model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.live_2d_backbone = create_feature_extractor(
            base_model, return_nodes={"features.3": "feat"}
        )
        for name, param in self.live_2d_backbone.named_parameters():
            if not name.startswith("features.3"):
                param.requires_grad = False
        self.dim_f2 = 1024

        # --- 2. 跨模态注意力（不变） ---
        self.cross_modal_attn = CrossModalAttention(
            dim_2d=self.dim_f2, dim_3d=dim_f3_mae, d_model=256, num_heads=8
        )

        # --- 3. 差异化专家池（v12.0，不变） ---
        self.expert_phys = PhysExpert(input_dim=dim_f3_stats)
        self.expert_geom = GeomExpert(input_dim=dim_f3_mae, hidden_dim=hidden_dim)
        self.expert_tex  = TexExpert(input_dim=self.dim_f2, hidden_dim=hidden_dim)

        # --- 4. 门控网络（不变） ---
        self.gating = MoMEGatingNetwork(dim_f3_stats, dim_f3_mae, self.dim_f2)

    def forward(self, img_tensor, rois_tensor, f3_stats, f3_mae, q_2d, valid_mask):
        B = img_tensor.size(0)
        num_patches = 63

        # --------------------------------------------------
        # Phase 1：视觉活体流处理（不变）
        # --------------------------------------------------
        spatial_feat = self.live_2d_backbone(img_tensor)["feat"]
        pooled_feat = roi_align(
            spatial_feat, rois_tensor, output_size=(2, 2), spatial_scale=1 / 8.0
        )
        # [B*63, C, 2, 2] → [B, 63, dim_f2]
        f2_tex = pooled_feat.view(pooled_feat.size(0), -1).view(B, num_patches, self.dim_f2)
        f2_tex = f2_tex * valid_mask.unsqueeze(-1)

        # --------------------------------------------------
        # Phase 2：跨模态深度交互（v14.0：传入 valid_mask 屏蔽盲区 key）
        # --------------------------------------------------
        f2_enhanced = self.cross_modal_attn(f2_tex, f3_mae, valid_mask)
        f2_enhanced = f2_enhanced * valid_mask.unsqueeze(-1)

        # --------------------------------------------------
        # Phase 3：差异化专家独立诊断
        # --------------------------------------------------
        # Phys：展平输入，输出 [B*63, 1]
        pred_phys_flat = self.expert_phys(
            f3_stats.view(B * num_patches, -1)
        )

        # Geom：序列输入 [B, 63, 384]，内部计算对比特征后展平输出
        pred_geom_flat = self.expert_geom(f3_mae)

        # Tex：序列输入 [B, 63, dim_f2]，传入 valid_mask 屏蔽盲区 patch
        pred_tex_flat = self.expert_tex(f2_enhanced, valid_mask)

        pred_phys = pred_phys_flat.view(B, num_patches)
        pred_geom = pred_geom_flat.view(B, num_patches)
        pred_tex  = pred_tex_flat.view(B, num_patches)

        # --------------------------------------------------
        # Phase 4：决策级动态融合（v13.1 回归单头）
        # --------------------------------------------------
        weights = self.gating(f3_stats, f3_mae, f2_enhanced, q_2d)

        expert_preds = torch.stack([pred_phys, pred_geom, pred_tex], dim=-1)
        final_logit  = (expert_preds * weights).sum(dim=-1)   # [B, 63]

        internals = {
            "pred_phys": pred_phys,
            "pred_geom": pred_geom,
            "pred_tex":  pred_tex,
            "weights":   weights,
        }

        return final_logit, internals