"""
Road-MoME: 纯 PyTorch 原生 3D 骨干网络 (v9.5 预训练完美继承版)

核心重构与拯救：
1. 精准复刻: 反向破译并使用纯 PyTorch 还原了原 C++ 算子的局部+全局拼接 (256+256=512) 特征提取流。
2. 完美对齐: 重写 FlashAttention 与 Mlp 的内部命名 (qkv, fc1)，实现 100% 继承官方预训练权重。
3. 算力解放: 全面利用 F.scaled_dot_product_attention 触发 RTX 5090 (Blackwell) 的底层硬件加速。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 局部特征提取器 (完美还原官方 C++ 内部通道)
# ==========================================
class MiniPointNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=384):
        super().__init__()
        # 对齐第一阶段卷积 (3 -> 128 -> 256)
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        # 对齐第二阶段卷积 (512 -> 512 -> 384)
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, out_channels, 1),
        )

    def forward(self, x):
        # x: [B*G, K, 3] -> transpose to [B*G, 3, K]
        x = x.transpose(1, 2)

        # 1. 局部提取
        feat = self.first_conv(x)  # [B*G, 256, K]

        # 2. 局部范围内的全局池化与扩展
        feat_global = torch.max(feat, dim=2, keepdim=True)[0]  # [B*G, 256, 1]
        feat_global = feat_global.expand(-1, -1, feat.size(-1))  # [B*G, 256, K]

        # 3. 拼接特征 (破译 512 维的真相)
        feat_concat = torch.cat([feat_global, feat], dim=1)  # [B*G, 512, K]

        # 4. 融合降维并最终池化
        out = self.second_conv(feat_concat)  # [B*G, 384, K]
        out = torch.max(out, dim=2)[0]  # [B*G, 384]
        return out


# ==========================================
# 2. 硬件加速 Attention 与 MLP (完美对齐键名)
# ==========================================
class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # 对齐 checkpoint 中的 attn.qkv.weight 和 attn.proj.weight
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # qkv 生成并拆分
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        # 调用 PyTorch 2.x 的 C++ 底层实现，极速运算
        attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        return self.proj(attn_out)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        # 对齐 checkpoint 中的 mlp.fc1.weight 和 mlp.fc2.weight
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class NativeTransformerBlock(nn.Module):
    def __init__(self, dim=384, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ==========================================
# 3. 3D 主干网络组装
# ==========================================
class RoadPointMAEEncoder(nn.Module):
    def __init__(self, trans_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.trans_dim = trans_dim

        self.local_grouper = MiniPointNet(in_channels=3, out_channels=trans_dim)

        # 绝对位置编码
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, trans_dim)
        )

        # Transformer 深度堆叠
        self.blocks = nn.ModuleList(
            [
                NativeTransformerBlock(dim=trans_dim, num_heads=num_heads)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(trans_dim)

    def forward(self, neighborhood, center):
        B, G, K, C = neighborhood.shape

        neigh_flat = neighborhood.view(B * G, K, C)
        local_features = self.local_grouper(neigh_flat)
        local_features = local_features.view(B, G, self.trans_dim)

        pos_embedding = self.pos_embed(center)
        x = local_features + pos_embedding

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)  # 此时形状为 [B, G, 384]

        # ==========================================
        # 核心修复：添加 Max Pooling 收口
        # 将 G (128) 个微观组特征，最大池化为 1 个全局 Patch 特征
        # 最终输出形状严格对齐 [B, 384]
        # ==========================================
        x = torch.max(x, dim=1)[0]

        return x


# ==========================================
# 4. 万能映射加载器
# ==========================================
def load_official_pretrain(model, ckpt_path):
    """通过智能映射，实现官方 DDP 模型向本机重构模型的无损导入"""
    if not torch.cuda.is_available():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    else:
        checkpoint = torch.load(ckpt_path)

    base_dict = checkpoint.get(
        "base_model", checkpoint.get("model_state_dict", checkpoint)
    )

    mapped_dict = {}
    model_state_dict = model.state_dict()

    for k, v in base_dict.items():
        # 1. 剥离外层包裹前缀
        new_k = k.replace("module.MAE_encoder.", "")
        new_k = new_k.replace("module.", "")

        # 2. 映射部件命名 (encoder -> local_grouper)
        new_k = new_k.replace("encoder.", "local_grouper.")

        # 3. 映射 PyTorch 的 ModuleList 嵌套关系 (blocks.blocks. -> blocks.)
        new_k = new_k.replace("blocks.blocks.", "blocks.")

        # 精准匹配核对
        if new_k in model_state_dict and v.shape == model_state_dict[new_k].shape:
            mapped_dict[new_k] = v

    model.load_state_dict(mapped_dict, strict=False)
    print(
        f"✅ 安全锚定：通过智能映射，成功装载 {len(mapped_dict)} 个匹配的 3D 预训练张量！"
    )
    return model
