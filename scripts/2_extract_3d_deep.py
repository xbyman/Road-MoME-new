"""
[Step 2] Road-MoME 3D 深度特征批量提取器 (v9.6 极速 GPU 采样版)

核心重构 (完美匹配 Step 0 v9.5 极速版):
1. 实时 FPS 算子集成: 直接读取 sampled_pts，在 5090 GPU 上实时计算 FPS 和 Grouping。
2. 抛弃旧包依赖: 完全无视 .npz 中的 centers/neighborhood 占位符，解决形状维度报错。
3. 索引映射填零: 利用 valid_mask 仅推理有效 Patch，确保输出特征 (63, 384) 刚性对齐。
4. 批量处理加速: 针对 RTX 5090 优化，支持多帧并行推理。
"""

import os
import sys
import uuid
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# 环境设置
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.backbones import RoadPointMAEEncoder, load_official_pretrain

def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fps_and_group_batched(pts_tensor, n_group=128, group_size=32):
    """
    针对 Patch Batch 进行并行 FPS 采样与分组 (Pure PyTorch)
    输入: [N, 8192, 3] (N 是有效 Patch 总数)
    输出: neighborhood [N, 128, 32, 3], center [N, 128, 3]
    """
    B, N, _ = pts_tensor.shape
    device = pts_tensor.device

    centroids = torch.zeros(B, n_group, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(n_group):
        centroids[:, i] = farthest
        centroid = pts_tensor[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((pts_tensor - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    center = torch.gather(pts_tensor, 1, centroids.unsqueeze(-1).expand(-1, -1, 3))
    dist_mat = torch.cdist(center, pts_tensor)
    idx = dist_mat.topk(k=group_size, dim=-1, largest=False)[1]

    B_idx = batch_indices.view(B, 1, 1).expand(-1, n_group, group_size)
    neighborhood = pts_tensor[B_idx, idx, :]
    neighborhood = neighborhood - center.unsqueeze(2) # 局部归一化

    return neighborhood, center

class ExtractionDataset(Dataset):
    """
    增量式数据集加载器 - 直接读取原始采样点
    """
    def __init__(self, npz_files, force_reextract=False):
        self.files = []
        print("🔍 正在执行 3D 推理进度审计...")
        for f in tqdm(npz_files, desc="完整性校验"):
            if not force_reextract:
                try:
                    with np.load(f, allow_pickle=True) as data:
                        if "deep_512d" in data:
                            continue
                except Exception:
                    pass
            self.files.append(f)
        print(f"✅ 审计完成。待处理: {len(self.files)} 帧 | 已跳过: {len(npz_files)-len(self.files)} 帧")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = self.files[idx]
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                # 核心修复: 不再读取 neighborhood，只读取 sampled_pts
                pts_array = data["sampled_pts"]  # [63, 8192, 3]
                meta_array = data["meta"]        # [63, 3]
                
            valid_mask = meta_array[:, 2] == 1.0
            
            return {
                "pts": torch.from_numpy(pts_array).float(),
                "valid_mask": torch.from_numpy(valid_mask).bool(),
                "path": str(npz_path)
            }
        except Exception as e:
            return {"error": f"Load error {npz_path.name}: {e}"}

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 因为在 GPU 上实时算 FPS 消耗显存，5090 建议先设为 8 或 16
    batch_size = cfg["inference"].get("inference_batch_size", 16) 
    npz_dir = Path(cfg["paths"]["output_dir"])
    all_npz_files = sorted(list(npz_dir.glob("pkg_*.npz")))

    if not all_npz_files:
        print(f"❌ 错误: 在 {npz_dir} 中未发现特征包。")
        return

    # 1. 进度审计
    dataset = ExtractionDataset(all_npz_files)
    if len(dataset) == 0:
        print("🎉 所有特征已完成提取。")
        return
        
    # collate_fn 鲁棒性：剔除加载失败的空件，防止 DataLoader 报错
    def safe_collate(batch):
        return [item for item in batch if item is not None and "error" not in item]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=safe_collate)

    # 2. 初始化编码器
    print("🤖 正在加载纯 PyTorch 版 Point-MAE 编码器 (v9.6)...")
    model = RoadPointMAEEncoder(trans_dim=384, depth=12, num_heads=6).to(device)
    ckpt_path = cfg["paths"]["weights"]["point_mae"]
    model = load_official_pretrain(model, ckpt_path).eval()

    print(f"🚀 启动批量提取 | Batch Size: {batch_size} | 实时 FPS 加速: 开启")
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="3D 推理中"):
            if not batch_data: continue 
            
            B = len(batch_data)
            all_pts = torch.stack([d["pts"] for d in batch_data]).to(device)          # [B, 63, 8192, 3]
            all_masks = torch.stack([d["valid_mask"] for d in batch_data]).to(device) # [B, 63]
            paths = [d["path"] for d in batch_data]

            # 展平所有 Patch 以便统一筛选
            pts_flat = all_pts.view(-1, 8192, 3)
            mask_flat = all_masks.view(-1)
            
            valid_indices = torch.where(mask_flat)[0]
            full_feats_flat = torch.zeros((B * 63, 384), device=device)

            if len(valid_indices) > 0:
                v_pts = pts_flat[valid_indices]
                
                # [核心重构]: 丢弃预处理占位符，在此处让 5090 实时计算 FPS
                v_neigh, v_center = fps_and_group_batched(v_pts)
                
                # 模型推理
                v_feats = model(v_neigh, v_center) 
                
                # 索引映射填零回填
                full_feats_flat[valid_indices] = v_feats

            final_feats = full_feats_flat.view(B, 63, 384).cpu().numpy()

            # 逐帧安全写回 .npz
            for i in range(B):
                npz_path = Path(paths[i])
                try:
                    with np.load(npz_path, allow_pickle=True) as existing_data:
                        data_dict = dict(existing_data)
                    data_dict["deep_512d"] = final_feats[i].astype(np.float32)
                    
                    unique_id = uuid.uuid4().hex[:6]
                    tmp_path = npz_path.parent / f"tmp_3d_{unique_id}_{npz_path.name}"
                    np.savez_compressed(tmp_path, **data_dict)
                    os.replace(tmp_path, npz_path)
                except Exception as e:
                    print(f"❌ 写入 {npz_path.name} 失败: {e}")

    print("\n✨ 3D 特征提取任务已圆满完成！")

if __name__ == "__main__":
    main()