"""
[Step 2] 3D 几何特征提取 (v7.2 索引映射对齐版)
核心升级：
1. 索引映射填零：利用 valid_mask 仅对有效 Patch 进行推理，极大节省显存。
2. 绝对维度对齐：通过底座填充法，确保输出特征永远保持 (63, 384) 的刚性形状。
3. 并发安全：引入 UUID 临时文件后缀，彻底解决多进程读写时的覆盖报错。
"""

import os
import sys
import uuid
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import yaml

# 环境设置
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.backbones import RoadPointMAEEncoder, load_official_pretrain


def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fps_and_group_batched(pts_tensor, n_group=128, group_size=32):
    """针对分片 Batch 进行并行 FPS 采样与分组"""
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
    neighborhood = neighborhood - center.unsqueeze(2)

    return neighborhood, center


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 显存控制参数
    INF_BATCH_SIZE = cfg["inference"].get("inference_batch_size", 16)
    npz_dir = Path(cfg["paths"]["output_dir"])
    all_npz_files = sorted(list(npz_dir.glob("pkg_*.npz")))

    if not all_npz_files:
        print(f"⚠️ 警告: 在 {npz_dir} 中未发现特征包，请先正确运行 Step 0。")
        return

    # --- 第一阶段：审计进度 ---
    pending_files = []
    skipped_count = 0
    print(f"🔍 正在执行 3D 推理进度审计...")
    for npz_path in tqdm(all_npz_files, desc="完整性校验"):
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "deep_512d" in data.files:
                    skipped_count += 1
                else:
                    pending_files.append(npz_path)
        except Exception:
            # 文件损坏则删除重建
            npz_path.unlink(missing_ok=True)

    if skipped_count > 0:
        print(f"ℹ️ 自动跳过 {skipped_count} 个已完成 3D 提取的包。")

    if not pending_files:
        print("✅ 检查完毕：所有样本的 3D 特征均已存在。")
        return

    # --- 第二阶段：Point-MAE 推理 ---
    print(
        f"\n🚀 启动 3D 深度特征推理 (索引映射隔离版) | 待处理: {len(pending_files)} 帧"
    )

    model = RoadPointMAEEncoder(trans_dim=384, depth=12, num_heads=6)
    ckpt_path = cfg["paths"]["weights"]["point_mae"]
    model = load_official_pretrain(model, ckpt_path).to(device).eval()

    processed_count = 0
    with torch.no_grad():
        for npz_path in tqdm(pending_files, desc="3D 推理中"):
            try:
                # 1. 安全解包入内存
                with np.load(npz_path, allow_pickle=True) as data:
                    if "sampled_pts" not in data or "meta" not in data:
                        continue
                    pts_array = data["sampled_pts"]  # [63, 8192, 3]
                    meta_array = data["meta"]  # [63, 3]
                    data_dict = dict(data)

                total_patches = len(pts_array)
                # 提取 valid_mask (第3个元素)
                valid_mask = meta_array[:, 2] == 1.0
                valid_indices = np.where(valid_mask)[0]

                # 初始化 100% 维度对齐的全 0 特征底座
                # 特征维度为 384 (与 Point-MAE 架构对应)
                full_patch_feats = np.zeros((total_patches, 384), dtype=np.float32)

                # 仅当存在有效 Patch 时才启动 GPU 推理
                if len(valid_indices) > 0:
                    valid_pts = pts_array[valid_indices]
                    valid_feats_list = []

                    # 2. 分片批处理（仅针对有效点，保护显存）
                    for i in range(0, len(valid_pts), INF_BATCH_SIZE):
                        sub_pts = valid_pts[i : i + INF_BATCH_SIZE]
                        input_tensor = torch.from_numpy(sub_pts).float().to(device)

                        # 并行采样与模型推理
                        neigh, center = fps_and_group_batched(input_tensor)
                        feats = model(neigh, center).cpu().numpy()
                        valid_feats_list.append(feats)

                    # 3. 合并有效特征
                    valid_feats_cat = np.concatenate(valid_feats_list, axis=0)

                    # 4. 索引映射填零法：将算出的特征填回原本对应的绝对位置
                    full_patch_feats[valid_indices] = valid_feats_cat

                # 5. 保存结果 (保持键名兼容性)
                data_dict["deep_512d"] = full_patch_feats

                # 6. 多进程安全的原子化写入
                unique_id = uuid.uuid4().hex[:8]
                tmp_path = npz_path.parent / f"tmp_3d_{unique_id}_{npz_path.name}"
                np.savez_compressed(tmp_path, **data_dict)
                os.replace(tmp_path, npz_path)

                processed_count += 1

                # 显存碎片清理
                if processed_count % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n❌ 处理 {npz_path.name} 出错: {e}")

    print(f"\n✨ 3D 特征提取圆满完成！新处理: {processed_count} 帧")


if __name__ == "__main__":
    main()
