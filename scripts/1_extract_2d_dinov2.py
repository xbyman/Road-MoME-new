"""
[Step 1] DINOv2 2D 特征提取器 (浅层纹理定点覆盖版)

核心重构：
1. 浅层视力觉醒：将 DINOv2 的提取目标从深层语义 (默认 n=4，即最后4层) 强行修改为前 4 层 (n=[0, 1, 2, 3])。
   保留极致的高频边缘和纹理信息，专门针对马路裂缝这种细粒度病害。
2. 定点外科手术：读取现有的 .npz 文件，仅覆盖更新 'deep_2d_768d' 字段，完美保留现有的 3D 和物理特征。
3. 路径寻址恢复：回归原生 O(1) 的时间戳目录推导逻辑与 dataset_index 索引，彻底解决找不到图像的 Bug。
"""

import os
import sys
import torch
import numpy as np
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torchvision.ops import roi_align

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------------------
# 图像预处理 (严格遵循 DINOv2 的输入标准)
# ------------------------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def process_frame(npz_path, img_dir, img_map, model, device):
    """
    对单个 npz 文件进行 2D 特征的定点重写
    """
    # 1. 安全加载现有数据
    try:
        data = np.load(npz_path, allow_pickle=True)
        # 将所有内容转存为 dict，以便修改和重新保存
        npz_dict = {k: data[k] for k in data.keys()}
    except Exception as e:
        print(f"❌ 无法读取 {npz_path.name}: {e}")
        return False

    frame_id = npz_path.stem.replace("pkg_", "")

    # =========================================================
    # 【核心修复】回归极其高效的 O(1) 原生目录推导逻辑
    # 帧格式: 20230317074848.400 -> 时间戳文件夹: 20230317074848
    # =========================================================
    timestamp_folder = frame_id.split(".")[0]
    img_path = Path(img_dir) / timestamp_folder / "left" / f"{frame_id}.jpg"

    # 如果推导路径不存在，则从官方索引 dataset_index.yaml 中安全回退寻找
    if not img_path.exists():
        img_path = img_map.get(frame_id)

    if img_path is None or not img_path.exists():
        print(f"⚠️ 找不到对应图像: {frame_id}.jpg，跳过此帧。")
        return False

    # 2. 读取并 Resize 图像 (固定为 DINOv2 Patch14 的整数倍)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"⚠️ 图像损坏无法读取: {img_path.name}")
        return False

    orig_h, orig_w = img_bgr.shape[:2]

    target_w, target_h = 1008, 560
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_w, target_h))

    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    # 3. 提取浅层特征 (核心修改所在)
    with torch.no_grad():
        # n=[0, 1, 2, 3] 提取前 4 个 Block 的特征，关注底层纹理、边缘和角点
        feat_list = model.get_intermediate_layers(
            img_tensor, n=[0, 1, 2, 3], return_class_token=False
        )
        stacked_feat = torch.stack(feat_list, dim=1)  # [1, 4, 2880, 768]

        feat_h, feat_w = target_h // 14, target_w // 14  # 40, 72
        spatial_feat = stacked_feat.reshape(1, 4, feat_h, feat_w, 768)
        spatial_feat = spatial_feat.permute(0, 1, 4, 2, 3).reshape(
            1, 4 * 768, feat_h, feat_w
        )

    # 4. 基于现有 npz 中的网格坐标 (patch_corners_uv) 提取局部特征
    patch_corners_uv = npz_dict.get("patch_corners_uv")
    meta = npz_dict.get("meta")

    if patch_corners_uv is None or meta is None:
        print(f"⚠️ {npz_path.name} 缺少投影坐标或 meta 信息，跳过。")
        return False

    valid_mask = meta[:, 2]
    rois = []

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    # 构建 RoI 列表
    for i in range(len(patch_corners_uv)):
        if valid_mask[i] == 0:
            rois.append([0, 0, 0, 1, 1])
            continue

        corners = patch_corners_uv[i]

        x_min = max(0, np.min(corners[:, 0]) * scale_x)
        x_max = min(target_w, np.max(corners[:, 0]) * scale_x)
        y_min = max(0, np.min(corners[:, 1]) * scale_y)
        y_max = min(target_h, np.max(corners[:, 1]) * scale_y)

        if x_max <= x_min or y_max <= y_min:
            rois.append([0, 0, 0, 1, 1])
        else:
            rois.append([0, x_min, y_min, x_max, y_max])

    rois_tensor = torch.tensor(rois, dtype=torch.float32).to(device)

    # 5. ROI Align 池化 (2x2)
    pooled_feat = roi_align(
        spatial_feat, rois_tensor, output_size=(2, 2), spatial_scale=1 / 14.0
    )
    final_2d_feat = pooled_feat.view(pooled_feat.size(0), -1).cpu().numpy()
    final_2d_feat[valid_mask == 0] = 0.0

    # 6. 【核心】更新字典并原位覆盖保存
    npz_dict["deep_2d_768d"] = final_2d_feat.astype(np.float32)
    np.savez_compressed(npz_path, **npz_dict)

    return True


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_dir = Path(cfg["paths"]["output_dir"])
    img_dir = Path(cfg["paths"]["raw_img_dir"])

    npz_files = sorted(list(npz_dir.glob("pkg_*.npz")))
    if not npz_files:
        print(f"❌ 找不到任何 npz 文件于: {npz_dir}")
        return

    print("=" * 60)
    print("🚀 Road-MoME DINOv2 浅层特征更新脚本已启动")
    print("🧠 逻辑: 放弃深层语义，强行提取前 4 层 (n=[0,1,2,3]) 高频边缘特征")
    print("📦 模式: 极速路径匹配，定点覆盖现有的 npz 文件")
    print(f"🎛️ 设备: {device}")
    print("=" * 60)

    # 加载官方 YAML 索引作为保底防线
    img_map = {}
    index_path = npz_dir.parent / "dataset_index.yaml"
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = yaml.safe_load(f)
                for item in index_data:
                    img_map[str(item["id"])] = Path(item["img"])
            print(f"✅ 已加载备用全局索引: 包含 {len(img_map)} 条映射。")
        except Exception as e:
            print(f"⚠️ 无法读取 dataset_index.yaml: {e}")

    # 加载预训练模型
    print("⏳ 正在加载 DINOv2 (ViT-B/14)...")
    dinov2_path = cfg["paths"]["weights"].get("dinov2", "")
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=False
        )
        if dinov2_path and os.path.exists(dinov2_path):
            model.load_state_dict(torch.load(dinov2_path))
            print("✅ 成功加载本地权重。")
        else:
            print("⚠️ 未找到本地权重，将从网络下载预训练参数。")
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
            )
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    model = model.to(device)
    model.eval()

    success_count = 0
    pbar = tqdm(npz_files, desc="Updating 2D Features")
    for npz_path in pbar:
        if process_frame(npz_path, img_dir, img_map, model, device):
            success_count += 1

    print("=" * 60)
    print(f"✨ 特征覆盖完成! 成功更新 {success_count}/{len(npz_files)} 个文件。")
    print("💡 下一步：请直接运行 3_train_mome.py 开始训练新的特征。")


if __name__ == "__main__":
    main()
