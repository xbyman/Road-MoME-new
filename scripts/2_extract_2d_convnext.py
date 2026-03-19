"""
[Step 1] ConvNeXt-Base 2D 前沿特征提取器 (定点覆盖版)

核心重构：
1. 学术前沿替换：使用 2022 年顶流卷积架构 ConvNeXt 替换掉 ResNet50。
2. 极细粒度保留：精准提取 features.3 (Stage 2) 的特征图，Stride=8，完美保留裂缝的高频边缘。
3. 维度计算：Stage 2 输出通道数为 256。经过 2x2 ROI Align 后，最终特征降至极致的 1024 维。
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
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.ops import roi_align

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ConvNeXt 标准预处理 (与官方 ImageNet 预训练一致)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def process_frame(npz_path, img_dir, img_map, model, device):
    try:
        data = np.load(npz_path, allow_pickle=True)
        npz_dict = {k: data[k] for k in data.keys()}
    except Exception as e:
        print(f"❌ 无法读取 {npz_path.name}: {e}")
        return False

    frame_id = npz_path.stem.replace("pkg_", "")

    # 极速 O(1) 时间戳目录推导
    timestamp_folder = frame_id.split(".")[0]
    img_path = Path(img_dir) / timestamp_folder / "left" / f"{frame_id}.jpg"

    # 如果推导失败，回退使用全局索引
    if not img_path.exists():
        img_path = img_map.get(frame_id)

    if img_path is None or not img_path.exists():
        print(f"⚠️ 找不到图像: {frame_id}.jpg")
        return False

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return False

    orig_h, orig_w = img_bgr.shape[:2]
    # ConvNeXt 同样兼容该分辨率
    target_w, target_h = 1008, 560
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_w, target_h))

    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    # ==========================================
    # 提取 ConvNeXt Stage 2 特征 (Stride = 8)
    # ==========================================
    with torch.no_grad():
        # 提取指定的 features.3 节点
        features = model(img_tensor)
        spatial_feat = features["feat"]

    patch_corners_uv = npz_dict.get("patch_corners_uv")
    meta = npz_dict.get("meta")
    if patch_corners_uv is None or meta is None:
        return False

    valid_mask = meta[:, 2]
    rois = []

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    # 构建包围盒 (Bounding Box)
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

    # ==========================================
    # ROI Align (基于 stride=8 的缩放率)
    # ==========================================
    # spatial_scale=1/8.0 精准对齐感受野
    pooled_feat = roi_align(
        spatial_feat, rois_tensor, output_size=(2, 2), spatial_scale=1 / 8.0
    )

    # 展平输出: [63, 256通道, 2, 2] -> [63, 1024维]
    final_2d_feat = pooled_feat.view(pooled_feat.size(0), -1).cpu().numpy()
    final_2d_feat[valid_mask == 0] = 0.0

    # 保持键名兼容性原位覆盖
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
        print("❌ 找不到 npz 文件")
        return

    print("=" * 60)
    print("🚀 Road-MoME [ConvNeXt-Base] 特征提取脚本启动")
    print("🧠 逻辑: 调用 torchvision 原生 ConvNeXt 提取高分辨率边缘特征")
    print(f"🎛️ 设备: {device}")
    print("=" * 60)

    img_map = {}
    index_path = npz_dir.parent / "dataset_index.yaml"
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                for item in yaml.safe_load(f):
                    img_map[str(item["id"])] = Path(item["img"])
        except Exception:
            pass

    # ==========================================
    # 核心：自动下载并加载 ConvNeXt-Base 官方预训练权重
    # ==========================================
    print("⏳ 正在请求 ConvNeXt-Base 预训练权重 (首次运行会自动下载)...")
    base_model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).to(device)

    # 切开网络，精准提取 Stage 2 的输出 (即 features.3 模块)
    model = create_feature_extractor(base_model, return_nodes={"features.3": "feat"})
    model.eval()

    success_count = 0
    pbar = tqdm(npz_files, desc="Extracting ConvNeXt Features")
    for npz_path in pbar:
        if process_frame(npz_path, img_dir, img_map, model, device):
            success_count += 1

    print("=" * 60)
    print(f"✨ ConvNeXt 前沿特征覆盖完成! ({success_count}/{len(npz_files)})")
    print("⚠️ 【必须执行】: 请立即打开 config.yaml，将 2d input_dim 改为 1024！")
    print("=" * 60)


if __name__ == "__main__":
    main()