import torch

# 请确保路径正确
ckpt_path = r"C:\Users\31078\Desktop\ROAD\pretrained_models\pretrain.pth"

try:
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # 精准提取真正的权重字典
    state_dict = checkpoint["base_model"]

    print("=" * 60)
    print("🔍 骨干网络的前 40 个 Key 命名 (寻找 PointNet 相关的层):")
    print("=" * 60)

    for i, key in enumerate(list(state_dict.keys())[:40]):
        # 安全读取形状
        tensor = state_dict[key]
        shape = tensor.shape if hasattr(tensor, "shape") else "Not a Tensor"
        print(f"[{i}] {key}: {shape}")

except Exception as e:
    print(f"❌ 读取失败: {e}")
