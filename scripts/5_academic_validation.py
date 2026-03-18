"""
[Step 5] MoME v2.8 学术压力测试
修复说明：适配 4 个返回值，并增加不确定性响应分析。
"""

import os
import sys
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.mome_model import build_mome_model


def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_academic_tests():
    model = build_mome_model(cfg).to(DEVICE)
    model.load_state_dict(
        torch.load(cfg["paths"]["weights"]["mome_model"], map_location=DEVICE)
    )
    model.eval()

    npz_dir = Path(cfg["paths"]["output_dir"])
    test_file = sorted(list(npz_dir.glob("*.npz")))[0]
    data = np.load(test_file, allow_pickle=True)

    phys = torch.from_numpy(data["phys_8d"]).float().unsqueeze(0).to(DEVICE)
    geom = torch.from_numpy(data["deep_512d"]).float().unsqueeze(0).to(DEVICE)
    tex = torch.from_numpy(data["deep_2d_768d"]).float().unsqueeze(0).to(DEVICE)
    q_geo = torch.from_numpy(data["meta"][:, 1:2]).float()

    # --- 实验 1: 鲁棒性扫描 ---
    q2_steps = np.linspace(0.01, 1.0, 15)
    w_history = []
    u_history = []

    for q2 in tqdm(q2_steps, desc="Scanning q2"):
        q_vec = (
            torch.cat([q_geo, torch.full_like(q_geo, q2)], dim=-1)
            .unsqueeze(0)
            .to(DEVICE)
        )
        with torch.no_grad():
            _, weights, _, uncerts = model(phys, geom, tex, q_vec)
            w_history.append(weights[0].mean(dim=0).cpu().numpy())
            # [修复点] 视觉不确定性 (Tex Expert) 对应索引为 2
            u_history.append(
                torch.exp(0.5 * uncerts[0, :, 2]).mean().item()
            )  # Avg Sigma

    w_history = np.array(w_history)

    plt.figure(figsize=(10, 6))
    labels = ["Phys", "3D-Geom", "2D-Tex", "Synergy"]
    for i in range(4):
        plt.plot(q2_steps, w_history[:, i], label=labels[i], marker="o")
    plt.xlabel("Image Quality (q2)")
    plt.ylabel("Gating Weight")
    plt.title("v2.8 Model Response to Visual Quality Scan")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = project_root / "logs" / "academic_v28_scan.png"
    plt.savefig(save_path)
    print(f"✅ 学术扫描图已生成: {save_path}")


if __name__ == "__main__":
    run_academic_tests()
