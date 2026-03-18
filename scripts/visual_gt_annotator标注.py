"""
Road-MoME 视觉真值标注工具 (v4.0 数据直读版 - SSOT)
核心重构：
1. 纯粹渲染器：彻底剔除 RSRDProjector 和所有 3D 物理计算逻辑。
2. 绝对对齐 (SSOT)：直接读取 Step 0 生成的 `.npz` 容器中的 `patch_corners_uv` 进行渲染。你画的网格，就是 Point-MAE 和 DINOv2 吃进去的物理边界，0 误差对齐。
3. 盲区跳过：自动读取 `.npz` 里的 `valid_mask`，屏蔽无效的点云盲区。
"""

import cv2
import numpy as np
import yaml
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

# ==================== 协作分工配置区 ====================
USER_NAME = "Annotatordense"  # 标注员姓名 (用于区分保存文件)

# 指定分配给该标注员的时间戳文件夹索引范围 [start, end)
FOLDER_RANGE = [0, 30]
# ======================================================

# --- 1. 配置加载与环境准备 ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

CONFIG_PATH = project_root / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_IMG_DIR = Path(cfg["paths"]["raw_img_dir"])
NPZ_DIR = Path(cfg["paths"]["output_dir"])
SAVE_PATH = project_root / "data" / f"manual_visual_gt_{USER_NAME}.json"


class ProfessionalAnnotator:
    def __init__(self):
        # 1. 加载已有标注实现“自动续标”
        self.all_manual_labels = {}
        if SAVE_PATH.exists():
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                self.all_manual_labels = json.load(f)
            print(
                f"📂 [{USER_NAME}] 检测到已有存档，已加载 {len(self.all_manual_labels)} 帧记录。"
            )

        # 2. 建立特征包索引 (确保只标注有特征数据支持的帧)
        print("🔍 正在扫描底层的 NPZ 特征包索引...")
        self.available_npz_stems = {
            p.stem.replace("pkg_", "") for p in NPZ_DIR.glob("*.npz")
        }

        # 3. 任务分配与过滤
        all_timestamp_dirs = sorted(
            [d for d in RAW_IMG_DIR.iterdir() if d.is_dir() and (d / "left").exists()]
        )

        total_folders = len(all_timestamp_dirs)
        start, end = FOLDER_RANGE
        end = min(end, total_folders)
        my_dirs = all_timestamp_dirs[start:end]

        self.todo_list = []
        for d in my_dirs:
            left_dir = d / "left"
            imgs = sorted(list(left_dir.glob("*.jpg")))
            for p in imgs:
                # 只有当预处理的 .npz 存在，且未被标注过时，才加入待办
                if p.stem in self.available_npz_stems:
                    if p.name not in self.all_manual_labels:
                        self.todo_list.append(p)

        print(f"📊 任务分配报告:")
        print(f"   - 负责目录范围: [{start}:{end}] (共 {len(my_dirs)} 个目录)")
        print(f"   - 已同步物理包: {len(self.available_npz_stems)} 帧")
        print(f"   - 已标注数量: {len(self.all_manual_labels)} 帧")
        print(f"   - 剩余待标注: {len(self.todo_list)} 帧 (重启后自动跳转)")

        # 刚性维度锁定
        self.num_patches = 63

        # 状态控制
        self.current_labels = [0] * self.num_patches
        self.current_polygons = []
        self.valid_mask_ui = []
        self.is_drawing = False
        self.draw_mode = 1
        self.should_save = False
        self.should_quit = False

        # UI 按钮布局
        self.footer_h = 80
        self.buttons = {
            "save": [20, 10, 220, 60, "SAVE & NEXT"],
            "clear": [260, 10, 120, 60, "CLEAR"],
            "quit": [400, 10, 120, 60, "QUIT"],
        }

    def _load_polygons_from_npz(self, frame_id):
        """核心重构：直接从 NPZ 提取物理预处理算好的梯形顶点"""
        self.current_polygons = []
        self.valid_mask_ui = []

        npz_path = NPZ_DIR / f"pkg_{frame_id}.npz"

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                patch_corners_uv = data["patch_corners_uv"]  # [63, 4, 2]
                meta = data["meta"]  # [63, 3]
        except Exception as e:
            print(f"\n❌ 读取特征包失败 {npz_path}: {e}")
            return False

        if len(patch_corners_uv) != self.num_patches:
            print(
                f"\n⚠️ 警告: 维度断层！该特征包包含 {len(patch_corners_uv)} 个Patch，而非刚性规定的 63个。"
            )
            return False

        for i in range(self.num_patches):
            # 获取物理有效性 (第3个元素)
            is_valid = meta[i, 2] == 1.0

            if not is_valid:
                self.valid_mask_ui.append(False)
                self.current_polygons.append(None)
                continue

            corners = patch_corners_uv[i]
            if np.any(np.isnan(corners)):
                self.valid_mask_ui.append(False)
                self.current_polygons.append(None)
            else:
                self.valid_mask_ui.append(True)
                self.current_polygons.append(corners.astype(np.int32))

        return True

    def get_patch_idx(self, x, y):
        """判定鼠标击中哪个梯形"""
        for idx, poly in enumerate(self.current_polygons):
            if not self.valid_mask_ui[idx]:
                continue
            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                return idx
        return None

    def on_mouse(self, event, x, y, flags, param):
        img_h = param["h"]

        if event == cv2.EVENT_LBUTTONDOWN:
            if y > img_h:
                local_y = y - img_h
                for btn_id, (bx, by, bw, bh, _) in self.buttons.items():
                    if bx <= x <= bx + bw and by <= local_y <= by + bh:
                        if btn_id == "save":
                            self.should_save = True
                        if btn_id == "clear":
                            self.current_labels = [0] * self.num_patches
                        if btn_id == "quit":
                            self.should_quit = True
                return

            idx = self.get_patch_idx(x, y)
            if idx is not None:
                self.is_drawing = True
                self.draw_mode = 1 - self.current_labels[idx]
                self.current_labels[idx] = self.draw_mode

        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            if y <= img_h:
                idx = self.get_patch_idx(x, y)
                if idx is not None:
                    self.current_labels[idx] = self.draw_mode

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False

    def render(self, img):
        h, w = img.shape[:2]
        canvas = np.zeros((h + self.footer_h, w, 3), dtype=np.uint8)
        canvas[:h, :w] = img.copy()

        grid_layer = canvas[:h, :w].copy()
        for idx, poly in enumerate(self.current_polygons):
            if not self.valid_mask_ui[idx]:
                continue

            color = (0, 0, 255) if self.current_labels[idx] == 1 else (0, 255, 0)

            cv2.fillPoly(grid_layer, [poly], color)
            cv2.polylines(
                canvas[:h, :w],
                [poly],
                isClosed=True,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        cv2.addWeighted(grid_layer, 0.4, canvas[:h, :w], 0.6, 0, canvas[:h, :w])

        footer = canvas[h:, :]
        footer[:] = (45, 45, 45)
        for btn_id, (bx, by, bw, bh, label) in self.buttons.items():
            color = (60, 160, 60) if btn_id == "save" else (100, 100, 100)
            if btn_id == "quit":
                color = (60, 60, 160)
            cv2.rectangle(footer, (bx, by), (bx + bw, by + bh), color, -1)
            cv2.putText(
                footer,
                label,
                (bx + 15, by + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        done_count = len(self.all_manual_labels)
        total_todo = done_count + len(self.todo_list)
        cv2.putText(
            canvas,
            f"Progress: {done_count}/{total_todo} | User: {USER_NAME} | Sync Mode: NPZ",
            (w - 480, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return canvas

    def run(self):
        if not self.todo_list:
            print(f"🎉 任务已全部完成！没有找到未标注且存在底层物理包的图像。")
            return

        win_name = f"Road-MoME Annotator - [{USER_NAME}]"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 850)

        for img_path in self.todo_list:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            frame_id = img_path.stem

            # 直接从 npz 中加载多边形
            success = self._load_polygons_from_npz(frame_id)
            if not success:
                continue

            self.current_labels = [0] * self.num_patches
            self.should_save = False

            h, w = img.shape[:2]
            cv2.setMouseCallback(win_name, self.on_mouse, {"h": h})

            while True:
                display = self.render(img)
                cv2.imshow(win_name, display)

                cv2.waitKey(10)

                if self.should_save:
                    self.all_manual_labels[img_path.name] = self.current_labels
                    with open(SAVE_PATH, "w", encoding="utf-8") as f:
                        json.dump(self.all_manual_labels, f)
                    print(f"✔️ Saved: {img_path.name}")
                    break

                if self.should_quit:
                    print("🚪 安全退出，进度已保存。")
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()
        print("✨ 标注流程已圆满结束。")


if __name__ == "__main__":
    ProfessionalAnnotator().run()
