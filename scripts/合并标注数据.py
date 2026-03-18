"""
Road-MoME 标注合并工具 (v1.1 协作优化版)
功能：
1. 自动搜索：扫描 data 目录下所有 manual_visual_gt_{User}.json。
2. 字典合并：将不同用户的标注结果汇总，冲突时以最新载入的为准。
3. 用户名净化：自动移除文件名中的副本后缀 (如 (1))，生成干净的报告。
4. 贡献度分析：输出各用户的帧数贡献及百分比占比。
"""

import json
import re
from pathlib import Path


def merge_manual_annotations():
    # 路径定义
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_path = data_dir / "manual_visual_gt.json"

    # 查找所有标注员的文件
    annotation_files = list(data_dir.glob("manual_visual_gt_*.json"))

    if not annotation_files:
        print("❌ 未在 data/ 目录下找到任何标注员的 JSON 文件。")
        print("请确保文件名格式为: manual_visual_gt_XXX.json")
        return

    merged_data = {}
    contributions = {}
    conflict_count = 0

    print(f"🚀 启动多用户标注合并程序...")
    print(f"📂 扫描目录: {data_dir}")
    print("-" * 40)

    # 按照文件名排序，确保合并顺序确定
    for file_path in sorted(annotation_files):
        try:
            # 净化用户名：移除 manual_visual_gt_ 前缀和类似 (1) 的后缀
            raw_name = file_path.stem.replace("manual_visual_gt_", "")
            user_name = re.sub(r"\(\d+\)$", "", raw_name).strip()

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 统计并合并
            count = 0
            for key, val in data.items():
                if key in merged_data:
                    conflict_count += 1
                merged_data[key] = val
                count += 1

            contributions[user_name] = contributions.get(user_name, 0) + count
            print(f"读取文件: {file_path.name:<30} | 有效帧数: {count:>5}")

        except Exception as e:
            print(f"  ⚠️ 处理文件 {file_path.name} 时出错: {e}")

    # 保存合并后的结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4)

    total_frames = len(merged_data)

    # 输出审计报告
    print("\n" + "=" * 45)
    print(f"📊 标注贡献审计报告 (Final Summary)")
    print("-" * 45)
    for user, count in contributions.items():
        percentage = (count / total_frames * 100) if total_frames > 0 else 0
        print(f"👤 标注员: {user:<15} | 贡献: {count:>5} 帧 ({percentage:>5.1f}%)")

    print("-" * 45)
    if conflict_count > 0:
        print(f"⚠️ 冲突提示: 发现 {conflict_count} 个重复标记的文件 ID，已自动去重。")

    print(f"✨ 合并成功！全量真值总计: {total_frames} 帧")
    print(f"💾 存储路径: {output_path}")
    print("=" * 45)


if __name__ == "__main__":
    merge_manual_annotations()
