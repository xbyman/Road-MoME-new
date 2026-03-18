import pickle
import pandas as pd


def load_trajectory_data(file_path):
    """
    读取包含位置、位姿和速度的 Pickle 文件，并转换为 DataFrame
    """
    # 1. 以二进制只读模式打开并反序列化数据
    with open(file_path, "rb") as f:
        raw_data = pickle.load(f)

    # 2. 转换为结构化的 DataFrame
    df = pd.DataFrame(raw_data)

    # 3. 时间戳格式化 (核心步骤)
    # 原始时间格式类似 '20230408031956.600'
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S.%f")
        # 将时间设为索引，方便后续基于时间窗的截取和对齐
        df.set_index("time", inplace=True)

    return df


if __name__ == "__main__":
    file_name = r"C:\Users\31078\Desktop\ROAD\data\RSRD-sparse1\2023-04-08-03-18-09-11-conti\loc_pose_vel.pkl"

    try:
        df = load_trajectory_data(file_name)

        print("✅ 数据加载与结构化成功！\n")
        print("--- 数据字段结构与类型 ---")
        df.info()

        print("\n--- 前 5 帧遥测数据预览 ---")
        # 限制显示的列宽以适应屏幕
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(df.head())

        # 可选：简单的数据完整性检查
        print("\n--- 缺失值统计 ---")
        print(df.isnull().sum())

    except FileNotFoundError:
        print(f"❌ 找不到文件：{file_name}，请检查路径。")
    except Exception as e:
        print(f"❌ 数据解析异常：{e}")
