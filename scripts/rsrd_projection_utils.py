"""
RSRD 投影工具 (v9.0 逆向对齐版)
核心功能：
1. 像素回溯物理 (Back-projection)：将图像上的像素坐标 [u, v] 还原为物理世界中的 [X, Y]。
2. 地平线对齐：利用射线-平面交点算法，强制将 2D 网格映射到 Z = -1.04m 的地面。
"""

import numpy as np
import pickle
from pathlib import Path
import re


class RSRDProjector:
    def __init__(self, calib_dir=None):
        self.calib_cache = {}
        self.calib_dir = Path(calib_dir) if calib_dir else None
        self.z_ground = -1.04  # 离地高度

    def _get_date_from_string(self, source_name):
        match = re.search(r"\d{8}", source_name)
        return match.group(0) if match else "default"

    def _load_calib_for_date(self, date_str):
        if date_str in self.calib_cache:
            return self.calib_cache[date_str]
        file_path = (
            list(self.calib_dir.glob(f"*{date_str}*.pkl"))[0]
            if self.calib_dir
            else None
        )

        with open(file_path, "rb") as f:
            cal = pickle.load(f)
            K = cal.get("cam_intrinsic_left", cal.get("K"))
            R = cal.get("lidar2cam_left_R", cal.get("R"))
            T = np.array(cal.get("lidar2cam_left_T", cal.get("T"))).reshape(3, 1)
            self.calib_cache[date_str] = {
                "K": K,
                "R": R,
                "T": T,
                "K_inv": np.linalg.inv(K),
                "R_inv": R.T,
            }
        return self.calib_cache[date_str]

    def pixel_to_lidar_ground(self, u, v, source_name):
        """
        [关键算法] 将像素坐标回溯到物理地平线 (Z = -1.04m)
        """
        cal = self._load_calib_for_date(self._get_date_from_string(source_name))

        # 1. 构造相机坐标系下的归一化射线 [u, v, 1] -> P_cam_ray
        pixel_homo = np.array([u, v, 1.0]).reshape(3, 1)
        ray_cam = cal["K_inv"] @ pixel_homo

        # 2. 转换射线到 LiDAR 坐标系
        # P_lidar = R_inv * (P_cam - T)
        # 射线方向 d = R_inv * ray_cam
        # 射线起点 O = -R_inv * T
        ray_dir_lidar = cal["R_inv"] @ ray_cam
        ray_origin_lidar = -cal["R_inv"] @ cal["T"]

        # 3. 计算射线与平面 Z = z_ground 的交点
        # ray_origin_z + t * ray_dir_z = z_ground
        t = (self.z_ground - ray_origin_lidar[2, 0]) / (ray_dir_lidar[2, 0] + 1e-9)

        if t < 0:
            return None  # 射线指向天空

        p_intersect = ray_origin_lidar + t * ray_dir_lidar
        return p_intersect.flatten()  # [X, Y, Z]

    def lidar_to_pixel(self, pts_lidar, source_name, is_aligned=True):
        """保持向前兼容的标准投影"""
        date_str = self._get_date_from_string(source_name)
        cal = self._load_calib_for_date(date_str)
        pts = np.atleast_2d(pts_lidar).copy()
        if pts.shape[1] != 3:
            pts = pts.T
        if is_aligned:
            pts[:, 2] += self.z_ground
        pts_cam = (cal["R"] @ pts.T) + cal["T"]
        z = pts_cam[2, :]
        pts_2d = cal["K"] @ pts_cam
        u = pts_2d[0, :] / (z + 1e-6)
        v = pts_2d[1, :] / (z + 1e-6)
        res = np.stack([u, v], axis=1)
        res[z < 0.5] = np.nan
        return res
