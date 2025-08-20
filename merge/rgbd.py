# merge/rgbd.py
from __future__ import annotations

import numpy as np
import open3d as o3d

from utils.helpers import prefilter_depth
from utils.logger import Logger

LOG = Logger.get_logger("rgbd")


def make_rgbd(
    rgb: np.ndarray, depth_m: np.ndarray, depth_trunc: float
) -> o3d.geometry.RGBDImage:
    """Create Open3D RGBD from numpy arrays (depth in meters)."""
    color = o3d.geometry.Image(rgb.astype(np.uint8))
    depth = o3d.geometry.Image(depth_m.astype(np.float32))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1.0,
        depth_trunc=float(depth_trunc),
        convert_rgb_to_intensity=False,
    )


def rgbd_to_pcd(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intr: o3d.camera.PinholeCameraIntrinsic,
    depth_trunc: float,
) -> o3d.geometry.PointCloud:
    """Back-project RGB-D to a point cloud (metric depth)."""
    rgbd = make_rgbd(rgb, depth_m, depth_trunc)
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


def prefilter_depth_m(depth_m: np.ndarray, cfg) -> np.ndarray:
    """Apply bilateral smoothing using utils.helpers."""
    if not cfg.depth_filter.use_bilateral:
        return depth_m
    return prefilter_depth(
        depth_m,
        use_bilateral=True,
        diameter_px=cfg.depth_filter.diameter_px,
        sigma_color_m=cfg.depth_filter.sigma_color_m,
        sigma_space_px=cfg.depth_filter.sigma_space_px,
    )
