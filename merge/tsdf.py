from __future__ import annotations

import numpy as np
import open3d as o3d

from utils.helpers import suppress_o3d_info
from utils.logger import Logger

LOG = Logger.get_logger("tsdf")


def build_tsdf(
    voxel: float,
    trunc: float,
    color_type,
) -> o3d.pipelines.integration.ScalableTSDFVolume:
    """Construct a scalable TSDF volume with given params."""
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=trunc,
        color_type=color_type,
    )
    LOG.info(f"voxel={voxel:.4f} trunc={trunc:.4f}")
    return vol


def integrate_frame(
    vol: o3d.pipelines.integration.ScalableTSDFVolume,
    intr: o3d.camera.PinholeCameraIntrinsic,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    T_base_cam: np.ndarray,
) -> None:
    """Integrate a single RGB-D frame; expects depth in meters."""
    rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    dep_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        dep_o3d,
        depth_scale=1.0,
        depth_trunc=10.0,  # trunc by pipeline
        convert_rgb_to_intensity=False,
    )
    extr = np.linalg.inv(T_base_cam)  # Open3D expects CAM <- BASE
    with suppress_o3d_info():
        vol.integrate(rgbd, intr, extr)


def extract_cloud(
    vol: o3d.pipelines.integration.ScalableTSDFVolume,
) -> o3d.geometry.PointCloud:
    """Extract a point cloud from TSDF."""
    with suppress_o3d_info():
        pcd = vol.extract_point_cloud()
    LOG.info(f"extracted {len(pcd.points)} points before ROI")
    return pcd
