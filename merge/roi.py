# merge/roi.py
from __future__ import annotations

import numpy as np
import open3d as o3d

from utils.helpers import fmt_array
from utils.logger import Logger

LOG = Logger.get_logger("roi")


def make_aabb(points: np.ndarray) -> o3d.geometry.AxisAlignedBoundingBox:
    """Axis-aligned box from 4 vertices."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    return pc.get_axis_aligned_bounding_box()


def crop_to_aabb(
    pcd: o3d.geometry.PointCloud, aabb: o3d.geometry.AxisAlignedBoundingBox
) -> o3d.geometry.PointCloud:
    """Crop to ROI."""
    return pcd.crop(aabb)


def count_inliers(
    pts: np.ndarray, aabb: o3d.geometry.AxisAlignedBoundingBox
) -> int:
    """Count points inside AABB."""
    mn = np.asarray(aabb.get_min_bound())
    mx = np.asarray(aabb.get_max_bound())
    m = (
        (pts[:, 0] >= mn[0])
        & (pts[:, 1] >= mn[1])
        & (pts[:, 2] >= mn[2])
        & (pts[:, 0] <= mx[0])
        & (pts[:, 1] <= mx[1])
        & (pts[:, 2] <= mx[2])
    )
    return int(m.sum())


def log_cloud_stats(tag: str, pcd: o3d.geometry.PointCloud) -> None:
    """Compact stats for a point cloud."""
    n = len(pcd.points)
    if n == 0:
        LOG.info(f"{tag}: 0 pts")
        return
    P = np.asarray(pcd.points)
    c = P.mean(0)
    mn = P.min(0)
    mx = P.max(0)
    LOG.info(
        f"{tag}: {n} pts | center={fmt_array(c)} | min={fmt_array(mn)} | max={fmt_array(mx)}"
    )
