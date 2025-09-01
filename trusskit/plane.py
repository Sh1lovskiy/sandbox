"""Dominant plane detection utilities."""
from __future__ import annotations

import numpy as np
import open3d as o3d

from utils.logger import Logger
from vision.skeleton.skeleton2d import _plane_basis_from_coeffs

LOG = Logger.get_logger("tk.plane")


def find_dominant_plane(cloud: o3d.geometry.PointCloud):
    """Return plane model and local basis (u, v, n, p0)."""
    if len(cloud.points) == 0:
        raise RuntimeError("empty cloud")
    model, inliers = cloud.segment_plane(0.01, 3, 1000)
    pts = np.asarray(cloud.points)[inliers]
    u, v, n, p0 = _plane_basis_from_coeffs(np.asarray(model, float), pts)
    LOG.info(f"plane inliers={len(inliers)}")
    return np.asarray(model, float), np.asarray(inliers, int), u, v, n, p0
