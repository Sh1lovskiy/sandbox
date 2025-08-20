# vision/skeleton/normals.py
from __future__ import annotations
import numpy as np
import open3d as o3d
from utils.logger import Logger

LOG = Logger.get_logger("norm")


def _ensure_normals(pcd: o3d.geometry.PointCloud):
    if pcd.has_normals() and len(pcd.normals) == len(pcd.points):
        return
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=60)
    )


def angle_between_normals_deg(N: np.ndarray, ref_n: np.ndarray) -> np.ndarray:
    """Angle in degrees between each row of N and ref normal (0..90]."""
    ref = ref_n / (np.linalg.norm(ref_n) + 1e-12)
    dots = np.clip(np.abs(N @ ref), 0.0, 1.0)
    return np.degrees(np.arccos(dots))


def colorize_by_angle_bins(
    angles_deg: np.ndarray,
    bins=(15.0, 35.0, 55.0, 75.0, 90.0),
    palette=((1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0, 0.7, 0), (0, 0.4, 1)),
) -> np.ndarray:
    cols = np.zeros((len(angles_deg), 3), float)
    for i, a in enumerate(angles_deg):
        for bi, hi in enumerate(bins):
            if a <= hi + 1e-9:
                cols[i] = palette[min(bi, len(palette) - 1)]
                break
    return cols


def make_normals_colored_cloud(
    base_cloud: o3d.geometry.PointCloud,
    ref_normal: np.ndarray,
    bins=(15.0, 35.0, 55.0, 75.0, 90.0),
    palette=((1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0, 0.7, 0), (0, 0.4, 1)),
):
    """Copy of cloud with colors encoding angle-to-plane normal."""
    cloud = o3d.geometry.PointCloud(base_cloud)
    _ensure_normals(cloud)
    N = np.asarray(cloud.normals)
    ang = angle_between_normals_deg(N, ref_normal)
    cloud.colors = o3d.utility.Vector3dVector(
        colorize_by_angle_bins(ang, bins, palette)
    )
    stats = dict(
        count=int(np.isfinite(ang).sum()),
        mean=float(np.nanmean(ang)),
        median=float(np.nanmedian(ang)),
        std=float(np.nanstd(ang)),
        min=float(np.nanmin(ang)),
        max=float(np.nanmax(ang)),
        bins_deg=list(bins),
    )
    return cloud, ang, stats
