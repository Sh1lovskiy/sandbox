# vision/skeleton/grouping.py
from __future__ import annotations
from typing import List
import math
import numpy as np
import open3d as o3d
from utils.logger import Logger

LOG = Logger.get_logger("group")


def _downsample(pcd: o3d.geometry.PointCloud, vox: float):
    return pcd.voxel_down_sample(vox) if vox and vox > 0 else pcd


def estimate_dominant_plane_normal(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Try RANSAC on downsample; fallback to PCA last axis."""
    ds = _downsample(pcd, 0.01)
    try:
        model, inl = ds.segment_plane(0.01, 3, 1500)
        n = np.array(model[:3], float)
        return n / (np.linalg.norm(n) + 1e-12)
    except Exception:
        P = np.asarray(ds.points)
        c = P.mean(0)
        A = P - c
        _, _, Vt = np.linalg.svd(A, False)
        n = Vt[-1]
        return n / (np.linalg.norm(n) + 1e-12)


def group_polylines_by_angle(
    polys: List[np.ndarray], plane_n: np.ndarray, bins=(15, 35, 55, 75)
) -> List[List[np.ndarray]]:
    """Bin polylines by angle to plane normal in degrees (0..90]."""
    n = plane_n / (np.linalg.norm(plane_n) + 1e-12)
    edges = list(bins) + [90.0]
    groups = [[] for _ in range(len(edges))]
    for P in polys:
        if len(P) < 2:
            continue
        d = P[-1] - P[0]
        nl = np.linalg.norm(d)
        if nl < 1e-9:
            continue
        d = d / nl
        ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(d, n))))))
        placed = False
        for gi, hi in enumerate(edges):
            if ang <= hi + 1e-6:
                groups[gi].append(P)
                placed = True
                break
        if not placed:
            groups[-1].append(P)
    return groups
