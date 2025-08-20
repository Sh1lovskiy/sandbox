# vision/skeleton/planes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from utils.config import HandEye, Pose, HAND_EYE
from utils.helpers import euler_deg_to_R, make_T
from utils.io import load_poses
from utils.logger import Logger

from .geometry import PlaneBasis, plane_basis_from_coeffs, to_plane_uv

LOG = Logger.get_logger("planes")


def _segment_plane_once(
    pcd: o3d.geometry.PointCloud, dist: float, ransac_n: int, iters: int
) -> Tuple[np.ndarray, np.ndarray]:
    model, inliers = pcd.segment_plane(
        distance_threshold=dist, ransac_n=ransac_n, num_iterations=iters
    )
    return np.asarray(model, float), np.asarray(inliers, int)


def segment_planes_iterative(
    pcd: o3d.geometry.PointCloud,
    max_planes: int,
    dist: float,
    ransac_n: int,
    iters: int,
    min_pts: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    planes: List[Tuple[np.ndarray, np.ndarray]] = []
    rest = pcd
    for i in range(max_planes):
        if len(rest.points) < min_pts:
            break
        m, inl = _segment_plane_once(rest, dist, ransac_n, iters)
        if inl.size < min_pts:
            break
        planes.append((m, inl))
        rest = rest.select_by_index(inl.tolist(), invert=True)
        LOG.info(f"[RANSAC] plane#{i}: {inl.size} inliers")
    return planes


def _reselect_inliers_full(
    model: np.ndarray, cloud: o3d.geometry.PointCloud, dist: float
) -> np.ndarray:
    a, b, c, d = model
    n = np.array([a, b, c], float)
    nn = np.linalg.norm(n) + 1e-12
    P = np.asarray(cloud.points)
    dists = np.abs(P @ n + d) / nn
    return np.where(dists <= dist)[0]


def _camera_centers_from_poses(poses: Dict[str, Pose]) -> List[np.ndarray]:
    if not poses:
        return []
    he = HandEye(R=HAND_EYE.R.copy(), t=HAND_EYE.t.copy(), direction="tcp_cam")
    T_tcp_cam = he.T_tcp_cam
    centers: List[np.ndarray] = []
    for k in sorted(poses.keys()):
        p = poses[k]
        R_tcp = euler_deg_to_R(p.rx, p.ry, p.rz, "XYZ")
        t_tcp = np.array([p.x, p.y, p.z], float) * 0.001
        T_base_tcp = make_T(R_tcp, t_tcp)
        T_base_cam = T_base_tcp @ T_tcp_cam
        centers.append(T_base_cam[:3, 3].copy())
    LOG.info(f"[POSES] camera centers: {len(centers)}")
    return centers


def pick_nearest_plane_to_cameras(
    planes: List[Tuple[np.ndarray, np.ndarray]],
    cloud: o3d.geometry.PointCloud,
    capture_root: Optional[Path],
    poses_json: str,
    use_capture_poses: bool,
    dist_for_reselect: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not planes:
        raise RuntimeError("No plane candidates.")
    cam_centers: List[np.ndarray] = []
    if use_capture_poses and capture_root is not None:
        poses = load_poses(capture_root / poses_json)
        cam_centers = _camera_centers_from_poses(poses)
    if not cam_centers:
        best = max(planes, key=lambda P: P[1].size)
        inl_full = _reselect_inliers_full(best[0], cloud, dist_for_reselect)
        LOG.info("[PICK] no cam centers; chose largest support.")
        return best[0], inl_full
    best_m, best_s = None, 1e9
    for i, (m, _) in enumerate(planes):
        a, b, c, d = m
        n = np.array([a, b, c])
        nn = np.linalg.norm(n) + 1e-12
        s = min(
            abs(a * x + b * y + c * z + d) / nn for (x, y, z) in cam_centers
        )
        LOG.info(f"[PICK] plane#{i} min-dist-to-cam={s:.4f} m")
        if s < best_s:
            best_s, best_m = s, m
    inl_full = _reselect_inliers_full(best_m, cloud, dist_for_reselect)
    return best_m, inl_full


def rasterize_points(
    uv: np.ndarray, res: float
) -> Tuple[np.ndarray, Tuple[float, float]]:
    umin, vmin = uv.min(0)
    umax, vmax = uv.max(0)
    W = max(1, int(np.ceil((umax - umin) / res)) + 3)
    H = max(1, int(np.ceil((vmax - vmin) / res)) + 3)
    img = np.zeros((H, W), np.uint8)
    ui = np.clip(((uv[:, 0] - umin) / res).astype(int), 0, W - 1)
    vi = np.clip(((uv[:, 1] - vmin) / res).astype(int), 0, H - 1)
    img[vi, ui] = 1
    return img, (umin, vmin)


def morph_close(img: np.ndarray, k: int) -> np.ndarray:
    try:
        import cv2
    except Exception as e:
        LOG.warning(f"[MORPH] OpenCV missing: {e}")
        return img
    k = max(1, int(k))
    k = k if (k % 2) == 1 else k + 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
