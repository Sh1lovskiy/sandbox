from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import open3d as o3d

from utils.config import HandEye, Pose
from utils.helpers import euler_deg_to_R, make_T, fmt_array
from utils.logger import Logger

LOG = Logger.get_logger("transforms")


def build_T_base_tcp(pose: Pose, unit_scale: float, euler: str) -> np.ndarray:
    """BASE <- TCP from Pose with given units and Euler order."""
    R = euler_deg_to_R(pose.rx, pose.ry, pose.rz, euler)
    t = np.array([pose.x, pose.y, pose.z], dtype=float) * unit_scale
    return make_T(R, t)


def _as_handeye_with_dir(he: HandEye, d: str) -> HandEye:
    """Return a copy of HandEye with a different direction."""
    return HandEye(R=he.R.copy(), t=he.t.copy(), direction=d)


def get_T_tcp_cam(handeye: HandEye) -> np.ndarray:
    """Return TCP <- CAM 4x4 using HandEye.direction."""
    return handeye.T_tcp_cam


def build_T_base_cam(
    pose: Pose,
    unit_scale: float,
    euler: str,
    handeye: HandEye,
) -> np.ndarray:
    """BASE <- CAM via BASE<-TCP and TCP<-CAM."""
    T_base_tcp = build_T_base_tcp(pose, unit_scale, euler)
    T_tcp_cam = get_T_tcp_cam(handeye)
    return T_base_tcp @ T_tcp_cam


def _iter_modes(
    pose_unit_mode: str, euler_mode: str, he_dir_mode: str
) -> Tuple[list, list, list]:
    """Enumerate search spaces for autotune."""
    units = (
        [("meters", 1.0), ("millimeters", 0.001)]
        if pose_unit_mode == "auto"
        else [(pose_unit_mode, 1.0 if pose_unit_mode == "meters" else 0.001)]
    )
    orders = ["XYZ", "ZYX"] if euler_mode == "auto" else [euler_mode]
    hdirs = (
        ["tcp_cam", "cam_tcp_inv"] if he_dir_mode == "auto" else [he_dir_mode]
    )
    return units, orders, hdirs


def _count_inliers(
    P: np.ndarray, aabb: o3d.geometry.AxisAlignedBoundingBox
) -> int:
    """Count AABB inliers for transformed points."""
    mn = np.asarray(aabb.get_min_bound())
    mx = np.asarray(aabb.get_max_bound())
    m = (
        (P[:, 0] >= mn[0])
        & (P[:, 1] >= mn[1])
        & (P[:, 2] >= mn[2])
        & (P[:, 0] <= mx[0])
        & (P[:, 1] <= mx[1])
        & (P[:, 2] <= mx[2])
    )
    return int(m.sum())


def _sample_points(
    pcd: o3d.geometry.PointCloud, n: int = 120_000
) -> np.ndarray:
    """Random point subset as ndarray (N,3)."""
    P = np.asarray(pcd.points)
    if len(P) <= n:
        return P
    idx = np.random.choice(len(P), n, replace=False)
    return P[idx]


def autotune_extrinsics(
    pcd_cam: o3d.geometry.PointCloud,
    pose: Pose,
    aabb: o3d.geometry.AxisAlignedBoundingBox,
    pose_unit_mode: Literal["auto", "meters", "millimeters"],
    euler_mode: Literal["auto", "XYZ", "ZYX"],
    he_dir_mode: Literal["auto", "tcp_cam", "cam_tcp_inv"],
    handeye: HandEye,
) -> tuple[float, str, str]:
    """
    Search units, Euler order, and HE direction to maximize AABB inliers.
    Returns (unit_scale, euler_order, he_dir).
    """
    units, orders, hdirs = _iter_modes(pose_unit_mode, euler_mode, he_dir_mode)
    P = _sample_points(pcd_cam)
    best = None

    for uname, scale in units:
        for order in orders:
            for hdir in hdirs:
                T = build_T_base_cam(
                    pose, scale, order, _as_handeye_with_dir(handeye, hdir)
                )
                Pt = (T[:3, :3] @ P.T + T[:3, 3:4]).T
                hit = _count_inliers(Pt, aabb)
                if not best or hit > best["hit"]:
                    best = dict(
                        scale=scale, order=order, hdir=hdir, hit=hit, T=T
                    )

    LOG.info(
        f"units={'m' if best['scale'] == 1.0 else 'mm'} "
        f"order={best['order']} handeye={best['hdir']} inliers={best['hit']}"
    )
    LOG.info(
        f"R=\n{fmt_array(best['T'][:3, :3])}\n t={fmt_array(best['T'][:3, 3])}"
    )
    return float(best["scale"]), str(best["order"]), str(best["hdir"])
