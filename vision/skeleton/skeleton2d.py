# vision/skeleton/skeleton2d.py
from __future__ import annotations
from typing import List, Tuple
import math
import numpy as np
import open3d as o3d

from utils.logger import Logger
from utils.io import load_poses
from utils.config import HAND_EYE
from utils.helpers import euler_deg_to_R, make_T

LOG = Logger.get_logger("sk2d")


def _downsample(pcd: o3d.geometry.PointCloud, vox: float):
    return pcd.voxel_down_sample(vox) if vox and vox > 0 else pcd


def _segment_plane(
    pcd: o3d.geometry.PointCloud, dist: float, n: int, iters: int
) -> tuple[np.ndarray, np.ndarray]:
    model, inl = pcd.segment_plane(dist, n, iters)
    return np.array(model, float), np.asarray(inl, int)


def _iter_planes(
    pcd_ds: o3d.geometry.PointCloud,
    max_planes: int,
    dist: float,
    n: int,
    iters: int,
    min_pts: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    planes, rest = [], pcd_ds
    for _ in range(max_planes):
        if len(rest.points) < min_pts:
            break
        m, inl = _segment_plane(rest, dist, n, iters)
        if inl.size < min_pts:
            break
        planes.append((m, inl))
        rest = rest.select_by_index(inl.tolist(), invert=True)
    return planes


def _camera_centers_from_poses(
    poses: dict[str, "ucfg.Pose"],
) -> list[np.ndarray]:
    if not poses:
        return []
    T_tcp_cam = make_T(HAND_EYE.R, HAND_EYE.t)
    out = []
    for k in sorted(poses):
        p = poses[k]
        R_tcp = euler_deg_to_R(p.rx, p.ry, p.rz, "XYZ")
        t_tcp = np.array([p.x, p.y, p.z]) * 0.001
        T_base_tcp = make_T(R_tcp, t_tcp)
        T_base_cam = T_base_tcp @ T_tcp_cam
        out.append(T_base_cam[:3, 3].copy())
    return out


def _reselect_inliers_full(
    m: np.ndarray, cloud: o3d.geometry.PointCloud, d: float
):
    a, b, c, D = m
    n = np.array([a, b, c], float)
    nn = np.linalg.norm(n) + 1e-12
    P = np.asarray(cloud.points)
    dist = np.abs(P @ n + D) / nn
    return np.where(dist <= d)[0]


def _pick_plane(
    planes, cloud, cam_centers, dist_for_reselect: float
) -> tuple[np.ndarray, np.ndarray]:
    if not planes:
        raise RuntimeError("No plane candidates.")
    if not cam_centers:
        best = max(planes, key=lambda P: P[1].size)
        inl = _reselect_inliers_full(best[0], cloud, dist_for_reselect)
        LOG.info("no cameras; choose largest plane.")
        return best[0], inl
    best_m, best_s = None, 1e9
    for m, _inl in planes:
        a, b, c, D = m
        n = np.array([a, b, c], float)
        nn = np.linalg.norm(n) + 1e-12
        s = min(
            abs(a * x + b * y + c * z + D) / nn for (x, y, z) in cam_centers
        )
        if s < best_s:
            best_s, best_m = s, m
    inl = _reselect_inliers_full(best_m, cloud, dist_for_reselect)
    return best_m, inl


def detect_plane_and_mask(
    cloud: o3d.geometry.PointCloud,
    cfg,
    capture_root: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick plane by nearest-to-camera heuristic."""
    pcd_ds = _downsample(cloud, cfg.ds_vox_seg)
    N = len(pcd_ds.points)
    min_pts = max(cfg.min_plane_pts_abs, int(N * cfg.min_plane_pts_frac))

    cams = []
    if cfg.use_capture_poses:
        poses = load_poses(
            (__import__("pathlib").Path(capture_root) / cfg.poses_json)
        )
        cams = _camera_centers_from_poses(poses) if poses else []

    planes, used = None, None
    for d in cfg.dist_schedule:
        planes = _iter_planes(
            pcd_ds, cfg.max_planes, d, cfg.ransac_n, cfg.iters, min_pts
        )
        if planes:
            used = d
            break
    if not planes:
        planes = _iter_planes(
            pcd_ds,
            cfg.max_planes,
            cfg.dist_schedule[-1],
            cfg.ransac_n,
            cfg.iters * 2,
            max(200, int(min_pts * 0.5)),
        )
        used = cfg.dist_schedule[-1] if planes else None
    if not planes:
        raise RuntimeError("no plane candidates after escalation")

    model, inliers_full = _pick_plane(planes, cloud, cams, used)
    return model, inliers_full


def rasterize_points_2d(
    pts: np.ndarray, u: np.ndarray, v: np.ndarray, p0: np.ndarray, res: float
) -> tuple[np.ndarray, tuple[float, float]]:
    q = pts - p0
    UV = np.stack([q @ u, q @ v], 1)
    umin, vmin = UV.min(0)
    umax, vmax = UV.max(0)
    W = max(1, int(np.ceil((umax - umin) / res)) + 3)
    H = max(1, int(np.ceil((vmax - vmin) / res)) + 3)
    img = np.zeros((H, W), np.uint8)
    ui = np.clip(((UV[:, 0] - umin) / res).astype(int), 0, W - 1)
    vi = np.clip(((UV[:, 1] - vmin) / res).astype(int), 0, H - 1)
    img[vi, ui] = 1
    return img, (umin, vmin)


def _plane_basis_from_coeffs(m: np.ndarray, pts: np.ndarray):
    a, b, c, d = m
    n = np.array([a, b, c], float)
    n /= np.linalg.norm(n) + 1e-12
    p0 = pts.mean(0)
    ref = np.array([0, 0, 1]) if abs(n[2]) < 0.9 else np.array([1, 0, 0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return u, v, n, p0


def skeletonize_2d_img(img: np.ndarray, method: str) -> np.ndarray:
    if method == "skimage":
        try:
            from skimage.morphology import skeletonize

            return skeletonize(img.astype(bool)).astype(np.uint8)
        except Exception as e:
            LOG.warning(f"skimage skeletonize fail: {e}")
    # naive Zhang-Suen-like thinning
    sk = img.copy().astype(np.uint8)
    changed = True
    nbrs = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]
    while changed:
        changed = False
        for pass_id in (0, 1):
            to_del = []
            H, W = sk.shape
            for r in range(1, H - 1):
                for c in range(1, W - 1):
                    if sk[r, c] == 0:
                        continue
                    nb = [sk[r + dr, c + dc] for dr, dc in nbrs]
                    C = sum(
                        (nb[i] == 0 and nb[(i + 1) % 8] == 1) for i in range(8)
                    )
                    N = sum(nb)
                    if 2 <= N <= 6 and C == 1:
                        if pass_id == 0:
                            if (
                                nb[0] * nb[2] * nb[4] == 0
                                and nb[2] * nb[4] * nb[6] == 0
                            ):
                                to_del.append((r, c))
                        else:
                            if (
                                nb[0] * nb[2] * nb[6] == 0
                                and nb[0] * nb[4] * nb[6] == 0
                            ):
                                to_del.append((r, c))
            for r, c in to_del:
                sk[r, c] = 0
                changed = True
    return sk


def build_graph_from_2d_skel(sk: np.ndarray):
    H, W = sk.shape
    nbrs = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]
    deg, px = {}, np.argwhere(sk == 1)
    for r, c in px:
        deg[(r, c)] = sum(
            1
            for dr, dc in nbrs
            if 0 <= r + dr < H and 0 <= c + dc < W and sk[r + dr, c + dc]
        )
    nodes = [rc for rc, d in deg.items() if d != 2]
    node_id = {rc: i for i, rc in enumerate(nodes)}
    visited, edges, polylines = set(), [], []
    for r, c in nodes:
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W) or sk[rr, cc] == 0:
                continue
            a, b = (r, c), (rr, cc)
            if (a, b) in visited:
                continue
            path = [a, b]
            u, v = a, b
            while True:
                nxt, cnt = None, 0
                for ddr, ddc in nbrs:
                    r2, c2 = v[0] + ddr, v[1] + ddc
                    if (
                        0 <= r2 < H
                        and 0 <= c2 < W
                        and sk[r2, c2] == 1
                        and (r2, c2) != u
                    ):
                        cnt += 1
                        nxt = (r2, c2)
                if v in node_id and v != a:
                    break
                if cnt == 0:
                    break
                u, v = v, nxt
                path.append(v)
            visited.add((path[0], path[1]))
            s, e = path[0], path[-1]
            if s in node_id and e in node_id and s != e:
                edges.append((node_id[s], node_id[e]))
                polylines.append(path)
    return nodes, edges, node_id, polylines


def px_to_world_2d(
    rc_path: list[tuple[int, int]],
    u: np.ndarray,
    v: np.ndarray,
    p0: np.ndarray,
    res: float,
    uv_min: tuple[float, float],
) -> np.ndarray:
    """Pixel chain -> world xyz via plane basis and raster grid."""
    umin, vmin = uv_min
    UV = np.array(
        [
            [(c + 0.5) * res + umin, (r + 0.5) * res + vmin]
            for (r, c) in rc_path
        ],
        float,
    )
    return p0[None, :] + UV[:, 0:1] * u[None, :] + UV[:, 1:2] * v[None, :]
