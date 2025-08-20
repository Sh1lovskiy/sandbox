# vision/skeleton/regions.py
from __future__ import annotations
from typing import List, Tuple
import math
import numpy as np
import open3d as o3d
from utils.logger import Logger
from .normals import angle_between_normals_deg

LOG = Logger.get_logger("regions")


def _ensure_normals(pcd: o3d.geometry.PointCloud):
    if pcd.has_normals() and len(pcd.normals) == len(pcd.points):
        return
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=60)
    )


def _bin_masks(angles_deg: np.ndarray, edges) -> list[np.ndarray]:
    eps = 1e-6
    masks = []
    for i in range(1, len(edges)):
        prev = edges[i - 1]
        hi = edges[i]
        m = (angles_deg > prev + eps) & (angles_deg <= hi + eps)
        masks.append(m)
    return masks


def build_side_surface_clouds(
    base_cloud: o3d.geometry.PointCloud,
    ref_n: np.ndarray,
    bins=(15.0, 35.0, 55.0, 75.0, 90.0),
    palette=((1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0, 0.7, 0), (0, 0.4, 1)),
) -> Tuple[List[o3d.geometry.PointCloud], List[np.ndarray]]:
    """Return per-side-bin point clouds (skip the 0-15 bin)."""
    c = o3d.geometry.PointCloud(base_cloud)
    _ensure_normals(c)
    P = np.asarray(c.points)
    ang = angle_between_normals_deg(np.asarray(c.normals), ref_n)
    masks = _bin_masks(ang, bins)  # 4 side bins
    clouds = []
    for bi, m in enumerate(masks, start=1):
        pc = o3d.geometry.PointCloud()
        if np.any(m):
            pc.points = o3d.utility.Vector3dVector(P[m])
            pc.paint_uniform_color(palette[bi])
        clouds.append(pc)
    return clouds, masks


def _distinct_colors(n: int) -> list[tuple[float, float, float]]:
    if n <= 0:
        return []
    cols = []
    for i in range(n):
        h = (i / max(1, n)) % 1.0
        s, v = 0.65, 0.95
        hi = int(h * 6) % 6
        f = h * 6 - hi
        p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
        if hi == 0:
            r, g, b = v, t, p
        elif hi == 1:
            r, g, b = q, v, p
        elif hi == 2:
            r, g, b = p, v, t
        elif hi == 3:
            r, g, b = p, q, v
        elif hi == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        cols.append((r, g, b))
    return cols


def _ip_basis(n: np.ndarray):
    n = n / (np.linalg.norm(n) + 1e-12)
    ref = (
        np.array([0.0, 0.0, 1.0])
        if abs(n[2]) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return u, v, n


def _project_normals_ip(N: np.ndarray, n: np.ndarray):
    T = N - (N @ n)[:, None] * n
    s = np.linalg.norm(T, axis=1)
    m = s > 1e-9
    T[m] /= s[m, None]
    T[~m] = 0.0
    return T, m


def _region_grow_ip(
    P: np.ndarray,
    T: np.ndarray,
    mask: np.ndarray,
    r: float,
    cos_thr: float,
    min_pts: int,
) -> list[np.ndarray]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    Pm, Tm = P[idx], T[idx]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
    kdt = o3d.geometry.KDTreeFlann(pc)
    vis = np.zeros(idx.size, bool)
    comps = []
    for seed in range(idx.size):
        if vis[seed]:
            continue
        q, comp = [seed], [seed]
        vis[seed] = True
        while q:
            i = q.pop()
            _, nbrs, _ = kdt.search_radius_vector_3d(Pm[i], r)
            ti = Tm[i]
            for j in nbrs:
                if vis[j]:
                    continue
                if float(np.dot(ti, Tm[j])) >= cos_thr:
                    vis[j] = True
                    q.append(j)
                    comp.append(j)
        if len(comp) >= min_pts:
            comps.append(idx[np.asarray(comp, int)])
    return comps


def _boundary_points(P: np.ndarray, labels: np.ndarray, k: int, frac: float):
    if P.shape[0] == 0:
        return np.zeros((0,), bool)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    kdt = o3d.geometry.KDTreeFlann(pc)
    out = np.zeros(P.shape[0], bool)
    k = max(3, int(k))
    for i in range(P.shape[0]):
        kk = min(k, P.shape[0])
        _, idxs, _ = kdt.search_knn_vector_3d(P[i], kk)
        same = int(np.sum(labels[idxs] == labels[i]))
        if (same / float(kk)) < frac:
            out[i] = True
    return out


def build_side_surface_regions_v2(
    base_cloud: o3d.geometry.PointCloud,
    ref_n: np.ndarray,
    cfg,
) -> tuple[list[o3d.geometry.PointCloud], o3d.geometry.PointCloud]:
    """Grow components by in-plane normal similarity; split into subfaces."""
    _ensure_normals(base_cloud)
    P = np.asarray(base_cloud.points)
    N = np.asarray(base_cloud.normals)
    u, v, n = _ip_basis(ref_n)
    ang = angle_between_normals_deg(N, n)
    side_mask = ang > (cfg.side_min_deg - 1e-6)
    T, valid = _project_normals_ip(N, n)
    side_mask &= valid
    cos_thr = math.cos(math.radians(cfg.ip_normal_thr_deg))
    comps = _region_grow_ip(
        P, T, side_mask, cfg.grow_radius, cos_thr, cfg.min_region_pts
    )

    labels = np.full(P.shape[0], -1, int)
    clouds, cols = [], _distinct_colors(16)
    color_id = 0
    for ids in comps:
        theta = np.arctan2(T[ids] @ v, T[ids] @ u)
        edges = [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi + 1e-9]
        for k in range(4):
            m = (theta > edges[k]) & (theta <= edges[k + 1])
            if np.count_nonzero(m) >= max(40, int(0.05 * ids.size)):
                sids = ids[m]
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(P[sids])
                pc.paint_uniform_color(cols[color_id % len(cols)])
                clouds.append(pc)
                labels[sids] = color_id
                color_id += 1

    bnd = o3d.geometry.PointCloud()
    keep = labels >= 0
    if np.any(keep):
        Pb = P[keep]
        Lb = labels[keep]
        bmask = _boundary_points(Pb, Lb, cfg.bnd_k, cfg.bnd_frac)
        if np.any(bmask):
            bnd.points = o3d.utility.Vector3dVector(Pb[bmask])
            bnd.paint_uniform_color((0.0, 0.0, 0.0))
    return clouds, bnd
