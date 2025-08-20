# vision/skeleton/nodes.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import open3d as o3d
from utils.logger import Logger

LOG = Logger.get_logger("nodes")


def merge_close_nodes_world(
    nodes_xyz: np.ndarray, edges: List[Tuple[int, int]], radius: float
):
    """Union-find merge of nodes closer than `radius`."""
    n = len(nodes_xyz)
    if n == 0:
        return nodes_xyz, edges, {i: i for i in range(n)}
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(nodes_xyz[i] - nodes_xyz[j]) < radius:
                union(i, j)

    groups, new_nodes, remap = {}, [], {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    for _, idxs in groups.items():
        new_idx = len(new_nodes)
        for i in idxs:
            remap[i] = new_idx
        new_nodes.append(np.mean(nodes_xyz[idxs], axis=0))
    new_edges = set()
    for u, v in edges:
        uu, vv = remap[u], remap[v]
        if uu != vv:
            a, b = (uu, vv) if uu < vv else (vv, uu)
            new_edges.add((a, b))
    return np.asarray(new_nodes), sorted(new_edges), remap


def _ensure_normals(
    pcd: o3d.geometry.PointCloud, radius: float, max_nn: int
) -> None:
    if pcd.has_normals() and len(pcd.normals) == len(pcd.points):
        return
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )


def refine_nodes_by_normals(
    nodes_xyz: np.ndarray,
    cloud: o3d.geometry.PointCloud,
    cfg,
) -> np.ndarray:
    """Shift node to mid-bridge using two opposite normal clusters."""
    if len(nodes_xyz) == 0:
        return nodes_xyz
    _ensure_normals(cloud, cfg.normal_radius, cfg.normal_max_nn)
    P = np.asarray(cloud.points)
    N = np.asarray(cloud.normals)
    kdt = o3d.geometry.KDTreeFlann(cloud)
    out = []
    for p in nodes_xyz:
        _, idx, _ = kdt.search_radius_vector_3d(p, cfg.search_radius)
        if len(idx) < cfg.min_side_pts * 2:
            out.append(p)
            continue
        Q, M = P[idx], N[idx]
        good = np.isfinite(M).all(1)
        Q, M = Q[good], M[good]
        if len(Q) < cfg.min_side_pts * 2:
            out.append(p)
            continue
        Mm = M - M.mean(0)
        try:
            _, _, Vt = np.linalg.svd(Mm, False)
            dirn = Vt[0]
        except Exception:
            out.append(p)
            continue
        s = M @ dirn
        plus = s >= 0
        minus = ~plus
        if plus.sum() < cfg.min_side_pts or minus.sum() < cfg.min_side_pts:
            out.append(p)
            continue
        c_plus = Q[plus].mean(0)
        c_minus = Q[minus].mean(0)
        sep = float(np.linalg.norm(c_plus - c_minus))
        low, hi = 0.4 * cfg.bridge_width_est, 1.8 * cfg.bridge_width_est
        if not (low <= sep <= hi):
            out.append(p)
            continue
        out.append(0.5 * (c_plus + c_minus))
    return np.asarray(out)
