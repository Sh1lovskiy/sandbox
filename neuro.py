# new.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple, List, Optional, Dict, Iterable, NamedTuple, Any

import os
import io
import time
import math
import contextlib
import concurrent.futures
import numpy as np
import open3d as o3d

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from itertools import product
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
from collections import deque

from tqdm import tqdm

from utils.logger import Logger

LOG = Logger.get_logger("new")

# ---------------- GPU backend (for neural nets) ----------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    _TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    TORCH_AVAILABLE = False
    _TORCH_DEVICE = "cpu"


# ===================== logging control =====================


def _make_executor(n_jobs: int, ctx):
    """Process pool with worker recycling (avoids native leaks/crashes)."""
    return ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=ctx,
        initializer=_worker_init,
        max_tasks_per_child=16,
    )


def _set_quiet_logs() -> None:
    """Silence logs during search."""
    try:
        LOG.level("WARNING")
        LOG.disable("")
    except Exception:
        pass
    try:
        import graph

        if hasattr(graph, "LOG"):
            graph.LOG.level("WARNING")
            graph.LOG.info = lambda *a, **k: None
            graph.LOG.debug = lambda *a, **k: None
    except Exception:
        pass


def _set_info_logs() -> None:
    """Re-enable logs for top-K stage."""
    try:
        LOG.level("INFO")
    except Exception:
        pass
    try:
        import graph

        if hasattr(graph, "LOG"):
            graph.LOG.level("INFO")
    except Exception:
        pass


def _worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    _set_quiet_logs()


# ===================== small helpers (viz/graph) =====================


def _distinct_colors(n: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 3), np.float64)
    h = np.linspace(0.0, 1.0, n, endpoint=False)
    s = np.full(n, 0.75)
    v = np.full(n, 0.95)
    import colorsys

    return np.asarray(
        [colorsys.hsv_to_rgb(hi, si, vi) for hi, si, vi in zip(h, s, v)],
        dtype=np.float64,
    )


def _ensure_normals_inplace(
    pcd: o3d.geometry.PointCloud, radius: float
) -> None:
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=32)
        )
        pcd.orient_normals_consistent_tangent_plane(50)


def _adj_from_edges(nv: int, E: np.ndarray) -> List[List[int]]:
    adj = [[] for _ in range(nv)]
    for a, b in E:
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _compress_deg2_lines(adj: List[List[int]]) -> List[List[int]]:
    deg = [len(v) for v in adj]

    def is_j(u: int) -> bool:
        return deg[u] != 2

    seen = set()
    out: List[List[int]] = []
    for u in range(len(adj)):
        if not is_j(u):
            continue
        for v in adj[u]:
            if (u, v) in seen or (v, u) in seen:
                continue
            chain = [u, v]
            a, b = u, v
            while deg[b] == 2:
                nxt = [w for w in adj[b] if w != a]
                if not nxt:
                    break
                a, b = b, nxt[0]
                chain.append(b)
            for x, y in zip(chain[:-1], chain[1:]):
                seen.add((x, y))
            out.append(chain)
    return out


def _to_lines_and_balls(
    nodes: np.ndarray,
    E: np.ndarray,
    node_r: float,
    edge_colors: Optional[np.ndarray],
) -> tuple[o3d.geometry.LineSet, o3d.geometry.TriangleMesh]:
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(nodes),
        o3d.utility.Vector2iVector(E if E.size else np.empty((0, 2), np.int32)),
    )
    if edge_colors is not None and len(edge_colors) == len(ls.lines):
        ls.colors = o3d.utility.Vector3dVector(edge_colors)

    balls = o3d.geometry.TriangleMesh()
    if len(nodes):
        deg = np.zeros(len(nodes), np.int32)
        for a, b in E:
            deg[a] += 1
            deg[b] += 1
        pick = np.where(deg != 2)[0]
        r = max(1e-4, float(node_r))
        for i in pick:
            s = o3d.geometry.TriangleMesh.create_sphere(r)
            s.translate(nodes[i])
            balls += s
        balls.merge_close_vertices(1e-6)
        balls.compute_vertex_normals()
        balls.paint_uniform_color((0.9, 0.2, 0.1))
    return ls, balls


def _refine_junctions_inplace(
    S: np.ndarray, E: np.ndarray, P: np.ndarray, radius: float
) -> None:
    if len(S) == 0 or len(E) == 0:
        return
    adj = _adj_from_edges(len(S), E)
    TP = cKDTree(P)
    for u, nbrs in enumerate(adj):
        if len(nbrs) < 3:
            continue
        idx = TP.query_ball_point(S[u], radius)
        if idx:
            S[u] = np.median(P[idx], axis=0)


def _mst_edges(nodes: np.ndarray, k: int = 8) -> np.ndarray:
    if len(nodes) < 2:
        return np.empty((0, 2), np.int32)
    T = cKDTree(nodes)
    rows, cols, data = [], [], []
    for i, p in enumerate(nodes):
        d, j = T.query(p, k=min(k + 1, len(nodes)))
        for dj, jj in zip(d[1:], j[1:]):
            rows.append(i)
            cols.append(int(jj))
            data.append(float(dj))
    G = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    MST = minimum_spanning_tree(G).tocoo()
    return np.unique(
        np.sort(np.stack([MST.row, MST.col], axis=1), axis=1), axis=0
    ).astype(np.int32)


# ===================== params =====================


@dataclass
class EMParams:
    max_iter: int = 10
    tol: float = 1e-4
    epsilon: float = 0.07
    sigma_init_scale: float = 0.2
    pca_radius_mult: float = 1.0
    alpha_pca: float = 1.0
    lap_lambda: float = 0.5
    lap_k: int = 1
    neighbor_sigma_mult: float = 0.8
    node_merge_r: float = 0.015
    node_vis_r: float = 0.006


@dataclass
class CtrParams:
    iters: int = 14
    k: int = 9
    step: float = 0.8
    anisotropy: float = 0.6
    node_merge_r: float = 0.019
    node_vis_r: float = 0.006


# ===================== math utils =====================


def _local_pca_mu(
    P: np.ndarray, tree: o3d.geometry.KDTreeFlann, radius: float
) -> np.ndarray:
    n = len(P)
    mu = np.zeros(n, np.float64)
    for i, p in enumerate(P):
        try:
            _, idx, _ = tree.search_radius_vector_3d(p, radius)
        except Exception:
            idx = [i]
        if len(idx) < 3:
            mu[i] = 0.0
            continue
        X = P[idx] - P[idx].mean(axis=0)
        C = (X.T @ X) / float(len(idx))
        w, _ = np.linalg.eigh(C)
        w = np.maximum(w, 1e-12)
        mu[i] = float(w[0] / np.sum(w))
    return mu


def _graph_laplacian(P: np.ndarray, k: int) -> np.ndarray:
    n = len(P)
    if n == 0:
        return np.empty((0, 0), np.float64)
    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    )
    W = np.zeros((n, n), np.float64)
    for i, p in enumerate(P):
        kk = min(k + 1, n)
        try:
            _, idx, _ = tree.search_knn_vector_3d(p, kk)
        except Exception:
            idx = [i]
        for j in idx:
            if j == i:
                continue
            d = np.linalg.norm(P[j] - p) + 1e-12
            w = 1.0 / d
            if w > W[i, j]:
                W[i, j] = w
                W[j, i] = w
    D = np.diag(np.sum(W, axis=1))
    return D - W


def _sigma0_from_scale(P: np.ndarray, S: np.ndarray, scale: float) -> float:
    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    )
    d = []
    for s in S:
        try:
            _, idx, _ = tree.search_knn_vector_3d(s, 1)
            d.append(np.linalg.norm(P[idx[0]] - s))
        except Exception:
            continue
    return float((np.median(d) if d else 0.01) * scale)


# ===================== init nodes =====================


def _build_initial_nodes(pcd, surf) -> tuple[np.ndarray, np.ndarray]:
    from graph import (
        estimate_local_radii,
        radius_clustering,
        adaptive_graph_3d,
        snap_nodes_to_mesh,
        merge_nodes_radius,
    )

    P = np.asarray(pcd.points, np.float64)
    radii = estimate_local_radii(P, k=8, scale=1.2)
    C, _ = radius_clustering(P, radii, strength=2.0)
    nodes, E0 = adaptive_graph_3d(C, k=10, m=2.2, angle_deg=70.0)
    nodes = snap_nodes_to_mesh(nodes, surf, k=9)
    nodes, E0 = merge_nodes_radius(nodes, E0, radius=0.012)
    return nodes, E0.astype(np.int32, copy=False)


def _track_edges(
    prev_nodes: np.ndarray, new_nodes: np.ndarray, E0: np.ndarray
) -> np.ndarray:
    """Map each old vertex -> nearest new vertex; rebuild edges accordingly."""
    if len(prev_nodes) == 0 or len(new_nodes) == 0 or len(E0) == 0:
        return np.empty((0, 2), np.int32)
    T = cKDTree(new_nodes)
    _, idx = T.query(prev_nodes, k=1)
    E = np.stack([idx[E0[:, 0]], idx[E0[:, 1]]], axis=1).astype(np.int32)
    E = E[E[:, 0] != E[:, 1]]  # drop zero-length
    E = np.unique(np.sort(E, axis=1), axis=0)
    return E


# ===================== EM (sparse) =====================


def _em_refine(
    pcd: o3d.geometry.PointCloud,
    nodes0: np.ndarray,
    params: EMParams,
    voxel_size: float,
) -> np.ndarray:
    P = np.asarray(pcd.points, np.float64)
    n, m = len(P), len(nodes0)
    if n == 0 or m == 0:
        return nodes0.copy()

    S = nodes0.copy()
    treeP_o3d = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    )
    pca_r = params.pca_radius_mult * voxel_size
    muP = _local_pca_mu(P, treeP_o3d, pca_r)

    sigma2 = _sigma0_from_scale(P, S, params.sigma_init_scale) ** 2
    eps = float(params.epsilon)
    L = _graph_laplacian(S, k=params.lap_k)

    TP = cKDTree(P)
    last_obj = np.inf
    LOG.info(
        f"[EM] start: n={n} m={m} sigma0={np.sqrt(sigma2):.5f} eps={eps:.3f} pca_r={pca_r:.5f}"
    )

    for it in range(params.max_iter):
        treeS_o3d = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(S))
        )
        muS = _local_pca_mu(S, treeS_o3d, pca_r)

        two_sigma2 = 2.0 * sigma2
        norm_const = (2.0 * np.pi * sigma2) ** 1.5
        r_query = max(
            3.0 * voxel_size, params.neighbor_sigma_mult * np.sqrt(sigma2)
        )

        rows, cols, data = [], [], []
        for i, si in enumerate(S):
            idxP = TP.query_ball_point(si, r_query)
            if not idxP:
                continue
            diff = P[idxP] - si
            d2 = np.einsum("ij,ij->i", diff, diff)
            a = np.exp(-params.alpha_pca * np.abs(muS[i] - muP[idxP]))
            vij = a * np.exp(-d2 / two_sigma2) / norm_const
            rows.extend([i] * len(idxP))
            cols.extend(idxP)
            data.extend(vij.tolist())

        R_raw = coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()
        v_sum = np.asarray(R_raw.sum(axis=0)).ravel()
        den = (eps / n) + (1.0 - eps) * v_sum
        den[den <= 1e-300] = 1e-300
        R = R_raw.multiply((1.0 - eps)).multiply(1.0 / den)

        w = np.asarray(R.sum(axis=1)).ravel()
        RPx = R @ P[:, 0]
        RPy = R @ P[:, 1]
        RPz = R @ P[:, 2]
        A = np.diag(w + 1e-12) + params.lap_lambda * L
        Sx = np.linalg.solve(A, RPx)
        Sy = np.linalg.solve(A, RPy)
        Sz = np.linalg.solve(A, RPz)
        S = np.stack([Sx, Sy, Sz], axis=1)

        p2 = np.einsum("ij,ij->i", P, P)
        s2 = np.einsum("ij,ij->i", S, S)
        sumR = w
        RP = np.column_stack([RPx, RPy, RPz])
        term1 = float(p2 @ np.asarray(R.sum(axis=0)).ravel())
        term2 = 2.0 * float(np.einsum("ij,ij->", S, RP))
        term3 = float(s2 @ sumR)
        num_sigma = term1 - term2 + term3
        den_sigma = 3.0 * float(sumR.sum())
        sigma2_new = max(1e-10, num_sigma / max(1e-12, den_sigma))

        obj = -float(sumR.sum())
        d_obj = abs(obj - last_obj) / (abs(last_obj) + 1e-9)
        LOG.debug(
            f"[EM] it={it+1} sigma={np.sqrt(sigma2_new):.6f} d_obj={d_obj:.3e}"
        )
        sigma2, last_obj = sigma2_new, obj
        if d_obj < params.tol:
            LOG.info(f"[EM] converged at it={it+1} d_obj={d_obj:.3e}")
            break

        L = _graph_laplacian(S, k=params.lap_k)
    return S


def _refine_junction_nodes(
    nodes: np.ndarray,
    edges: np.ndarray,
    cloud: o3d.geometry.PointCloud,
    merge_r: float = 0.01,
) -> np.ndarray:
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from([tuple(e) for e in edges])
    deg = dict(G.degree)
    P = np.asarray(cloud.points, np.float64)
    N = np.asarray(cloud.normals, np.float64) if cloud.has_normals() else None

    refined = nodes.copy()
    for i, p in enumerate(nodes):
        if deg.get(i, 0) < 3:
            continue
        tree = o3d.geometry.KDTreeFlann(cloud)
        _, idx, _ = tree.search_radius_vector_3d(p, merge_r)
        if len(idx) > 0:
            pts = P[idx]
            if N is not None:
                nors = N[idx]
                cos_thr = np.cos(np.deg2rad(15))
                mask = (nors @ nors.mean(0)) > cos_thr
                refined[i] = pts[mask].mean(0) if mask.any() else pts.mean(0)
            else:
                refined[i] = pts.mean(0)
    return refined


# ===================== pipelines =====================


def skeletonize_em(
    raw: o3d.geometry.PointCloud,
    *,
    voxel_size: float = 0.004,
    params: EMParams | None = None,
) -> Tuple[
    o3d.geometry.LineSet, o3d.geometry.TriangleMesh, o3d.geometry.PointCloud
]:
    from graph import preprocess_cloud, reconstruct_mesh, merge_nodes_radius

    cfg = params or EMParams()
    LOG.info("[Skel-EM] start")
    pcd = preprocess_cloud(raw)
    _ensure_normals_inplace(pcd, radius=voxel_size * 3.0)

    surf = reconstruct_mesh(pcd)
    surf.paint_uniform_color((0.82, 0.82, 0.82))
    nodes0, E0 = _build_initial_nodes(pcd, surf)
    if len(nodes0) == 0:
        LOG.warning("[Skel-EM] no initial nodes")
        return o3d.geometry.LineSet(), o3d.geometry.TriangleMesh(), pcd

    nodes = _em_refine(pcd, nodes0, cfg, voxel_size=voxel_size)
    E = _track_edges(nodes0, nodes, E0)
    nodes_m, E_m = merge_nodes_radius(nodes, E, radius=cfg.node_merge_r)
    _refine_junctions_inplace(
        nodes_m, E_m, np.asarray(pcd.points), radius=3.0 * voxel_size
    )
    nodes_ref = _refine_junction_nodes(nodes_m, E_m, pcd)
    edge_cols = _distinct_colors(max(1, len(E_m)))
    lines, balls = _to_lines_and_balls(
        nodes_ref, E_m, cfg.node_vis_r, edge_cols
    )
    LOG.info(f"[Skel-EM] done: nodes={len(nodes_ref)} edges={len(E_m)}")
    return lines, balls, pcd


def _taubin_smooth_polyline(
    V: np.ndarray, E: np.ndarray, it=8, lam=0.5, mu=-0.53
) -> np.ndarray:
    """Edge-preserving smoothing: only deg==2 vertices move."""
    if len(E) == 0:
        return V
    V = V.copy()
    adj = [[] for _ in range(len(V))]
    for a, b in E:
        adj[a].append(b)
        adj[b].append(a)
    movable = np.array([len(nbr) == 2 for nbr in adj], bool)

    def laplacian(X):
        L = np.zeros_like(X)
        for i, nbr in enumerate(adj):
            if not movable[i] or not nbr:
                continue
            m = np.mean(X[nbr], axis=0) - X[i]
            L[i] = m
        return L

    for _ in range(it):
        V += lam * laplacian(V)
        V += mu * laplacian(V)
    return V


def _contract_points(
    P: np.ndarray, iters: int, k: int, step: float, anisotropy: float
) -> np.ndarray:
    X = P.copy()
    for _ in range(iters):
        T = cKDTree(X)
        new = np.empty_like(X)
        for i, x in enumerate(X):
            d, j = T.query(x, k=min(k + 1, len(X)))
            jj = j[1:] if len(j) > 1 else j
            neigh = X[jj] if len(jj) else X[i : i + 1]
            mu = neigh.mean(axis=0)
            C = np.cov((neigh - mu).T) + 1e-9 * np.eye(3)
            w, V = np.linalg.eigh(C)
            u = V[:, np.argmin(w)]
            proj = mu + np.dot(x - mu, u) * u
            target = (1.0 - anisotropy) * mu + anisotropy * proj
            new[i] = x + step * (target - x)
        X = new
    return X


def skeletonize_contraction(
    raw: o3d.geometry.PointCloud,
    *,
    voxel_size: float = 0.004,
    params: CtrParams | None = None,
) -> Tuple[
    o3d.geometry.LineSet, o3d.geometry.TriangleMesh, o3d.geometry.PointCloud
]:
    from graph import preprocess_cloud, merge_nodes_radius

    cfg = params or CtrParams()
    pcd = preprocess_cloud(raw)
    _ensure_normals_inplace(pcd, radius=voxel_size * 3.0)

    P = np.asarray(pcd.points, np.float64)
    if len(P) == 0:
        return o3d.geometry.LineSet(), o3d.geometry.TriangleMesh(), pcd

    X = _contract_points(P, cfg.iters, cfg.k, cfg.step, cfg.anisotropy)
    T = cKDTree(X)
    used = np.zeros(len(X), bool)
    centers = []
    rad = 1.6 * voxel_size
    for i in range(len(X)):
        if used[i]:
            continue
        idx = T.query_ball_point(X[i], rad)
        used[idx] = True
        centers.append(X[idx].mean(axis=0))
    nodes = np.asarray(centers, np.float64)

    E = _mst_edges(nodes, k=8)
    nodes_m, E_m = merge_nodes_radius(nodes, E, radius=cfg.node_merge_r)
    _refine_junctions_inplace(nodes_m, E_m, P, radius=3.0 * voxel_size)

    edge_cols = _distinct_colors(max(1, len(E_m)))
    lines, balls = _to_lines_and_balls(nodes_m, E_m, cfg.node_vis_r, edge_cols)
    LOG.info(f"[Skel-CTR] done: nodes={len(nodes_m)} edges={len(E_m)}")
    return lines, balls, pcd


# ===================== metrics & objective =====================

try:
    from metrics import report_all_metrics, MetricConfig

    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False

    class MetricConfig:
        pass


def _safe_report_metrics(
    cloud: o3d.geometry.PointCloud,
    em_lines: o3d.geometry.LineSet,
    ct_lines: o3d.geometry.LineSet,
    ct_pts: o3d.geometry.PointCloud,
) -> dict:
    if not _HAS_METRICS:
        return {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = report_all_metrics(
            cloud, em_lines, ct_lines, ct_pts, MetricConfig()
        )
    return out


def _objective_from_metrics(m: dict) -> float:
    """Lower is better."""
    topo = m.get("topology_contraction", {})
    bn = float(topo.get("bottleneck", np.nan))
    w2 = float(topo.get("wasserstein", np.nan))
    emS = float(m.get("smooth_EM", {}).get("S", np.nan))
    ctS = float(m.get("smooth_Contraction", {}).get("S", np.nan))
    emB = float(m.get("bounded_EM", {}).get("fraction", np.nan))
    ctB = float(m.get("bounded_Contraction", {}).get("fraction", np.nan))
    emC = float(m.get("centered_EM", {}).get("fraction", np.nan))
    ctC = float(m.get("centered_Contraction", {}).get("fraction", np.nan))

    def nz(x: float, default: float) -> float:
        return x if x == x else default

    bn = nz(bn, 1.0)
    w2 = nz(w2, 1.0)
    emS = nz(emS, 0.0)
    ctS = nz(ctS, 0.0)
    emB = nz(emB, 0.0)
    ctB = nz(ctB, 0.0)
    emC = nz(emC, 0.0)
    ctC = nz(ctC, 0.0)

    loss = (
        1.0 * bn
        + 0.7 * w2
        + 0.30 * (1.0 - min(emS, 1.0))
        + 0.45 * (1.0 - min(ctS, 1.0))  # give CTR-smoothness higher weight
        + 0.15 * (1.0 - 0.5 * (emB + ctB))
        + 0.15 * (1.0 - 0.5 * (emC + ctC))
    )
    return float(loss)


class TrialResult(NamedTuple):
    loss: float
    metrics: dict
    em_params: EMParams
    ct_params: CtrParams
    voxel: float


def _metrics_postfix(m: dict) -> dict:
    topo = m.get("topology_contraction", {})
    emS = float(m.get("smooth_EM", {}).get("S", np.nan))
    ctS = float(m.get("smooth_Contraction", {}).get("S", np.nan))
    emB = float(m.get("bounded_EM", {}).get("fraction", np.nan))
    ctB = float(m.get("bounded_Contraction", {}).get("fraction", np.nan))
    bnd_avg = 0.5 * (
        (emB if emB == emB else 0.0) + (ctB if ctB == ctB else 0.0)
    )
    return {
        "BN": f"{float(topo.get('bottleneck', np.nan)):.4f}" if topo else "nan",
        "W2": (
            f"{float(topo.get('wasserstein', np.nan)):.4f}" if topo else "nan"
        ),
        "EM_S": f"{emS:.3f}" if emS == emS else "nan",
        "CT_S": f"{ctS:.3f}" if ctS == ctS else "nan",
        "Bnd": f"{bnd_avg:.3f}",
    }


def _expand(values_or_single) -> Iterable:
    return (
        values_or_single
        if isinstance(values_or_single, (list, tuple))
        else [values_or_single]
    )


# ===================== worker eval =====================


def _points_to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    )


def _eval_trial_worker(
    points: np.ndarray, em: EMParams, ct: CtrParams, voxel: float
) -> tuple:
    _worker_init()
    cloud = _points_to_pcd(points)
    em_lines, _, _ = skeletonize_em(cloud, voxel_size=voxel, params=em)
    ct_lines, _, ct_pts = skeletonize_contraction(
        cloud, voxel_size=voxel, params=ct
    )
    metrics = _safe_report_metrics(cloud, em_lines, ct_lines, ct_pts)
    loss = _objective_from_metrics(metrics)
    return (loss, metrics, em, ct, float(voxel))


# ===================== neural surrogate optimizer (GPU nets) =====================

# --- Search space definition ---
SPACE: List[Dict[str, Any]] = [
    {
        "name": "voxel",
        "type": "float",
        "low": 0.0035,
        "high": 0.0060,
        "is_int": False,
    },
    {
        "name": "em_max_iter",
        "type": "int",
        "low": 8,
        "high": 14,
        "is_int": True,
    },
    {
        "name": "em_eps",
        "type": "float",
        "low": 0.02,
        "high": 0.08,
        "is_int": False,
    },
    {
        "name": "em_sigma_scale",
        "type": "float",
        "low": 0.15,
        "high": 0.45,
        "is_int": False,
    },
    {
        "name": "em_pca_mult",
        "type": "float",
        "low": 0.6,
        "high": 1.5,
        "is_int": False,
    },
    {
        "name": "em_lap_lambda",
        "type": "float",
        "low": 0.2,
        "high": 0.8,
        "is_int": False,
    },
    {"name": "em_lap_k", "type": "int", "low": 1, "high": 4, "is_int": True},
    {
        "name": "em_neighbor_sigma",
        "type": "float",
        "low": 0.6,
        "high": 1.4,
        "is_int": False,
    },
    {
        "name": "em_merge_r",
        "type": "float",
        "low": 0.010,
        "high": 0.022,
        "is_int": False,
    },
    {"name": "ct_iters", "low": 8, "high": 24, "is_int": True},
    {"name": "ct_k", "low": 7, "high": 17, "is_int": True},
    {"name": "ct_step", "low": 0.65, "high": 0.92, "is_int": False},
    {
        "name": "ct_aniso",
        "type": "float",
        "low": 0.5,
        "high": 0.9,
        "is_int": False,
    },
    {
        "name": "ct_merge_r",
        "type": "float",
        "low": 0.015,
        "high": 0.023,
        "is_int": False,
    },
]


def _encode_sample(sample: Dict[str, float]) -> np.ndarray:
    xs = []
    for spec in SPACE:
        v = float(sample[spec["name"]])
        lo, hi = spec["low"], spec["high"]
        x = (v - lo) / (hi - lo + 1e-12)
        x = np.clip(x, 0.0, 1.0)
        xs.append(x)
    return np.asarray(xs, np.float64)


def _decode_vector(x: np.ndarray) -> Dict[str, float]:
    out = {}
    for i, spec in enumerate(SPACE):
        lo, hi = spec["low"], spec["high"]
        v = lo + float(np.clip(x[i], 0.0, 1.0)) * (hi - lo)
        if spec["is_int"]:
            v = int(round(v))
        out[spec["name"]] = v
    return out


def _vec_to_params(v: Dict[str, float]) -> tuple[float, EMParams, CtrParams]:
    em = EMParams(
        max_iter=int(v["em_max_iter"]),
        epsilon=float(v["em_eps"]),
        sigma_init_scale=float(v["em_sigma_scale"]),
        pca_radius_mult=float(v["em_pca_mult"]),
        lap_lambda=float(v["em_lap_lambda"]),
        lap_k=int(v["em_lap_k"]),
        neighbor_sigma_mult=float(v["em_neighbor_sigma"]),
        node_merge_r=float(v["em_merge_r"]),
    )
    ct = CtrParams(
        iters=int(v["ct_iters"]),
        k=int(v["ct_k"]),
        step=float(v["ct_step"]),
        anisotropy=float(v["ct_aniso"]),
        node_merge_r=float(v["ct_merge_r"]),
    )
    vox = float(v["voxel"])
    return vox, em, ct


def _random_sample(rng: np.random.Generator) -> Dict[str, float]:
    s = {}
    for spec in SPACE:
        lo, hi = spec["low"], spec["high"]
        if spec["is_int"]:
            s[spec["name"]] = int(rng.integers(lo, hi + 1))
        else:
            s[spec["name"]] = float(rng.uniform(lo, hi))
    return s


# --- GPU MLP (torch) + CPU fallback ---


if TORCH_AVAILABLE:

    class TorchMLP(nn.Module):
        """Two-layer MLP (ReLU) optimized with Adam. Runs on CUDA if available."""

        def __init__(self, dim: int, hidden: int = 64):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, 1)

            # Kaiming-like init
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
            nn.init.zeros_(self.fc1.bias)
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = F.relu(self.fc1(x))
            y = self.fc2(h)
            return y

        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 220,
            batch: int = 64,
            lr: float = 8e-3,
            weight_decay: float = 1e-6,
            device: str = _TORCH_DEVICE,
        ):
            self.to(device)
            Xt = torch.from_numpy(X.astype(np.float32)).to(device)
            yt = torch.from_numpy(y.astype(np.float32).reshape(-1, 1)).to(
                device
            )
            opt = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
            n = Xt.shape[0]
            for ep in range(epochs):
                perm = torch.randperm(n, device=device)
                for s in range(0, n, batch):
                    idx = perm[s : s + batch]
                    xb = Xt[idx]
                    yb = yt[idx]
                    pred = self(xb)
                    loss = F.mse_loss(pred, yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            return self

        @torch.no_grad()
        def predict(
            self, X: np.ndarray, device: str = _TORCH_DEVICE
        ) -> np.ndarray:
            self.to(device)
            Xt = torch.from_numpy(X.astype(np.float32)).to(device)
            y = self(Xt).squeeze(1)
            return y.detach().cpu().numpy()

else:

    class TorchMLP:
        """CPU fallback (NumPy). API-compatible subset."""

        def __init__(self, dim: int, hidden: int = 64):
            self.rng = np.random.default_rng(0)
            k1 = math.sqrt(2.0 / dim)
            k2 = math.sqrt(2.0 / hidden)
            self.W1 = self.rng.normal(0, k1, (dim, hidden))
            self.b1 = np.zeros(hidden)
            self.W2 = self.rng.normal(0, k2, (hidden, 1))
            self.b2 = np.zeros(1)
            self.lr = 8e-3

        @staticmethod
        def _relu(z):
            return np.maximum(z, 0.0)

        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 220,
            batch: int = 64,
            lr: float = 8e-3,
            weight_decay: float = 1e-6,
            device: str = "cpu",
        ):
            n, d = X.shape
            y = y.reshape(-1, 1)
            self.lr = lr
            for _ in range(epochs):
                idx = np.arange(n)
                self.rng.shuffle(idx)
                for s in range(0, n, batch):
                    b = idx[s : s + batch]
                    xb = X[b]
                    yb = y[b]
                    H = self._relu(xb @ self.W1 + self.b1)
                    pred = H @ self.W2 + self.b2
                    err = pred - yb
                    gW2 = H.T @ (2.0 * err / len(b)) + weight_decay * self.W2
                    gb2 = (2.0 * err / len(b)).sum(0)
                    dH = (2.0 * err / len(b)) @ self.W2.T
                    dH[H <= 0] = 0.0
                    gW1 = xb.T @ dH + weight_decay * self.W1
                    gb1 = dH.sum(0)
                    self.W2 -= lr * gW2
                    self.b2 -= lr * gb2
                    self.W1 -= lr * gW1
                    self.b1 -= lr * gb1
            return self

        def predict(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
            H = self._relu(X @ self.W1 + self.b1)
            Y = H @ self.W2 + self.b2
            return Y.ravel()


def _train_ensemble(
    X: np.ndarray, y: np.ndarray, ensemble: int, device: str
) -> List[TorchMLP]:
    """Bootstrap-trained ensemble of MLPs on GPU (fallback to CPU)."""
    models: List[TorchMLP] = []
    n = len(X)
    rng = np.random.default_rng(2025)
    for m in range(ensemble):
        idx = rng.integers(0, n, n) if n > 1 else np.zeros(n, int)
        Xm, ym = X[idx], y[idx]
        mdl = TorchMLP(dim=X.shape[1], hidden=64)
        mdl.fit(
            Xm,
            ym,
            epochs=220,
            batch=min(64, len(Xm)),
            lr=8e-3,
            weight_decay=1e-6,
            device=device,
        )
        models.append(mdl)
    return models


def _ensemble_predict(
    models: List[TorchMLP], X: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) of predictions for X. Uses GPU when available."""
    if TORCH_AVAILABLE:
        with torch.no_grad():
            Xt = torch.from_numpy(X.astype(np.float32))
            mu_all = []
            # chunk to avoid OOM
            chunk = 32768
            for s in range(0, len(X), chunk):
                xs = Xt[s : s + chunk].to(device)
                preds = []
                for mdl in models:
                    preds.append(mdl(xs).squeeze(1))
                P = torch.stack(preds, dim=0)  # [M, B]
                mu_all.append((P.mean(0), P.std(0)))
            mu = torch.cat([m for m, _ in mu_all], dim=0).cpu().numpy()
            sd = torch.cat([s for _, s in mu_all], dim=0).cpu().numpy()
            return mu, sd
    else:
        preds = np.stack([mdl.predict(X) for mdl in models], axis=0)  # [M,N]
        return preds.mean(0), preds.std(0)


def _propose_batch(
    models: List[TorchMLP],
    rng: np.random.Generator,
    batch_size: int,
    pool_size: int = 8192,
    kappa: float = 0.20,
    already: set | None = None,
    device: str = _TORCH_DEVICE,
) -> List[Dict[str, float]]:
    """GPU-scored pool: sample candidates, score by mean - kappa*std, pick best."""
    pool = [_random_sample(rng) for _ in range(pool_size)]
    X = np.stack([_encode_sample(s) for s in pool], axis=0)
    mu, sd = _ensemble_predict(models, X, device=device)
    score = mu - kappa * sd
    order = np.argsort(score)  # minimize
    out = []
    seen = already or set()
    for idx in order:
        cand = pool[idx]
        key = tuple(
            int(round(v * 1e6)) if isinstance(v, float) else int(v)
            for v in _encode_sample(cand)
        )
        if key in seen:
            continue
        out.append(cand)
        seen.add(key)
        if len(out) >= batch_size:
            break
    return out


# ===================== neural search loop (parallel eval + panel) =====================


def neural_search_on_cloud(
    cloud: o3d.geometry.PointCloud,
    *,
    total_evals: int = 120,
    init_evals: int = 16,
    batch_size: int = 8,
    ensemble: int = 5,
    n_jobs: int | None = None,
    top_k: int = 5,
) -> List[TrialResult]:
    """
    One tqdm bar + live metrics. Surrogate models (MLP ensemble) run on GPU.
    """
    _set_quiet_logs()
    P = np.asarray(cloud.points, np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) == 0:
        raise ValueError("cloud has no points")

    device = _TORCH_DEVICE

    # warm-up single synchronous trial
    rng = np.random.default_rng(42)
    warm = _random_sample(rng)
    vox, em, ct = _vec_to_params(warm)
    w_loss, w_metrics, w_em, w_ct, w_vox = _eval_trial_worker(P, em, ct, vox)
    warm_tr = TrialResult(
        loss=w_loss,
        metrics=w_metrics,
        em_params=w_em,
        ct_params=w_ct,
        voxel=w_vox,
    )

    # storage
    X = [_encode_sample(warm)]
    y = [w_loss]
    hist: List[TrialResult] = [warm_tr]
    best: List[TrialResult] = [warm_tr]
    win = deque([w_metrics], maxlen=20)

    # progress widgets
    pbar = tqdm(
        total=total_evals,
        desc="NeuralSearch",
        ncols=100,
        leave=True,
        dynamic_ncols=True,
    )
    met = tqdm(total=0, position=1, leave=False, ncols=100, bar_format="{desc}")
    pbar.set_postfix(_metrics_postfix(warm_tr.metrics))
    pbar.update(1)

    # live panel
    def _update_panel(current_metrics: dict | None):
        cur_m = current_metrics or best[-1].metrics
        cur = _metrics_postfix(cur_m)
        b = _metrics_postfix(best[0].metrics)

        def _avg(field: str):
            vals = []
            for m in win:
                p = _metrics_postfix(m)
                try:
                    vals.append(float(p[field]))
                except Exception:
                    pass
            return (sum(vals) / len(vals)) if vals else None

        avg_em, avg_ct, avg_bn = _avg("EM_S"), _avg("CT_S"), _avg("Bnd")
        avg_str = (
            f"avg(20): EM_S={avg_em:.3f} CT_S={avg_ct:.3f} Bnd={avg_bn:.3f}"
            if avg_em is not None
            else "avg(20): —"
        )
        met.set_description_str(
            f"cur: BN={cur.get('BN','nan')} W2={cur.get('W2','nan')} | EM_S={cur.get('EM_S','nan')} CT_S={cur.get('CT_S','nan')} | Bnd={cur.get('Bnd','nan')}\n"
            f"best: BN={b.get('BN','nan')} W2={b.get('W2','nan')} | EM_S={b.get('EM_S','nan')} CT_S={b.get('CT_S','nan')} | Bnd={b.get('Bnd','nan')} | loss={best[0].loss:.4f}\n"
            f"{avg_str}"
        )

    _update_panel(warm_tr.metrics)

    # initial random batch (besides first)
    to_go = max(0, init_evals - 1)
    init_pool = [_random_sample(rng) for _ in range(to_go)]

    # parallel settings
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 2)
    ctx = get_context("spawn")

    backoff = 0.5
    done_total = 1  # warmup accounted

    def _submit_many(ex, samples: List[Dict[str, float]], inflight: dict):
        for s in samples:
            vox, em, ct = _vec_to_params(s)
            fut = ex.submit(_eval_trial_worker, P, em, ct, vox)
            inflight[fut] = s

    inflight: Dict[Any, Dict[str, float]] = {}

    # evaluate initial randoms in parallel
    if init_pool:
        try:
            ex = _make_executor(n_jobs, ctx)
            _submit_many(
                ex, init_pool[: min(batch_size, len(init_pool))], inflight
            )
            rest_queue = deque(init_pool[min(batch_size, len(init_pool)) :])
        except Exception:
            rest_queue = deque()
    else:
        rest_queue = deque()

    last_metrics = warm_tr.metrics

    # kappa annealing: exploration -> exploitation
    kappa_start, kappa_end = 0.35, 0.15

    while done_total < total_evals:
        try:
            if "ex" not in locals() or getattr(ex, "_shutdown_thread", None):
                ex = _make_executor(n_jobs, ctx)

            if not inflight:
                X_np = np.stack(X, axis=0)
                y_np = np.asarray(y, np.float64)

                models = _train_ensemble(
                    X_np, y_np, ensemble=ensemble, device=device
                )

                seen = set(
                    tuple(int(round(xx * 1e6)) for xx in x) for x in X_np
                )

                prog = min(1.0, done_total / float(total_evals))
                kappa_now = kappa_start + (kappa_end - kappa_start) * prog

                batch = _propose_batch(
                    models,
                    np.random.default_rng(777 + done_total),
                    batch_size,
                    pool_size=8192,
                    kappa=kappa_now,
                    already=seen,
                    device=device,
                )
                _submit_many(ex, batch, inflight)

            done, _ = wait(
                set(inflight.keys()), timeout=5.0, return_when=FIRST_COMPLETED
            )
            if not done:
                # keep workers busy with remaining init samples
                while rest_queue and len(inflight) < batch_size:
                    s = rest_queue.popleft()
                    _submit_many(ex, [s], inflight)
                _update_panel(last_metrics)
                continue

            # consume finished futures
            for fut in done:
                s = inflight.pop(fut, None)
                try:
                    loss, metrics, em_r, ct_r, vox_r = fut.result()
                except Exception:
                    done_total += 1
                    pbar.update(1)
                    continue

                tr = TrialResult(
                    loss=loss,
                    metrics=metrics,
                    em_params=em_r,
                    ct_params=ct_r,
                    voxel=vox_r,
                )
                hist.append(tr)
                last_metrics = metrics
                win.append(metrics)

                X.append(_encode_sample(s))
                y.append(loss)

                best.append(tr)
                best.sort(key=lambda t: t.loss)
                if len(best) > top_k:
                    best.pop(-1)

                pbar.set_postfix(_metrics_postfix(metrics))
                pbar.update(1)
                done_total += 1

            while (
                rest_queue
                and len(inflight) < batch_size
                and done_total < total_evals
            ):
                s = rest_queue.popleft()
                _submit_many(ex, [s], inflight)

            _update_panel(last_metrics)
            backoff = 0.5

        except concurrent.futures.process.BrokenProcessPool:
            try:
                ex.shutdown(cancel_futures=True)
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 4.0)
            continue

    try:
        ex.shutdown(cancel_futures=True)
    except Exception:
        pass
    pbar.close()
    met.close()

    best.sort(key=lambda t: t.loss)
    return best[:top_k]


# ===================== visualization of top-K =====================


def _log_trial_info(rank: int, tr: TrialResult) -> None:
    LOG.info(f"[Top#{rank}] loss={tr.loss:.6f} voxel={tr.voxel:.5f}")
    LOG.info(f"[Top#{rank}] EM: {tr.em_params}")
    LOG.info(f"[Top#{rank}] CTR: {tr.ct_params}")
    LOG.info(f"[Top#{rank}] metrics: {tr.metrics}")


def visualize_top_k(
    cloud: o3d.geometry.PointCloud, top: List[TrialResult]
) -> None:
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    _set_info_logs()

    state = {"idx": 0, "mode": 2, "line_w": 4.0, "em": None, "ct": None}

    def build_current():
        tr = top[state["idx"]]
        em_lines, em_balls, _ = skeletonize_em(
            cloud, voxel_size=tr.voxel, params=tr.em_params
        )
        ct_lines, ct_balls, _ = skeletonize_contraction(
            cloud, voxel_size=tr.voxel, params=tr.ct_params
        )
        state["em"] = (em_lines, em_balls)
        state["ct"] = (ct_lines, ct_balls)
        _log_trial_info(state["idx"] + 1, tr)

    build_current()

    line_mat = rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = state["line_w"]
    pts_mat = rendering.MaterialRecord()
    pts_mat.shader = "defaultUnlit"
    pts_mat.point_size = 2.0
    ball_mat = rendering.MaterialRecord()
    ball_mat.shader = "defaultLit"

    gui.Application.instance.initialize()
    win = gui.Application.instance.create_window(
        "Top-K: 1 cloud | 2 EM | 3 Contraction | Esc next", 1280, 800
    )
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(win.renderer)
    scene.scene.set_background([1, 1, 1, 1])
    win.add_child(scene)

    scene.scene.add_geometry("cloud", cloud, pts_mat)

    def set_mode(m: int):
        state["mode"] = m
        scene.scene.show_geometry("cloud", m == 1)
        scene.scene.show_geometry("em_lines", m == 2)
        scene.scene.show_geometry("em_balls", m == 2)
        scene.scene.show_geometry("ct_lines", m == 3)
        scene.scene.show_geometry("ct_balls", m == 3)

    def sync_scene(first=False):
        em_lines, em_balls = state["em"]
        ct_lines, ct_balls = state["ct"]
        if first:
            scene.scene.add_geometry("em_lines", em_lines, line_mat)
            scene.scene.add_geometry("em_balls", em_balls, ball_mat)
            scene.scene.add_geometry("ct_lines", ct_lines, line_mat)
            scene.scene.add_geometry("ct_balls", ct_balls, ball_mat)
            bbox = cloud.get_axis_aligned_bounding_box()
            scene.setup_camera(60.0, bbox, bbox.get_center())
        else:
            for name in ("em_lines", "em_balls", "ct_lines", "ct_balls"):
                scene.scene.remove_geometry(name)
            scene.scene.add_geometry("em_lines", em_lines, line_mat)
            scene.scene.add_geometry("em_balls", em_balls, ball_mat)
            scene.scene.add_geometry("ct_lines", ct_lines, line_mat)
            scene.scene.add_geometry("ct_balls", ct_balls, ball_mat)
        set_mode(state["mode"])

    sync_scene(first=True)

    HANDLED = True
    IGNORED = False

    def key_is(event, *names: str) -> bool:
        KN = getattr(gui, "KeyName", None)
        if KN is None:
            return False
        for nm in names:
            if getattr(KN, nm, None) == getattr(event, "key", None):
                return True
        return False

    def key_chr(event, ch: str) -> bool:
        k = getattr(event, "key", None)
        if isinstance(k, int) and len(ch) == 1:
            return k == ord(ch)
        s = getattr(event, "key_name", None) or getattr(
            event, "key_string", None
        )
        return isinstance(s, str) and s.lower() == ch.lower()

    def rebuild_and_refresh():
        build_current()
        sync_scene(first=False)

    def on_key(event):
        if key_is(event, "DIGIT1", "NUM_1", "KP_1") or key_chr(event, "1"):
            set_mode(1)
            return HANDLED
        if key_is(event, "DIGIT2", "NUM_2", "KP_2") or key_chr(event, "2"):
            set_mode(2)
            return HANDLED
        if key_is(event, "DIGIT3", "NUM_3", "KP_3") or key_chr(event, "3"):
            set_mode(3)
            return HANDLED
        if key_is(event, "BRACKET_RIGHT") or key_chr(event, "]"):
            state["line_w"] += 1.0
            line_mat.line_width = state["line_w"]
            scene.scene.modify_geometry_material("em_lines", line_mat)
            scene.scene.modify_geometry_material("ct_lines", line_mat)
            return HANDLED
        if key_is(event, "BRACKET_LEFT") or key_chr(event, "["):
            state["line_w"] = max(1.0, state["line_w"] - 1.0)
            line_mat.line_width = state["line_w"]
            scene.scene.modify_geometry_material("em_lines", line_mat)
            scene.scene.modify_geometry_material("ct_lines", line_mat)
            return HANDLED
        if key_is(event, "ESCAPE") or key_chr(event, "\x1b"):
            if state["idx"] + 1 < len(top):
                state["idx"] += 1
                rebuild_and_refresh()
                return HANDLED
            else:
                gui.Application.instance.quit()
                return HANDLED
        return IGNORED

    win.set_on_key(on_key)
    gui.Application.instance.run()


# ===================== end-to-end =====================


def run_param_sweep(path_loader_callable) -> None:
    """
    1) Load cloud
    2) Neural search (GPU surrogate; one tqdm with live metrics)
    3) Log Top-5
    4) Visualize Top-K
    """
    _set_quiet_logs()
    LOG.info("[Neural] start search")

    loaded = path_loader_callable()
    cloud = (
        loaded
        if isinstance(loaded, o3d.geometry.PointCloud)
        else o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.asarray(loaded, np.float64))
        )
    )

    top = neural_search_on_cloud(
        cloud, total_evals=256, init_evals=24, batch_size=8, ensemble=6
    )

    _set_info_logs()
    for rank, tr in enumerate(top, 1):
        _log_trial_info(rank, tr)

    import json, pathlib

    best = top[0]
    out = {
        "voxel": best.voxel,
        "EM": vars(best.em_params),
        "CTR": vars(best.ct_params),
        "loss": best.loss,
        "metrics": best.metrics,
    }
    pathlib.Path(".neural_search").mkdir(exist_ok=True)
    with open(".neural_search/best_params.json", "w") as f:
        json.dump(out, f, indent=2)
    LOG.info(
        f"[BEST] loss={best.loss:.4f} voxel={best.voxel:.5f} | saved to .neural_search/best_params.json"
    )

    visualize_top_k(cloud, top)


# ===================== main =====================

if __name__ == "__main__":
    from main import _load_cloud
    from vision.skeleton.config import SkelPipelineCfg

    cfg = SkelPipelineCfg()
    loaded = _load_cloud(cfg)
    cloud = (
        loaded
        if isinstance(loaded, o3d.geometry.PointCloud)
        else o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.asarray(loaded, np.float64))
        )
    )
    run_param_sweep(lambda: cloud)
