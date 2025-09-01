# new.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple, List, Optional, Dict, Iterable, NamedTuple

import os
import io
import sys
import time
import contextlib
import logging
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


# ===================== utility: logging control =====================


def _make_executor(n_jobs: int, ctx):
    """Process pool with worker recycling to avoid long-lived native leaks."""
    return ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=ctx,
        initializer=_worker_init,
        max_tasks_per_child=16,
    )


def _set_quiet_logs() -> None:
    """Silence our logs and noisy deps while grid search runs."""
    try:
        LOG.level("WARNING")
        LOG.disable("")  # loguru noop when bound
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
    """Re-enable informative logs for final top-K stage."""
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


# ===================== helpers (viz, graph, colors) =====================


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
        LOG.debug(f"Normals estimated: radius={radius:.5f}")


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


def _branch_edge_colors(nodes: np.ndarray, E: np.ndarray) -> np.ndarray:
    if len(E) == 0:
        return np.empty((0, 3), np.float64)
    adj = _adj_from_edges(len(nodes), E)
    chains = _compress_deg2_lines(adj)
    e2i = {tuple(sorted(map(int, e))): i for i, e in enumerate(E)}
    cols = np.ones((len(E), 3), np.float64)
    pal = _distinct_colors(max(1, len(chains)))
    for k, ch in enumerate(chains):
        for a, b in zip(ch[:-1], ch[1:]):
            key = tuple(sorted((int(a), int(b))))
            if key in e2i:
                cols[e2i[key]] = pal[k]
    return cols


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


def _build_initial_nodes(
    pcd: o3d.geometry.PointCloud, surf: o3d.geometry.TriangleMesh
) -> np.ndarray:
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
    nodes, E = adaptive_graph_3d(C, k=10, m=2.2, angle_deg=70.0)
    nodes = snap_nodes_to_mesh(nodes, surf, k=9)
    nodes, _ = merge_nodes_radius(nodes, E, radius=0.012)
    LOG.info(f"[Init] initial nodes={len(nodes)}")
    return nodes


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

    nodes0 = _build_initial_nodes(pcd, surf)
    if len(nodes0) == 0:
        LOG.warning("[Skel-EM] no initial nodes")
        return o3d.geometry.LineSet(), o3d.geometry.TriangleMesh(), pcd

    nodes = _em_refine(pcd, nodes0, cfg, voxel_size=voxel_size)
    E = _mst_edges(nodes, k=8)
    nodes_m, E_m = merge_nodes_radius(nodes, E, radius=cfg.node_merge_r)
    _refine_junctions_inplace(
        nodes_m, E_m, np.asarray(pcd.points), radius=3.0 * voxel_size
    )
    nodes_ref = _refine_junction_nodes(nodes_m, E_m, pcd)
    edge_cols = _branch_edge_colors(nodes_ref, E_m)
    lines, balls = _to_lines_and_balls(
        nodes_ref, E_m, cfg.node_vis_r, edge_cols
    )
    LOG.info(f"[Skel-EM] done: nodes={len(nodes_ref)} edges={len(E_m)}")
    return lines, balls, pcd


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

    edge_cols = _branch_edge_colors(nodes_m, E_m)
    lines, balls = _to_lines_and_balls(nodes_m, E_m, cfg.node_vis_r, edge_cols)
    LOG.info(f"[Skel-CTR] done: nodes={len(nodes_m)} edges={len(E_m)}")
    return lines, balls, pcd


# ===================== metrics & objective =====================

# We keep metrics self-contained in repo via local module `metrics.py`.
# We suppress any prints from metrics during grid search.

try:
    from metrics import report_all_metrics, MetricConfig

    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False

    class MetricConfig:  # stub
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
    """Combine metrics into a scalar loss (lower is better)."""
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
        + 0.30 * (1.0 - min(ctS, 1.0))
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
    """Compact, stable set of fields to show in tqdm postfix."""
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
    points: np.ndarray,
    em: EMParams,
    ct: CtrParams,
    voxel: float,
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


# ===================== grid search =====================


def grid_search_on_cloud(
    cloud: o3d.geometry.PointCloud,
    *,
    # EM grid
    em_max_iter=(8, 12),
    em_eps=(0.03, 0.06),
    em_sigma_scale=(0.2, 0.4),
    em_pca_mult=(0.8, 1.2),
    em_lap_lambda=(0.3, 0.6),
    em_lap_k=(1, 3),
    em_neighbor_sigma=(0.8, 1.2),
    em_merge_r=(0.012, 0.018),
    # Contraction grid
    ct_iters=(10, 14),
    ct_k=(7, 9, 11),
    ct_step=(0.75, 0.85),
    ct_aniso=(0.6, 0.8),
    ct_merge_r=(0.017, 0.020),
    # voxel grid
    voxel=(0.004, 0.005),
    # parallel
    n_jobs: int | None = None,
    top_k: int = 5,
) -> List[TrialResult]:
    """
    Robust grid-search:
      - one tqdm bar + live metrics panel (cur/best/avg)
      - warm-up first trial in main proc (no NaN at start)
      - worker recycling & pool auto-restart on BrokenProcessPool
    """
    _set_quiet_logs()

    P = np.asarray(cloud.points, np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) == 0:
        raise ValueError("cloud has no points")

    em0 = EMParams()
    ct0 = CtrParams()

    # materialize all combos once (we will warm up on the first)
    grids = (
        tuple(_expand(voxel)),
        tuple(_expand(em_max_iter)),
        tuple(_expand(em_eps)),
        tuple(_expand(em_sigma_scale)),
        tuple(_expand(em_pca_mult)),
        tuple(_expand(em_lap_lambda)),
        tuple(_expand(em_lap_k)),
        tuple(_expand(em_neighbor_sigma)),
        tuple(_expand(em_merge_r)),
        tuple(_expand(ct_iters)),
        tuple(_expand(ct_k)),
        tuple(_expand(ct_step)),
        tuple(_expand(ct_aniso)),
        tuple(_expand(ct_merge_r)),
    )
    combos_raw = list(product(*grids))
    total = len(combos_raw)

    def make_params(raw):
        (
            vox,
            e_maxit,
            e_eps,
            e_sig,
            e_pca,
            e_lap,
            e_k,
            e_ns,
            e_mr,
            c_it,
            c_k,
            c_step,
            c_an,
            c_mr,
        ) = raw
        em = replace(
            em0,
            max_iter=int(e_maxit),
            epsilon=float(e_eps),
            sigma_init_scale=float(e_sig),
            pca_radius_mult=float(e_pca),
            lap_lambda=float(e_lap),
            lap_k=int(e_k),
            neighbor_sigma_mult=float(e_ns),
            node_merge_r=float(e_mr),
        )
        ct = replace(
            ct0,
            iters=int(c_it),
            k=int(c_k),
            step=float(c_step),
            anisotropy=float(c_an),
            node_merge_r=float(c_mr),
        )
        return float(vox), em, ct

    # warm-up trial synchronously (robust start and sane panel)
    warm_vox, warm_em, warm_ct = make_params(combos_raw[0])
    try:
        warm_loss, warm_metrics, warm_em_r, warm_ct_r, warm_vox_r = (
            _eval_trial_worker(P, warm_em, warm_ct, warm_vox)
        )
        warm_trial = TrialResult(
            loss=warm_loss,
            metrics=warm_metrics,
            em_params=warm_em_r,
            ct_params=warm_ct_r,
            voxel=warm_vox_r,
        )
    except Exception as e:
        raise RuntimeError(f"Warmup trial failed: {e}")

    # state
    best: List[TrialResult] = [warm_trial]
    inflight: dict = {}
    pending = deque(combos_raw[1:])  # skip warm-up
    win = deque(maxlen=20)

    # progress + metrics panel
    pbar = tqdm(
        total=total, desc="Grid", ncols=100, leave=True, dynamic_ncols=True
    )
    met = tqdm(total=0, position=1, leave=False, ncols=100, bar_format="{desc}")

    # draw warm-up
    pbar.set_postfix(_metrics_postfix(warm_trial.metrics))
    pbar.update(1)
    win.append(warm_trial.metrics)

    def _update_metrics_panel(current_metrics: dict | None):
        if not best and current_metrics is None:
            met.set_description_str("waiting for first result…")
            return

        cur_m = current_metrics or best[-1].metrics
        cur = _metrics_postfix(cur_m)
        cur_str = (
            f"cur: BN={cur.get('BN','nan')} W2={cur.get('W2','nan')} "
            f"| EM_S={cur.get('EM_S','nan')} CT_S={cur.get('CT_S','nan')} "
            f"| Bnd={cur.get('Bnd','nan')}"
        )

        b = _metrics_postfix(best[0].metrics)
        best_str = (
            f"best: BN={b.get('BN','nan')} W2={b.get('W2','nan')} "
            f"| EM_S={b.get('EM_S','nan')} CT_S={b.get('CT_S','nan')} "
            f"| Bnd={b.get('Bnd','nan')} | loss={best[0].loss:.4f}"
        )

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

        met.set_description_str(f"{cur_str}\n{best_str}\n{avg_str}")

    _update_metrics_panel(warm_trial.metrics)

    # main loop with pool auto-restart
    backoff = 0.5
    last_metrics = warm_trial.metrics

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 2)
    ctx = get_context("spawn")

    while inflight or pending:
        try:
            if 'ex' not in locals() or getattr(ex, "_shutdown_thread", None):
                ex = _make_executor(n_jobs, ctx)
                inflight = {}
                max_inflight = min(2 * n_jobs, 8 * n_jobs)
                while pending and len(inflight) < max_inflight:
                    combo = pending.popleft()
                    vox, em, ct = make_params(combo)
                    fut = ex.submit(_eval_trial_worker, P, em, ct, vox)
                    inflight[fut] = combo

            done, _ = wait(
                set(inflight.keys()), timeout=5.0, return_when=FIRST_COMPLETED
            )

            if not done:
                while pending and len(inflight) < max_inflight:
                    combo = pending.popleft()
                    vox, em, ct = make_params(combo)
                    fut = ex.submit(_eval_trial_worker, P, em, ct, vox)
                    inflight[fut] = combo
                _update_metrics_panel(last_metrics)
                continue

            for fut in done:
                combo = inflight.pop(fut, None)
                try:
                    loss, metrics, em_r, ct_r, vox_r = fut.result()
                except Exception:
                    pbar.update(1)
                    continue

                res = TrialResult(
                    loss=loss,
                    metrics=metrics,
                    em_params=em_r,
                    ct_params=ct_r,
                    voxel=vox_r,
                )
                best.append(res)
                best.sort(key=lambda x: x.loss)
                if len(best) > top_k:
                    best.pop(-1)

                last_metrics = metrics
                win.append(metrics)
                pbar.set_postfix(_metrics_postfix(metrics))
                pbar.update(1)

            while pending and len(inflight) < max_inflight:
                combo = pending.popleft()
                vox, em, ct = make_params(combo)
                fut = ex.submit(_eval_trial_worker, P, em, ct, vox)
                inflight[fut] = combo

            _update_metrics_panel(last_metrics)
            backoff = 0.5

        except concurrent.futures.process.BrokenProcessPool:
            try:
                ex.shutdown(cancel_futures=True)
            except Exception:
                pass
            for fut, combo in list(inflight.items()):
                if not fut.done():
                    pending.appendleft(combo)
            inflight.clear()
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 4.0)
            continue

    try:
        ex.shutdown(cancel_futures=True)
    except Exception:
        pass
    pbar.close()
    met.close()
    return best


# ===================== visualization of top-K =====================


def _log_trial_info(rank: int, tr: TrialResult) -> None:
    LOG.info(f"[Top#{rank}] loss={tr.loss:.6f} voxel={tr.voxel:.5f}")
    LOG.info(f"[Top#{rank}] EM: {tr.em_params}")
    LOG.info(f"[Top#{rank}] CTR: {tr.ct_params}")
    LOG.info(f"[Top#{rank}] metrics: {tr.metrics}")


def visualize_top_k(
    cloud: o3d.geometry.PointCloud,
    top: List[TrialResult],
) -> None:
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    _set_info_logs()

    state = {
        "idx": 0,
        "mode": 2,  # 2=EM, 3=CTR, 1=cloud
        "line_w": 4.0,
        "em": None,
        "ct": None,
    }

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
        "Top-K Grid Search: 1 cloud | 2 EM | 3 Contraction | Esc next",
        1280,
        800,
    )
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(win.renderer)
    scene.scene.set_background([1, 1, 1, 1])
    win.add_child(scene)

    scene.scene.add_geometry("cloud", cloud, pts_mat)

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
            scene.scene.remove_geometry("em_lines")
            scene.scene.remove_geometry("em_balls")
            scene.scene.remove_geometry("ct_lines")
            scene.scene.remove_geometry("ct_balls")
            scene.scene.add_geometry("em_lines", em_lines, line_mat)
            scene.scene.add_geometry("em_balls", em_balls, ball_mat)
            scene.scene.add_geometry("ct_lines", ct_lines, line_mat)
            scene.scene.add_geometry("ct_balls", ct_balls, ball_mat)
        set_mode(state["mode"])

    def set_mode(m: int):
        state["mode"] = m
        scene.scene.show_geometry("cloud", m == 1)
        scene.scene.show_geometry("em_lines", m == 2)
        scene.scene.show_geometry("em_balls", m == 2)
        scene.scene.show_geometry("ct_lines", m == 3)
        scene.scene.show_geometry("ct_balls", m == 3)

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


# ===================== end-to-end: sweep then visualize top-K =====================


def run_param_sweep(path_loader_callable) -> None:
    """
    1) Load cloud.
    2) Run parallel grid search with a single tqdm line; postfix shows metrics.
    3) Extract top-5 by loss and log only those.
    4) Visualize top-K with EM/Contraction toggles.
    """
    _set_quiet_logs()
    LOG.info("[Grid] start param sweep")

    loaded = path_loader_callable()
    cloud = (
        loaded
        if isinstance(loaded, o3d.geometry.PointCloud)
        else o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.asarray(loaded, np.float64))
        )
    )

    top = grid_search_on_cloud(cloud)

    _set_info_logs()
    for rank, tr in enumerate(top, 1):
        _log_trial_info(rank, tr)

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
