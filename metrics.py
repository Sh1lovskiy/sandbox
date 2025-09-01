# metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

__all__ = [
    "MetricConfig",
    "report_all_metrics",
    "topo_similarity",
    "boundedness",
    "centeredness_curve",
    "smoothness_curve",
]

# ---------- utils ----------


def _np_pts(pcd_or_np) -> np.ndarray:
    """Return Nx3 float64 points from Open3D PointCloud or array-like."""
    if isinstance(pcd_or_np, o3d.geometry.PointCloud):
        return np.asarray(pcd_or_np.points, dtype=np.float64)
    a = np.asarray(pcd_or_np, dtype=np.float64)
    return a.reshape((-1, 3))


def _lineset_to_graph(
    ls: o3d.geometry.LineSet,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (V,E) from LineSet; E is int32, shape (M,2)."""
    V = np.asarray(ls.points, dtype=np.float64)
    E = (
        np.asarray(ls.lines, dtype=np.int32)
        if len(ls.lines)
        else np.empty((0, 2), np.int32)
    )
    return V, E


def _sample_on_edges(V: np.ndarray, E: np.ndarray, step: float) -> np.ndarray:
    """Uniform samples along edges with step ~ 'step'."""
    if len(E) == 0 or len(V) == 0:
        return np.empty((0, 3), np.float64)
    out = []
    for a, b in E:
        pa, pb = V[a], V[b]
        L = np.linalg.norm(pb - pa)
        n = max(2, int(np.ceil(L / max(step, 1e-9))))
        t = np.linspace(0.0, 1.0, n, endpoint=True)
        seg = pa[None, :] * (1 - t[:, None]) + pb[None, :] * t[:, None]
        out.append(seg)
    return np.vstack(out) if out else np.empty((0, 3), np.float64)


# ---------- Topology via H0 barcode (MST edge lengths) ----------


def _mst_edge_lengths(P: np.ndarray, k: int = 12) -> np.ndarray:
    """
    H0 persistence through MST of kNN-graph.
    Return edge weights selected by the (directed) MST.
    """
    if len(P) <= 1:
        return np.empty(0, np.float64)
    tree = cKDTree(P)
    rows, cols, data = [], [], []
    for i, p in enumerate(P):
        d, j = tree.query(p, k=min(k + 1, len(P)))
        for dj, jj in zip(d[1:], j[1:]):
            rows.append(i)
            cols.append(int(jj))
            data.append(float(dj))
    G = csr_matrix((data, (rows, cols)), shape=(len(P), len(P)))
    MST = minimum_spanning_tree(G).tocoo()
    # We use raw positive weights; duplicates are negligible for scoring.
    return np.asarray(MST.data, dtype=np.float64)


def _eps_star(P: np.ndarray) -> float:
    """Threshold from max 2-NN distance of P (denoising tiny bars)."""
    d, _ = cKDTree(P).query(P, k=min(2, len(P)))
    return float(np.max(d[:, -1])) if len(d) else 0.0


def _filter_barcode(deaths: np.ndarray, thr: float) -> np.ndarray:
    if deaths.size == 0:
        return deaths
    return deaths[deaths >= thr]


def topo_similarity(
    Po: np.ndarray, Ps: np.ndarray, p: int = 2, k: int = 12
) -> Dict[str, float]:
    """
    Compare H0 barcodes of Po and Ps after thresholding short bars by eps*(Po).
    Return bottleneck and p-Wasserstein distances between sorted deaths.
    """
    Po = _np_pts(Po)
    Ps = _np_pts(Ps)
    if len(Po) == 0 or len(Ps) == 0:
        return {
            "bottleneck": float("nan"),
            "wasserstein": float("nan"),
            "nb": 0,
        }

    d_o = np.sort(_mst_edge_lengths(Po, k))
    d_s = np.sort(_mst_edge_lengths(Ps, k))
    thr = _eps_star(Po)
    d_o = _filter_barcode(d_o, thr)
    d_s = _filter_barcode(d_s, thr)
    if d_o.size == 0 or d_s.size == 0:
        return {
            "bottleneck": float("inf"),
            "wasserstein": float("inf"),
            "nb": 0,
            "thr": thr,
        }

    n = min(d_o.size, d_s.size)
    a = d_o[-n:]
    b = d_s[-n:]
    bottleneck = float(np.max(np.abs(a - b)))
    wasserstein = float((np.mean(np.abs(a - b) ** p)) ** (1.0 / p))
    return {
        "bottleneck": bottleneck,
        "wasserstein": wasserstein,
        "nb": int(n),
        "thr": thr,
    }


# ---------- Boundedness (spherical coverage around nodes) ----------


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    phi = (1 + 5**0.5) / 2
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2 * np.pi * i / phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def boundedness(
    points_to_test: np.ndarray,
    Po: np.ndarray,
    k: int = 200,
    dirs: int = 512,
    ang_deg: float = 10.0,
    thresh: float = 0.85,
) -> Dict[str, object]:
    """
    For each skeleton node x, compute coverage of unit directions by vectors to k-NN points in Po.
    Score = mean directional hits; fraction = share with score >= thresh.
    """
    X = _np_pts(points_to_test)
    Po = _np_pts(Po)
    if len(X) == 0 or len(Po) == 0:
        return {"per_point": np.zeros(0), "fraction": 0.0}
    tree = cKDTree(Po)
    U = _fibonacci_sphere(max(8, int(dirs)))
    cos_thr = float(np.cos(np.deg2rad(ang_deg)))
    cov = np.zeros(len(X), np.float64)
    for i, x in enumerate(X):
        d, j = tree.query(x, k=min(k, len(Po)))
        vecs = Po[j] - x
        nn = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / nn
        M = U @ vecs.T  # [dirs, k]
        hit = (M >= cos_thr).any(axis=1)
        cov[i] = hit.mean()
    return {"per_point": cov, "fraction": float(np.mean(cov >= thresh))}


# ---------- Centeredness (transverse slabs along edges) ----------


def _orthobasis(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u = u / (np.linalg.norm(u) + 1e-12)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(a, u)) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    g = np.cross(u, a)
    g /= np.linalg.norm(g) + 1e-12
    h = np.cross(u, g)
    h /= np.linalg.norm(h) + 1e-12
    return g, h


def centeredness_curve(
    V: np.ndarray,
    E: np.ndarray,
    Po: np.ndarray,
    alpha: float = 0.4,
    samples_per_edge: int = 10,
    min_pts: int = 15,
    c_thr: float = 0.75,
) -> Dict[str, object]:
    """
    For samples along each edge, fit local cross-section in a transverse slab and
    score how close sample is to the centroid normalized by section radii.
    """
    V = _np_pts(V)
    Po = _np_pts(Po)
    E = (
        np.asarray(E, dtype=np.int32).reshape((-1, 2))
        if len(E)
        else np.empty((0, 2), np.int32)
    )
    if len(E) == 0 or len(V) == 0 or len(Po) == 0:
        return {"per_sample": np.zeros(0), "fraction": 0.0}

    if len(E):
        Lmin = np.min(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1))
    else:
        Lmin = np.max(np.ptp(Po, axis=0)) * 0.02
    w = max(1e-6, alpha * Lmin * 0.5)

    tree = cKDTree(Po)
    vals: List[float] = []
    for a, b in E:
        pa, pb = V[a], V[b]
        u = pb - pa
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            continue
        u /= nu
        g, h = _orthobasis(u)
        t = np.linspace(0, 1, max(2, int(samples_per_edge)), endpoint=True)
        S = pa[None, :] * (1 - t[:, None]) + pb[None, :] * t[:, None]
        for x in S:
            idx = tree.query_ball_point(x, r=max(3 * w, 1e-3))
            if not idx:
                vals.append(0.0)
                continue
            P = Po[idx]

            # NEW: адаптивный радиус по локальному шагу точек
            # (чтобы слэб не был «слишком тонким» для разреженных облаков)
            d_loc, _ = tree.query(x, k=min(10, len(Po)))
            nn_med = float(np.median(d_loc[1:])) if len(d_loc) > 1 else 0.0
            r_adapt = max(3 * w, 3.0 * nn_med, 1e-3)

            offs = P - x
            proj = offs @ u
            slab = np.abs(proj) <= r_adapt
            Q = P[slab]
            if len(Q) < min_pts:
                vals.append(0.0)
                continue
            Y = np.stack([(Q - x) @ g, (Q - x) @ h], axis=1)  # Nx2
            c = np.mean(Y, axis=0)
            C = Y - c
            # robust radii via singular values
            _, Sig, _ = np.linalg.svd(C, full_matrices=False)
            la = np.sqrt(Sig[0] / len(Y)) + 1e-12
            lb = np.sqrt(Sig[1] / len(Y)) + 1e-12
            d = np.linalg.norm(
                c
            )  # distance from x (origin) to centroid in local frame
            score = 1.0 - d / (0.5 * (la + lb))
            vals.append(float(np.clip(score, 0.0, 1.0)))
    vals = np.asarray(vals, dtype=np.float64) if vals else np.zeros(0)
    return {"per_sample": vals, "fraction": float(np.mean(vals >= c_thr))}


# ---------- Smoothness (angle deviation at degree-2 nodes) ----------


def smoothness_curve(V: np.ndarray, E: np.ndarray) -> Dict[str, float]:
    """
    S in [0,1]. At deg-2 nodes, penalize deviation from collinearity weighted by local edge length.
    """
    V = _np_pts(V)
    E = (
        np.asarray(E, dtype=np.int32).reshape((-1, 2))
        if len(E)
        else np.empty((0, 2), np.int32)
    )
    if len(E) == 0 or len(V) == 0:
        return {"S": 1.0}

    from collections import defaultdict

    adj = defaultdict(list)
    for a, b in E:
        adj[a].append(b)
        adj[b].append(a)

    def edge_len(i, j) -> float:
        return float(np.linalg.norm(V[i] - V[j]))

    total_len = 0.0
    penalty = 0.0
    for i, nbrs in adj.items():
        if len(nbrs) != 2:
            continue
        v0, v1 = nbrs[0], nbrs[1]
        u = V[v0] - V[i]
        v = V[v1] - V[i]
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            continue
        u /= nu
        v /= nv
        ang = np.arccos(np.clip(u @ v, -1.0, 1.0))  # [0,pi]
        Dn = ang / np.pi  # normalize to [0,1]
        s = abs(1.0 - 2.0 * Dn)  # 1 at 0/π (straight), 0 at π/2 (right angle)
        w = 0.5 * (edge_len(i, v0) + edge_len(i, v1))
        total_len += w
        penalty += w * (1.0 - s)
    S = 1.0 - (penalty / (total_len + 1e-12))
    return {"S": float(np.clip(S, 0.0, 1.0))}


# ---------- Aggregation ----------


@dataclass(frozen=True)
class MetricConfig:
    topo_k: int = 12
    topo_p: int = 2
    bounded_k: int = 768
    bounded_dirs: int = 2048
    bounded_ang_deg: float = 25.0
    bounded_thr: float = 0.5
    cent_alpha: float = 0.6
    cent_samples_per_edge: int = 16
    cent_min_pts: int = 10
    cent_thr: float = 0.65
    sample_step: float = 0.01


def report_all_metrics(
    cloud: o3d.geometry.PointCloud,
    em_lines: o3d.geometry.LineSet | None,
    ct_lines: o3d.geometry.LineSet | None,
    ct_points: np.ndarray | o3d.geometry.PointCloud | None,
    cfg: MetricConfig = MetricConfig(),
) -> Dict[str, Dict]:
    """
    Compute metrics for EM and Contraction skeletons + topology vs contraction points.
    Keys:
      - "topology_contraction": {"bottleneck","wasserstein","nb","thr"}
      - "bounded_EM" / "bounded_Contraction": {"fraction","mean"}
      - "centered_EM" / "centered_Contraction": {"fraction","mean"}
      - "smooth_EM" / "smooth_Contraction": {"S"}
    """
    Po = _np_pts(cloud)
    out: Dict[str, Dict] = {}

    if ct_points is not None:
        Ps = _np_pts(ct_points)
        topo = topo_similarity(Po, Ps, p=cfg.topo_p, k=cfg.topo_k)
        out["topology_contraction"] = topo

    for name, ls in [("EM", em_lines), ("Contraction", ct_lines)]:
        if ls is None:
            continue
        V, E = _lineset_to_graph(ls)

        # NEW: семплируем вдоль рёбер
        S = _sample_on_edges(
            V, E, step=cfg.sample_step if cfg.sample_step > 0 else 0.01
        )

        # boundedness теперь проверяем на S, а не на V
        bnd = boundedness(
            S,
            Po,
            k=cfg.bounded_k,
            dirs=cfg.bounded_dirs,
            ang_deg=cfg.bounded_ang_deg,
            thresh=cfg.bounded_thr,
        )

        cent = centeredness_curve(
            V,
            E,
            Po,
            alpha=cfg.cent_alpha,
            samples_per_edge=cfg.cent_samples_per_edge,
            min_pts=cfg.cent_min_pts,
            c_thr=cfg.cent_thr,
        )
        sm = smoothness_curve(V, E)

        out[f"bounded_{name}"] = {
            "fraction": float(bnd["fraction"]),
            "mean": (
                float(np.mean(bnd["per_point"]))
                if len(bnd["per_point"])
                else 0.0
            ),
        }
        out[f"centered_{name}"] = {
            "fraction": float(cent["fraction"]),
            "mean": (
                float(np.mean(cent["per_sample"]))
                if len(cent["per_sample"])
                else 0.0
            ),
        }
        out[f"smooth_{name}"] = sm

    # Optional console summary (new.py подавляет stdout во время грид-серча)
    def _fmt(v: float) -> str:
        return "nan" if not np.isfinite(v) else f"{v:.4f}"

    print("\n=== Skeleton metrics ===")
    if "topology_contraction" in out:
        t = out["topology_contraction"]
        thr = t.get("thr", float("nan"))
        print(
            f"[Topo (Contraction)] bottleneck={_fmt(t['bottleneck'])}, "
            f"W{cfg.topo_p}={_fmt(t['wasserstein'])}, nbars={t['nb']}, "
            f"eps*={_fmt(thr)}"
        )
    for name in ("EM", "Contraction"):
        if f"bounded_{name}" in out:
            b = out[f"bounded_{name}"]
            c = out[f"centered_{name}"]
            s = out[f"smooth_{name}"]
            print(
                f"[{name}] bounded(frac/mean)={_fmt(b['fraction'])}/{_fmt(b['mean'])} "
                f"| centered(frac/mean)={_fmt(c['fraction'])}/{_fmt(c['mean'])} "
                f"| smooth S={_fmt(s['S'])}"
            )
    print("========================\n")
    return out
