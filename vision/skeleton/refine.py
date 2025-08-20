# vision/skeleton/refine.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import open3d as o3d
from utils.logger import Logger

LOG = Logger.get_logger("refine")


def resample_polyline_world(
    P: np.ndarray, step: float, keep_tail_min_frac: float
) -> np.ndarray:
    """Even spacing along polyline; keep tail if long enough."""
    if len(P) < 2:
        return P
    seg = P[1:] - P[:-1]
    L = np.linalg.norm(seg, axis=1)
    total = float(L.sum())
    if total < step:
        return P
    out = [P[0]]
    acc = 0.0
    target = step
    for i in range(1, len(P)):
        d = float(np.linalg.norm(P[i] - P[i - 1]))
        if d <= 1e-12:
            continue
        dirv = (P[i] - P[i - 1]) / d
        while acc + d >= target:
            out.append(P[i - 1] + dirv * (target - acc))
            target += step
        acc += d
    tail = acc - ((len(out) - 1) * step)
    if tail >= keep_tail_min_frac * step:
        out.append(P[-1])
    return np.asarray(out) if len(out) >= 2 else P


def _ensure_normals(pcd: o3d.geometry.PointCloud) -> None:
    if pcd.has_normals() and len(pcd.normals) == len(pcd.points):
        return
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=60)
    )


def refine_polylines_centers(
    polys: List[np.ndarray],
    cloud: o3d.geometry.PointCloud,
    cfg,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """For each poly point, take mid of two opposite normal clusters."""
    _ensure_normals(cloud)
    P = np.asarray(cloud.points)
    N = np.asarray(cloud.normals)
    kdt = o3d.geometry.KDTreeFlann(cloud)
    out_polys, widths = [], []
    for poly in polys:
        if len(poly) == 0:
            continue
        refined, w_arr = [], []
        for p in poly:
            _, idx, _ = kdt.search_radius_vector_3d(p, cfg.search_r)
            if len(idx) < cfg.min_side_pts * 2:
                refined.append(p)
                w_arr.append(np.nan)
                continue
            Q, M = P[idx], N[idx]
            good = np.isfinite(M).all(1)
            Q, M = Q[good], M[good]
            if len(Q) < cfg.min_side_pts * 2:
                refined.append(p)
                w_arr.append(np.nan)
                continue
            Mm = M - M.mean(0)
            try:
                _, _, Vt = np.linalg.svd(Mm, False)
                dirn = Vt[0]
            except Exception:
                refined.append(p)
                w_arr.append(np.nan)
                continue
            s = M @ dirn
            plus = s >= 0
            minus = ~plus
            if plus.sum() < cfg.min_side_pts or minus.sum() < cfg.min_side_pts:
                refined.append(p)
                w_arr.append(np.nan)
                continue
            c_plus = Q[plus].mean(0)
            c_minus = Q[minus].mean(0)
            sep = float(np.linalg.norm(c_plus - c_minus))
            lo, hi = 0.3 * cfg.exp_width, 2.2 * cfg.exp_width
            if not (lo <= sep <= hi):
                refined.append(p)
                w_arr.append(np.nan)
                continue
            refined.append(0.5 * (c_plus + c_minus))
            w_arr.append(sep)
        out_polys.append(np.asarray(refined))
        widths.append(np.asarray(w_arr))
    return out_polys, widths


def _fit_line_svd(X: np.ndarray):
    c = X.mean(0)
    A = X - c
    U, S, Vt = np.linalg.svd(A, False)
    d = Vt[0]
    t = (X - c) @ d
    p0 = c + t.min() * d
    p1 = c + t.max() * d
    lam = (S**2) / (len(X) - 1 + 1e-9)
    lam = np.pad(lam, (0, max(0, 3 - len(lam))), constant_values=0)
    linearity = (lam[1] + lam[2]) / (lam[0] + 1e-12)
    L = float(np.linalg.norm(p1 - p0))
    return p0, p1, L, float(linearity)


def straighten_polylines(
    polys: List[np.ndarray],
    min_len: float,
    linearity_thr: float,
    step: float,
) -> List[np.ndarray]:
    """Replace near-linear long polylines by straight segments."""
    out = []
    for P in polys:
        if len(P) < 2:
            out.append(P)
            continue
        p0, p1, L, lin = _fit_line_svd(P)
        if (L >= min_len) and (lin <= linearity_thr):
            npts = max(2, int(round(L / max(1e-6, step))) + 1)
            t = np.linspace(0.0, 1.0, npts)
            out.append(
                (p0[None, :] * (1 - t)[:, None] + p1[None, :] * t[:, None])
            )
        else:
            out.append(P)
    return out
