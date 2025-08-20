# vision/skeleton/geometry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from utils.logger import Logger

LOG = Logger.get_logger("geom")


def lineset_has_points(ls) -> bool:
    try:
        return np.asarray(ls.points).shape[0] > 0
    except Exception:
        return False


def _fit_line_svd(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    c = X.mean(0)
    A = X - c
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    d = Vt[0]
    t = (X - c) @ d
    p0 = c + t.min() * d
    p1 = c + t.max() * d
    lam = (S**2) / (len(X) - 1 + 1e-12)
    lam = np.pad(lam, (0, max(0, 3 - len(lam))), constant_values=0)
    linearity = (lam[1] + lam[2]) / (lam[0] + 1e-12)
    length = float(np.linalg.norm(p1 - p0))
    return p0, p1, length, float(linearity)


def resample_polyline_world(
    P: np.ndarray, step: float, keep_tail_min_frac: float
) -> np.ndarray:
    if len(P) < 2:
        return P
    total = float(np.sum(np.linalg.norm(P[1:] - P[:-1], axis=1)))
    if total < step:
        return P
    out = [P[0]]
    acc = 0.0
    target = step
    for i in range(1, len(P)):
        seg = P[i] - P[i - 1]
        L = float(np.linalg.norm(seg))
        if L <= 1e-12:
            continue
        dirv = seg / L
        while acc + L >= target:
            d = target - acc
            out.append(P[i - 1] + dirv * d)
            target += step
        acc += L
    tail = acc - ((len(out) - 1) * step)
    if tail >= keep_tail_min_frac * step:
        out.append(P[-1])
    return np.asarray(out) if len(out) >= 2 else P


def straighten_polylines(
    polys: List[np.ndarray], min_len: float, lin_thr: float, step: float
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for P in polys:
        if len(P) < 2:
            out.append(P)
            continue
        p0, p1, L, lin = _fit_line_svd(P)
        if (L >= min_len) and (lin <= lin_thr):
            npts = max(2, int(round(L / max(1e-6, step))) + 1)
            t = np.linspace(0.0, 1.0, npts)
            line = p0[None, :] * (1 - t)[:, None] + p1[None, :] * t[:, None]
            out.append(line)
        else:
            out.append(P)
    return out


@dataclass(frozen=True)
class PlaneBasis:
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    p0: np.ndarray


def plane_basis_from_coeffs(model: np.ndarray, pts: np.ndarray) -> PlaneBasis:
    a, b, c, _ = model
    n = np.array([a, b, c], float)
    n /= np.linalg.norm(n) + 1e-12
    p0 = pts.mean(0)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1, 0, 0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return PlaneBasis(u=u, v=v, n=n, p0=p0)


def to_plane_uv(pts: np.ndarray, basis: PlaneBasis) -> np.ndarray:
    q = pts - basis.p0
    return np.stack([q @ basis.u, q @ basis.v], axis=1)
