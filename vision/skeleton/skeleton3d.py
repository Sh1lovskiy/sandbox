# vision/skeleton/skeleton3d.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, List, Tuple, Dict, Optional, Set

import math
import numpy as np

try:  # optional for KD queries in graph-from-points helpers
    from scipy.spatial import cKDTree as _KDTree  # type: ignore
except Exception:  # pragma: no cover
    _KDTree = None  # fallback to brute-force

from utils.logger import Logger

LOG = Logger.get_logger("sk3d")


# ============================== DATA MODELS ==================================


@dataclass(frozen=True)
class VoxelSize:
    """Uniform voxel size in world units."""

    v: float

    def to_world(self, ijk: np.ndarray, origin: np.ndarray) -> np.ndarray:
        return origin[None, :] + (ijk + 0.5) * float(self.v)


# ============================== SKELETON CORE ================================


def skeletonize_3d(vol: np.ndarray) -> np.ndarray:
    """
    3D skeletonization with detailed logging.

    - Accepts: binary or grayscale volume (Z,Y,X).
    - Returns: binary volume (uint8) with 1 at skeleton voxels.
    - Falls back to per-slice 2D skeletonization if 3D op is missing.
    """
    t0 = perf_counter()
    if vol.ndim != 3:
        raise ValueError(f"vol must be 3D, got shape={vol.shape}")

    Z, Y, X = vol.shape
    vol_bool = vol > 0
    nz = int(vol_bool.sum())
    fill = (nz / float(vol_bool.size)) if vol_bool.size else 0.0
    LOG.info(f"[SK3D] input shape=({Z},{Y},{X}), nnz={nz}, fill={fill:.4f}")

    try:
        from skimage.morphology import skeletonize_3d as _sk3d  # type: ignore

        t1 = perf_counter()
        out = _sk3d(vol_bool).astype(np.uint8)
        t2 = perf_counter()
        LOG.info(
            f"[SK3D] 3D thinning ok | vox_out={int(out.sum())} | "
            f"time_load={t1-t0:.3f}s time_sk={t2-t1:.3f}s"
        )
        return out
    except Exception as e:
        LOG.warning(f"[SK3D] 3D op unavailable ({e}); fallback to 2D slices")

    try:
        from skimage.morphology import skeletonize as _sk2d  # type: ignore
    except Exception as e2:  # pragma: no cover
        raise RuntimeError(f"skel ops unavailable: {e2}") from e2

    out = np.zeros_like(vol_bool, dtype=np.uint8)
    for z in range(Z):
        out[z] = _sk2d(vol_bool[z]).astype(np.uint8)
    t3 = perf_counter()
    LOG.info(
        f"[SK3D] 2D fallback ok | vox_out={int(out.sum())} | time={t3-t0:.3f}s"
    )
    return out


# ============================ TOPOLOGY HELPERS ===============================


def _nbrs(Z: int, Y: int, X: int, v: Tuple[int, int, int]):
    """26-connected neighborhood, clipped to bounds."""
    z, y, x = v
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                w = (z + dz, y + dy, x + dx)
                if 0 <= w[0] < Z and 0 <= w[1] < Y and 0 <= w[2] < X:
                    yield w


def _degree(
    vox: Set[Tuple[int, int, int]],
    Z: int,
    Y: int,
    X: int,
    v: Tuple[int, int, int],
) -> int:
    return sum((w in vox) for w in _nbrs(Z, Y, X, v))


def skeleton_nodes_vox(sk: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Return voxel indices where skeleton degree != 2 (endpoints or junctions).
    """
    if sk.ndim != 3:
        raise ValueError("sk must be 3D")
    Z, Y, X = sk.shape
    vox = set(map(tuple, np.argwhere(sk == 1)))
    if not vox:
        LOG.info("[NODES] empty skeleton")
        return []
    out: List[Tuple[int, int, int]] = []
    for v in vox:
        if _degree(vox, Z, Y, X, v) != 2:
            out.append(v)
    LOG.info(f"[NODES] nodes={len(out)} from vox={len(vox)}")
    return out


# ============================== POLYLINES / GRAPH ============================


def _ek(u: Tuple[int, int, int], v: Tuple[int, int, int]):
    return (u, v) if u < v else (v, u)


def _trace_path(
    vox: Set[Tuple[int, int, int]],
    Z: int,
    Y: int,
    X: int,
    a: Tuple[int, int, int],
    b: Tuple[int, int, int],
    visited: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
) -> List[Tuple[int, int, int]]:
    """Trace along degree==2 voxels from node a toward b until node/stop."""
    path = [a, b]
    prv, cur = a, b

    while True:
        nb = [w for w in _nbrs(Z, Y, X, cur) if w in vox and w != prv]
        if _degree(vox, Z, Y, X, cur) != 2 or not nb:
            break
        nxt = nb[0]
        if _ek(cur, nxt) in visited:
            break
        path.append(nxt)
        prv, cur = cur, nxt
    return path


def skeleton_to_polylines_3d(
    sk: np.ndarray,
) -> List[List[Tuple[int, int, int]]]:
    """
    Convert a binary skeleton volume to voxel polylines.
    Each polyline runs between nodes (degree!=2) or around cycles.
    """
    if sk.ndim != 3:
        raise ValueError("sk must be 3D")
    Z, Y, X = sk.shape
    vox = set(map(tuple, np.argwhere(sk == 1)))
    if not vox:
        LOG.info("[POLY] empty skeleton")
        return []

    visited: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()
    nodes = [v for v in vox if _degree(vox, Z, Y, X, v) != 2]
    polylines: List[List[Tuple[int, int, int]]] = []

    # edges between nodes
    for v in nodes:
        for u in _nbrs(Z, Y, X, v):
            if u not in vox:
                continue
            e = _ek(v, u)
            if e in visited:
                continue
            path = _trace_path(vox, Z, Y, X, v, u, visited)
            for i in range(len(path) - 1):
                visited.add(_ek(path[i], path[i + 1]))
            polylines.append(path)

    # residual cycles (no explicit nodes on them)
    for v in list(vox):
        for u in _nbrs(Z, Y, X, v):
            if u not in vox:
                continue
            e = _ek(v, u)
            if e in visited:
                continue
            path = _trace_path(vox, Z, Y, X, v, u, visited)
            for i in range(len(path) - 1):
                visited.add(_ek(path[i], path[i + 1]))
            polylines.append(path)

    LOG.info(
        f"[POLY] polylines={len(polylines)} | nodes={len(nodes)} | vox={len(vox)}"
    )
    return polylines


def skeleton_graph_vox(
    sk: np.ndarray,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    """
    Build a sparse graph from the skeleton volume.

    Returns:
        nodes_vox: list of voxel-node coords (degree!=2)
        edges: pairs of indices into nodes_vox
    """
    if sk.ndim != 3:
        raise ValueError("sk must be 3D")
    Z, Y, X = sk.shape
    vox = set(map(tuple, np.argwhere(sk == 1)))
    if not vox:
        LOG.info("[GRAPH] empty skeleton")
        return [], []

    nodes = [v for v in vox if _degree(vox, Z, Y, X, v) != 2]
    id_of: Dict[Tuple[int, int, int], int] = {v: i for i, v in enumerate(nodes)}
    edges: Set[Tuple[int, int]] = set()

    visited: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()
    for a in nodes:
        for b in _nbrs(Z, Y, X, a):
            if b not in vox:
                continue
            path = _trace_path(vox, Z, Y, X, a, b, visited)
            s, e = path[0], path[-1]
            if s in id_of and e in id_of and s != e:
                i, j = id_of[s], id_of[e]
                edges.add((i, j) if i < j else (j, i))
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                visited.add(_ek(u, v))

    E = sorted(edges)
    LOG.info(f"[GRAPH] nodes={len(nodes)}, edges={len(E)}")
    return nodes, E


# ============================== VOX <-> WORLD ================================


def vox_path_to_world(
    path_vox: Iterable[Tuple[int, int, int]], origin: np.ndarray, v: float
) -> np.ndarray:
    """Convert [(z,y,x)] path to world XYZ using voxel centroids."""
    V = np.array(
        [[x + 0.5, y + 0.5, z + 0.5] for (z, y, x) in path_vox], dtype=float
    )
    return origin[None, :] + V[:, [0, 1, 2]] * float(v)


def node_vox_to_world(
    nodes_vox: Iterable[Tuple[int, int, int]], origin: np.ndarray, v: float
) -> np.ndarray:
    """Convert node voxel indices to world XYZ (centroids)."""
    pts = [[x + 0.5, y + 0.5, z + 0.5] for (z, y, x) in nodes_vox]
    V = np.asarray(pts, dtype=float)
    return origin[None, :] + V[:, [0, 1, 2]] * float(v)


def vox_to_world(
    path_vox: List[Tuple[int, int, int]], origin: np.ndarray, v: float
) -> np.ndarray:
    """Backward-compat wrapper kept for existing imports."""
    return vox_path_to_world(path_vox, origin, v)


# ====================== GRAPH FROM SKELETON POINTS (OPTIONAL) ===============
# These helpers follow the spirit of the paper's Sec. 3.1.


def _ensure_kdtree(P: np.ndarray):
    if _KDTree is None:
        LOG.warning("[KD] scipy not available; using brute-force distances")
        return None
    return _KDTree(P)


def radius_clustering(
    C: np.ndarray,
    r: np.ndarray,
    strength: float = 0.75,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radius-based clustering: merge nearby skeletal points by radius-weighted balls.

    Args:
        C: (N,3) centers, r: (N,) radii.
        strength: cluster radius multiplier.
    Returns:
        (C', r') reduced centers and radii.
    """
    if C.size == 0:
        return C, r
    idx = np.arange(len(C))
    if shuffle:
        rng = np.random.default_rng()
        rng.shuffle(idx)
    Cw, rw = C[idx], r[idx]

    tree = _ensure_kdtree(Cw)
    used = np.zeros(len(Cw), dtype=bool)
    out_c, out_r = [], []

    for i in range(len(Cw)):
        if used[i]:
            continue
        ci, ri = Cw[i], rw[i]
        if tree is not None:
            I = tree.query_ball_point(ci, r=strength * ri)
        else:
            d = np.linalg.norm(Cw - ci[None, :], axis=1)
            I = np.where(d <= strength * ri + 1e-9)[0].tolist()

        sel = [j for j in I if not used[j]]
        if not sel:
            used[i] = True
            out_c.append(ci)
            out_r.append(ri)
            continue
        used[sel] = True
        out_c.append(Cw[sel].mean(axis=0))
        out_r.append(rw[sel].mean())

    C2 = np.asarray(out_c, dtype=float)
    r2 = np.asarray(out_r, dtype=float)
    LOG.info(f"[RC] in={len(C)} out={len(C2)} strength={strength:.2f}")
    return C2, r2


def adaptive_graph_construction(
    C: np.ndarray,
    k: int = 5,
    m: float = 2.5,
    angle_thresh_deg: float = 75.0,
) -> List[Tuple[int, int]]:
    """
    Build undirected edges among points using KNN with distance/angle vetoes.

    For each point v, consider neighbors sorted by distance. Connect to a
    neighbor vi if dist(v,vi) <= m * d_min and direction is new w.r.t. the
    set of already accepted edge directions (angle >= angle_thresh_deg).

    Returns:
        List of undirected edges (i,j) with i<j over indices in C.
    """
    if len(C) == 0:
        return []

    tree = _ensure_kdtree(C)
    edges: Set[Tuple[int, int]] = set()
    cos_th = math.cos(math.radians(angle_thresh_deg))

    for idx in range(len(C)):
        v = C[idx]
        if tree is not None:
            dists, inds = tree.query(v, k=min(k + 1, len(C)))
            inds = np.atleast_1d(inds)
            dists = np.atleast_1d(dists)
        else:
            d = np.linalg.norm(C - v[None, :], axis=1)
            inds = np.argsort(d)[: k + 1]
            dists = d[inds]

        if len(inds) <= 1:
            continue
        dmin = float(dists[1])
        dirset: List[np.ndarray] = []

        for i in inds[1:]:
            vi = C[i]
            if np.linalg.norm(vi - v) > m * dmin:
                break
            dir_vec = vi - v
            nrm = np.linalg.norm(dir_vec)
            if nrm == 0:
                continue
            dir_vec /= nrm

            is_new = True
            for dvec in dirset:
                if float(np.dot(dir_vec, dvec)) >= cos_th:
                    is_new = False
                    break
            if not is_new:
                continue
            dirset.append(dir_vec)
            e = (idx, i) if idx < i else (i, idx)
            edges.add(e)

    E = sorted(edges)
    LOG.info(
        f"[AG] points={len(C)} edges={len(E)} k={k} m={m:.2f} "
        f"ang>={angle_thresh_deg:.1f}"
    )
    return E
