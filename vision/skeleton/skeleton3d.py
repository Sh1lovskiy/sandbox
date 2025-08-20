# vision/skeleton/skeleton3d.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from utils.logger import Logger

LOG = Logger.get_logger("sk3d")


def skeletonize_3d(vol: np.ndarray) -> np.ndarray:
    """3D skeleton; fall back to 2D-slices if skimage lacks 3D."""
    try:
        from skimage.morphology import skeletonize_3d as _sk3d

        return _sk3d(vol.astype(bool)).astype(np.uint8)
    except Exception:
        try:
            from skimage.morphology import skeletonize as _sk2d

            out = np.zeros_like(vol)
            for z in range(vol.shape[0]):
                out[z] = _sk2d(vol[z].astype(bool)).astype(np.uint8)
            return out
        except Exception as e:
            raise RuntimeError(f"skel3d unavailable: {e}") from e


def _nbrs(Z: int, Y: int, X: int, v: tuple[int, int, int]):
    z, y, x = v
    offs = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]
    for dz, dy, dx in offs:
        w = (z + dz, y + dy, x + dx)
        if 0 <= w[0] < Z and 0 <= w[1] < Y and 0 <= w[2] < X:
            yield w


def skeleton_nodes_vox(sk: np.ndarray) -> List[tuple[int, int, int]]:
    """Vox indices with degree != 2."""
    Z, Y, X = sk.shape
    vox = set(map(tuple, np.argwhere(sk == 1)))
    if not vox:
        return []
    out = []
    for v in vox:
        d = sum((w in vox) for w in _nbrs(Z, Y, X, v))
        if d != 2:
            out.append(v)
    return out


def skeleton_to_polylines_3d(
    sk: np.ndarray,
) -> List[list[tuple[int, int, int]]]:
    """Trace polyline voxel paths between nodes and around cycles."""
    Z, Y, X = sk.shape
    vox = set(map(tuple, np.argwhere(sk == 1)))
    if not vox:
        return []
    visited = set()

    def deg(v):  # degree
        return sum((w in vox) for w in _nbrs(Z, Y, X, v))

    def ek(a, b):
        return (a, b) if a < b else (b, a)

    def trace(a, b):
        path = [a, b]
        prv, cur = a, b
        while True:
            nb = [w for w in _nbrs(Z, Y, X, cur) if w in vox and w != prv]
            if deg(cur) != 2 or not nb:
                break
            nxt = nb[0]
            if ek(cur, nxt) in visited:
                break
            path.append(nxt)
            prv, cur = cur, nxt
        return path

    nodes = [v for v in vox if deg(v) != 2]
    polylines = []
    for v in nodes:
        for u in _nbrs(Z, Y, X, v):
            if u not in vox:
                continue
            e = ek(v, u)
            if e in visited:
                continue
            path = trace(v, u)
            for i in range(len(path) - 1):
                visited.add(ek(path[i], path[i + 1]))
            polylines.append(path)

    # cycles
    for v in list(vox):
        for u in _nbrs(Z, Y, X, v):
            if u not in vox:
                continue
            e = ek(v, u)
            if e in visited:
                continue
            path = trace(v, u)
            for i in range(len(path) - 1):
                visited.add(ek(path[i], path[i + 1]))
            polylines.append(path)
    return polylines


def vox_to_world(
    path_vox: list[tuple[int, int, int]],
    origin: np.ndarray,
    v: float,
) -> np.ndarray:
    """Convert [(z,y,x)] -> world XYZ with voxel centroids."""
    V = np.array([[x + 0.5, y + 0.5, z + 0.5] for (z, y, x) in path_vox])
    return origin[None, :] + V[:, [0, 1, 2]] * v
