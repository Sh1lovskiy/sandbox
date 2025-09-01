"""2D projection helpers."""
from __future__ import annotations

import numpy as np

from vision.skeleton.skeleton2d import rasterize_points_2d, px_to_world_2d


def rasterize(points: np.ndarray, u: np.ndarray, v: np.ndarray, p0: np.ndarray, res: float):
    return rasterize_points_2d(points, u, v, p0, res)


def pixels_to_world(
    path: list[tuple[int, int]],
    u: np.ndarray,
    v: np.ndarray,
    p0: np.ndarray,
    res: float,
    uv_min: tuple[float, float],
) -> np.ndarray:
    return px_to_world_2d(path, u, v, p0, res, uv_min)
