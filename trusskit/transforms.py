"""Coordinate transform utilities."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PlaneFrame:
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    p0: np.ndarray

    def world_to_plane(self, pts: np.ndarray) -> np.ndarray:
        q = pts - self.p0[None, :]
        return np.stack([q @ self.u, q @ self.v], 1)

    def plane_to_world(self, uv: np.ndarray) -> np.ndarray:
        return self.p0[None, :] + uv[:, 0:1] * self.u[None, :] + uv[:, 1:2] * self.v[None, :]
