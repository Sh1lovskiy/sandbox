"""I/O helpers for clouds, skeletons, and graphs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d

from utils.logger import Logger

LOG = Logger.get_logger("tk.io")


def load_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from PLY."""
    pc = o3d.io.read_point_cloud(path)
    if len(pc.points) == 0:
        raise RuntimeError(f"empty cloud at {path}")
    LOG.info(f"loaded cloud: {path}")
    return pc


def save_skeleton(polys: List[np.ndarray], path: Path) -> None:
    """Save skeleton polylines as JSON of lists of points."""
    data = [[[float(c) for c in p] for p in poly] for poly in polys]
    path.write_text(json.dumps(data))
    LOG.info(f"saved skeleton: {path}")


def save_graph(nodes: np.ndarray, edges: List[Tuple[int, int]], path: Path) -> None:
    """Save graph nodes and edges as JSON."""
    data = {
        "nodes": [[float(c) for c in p] for p in nodes],
        "edges": [[int(u), int(v)] for (u, v) in edges],
    }
    path.write_text(json.dumps(data))
    LOG.info(f"saved graph: {path}")
