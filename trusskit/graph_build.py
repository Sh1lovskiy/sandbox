"""Graph construction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from utils.logger import Logger


@dataclass
class Graph:
    nodes: np.ndarray
    edges: List[Tuple[int, int]]
    polylines: List[np.ndarray]
    lengths: List[float]
    orientations: List[np.ndarray]


def build(
    nodes_xyz: np.ndarray, edges: List[Tuple[int, int]], polys: List[np.ndarray]
) -> Graph:
    """Assemble graph with basic edge attributes."""
    LOG = Logger.get_logger("tk.graph")
    LOG.info(
        f"build graph: {len(nodes_xyz)} nodes {len(edges)} edges polylines={len(polys)}"
    )
    lengths, ori = [], []
    for poly in polys:
        if len(poly) < 2:
            lengths.append(0.0)
            ori.append(np.zeros(3))
            continue
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        lengths.append(float(seg.sum()))
        vec = poly[-1] - poly[0]
        norm = np.linalg.norm(vec) + 1e-12
        ori.append(vec / norm)
    LOG.info("graph build done")
    return Graph(nodes_xyz, edges, polys, lengths, ori)
