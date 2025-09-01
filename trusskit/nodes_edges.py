"""Node and edge detection from skeleton."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from utils.logger import Logger
from vision.skeleton.skeleton2d import build_graph_from_2d_skel
from vision.skeleton.nodes import merge_close_nodes_world

LOG = Logger.get_logger("tk.nodes")


def detect(sk: np.ndarray):
    """Extract nodes, edges, polylines in pixel space."""
    LOG.info(f"skeleton pixels={int(sk.sum())}")
    nodes, edges, _, polys = build_graph_from_2d_skel(sk)
    LOG.info(f"raw nodes={len(nodes)} edges={len(edges)}")
    return nodes, edges, polys


def merge_world(
    nodes_xyz: np.ndarray, edges: List[Tuple[int, int]], radius: float
) -> tuple[np.ndarray, List[Tuple[int, int]]]:
    """Cluster close nodes in world coordinates."""
    LOG.info(
        f"merge nodes: {len(nodes_xyz)} nodes {len(edges)} edges r={radius:.3f}"
    )
    new_nodes, new_edges, _ = merge_close_nodes_world(nodes_xyz, edges, radius)
    LOG.info(
        f"after merge: {len(new_nodes)} nodes {len(new_edges)} edges"
    )
    return new_nodes, new_edges
