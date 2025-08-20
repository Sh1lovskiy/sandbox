# vision/skeleton/export.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np
import open3d as o3d
from utils.logger import Logger

LOG = Logger.get_logger("skexp")


def export_skeleton_model(
    out_dir: Path, nodes_xyz: np.ndarray, polylines_world: List[np.ndarray]
):
    """Minimal Azman-like JSON model with unique points and line indices."""
    out_dir.mkdir(parents=True, exist_ok=True)
    points: List[List[float]] = []
    index: dict[Tuple[float, float, float], int] = {}

    def add(p) -> int:
        k = (float(p[0]), float(p[1]), float(p[2]))
        if k in index:
            return index[k]
        idx = len(points)
        index[k] = idx
        points.append([k[0], k[1], k[2]])
        return idx

    lines: List[Tuple[int, int]] = []
    for poly in polylines_world:
        if len(poly) < 2:
            continue
        a = add(poly[0])
        for i in range(1, len(poly)):
            b = add(poly[i])
            if a != b:
                lines.append((a, b))
            a = b

    path = out_dir / "skeleton_model.json"
    path.write_text(json.dumps({"points": points, "lines": lines}, indent=2))
    LOG.info(
        f"saved skeleton_model.json with {len(points)} pts, {len(lines)} lines"
    )


def write_lineset_ply(path: Path, ls: o3d.geometry.LineSet) -> None:
    o3d.io.write_line_set(str(path), ls)


def write_mesh_ply(path: Path, mesh: o3d.geometry.TriangleMesh) -> None:
    o3d.io.write_triangle_mesh(str(path), mesh)
