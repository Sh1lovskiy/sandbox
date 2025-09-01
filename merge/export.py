# merge/export.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d

from utils.logger import Logger

LOG = Logger.get_logger("export")


def save_cloud(
    cloud: o3d.geometry.PointCloud, root: Path, name: str = "final_merged.ply"
) -> Path:
    """Save cloud under <root>/debug/<name>."""
    out_dir = root / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    ok = o3d.io.write_point_cloud(str(out_path), cloud)
    n_final = len(cloud.points)
    if ok:
        LOG.info(f"Save wrote {n_final} points to {out_path}")
    else:
        LOG.warning(f"Save failed: {out_path}")
    return out_path


def _write_picks_bundle(
    picks: List[int],
    cloud: o3d.geometry.PointCloud,
    out_dir: Path,
    prefix: str,
    sphere_r: float,
) -> None:
    P = np.asarray(cloud.points)
    coords = [P[i].tolist() for i in picks] if picks else []
    (out_dir / f"{prefix}_picks.json").write_text(
        json.dumps(
            {"count": len(picks), "indices": picks, "points_xyz": coords},
            indent=2,
        )
    )

    spheres, lines = [], []
    for i, pi in enumerate(picks):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r)
        s.translate(P[pi])
        s.compute_vertex_normals()
        s.paint_uniform_color((1, 0.25, 0))
        spheres.append(s)
        if i > 0:
            lines.append([i - 1, i])

    markers = o3d.geometry.TriangleMesh()
    for s in spheres:
        markers += s
    o3d.io.write_triangle_mesh(
        str(out_dir / f"{prefix}_pick_markers.ply"), markers
    )
    if len(picks) >= 2:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector([P[i] for i in picks])
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(lines))
        o3d.io.write_line_set(str(out_dir / f"{prefix}_pick_polyline.ply"), ls)


def interactive_picks_and_save(
    cloud: o3d.geometry.PointCloud, root: Path, prefix: str
) -> None:
    """Open editing viewer, save picks bundle to <root>/debug."""
    out_dir = root / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Final edit & save picks", width=1280, height=800
    )
    vis.add_geometry(cloud)
    vis.run()
    vis.destroy_window()
    picks = vis.get_picked_points()
    LOG.info(f"Picks {len(picks)} points")
    _write_picks_bundle(picks, cloud, out_dir, prefix, sphere_r=0.004)
