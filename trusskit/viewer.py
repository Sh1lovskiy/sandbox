"""Open3D interactive viewer for the truss pipeline."""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import open3d as o3d

from utils.logger import Logger
from utils.keyboard import KeyHandler

LOG = Logger.get_logger("tk.viewer")


def _toggle_group(
    vis: o3d.visualization.Visualizer, geoms: Dict[str, List], visible: Dict[str, bool], name: str
) -> None:
    cur = visible.get(name, False)
    visible[name] = not cur
    for g in geoms.get(name, []):
        try:
            if visible[name]:
                vis.add_geometry(g, reset_bounding_box=False)
            else:
                vis.remove_geometry(g, reset_bounding_box=False)
        except Exception:
            pass
    vis.update_renderer()
    LOG.info(f"toggle {name} -> {visible[name]}")


def open_view(
    cloud: o3d.geometry.PointCloud,
    build_mesh: Callable[[], o3d.geometry.TriangleMesh],
    skeleton_ls: o3d.geometry.LineSet,
    graph_nodes: o3d.geometry.TriangleMesh,
    graph_edges: o3d.geometry.LineSet,
    nodes_xyz: np.ndarray,
    navigator,
    save_cb: Callable[[], None],
) -> None:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="TrussKit", width=1024, height=768)
    vis.add_geometry(cloud)
    geoms = {
        "cloud": [cloud],
        "mesh": [],
        "skeleton": [skeleton_ls],
        "graph": [graph_nodes, graph_edges],
        "highlight": [],
    }
    visible = {"cloud": True, "mesh": False, "skeleton": False, "graph": False}
    highlight = o3d.geometry.LineSet()
    geoms["highlight"].append(highlight)

    mesh_cache: List[o3d.geometry.TriangleMesh] = []

    def _ensure_mesh() -> None:
        if mesh_cache:
            return
        mesh = build_mesh()
        mesh_cache.append(mesh)
        geoms["mesh"] = [mesh]

    def t(name: str) -> Callable[[], None]:
        return lambda: _toggle_group(vis, geoms, visible, name)

    def toggle_mesh() -> None:
        _ensure_mesh()
        if visible["mesh"]:
            _toggle_group(vis, geoms, visible, "mesh")
            if not visible["cloud"]:
                _toggle_group(vis, geoms, visible, "cloud")
        else:
            if visible["cloud"]:
                _toggle_group(vis, geoms, visible, "cloud")
            _toggle_group(vis, geoms, visible, "mesh")

    def toggle_graph() -> None:
        _toggle_group(vis, geoms, visible, "graph")
        if visible["graph"]:
            if highlight not in geoms["graph"]:
                geoms["graph"].append(highlight)
                vis.add_geometry(highlight, reset_bounding_box=False)
            update_highlight(navigator.current())
        else:
            if highlight in geoms["graph"]:
                vis.remove_geometry(highlight, reset_bounding_box=False)
                geoms["graph"].remove(highlight)

    def update_highlight(edge) -> None:
        if edge is None:
            return
        u, v = edge
        pts = nodes_xyz[[u, v]]
        highlight.points = o3d.utility.Vector3dVector(pts)
        highlight.lines = o3d.utility.Vector2iVector([[0, 1]])
        highlight.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])
        vis.update_geometry(highlight)

    def step_prev() -> None:
        if not visible["graph"]:
            toggle_graph()
        update_highlight(navigator.prev())

    def step_next() -> None:
        if not visible["graph"]:
            toggle_graph()
        update_highlight(navigator.next())

    mapping = {
        "2": toggle_mesh,
        "3": t("skeleton"),
        "4": toggle_graph,
        "[": step_prev,
        "]": step_next,
        "S": save_cb,
    }
    KeyHandler(vis).register(mapping)
    vis.run()
    vis.destroy_window()
