"""TrussKit pipeline orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from utils.logger import Logger
from utils.error_tracker import ErrorTracker

from . import config as cfgmod
from . import io, plane, project2d, skeleton2d, nodes_edges, graph_build, graph_traverse, mesh, viewer
from .transforms import PlaneFrame

from vision.viz import overlays

LOG = Logger.get_logger("tk.pipeline")


def run(
    *,
    cloud_path: str,
    merge_node_radius: float,
    raster_res_px: int,
    save_tag: str,
) -> None:
    cfg = cfgmod.RunConfig(cloud_path, merge_node_radius, raster_res_px, save_tag)

    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()
    ErrorTracker.install_keyboard_listener()

    cloud = io.load_cloud(cfg.cloud_path)

    model, inliers, u, v, n, p0 = plane.find_dominant_plane(cloud)
    frame = PlaneFrame(u, v, n, p0)
    pts = np.asarray(cloud.points)[inliers]
    uv = frame.world_to_plane(pts)
    umin, vmin = uv.min(0)
    umax, vmax = uv.max(0)
    span = max(umax - umin, vmax - vmin)
    res = span / float(cfg.raster_res_px)
    img, uv_min = project2d.rasterize(pts, u, v, p0, res)
    LOG.info(f"raster image: {img.shape}")

    sk = skeleton2d.skeletonize(img)
    nodes_rc, edges, polys_px = nodes_edges.detect(sk)
    nodes_xyz = project2d.pixels_to_world(nodes_rc, u, v, p0, res, uv_min)
    polys_xyz: List[np.ndarray] = [
        project2d.pixels_to_world(path, u, v, p0, res, uv_min) for path in polys_px
    ]
    nodes_xyz, edges = nodes_edges.merge_world(nodes_xyz, edges, cfg.merge_node_radius)

    graph = graph_build.build(nodes_xyz, edges, polys_xyz)
    LOG.info(
        f"graph summary: {graph.nodes.shape[0]} nodes {len(graph.edges)} edges"
    )
    navigator = graph_traverse.EdgeNavigator(graph.edges)

    skel_ls = overlays.make_lineset_from_polylines(polys_xyz, color=(0.0, 0.0, 1.0))
    edge_polys = [graph.nodes[[u, v]] for (u, v) in graph.edges]
    graph_ls = overlays.make_lineset_from_polylines(edge_polys, color=(1.0, 0.5, 0.0))
    node_mesh = overlays.make_nodes_mesh(graph.nodes, color=(1.0, 0.0, 0.0))

    def save_outputs() -> None:
        io.save_skeleton(polys_xyz, Path(f"{cfg.save_tag}_skeleton_2d.json"))
        io.save_graph(graph.nodes, graph.edges, Path(f"{cfg.save_tag}_graph.json"))

    viewer.open_view(
        cloud,
        lambda: mesh.build_mesh(cloud),
        skel_ls,
        node_mesh,
        graph_ls,
        graph.nodes,
        navigator,
        save_outputs,
    )
