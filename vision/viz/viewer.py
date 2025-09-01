# vision/viz/viewer.py
from __future__ import annotations

from typing import Dict, List
import numpy as np
import open3d as o3d

from utils.logger import Logger
from .hotkeys import register as register_hotkeys
from .overlays import build_side_region_meshes

from vision.skeleton.config import ViewerCfg

LOG = Logger.get_logger("viewer")


def _set_point_size(vis: o3d.visualization.Visualizer, size: float) -> None:
    try:
        opt = vis.get_render_option()
        opt.point_size = float(size)
        opt.mesh_show_back_face = True
        opt.light_on = True
    except Exception:
        pass


def _toggle_group(vis, geoms, visible, name: str) -> None:
    cur = visible.get(name, False)
    visible[name] = not cur
    geoms.setdefault(name, [])
    if visible[name]:
        for g in geoms[name]:
            vis.add_geometry(g, reset_bounding_box=False)
    else:
        for g in geoms[name]:
            vis.remove_geometry(g, reset_bounding_box=False)
    LOG.info(f"toggle {name} -> {visible[name]}")
    vis.update_renderer()


def _toggle_many(vis, geoms, visible, names: list[str]) -> None:
    turn_on = not any(visible.get(n, False) for n in names)
    for n in names:
        for g in geoms.get(n, []):
            vis.remove_geometry(g, reset_bounding_box=False)
        visible[n] = False
    if turn_on:
        for n in names:
            for g in geoms.get(n, []):
                vis.add_geometry(g, reset_bounding_box=False)
            visible[n] = True
    vis.update_renderer()
    LOG.info(f"toggle bundle {names} -> {'ON' if turn_on else 'OFF'}")


def _install_shift_click_pick(vis, base_cloud):
    if not hasattr(vis, "register_mouse_callback"):
        return

    def _mouse_cb(action):
        if (
            getattr(action, "button", None) == 1
            and getattr(action, "type", None) == 1
            and getattr(action, "mods", None)
            and getattr(action.mods, "is_shift", False)
        ):
            if hasattr(vis, "pick_point"):
                idx = vis.pick_point(action.x, action.y)
                if 0 <= idx < len(base_cloud.points):
                    p = np.asarray(base_cloud.points)[idx]
                    LOG.info(f"Picked XYZ = {p.tolist()}")
        return True

    try:
        vis.register_mouse_callback(_mouse_cb)
    except Exception:
        pass


def _open_edit_picker_print_xyz(pc: o3d.geometry.PointCloud) -> None:
    if len(pc.points) == 0:
        return
    vv = o3d.visualization.VisualizerWithEditing()
    vv.create_window(window_name="Pick: double-click", width=900, height=600)
    vv.add_geometry(pc)
    vv.run()
    vv.destroy_window()
    picks = vv.get_picked_points()
    P = np.asarray(pc.points)
    for i in picks:
        if 0 <= i < len(P):
            LOG.info(f"Picked XYZ = {P[i].tolist()}")


def _ensure_side_mesh_layer(geoms: Dict[str, List], cfg: ViewerCfg) -> None:
    if "side_regions_mesh" in geoms:
        return
    regions = geoms.get("side_regions", [])
    geoms["side_regions_mesh"] = build_side_region_meshes(
        regions,
        prefer_poisson=False,
        target=cfg.mesh_decimate_target,
    )


def open_view(geoms: Dict[str, List], cfg: ViewerCfg) -> None:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=(
            f"{cfg.title_prefix} - "
            "1 cloud  3 2D-skel+nodes  4 centers  5 members  "
            "6 normals  7 regions  8 3D-skel+nodes  9 side-mesh  P pick"
        ),
        width=cfg.window[0],
        height=cfg.window[1],
    )

    base_cloud = next(
        (
            g
            for g in geoms.get("cloud", [])
            if isinstance(g, o3d.geometry.PointCloud)
        ),
        o3d.geometry.PointCloud(),
    )
    for g in geoms.get("cloud", []):
        vis.add_geometry(g)
    _set_point_size(vis, cfg.point_size)
    _install_shift_click_pick(vis, base_cloud)

    visible = {
        "cloud": True,
        "skeleton": False,
        "graph": False,
        "skel3d": False,
        "centers": False,
        "members": False,
        "normals": False,
        "side_regions": False,
        "sides_bins": False,
        "side_regions_mesh": False,
    }

    def t(name: str):
        return lambda: _toggle_group(vis, geoms, visible, name)

    mapping = {
        "1": t("cloud"),
        "2": t("skeleton"),
        "3": lambda: (
            _toggle_group(vis, geoms, visible, "skeleton"),
            _toggle_group(vis, geoms, visible, "graph"),
        ),
        "4": t("centers"),
        "5": t("members"),
        "6": t("normals"),
        "7": t("side_regions"),
        "8": t("sides_bins"),
        "9": (
            lambda: (
                _ensure_side_mesh_layer(geoms, cfg),
                _toggle_group(vis, geoms, visible, "side_regions_mesh"),
            )
        ),
        "P": lambda: _open_edit_picker_print_xyz(base_cloud),
    }
    register_hotkeys(vis, mapping)
    LOG.info("Viewer ready (Shift+LeftClick or 'P' to pick XYZ).")
    vis.run()
    vis.destroy_window()
