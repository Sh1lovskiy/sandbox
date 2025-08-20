# apps/skel_viewer/main.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import numpy as np
import open3d as o3d

from utils.logger import Logger
from utils.error_tracker import ErrorTracker
from utils import config as ucfg

from vision.skeleton.config import SkelPipelineCfg
from vision.skeleton.voxelize import (
    compute_voxel_grid_params,
    voxelize_points,
    dilate_3d,
)
from vision.skeleton.skeleton3d import (
    skeletonize_3d,
    skeleton_nodes_vox,
    skeleton_to_polylines_3d,
    vox_to_world,
)
from vision.skeleton.skeleton2d import (
    detect_plane_and_mask,
    rasterize_points_2d,
    skeletonize_2d_img,
    px_to_world_2d,
    build_graph_from_2d_skel,
)
from vision.skeleton.nodes import (
    merge_close_nodes_world,
    refine_nodes_by_normals,
)
from vision.skeleton.refine import (
    resample_polyline_world,
    refine_polylines_centers,
    straighten_polylines,
)
from vision.skeleton.grouping import (
    estimate_dominant_plane_normal,
    group_polylines_by_angle,
)
from vision.skeleton.normals import make_normals_colored_cloud
from vision.skeleton.regions import (
    build_side_surface_clouds,
    build_side_surface_regions_v2,
)
from vision.skeleton.export import export_skeleton_model
from vision.viz.overlays import make_lineset_from_polylines, make_nodes_mesh
from vision.viz.viewer import open_view

LOG = Logger.get_logger("skel_app")


def _load_cloud(cfg: SkelPipelineCfg) -> o3d.geometry.PointCloud:
    root = Path(cfg.capture_root)
    if cfg.input_cloud_path:
        p = Path(cfg.input_cloud_path)
    else:
        dbg = root / cfg.debug_dir_name
        for name in ("final_merged.ply", "merged.ply"):
            cand = dbg / name
            if cand.exists():
                p = cand
                break
        else:
            any_ply = sorted(dbg.glob("*.ply"))
            if not any_ply:
                raise FileNotFoundError(f"No clouds under {dbg}")
            p = any_ply[0]
    LOG.info(f"load cloud: {p}")
    pc = o3d.io.read_point_cloud(str(p))
    if len(pc.points) == 0:
        raise RuntimeError("empty cloud")
    return pc


def _smooth_if_enabled(raw: o3d.geometry.PointCloud, cfg: SkelPipelineCfg):
    if not cfg.smoothing.enabled:
        return raw, None
    P0 = np.asarray(raw.points)
    P = P0.copy()
    # simple adaptive MLS inspired smoothing; small & deterministic
    # reuse local PCA plane projection with gaussian weights
    pts_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    kdt = o3d.geometry.KDTreeFlann(pts_pc)
    sigma2 = (cfg.smoothing.radius * 0.5) ** 2
    for _ in range(max(0, cfg.smoothing.iters)):
        Pn = P.copy()
        for i in range(len(P)):
            _, idx, _ = kdt.search_radius_vector_3d(P[i], cfg.smoothing.radius)
            if len(idx) < cfg.smoothing.min_nn:
                continue
            Q = P[idx]
            d = np.linalg.norm(Q - P[i], axis=1)
            w = np.exp(-(d * d) / (2.0 * sigma2)) + 1e-8
            c = (Q * w[:, None]).sum(0) / w.sum()
            A = (Q - c) * np.sqrt(w[:, None])
            try:
                _, S, Vt = np.linalg.svd(A, False)
                n = Vt[-1]
            except Exception:
                continue
            lam = (S**2) / (len(Q) - 1 + 1e-9)
            lam = np.pad(lam, (0, max(0, 3 - len(lam))), constant_values=0)
            sv = lam[-1] / (lam.sum() + 1e-12)
            w_pl = 1.0 - float(sv)
            alpha = (
                cfg.smoothing.alpha_min
                + (cfg.smoothing.alpha_max - cfg.smoothing.alpha_min) * w_pl
            )
            off = float((P[i] - c) @ n)
            Pn[i] = P[i] - alpha * off * n
        P = Pn
        pts_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        kdt = o3d.geometry.KDTreeFlann(pts_pc)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(P)
    return out, o3d.geometry.PointCloud(raw)


def _resample_all(polys: list[np.ndarray], cfg: SkelPipelineCfg):
    out = []
    for xyz in polys:
        out.append(
            resample_polyline_world(
                xyz,
                cfg.center_refine.resample_step,
                cfg.center_refine.keep_tail_min_frac,
            )
        )
    return out


def _final_centerlines(
    polys_w: list[np.ndarray],
    base_cloud: o3d.geometry.PointCloud,
    cfg: SkelPipelineCfg,
):
    ctr, _w = refine_polylines_centers(polys_w, base_cloud, cfg.center_refine)
    if cfg.center_refine.straighten:
        ctr = straighten_polylines(
            ctr,
            cfg.center_refine.straight_min_len,
            cfg.center_refine.straight_lin_thr,
            cfg.center_refine.resample_step,
        )
    return ctr


def _export_layers(
    root: Path,
    geoms: Dict[str, List],
    nodes_xyz: np.ndarray,
    center_polys: list[np.ndarray],
    cfg: SkelPipelineCfg,
):
    out = root / cfg.debug_dir_name
    out.mkdir(parents=True, exist_ok=True)

    # members
    for i, ls in enumerate(geoms.get("members", [])):
        o3d.io.write_line_set(str(out / f"members_group{i}.ply"), ls)
    # normals
    for i, c in enumerate(geoms.get("normals", [])):
        o3d.io.write_point_cloud(str(out / f"normals_angle_colored.ply"), c)
    # side regions & sides
    for i, pc in enumerate(geoms.get("side_regions", [])):
        o3d.io.write_point_cloud(str(out / f"side_region_{i:02d}.ply"), pc)
    if "sides" in geoms:
        names = ("sides_15_35", "sides_35_55", "sides_55_75", "sides_75_90")
        for name, pc in zip(names, geoms["sides"]):
            o3d.io.write_point_cloud(str(out / f"{name}.ply"), pc)
    # graph + centers + skeleton
    if geoms.get("graph"):
        o3d.io.write_triangle_mesh(
            str(out / "graph_nodes.ply"), geoms["graph"][-1]
        )
    if geoms.get("centers"):
        o3d.io.write_line_set(str(out / "centerlines.ply"), geoms["centers"][0])
    if geoms.get("skeleton"):
        o3d.io.write_line_set(
            str(out / "skeleton_raw.ply"), geoms["skeleton"][0]
        )

    export_skeleton_model(out, nodes_xyz, center_polys)


def run(cfg: SkelPipelineCfg | None = None) -> Path | None:
    Logger.configure()
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()

    cfg = cfg or SkelPipelineCfg()
    root = Path(cfg.capture_root)
    base_cloud = _load_cloud(cfg)
    LOG.info(f"points: {len(base_cloud.points)}")

    smoothed, raw = _smooth_if_enabled(base_cloud, cfg)
    base = smoothed if smoothed is not None else base_cloud

    geoms: Dict[str, List] = {
        "cloud": [],
        "skeleton": [],
        "graph": [],
        "centers": [],
        "members": [],
        "normals": [],
        "side_regions": [],
        "sides": [],
    }

    raw_col = o3d.geometry.PointCloud(base_cloud)
    raw_col.paint_uniform_color((0.60, 0.60, 0.60))
    geoms["cloud"].append(raw_col)
    if smoothed is not None:
        sc = o3d.geometry.PointCloud(smoothed)
        sc.paint_uniform_color((0.90, 0.90, 0.90))
        geoms["cloud"].append(sc)

    nodes_xyz = np.zeros((0, 3), float)
    center_polys: list[np.ndarray] = []
    ref_plane_n = None

    if cfg.mode == "plane2d":
        model, inliers = detect_plane_and_mask(base, cfg.sk2d, cfg.capture_root)
        plane_pcd = base.select_by_index(inliers.tolist(), invert=False)
        rest_pcd = base.select_by_index(inliers.tolist(), invert=True)
        plane_pcd.paint_uniform_color((0.75, 0.75, 0.75))
        rest_pcd.paint_uniform_color((0.55, 0.55, 0.55))
        geoms["cloud"] += [rest_pcd, plane_pcd]

        P = np.asarray(plane_pcd.points)
        # basis
        a, b, c, d = model
        n = np.array([a, b, c], float)
        n /= np.linalg.norm(n) + 1e-12
        ref_plane_n = n
        ref = np.array([0, 0, 1]) if abs(n[2]) < 0.9 else np.array([1, 0, 0])
        u = np.cross(n, ref)
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(n, u)
        img, uvmin = rasterize_points_2d(P, u, v, P.mean(0), cfg.sk2d.grid_res)

        # close morph
        try:
            import cv2

            k = cfg.sk2d.morph_close_k
            k = k if (k % 2) == 1 else k + 1
            img = cv2.morphologyEx(
                img, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)
            )
        except Exception:
            pass

        sk2 = skeletonize_2d_img(img, cfg.sk2d.method)
        node_px, edges2d, node_map2d, polylines_px = build_graph_from_2d_skel(
            sk2
        )

        nodes_xyz = np.array(
            [
                px_to_world_2d([rc], u, v, P.mean(0), cfg.sk2d.grid_res, uvmin)[
                    0
                ]
                for rc in node_px
            ],
            float,
        )
        nodes_xyz, _, _ = merge_close_nodes_world(
            nodes_xyz, [], cfg.node_refine.merge_radius_m
        )
        nodes_xyz = refine_nodes_by_normals(
            nodes_xyz, plane_pcd, cfg.node_refine
        )

        # polylines -> world
        polys_w = []
        for path in polylines_px:
            if len(path) < 2:
                continue
            xyz = px_to_world_2d(
                path, u, v, P.mean(0), cfg.sk2d.grid_res, uvmin
            )
            polys_w.append(xyz)
    else:
        P = np.asarray(base.points)
        origin, v, dims = compute_voxel_grid_params(
            P, cfg.vox.vsize, cfg.vox.pad, cfg.vox.max_voxels
        )
        LOG.info(f"vox origin={origin}, v={v:.4f}, dims={tuple(dims)}")
        vol = voxelize_points(P, origin, v, dims)
        vol = dilate_3d(vol, cfg.vox.dilate_r)
        sk = skeletonize_3d(vol)
        if int(sk.sum()) < cfg.vox.min_skel_voxels:
            raise RuntimeError(
                "3D skeleton too small; increase dilation or voxel."
            )
        polylines_vox = skeleton_to_polylines_3d(sk)
        nodes = skeleton_nodes_vox(sk)
        nodes_xyz = np.array(
            [vox_to_world([n], origin, v)[0] for n in nodes], float
        )
        nodes_xyz, _, _ = merge_close_nodes_world(
            nodes_xyz, [], cfg.node_refine.merge_radius_m
        )
        nodes_xyz = refine_nodes_by_normals(nodes_xyz, base, cfg.node_refine)

        polys_w = []
        for path in polylines_vox:
            if len(path) < 2:
                continue
            xyz = vox_to_world(path, origin, v)
            polys_w.append(xyz)

    # resample + centers
    polys_w = _resample_all(polys_w, cfg)
    center_polys = _final_centerlines(polys_w, base, cfg)

    ls_centers = make_lineset_from_polylines(center_polys, color=(0, 0, 1))
    geoms["centers"] = [ls_centers]
    ls_raw = make_lineset_from_polylines(polys_w, color=(0, 1, 0))
    geoms["skeleton"] = [ls_raw]
    if len(nodes_xyz) > 0:
        geoms["graph"] = [ls_raw, make_nodes_mesh(nodes_xyz, (1, 0, 0), 0.004)]

    if ref_plane_n is None:
        ref_plane_n = estimate_dominant_plane_normal(base)

    # members
    groups = group_polylines_by_angle(
        center_polys, ref_plane_n, cfg.grouping.bins_deg
    )
    from vision.viz.overlays import make_lineset_from_polylines as _mk

    palette = [(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0, 0.7, 0), (0, 0.4, 1)]
    geoms["members"] = [
        _mk(groups[i], palette[min(i, len(palette) - 1)])
        for i in range(len(groups))
    ]

    # normals & sides
    colored, angles, report = make_normals_colored_cloud(
        base, ref_plane_n, cfg.normals.angle_bins_deg, cfg.normals.palette_rgb
    )
    geoms["normals"] = [colored]
    try:
        sides, _masks = build_side_surface_clouds(
            base,
            ref_plane_n,
            cfg.normals.angle_bins_deg,
            cfg.normals.palette_rgb,
        )
        geoms["sides"] = sides
    except Exception as e:
        LOG.warning(f"side bins failed: {e}")

    try:
        regions, boundary = build_side_surface_regions_v2(
            base, ref_plane_n, cfg.regions
        )
        geoms["side_regions"] = regions + (
            [boundary] if len(boundary.points) else []
        )
    except Exception as e:
        LOG.warning(f"regions v2 failed: {e}")

    # export
    _export_layers(root, geoms, nodes_xyz, center_polys, cfg)

    # view
    open_view(geoms, cfg.viewer)
    return root / cfg.debug_dir_name / "skeleton_model.json"


def _main():
    run(SkelPipelineCfg())


if __name__ == "__main__":
    _main()
