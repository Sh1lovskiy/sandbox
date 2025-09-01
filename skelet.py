# skelet.py
"""
Mesh-plane -> 2D raster -> auto morphology -> edge-preserving smoothing ->
part contour -> geodesic-distance watershed-ridges skeleton.
Saves a single matplotlib grid with all steps to out1/ and then opens an
Open3D viewer where key 1 shows the original cleaned cloud and key 2 shows the mesh.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import (
    distance_transform_edt,
    label as nd_label,
    binary_fill_holes,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ------------------------------ config ---------------------------------------
@dataclass
class Cfg:
    cloud_path: str = (
        "/home/sha/Documents/aitech-robotics/ai-robo-sandbox/data_captures/first/debug/final_merged.ply"
    )

    # outlier removal only (no cropping)
    nb_neighbors: int = 20
    std_ratio: float = 2.0

    # Poisson and mesh cleanup
    poisson_depth: int = 9
    mesh_to_cloud_thr: float = 0.012
    aabb_expand_surf: float = 1.05
    target_tris: int = 120_000
    smooth_taubin_it: int = 30

    # plane segmentation on mesh vertices
    plane_thr: float = 0.003
    plane_ransac_n: int = 5
    plane_iters: int = 800

    # image domain
    img_res: int = 1024

    # edge-preserving smoothing for the mask
    bf_d: int = 11
    bf_sigma_color: float = 55.0
    bf_sigma_space_mul: float = 1.0

    out_dir: str = "out1"


# ------------------------------ fs utils -------------------------------------
def _ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def _save_fig(path: str, fig: plt.Figure) -> None:
    _ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------ cloud IO -------------------------------------
def load_and_clean_cloud(cfg: Cfg) -> o3d.geometry.PointCloud:
    print(f"Loading point cloud: {cfg.cloud_path}")
    pcd = o3d.io.read_point_cloud(cfg.cloud_path)
    if len(pcd.points) == 0:
        raise ValueError("Empty point cloud")
    pcd.estimate_normals()
    pcd, _ = pcd.remove_statistical_outlier(cfg.nb_neighbors, cfg.std_ratio)
    print(f"Cloud: {len(pcd.points)} pts -> after clean: {len(pcd.points)}")
    return pcd


# ------------------------------ mesh build -----------------------------------
def _crop_mesh_to_cloud_aabb(
    mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, expand: float
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0 or len(pcd.points) == 0:
        return mesh
    aabb = pcd.get_axis_aligned_bounding_box()
    c = aabb.get_center()
    half = (aabb.get_max_bound() - aabb.get_min_bound()) * 0.5 * float(expand)
    aabb_e = o3d.geometry.AxisAlignedBoundingBox(c - half, c + half)
    return mesh.crop(aabb_e)


def _prune_mesh_by_cloud_distance(
    mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, thr: float
) -> o3d.geometry.TriangleMesh:
    if len(mesh.vertices) == 0 or len(pcd.points) == 0:
        return mesh
    tree = o3d.geometry.KDTreeFlann(pcd)
    V = np.asarray(mesh.vertices, float)
    P = np.asarray(pcd.points, float)
    thr2 = float(thr) ** 2
    ok = np.zeros(len(V), bool)
    for i, v in enumerate(V):
        try:
            _, idx, _ = tree.search_knn_vector_3d(v, 1)
            q = P[idx[0]]
            ok[i] = np.sum((v - q) ** 2) <= thr2
        except Exception:
            ok[i] = False
    T = np.asarray(mesh.triangles, np.int64)
    keep_tri = ok[T].any(axis=1)
    if keep_tri.size:
        mesh.remove_triangles_by_mask((~keep_tri).tolist())
        mesh.remove_unreferenced_vertices()
    return mesh


def build_mesh(
    cfg: Cfg, pcd: o3d.geometry.PointCloud
) -> o3d.geometry.TriangleMesh:
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=cfg.poisson_depth
        )
        mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    except Exception:
        # safe fallback if Poisson fails
        r = 0.01
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([r, 2.0 * r])
        )

    mesh = _crop_mesh_to_cloud_aabb(mesh, pcd, cfg.aabb_expand_surf)
    mesh = _prune_mesh_by_cloud_distance(mesh, pcd, cfg.mesh_to_cloud_thr)
    if cfg.target_tris and len(mesh.triangles) > cfg.target_tris:
        mesh = mesh.simplify_quadric_decimation(cfg.target_tris)

    try:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=cfg.smooth_taubin_it
        )
    except Exception:
        mesh = mesh.filter_smooth_simple(
            number_of_iterations=cfg.smooth_taubin_it
        )

    mesh.compute_vertex_normals()
    print(f"Mesh: V={len(mesh.vertices)}, F={len(mesh.triangles)}")
    return mesh


# ------------------------------ plane on mesh --------------------------------
def plane_from_mesh_vertices(
    cfg: Cfg, mesh: o3d.geometry.TriangleMesh
) -> np.ndarray:
    V = np.asarray(mesh.vertices, float)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V))
    _, inliers = pc.segment_plane(
        distance_threshold=cfg.plane_thr,
        ransac_n=cfg.plane_ransac_n,
        num_iterations=cfg.plane_iters,
    )
    plane = pc.select_by_index(inliers)
    pts = np.asarray(plane.points, float)
    print(f"Plane inliers: {len(pts)}")
    if pts.size == 0:
        raise ValueError("Plane segmentation failed (0 inliers)")
    return pts


# ------------------------------ projection -----------------------------------
def project_to_image(
    points: np.ndarray, img_res: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    basis = vt[:2]
    coords_2d = (points - center) @ basis.T
    mn = coords_2d.min(axis=0)
    mx = coords_2d.max(axis=0)
    uv = (coords_2d - mn) / (mx - mn + 1e-9)
    img_xy = np.clip((uv * (img_res - 1)).astype(np.int32), 0, img_res - 1)
    mask2d = np.zeros((img_res, img_res), np.uint8)
    mask2d[img_xy[:, 1], img_xy[:, 0]] = 1
    return coords_2d, img_xy, mask2d


# ------------------------------ raster + morph -------------------------------
def auto_disk_radius_px(img_xy: np.ndarray) -> int:
    if len(img_xy) < 8:
        return 1
    pts = img_xy.astype(np.float32)
    nn = NearestNeighbors(n_neighbors=4, algorithm="kd_tree").fit(pts)
    d, _ = nn.kneighbors(pts)
    step = float(np.median(d[:, 3]))
    r = max(1, int(np.ceil(0.55 * step)))
    return min(r, 16)


def _odd(n: int) -> int:
    return int(n) | 1


def _largest_component(mask01: np.ndarray) -> np.ndarray:
    lab, num = nd_label(mask01.astype(bool))
    if int(num) <= 1:
        return mask01.astype(np.uint8)
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    keep = int(np.argmax(cnt))
    return (lab == keep).astype(np.uint8)


def rasterize_and_morph_auto(
    mask2d: np.ndarray, img_xy: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray, int, int, Dict]:
    r_disk = auto_disk_radius_px(img_xy)
    se_disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r_disk + 1, 2 * r_disk + 1)
    )
    mask_raster = cv2.dilate(mask2d, se_disk).astype(np.uint8)

    close_muls = (2.5, 3.5, 4.5, 5.5)
    open_muls = (1.0, 1.5, 2.0, 2.5)
    k_close_grid = sorted(
        {_odd(max(3, int(round(2 * r_disk * m)))) for m in close_muls}
    )
    k_open_grid = sorted(
        {_odd(max(3, int(round(2 * r_disk * m)))) for m in open_muls}
    )

    best = {"score": 1e9, "kc": 3, "ko": 3, "mask": None, "stats": None}
    base255 = (mask_raster * 255).astype(np.uint8)

    def score(mask01: np.ndarray) -> Tuple[float, Dict]:
        lab, num = nd_label(mask01.astype(bool))
        cnt = np.bincount(lab.ravel())
        cnt[0] = 0
        if cnt.size <= 1:
            return 1e9, {
                "frac_largest": 0.0,
                "holes_frac": 1.0,
                "components": int(num),
                "overgrow_frac": 1.0,
            }
        keep = int(np.argmax(cnt))
        area_tot = float(mask01.sum())
        area_lrg = float((lab == keep).sum())
        frac_largest = area_lrg / max(area_tot, 1.0)
        filled = binary_fill_holes(mask01.astype(bool)).astype(np.uint8)
        holes = (filled - mask01).sum() / max(filled.sum(), 1)
        overgrow = max(0.0, mask01.sum() / max(mask_raster.sum(), 1) - 1.0)
        s = (
            (1.0 - frac_largest)
            + 0.35 * holes
            + 0.30 * min(overgrow, 1.0)
            + 0.25 * min(max(int(num) - 1, 0) / 10.0, 1.0)
        )
        return float(s), {
            "frac_largest": frac_largest,
            "holes_frac": float(holes),
            "components": int(num),
            "overgrow_frac": float(min(overgrow, 1.0)),
        }

    for kc in k_close_grid:
        se_c = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (21, 21)
        )  # fixed: use kc
        img_c = cv2.morphologyEx(base255, cv2.MORPH_CLOSE, se_c)
        for ko in k_open_grid:
            se_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            img_o = cv2.morphologyEx(img_c, cv2.MORPH_OPEN, se_o)
            m = (img_o > 32).astype(np.uint8)
            m = binary_fill_holes(m.astype(bool)).astype(np.uint8)
            sc, st = score(m)
            if sc < best["score"]:
                best.update(score=sc, kc=kc, ko=ko, mask=m, stats=st)
            if (
                st["frac_largest"] > 0.97
                and st["holes_frac"] < 0.02
                and st["components"] <= 2
            ):
                break

    final = _largest_component(best["mask"])
    final = binary_fill_holes(final.astype(bool)).astype(np.uint8)
    diagnostics = {
        "r_disk": int(r_disk),
        "k_close": int(best["kc"]),
        "k_open": int(best["ko"]),
        "score": float(best["score"]),
        **(best["stats"] or {}),
    }
    print("[auto-morph]", diagnostics)
    return (
        r_disk,
        mask_raster,
        final,
        int(best["kc"]),
        int(best["ko"]),
        diagnostics,
    )


# ------------------------------ contour --------------------------------------
def find_part_contour(mask01: np.ndarray) -> np.ndarray:
    comp = _largest_component(mask01.astype(np.uint8))
    comp = binary_fill_holes(comp.astype(bool)).astype(np.uint8)
    contour = (cv2.Canny((comp * 255).astype(np.uint8), 50, 100) > 0).astype(
        np.uint8
    )
    if contour.sum() == 0:
        from skimage.segmentation import find_boundaries

        contour = find_boundaries(comp.astype(bool), mode="outer").astype(
            np.uint8
        )
    return contour


# ------------------------------ skeleton -------------------------------------
def geodesic_watershed_skeleton(
    mask01: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from skimage.segmentation import watershed, find_boundaries

    m = mask01.astype(bool)
    contour = find_boundaries(m, mode="outer").astype(np.uint8)
    edges = (cv2.Canny((m.astype(np.uint8) * 255), 80, 160) > 0).astype(
        np.uint8
    )
    inv_edges = ~edges.astype(bool)
    edt = distance_transform_edt(inv_edges).astype(np.float32)
    geo = np.zeros_like(edt, np.float32)
    geo[m] = edt[m]
    elev = geo + contour.astype(np.float32) * 1e-3
    markers, _ = nd_label(m.astype(np.uint8))
    labels = watershed(elev, markers=markers, mask=m)
    ridges = find_boundaries(labels, mode="inner").astype(np.uint8)
    ridges[contour.astype(bool)] = 0
    return ridges.astype(np.uint8), contour, geo


# ------------------------------ grid plot ------------------------------------
def save_grid(
    cfg: Cfg,
    coords_2d: np.ndarray,
    mask2d: np.ndarray,
    r_disk: int,
    mask_raster: np.ndarray,
    mask_morph: np.ndarray,
    mask_smooth: np.ndarray,
    contour: np.ndarray,
    geo_dist: np.ndarray,
    skeleton: np.ndarray,
) -> None:
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Mesh-plane -> 2D -> Morph -> Skeleton", fontsize=13)

    axs[0, 0].scatter(coords_2d[:, 0], coords_2d[:, 1], s=1, c="k")
    axs[0, 0].set_title("Projected points")
    axs[0, 0].axis("equal")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(mask2d, cmap="gray")
    axs[0, 1].set_title("Raw mask2d")
    axs[0, 1].axis("off")
    axs[0, 2].imshow(mask_raster, cmap="gray")
    axs[0, 2].set_title(f"Raster (r={r_disk}px)")
    axs[0, 2].axis("off")
    axs[0, 3].imshow(mask_morph, cmap="gray")
    axs[0, 3].set_title("Auto morphology")
    axs[0, 3].axis("off")

    im = axs[1, 0].imshow(geo_dist, cmap="magma")
    axs[1, 0].set_title("Geodesic distance")
    axs[1, 0].axis("off")
    fig.colorbar(im, ax=axs[1, 0], shrink=0.8)

    axs[1, 1].imshow(mask_smooth, cmap="gray")
    axs[1, 1].set_title("Edge-preserving mask")
    axs[1, 1].axis("off")
    axs[1, 2].imshow(contour, cmap="gray")
    axs[1, 2].set_title("Part contour")
    axs[1, 2].axis("off")

    overlay = np.stack(
        [skeleton * 255, mask_smooth * 255, np.zeros_like(mask_smooth)], axis=-1
    ).astype(np.uint8)
    axs[1, 3].imshow(overlay)
    axs[1, 3].set_title("Skeleton over mask")
    axs[1, 3].axis("off")

    _save_fig(os.path.join(cfg.out_dir, "mesh_plane_grid.png"), fig)


# ------------------------------ run pipeline ---------------------------------
def run(cfg: Cfg) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    _ensure_dir(cfg.out_dir)

    pcd = load_and_clean_cloud(cfg)
    mesh = build_mesh(cfg, pcd)
    print("[viewer] press 1 for cloud, 2 for mesh after save")

    plane_pts = plane_from_mesh_vertices(cfg, mesh)

    coords_2d, img_xy, mask2d = project_to_image(plane_pts, cfg.img_res)
    r_disk, mask_raster, mask_morph, k_close, k_open, diag = (
        rasterize_and_morph_auto(mask2d, img_xy)
    )

    sigma_space = max(1.0, cfg.bf_sigma_space_mul * r_disk)
    m255 = (mask_morph * 255).astype(np.uint8)
    m255 = cv2.bilateralFilter(
        m255,
        d=cfg.bf_d,
        sigmaColor=cfg.bf_sigma_color,
        sigmaSpace=float(sigma_space),
    )
    mask_smooth = (m255 > 32).astype(np.uint8)
    mask_smooth = binary_fill_holes(mask_smooth.astype(bool)).astype(np.uint8)

    contour = find_part_contour(mask_smooth)

    skeleton, _contour_for_skel, geo = geodesic_watershed_skeleton(mask_smooth)

    save_grid(
        cfg,
        coords_2d,
        mask2d,
        r_disk,
        mask_raster,
        mask_morph,
        mask_smooth,
        contour,
        geo,
        skeleton,
    )

    # save simple layers for quick inspection
    for name, arr, cmap in [
        ("mask2d", mask2d, "gray"),
        ("mask_raster", mask_raster, "gray"),
        ("mask_morph", mask_morph, "gray"),
        ("mask_smooth", mask_smooth, "gray"),
        ("contour", contour, "gray"),
        ("skeleton", skeleton, "gray"),
    ]:
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(arr, cmap=cmap)
        plt.axis("off")
        _save_fig(os.path.join(cfg.out_dir, f"{name}.png"), fig)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(geo, cmap="magma")
    plt.axis("off")
    _save_fig(os.path.join(cfg.out_dir, "geo.png"), fig)

    print(f"Saved grid and layers to: {os.path.abspath(cfg.out_dir)}")
    return pcd, mesh


# ------------------------------ viewer ---------------------------------------
def view_cloud_and_mesh(
    pcd: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh
) -> None:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Cloud/Mesh viewer", width=1280, height=900)

    cloud_show = o3d.geometry.PointCloud(pcd)
    mesh_show = o3d.geometry.TriangleMesh(mesh)

    state = {"mode": 1}  # 1=cloud, 2=mesh

    def redraw():
        vis.clear_geometries()
        if state["mode"] == 1:
            vis.add_geometry(cloud_show, reset_bounding_box=True)
        else:
            vis.add_geometry(mesh_show, reset_bounding_box=True)
        vis.update_renderer()

    def set_cloud(_):
        state["mode"] = 1
        redraw()
        return True

    def set_mesh(_):
        state["mode"] = 2
        redraw()
        return True

    redraw()
    vis.register_key_callback(ord("1"), set_cloud)
    vis.register_key_callback(ord("2"), set_mesh)
    vis.run()
    vis.destroy_window()


# ------------------------------ main -----------------------------------------
def main():
    cfg = Cfg()
    pcd, mesh = run(cfg)
    view_cloud_and_mesh(pcd, mesh)


if __name__ == "__main__":
    main()
