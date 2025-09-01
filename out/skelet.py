# skelet.py
# -*- coding: utf-8 -*-
"""
Cloud -> (A) RAW plane pipeline, (B) MESH plane pipeline.
2D rasterization with auto radius -> auto morphology ->
geodesic distance + watershed ridges skeleton (no thinning libs).
All figures saved to 'out/' (no GUI popups).
Open3D hotkeys: '1' show raw cloud, '2' show mesh (and run mesh pipeline -> out/).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import distance_transform_edt, label as nd_label
from skimage.segmentation import watershed, find_boundaries


# ----------------------------- config -------------------------------------- #


@dataclass
class Params:
    cloud_path: str = (
        "/home/sha/Documents/aitech-robotics/ai-robo-sandbox/data_captures/first/debug/final_merged.ply"
    )

    # cloud denoise
    nb_neighbors: int = 20
    std_ratio: float = 2.0

    # plane (RANSAC)
    plane_thr: float = 0.04
    plane_ransac_n: int = 5
    plane_iters: int = 500

    # image domain
    img_res: int = 1024

    # mesh recon (as in your snippet)
    VOXEL_SIZE: float = 0.005
    NB_NEIGHBORS: int = 20
    STD_RATIO: float = 2.0

    POISSON_DEPTH: int = 9
    BPA_RADIUS: float = 0.01
    TARGET_TRIS: int = 120_000
    AABB_EXPAND_SURF: float = 1.05
    MESH_TO_CLOUD_THR: float = 0.012
    SMOOTH_TAUBIN_IT: int = 30

    out_dir: str = "out"


# ----------------------------- helpers ------------------------------------- #


def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_grid_png(path: str, fig):
    ensure_outdir(os.path.dirname(path))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- IO & clean ---------------------------------- #


def load_and_clean_cloud(p: Params) -> o3d.geometry.PointCloud:
    print(f"Loading point cloud: {p.cloud_path}")
    pcd = o3d.io.read_point_cloud(p.cloud_path)
    if len(pcd.points) == 0:
        raise ValueError("Loaded point cloud is empty")
    pcd.estimate_normals()
    pcd, _ = pcd.remove_statistical_outlier(p.nb_neighbors, p.std_ratio)
    print(f"Point cloud: {len(pcd.points)} pts after clean")
    return pcd


# ----------------------------- mesh (your method) -------------------------- #


def _ensure_normals(pcd: o3d.geometry.PointCloud, vox: float):
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 3.0, max_nn=22)
        )
        pcd.orient_normals_consistent_tangent_plane(50)


def preprocess_cloud_for_mesh(
    raw: o3d.geometry.PointCloud, p: Params
) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud(raw)
    if len(pc.points) == 0:
        return pc
    pc = pc.voxel_down_sample(p.VOXEL_SIZE)
    pc, _ = pc.remove_statistical_outlier(p.NB_NEIGHBORS, p.STD_RATIO)
    _ensure_normals(pc, p.VOXEL_SIZE)
    return pc


def _crop_mesh_to_cloud_aabb(
    mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, expand: float
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0 or len(pcd.points) == 0:
        return mesh
    aabb = pcd.get_axis_aligned_bounding_box()
    c = aabb.get_center()
    half = (aabb.get_max_bound() - aabb.get_min_bound()) * 0.5 * float(expand)
    return mesh.crop(o3d.geometry.AxisAlignedBoundingBox(c - half, c + half))


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
        except Exception:
            continue
        ok[i] = np.sum((v - P[idx[0]]) ** 2) <= thr2
    T = np.asarray(mesh.triangles, np.int64)
    keep_tri = ok[T].any(axis=1)
    mesh.remove_triangles_by_mask((~keep_tri).tolist())
    mesh.remove_unreferenced_vertices()
    return mesh


def _keep_largest_component(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    labels, _, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels, np.int64)
    if labels.size == 0:
        return mesh
    keep = int(np.argmax(np.bincount(labels)))
    mesh.remove_triangles_by_mask((labels != keep).tolist())
    mesh.remove_unreferenced_vertices()
    return mesh


def reconstruct_and_smooth_mesh(
    pcd: o3d.geometry.PointCloud, p: Params
) -> o3d.geometry.TriangleMesh:
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=p.POISSON_DEPTH
        )
        mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
        if len(mesh.triangles) == 0:
            raise RuntimeError("Empty Poisson mesh")
    except Exception:
        r = p.BPA_RADIUS
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([r, 2.0 * r])
        )
    mesh = _crop_mesh_to_cloud_aabb(mesh, pcd, p.AABB_EXPAND_SURF)
    mesh = _keep_largest_component(mesh)
    mesh = _prune_mesh_by_cloud_distance(mesh, pcd, p.MESH_TO_CLOUD_THR)
    mesh = _keep_largest_component(mesh)
    if p.TARGET_TRIS and len(mesh.triangles) > p.TARGET_TRIS:
        mesh = mesh.simplify_quadric_decimation(p.TARGET_TRIS)
    try:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=p.SMOOTH_TAUBIN_IT
        )
    except Exception:
        mesh = mesh.filter_smooth_simple(
            number_of_iterations=p.SMOOTH_TAUBIN_IT
        )
    mesh.compute_vertex_normals()
    return mesh


# ----------------------------- plane & PCA --------------------------------- #


def plane_from_points_array(
    points_xyz: np.ndarray, p: Params
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_xyz))
    _, inliers = pc.segment_plane(
        distance_threshold=p.plane_thr,
        ransac_n=p.plane_ransac_n,
        num_iterations=p.plane_iters,
    )
    plane = pc.select_by_index(inliers)
    pts = np.asarray(plane.points)
    if pts.size == 0:
        raise ValueError("RANSAC returned 0 inliers")
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    basis = vt[:2]  # 2D basis on plane
    normal = vt[2]
    return pts, basis, normal, center


# ----------------------------- 2D projection & raster ---------------------- #


def project_to_image(
    points: np.ndarray, basis: np.ndarray, center: np.ndarray, img_res: int
):
    coords_2d = (points - center) @ basis.T
    mn = coords_2d.min(axis=0)
    mx = coords_2d.max(axis=0)
    norm = (coords_2d - mn) / (mx - mn + 1e-9)
    img_xy = np.clip((norm * (img_res - 1)).astype(np.int32), 0, img_res - 1)
    mask2d = np.zeros((img_res, img_res), np.uint8)
    mask2d[img_xy[:, 1], img_xy[:, 0]] = 1
    return coords_2d, img_xy, mask2d


def auto_disk_radius_px(img_xy: np.ndarray) -> int:
    if len(img_xy) < 8:
        return 1
    pts = img_xy.astype(np.float32)
    nn = NearestNeighbors(n_neighbors=4, algorithm="kd_tree").fit(pts)
    d, _ = nn.kneighbors(pts)
    step = float(np.median(d[:, 3]))
    r = max(1, int(np.ceil(0.45 * step)))
    return min(r, 12)


def auto_kernels_from_radius(r: int) -> Tuple[int, int]:
    k_close = max(3, 2 * r + 1)  # stronger closing
    k_open = max(3, (int(round(1.5 * r)) * 2 + 1))
    return k_close, k_open


def rasterize_and_morph(
    mask2d: np.ndarray, img_xy: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray, int, int]:
    r_disk = auto_disk_radius_px(img_xy)
    se = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r_disk + 1, 2 * r_disk + 1)
    )
    mask_raster = cv2.dilate(mask2d, se)  # 0/1
    k_close, k_open = 25, 25
    mask255 = (mask_raster * 255).astype(np.uint8)
    mask255 = cv2.morphologyEx(
        mask255, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8)
    )
    mask255 = cv2.morphologyEx(
        mask255, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8)
    )
    mask255 = cv2.GaussianBlur(mask255, (11, 11), 5.0)
    mask = (mask255 > 32).astype(np.uint8)
    return r_disk, mask_raster, mask, k_close, k_open


# ----------------------------- geodesic + watershed ridges ----------------- #


def watershed_ridge_skeleton(mask: np.ndarray):
    mask_bool = mask.astype(bool)
    contour = find_boundaries(mask_bool, mode="outer").astype(np.uint8)
    edges = cv2.Canny((mask * 255).astype(np.uint8), 100, 200) // 255
    inv_edges = ~edges.astype(bool)
    edt = distance_transform_edt(inv_edges).astype(np.float32)
    geo = np.zeros_like(edt, np.float32)
    geo[mask_bool] = edt[mask_bool]
    elev = geo + contour.astype(np.float32) * 1e-3
    markers, _ = nd_label(mask)
    labels = watershed(elev, markers=markers, mask=mask_bool)
    ridges = find_boundaries(labels, mode="inner").astype(np.uint8)
    ridges[contour.astype(bool)] = 0
    return ridges, contour, geo


# ----------------------------- matplotlib grid (save only) ----------------- #


def save_grid(
    title: str,
    coords_2d,
    mask2d,
    r_disk,
    mask_raster,
    mask,
    contour,
    geo_dist,
    skeleton,
    k_close: int,
    k_open: int,
    out_png: str,
):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=13)

    axs[0, 0].scatter(coords_2d[:, 0], coords_2d[:, 1], s=1, c="k")
    axs[0, 0].set_title("Projected points")
    axs[0, 0].axis("equal")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(mask2d, cmap="gray")
    axs[0, 1].set_title("Raw mask2d")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(mask_raster, cmap="gray")
    axs[0, 2].set_title(f"Rasterized (r={r_disk}px)")
    axs[0, 2].axis("off")

    axs[0, 3].imshow(mask, cmap="gray")
    axs[0, 3].set_title(f"Morph (close={k_close}, open={k_open})")
    axs[0, 3].axis("off")

    axs[1, 0].imshow(contour, cmap="gray")
    axs[1, 0].set_title("Contour")
    axs[1, 0].axis("off")

    im = axs[1, 1].imshow(geo_dist, cmap="magma")
    axs[1, 1].set_title("Geodesic distance")
    axs[1, 1].axis("off")
    fig.colorbar(im, ax=axs[1, 1], shrink=0.8)

    axs[1, 2].imshow(skeleton, cmap="gray")
    axs[1, 2].set_title("Skeleton (watershed ridges)")
    axs[1, 2].axis("off")

    overlay = np.stack(
        [skeleton * 255, mask * 255, np.zeros_like(mask)], axis=-1
    ).astype(np.uint8)
    axs[1, 3].imshow(overlay)
    axs[1, 3].set_title("Overlay (red=skeleton, green=mask)")
    axs[1, 3].axis("off")

    plt.tight_layout()
    save_grid_png(out_png, fig)


# ----------------------------- one pipeline run ---------------------------- #


def run_pipeline(points_xyz: np.ndarray, tag: str, p: Params):
    pts, basis, normal, center = plane_from_points_array(points_xyz, p)
    coords_2d, img_xy, mask2d = project_to_image(pts, basis, center, p.img_res)
    r_disk, mask_raster, mask, k_close, k_open = rasterize_and_morph(
        mask2d, img_xy
    )
    skeleton, contour, geo = watershed_ridge_skeleton(mask)

    od = ensure_outdir(p.out_dir)

    # save individual layers
    def _save_img(name: str, arr, cmap="gray"):
        path = os.path.join(od, f"{tag}_{name}.png")
        if arr.ndim == 2:
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(arr, cmap=cmap)
            plt.axis("off")
        else:
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(arr)
            plt.axis("off")
        save_grid_png(path, fig)

    _save_img("mask2d", mask2d)
    _save_img("mask_raster", mask_raster)
    _save_img("mask", mask)
    _save_img("contour", contour)
    _save_img("skeleton", skeleton)
    # distance map
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(geo, cmap="magma")
    plt.axis("off")
    save_grid_png(os.path.join(od, f"{tag}_geo.png"), fig)
    # grid
    save_grid(
        title=f"{tag}: plane->mask->skeleton",
        coords_2d=coords_2d,
        mask2d=mask2d,
        r_disk=r_disk,
        mask_raster=mask_raster,
        mask=mask,
        contour=contour,
        geo_dist=geo,
        skeleton=skeleton,
        k_close=k_close,
        k_open=k_open,
        out_png=os.path.join(od, f"{tag}_grid.png"),
    )
    print(f"[{tag}] saved to: {od}")


# ----------------------------- viewer & main ------------------------------- #


def main():
    p = Params()
    od = ensure_outdir(p.out_dir)

    # load/clean raw
    raw = load_and_clean_cloud(p)

    # --- RUN A: RAW pipeline upfront -> out/ ---
    run_pipeline(np.asarray(raw.points), tag="raw", p=p)

    # prepare mesh for viewer (but do mesh pipeline on key '2')
    mesh_src = preprocess_cloud_for_mesh(raw, p)
    surf = reconstruct_and_smooth_mesh(mesh_src, p)
    print(f"Mesh: V={len(surf.vertices)}, F={len(surf.triangles)}")

    # build simple viewer: 1=raw cloud, 2=mesh (+ run mesh pipeline -> out)
    raw_vis = o3d.geometry.PointCloud(raw)  # show cleaned cloud
    raw_vis.paint_uniform_color([0.05, 0.05, 0.05])

    surf_vis = o3d.geometry.TriangleMesh(surf)
    surf_vis.paint_uniform_color([0.82, 0.82, 0.82])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Cloud/Mesh viewer", width=1280, height=900)

    state = {"cur": 1}

    def redraw():
        vis.clear_geometries()
        if state["cur"] == 1:
            vis.add_geometry(raw_vis, reset_bounding_box=True)
        else:
            vis.add_geometry(surf_vis, reset_bounding_box=True)
        vis.update_renderer()

    def show_raw(_):
        state["cur"] = 1
        redraw()
        return True

    def show_mesh_and_run_pipeline(_):
        state["cur"] = 2
        redraw()
        # run mesh pipeline on its vertices and save to out/
        V = np.asarray(surf.vertices)
        if V.size > 0:
            run_pipeline(V, tag="mesh", p=p)
        else:
            print("[mesh] empty, skipped pipeline.")
        return True

    redraw()
    vis.register_key_callback(ord("1"), show_raw)
    vis.register_key_callback(ord("2"), show_mesh_and_run_pipeline)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
