# vision/viz/overlays.py
from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np
import open3d as o3d

from utils.logger import Logger

LOG = Logger.get_logger("overlays")


def _ensure_normals(pc: o3d.geometry.PointCloud, radius: float) -> None:
    if pc.has_normals() and len(pc.normals) == len(pc.points):
        return
    if len(pc.points) == 0:
        return
    pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60)
    )
    pc.orient_normals_consistent_tangent_plane(50)


def _mean_spacing(pc: o3d.geometry.PointCloud, k: int = 12) -> float:
    if len(pc.points) < 2:
        return 0.005
    kdt = o3d.geometry.KDTreeFlann(pc)
    P = np.asarray(pc.points)
    d = []
    for i in range(min(len(P), 500)):
        _, idx, _ = kdt.search_knn_vector_3d(P[i], min(k, len(P)))
        if len(idx) >= 2:
            dd = np.linalg.norm(P[idx[1:]] - P[i], axis=1)
            d.append(dd.mean())
    return float(np.median(d)) if d else 0.005


def build_side_region_meshes(
    clouds: List[o3d.geometry.PointCloud],
    *,
    prefer_poisson: bool = False,
) -> List[o3d.geometry.TriangleMesh]:
    meshes: List[o3d.geometry.TriangleMesh] = []
    for i, pc in enumerate(clouds):
        if len(pc.points) < 100:
            meshes.append(o3d.geometry.TriangleMesh())
            continue
        pc = o3d.geometry.PointCloud(pc)  # копия
        h = _mean_spacing(pc)
        _ensure_normals(pc, radius=max(h * 3.0, 0.006))

        try:
            if prefer_poisson:
                raise RuntimeError("force_poisson")

            r = [h * 1.5, h * 2.2, h * 3.0]
            rec = (
                o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pc, o3d.utility.DoubleVector(r)
                )
            )
            rec.remove_degenerate_triangles()
            rec.remove_duplicated_triangles()
            rec.remove_duplicated_vertices()
            rec.remove_non_manifold_edges()
            meshes.append(rec)
        except Exception as e:
            LOG.warning(f"BPA failed on region {i}: {e}")
            try:
                rec, _ = (
                    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pc, depth=8
                    )
                )
                bbox = pc.get_axis_aligned_bounding_box()
                rec = rec.crop(bbox)
                meshes.append(rec)
            except Exception as e2:
                LOG.warning(f"Poisson failed on region {i}: {e2}")
                meshes.append(o3d.geometry.TriangleMesh())
    return meshes


def make_lineset_from_polylines(
    polys: Iterable[np.ndarray], color: Tuple[float, float, float]
):
    """Polyline -> LineSet"""
    pts, segs, base = [], [], 0
    for P in polys:
        if P is None or len(P) < 2:
            continue
        n = len(P)
        pts.append(P)
        a = np.arange(base, base + n - 1)
        b = a + 1
        segs.append(np.stack([a, b], axis=1))
        base += n
    if not pts:
        return o3d.geometry.LineSet()
    V = np.vstack(pts)
    E = np.vstack(segs).astype(np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(V)
    ls.lines = o3d.utility.Vector2iVector(E)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, float), (len(E), 1))
    )
    return ls


def make_nodes_mesh(
    nodes_xyz: np.ndarray, color: Tuple[float, float, float], r: float = 0.004
) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh()
    for p in nodes_xyz:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        s.translate(np.asarray(p, float))
        s.compute_vertex_normals()
        s.paint_uniform_color(color)
        m += s
    return m


def aabb_overlay(
    aabb: o3d.geometry.AxisAlignedBoundingBox,
) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    ls.paint_uniform_color((1.0, 0.1, 0.1))
    return ls


def axes(size: float = 0.05) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def _ensure_normals_inplace(pc: o3d.geometry.PointCloud) -> None:
    if pc.has_normals() and len(pc.normals) == len(pc.points):
        return
    if len(pc.points) == 0:
        return
    aabb = pc.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(
        np.asarray(aabb.get_max_bound()) - np.asarray(aabb.get_min_bound())
    )
    rad = max(1e-3, 0.02 * float(diag))
    pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=60)
    )
    try:
        pc.orient_normals_consistent_tangent_plane(30)
    except Exception:
        pass


def points_to_mesh_surface(
    cloud: o3d.geometry.PointCloud,
    *,
    ball_radius: float = 0.01,
    target: int | None = 30_000,
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    pc = o3d.geometry.PointCloud(cloud)
    if len(pc.points) == 0:
        return mesh

    _ensure_normals_inplace(pc)
    try:
        rec = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pc, o3d.utility.DoubleVector([ball_radius, ball_radius * 2.0])
        )
        if len(rec.triangles) == 0:
            raise RuntimeError("BPA produced 0 tris")
        mesh = rec
    except Exception as e:
        LOG.warning(f"BPA failed: {e}")
        try:
            rec, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pc, depth=8
            )
            aabb = pc.get_axis_aligned_bounding_box()
            mesh = rec.crop(aabb)
        except Exception as e2:
            LOG.warning(f"Poisson failed: {e2}")
            return o3d.geometry.TriangleMesh()

    if target and len(mesh.triangles) > target:
        try:
            mesh = mesh.simplify_quadric_decimation(target)
        except Exception:
            pass
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.85, 0.85, 0.85))
    return mesh


def _estimate_normals_if_needed(
    pc: o3d.geometry.PointCloud, r: float = 0.01
) -> None:
    if pc.has_normals() and len(pc.normals) == len(pc.points):
        return
    pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=60)
    )


def clouds_to_meshes(
    clouds: List[o3d.geometry.PointCloud],
    method: str = "bpa",  # "bpa" | "poisson"
    bpa_radii: Tuple[float, float, float] = (0.004, 0.006, 0.008),
    poisson_depth: int = 8,
) -> List[o3d.geometry.TriangleMesh]:
    """
    Convert region point clouds into triangle meshes for nicer display.
    Colors are copied from per-point average color of each cloud.
    """
    meshes: List[o3d.geometry.TriangleMesh] = []
    for pc in clouds:
        if len(pc.points) < 200:
            meshes.append(o3d.geometry.TriangleMesh())
            continue
        col = (
            np.asarray(pc.colors).mean(0).tolist()
            if pc.has_colors() and len(pc.colors)
            else [0.7, 0.7, 0.7]
        )
        _estimate_normals_if_needed(pc, r=0.012)
        if method == "poisson":
            try:
                mesh, _ = (
                    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pc, depth=int(poisson_depth)
                    )
                )
                mesh = mesh.filter_smooth_simple(number_of_iterations=1)
            except Exception as e:
                LOG.warning(f"poisson failed, fallback to BPA: {e}")
                method = "bpa"  # fallthrough
        if method == "bpa":
            radii = o3d.utility.DoubleVector(list(bpa_radii))
            pcd = pc.voxel_down_sample(voxel_size=min(bpa_radii) * 0.6)
            _estimate_normals_if_needed(pcd, r=min(bpa_radii) * 2.0)
            mesh = (
                o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, radii
                )
            )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(col)
        meshes.append(mesh)
    return meshes
