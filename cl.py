"""
3D point-cloud curve skeleton + tube reconstruction (Open3D).

Fixes
- Robust mesh cleanup: crop to cloud AABB, distance pruning vs. cloud, keep largest component.
- Open3D compatibility: select_by_index(list, cleanup=True).
- Visualizer hotkeys: '2' toggles graph (from adaptive construction), '3' toggles skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from collections import defaultdict, deque
import heapq
import numpy as np
import open3d as o3d

# --------------------------- PARAMETERS ---------------------------------------

VOXEL_SIZE: float = 0.005  # preprocessing voxel for raw cloud
NB_NEIGHBORS: int = 20  # SO-filter K
STD_RATIO: float = 2.0  # SO-filter std threshold

POISSON_DEPTH: int = 9
BPA_RADIUS: float = 0.01
TARGET_TRIS: int = 120_000
AABB_EXPAND_SURF: float = 1.05  # mesh crop vs input cloud AABB
MESH_TO_CLOUD_THR: float = 0.012  # meters; drop mesh verts farther than this

LANDMARKS: int = 24
TUBE_RADIUS: float = 0.01
AABB_EXPAND: float = 1.10  # AABB pruning for skeleton lines
MIN_SEG_LEN: float = 5e-4  # drop degenerate segments (<0.5 mm)

# Adaptive graph (paper)
GRAPH_K: int = 8  # KNN neighbors
GRAPH_MUL: float = 2.5  # distance multiplier m
GRAPH_ANGLE_DEG: float = 75.0  # angle threshold (degrees)

# --------------------------- TYPES -------------------------------------------

Vec3 = Tuple[float, float, float]
Index = int


@dataclass(frozen=True)
class Edge:
    u: Index
    v: Index


@dataclass
class Polyline:
    verts: List[Index]


# --------------------------- UTILS -------------------------------------------


def _np(a: o3d.utility.Vector3dVector) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _ensure_normals(pcd: o3d.geometry.PointCloud) -> None:
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=VOXEL_SIZE * 3, max_nn=22
            )
        )
        pcd.orient_normals_consistent_tangent_plane(50)


# --------------------------- PREPROCESS --------------------------------------


def preprocess_cloud(raw: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(raw)
    if len(pcd.points) == 0:
        return pcd
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(NB_NEIGHBORS, STD_RATIO)
    _ensure_normals(pcd)
    return pcd


# --------------------------- AABB / DIST CLEANUP -----------------------------


def _crop_mesh_to_cloud_aabb(
    mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, expand: float
) -> o3d.geometry.TriangleMesh:
    """Hard crop surface to the expanded AABB of the input cloud."""
    if len(mesh.triangles) == 0 or len(pcd.points) == 0:
        return mesh
    aabb = pcd.get_axis_aligned_bounding_box()
    c = aabb.get_center()
    half = (aabb.get_max_bound() - aabb.get_min_bound()) * 0.5 * float(expand)
    aabb_e = o3d.geometry.AxisAlignedBoundingBox(c - half, c + half)
    return mesh.crop(aabb_e)


def prune_mesh_by_cloud_distance(
    mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, thr: float
) -> o3d.geometry.TriangleMesh:
    """Remove triangles that are not supported by the input cloud."""
    if len(mesh.vertices) == 0 or len(pcd.points) == 0:
        return mesh

    tree = o3d.geometry.KDTreeFlann(pcd)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    P = np.asarray(pcd.points, dtype=np.float64)
    thr2 = float(thr) * float(thr)

    ok_vert = np.zeros(len(V), dtype=bool)
    for i, p in enumerate(V):
        # nearest neighbor distance^2
        try:
            _, idx, _ = tree.search_knn_vector_3d(p.astype(np.float64), 1)
        except Exception:
            continue
        q = P[idx[0]]
        ok_vert[i] = np.sum((p - q) * (p - q)) <= thr2

    T = np.asarray(mesh.triangles, dtype=np.int64)
    keep_tri = ok_vert[T].any(
        axis=1
    )  # keep triangles with at least one supported vertex
    # remove triangles by mask (avoid select_by_index on CUDA builds)
    rm_mask = (~keep_tri).tolist()
    if any(rm_mask):
        mesh.remove_triangles_by_mask(rm_mask)
        mesh.remove_unreferenced_vertices()
    return mesh


# --------------------------- SURFACE RECON -----------------------------------


def _poisson(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    rec, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )
    aabb = pcd.get_axis_aligned_bounding_box()
    return rec.crop(aabb)


def _bpa(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    r = BPA_RADIUS
    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([r, r * 2.0])
    )


def _keep_largest_component(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    labels, _, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return mesh
    counts = np.bincount(labels)
    keep = int(np.argmax(counts))
    rm_mask = (labels != keep).tolist()
    if any(rm_mask):
        mesh.remove_triangles_by_mask(rm_mask)
        mesh.remove_unreferenced_vertices()
    return mesh


def reconstruct_mesh(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    if len(pcd.points) == 0:
        return o3d.geometry.TriangleMesh()
    try:
        mesh = _poisson(pcd)
        if len(mesh.triangles) == 0:
            raise RuntimeError("empty Poisson")
    except Exception:
        mesh = _bpa(pcd)

    mesh = _crop_mesh_to_cloud_aabb(mesh, pcd, AABB_EXPAND_SURF)
    mesh = _keep_largest_component(mesh)
    mesh = prune_mesh_by_cloud_distance(mesh, pcd, MESH_TO_CLOUD_THR)
    mesh = _keep_largest_component(mesh)

    if TARGET_TRIS and len(mesh.triangles) > TARGET_TRIS:
        mesh = mesh.simplify_quadric_decimation(TARGET_TRIS)

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


# --------------------------- GRAPH FROM MESH ---------------------------------


def mesh_graph(
    mesh: o3d.geometry.TriangleMesh,
) -> Tuple[np.ndarray, List[Edge]]:
    V = _np(mesh.vertices)
    T = np.asarray(mesh.triangles, dtype=np.int64)
    edges_set = set()
    for a, b, c in T:
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edges_set.add((int(u), int(v)))
    edges = [Edge(u, v) for (u, v) in edges_set]
    return V, edges


def build_adjacency(
    nv: int, edges: Sequence[Edge], V: np.ndarray
) -> List[List[Tuple[Index, float]]]:
    adj: List[List[Tuple[Index, float]]] = [[] for _ in range(nv)]
    for e in edges:
        w = float(np.linalg.norm(V[e.u] - V[e.v]))
        if w < MIN_SEG_LEN:
            continue
        adj[e.u].append((e.v, w))
        adj[e.v].append((e.u, w))
    return adj


# --------------------------- SHORTEST PATHS ----------------------------------


def dijkstra(
    adj: List[List[Tuple[Index, float]]], s: Index
) -> Tuple[np.ndarray, List[Index]]:
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float64)
    parent = [-1] * n
    dist[s] = 0.0
    pq: List[Tuple[float, Index]] = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent


def backtrack(parent: List[Index], t: Index) -> List[Index]:
    out = deque()
    u = t
    while u != -1:
        out.appendleft(u)
        u = parent[u]
    return list(out)


# --------------------------- LANDMARKS / SKELETON ----------------------------


def farthest_point_sampling(
    adj: List[List[Tuple[Index, float]]], seeds: int
) -> List[Index]:
    dist, _ = dijkstra(adj, 0)
    cur = int(np.argmax(dist))
    L = [cur]
    for _ in range(seeds - 1):
        dist, _ = dijkstra(adj, cur)
        cur = int(np.argmax(dist))
        L.append(cur)
    return L


def steiner_tree_union(
    adj: List[List[Tuple[Index, float]]], L: Sequence[Index]
) -> Dict[Index, List[Index]]:
    tree: Dict[Index, List[Index]] = defaultdict(list)
    _, parent = dijkstra(adj, L[0])
    for l in L[1:]:
        path = backtrack(parent, l)
        for a, b in zip(path[:-1], path[1:]):
            if a == b:
                continue
            tree[a].append(b)
            tree[b].append(a)
    return tree


def compress_degree2(tree: Dict[Index, List[Index]]) -> List[Polyline]:
    deg = {u: len(vs) for u, vs in tree.items()}
    visited = set()
    lines: List[Polyline] = []

    def is_junction(u: Index) -> bool:
        return deg.get(u, 0) != 2

    for u in list(tree.keys()):
        if not is_junction(u):
            continue
        for v in tree[u]:
            if (u, v) in visited or (v, u) in visited:
                continue
            chain = [u, v]
            a, b = u, v
            while not is_junction(b):
                nxts = [w for w in tree[b] if w != a]
                if not nxts:
                    break
                a, b = b, nxts[0]
                chain.append(b)
            for x, y in zip(chain[:-1], chain[1:]):
                visited.add((x, y))
            lines.append(Polyline(chain))
    return lines


def unique_endpoints(lines: Sequence[Polyline]) -> List[Index]:
    """All unique endpoints of skeleton polylines (junctions + endpoints)."""
    nodes: set[Index] = set()
    for pl in lines:
        if len(pl.verts) >= 1:
            nodes.add(int(pl.verts[0]))
            nodes.add(int(pl.verts[-1]))
    return sorted(nodes)


def graph_from_polylines(
    lines: Sequence[Polyline],
    V: np.ndarray,
    node_radius: float,
) -> tuple[o3d.geometry.LineSet, o3d.geometry.TriangleMesh]:
    """
    Build a clean graph (nodes + straight edges) from skeleton polylines.
    - Nodes = unique endpoints of polylines
    - Each polyline -> one straight edge between its endpoints
    """
    nodes = unique_endpoints(lines)
    if not nodes:
        empty_ls = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.empty((0, 3), np.float64)),
            o3d.utility.Vector2iVector(np.empty((0, 2), np.int32)),
        )
        return empty_ls, o3d.geometry.TriangleMesh()

    # map original vertex-index -> compact graph index
    id_map = {int(i): k for k, i in enumerate(nodes)}
    pts = np.asarray([V[i] for i in nodes], dtype=np.float64)

    # edges: one segment per polyline (endpoint -> endpoint)
    edges: list[tuple[int, int]] = []
    for pl in lines:
        if len(pl.verts) < 2:
            continue
        a = id_map[int(pl.verts[0])]
        b = id_map[int(pl.verts[-1])]
        if a != b:
            if a > b:
                a, b = b, a
            edges.append((a, b))
    # deduplicate edges
    if edges:
        edges = sorted(set(edges))
        L = np.asarray(edges, dtype=np.int32)
    else:
        L = np.empty((0, 2), dtype=np.int32)

    graph_ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(pts),
        o3d.utility.Vector2iVector(L),
    )

    # small spheres at node positions (for junctions visualization)
    node_mesh = o3d.geometry.TriangleMesh()
    for p in pts:
        s = o3d.geometry.TriangleMesh.create_sphere(max(1e-4, node_radius))
        s.translate(p)
        node_mesh += s
    node_mesh.merge_close_vertices(1e-6)
    node_mesh.compute_vertex_normals()

    return graph_ls, node_mesh


from collections import Counter


def nodes_from_polylines(lines: Sequence[Polyline]) -> List[Index]:
    """Find graph nodes: endpoints + junctions (degree >= 3)."""
    counts = Counter()
    for pl in lines:
        if len(pl.verts) < 2:
            continue
        counts[pl.verts[0]] += 1
        counts[pl.verts[-1]] += 1
    nodes = [i for i, c in counts.items() if c >= 1]
    return nodes


def nodes_to_spheres_mesh(
    nodes: Sequence[Index], V: np.ndarray, radius: float
) -> o3d.geometry.TriangleMesh:
    """Small sphere per graph node (for visualization)."""
    mesh = o3d.geometry.TriangleMesh()
    if not nodes:
        return mesh
    for i in nodes:
        s = o3d.geometry.TriangleMesh.create_sphere(radius)
        s.translate(V[int(i)])
        mesh += s
    mesh.merge_close_vertices(1e-6)
    mesh.compute_vertex_normals()
    return mesh


# --------------------------- GEOMETRY ----------------------------------------


def _rot_from_z(v: np.ndarray) -> np.ndarray:
    v = v / (np.linalg.norm(v) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, v)
    s = np.linalg.norm(axis)
    c = float(np.dot(z, v))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1])
    K = (
        np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        / s
    )
    return np.eye(3) + K * s + (K @ K) * ((1 - c) / (s * s))


def _cylinder(
    p0: np.ndarray, p1: np.ndarray, r: float
) -> o3d.geometry.TriangleMesh:
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < MIN_SEG_LEN:
        return o3d.geometry.TriangleMesh()
    m = o3d.geometry.TriangleMesh.create_cylinder(r, L, 24)
    m.rotate(_rot_from_z(v), center=(0, 0, 0))
    m.translate(p0)
    return m


# --------------------------- EXPORT ------------------------------------------


def polylines_to_lineset(
    lines: Sequence[Polyline], V: np.ndarray
) -> o3d.geometry.LineSet:
    pts: List[np.ndarray] = []
    segs: List[Tuple[int, int]] = []
    base = 0
    for pl in lines:
        idx = pl.verts
        if len(idx) < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            pa, pb = V[a], V[b]
            if np.linalg.norm(pa - pb) < MIN_SEG_LEN:
                continue
            pts.append(pa)
            pts.append(pb)
            segs.append((base, base + 1))
            base += 2
    if not pts:
        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.empty((0, 3), np.float64)),
            o3d.utility.Vector2iVector(np.empty((0, 2), np.int32)),
        )
    P = np.asarray(pts, np.float64).reshape(-1, 3)
    L = np.asarray(segs, np.int32).reshape(-1, 2)
    return o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(P), o3d.utility.Vector2iVector(L)
    )


def polylines_to_tube_mesh(
    lines: Sequence[Polyline], V: np.ndarray, r: float
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    for pl in lines:
        idx = pl.verts
        if len(idx) < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            pa, pb = V[a], V[b]
            if np.linalg.norm(pa - pb) < MIN_SEG_LEN:
                continue
            mesh += _cylinder(pa, pb, r)
    if len(mesh.triangles) == 0:
        return mesh
    mesh.merge_close_vertices(1e-6)
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


# --------------------------- PRUNING BY AABB ---------------------------------


def prune_polylines_by_aabb(
    lines: List[Polyline],
    V: np.ndarray,
    aabb: o3d.geometry.AxisAlignedBoundingBox,
    expand: float,
) -> List[Polyline]:
    """Keep polylines whose midpoint lies inside the expanded input AABB."""
    if len(V) == 0:
        return []
    mn = aabb.get_min_bound()
    mx = aabb.get_max_bound()
    c = aabb.get_center()
    half = (mx - mn) * 0.5 * float(expand)
    mn_e = c - half
    mx_e = c + half

    def inside(p: np.ndarray) -> bool:
        return bool(np.all(p >= mn_e - 1e-12) and np.all(p <= mx_e + 1e-12))

    keep: List[Polyline] = []
    for pl in lines:
        if not pl.verts:
            continue
        pts = V[np.asarray(pl.verts, dtype=int)]
        mid = pts[len(pts) // 2]
        if inside(mid):
            keep.append(pl)
    return keep


# --------------------------- ADAPTIVE GRAPH (paper) --------------------------


def adaptive_graph(
    points: np.ndarray,
    k: int = GRAPH_K,
    m: float = GRAPH_MUL,
    angle_deg: float = GRAPH_ANGLE_DEG,
) -> o3d.geometry.LineSet:
    """
    Adaptive graph (Alg.1 style):
      - KNN per vertex
      - stop when dist > m * d_min
      - add edge only if new direction differs by >= angle_deg from existing ones
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2:
        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.empty((0, 3), np.float64)),
            o3d.utility.Vector2iVector(np.empty((0, 2), np.int32)),
        )

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    tree = o3d.geometry.KDTreeFlann(pcd)
    cos_thr = float(np.cos(np.deg2rad(angle_deg)))

    segs = []
    for idx in range(n):
        v = points[idx]
        kq = max(2, min(k + 1, n))  # include self; ensure >=2
        try:
            _, I, _ = tree.search_knn_vector_3d(v.astype(np.float64), kq)
        except Exception:
            continue
        I = [i for i in I if i != idx]
        if not I:
            continue

        dists = [float(np.linalg.norm(points[i] - v)) for i in I]
        order = np.argsort(dists).tolist()
        dmin = dists[order[0]] if dists else 0.0

        dirs: List[np.ndarray] = []
        for oi in order:
            j = I[oi]
            dv = points[j] - v
            dist = dists[oi]
            if dmin > 0.0 and dist > m * dmin:
                break
            L = np.linalg.norm(dv)
            if L < MIN_SEG_LEN:
                continue
            d = dv / L
            # check angular separation
            add = True
            for ex in dirs:
                if float(np.dot(d, ex)) >= cos_thr:
                    add = False
                    break
            if add:
                dirs.append(d)
                a, b = (idx, j) if idx < j else (j, idx)
                segs.append((a, b))

    # deduplicate
    segs = sorted(set(segs))
    return o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(np.asarray(segs, dtype=np.int32)),
    )


# --------------------------- MAIN API ----------------------------------------


def cloud_to_skeleton_and_mesh(
    raw: o3d.geometry.PointCloud,
    *,
    landmarks: int = LANDMARKS,
    tube_radius: float = TUBE_RADIUS,
):
    pcd = preprocess_cloud(raw)
    if len(pcd.points) == 0:
        return (
            o3d.geometry.LineSet(),
            o3d.geometry.TriangleMesh(),
            o3d.geometry.TriangleMesh(),
            o3d.geometry.LineSet(),
        )

    surf = reconstruct_mesh(pcd)
    if len(surf.triangles) == 0:
        return (
            o3d.geometry.LineSet(),
            o3d.geometry.TriangleMesh(),
            surf,
            o3d.geometry.LineSet(),
        )

    V, E = mesh_graph(surf)
    adj = build_adjacency(len(V), E, V)

    L = farthest_point_sampling(adj, max(2, landmarks))
    tree = steiner_tree_union(adj, L)
    lines = compress_degree2(tree)
    lines = prune_polylines_by_aabb(
        lines, V, pcd.get_axis_aligned_bounding_box(), AABB_EXPAND
    )

    skel_ls = polylines_to_lineset(lines, V)
    tube = polylines_to_tube_mesh(lines, V, tube_radius)
    node_idx = nodes_from_polylines(lines)
    graph_ls = polylines_to_lineset(lines, V)
    node_mesh = nodes_to_spheres_mesh(
        node_idx, V, radius=max(0.6 * tube_radius, 1e-4)
    )
    node_mesh.paint_uniform_color((0.9, 0.2, 0.1))
    # adaptive graph from the paper (on mesh vertices; for visualization)
    graph_ls = adaptive_graph(
        V, k=GRAPH_K, m=GRAPH_MUL, angle_deg=GRAPH_ANGLE_DEG
    )

    surf.paint_uniform_color((0.82, 0.82, 0.82))
    return skel_ls, tube, surf, graph_ls, node_mesh


# --------------------------- DEMO / VIEWER -----------------------------------

if __name__ == "__main__":
    # Load your cloud here. Replace with your loader if needed.
    import os
    from main import _load_cloud
    from vision.skeleton.config import SkelPipelineCfg

    cfg = SkelPipelineCfg()
    base_cloud = _load_cloud(cfg)
    raw = o3d.geometry.PointCloud(base_cloud)

    skel_ls, tube, surf, graph_ls, node_mesh = cloud_to_skeleton_and_mesh(
        raw, tube_radius=TUBE_RADIUS
    )

    # colors
    if len(skel_ls.lines) > 0:
        skel_ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.10, 0.50, 0.90]]), (len(skel_ls.lines), 1))
        )
    if len(graph_ls.lines) > 0:
        graph_ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.10, 0.70, 0.30]]), (len(graph_ls.lines), 1))
        )

    # Visualizer with hotkeys:
    #  - '2' toggle graph edges
    #  - '3' toggle skeleton (line+ tube)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Skeleton & Graph", width=1280, height=800)

    mode = {"cur": 1}

    def redraw():
        vis.clear_geometries()
        if mode["cur"] == 1:
            vis.add_geometry(surf, reset_bounding_box=True)
        elif mode["cur"] == 2:
            vis.add_geometry(skel_ls, reset_bounding_box=True)
            vis.add_geometry(tube, reset_bounding_box=False)
        elif mode["cur"] == 3:
            vis.add_geometry(graph_ls, reset_bounding_box=True)
            vis.add_geometry(node_mesh, reset_bounding_box=False)
        vis.update_renderer()

    def set_surf(_):
        mode.update(cur=1)
        redraw()
        return True

    def set_skel(_):
        mode.update(cur=2)
        redraw()
        return True

    def set_graph(_):
        mode.update(cur=3)
        redraw()
        return True

    redraw()
    vis.register_key_callback(ord('1'), set_surf)
    vis.register_key_callback(ord('2'), set_skel)
    vis.register_key_callback(ord('3'), set_graph)
    vis.run()
    vis.destroy_window()
