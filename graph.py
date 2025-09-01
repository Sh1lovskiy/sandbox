# graph.py
"""
Truss graph extraction in pure 3D (Open3D). No plane projection.

Keys (demo):
  1 - cleaned surface
  2 - Graph-PCD    : adaptive 3D graph on the preprocessed point cloud
  3 - Graph-Ridges : adaptive 3D graph on mesh ridge vertices
  4 - Graph-Skel3D : graph from 3D curve skeletonization over the mesh graph
  5 - Surface-Segments: region-growing segments on the surface from [1]
  6 - Point-Segments  : DBSCAN segments on the preprocessed point cloud from [1]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Dict
import numpy as np
import open3d as o3d

from utils.logger import Logger

LOG = Logger.get_logger("graph")

# --------------------------- Parameters --------------------------------------

VOXEL_SIZE = 0.005
NB_NEIGHBORS = 20
STD_RATIO = 2.0

POISSON_DEPTH = 9
BPA_RADIUS = 0.01
TARGET_TRIS = 120_000
AABB_EXPAND_SURF = 1.05
MESH_TO_CLOUD_THR = 0.012

# Local scale & clustering
K_RADIUS = 8
RADIUS_SCALE = 1.2
CLUST_STRENGTH = 2.0

# Adaptive graph (Alg.1)
GRAPH_K = 10
GRAPH_MUL = 2.2
GRAPH_ANGLE_DEG = 70.0

# Skeletonization (method #4)
SKEL_LANDMARKS = 32
AABB_EXPAND = 1.10
SKEL_MIN_SEG = 5e-4

# Node snap/merge
NODE_SNAP_K = 9
NODE_MERGE_R = 0.02  # meters

# Ridges (method #3)
RIDGE_DIHEDRAL_DEG = 35.0

# Surface segments (method #5)
SEG_NORMAL_ANGLE_DEG = 25.0
SEG_MIN_TRIS = 150
SEG_MAX_SHOW = 128

# Point segments (method #6)
PSEG_EPS = 3.0 * VOXEL_SIZE  # DBSCAN radius in meters
PSEG_MIN_PTS = 45  # min cluster size
PSEG_MAX_SHOW = 128

# Misc
MIN_SEG_LEN = 5e-3
NODE_VIS_R = 0.01

# --------------------------- Types & utils -----------------------------------

Index = int


@dataclass(frozen=True)
class Edge:
    u: Index
    v: Index


def _np(v: o3d.utility.Vector3dVector) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def _ensure_normals(pcd: o3d.geometry.PointCloud) -> None:
    if not pcd.has_normals():
        LOG.debug(f"Estimating normals: radius={VOXEL_SIZE*3:.6f}, max_nn=22")
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=VOXEL_SIZE * 3, max_nn=22
            )
        )
        pcd.orient_normals_consistent_tangent_plane(50)
        LOG.debug("Normals estimated & oriented")


# --------------------------- Preprocess & Mesh -------------------------------


def preprocess_cloud(raw: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    LOG.info(f"[Preprocess] input points={len(raw.points)}")
    pcd = o3d.geometry.PointCloud(raw)
    if len(pcd.points) == 0:
        LOG.warning("[Preprocess] empty cloud")
        return pcd
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    LOG.debug(
        f"[Preprocess] voxel DS @ {VOXEL_SIZE:.4f} -> {len(pcd.points)} pts"
    )
    pcd, idx = pcd.remove_statistical_outlier(NB_NEIGHBORS, STD_RATIO)
    LOG.debug(
        f"[Preprocess] SO-filter K={NB_NEIGHBORS} std={STD_RATIO} "
        f"-> kept {len(idx)} pts"
    )
    _ensure_normals(pcd)
    LOG.info(f"[Preprocess] done: {len(pcd.points)} points")
    return pcd


def _crop_mesh_to_cloud_aabb(
    mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, expand: float
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0 or len(pcd.points) == 0:
        return mesh
    aabb = pcd.get_axis_aligned_bounding_box()
    c = aabb.get_center()
    half = (aabb.get_max_bound() - aabb.get_min_bound()) * 0.5 * float(expand)
    LOG.debug(f"[Mesh] crop to expanded AABB x{expand:.3f}")
    return mesh.crop(o3d.geometry.AxisAlignedBoundingBox(c - half, c + half))


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
    rm = (labels != keep).tolist()
    if any(rm):
        LOG.debug(
            f"[Mesh] keep largest connected, remove {np.count_nonzero(rm)} tris"
        )
        mesh.remove_triangles_by_mask(rm)
        mesh.remove_unreferenced_vertices()
    return mesh


def prune_mesh_by_cloud_distance(
    mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, thr: float
) -> o3d.geometry.TriangleMesh:
    if len(mesh.vertices) == 0 or len(pcd.points) == 0:
        return mesh
    LOG.debug(f"[Mesh] prune by cloud distance thr={thr:.4f} m")
    tree = o3d.geometry.KDTreeFlann(pcd)
    V = np.asarray(mesh.vertices, np.float64)
    P = np.asarray(pcd.points, np.float64)
    thr2 = float(thr) ** 2
    ok = np.zeros(len(V), bool)
    for i, p in enumerate(V):
        try:
            _, idx, _ = tree.search_knn_vector_3d(p, 1)
        except Exception:
            continue
        q = P[idx[0]]
        ok[i] = np.sum((p - q) ** 2) <= thr2
    T = np.asarray(mesh.triangles, np.int64)
    keep_tri = ok[T].any(axis=1)
    rm = (~keep_tri).tolist()
    if any(rm):
        LOG.debug(f"[Mesh] remove unsupported tris={np.count_nonzero(rm)}")
        mesh.remove_triangles_by_mask(rm)
        mesh.remove_unreferenced_vertices()
    return mesh


def reconstruct_mesh(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    LOG.info(f"[Mesh] reconstruct from {len(pcd.points)} points")
    if len(pcd.points) == 0:
        return o3d.geometry.TriangleMesh()
    try:
        rec, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=POISSON_DEPTH
        )
        mesh = rec
        LOG.debug(
            f"[Mesh] Poisson depth={POISSON_DEPTH}: "
            f"{len(mesh.vertices)}V {len(mesh.triangles)}T"
        )
        if len(mesh.triangles) == 0:
            raise RuntimeError()
    except Exception:
        r = BPA_RADIUS
        LOG.debug(f"[Mesh] fallback BPA r={r:.4f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([r, r * 2.0])
        )
    mesh = _crop_mesh_to_cloud_aabb(mesh, pcd, AABB_EXPAND_SURF)
    mesh = _keep_largest_component(mesh)
    mesh = prune_mesh_by_cloud_distance(mesh, pcd, MESH_TO_CLOUD_THR)
    mesh = _keep_largest_component(mesh)
    if TARGET_TRIS and len(mesh.triangles) > TARGET_TRIS:
        LOG.debug(f"[Mesh] QEM simplify -> {TARGET_TRIS} tris")
        mesh = mesh.simplify_quadric_decimation(TARGET_TRIS)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    LOG.info(f"[Mesh] done: {len(mesh.vertices)}V {len(mesh.triangles)}T")
    return mesh


# --------------------------- Local scale (r_i) --------------------------------


def estimate_local_radii(
    P: np.ndarray, k: int = K_RADIUS, scale: float = RADIUS_SCALE
) -> np.ndarray:
    if len(P) == 0:
        return np.empty(0, np.float64)
    LOG.debug(f"[Radius] estimate k={k} scale={scale:.3f} for {len(P)} pts")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    tree = o3d.geometry.KDTreeFlann(pcd)
    r = np.empty(len(P), np.float64)
    for i, p in enumerate(P):
        _, idx, _ = tree.search_knn_vector_3d(p, min(k + 1, len(P)))
        if len(idx) <= 1:
            r[i] = VOXEL_SIZE
        else:
            d = np.linalg.norm(P[idx[1:]] - p, axis=1)
            r[i] = scale * float(np.median(d))
    LOG.debug(f"[Radius] median={np.median(r):.6f}, mean={np.mean(r):.6f}")
    return r


# --------------------------- Radius clustering (Alg.2) -----------------------


def radius_clustering(
    P: np.ndarray, R: np.ndarray, strength: float = CLUST_STRENGTH
) -> tuple[np.ndarray, np.ndarray]:
    if len(P) == 0:
        return P, R
    LOG.info(f"[Cluster] start: points={len(P)} strength={strength:.2f}")
    order = np.arange(len(P))
    np.random.shuffle(order)
    flags = np.zeros(len(P), dtype=bool)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    tree = o3d.geometry.KDTreeFlann(pc)
    C_list, R_list = [], []
    for i in order:
        if flags[i]:
            continue
        radius = max(MIN_SEG_LEN, float(strength) * float(R[i]))
        try:
            _, idx, _ = tree.search_radius_vector_3d(P[i], radius)
        except Exception:
            idx = [i]
        idx = [j for j in idx if not flags[j]]
        if not idx:
            continue
        flags[idx] = True
        C_list.append(P[idx].mean(axis=0))
        R_list.append(R[idx].mean())
    C = np.asarray(C_list, np.float64)
    Rc = np.asarray(R_list, np.float64)
    LOG.info(f"[Cluster] done: clusters={len(C)}")
    return C, Rc


# --------------------------- Adaptive Graph (Alg.1) --------------------------


def adaptive_graph_3d(
    P: np.ndarray,
    k: int = GRAPH_K,
    m: float = GRAPH_MUL,
    angle_deg: float = GRAPH_ANGLE_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    if len(P) < 2:
        return P.copy(), np.empty((0, 2), np.int32)
    LOG.info(
        f"[Graph-3D] build on {len(P)} nodes | k={k} m={m:.2f} angle={angle_deg:.1f}"
    )
    cos_thr = np.cos(np.deg2rad(angle_deg))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    tree = o3d.geometry.KDTreeFlann(pcd)

    segs: list[tuple[int, int]] = []
    for v_idx, v in enumerate(P):
        kq = min(k + 1, len(P))
        try:
            _, I, _ = tree.search_knn_vector_3d(v, kq)
        except Exception:
            continue
        I = [i for i in I if i != v_idx]
        if not I:
            continue
        D = np.linalg.norm(P[I] - v, axis=1)
        order = np.argsort(D)
        dmin = D[order[0]] if D.size else 0.0
        dirs: list[np.ndarray] = []
        for oi in order:
            j = I[oi]
            d = D[oi]
            if dmin > 0.0 and d > m * dmin:
                break
            dv = P[j] - v
            L = np.linalg.norm(dv)
            if L < MIN_SEG_LEN:
                continue
            u = dv / L
            if any(float(np.dot(u, ex)) >= cos_thr for ex in dirs):
                continue
            dirs.append(u)
            a, b = (v_idx, j) if v_idx < j else (j, v_idx)
            segs.append((a, b))
    E = (
        np.unique(np.asarray(segs, np.int32), axis=0)
        if segs
        else np.empty((0, 2), np.int32)
    )
    LOG.info(f"[Graph-3D] edges={len(E)}")
    return P.copy(), E


# --------------------------- Ridges (Alg.3) ----------------------------------


def ridge_vertices(
    mesh: o3d.geometry.TriangleMesh, deg_thr: float = RIDGE_DIHEDRAL_DEG
) -> np.ndarray:
    if len(mesh.triangles) == 0:
        return np.empty((0, 3), np.float64)
    LOG.info(f"[Ridge] select vertices with dihedral > {deg_thr:.1f}°")
    V = _np(mesh.vertices)
    T = np.asarray(mesh.triangles, np.int32)
    F = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    nF = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    from collections import defaultdict

    ef = defaultdict(list)
    for fi, (a, b, c) in enumerate(T):
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            ef[(u, v)].append(fi)
    sel = np.zeros(len(V), dtype=bool)
    cos_thr = np.cos(np.deg2rad(deg_thr))
    for (u, v), fs in ef.items():
        if len(fs) != 2:
            continue
        a, b = fs
        if float(np.dot(nF[a], nF[b])) < cos_thr:
            sel[u] = True
            sel[v] = True
    out = V[sel]
    LOG.info(f"[Ridge] selected vertices={len(out)} / {len(V)}")
    return out


# --------------------------- Snap & Merge ------------------------------------


def snap_nodes_to_mesh(
    nodes: np.ndarray, surf: o3d.geometry.TriangleMesh, k: int = NODE_SNAP_K
) -> np.ndarray:
    if len(nodes) == 0 or len(surf.vertices) == 0:
        return nodes
    V = _np(surf.vertices)
    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V))
    )
    out = np.empty_like(nodes)
    for i, p in enumerate(nodes):
        _, idx, _ = tree.search_knn_vector_3d(p, min(k, len(V)))
        out[i] = V[idx].mean(axis=0)
    LOG.debug(f"[Snap] snapped {len(nodes)} nodes to mesh (k={k})")
    return out


def merge_nodes_radius(
    P: np.ndarray, E: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    if len(P) == 0:
        return P, E
    LOG.info(f"[Merge] radius={radius:.4f} on {len(P)} nodes, edges={len(E)}")
    cell = float(radius)
    keys = np.floor(P / cell).astype(np.int64)
    bins: Dict[tuple, tuple[np.ndarray, int, int]] = {}
    for i, k in enumerate(map(tuple, keys)):
        if k not in bins:
            bins[k] = (P[i].copy(), 1, -1)
        else:
            s, c, _ = bins[k]
            bins[k] = (s + P[i], c + 1, -1)
    new_pts = []
    id_map = np.full(len(P), -1, np.int32)
    for k, (s, c, _) in bins.items():
        idx = len(new_pts)
        bins[k] = (s, c, idx)
        new_pts.append(s / c)
    for i, k in enumerate(map(tuple, keys)):
        id_map[i] = bins[k][2]
    if E.size:
        e = id_map[E]
        e = e[e[:, 0] != e[:, 1]]
        e = np.unique(np.sort(e, axis=1), axis=0) if e.size else e
    else:
        e = np.empty((0, 2), np.int32)
    Pm = np.asarray(new_pts, np.float64)
    LOG.info(f"[Merge] result nodes={len(Pm)} edges={len(e)}")
    return Pm, e


def _ls_and_balls(
    nodes: np.ndarray, E: np.ndarray
) -> tuple[o3d.geometry.LineSet, o3d.geometry.TriangleMesh]:
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(nodes),
        o3d.utility.Vector2iVector(E if E.size else np.empty((0, 2), np.int32)),
    )
    balls = o3d.geometry.TriangleMesh()
    for p in nodes:
        s = o3d.geometry.TriangleMesh.create_sphere(max(1e-4, NODE_VIS_R))
        s.translate(p)
        balls += s
    balls.merge_close_vertices(1e-6)
    balls.compute_vertex_normals()
    return ls, balls


# --------------------------- Graphs (#2, #3, #4) -----------------------------


def build_graph_pcd(
    pcd: o3d.geometry.PointCloud, surf: o3d.geometry.TriangleMesh
) -> tuple[o3d.geometry.LineSet, o3d.geometry.TriangleMesh]:
    P = _np(pcd.points)
    LOG.info(f"[Graph-PCD] source points={len(P)}")
    if len(P) == 0:
        return _ls_and_balls(
            np.empty((0, 3), np.float64), np.empty((0, 2), np.int32)
        )
    radii = estimate_local_radii(P, K_RADIUS, RADIUS_SCALE)
    C, _ = radius_clustering(P, radii, CLUST_STRENGTH)
    nodes, E = adaptive_graph_3d(C, GRAPH_K, GRAPH_MUL, GRAPH_ANGLE_DEG)
    nodes = snap_nodes_to_mesh(nodes, surf, NODE_SNAP_K)
    nodes, E = merge_nodes_radius(nodes, E, NODE_MERGE_R)
    LOG.info(f"[Graph-PCD] final nodes={len(nodes)} edges={len(E)}")
    return _ls_and_balls(nodes, E)


def build_graph_ridges(
    pcd: o3d.geometry.PointCloud, surf: o3d.geometry.TriangleMesh
) -> tuple[o3d.geometry.LineSet, o3d.geometry.TriangleMesh]:
    V_sel = ridge_vertices(surf, RIDGE_DIHEDRAL_DEG)
    if len(V_sel) == 0:
        LOG.warning(
            "[Graph-Ridges] no ridge vertices; fallback to all mesh vertices"
        )
        V_sel = _np(surf.vertices)
    LOG.info(f"[Graph-Ridges] candidate points={len(V_sel)}")
    radii = estimate_local_radii(V_sel, K_RADIUS, RADIUS_SCALE)
    C, _ = radius_clustering(V_sel, radii, CLUST_STRENGTH)
    nodes, E = adaptive_graph_3d(C, GRAPH_K, GRAPH_MUL, GRAPH_ANGLE_DEG)
    nodes = snap_nodes_to_mesh(nodes, surf, NODE_SNAP_K)
    nodes, E = merge_nodes_radius(nodes, E, NODE_MERGE_R)
    LOG.info(f"[Graph-Ridges] final nodes={len(nodes)} edges={len(E)}")
    return _ls_and_balls(nodes, E)


# --------------------------- Skeletonization (#4) ----------------------------


def _mesh_edges(
    mesh: o3d.geometry.TriangleMesh,
) -> tuple[np.ndarray, List[Edge]]:
    V = _np(mesh.vertices)
    T = np.asarray(mesh.triangles, np.int64)
    es = set()
    for a, b, c in T:
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            es.add((int(u), int(v)))
    E = [Edge(u, v) for (u, v) in es]
    LOG.debug(f"[Skel3D] mesh edges built: V={len(V)} E={len(E)}")
    return V, E


def _adjacency(
    nv: int, edges: Sequence[Edge], V: np.ndarray
) -> List[List[tuple[int, float]]]:
    adj: List[List[tuple[int, float]]] = [[] for _ in range(nv)]
    for e in edges:
        w = float(np.linalg.norm(V[e.u] - V[e.v]))
        if w < SKEL_MIN_SEG:
            continue
        adj[e.u].append((e.v, w))
        adj[e.v].append((e.u, w))
    return adj


def _dijkstra(
    adj: List[List[tuple[int, float]]], s: int
) -> tuple[np.ndarray, List[int]]:
    n = len(adj)
    dist = np.full(n, np.inf, np.float64)
    parent = [-1] * n
    dist[s] = 0.0
    import heapq

    pq: List[tuple[float, int]] = [(0.0, s)]
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


def _backtrack(parent: List[int], t: int) -> List[int]:
    out = []
    u = t
    while u != -1:
        out.append(u)
        u = parent[u]
    out.reverse()
    return out


def _fps_on_graph(adj: List[List[tuple[int, float]]], seeds: int) -> List[int]:
    dist, _ = _dijkstra(adj, 0)
    cur = int(np.argmax(dist))
    L = [cur]
    LOG.debug(f"[Skel3D] FPS seed-0={cur}")
    for _ in range(seeds - 1):
        dist, _ = _dijkstra(adj, cur)
        cur = int(np.argmax(dist))
        L.append(cur)
    LOG.debug(f"[Skel3D] FPS seeds={len(L)}")
    return L


def _steiner_union(
    adj: List[List[tuple[int, float]]], L: Sequence[int]
) -> Dict[int, List[int]]:
    from collections import defaultdict

    tree: Dict[int, List[int]] = defaultdict(list)
    _, parent = _dijkstra(adj, L[0])
    for l in L[1:]:
        path = _backtrack(parent, l)
        for a, b in zip(path[:-1], path[1:]):
            if a == b:
                continue
            tree[a].append(b)
            tree[b].append(a)
    LOG.debug(f"[Skel3D] union edges≈{sum(len(vs) for vs in tree.values())//2}")
    return tree


def _compress_deg2(tree: Dict[int, List[int]]) -> List[List[int]]:
    deg = {u: len(vs) for u, vs in tree.items()}
    visited = set()
    lines: List[List[int]] = []

    def is_junction(u: int) -> bool:
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
            lines.append(chain)
    LOG.debug(f"[Skel3D] compressed polylines={len(lines)}")
    return lines


def _prune_by_aabb(
    lines: List[List[int]],
    V: np.ndarray,
    aabb: o3d.geometry.AxisAlignedBoundingBox,
) -> List[List[int]]:
    if len(V) == 0:
        return []
    mn = aabb.get_min_bound()
    mx = aabb.get_max_bound()
    c = aabb.get_center()
    half = (mx - mn) * 0.5 * float(AABB_EXPAND)
    mn_e = c - half
    mx_e = c + half

    def inside(p: np.ndarray) -> bool:
        return bool(np.all(p >= mn_e - 1e-12) and np.all(p <= mx_e + 1e-12))

    keep: List[List[int]] = []
    for pl in lines:
        if not pl:
            continue
        pts = V[np.asarray(pl, int)]
        mid = pts[len(pts) // 2]
        if inside(mid):
            keep.append(pl)
    return keep


def _polylines_to_graph(
    lines: List[List[int]], V: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not lines:
        return np.empty((0, 3), np.float64), np.empty((0, 2), np.int32)
    nodes_idx: List[int] = []
    for pl in lines:
        if len(pl) >= 2:
            nodes_idx.append(int(pl[0]))
            nodes_idx.append(int(pl[-1]))
    if not nodes_idx:
        return np.empty((0, 3), np.float64), np.empty((0, 2), np.int32)
    nodes_idx = sorted(set(nodes_idx))
    id_map = {i: k for k, i in enumerate(nodes_idx)}
    P = V[nodes_idx].copy()
    edges = []
    for pl in lines:
        if len(pl) < 2:
            continue
        a = id_map[int(pl[0])]
        b = id_map[int(pl[-1])]
        if a == b:
            continue
        if a > b:
            a, b = b, a
        edges.append((a, b))
    E = (
        np.unique(np.asarray(edges, np.int32), axis=0)
        if edges
        else np.empty((0, 2), np.int32)
    )
    return P, E


def build_graph_skeleton(
    pcd: o3d.geometry.PointCloud, surf: o3d.geometry.TriangleMesh
) -> tuple[o3d.geometry.LineSet, o3d.geometry.TriangleMesh]:
    LOG.info("[Skel3D] start skeletonization")
    if len(surf.triangles) == 0 or len(surf.vertices) == 0:
        LOG.warning("[Skel3D] empty surface; nothing to do")
        return _ls_and_balls(
            np.empty((0, 3), np.float64), np.empty((0, 2), np.int32)
        )
    V, E = _mesh_edges(surf)
    adj = _adjacency(len(V), E, V)
    if len(adj) == 0:
        LOG.warning("[Skel3D] empty adjacency")
        return _ls_and_balls(
            np.empty((0, 3), np.float64), np.empty((0, 2), np.int32)
        )
    seeds = max(2, int(SKEL_LANDMARKS))
    L = _fps_on_graph(adj, seeds)
    tree = _steiner_union(adj, L)
    lines = _compress_deg2(tree)
    lines = _prune_by_aabb(lines, V, pcd.get_axis_aligned_bounding_box())
    nodes, E2 = _polylines_to_graph(lines, V)
    LOG.info(f"[Skel3D] endpoints={len(nodes)} raw_edges={len(E2)}")
    nodes, E2 = merge_nodes_radius(nodes, E2, NODE_MERGE_R)
    LOG.info(f"[Skel3D] merged endpoints={len(nodes)} edges={len(E2)}")
    return _ls_and_balls(nodes, E2)


# --------------------------- Surface Segments (#5) ---------------------------


def _triangle_adjacency(mesh: o3d.geometry.TriangleMesh) -> List[List[int]]:
    T = np.asarray(mesh.triangles, np.int64)
    n = len(T)
    adj: List[List[int]] = [[] for _ in range(n)]
    from collections import defaultdict

    edge2tris: Dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(T):
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge2tris[(u, v)].append(fi)
    for tris in edge2tris.values():
        if len(tris) < 2:
            continue
        a, b = tris[0], tris[1]
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _distinct_colors(n: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 3), np.float64)
    hue = np.linspace(0.0, 1.0, n, endpoint=False)
    sat = np.full(n, 0.75)
    val = np.full(n, 0.95)
    import colorsys

    cols = [colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hue, sat, val)]
    return np.asarray(cols, np.float64)


def segment_surface_regions(
    mesh: o3d.geometry.TriangleMesh,
    angle_deg: float | None = SEG_NORMAL_ANGLE_DEG,
    min_tris: int = SEG_MIN_TRIS,
) -> List[o3d.geometry.TriangleMesh]:
    if len(mesh.triangles) == 0:
        LOG.warning("[Segments] empty surface; nothing to do")
        return []
    T = np.asarray(mesh.triangles, np.int64)
    V = _np(mesh.vertices)
    adj = _triangle_adjacency(mesh)
    if angle_deg is not None:
        F = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        nF = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        cos_thr = float(np.cos(np.deg2rad(angle_deg)))
    visited = np.zeros(len(T), dtype=bool)
    comp: List[List[int]] = []
    for seed in range(len(T)):
        if visited[seed]:
            continue
        stack = [seed]
        cur: List[int] = []
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            cur.append(u)
            for v in adj[u]:
                if visited[v]:
                    continue
                if angle_deg is None:
                    stack.append(v)
                else:
                    if float(np.dot(nF[u], nF[v])) >= cos_thr:
                        stack.append(v)
        if len(cur) >= int(min_tris):
            comp.append(cur)
    comp.sort(key=len, reverse=True)
    if len(comp) > SEG_MAX_SHOW:
        LOG.warning(
            f"[Segments] too many ({len(comp)}), showing top {SEG_MAX_SHOW}"
        )
        comp = comp[:SEG_MAX_SHOW]
    colors = _distinct_colors(len(comp))
    out: List[o3d.geometry.TriangleMesh] = []
    for i, tri_idx in enumerate(comp):
        verts = np.unique(T[tri_idx].reshape(-1))
        seg = mesh.select_by_index(verts.tolist(), cleanup=True)
        seg.remove_unreferenced_vertices()
        seg.compute_vertex_normals()
        seg.paint_uniform_color(tuple(colors[i]))
        out.append(seg)
    LOG.info(
        f"[Segments] angle_thr="
        f"{'none' if angle_deg is None else f'{angle_deg:.1f}°'} "
        f"min_tris={min_tris} -> segments={len(out)}"
    )
    return out


# --------------------------- Point Segments (#6) -----------------------------
def segment_point_cloud_dbscan(
    pcd: o3d.geometry.PointCloud,
    eps: float = PSEG_EPS,
    min_points: int = PSEG_MIN_PTS,
) -> List[o3d.geometry.PointCloud]:
    r"""
    Segment a point cloud into structural elements using DBSCAN.

    DBSCAN setup
    ------------
    Point set:  P = {x_i ∈ ℝ³}, i=1..N
    Neighborhood (L2):  N_ε(p) = { q ∈ P | ||p - q||₂ ≤ ε }
    Core point:         |N_ε(p)| ≥ minPts
    Density reachability / connectivity: standard DBSCAN definitions.

    Practical ε choice:
        Let r_i = median_k(||x_t - x_i||₂), t ∈ kNN(i).
        Use ε = α · median_i r_i  (α ≈ 1.5…2.5).  ← scale to sampling density.

    PCA axis (next step, not here):
        u* = argmin_u ∑ ||(x_i - μ) - ((x_i - μ)·u) u||²,  ||u||=1,  μ = mean(C_j)
    """
    if len(pcd.points) == 0:
        LOG.warning("[P-Seg] empty cloud; nothing to do")
        return []

    # Auto-eps if not positive: use kNN-median scale (matches the doc above)
    if eps is None or eps <= 0:
        P = np.asarray(pcd.points, dtype=np.float64)
        r = estimate_local_radii(P, k=K_RADIUS, scale=1.0)
        eps = float(2.0 * np.median(r))  # α=2.0 by default
        LOG.debug(f"[P-Seg] auto-eps from kNN scale -> {eps:.6f}")

    LOG.info(f"[P-Seg] DBSCAN eps={eps:.4f} min_pts={int(min_points)}")
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=eps, min_points=int(min_points), print_progress=False
        )
    )

    if labels.size == 0 or labels.max() < 0:
        LOG.warning("[P-Seg] no clusters found")
        return []

    ncl = int(labels.max() + 1)
    if ncl > PSEG_MAX_SHOW:
        LOG.warning(f"[P-Seg] too many ({ncl}), showing top {PSEG_MAX_SHOW}")

    # Stable palette (HSV spread) for deterministic visualization
    colors = _distinct_colors(min(ncl, PSEG_MAX_SHOW))

    # Sort clusters by size desc
    counts = [(cid, int((labels == cid).sum())) for cid in range(ncl)]
    counts.sort(key=lambda x: x[1], reverse=True)

    segs: List[o3d.geometry.PointCloud] = []
    for rank, (cid, cnt) in enumerate(counts[:PSEG_MAX_SHOW]):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        seg = pcd.select_by_index(idx.tolist())
        seg.paint_uniform_color(tuple(colors[rank]))
        LOG.debug(f"[P-Seg] cluster {cid}: {cnt} pts")
        segs.append(seg)

    LOG.info(f"[P-Seg] clusters_shown={len(segs)} / found={ncl}")
    return segs


def segment_point_cloud_region_growing(
    pcd: o3d.geometry.PointCloud,
    angle_deg: float = 25.0,  # normal consistency threshold θ
    k_radius: int = 16,  # kNN for local scale r_i
    radius_scale: float = 1.5,  # ρ_i = scale · r_i
    min_points: int = 60,  # drop tiny clusters
    max_show: int = PSEG_MAX_SHOW,
) -> List[o3d.geometry.PointCloud]:
    r"""
    Region-growing segmentation directly on a point cloud.

    Model
    -----
    For each point x_i with normal n_i and local scale r_i:
        r_i = median_k(||x_t - x_i||₂),  t ∈ kNN(i)
        ρ_i = s · r_i,  s = radius_scale
    Grow edges (i,j) if:
        ||x_j - x_i||₂ ≤ ρ_i  and  n_i · n_j ≥ cos(θ),  θ = angle_deg
    Connected components of the resulting undirected graph are segments.

    Notes
    -----
    - Adapts radius per point (handles nonuniform sampling).
    - Normal similarity avoids merging adjacent but differently oriented bars.
    """
    if len(pcd.points) == 0:
        LOG.warning("[P-RG] empty cloud; nothing to do")
        return []

    _ensure_normals(pcd)
    P = np.asarray(pcd.points, dtype=np.float64)
    N = np.asarray(pcd.normals, dtype=np.float64)

    # Guard against zero-normals
    nn = np.linalg.norm(N, axis=1, keepdims=True)
    np.divide(N, nn, out=N, where=(nn > 1e-12))

    r = estimate_local_radii(P, k=k_radius, scale=1.0)
    rho = np.maximum(MIN_SEG_LEN, radius_scale * r)

    LOG.info(
        f"[P-RG] angle={angle_deg:.1f}° k={k_radius} scale={radius_scale:.2f} min_pts={int(min_points)}"
    )

    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    )
    cos_thr = float(np.cos(np.deg2rad(angle_deg)))

    # Build symmetric adjacency without duplicates
    n = len(P)
    adj: List[set[int]] = [set() for _ in range(n)]
    for i, p in enumerate(P):
        try:
            _, idx, _ = tree.search_radius_vector_3d(p, float(rho[i]))
        except Exception:
            idx = [i]
        for j in idx:
            if j == i:
                continue
            if float(np.dot(N[i], N[j])) >= cos_thr:
                adj[i].add(j)
                adj[j].add(i)

    # Connected components (BFS)
    visited = np.zeros(n, dtype=bool)
    comps: List[np.ndarray] = []
    for s in range(n):
        if visited[s]:
            continue
        q = [s]
        cur = []
        visited[s] = True
        while q:
            u = q.pop()
            cur.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        if len(cur) >= int(min_points):
            comps.append(np.asarray(cur, dtype=np.int64))

    if not comps:
        LOG.warning("[P-RG] no clusters after growing")
        return []

    comps.sort(key=lambda a: a.size, reverse=True)
    if len(comps) > max_show:
        LOG.warning(f"[P-RG] too many ({len(comps)}), showing top {max_show}")
        comps = comps[:max_show]

    colors = _distinct_colors(len(comps))
    segs: List[o3d.geometry.PointCloud] = []
    for i, idx in enumerate(comps):
        seg = pcd.select_by_index(idx.tolist())
        seg.paint_uniform_color(tuple(colors[i]))
        LOG.debug(f"[P-RG] cluster#{i}: {len(idx)} pts")
        segs.append(seg)

    LOG.info(f"[P-RG] clusters_shown={len(segs)} / found={len(comps)}")
    return segs


# --------------------------- Public API --------------------------------------


def build_surface_and_graphs(raw: o3d.geometry.PointCloud):
    """
    Return:
      surface_mesh,
      graph_pcd,    nodes_pcd,
      graph_ridges, nodes_ridges,
      graph_skel,   nodes_skel,
      surface_segments (list of meshes),
      point_segments   (list of point clouds)
    """
    pcd = preprocess_cloud(raw)
    surf = reconstruct_mesh(pcd)
    surf.paint_uniform_color((0.82, 0.82, 0.82))

    g2, n2 = build_graph_pcd(pcd, surf)
    g3, n3 = build_graph_ridges(pcd, surf)
    g4, n4 = build_graph_skeleton(pcd, surf)

    segs_mesh = segment_surface_regions(
        surf, angle_deg=SEG_NORMAL_ANGLE_DEG, min_tris=SEG_MIN_TRIS
    )
    segs_pts = segment_point_cloud_region_growing(
        pcd,
        angle_deg=25.0,
        k_radius=16,
        radius_scale=1.6,
        min_points=60,
    )

    if len(g2.lines) > 0:
        g2.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.10, 0.50, 0.90]]), (len(g2.lines), 1))
        )
    if len(g3.lines) > 0:
        g3.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.10, 0.70, 0.30]]), (len(g3.lines), 1))
        )
    if len(g4.lines) > 0:
        g4.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.60, 0.25, 0.85]]), (len(g4.lines), 1))
        )
    for nm in (n2, n3, n4):
        nm.paint_uniform_color((0.9, 0.2, 0.1))
    LOG.info("[API] surface+graphs+segments ready")
    return surf, g2, n2, g3, n3, g4, n4, segs_mesh, segs_pts


def mode7_geometries(
    raw: o3d.geometry.PointCloud,
) -> List[o3d.geometry.Geometry]:
    from all import skeletonize_em

    lines, balls = skeletonize_em(raw)
    return [lines, balls]


# --------------------------- Demo --------------------------------------------

if __name__ == "__main__":
    from main import _load_cloud
    from vision.skeleton.config import SkelPipelineCfg

    cfg = SkelPipelineCfg()
    raw_any = _load_cloud(cfg)
    raw = o3d.geometry.PointCloud(_load_cloud(cfg))
    LOG.info("[Demo] start viewer")
    if isinstance(raw_any, o3d.geometry.PointCloud):
        raw = raw_any
    else:
        raw = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.asarray(raw_any, dtype=np.float64))
        )
    surf, g2, n2, g3, n3, g4, n4, segs_mesh, segs_pts = (
        build_surface_and_graphs(raw)
    )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("3D Graphs + Segments", 1280, 800)
    mode = {"cur": 1}

    def redraw():
        vis.clear_geometries()
        if mode["cur"] == 1:
            vis.add_geometry(surf, True)
        elif mode["cur"] == 2:
            vis.add_geometry(surf, True)
            vis.add_geometry(g2, False)
            vis.add_geometry(n2, False)
        elif mode["cur"] == 3:
            vis.add_geometry(surf, True)
            vis.add_geometry(g3, False)
            vis.add_geometry(n3, False)
        elif mode["cur"] == 4:
            vis.add_geometry(surf, True)
            vis.add_geometry(g4, False)
            vis.add_geometry(n4, False)
        elif mode["cur"] == 5:
            for m in segs_mesh:
                vis.add_geometry(m, False)
        elif mode["cur"] == 6:
            for pc in segs_pts:
                vis.add_geometry(pc, False)
        elif mode["cur"] == 7:
            for g in mode7_geometries(raw):
                vis.add_geometry(g, False)
        vis.update_renderer()
        return True

    vis.register_key_callback(
        ord('1'), lambda _: (mode.update(cur=1), redraw())[1]
    )
    vis.register_key_callback(
        ord('2'), lambda _: (mode.update(cur=2), redraw())[1]
    )
    vis.register_key_callback(
        ord('3'), lambda _: (mode.update(cur=3), redraw())[1]
    )
    vis.register_key_callback(
        ord('4'), lambda _: (mode.update(cur=4), redraw())[1]
    )
    vis.register_key_callback(
        ord('5'), lambda _: (mode.update(cur=5), redraw())[1]
    )
    vis.register_key_callback(
        ord('6'), lambda _: (mode.update(cur=6), redraw())[1]
    )
    vis.register_key_callback(
        ord("7"), lambda _: (mode.update(cur=7), redraw())[1]
    )
    redraw()
    vis.run()
    vis.destroy_window()
