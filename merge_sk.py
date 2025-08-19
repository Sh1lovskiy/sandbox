# merge_sk.py
# viewer keys:
#   1: cloud           (raw + smoothed)
#   2: skeleton        (raw skeleton polylines)
#   3: graph           (skeleton + node spheres)
#   4: centers         (refined centerlines)
#   5: members         (centerlines grouped by angle to main plane)
#   6: normals         (point colors by angle between point normal and main plane normal)
#
# Modes:
#   - "plane2d": detects a dominant plane and skeletonizes in 2D
#   - "vol3d" : global voxelization + 3D skeletonization (no OBB)
#
# What's new in this version:
#   - Key "6": per-point coloring by normal deviation from the main plane normal.
#     Palette matches the angle bins used for member grouping (0–15, 15–35, 35–55, 55–75, 75–90 deg).
#   - Saves: <tag>_normals_angle_colored.ply and <tag>_normals_info.json in debug folder.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json, math, logging, glob
import numpy as np
import open3d as o3d

# =============================== CONSTANTS ===================================
CAPTURE_ROOT = Path("captures/20250813_143934")
INPUT_CLOUD_PATH: Optional[Path] = None  # auto-discover if None

MODE = "vol3d"  # "plane2d" | "vol3d"

# poses (for plane2d camera-nearest plane selection)
USE_CAPTURE_POSES_FOR_CAMERA = True
POSES_JSON = "poses.json"

# hand-eye (must match merge.py)
HAND_EYE_R = np.array(
    [
        [0.999048, 0.00428, -0.00625],
        [-0.00706, 0.99658, -0.00804],
        [0.00423, 0.00895, 0.99629],
    ],
    dtype=np.float64,
)
HAND_EYE_T = np.array([-0.036, -0.078, 0.006], dtype=np.float64)  # meters

# ---------- plane2d params ----------
DS_VOX_SEG = 0.0035
DIST_SCHEDULE = [0.003, 0.005, 0.008, 0.012, 0.02]
PLANE_RANSAC_N = 3
PLANE_ITERATIONS = 4000
MAX_PLANE_CANDIDATES = 6
MIN_PLANE_POINTS_ABS = 500
MIN_PLANE_POINTS_FRAC = 0.03
GRID_RES = 0.002
MORPH_CLOSE_KERNEL = 3
SKELETON_2D_METHOD = "skimage"  # "skimage" | "naive"

# ---------- vol3d base params ----------
VOXEL_3D = 0.0035
PAD_VOX = 3
MAX_3D_VOXELS = 60_000_000
DILATE_3D_RADIUS = 1
MIN_SKEL_VOXELS = 200

# ---------- resampling ----------
RESAMPLE_STEP_M = 0.10
KEEP_TAIL_MIN_FRAC = 0.7

# ---------- node refinement via normals ----------
BRIDGE_WIDTH_EST = 0.07
NODE_NORMAL_REFINE_RADIUS = 0.03
NODE_MIN_SIDE_PTS = 20
NORMAL_EST_RADIUS = 0.012
NORMAL_EST_MAXNN = 60

# --- centerline refinement ---
CENTER_SEARCH_R = 0.5 * BRIDGE_WIDTH_EST
CENTER_MIN_SIDE_PTS = 20

# ---------- node merge in world ----------
NODE_MERGE_RADIUS_M = 0.010

# ---------- smoothing (adaptive MLS) ----------
ENABLE_SMOOTHING = True
SMOOTH_ITERS = 2
SMOOTH_RADIUS = 0.010
SMOOTH_MIN_NN = 20
SMOOTH_ALPHA_MIN = 0.15
SMOOTH_ALPHA_MAX = 1.00

# --- skeleton endpoints extension (kept as before; not used by default here) ---
EXTEND_SKELETON = True
EXTEND_STEP_M = 0.005
EXTEND_MAX_M = 0.15
EXTEND_SUPPORT_R = 0.04
EXTEND_MIN_SUPPORT = 25
EXTEND_MAX_MEAN_DIST = 0.010

# ---------- FPFH for smoothing weighting ----------
USE_FPFH_FOR_SMOOTH = True
FPFH_RADIUS = 0.020
FPFH_MAX_NN = 100
FPFH_DOWNSAMPLE = 0.006

# ---------- straightening ----------
STRAIGHTEN_BRIDGES = True
STRAIGHT_MIN_LEN = 0.08
STRAIGHT_LINEARITY_THR = 0.08

# --- side-surface region growing (normals + spatial connectivity) ---
SIDE_REGION_RADIUS = 0.015  # meters, neighbor radius for components
SIDE_NORMAL_THR_DEG = 18.0  # max normal-angle difference within a component
SIDE_MIN_REGION_PTS = 250  # drop tiny residuals
SIDE_BOUNDARY_K = 24  # k-NN for boundary detection
SIDE_BOUNDARY_FRAC = 0.75  # if < this fraction of k are same region => boundary

# viz / export
DEBUG_DIR_NAME = "debug"
NODE_SPHERE_R = 0.004
LOG_LEVEL = logging.INFO

# ================================ LOGGING ====================================
logger = logging.getLogger("merge_sk")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(LOG_LEVEL)
np.set_printoptions(suppress=True, precision=6, linewidth=180)


# ================================ HELPERS ====================================
@dataclass
class Pose:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float  # degrees


def euler_deg_to_R(rx, ry, rz, order="XYZ") -> np.ndarray:
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array(
        [[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]]
    )
    Ry = np.array(
        [[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]]
    )
    Rz = np.array(
        [[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]]
    )
    d = dict(X=Rx, Y=Ry, Z=Rz)
    R = np.eye(3)
    for ch in order:
        R = d[ch] @ R
    return R


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def load_poses(path: Path) -> Dict[str, Pose]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for k, v in data.items():
        out[k] = Pose(
            float(v["x"]),
            float(v["y"]),
            float(v["z"]),
            float(v["rx"]),
            float(v["ry"]),
            float(v["rz"]),
        )
    return out


def camera_centers_from_poses(poses: Dict[str, Pose]) -> List[np.ndarray]:
    if not poses:
        return []
    T_tcp_cam = make_T(HAND_EYE_R, HAND_EYE_T)
    centers = []
    for k in sorted(poses.keys()):
        p = poses[k]
        R_tcp = euler_deg_to_R(p.rx, p.ry, p.rz, "XYZ")
        t_tcp = np.array([p.x, p.y, p.z]) * 0.001  # mm -> m
        T_base_tcp = make_T(R_tcp, t_tcp)
        T_base_cam = T_base_tcp @ T_tcp_cam
        centers.append(T_base_cam[:3, 3].copy())
    logger.info(f"[POSES] camera centers: {len(centers)}")
    return centers


def find_input_cloud() -> Path:
    if INPUT_CLOUD_PATH is not None:
        return INPUT_CLOUD_PATH
    dbg = CAPTURE_ROOT / DEBUG_DIR_NAME
    for c in [dbg / "final_merged.ply", dbg / "merged.ply"]:
        if c.exists():
            return c
    any_ply = sorted(glob.glob(str(dbg / "*.ply")))
    if any_ply:
        return Path(any_ply[0])
    raise FileNotFoundError(
        f"No input cloud found in {dbg}. Expected final_merged.ply or merged.ply."
    )


def downsample(pcd: o3d.geometry.PointCloud, vox: float) -> o3d.geometry.PointCloud:
    return pcd.voxel_down_sample(voxel_size=vox) if vox and vox > 0 else pcd


def ensure_normals(
    pcd: o3d.geometry.PointCloud, radius=NORMAL_EST_RADIUS, max_nn=NORMAL_EST_MAXNN
):
    if not pcd.has_normals() or len(pcd.normals) != len(pcd.points):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )


# ------------------------------ tiny FPFH bits (for smoothing weights) -------
def compute_fpfh(
    pcd: o3d.geometry.PointCloud, radius: float = FPFH_RADIUS, max_nn: int = FPFH_MAX_NN
) -> np.ndarray:
    """Returns (N,33) FPFH (Open3D), L1-normalized per point."""
    ensure_normals(pcd)
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    F = np.asarray(feat.data).T  # (33,N)->(N,33)
    s = F.sum(1, keepdims=True) + 1e-12
    return F / s


def fpfh_entropy_weight(F: np.ndarray) -> np.ndarray:
    """Low entropy -> planar/simple -> high weight in [0-1]."""
    N, D = F.shape
    H = -np.sum(F * (np.log(F + 1e-12)), axis=1)
    Hn = H / math.log(D)
    W = 1.0 - Hn
    return np.clip(W, 0.0, 1.0)


def map_weights_nn(
    src_pts: np.ndarray, dst_pts: np.ndarray, w_src: np.ndarray
) -> np.ndarray:
    """Nearest-neighbor map from downsampled weights."""
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts))
    kdt = o3d.geometry.KDTreeFlann(src)
    out = np.zeros((len(dst_pts),), float)
    for i, p in enumerate(dst_pts):
        _, idx, _ = kdt.search_knn_vector_3d(p, 1)
        out[i] = w_src[idx[0]]
    return out


# ------------------------------ adaptive MLS smoothing -----------------------
def smooth_cloud_mls_plane_adaptive(
    pcd: o3d.geometry.PointCloud,
    radius: float = SMOOTH_RADIUS,
    iters: int = SMOOTH_ITERS,
    min_nn: int = SMOOTH_MIN_NN,
    alpha_min: float = SMOOTH_ALPHA_MIN,
    alpha_max: float = SMOOTH_ALPHA_MAX,
    use_fpfh: bool = USE_FPFH_FOR_SMOOTH,
) -> o3d.geometry.PointCloud:
    """
    Iteratively projects each point onto a local PCA plane. The projection
    strength α_i ∈ [alpha_min, alpha_max] is adapted with:
      - FPFH entropy (lower entropy -> closer to plane),
      - local planarity from covariance eigenvalues.
    """
    P0 = np.asarray(pcd.points)
    P = P0.copy()

    if use_fpfh:
        pcd_ds = downsample(pcd, FPFH_DOWNSAMPLE) if FPFH_DOWNSAMPLE else pcd
        F_ds = compute_fpfh(pcd_ds)
        W_ds = fpfh_entropy_weight(F_ds)  # 0-1
        W_fpfh = map_weights_nn(np.asarray(pcd_ds.points), P, W_ds)
    else:
        W_fpfh = np.zeros((len(P),), float)

    pts_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    kdt = o3d.geometry.KDTreeFlann(pts_pc)
    sigma2 = (radius * 0.5) ** 2

    for _ in range(max(0, iters)):
        P_new = P.copy()
        for i in range(len(P)):
            _, idx, _ = kdt.search_radius_vector_3d(P[i], radius)
            if len(idx) < min_nn:
                continue
            Q = P[idx]
            d = np.linalg.norm(Q - P[i], axis=1)
            w = np.exp(-(d * d) / (2.0 * sigma2)) + 1e-8

            c = (Q * w[:, None]).sum(0) / w.sum()
            A = (Q - c) * np.sqrt(w[:, None])
            try:
                _, S, Vt = np.linalg.svd(A, full_matrices=False)
                n = Vt[-1]
            except Exception:
                continue

            lam = (S**2) / (len(Q) - 1 + 1e-9)
            lam = np.pad(lam, (0, max(0, 3 - len(lam))), constant_values=0)
            sv = lam[-1] / (lam.sum() + 1e-12)
            w_eig = 1.0 - float(sv)

            w_pl = max(w_eig, float(W_fpfh[i]))
            alpha = alpha_min + (alpha_max - alpha_min) * w_pl
            off = float((P[i] - c) @ n)
            P_new[i] = P[i] - alpha * off * n

        P = P_new
        pts_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        kdt = o3d.geometry.KDTreeFlann(pts_pc)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(P)
    return out


# ------------------------------ plane2d helpers ------------------------------
def segment_planes_o3d(pcd, dist, ransac_n, iters):
    model, inliers = pcd.segment_plane(dist, ransac_n, iters)
    return np.array(model, dtype=float), np.asarray(inliers, dtype=int)


def segment_planes_iterative(pcd, max_planes, dist, ransac_n, iters, min_pts):
    planes = []
    rest = pcd
    for i in range(max_planes):
        if len(rest.points) < min_pts:
            break
        m, inl = segment_planes_o3d(rest, dist, ransac_n, iters)
        if inl.size < min_pts:
            break
        planes.append((m, inl))
        rest = rest.select_by_index(inl.tolist(), invert=True)
        logger.info(f"[RANSAC] plane#{i}: {inl.size} inliers, rest={len(rest.points)}")
    return planes


def reselect_inliers_full(model, cloud_full, dist):
    a, b, c, d = model
    n = np.array([a, b, c], float)
    nn = np.linalg.norm(n) + 1e-12
    P = np.asarray(cloud_full.points)
    dists = np.abs(P @ n + d) / nn
    return np.where(dists <= dist)[0]


def pick_nearest_plane_to_cameras(planes, cloud_full, cam_centers, dist_for_reselect):
    if not planes:
        raise RuntimeError("No plane candidates.")
    if not cam_centers:
        best = max(planes, key=lambda P: P[1].size)
        inl_full = reselect_inliers_full(best[0], cloud_full, dist_for_reselect)
        logger.info("[PICK] no camera centers; using largest support.")
        return best[0], inl_full
    best_m, best_s = None, 1e9
    for i, (m, inl) in enumerate(planes):
        a, b, c, d = m
        n = np.array([a, b, c])
        nn = np.linalg.norm(n) + 1e-12
        s = min(abs(a * x + b * y + c * z + d) / nn for (x, y, z) in cam_centers)
        logger.info(
            f"[PICK] plane#{i} min-dist-to-cam={s:.4f} m | inliers(ds)={inl.size}"
        )
        if s < best_s:
            best_s, best_m = s, m
    inl_full = reselect_inliers_full(best_m, cloud_full, dist_for_reselect)
    return best_m, inl_full


def plane_basis_from_coeffs(m, pts):
    a, b, c, d = m
    n = np.array([a, b, c], float)
    n /= np.linalg.norm(n) + 1e-12
    p0 = pts.mean(0)
    ref = np.array([0, 0, 1]) if abs(n[2]) < 0.9 else np.array([1, 0, 0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return u, v, n, p0


def to_plane_uv(pts, u, v, p0):
    q = pts - p0
    return np.stack([q @ u, q @ v], axis=1)


# ------------------------------ skeletonization (2D/3D) ----------------------
def rasterize_points(uv, res):
    umin, vmin = uv.min(0)
    umax, vmax = uv.max(0)
    W = max(1, int(np.ceil((umax - umin) / res)) + 3)
    H = max(1, int(np.ceil((vmax - vmin) / res)) + 3)
    img = np.zeros((H, W), np.uint8)
    ui = np.clip(((uv[:, 0] - umin) / res).astype(int), 0, W - 1)
    vi = np.clip(((uv[:, 1] - vmin) / res).astype(int), 0, H - 1)
    img[vi, ui] = 1
    return img, (umin, vmin)


def morph_close(img, k):
    try:
        import cv2

        k = max(1, int(k))
        k = k if (k % 2) == 1 else k + 1
        kernel = np.ones((k, k), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    except Exception:
        return img


def skeletonize_2d(img, method="skimage"):
    if method == "skimage":
        try:
            from skimage.morphology import skeletonize

            return skeletonize(img.astype(bool)).astype(np.uint8)
        except Exception as e:
            logger.warning(f"[SKEL-2D] skimage not available ({e}); fallback to naive.")

    # fallback thinning
    def neighbors8(m, r, c):
        return [
            m[r - 1, c],
            m[r - 1, c + 1],
            m[r, c + 1],
            m[r + 1, c + 1],
            m[r + 1, c],
            m[r + 1, c - 1],
            m[r, c - 1],
            m[r - 1, c - 1],
        ]

    sk = img.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for pass_id in [0, 1]:
            to_del = []
            for r in range(1, sk.shape[0] - 1):
                for c in range(1, sk.shape[1] - 1):
                    if sk[r, c] == 0:
                        continue
                    nb = neighbors8(sk, r, c)
                    C = sum((nb[i] == 0 and nb[(i + 1) % 8] == 1) for i in range(8))
                    N = sum(nb)
                    if 2 <= N <= 6 and C == 1:
                        if pass_id == 0:
                            if (
                                nb[0] * nb[2] * nb[4] == 0
                                and nb[2] * nb[4] * nb[6] == 0
                            ):
                                to_del.append((r, c))
                        else:
                            if (
                                nb[0] * nb[2] * nb[6] == 0
                                and nb[0] * nb[4] * nb[6] == 0
                            ):
                                to_del.append((r, c))
            for r, c in to_del:
                sk[r, c] = 0
                changed = True
    return sk


def skeleton_to_graph_2d(sk):
    H, W = sk.shape
    nbrs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    deg = {}
    pixels = np.argwhere(sk == 1)
    for r, c in pixels:
        d = 0
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and sk[rr, cc]:
                d += 1
        deg[(r, c)] = d
    node_px = [rc for rc, d in deg.items() if d != 2]
    node_map = {rc: i for i, rc in enumerate(node_px)}
    edges, polylines, visited = [], [], set()
    for r, c in node_px:
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W) or sk[rr, cc] == 0:
                continue
            a, b = r, c
            u, v = rr, cc
            if (a, b, u, v) in visited:
                continue
            path = [(a, b), (u, v)]
            while True:
                nxt = None
                cnt = 0
                for ddr, ddc in nbrs:
                    r2, c2 = u + ddr, v + ddc
                    if (
                        0 <= r2 < H
                        and 0 <= c2 < W
                        and sk[r2, c2] == 1
                        and (r2, c2) != (a, b)
                    ):
                        cnt += 1
                        nxt = (r2, c2)
                if (u, v) in node_map and (u, v) != (a, b):
                    break
                if cnt == 0:
                    break
                a, b, u, v = u, v, nxt[0], nxt[1]
                path.append((u, v))
            visited.add((path[0][0], path[0][1], path[1][0], path[1][1]))
            s, e = path[0], path[-1]
            if s in node_map and e in node_map and s != e:
                edges.append((node_map[s], node_map[e]))
                polylines.append(path)
    return node_px, edges, node_map, polylines


# ------------------------------ centerline refinement ------------------------
def refine_polylines_centers(
    polylines_world: List[np.ndarray],
    cloud: o3d.geometry.PointCloud,
    search_r: float = CENTER_SEARCH_R,
    min_side_pts: int = CENTER_MIN_SIDE_PTS,
    exp_width: float = BRIDGE_WIDTH_EST,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    At each poly point: take neighbors within search_r, split into two opposite
    sides by normals, and take midpoint. Returns refined polylines and local widths.
    """
    ensure_normals(cloud)
    P = np.asarray(cloud.points)
    N = np.asarray(cloud.normals)
    kdt = o3d.geometry.KDTreeFlann(cloud)

    out_polys, widths = [], []
    for poly in polylines_world:
        if len(poly) == 0:
            continue
        refined = []
        w_arr = []
        for p in poly:
            _, idx, _ = kdt.search_radius_vector_3d(p, search_r)
            if len(idx) < min_side_pts * 2:
                refined.append(p)
                w_arr.append(np.nan)
                continue
            Q = P[idx]
            M = N[idx]
            good = np.isfinite(M).all(1)
            Q = Q[good]
            M = M[good]
            if len(Q) < min_side_pts * 2:
                refined.append(p)
                w_arr.append(np.nan)
                continue
            Mm = M - M.mean(0)
            try:
                _, _, Vt = np.linalg.svd(Mm, full_matrices=False)
                dirn = Vt[0]
            except Exception:
                refined.append(p)
                w_arr.append(np.nan)
                continue
            s = M @ dirn
            plus = s >= 0
            minus = ~plus
            if plus.sum() < min_side_pts or minus.sum() < min_side_pts:
                refined.append(p)
                w_arr.append(np.nan)
                continue
            c_plus = Q[plus].mean(0)
            c_minus = Q[minus].mean(0)
            sep = float(np.linalg.norm(c_plus - c_minus))
            if not (0.3 * exp_width <= sep <= 2.2 * exp_width):
                refined.append(p)
                w_arr.append(np.nan)
                continue
            refined.append(0.5 * (c_plus + c_minus))
            w_arr.append(sep)
        out_polys.append(np.asarray(refined))
        widths.append(np.asarray(w_arr))
    return out_polys, widths


def make_lineset_from_world_polylines(polys_world, color=(0, 1, 0)):
    pts_all, lines_all, base = [], [], 0
    for arr in polys_world:
        if len(arr) < 2:
            continue
        n = len(arr)
        pts_all.append(arr)
        lines_all.append(
            np.column_stack(
                [np.arange(base, base + n - 1), np.arange(base + 1, base + n)]
            )
        )
        base += n
    if not pts_all:
        return o3d.geometry.LineSet()
    P = np.vstack(pts_all)
    L = np.vstack(lines_all).astype(np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(P)
    ls.lines = o3d.utility.Vector2iVector(L)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, float), (len(L), 1)))
    return ls


def make_nodes_mesh(nodes_xyz: np.ndarray, color=(1, 0, 0), r: float = NODE_SPHERE_R):
    mesh = o3d.geometry.TriangleMesh()
    for p in nodes_xyz:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        s.translate(p)
        s.compute_vertex_normals()
        s.paint_uniform_color(color)
        mesh += s
    return mesh


def resample_polyline_world(
    P: np.ndarray, step: float, keep_tail_min_frac: float = 0.7
) -> np.ndarray:
    if len(P) < 2:
        return P
    total = float(np.sum(np.linalg.norm(P[1:] - P[:-1], axis=1)))
    if total < step:
        return P
    out = [P[0]]
    acc = 0.0
    target = step
    for i in range(1, len(P)):
        seg = P[i] - P[i - 1]
        L = float(np.linalg.norm(seg))
        if L <= 1e-12:
            continue
        dirv = seg / L
        while acc + L >= target:
            d = target - acc
            out.append(P[i - 1] + dirv * d)
            target += step
        acc += L
    tail = acc - ((len(out) - 1) * step)
    if tail >= keep_tail_min_frac * step:
        out.append(P[-1])
    if len(out) < 2:
        return P
    return np.asarray(out)


def lineset_has_points(ls) -> bool:
    try:
        return np.asarray(ls.points).shape[0] > 0
    except Exception:
        return False


# ------------------------------ straightening --------------------------------
def _fit_line_svd(X: np.ndarray):
    c = X.mean(0)
    A = X - c
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    d = Vt[0]
    t = (X - c) @ d
    p0 = c + t.min() * d
    p1 = c + t.max() * d
    lam = (S**2) / (len(X) - 1 + 1e-12)
    lam = np.pad(lam, (0, max(0, 3 - len(lam))), constant_values=0)
    linearity = (lam[1] + lam[2]) / (lam[0] + 1e-12)
    length = float(np.linalg.norm(p1 - p0))
    return p0, p1, length, float(linearity)


def straighten_polylines(
    polys: List[np.ndarray],
    min_len: float = STRAIGHT_MIN_LEN,
    lin_thr: float = STRAIGHT_LINEARITY_THR,
    step: float = RESAMPLE_STEP_M,
) -> List[np.ndarray]:
    out = []
    for P in polys:
        if len(P) < 2:
            out.append(P)
            continue
        p0, p1, L, lin = _fit_line_svd(P)
        if (L >= min_len) and (lin <= lin_thr):
            npts = max(2, int(round(L / max(1e-6, step))) + 1)
            t = np.linspace(0.0, 1.0, npts)
            out.append((p0[None, :] * (1 - t)[:, None] + p1[None, :] * t[:, None]))
        else:
            out.append(P)
    return out


# ------------------------------ vol3d helpers --------------------------------
def compute_voxel_grid_params(P, vsize, pad_vox, max_voxels):
    mn = P.min(0)
    mx = P.max(0)
    span = mx - mn
    dims = np.maximum(1, np.ceil(span / vsize).astype(int)) + 1 + 2 * pad_vox
    voxels = int(dims.prod())
    max_3d = max_voxels
    if voxels > max_3d:
        scale = (voxels / max_3d) ** (1 / 3)
        vsize_new = float(vsize * scale * 1.1)
        dims = np.maximum(1, np.ceil(span / vsize_new).astype(int)) + 1 + 2 * pad_vox
        logger.info(
            f"[VOX] resized voxel size {vsize:.4f} -> {vsize_new:.4f} to fit {max_3d} voxels"
        )
        vsize = vsize_new
    origin = mn - pad_vox * vsize
    dims = np.maximum(1, np.ceil((mx - origin)) / vsize).astype(int) + 1
    return origin, vsize, dims.astype(int)


def voxelize_points(P, origin, vsize, dims):
    I = np.floor((P - origin) / vsize).astype(int)
    valid = (
        (I[:, 0] >= 0)
        & (I[:, 1] >= 0)
        & (I[:, 2] >= 0)
        & (I[:, 0] < dims[0])
        & (I[:, 1] < dims[1])
        & (I[:, 2] < dims[2])
    )
    I = I[valid]
    vol = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint8)  # z,y,x
    vol[I[:, 2], I[:, 1], I[:, 0]] = 1
    return vol


def dilate_3d(vol, r):
    if r <= 0:
        return vol
    try:
        from skimage.morphology import ball, binary_dilation

        se = ball(int(r))
        return binary_dilation(vol.astype(bool), se).astype(np.uint8)
    except Exception as e:
        logger.warning(f"[VOX] dilation skipped ({e})")
        return vol


def skeletonize_3d_volume(vol):
    try:
        from skimage.morphology import skeletonize_3d as _sk3d

        return _sk3d(vol.astype(bool)).astype(np.uint8)
    except Exception:
        try:
            from skimage.morphology import skeletonize as _sk2d

            sk = np.zeros_like(vol)
            for z in range(vol.shape[0]):
                sk[z] = _sk2d(vol[z].astype(bool)).astype(np.uint8)
            return sk
        except Exception as e:
            raise RuntimeError(
                f"scikit-image not available for 3D skeletonization: {e}"
            )


def skeleton_nodes_vox(sk) -> List[Tuple[int, int, int]]:
    Z, Y, X = sk.shape
    vox = np.argwhere(sk == 1)
    if len(vox) == 0:
        return []
    S = set(map(tuple, vox.tolist()))
    offs = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    def degree(v):
        z, y, x = v
        d = 0
        for dz, dy, dx in offs:
            w = (z + dz, y + dy, x + dx)
            if 0 <= w[0] < Z and 0 <= w[1] < Y and 0 <= w[2] < X and (w in S):
                d += 1
        return d

    return [v for v in S if degree(v) != 2]


def skeleton_to_polylines_3d(sk):
    Z, Y, X = sk.shape
    vox = np.argwhere(sk == 1)
    if len(vox) == 0:
        return []
    S = set(map(tuple, vox.tolist()))
    offs = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    def nbrs(v):
        z, y, x = v
        out = []
        for dz, dy, dx in offs:
            w = (z + dz, y + dy, x + dx)
            if 0 <= w[0] < Z and 0 <= w[1] < Y and 0 <= w[2] < X and (w in S):
                out.append(w)
        return out

    def degree(v):
        return len(nbrs(v))

    visited_edges = set()
    polylines = []

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    def trace(a, b):
        path = [a, b]
        prev, cur = a, b
        while True:
            nb = [w for w in nbrs(cur) if w != prev]
            if degree(cur) != 2 or len(nb) == 0:
                break
            nxt = nb[0]
            if edge_key(cur, nxt) in visited_edges:
                break
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    nodes = [v for v in S if degree(v) != 2]
    for v in nodes:
        for u in nbrs(v):
            e = edge_key(v, u)
            if e in visited_edges:
                continue
            path = trace(v, u)
            for i in range(len(path) - 1):
                visited_edges.add(edge_key(path[i], path[i + 1]))
            polylines.append(path)

    # catch pure cycles
    for v in S:
        for u in nbrs(v):
            e = edge_key(v, u)
            if e in visited_edges:
                continue
            path = trace(v, u)
            for i in range(len(path) - 1):
                visited_edges.add(edge_key(path[i], path[i + 1]))
            polylines.append(path)

    return polylines


# ----------------------- node merge / refine ---------------------------------
def merge_close_nodes_world(
    nodes_xyz: np.ndarray, edges: List[Tuple[int, int]], radius: float
):
    n = len(nodes_xyz)
    if n == 0:
        return nodes_xyz, edges, {i: i for i in range(n)}
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[parent[a]]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(nodes_xyz[i] - nodes_xyz[j]) < radius:
                union(i, j)
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    new_index = {}
    new_nodes = []
    for _, idxs in groups.items():
        new_idx = len(new_nodes)
        new_index.update({i: new_idx for i in idxs})
        new_nodes.append(np.mean(nodes_xyz[idxs], axis=0))
    new_nodes = np.asarray(new_nodes)
    new_edges_set = set()
    for u, v in edges:
        uu, vv = new_index[u], new_index[v]
        if uu == vv:
            continue
        a, b = (uu, vv) if uu < vv else (vv, uu)
        new_edges_set.add((a, b))
    new_edges = sorted(list(new_edges_set))
    return new_nodes, new_edges, new_index


def refine_nodes_by_normals(
    nodes_xyz: np.ndarray,
    cloud: o3d.geometry.PointCloud,
    search_radius: float = NODE_NORMAL_REFINE_RADIUS,
    exp_width: float = BRIDGE_WIDTH_EST,
    min_side_pts: int = NODE_MIN_SIDE_PTS,
) -> np.ndarray:
    """Shift node to mid-bridge using two opposite normal clusters."""
    if len(nodes_xyz) == 0:
        return nodes_xyz
    ensure_normals(cloud)
    P = np.asarray(cloud.points)
    N = np.asarray(cloud.normals)
    kdt = o3d.geometry.KDTreeFlann(cloud)
    out = []
    for p in nodes_xyz:
        _, idx, _ = kdt.search_radius_vector_3d(p, search_radius)
        if len(idx) < min_side_pts * 2:
            out.append(p)
            continue
        Q = P[idx]
        M = N[idx]
        good = np.isfinite(M).all(1)
        Q = Q[good]
        M = M[good]
        if len(Q) < min_side_pts * 2:
            out.append(p)
            continue
        Mm = M - M.mean(0)
        try:
            _, _, Vt = np.linalg.svd(Mm, full_matrices=False)
            dirn = Vt[0]
        except Exception:
            out.append(p)
            continue
        s = M @ dirn
        plus = s >= 0
        minus = ~plus
        if plus.sum() < min_side_pts or minus.sum() < min_side_pts:
            out.append(p)
            continue
        c_plus = Q[plus].mean(0)
        c_minus = Q[minus].mean(0)
        sep = float(np.linalg.norm(c_plus - c_minus))
        if not (0.4 * exp_width <= sep <= 1.8 * exp_width):
            out.append(p)
            continue
        out.append(0.5 * (c_plus + c_minus))
    return np.asarray(out)


# ----------------------- grouping / members by angle -------------------------
def estimate_dominant_plane_normal(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Try RANSAC on downsample. Fallback to last PCA axis."""
    ds = downsample(pcd, 0.01)
    try:
        model, inl = ds.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1500
        )
        n = np.array(model[:3], float)
        n /= np.linalg.norm(n) + 1e-12
        return n
    except Exception:
        P = np.asarray(ds.points)
        c = P.mean(0)
        A = P - c
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        n = Vt[-1]
        n /= np.linalg.norm(n) + 1e-12
        return n


def group_polylines_by_angle(
    polys: List[np.ndarray], plane_normal: np.ndarray, bins_deg=(15, 35, 55, 75)
) -> List[List[np.ndarray]]:
    """
    Put polylines into angle bins by |dot(dir, n)|, angle in [0-90].
    """
    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    edges = list(bins_deg) + [90.0]
    groups = [[] for _ in range(len(edges))]
    for P in polys:
        if len(P) < 2:
            continue
        d = P[-1] - P[0]
        if np.linalg.norm(d) < 1e-9:
            continue
        d = d / np.linalg.norm(d)
        ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(d, n))))))
        put = False
        for gi, hi in enumerate(edges):
            if ang <= hi + 1e-6:
                groups[gi].append(P)
                put = True
                break
        if not put:
            groups[-1].append(P)
    return groups


def make_group_linesets(groups: List[List[np.ndarray]]) -> List[o3d.geometry.LineSet]:
    palette = [
        (1.0, 0.0, 0.0),  # 0-15 (almost in-plane)
        (1.0, 0.5, 0.0),  # 15-35
        (1.0, 1.0, 0.0),  # 35-55
        (0.0, 0.7, 0.0),  # 55-75
        (0.0, 0.4, 1.0),  # 75-90 (almost perpendicular)
    ]
    out = []
    for i, polys in enumerate(groups):
        col = palette[min(i, len(palette) - 1)]
        out.append(make_lineset_from_world_polylines(polys, color=col))
    return out


# ----------------------- export skeleton model (Azman-style) -----------------
def export_skeleton_model(
    out_dir: Path, nodes_xyz: np.ndarray, polylines_world: List[np.ndarray]
):
    """
    Export a simple skeleton model:
      points: all unique poly points,
      lines: consecutive pairs by index.
    """
    points: List[List[float]] = []
    index_map: Dict[Tuple[float, float, float], int] = {}

    def add_point(p):
        key = (float(p[0]), float(p[1]), float(p[2]))
        if key in index_map:
            return index_map[key]
        idx = len(points)
        points.append([key[0], key[1], key[2]])
        index_map[key] = idx
        return idx

    lines: List[Tuple[int, int]] = []
    for poly in polylines_world:
        if len(poly) < 2:
            continue
        prev_idx = add_point(poly[0])
        for i in range(1, len(poly)):
            cur_idx = add_point(poly[i])
            if prev_idx != cur_idx:
                lines.append((prev_idx, cur_idx))
            prev_idx = cur_idx

    data = {"points": points, "lines": lines, "sections": [], "joints": []}
    (out_dir / "skeleton_model.json").write_text(json.dumps(data, indent=2))
    logger.info(
        f"[EXPORT] skeleton_model.json with {len(points)} points, {len(lines)} lines"
    )


# ========================== normals deviation visualization ==================
ANGLE_BINS_DEG = (15.0, 35.0, 55.0, 75.0, 90.0)
PALETTE_BINS = [
    (1.0, 0.0, 0.0),  # 0-15
    (1.0, 0.5, 0.0),  # 15-35
    (1.0, 1.0, 0.0),  # 35-55
    (0.0, 0.7, 0.0),  # 55-75
    (0.0, 0.4, 1.0),  # 75-90
]


def compute_main_plane_normal_for_mode(
    mode: str,
    base_cloud: o3d.geometry.PointCloud,
    plane2d_model: Optional[np.ndarray],
) -> np.ndarray:
    """
    For 'plane2d': prefer exact plane normal; for 'vol3d': estimate dominant plane.
    """
    if mode == "plane2d" and plane2d_model is not None:
        n = np.array(plane2d_model[:3], float)
        n /= np.linalg.norm(n) + 1e-12
        return n
    return estimate_dominant_plane_normal(base_cloud)


def angle_between_normals_deg(normals: np.ndarray, ref_n: np.ndarray) -> np.ndarray:
    """Angle in degrees between each normal and ref normal, using |dot| (0-90]."""
    ref = ref_n / (np.linalg.norm(ref_n) + 1e-12)
    dots = np.clip(np.abs(normals @ ref), 0.0, 1.0)
    return np.degrees(np.arccos(dots))


def colorize_by_angle_bins(angles_deg: np.ndarray) -> np.ndarray:
    """Map each angle to a discrete palette color based on ANGLE_BINS_DEG."""
    cols = np.zeros((len(angles_deg), 3), float)
    for i, a in enumerate(angles_deg):
        for bi, hi in enumerate(ANGLE_BINS_DEG):
            if a <= hi + 1e-9:
                cols[i] = PALETTE_BINS[min(bi, len(PALETTE_BINS) - 1)]
                break
    return cols


def normals_deviation_report(angles_deg: np.ndarray) -> Dict[str, object]:
    """Compute a compact JSON-serializable report for normal deviations."""
    bins = list(ANGLE_BINS_DEG)
    hist = [0] * len(bins)
    for a in angles_deg:
        for bi, hi in enumerate(bins):
            if a <= hi + 1e-9:
                hist[bi] += 1
                break
    angles = angles_deg[np.isfinite(angles_deg)]
    stats = {
        "count": int(angles.size),
        "mean_deg": float(np.mean(angles)) if angles.size else float("nan"),
        "median_deg": float(np.median(angles)) if angles.size else float("nan"),
        "std_deg": float(np.std(angles)) if angles.size else float("nan"),
        "min_deg": float(np.min(angles)) if angles.size else float("nan"),
        "max_deg": float(np.max(angles)) if angles.size else float("nan"),
    }
    report = {
        "bins_deg": bins,
        "hist_counts": hist,
        "stats": stats,
        "note": "Angles are between per-point normals and main plane normal using |dot|, so range is [0,90] deg.",
        "palette_rgb": PALETTE_BINS,
    }
    return report


def make_normals_colored_cloud(
    base_cloud: o3d.geometry.PointCloud,
    ref_normal: np.ndarray,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, Dict[str, object]]:
    """
    Returns a copy of base_cloud where point colors encode the deviation angle
    between each point's normal and ref_normal. Also returns angles and a report.
    """
    cloud = o3d.geometry.PointCloud(base_cloud)  # shallow copy of geometry/attrs
    ensure_normals(cloud)
    N = np.asarray(cloud.normals)
    angles = angle_between_normals_deg(N, ref_normal)
    colors = colorize_by_angle_bins(angles)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    report = normals_deviation_report(angles)
    return cloud, angles, report


def _bin_masks_for_side_surfaces(angles_deg: np.ndarray) -> List[np.ndarray]:
    """
    Build boolean masks per side bin (exclude 0–15° main plane bin).
    Bins are (15,35], (35,55], (55,75], (75,90].
    """
    edges = list(ANGLE_BINS_DEG)  # [15,35,55,75,90]
    eps = 1e-6
    masks = []
    # side bins: indices 1..4
    lo = edges[0]
    for i in range(1, len(edges)):
        hi = edges[i]
        if i == 1:
            # (15,35]
            m = (angles_deg > lo + eps) & (angles_deg <= hi + eps)
        else:
            # (prev,hi]
            prev = edges[i - 1]
            m = (angles_deg > prev + eps) & (angles_deg <= hi + eps)
        masks.append(m)
    return masks  # len=4


def build_side_surface_clouds(
    base_cloud: o3d.geometry.PointCloud,
    ref_normal: np.ndarray,
) -> Tuple[List[o3d.geometry.PointCloud], List[np.ndarray]]:
    """
    Returns colored point clouds per side bin and their boolean masks.
    Each cloud contains only points of its bin; color = PALETTE_BINS[bin_index].
    """
    cloud = o3d.geometry.PointCloud(base_cloud)  # shallow copy for normals
    ensure_normals(cloud)
    N = np.asarray(cloud.normals)
    angles = angle_between_normals_deg(N, ref_normal)
    masks = _bin_masks_for_side_surfaces(angles)

    P = np.asarray(base_cloud.points)
    side_clouds: List[o3d.geometry.PointCloud] = []
    for bi, m in enumerate(masks, start=1):  # bins 1..4
        if not np.any(m):
            side_clouds.append(o3d.geometry.PointCloud())
            continue
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(P[m])
        pc.paint_uniform_color(PALETTE_BINS[bi])  # (1,0.5,0) etc.
        side_clouds.append(pc)
    return side_clouds, masks


def _distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    """Simple HSV palette, returns RGB in [0,1]."""
    if n <= 0:
        return []
    cols = []
    for i in range(n):
        h = (i / max(1, n)) % 1.0
        s = 0.65
        v = 0.95
        # HSV -> RGB
        hi = int(h * 6) % 6
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        if hi == 0:
            r, g, b = v, t, p
        elif hi == 1:
            r, g, b = q, v, p
        elif hi == 2:
            r, g, b = p, v, t
        elif hi == 3:
            r, g, b = p, q, v
        elif hi == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        cols.append((r, g, b))
    return cols


def _region_grow_components(
    P: np.ndarray,
    N: np.ndarray,
    mask: np.ndarray,
    radius: float,
    cos_thr: float,
    min_pts: int,
) -> List[np.ndarray]:
    """
    Connected components under: (i) within `radius`, (ii) normal cosine >= cos_thr.
    Returns list of index arrays (into the full point set P).
    """
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    Pm = P[idx]
    Nm = N[idx]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
    kdt = o3d.geometry.KDTreeFlann(pc)
    visited = np.zeros(idx.size, dtype=bool)
    comps = []
    for seed in range(idx.size):
        if visited[seed]:
            continue
        q = [seed]
        visited[seed] = True
        comp_local = [seed]
        n0 = Nm[seed]
        while q:
            i = q.pop()
            _, nbrs, _ = kdt.search_radius_vector_3d(Pm[i], radius)
            for j in nbrs:
                if visited[j]:
                    continue
                # same side-bin already guaranteed by `mask`; check normal similarity
                if float(np.dot(Nm[i], Nm[j])) >= cos_thr:
                    visited[j] = True
                    q.append(j)
                    comp_local.append(j)
        if len(comp_local) >= min_pts:
            comps.append(idx[np.array(comp_local, dtype=int)])
    return comps


def _boundary_points(
    P: np.ndarray, labels: np.ndarray, k: int, same_frac: float
) -> np.ndarray:
    """
    Mark boundary points: among k-NN, fraction with same label is < same_frac.
    Returns boolean mask over P.
    """
    if P.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    kdt = o3d.geometry.KDTreeFlann(pc)
    out = np.zeros((P.shape[0],), dtype=bool)
    k = max(3, int(k))
    for i in range(P.shape[0]):
        k_eff = min(k, P.shape[0])
        _, idxs, _ = kdt.search_knn_vector_3d(P[i], k_eff)
        same = np.sum(labels[idxs] == labels[i])
        if (same / float(k_eff)) < same_frac:
            out[i] = True
    return out


def build_side_surface_regions(
    base_cloud: o3d.geometry.PointCloud,
    ref_normal: np.ndarray,
    region_radius: float = SIDE_REGION_RADIUS,
    normal_thr_deg: float = SIDE_NORMAL_THR_DEG,
    min_region_pts: int = SIDE_MIN_REGION_PTS,
) -> Tuple[List[o3d.geometry.PointCloud], o3d.geometry.PointCloud]:
    """
    Split side bins into spatially connected normal-consistent regions (left/right, internals).
    Returns:
      - list of colored point clouds (one per region),
      - point cloud of boundary points (black).
    """
    ensure_normals(base_cloud)
    P = np.asarray(base_cloud.points)
    N = np.asarray(base_cloud.normals)
    # angles to main plane -> side-bin masks
    ang = angle_between_normals_deg(N, ref_normal)
    side_masks = _bin_masks_for_side_surfaces(
        ang
    )  # 4 masks for bins 15–35, 35–55, 55–75, 75–90

    cos_thr = math.cos(math.radians(normal_thr_deg))
    region_indices: List[np.ndarray] = []
    # grow components per side-bin
    for m in side_masks:
        comps = _region_grow_components(P, N, m, region_radius, cos_thr, min_region_pts)
        region_indices.extend(comps)

    # build colored clouds
    regions_clouds: List[o3d.geometry.PointCloud] = []
    cols = _distinct_colors(len(region_indices))
    labels = np.full((P.shape[0],), -1, dtype=int)
    for r_id, ids in enumerate(region_indices):
        labels[ids] = r_id
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(P[ids])
        pc.paint_uniform_color(cols[r_id])
        regions_clouds.append(pc)

    # boundary points (black)
    bnd = o3d.geometry.PointCloud()
    if len(region_indices) > 0:
        keep = labels >= 0
        Pb = P[keep]
        Lb = labels[keep]
        bmask = _boundary_points(Pb, Lb, SIDE_BOUNDARY_K, SIDE_BOUNDARY_FRAC)
        if np.any(bmask):
            bnd.points = o3d.utility.Vector3dVector(Pb[bmask])
            bnd.paint_uniform_color((0.0, 0.0, 0.0))
    return regions_clouds, bnd


# --- improved side-region detection (in-plane normals) ---
SIDE_MIN_SIDE_DEG = 15.0  # side points: angle-to-plane > this
SIDE_IP_NORMAL_THR_DEG = 15.0  # max in-plane normal angle within a component
SIDE_GROW_RADIUS = 0.016  # meters
SIDE_MIN_REGION_PTS = 180
SIDE_SUBFACES_K = 4  # split each region into 4 faces by in-plane angle
SIDE_BOUNDARY_K = 24
SIDE_BOUNDARY_FRAC = 0.75


def _plane_basis_from_n(n: np.ndarray):
    n = n / (np.linalg.norm(n) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return u, v, n


def _project_normals_to_plane(
    N: np.ndarray, n: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit in-plane normal directions T and valid mask."""
    T = N - (N @ n)[:, None] * n
    s = np.linalg.norm(T, axis=1)
    m = s > 1e-9
    T[m] /= s[m, None]
    T[~m] = 0.0
    return T, m


def _angles_in_plane(T: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle of in-plane vector T w.r.t. basis (u,v) in [-pi, pi]."""
    x = T @ u
    y = T @ v
    return np.arctan2(y, x)


def _region_grow_ip(
    P: np.ndarray,
    T: np.ndarray,
    mask: np.ndarray,
    radius: float,
    cos_thr: float,
    min_pts: int,
) -> List[np.ndarray]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    Pm = P[idx]
    Tm = T[idx]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
    kdt = o3d.geometry.KDTreeFlann(pc)
    visited = np.zeros(idx.size, bool)
    comps = []
    for seed in range(idx.size):
        if visited[seed]:
            continue
        q = [seed]
        visited[seed] = True
        comp = [seed]
        while q:
            i = q.pop()
            _, nbrs, _ = kdt.search_radius_vector_3d(Pm[i], radius)
            ti = Tm[i]
            for j in nbrs:
                if visited[j]:
                    continue
                if float(np.dot(ti, Tm[j])) >= cos_thr:  # in-plane normal similarity
                    visited[j] = True
                    q.append(j)
                    comp.append(j)
        if len(comp) >= min_pts:
            comps.append(idx[np.asarray(comp, int)])
    return comps


def build_side_surface_regions_v2(
    base_cloud: o3d.geometry.PointCloud,
    ref_normal: np.ndarray,
    grow_radius: float = SIDE_GROW_RADIUS,
    ip_normal_thr_deg: float = SIDE_IP_NORMAL_THR_DEG,
    min_region_pts: int = SIDE_MIN_REGION_PTS,
    subfaces_k: int = SIDE_SUBFACES_K,
) -> Tuple[List[o3d.geometry.PointCloud], o3d.geometry.PointCloud]:
    """
    1) side mask: angle(N, plane_n) > SIDE_MIN_SIDE_DEG
    2) region growing by spatial proximity + in-plane normal similarity
    3) split each region into `subfaces_k` faces by in-plane angle quadrants
    """
    ensure_normals(base_cloud)
    P = np.asarray(base_cloud.points)
    N = np.asarray(base_cloud.normals)

    # main plane basis
    u, v, n = _plane_basis_from_n(ref_normal)

    # side points (no upper cap): (> 15°)
    ang = angle_between_normals_deg(N, n)
    side_mask = ang > (SIDE_MIN_SIDE_DEG - 1e-6)

    # in-plane normal vectors
    T, valid = _project_normals_to_plane(N, n)
    side_mask &= valid

    cos_thr = math.cos(math.radians(ip_normal_thr_deg))
    comps = _region_grow_ip(P, T, side_mask, grow_radius, cos_thr, min_region_pts)

    # label map for boundaries
    labels = np.full(P.shape[0], -1, int)
    regions_clouds: List[o3d.geometry.PointCloud] = []
    color_id = 0

    for ids in comps:
        # subfaces by in-plane angle
        theta = _angles_in_plane(T[ids], u, v)  # [-pi, pi]
        # fixed 4 quadrants (robust, deterministic)
        # bins: (-pi,-pi/2], (-pi/2,0], (0,pi/2], (pi/2,pi]
        edges = [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi + 1e-9]
        sub_idx = []
        for k in range(4):
            m = (theta > edges[k]) & (theta <= edges[k + 1])
            if np.count_nonzero(m) >= max(40, int(0.05 * ids.size)):
                sub_idx.append(ids[m])

        # make colored clouds
        for sids in sub_idx:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(P[sids])
            # generate stable vivid color per subface
            hue = (hash((color_id, sids.size)) % 360) / 360.0
            cols = _distinct_colors(16)
            pc.paint_uniform_color(cols[color_id % len(cols)])
            regions_clouds.append(pc)
            labels[sids] = color_id
            color_id += 1

    # boundaries across all labeled points
    bnd = o3d.geometry.PointCloud()
    keep = labels >= 0
    if np.any(keep):
        Pb = P[keep]
        Lb = labels[keep]
        bmask = _boundary_points(Pb, Lb, SIDE_BOUNDARY_K, SIDE_BOUNDARY_FRAC)
        if np.any(bmask):
            bnd.points = o3d.utility.Vector3dVector(Pb[bmask])
            bnd.paint_uniform_color((0.0, 0.0, 0.0))

    return regions_clouds, bnd


# =============================== PIPELINE ====================================
def run():
    # load cloud
    cloud_path = find_input_cloud()
    logger.info(f"[LOAD] cloud: {cloud_path}")
    raw = o3d.io.read_point_cloud(str(cloud_path))
    if len(raw.points) == 0:
        raise RuntimeError("empty point cloud")

    # smoothing (optional)
    if ENABLE_SMOOTHING:
        smoothed = smooth_cloud_mls_plane_adaptive(raw)
        base_cloud = smoothed
        logger.info("[SMOOTH] adaptive MLS done.")
    else:
        smoothed = None
        base_cloud = raw

    # containers
    geoms: Dict[str, List] = {
        "cloud": [],
        "skeleton": [],
        "graph": [],
        "centers": [],
        "members": [],
        "normals": [],
        "side_regions": [],
        "sides_bins": [],
    }

    # store raw + smoothed clouds as grey references
    raw_col = o3d.geometry.PointCloud(raw)
    raw_col.paint_uniform_color((0.6, 0.6, 0.6))
    geoms["cloud"].append(raw_col)
    if smoothed is not None:
        sm = o3d.geometry.PointCloud(smoothed)
        sm.paint_uniform_color((0.9, 0.9, 0.9))
        geoms["cloud"].append(sm)

    # variables shared for normals visualization
    plane2d_model_used = None
    main_plane_normal = None
    normals_report = None

    if MODE == "plane2d":
        # detect plane on downsample; pick nearest to cameras
        pcd_ds = downsample(base_cloud, DS_VOX_SEG)
        N = len(pcd_ds.points)
        min_pts = max(MIN_PLANE_POINTS_ABS, int(N * MIN_PLANE_POINTS_FRAC))
        cam_centers = []
        if USE_CAPTURE_POSES_FOR_CAMERA:
            poses = load_poses(CAPTURE_ROOT / POSES_JSON)
            if poses:
                cam_centers = camera_centers_from_poses(poses)

        planes = None
        used_dist = None
        for dist in DIST_SCHEDULE:
            planes = segment_planes_iterative(
                pcd_ds,
                MAX_PLANE_CANDIDATES,
                dist,
                PLANE_RANSAC_N,
                PLANE_ITERATIONS,
                min_pts,
            )
            if planes:
                used_dist = dist
                break
        if not planes:
            min_relaxed = max(200, int(min_pts * 0.5))
            planes = segment_planes_iterative(
                pcd_ds,
                MAX_PLANE_CANDIDATES,
                DIST_SCHEDULE[-1],
                PLANE_RANSAC_N,
                PLANE_ITERATIONS * 2,
                min_relaxed,
            )
            used_dist = DIST_SCHEDULE[-1] if planes else None
        if not planes:
            raise RuntimeError("no plane candidates after escalation")

        model, inliers_full = pick_nearest_plane_to_cameras(
            planes, base_cloud, cam_centers, dist_for_reselect=used_dist
        )
        plane2d_model_used = model
        plane_pcd = base_cloud.select_by_index(inliers_full.tolist(), invert=False)
        rest_pcd = base_cloud.select_by_index(inliers_full.tolist(), invert=True)
        plane_pcd.paint_uniform_color((0.75, 0.75, 0.75))
        rest_pcd.paint_uniform_color((0.55, 0.55, 0.55))
        geoms["cloud"] += [rest_pcd, plane_pcd]

        P = np.asarray(plane_pcd.points)
        u, v, n, p0 = plane_basis_from_coeffs(model, P)
        main_plane_normal = n  # for normals visualization
        UV = to_plane_uv(P, u, v, p0)
        img, (umin, vmin) = rasterize_points(UV, GRID_RES)
        img = morph_close(img, MORPH_CLOSE_KERNEL)
        sk2 = skeletonize_2d(img, SKELETON_2D_METHOD)
        node_rc, edges2d, node_map2d, polylines_px = skeleton_to_graph_2d(sk2)

        nodes_xyz = np.array(
            [
                p0
                + ((c + 0.5) * GRID_RES + umin) * u
                + ((r + 0.5) * GRID_RES + vmin) * v
                for (r, c) in node_rc
            ],
            float,
        )
        nodes_xyz, _, _ = merge_close_nodes_world(nodes_xyz, [], NODE_MERGE_RADIUS_M)
        ensure_normals(plane_pcd)
        nodes_xyz = refine_nodes_by_normals(
            nodes_xyz,
            plane_pcd,
            NODE_NORMAL_REFINE_RADIUS,
            BRIDGE_WIDTH_EST,
            NODE_MIN_SIDE_PTS,
        )

        polys_w = []
        for path in polylines_px:
            if len(path) < 2:
                continue
            uv = np.array(
                [
                    [(c + 0.5) * GRID_RES + umin, (r + 0.5) * GRID_RES + vmin]
                    for (r, c) in path
                ],
                float,
            )
            xyz = p0[None, :] + uv[:, 0:1] * u[None, :] + uv[:, 1:2] * v[None, :]
            polys_w.append(
                resample_polyline_world(xyz, RESAMPLE_STEP_M, KEEP_TAIL_MIN_FRAC)
            )

        center_polys, _ = refine_polylines_centers(
            polys_w,
            plane_pcd,
            search_r=CENTER_SEARCH_R,
            min_side_pts=CENTER_MIN_SIDE_PTS,
            exp_width=BRIDGE_WIDTH_EST,
        )
        if STRAIGHTEN_BRIDGES:
            center_polys = straighten_polylines(center_polys)

        ls_centers = make_lineset_from_world_polylines(center_polys, color=(0, 0, 1))
        geoms["centers"] = [ls_centers]
        ls_raw = make_lineset_from_world_polylines(polys_w, color=(0, 1, 0))
        geoms["skeleton"] = [ls_raw]
        nodes_mesh = (
            make_nodes_mesh(nodes_xyz, (1, 0, 0), NODE_SPHERE_R)
            if len(nodes_xyz) > 0
            else None
        )
        grp = [ls_raw]
        if nodes_mesh is not None:
            grp.append(nodes_mesh)
        geoms["graph"] = grp

        groups = group_polylines_by_angle(center_polys, main_plane_normal)
        geoms["members"] = make_group_linesets(groups)

        out = CAPTURE_ROOT / DEBUG_DIR_NAME
        export_skeleton_model(out, nodes_xyz, center_polys)

    else:
        # 3D voxel skeletonization
        P = np.asarray(base_cloud.points)
        origin, vsize, dims = compute_voxel_grid_params(
            P, VOXEL_3D, PAD_VOX, MAX_3D_VOXELS
        )
        logger.info(
            f"[VOX] origin={origin}, v={vsize:.4f}, dims={tuple(dims)} ~ {int(np.prod(dims))} vox"
        )
        vol = voxelize_points(P, origin, vsize, dims)
        if DILATE_3D_RADIUS > 0:
            vol = dilate_3d(vol, DILATE_3D_RADIUS)

        sk = skeletonize_3d_volume(vol)
        sk_count = int(sk.sum())
        logger.info(f"[SKEL-3D] voxels: {sk_count}")
        if sk_count < MIN_SKEL_VOXELS:
            raise RuntimeError(
                f"3d skeleton too small ({sk_count} voxels). try bigger DILATE_3D_RADIUS or coarser VOXEL_3D."
            )

        polylines_vox = skeleton_to_polylines_3d(sk)
        logger.info(f"[SKEL-3D] polylines: {len(polylines_vox)}")

        node_vox = skeleton_nodes_vox(sk)
        nodes_xyz = np.array(
            [
                origin + (np.array([x + 0.5, y + 0.5, z + 0.5]) * vsize)
                for (z, y, x) in node_vox
            ],
            float,
        )
        nodes_xyz, _, _ = merge_close_nodes_world(nodes_xyz, [], NODE_MERGE_RADIUS_M)
        ensure_normals(base_cloud)
        nodes_xyz = refine_nodes_by_normals(
            nodes_xyz,
            base_cloud,
            NODE_NORMAL_REFINE_RADIUS,
            BRIDGE_WIDTH_EST,
            NODE_MIN_SIDE_PTS,
        )

        polys_w = []
        for path in polylines_vox:
            if len(path) < 2:
                continue
            V = np.array([[x + 0.5, y + 0.5, z + 0.5] for (z, y, x) in path], float)
            xyz = origin[None, :] + V[:, [0, 1, 2]] * vsize
            polys_w.append(
                resample_polyline_world(xyz, RESAMPLE_STEP_M, KEEP_TAIL_MIN_FRAC)
            )

        center_polys, _ = refine_polylines_centers(
            polys_w,
            base_cloud,
            search_r=CENTER_SEARCH_R,
            min_side_pts=CENTER_MIN_SIDE_PTS,
            exp_width=BRIDGE_WIDTH_EST,
        )
        if STRAIGHTEN_BRIDGES:
            center_polys = straighten_polylines(center_polys)

        ls_centers = make_lineset_from_world_polylines(center_polys, color=(0, 0, 1))
        geoms["centers"] = [ls_centers]
        ls_raw = make_lineset_from_world_polylines(polys_w, color=(0, 1, 0))

        cloud = o3d.geometry.PointCloud(base_cloud)
        cloud.paint_uniform_color((0.65, 0.65, 0.65))
        geoms["cloud"].append(cloud)
        geoms["skeleton"] = [ls_raw]
        nodes_mesh = (
            make_nodes_mesh(nodes_xyz, (1, 0, 0), NODE_SPHERE_R)
            if len(nodes_xyz) > 0
            else None
        )
        grp = [ls_raw]
        if nodes_mesh is not None:
            grp.append(nodes_mesh)
        geoms["graph"] = grp

        main_plane_normal = estimate_dominant_plane_normal(base_cloud)
        groups = group_polylines_by_angle(center_polys, main_plane_normal)
        geoms["members"] = make_group_linesets(groups)

        out = CAPTURE_ROOT / DEBUG_DIR_NAME
        export_skeleton_model(out, nodes_xyz, center_polys)

    # ----------- NEW: compute normals-based coloring (key "6") ---------------
    try:
        ref_n = compute_main_plane_normal_for_mode(MODE, base_cloud, plane2d_model_used)
        colored_cloud, angles, normals_report = make_normals_colored_cloud(
            base_cloud, ref_n
        )
        geoms["normals"].append(colored_cloud)
        try:
            side_clouds, side_masks = build_side_surface_clouds(base_cloud, ref_n)
            geoms["sides"] = (
                side_clouds  # 4 entries for bins (15–35], (35–55], (55–75], (75–90]
            )
            logger.info(
                "[SIDES] per-bin counts: %s",
                [int(np.asarray(sc.points).shape[0]) for sc in side_clouds],
            )
        except Exception as e:
            geoms["sides"] = []
            logger.warning(f"[SIDES] failed to segment side surfaces: {e}")
        logger.info(
            "[NORMALS] mean=%.2f, median=%.2f, min=%.2f, max=%.2f | bins=%s",
            normals_report["stats"]["mean_deg"],
            normals_report["stats"]["median_deg"],
            normals_report["stats"]["min_deg"],
            normals_report["stats"]["max_deg"],
            normals_report["hist_counts"],
        )
        logger.info(
            "[NORMALS] palette bins (deg): 0-15 red, 15-35 orange, 35-55 yellow, 55-75 green, 75-90 blue"
        )
    except Exception as e:
        logger.warning(f"[NORMALS] failed to compute normals visualization: {e}")
    # --- Side regions by normals + connectivity (key 7) ---
    try:
        side_regions, side_boundaries = build_side_surface_regions_v2(base_cloud, ref_n)
        geoms["side_regions"] = side_regions + (
            [side_boundaries] if len(side_boundaries.points) > 0 else []
        )
        logger.info(
            "[SIDE-REGIONS v2] regions=%d%s",
            len(side_regions),
            " + boundaries" if len(side_boundaries.points) > 0 else "",
        )
    except Exception as e:
        geoms["side_regions"] = []
        logger.warning(f"[SIDE-REGIONS] failed: {e}")

    # keep the old per-bin sides as a separate layer (key 8)
    try:
        side_clouds, _side_masks = build_side_surface_clouds(base_cloud, ref_n)
        geoms["sides_bins"] = side_clouds
    except Exception:
        geoms["sides_bins"] = []
    # ---------- viewer with toggles ----------
    visible = {
        "cloud": True,
        "skeleton": False,
        "graph": False,
        "centers": False,
        "members": False,
        "normals": False,
        "side_regions": False,
        "sides_bins": False,
    }

    vis = o3d.visualization.VisualizerWithKeyCallback()
    title = "Plane2D" if MODE == "plane2d" else "Vol3D Global"
    vis.create_window(
        window_name=f"Skeleton ({title}) - 1 cloud, 2 skeleton, 3 graph, 4 centers, 5 members, 6 normals, 7 side-regions, 8 side-bins",
        width=1280,
        height=800,
    )

    for g in geoms["cloud"]:
        vis.add_geometry(g)

    def toggle(group: str):
        def _cb(viz):
            # make robust if group was not predeclared
            cur = visible.get(group, False)
            visible[group] = not cur
            geoms.setdefault(group, [])  # ensure list exists

            if visible[group]:
                for g in geoms[group]:
                    viz.add_geometry(g)
            else:
                for g in geoms[group]:
                    viz.remove_geometry(g, reset_bounding_box=False)
            return False

        return _cb

    geoms.setdefault("side_regions", [])
    geoms.setdefault("sides_bins", [])

    vis.register_key_callback(ord("1"), toggle("cloud"))
    vis.register_key_callback(ord("2"), toggle("skeleton"))
    vis.register_key_callback(ord("3"), toggle("graph"))
    vis.register_key_callback(ord("4"), toggle("centers"))
    vis.register_key_callback(ord("5"), toggle("members"))
    vis.register_key_callback(ord("6"), toggle("normals"))
    vis.register_key_callback(ord("7"), toggle("side_regions"))
    vis.register_key_callback(ord("8"), toggle("sides_bins"))
    vis.run()
    vis.destroy_window()

    # save artifacts
    out = CAPTURE_ROOT / DEBUG_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    tag = "plane2d" if MODE == "plane2d" else "vol3d"
    if (
        geoms["skeleton"]
        and isinstance(geoms["skeleton"][0], o3d.geometry.LineSet)
        and lineset_has_points(geoms["skeleton"][0])
    ):
        o3d.io.write_line_set(
            str(out / f"{tag}_skeleton_raw.ply"), geoms["skeleton"][0]
        )
    if (
        geoms["graph"]
        and isinstance(geoms["graph"][0], o3d.geometry.LineSet)
        and lineset_has_points(geoms["graph"][0])
    ):
        o3d.io.write_line_set(str(out / f"{tag}_graph_skeleton.ply"), geoms["graph"][0])
    if (
        geoms.get("centers")
        and isinstance(geoms["centers"][0], o3d.geometry.LineSet)
        and lineset_has_points(geoms["centers"][0])
    ):
        o3d.io.write_line_set(str(out / f"{tag}_centerlines.ply"), geoms["centers"][0])

    if geoms["members"]:
        for gi, ls in enumerate(geoms["members"]):
            if isinstance(ls, o3d.geometry.LineSet) and lineset_has_points(ls):
                o3d.io.write_line_set(str(out / f"{tag}_members_group{gi}.ply"), ls)

    if len(geoms["graph"]) >= 2 and isinstance(
        geoms["graph"][1], o3d.geometry.TriangleMesh
    ):
        o3d.io.write_triangle_mesh(
            str(out / f"{tag}_graph_nodes.ply"), geoms["graph"][1]
        )

    # save normals visualization & report
    if geoms["normals"]:
        try:
            o3d.io.write_point_cloud(
                str(out / f"{tag}_normals_angle_colored.ply"), geoms["normals"][0]
            )
            if normals_report is not None:
                (out / f"{tag}_normals_info.json").write_text(
                    json.dumps(normals_report, indent=2)
                )
        except Exception as e:
            logger.warning(f"[SAVE] normals artifacts failed: {e}")
    # save side regions
    if geoms.get("side_regions"):
        try:
            for i, sc in enumerate(geoms["side_regions"]):
                if isinstance(sc, o3d.geometry.PointCloud) and len(sc.points) > 0:
                    name = (
                        "boundary"
                        if i == (len(geoms["side_regions"]) - 1)
                        and np.all(np.asarray(sc.colors)[0] == [0, 0, 0])
                        else f"region_{i:02d}"
                    )
                    o3d.io.write_point_cloud(str(out / f"{tag}_side_{name}.ply"), sc)
        except Exception as e:
            logger.warning(f"[SAVE] side regions failed: {e}")

    # save simple per-bin sides (legacy)
    if geoms.get("sides_bins"):
        try:
            side_names = ["15_35", "35_55", "55_75", "75_90"]
            for i, sc in enumerate(geoms["sides_bins"]):
                if isinstance(sc, o3d.geometry.PointCloud) and len(sc.points) > 0:
                    o3d.io.write_point_cloud(
                        str(out / f"{tag}_sides_{side_names[i]}.ply"), sc
                    )
        except Exception as e:
            logger.warning(f"[SAVE] side bins failed: {e}")

    logger.info(f"[SAVE] artifacts in {out}")


if __name__ == "__main__":
    run()
