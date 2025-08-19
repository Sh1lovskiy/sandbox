# merge.py - robust RGB-D merge (FGR->robust ICP->safe Colored ICP) + optional TSDF
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json, math, time, logging
import numpy as np
import open3d as o3d

np.random.seed(42)

# =============================== CONSTANTS ===================================
CAPTURE_ROOT = Path("captures/20250813_143934")
IMG_DIR_NAME = "imgs"
POSES_JSON = "poses.json"

# intrinsics fallback (if no YAML in root)
CAM_W, CAM_H = 1280, 720
CAM_FX, CAM_FY, CAM_CX, CAM_CY = 920.0, 920.0, 640.0, 360.0
DEPTH_TRUNC = 2.5
DEPTH_UNIT_MODE = "auto"  # "auto" | "meters" | "millimeters"

# hand-eye (camera on TCP of the last joint)
HAND_EYE_R = np.array(
    [
        [0.999048, 0.00428, -0.00625],
        [-0.00706, 0.99658, -0.00804],
        [0.00423, 0.00895, 0.99629],
    ],
    dtype=np.float64,
)
HAND_EYE_T = np.array([-0.036, -0.078, 0.006], dtype=np.float64)  # meters

# Work AABB in BASE (after transforms)
BBOX_POINTS = np.array(
    [
        [-0.57, -0.34, 0.46],
        [-0.57, 0.2, 0.2],
        [-0.3, 0.2, 0.2],
        [-0.3, 0.2, 0.46],
    ],
    dtype=np.float64,
)

# Per-frame / final processing
FRAME_VOX = 0.003
MERGE_VOX = 0.002
REMOVE_OUTLIERS = True
OUTLIER_NN, OUTLIER_STD = 20, 2.0

# Hybrid refine (global + local)
APPLY_PAIRWISE_REFINEMENT = True
RD, RN, RF = 0.01, 0.02, 0.05  # rd:rn:rf | 1:2:5
FGR_MAX_CORR_DIST = RD * 1.5
ICP_MAX_CORR_DIST = RD * 0.8
ICP_MAX_ITERS = 40
ICP_TUKEY_K = RD * 3.0

# --- Quality/strategy switches ---
QUALITY_PRESET = "best"  # "fast" | "best"
MERGE_STRATEGY = "tsdf"  # "pcd" | "tsdf"
USE_COLORED_ICP = True  # add multi-scale colored ICP after ICP
COLORED_ICP_PYRAMID = [0.02, 0.01, 0.005]  # thresholds (m): coarse->fine
COLORED_ICP_MAXIT = [30, 20, 15]

# Colored ICP safety & tuning
COLORED_ICP_DS = 0.006  # voxel size for CICP inputs
CICP_ENABLE_MIN_FITNESS = 0.20  # run CICP only if ICP fitness >= this
CICP_CONTINUE_MIN_FITNESS = 0.30  # continue next pyramid level if >= this
CICP_FAIL_SCALE_UP = 2.0  # inflate thresholds when start fitness is weak

# TSDF params (Open3D classic pipeline)
TSDF_VOX = 0.004
TSDF_TRUNC = TSDF_VOX * 4.0
TSDF_COLOR_TYPE = o3d.pipelines.integration.TSDFVolumeColorType.RGB8

# Depth prefilter
DEPTH_BILATERAL = True
DEPTH_BILATERAL_DIAM = 5  # odd (pixels)
DEPTH_BI_SIGMA_COLOR = 0.03  # meters
DEPTH_BI_SIGMA_SPACE = 3.0  # pixels

# Visualization / logs
LOG_LEVEL = logging.INFO
VIS_STAGES = True
VIS_EVERY_N = 1
VIS_POINT_SIZE = 2.0
VIS_COORD_FRAME_SIZE = 0.05
SAVE_INTERMEDIATE = False
DEBUG_DIR_NAME = "debug"

# Auto: pose units / Euler order / hand-eye direction (first frame)
POSE_UNIT_MODE = "auto"  # "auto"|"meters"|"millimeters"
POSE_EULER_ORDER = "auto"  # "auto"|"XYZ"|"ZYX"
HAND_EYE_DIR = "auto"  # "auto"|"tcp_cam"|"cam_tcp_inv"

# Final editing/picks export
CAPTURE_PICKS = False
PICK_SPHERE_R = 0.004

# ================================= LOGGING ===================================
logger = logging.getLogger("merge")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(LOG_LEVEL)
np.set_printoptions(suppress=True, precision=6, linewidth=180)


def _fmt(v):
    return np.array2string(np.asarray(v), separator=", ")


# =============================== DATA TYPES ==================================
@dataclass
class Pose:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float  # degrees


# =============================== HELPERS =====================================
def try_load_intrinsics_from_yaml(root: Path):
    for name in ["intrinsics.yml", "intrinsics.yaml", "cam.yml", "cam.yaml"]:
        p = root / name
        if p.exists():
            try:
                import yaml

                y = yaml.safe_load(p.read_text())
                w = int(y.get("width", CAM_W))
                h = int(y.get("height", CAM_H))
                fx = float(y.get("fx", CAM_FX))
                fy = float(y.get("fy", CAM_FY))
                cx = float(y.get("cx", CAM_CX))
                cy = float(y.get("cy", CAM_CY))
                logger.info(
                    f"[INTR] YAML {p.name}: w={w} h={h} fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}"
                )
                return (w, h, fx, fy, cx, cy)
            except Exception as e:
                logger.warning(f"[INTR] YAML read fail: {e}")
    logger.info("[INTR] YAML not found; using fallback.")
    return (CAM_W, CAM_H, CAM_FX, CAM_FY, CAM_CX, CAM_CY)


def o3d_intrinsics(w, h, fx, fy, cx, cy):
    return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)


def guess_depth_units(depth: np.ndarray) -> str:
    if DEPTH_UNIT_MODE in ("meters", "millimeters"):
        return DEPTH_UNIT_MODE
    p95 = float(np.nanpercentile(depth, 95))
    return "meters" if p95 < 10 else "millimeters"


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


def prefilter_depth(depth_m: np.ndarray) -> np.ndarray:
    """Bilateral filter on metric depth (meters)."""
    if not DEPTH_BILATERAL:
        return depth_m
    import cv2

    d = depth_m.copy()
    d[~np.isfinite(d)] = 0.0
    d = cv2.bilateralFilter(
        d, DEPTH_BILATERAL_DIAM, DEPTH_BI_SIGMA_COLOR, DEPTH_BI_SIGMA_SPACE
    )
    d = np.where(d < 1e-6, 0.0, d).astype(np.float32)
    return d


def rgbd_to_pcd(rgb: np.ndarray, depth_m: np.ndarray, intr) -> o3d.geometry.PointCloud:
    color = o3d.geometry.Image(rgb.astype(np.uint8))
    depth = o3d.geometry.Image(depth_m.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1.0,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


def make_rgbd(rgb: np.ndarray, depth_m: np.ndarray) -> o3d.geometry.RGBDImage:
    color = o3d.geometry.Image(rgb.astype(np.uint8))
    depth = o3d.geometry.Image(depth_m.astype(np.float32))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1.0,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False,
    )


def downsample(pcd, vox):
    return pcd.voxel_down_sample(vox) if vox and vox > 0 else pcd


def clean_outliers(pcd, nn=20, std=2.0):
    n0 = len(pcd.points)
    _, idx = pcd.remove_statistical_outlier(nb_neighbors=nn, std_ratio=std)
    pc = pcd.select_by_index(idx)
    return pc, (n0 - len(pc.points))


def compute_normals(pcd, radius, max_nn=60):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(50)


def compute_fpfh(pcd, radius):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )


def fgr(source, target, src_feat, tgt_feat, max_corr):
    opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=max_corr, iteration_number=64
    )
    res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source, target, src_feat, tgt_feat, opt
    )
    return res.transformation


def make_robust_kernel(tukey_k: float):
    """Return a robust kernel compatible with the installed Open3D build, or None."""
    rk = o3d.pipelines.registration
    if hasattr(rk, "TukeyLoss"):
        logger.info("[ICP] Using TukeyLoss API (rk.TukeyLoss).")
        return rk.TukeyLoss(k=float(tukey_k))
    if hasattr(rk, "RobustKernel") and hasattr(rk, "RobustKernelType"):
        logger.info("[ICP] Using legacy RobustKernel(Tukey) API.")
        return rk.RobustKernel(rk.RobustKernelType.Tukey, float(tukey_k))
    logger.info("[ICP] Robust kernels unavailable; falling back to plain L2.")
    return None


def icp_pt2pl(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    init: np.ndarray,
    max_corr: float,
    iters: int,
    tukey_k: float,
) -> Tuple[np.ndarray, float, float]:
    """Robust point-to-plane ICP returning (T, fitness, rmse)."""
    rk = o3d.pipelines.registration
    loss = make_robust_kernel(tukey_k)
    estimation = (
        rk.TransformationEstimationPointToPlane(loss)
        if loss is not None
        else rk.TransformationEstimationPointToPlane()
    )
    criteria = rk.ICPConvergenceCriteria(max_iteration=iters)
    res = rk.registration_icp(src, tgt, max_corr, init, estimation, criteria)
    logger.info(f"[ICP] fitness={res.fitness:.4f}  inlier_rmse={res.inlier_rmse:.6f}")
    return res.transformation, float(res.fitness), float(res.inlier_rmse)


def colored_icp_multiscale(
    src_in: o3d.geometry.PointCloud,
    tgt_in: o3d.geometry.PointCloud,
    init: np.ndarray,
    dists: List[float],
    iters: List[int],
    start_fitness: float,
) -> np.ndarray:
    """
    Safe multi-scale Colored ICP:
    - runs only if start_fitness >= CICP_ENABLE_MIN_FITNESS
    - inflates thresholds if start fitness is weak
    - early-stops if fitness drops below CICP_CONTINUE_MIN_FITNESS
    - catches 'No correspondences' and returns last good T
    """
    rk = o3d.pipelines.registration
    if start_fitness < CICP_ENABLE_MIN_FITNESS:
        logger.info(
            f"[CICP] skipped (ICP fitness {start_fitness:.3f} < {CICP_ENABLE_MIN_FITNESS})"
        )
        return init

    # Stable, downsampled copies (keep colors)
    src = downsample(o3d.geometry.PointCloud(src_in), COLORED_ICP_DS)
    tgt = downsample(o3d.geometry.PointCloud(tgt_in), COLORED_ICP_DS)
    if not src.has_normals():
        compute_normals(src, dists[0] * 2.0)
    if not tgt.has_normals():
        compute_normals(tgt, dists[0] * 2.0)

    dists = list(dists)
    if start_fitness < CICP_CONTINUE_MIN_FITNESS:
        dists = [d * CICP_FAIL_SCALE_UP for d in dists]
        logger.info(
            f"[CICP] inflating thresholds x{CICP_FAIL_SCALE_UP:.1f} (start fitness {start_fitness:.3f})"
        )

    T = init.copy()
    last_ok_T = T
    last_ok_fit = start_fitness

    for lvl, (dist, itn) in enumerate(zip(dists, iters)):
        try:
            res = rk.registration_colored_icp(
                src,
                tgt,
                dist,
                T,
                rk.TransformationEstimationForColoredICP(),
                rk.ICPConvergenceCriteria(max_iteration=itn),
            )
            T = res.transformation
            last_ok_T = T
            last_ok_fit = float(res.fitness)
            logger.info(
                f"[CICP L{lvl}] thr={dist:.4f} it={itn}  fitness={res.fitness:.4f}  rmse={res.inlier_rmse:.6f}"
            )
            if last_ok_fit < CICP_CONTINUE_MIN_FITNESS:
                logger.info(
                    f"[CICP] early stop at L{lvl} (fitness {last_ok_fit:.3f} < {CICP_CONTINUE_MIN_FITNESS})"
                )
                break
        except RuntimeError as e:
            logger.warning(f"[CICP L{lvl}] aborted: {e}")
            break

    return last_ok_T


def crop_aabb(pcd, bbox_pts):
    box = o3d.geometry.PointCloud()
    box.points = o3d.utility.Vector3dVector(bbox_pts)
    aabb = box.get_axis_aligned_bounding_box()
    return pcd.crop(aabb), aabb


def bbox_inliers_count(
    pts: np.ndarray, aabb: o3d.geometry.AxisAlignedBoundingBox
) -> int:
    mn = np.asarray(aabb.get_min_bound())
    mx = np.asarray(aabb.get_max_bound())
    m = (
        (pts[:, 0] >= mn[0])
        & (pts[:, 1] >= mn[1])
        & (pts[:, 2] >= mn[2])
        & (pts[:, 0] <= mx[0])
        & (pts[:, 1] <= mx[1])
        & (pts[:, 2] <= mx[2])
    )
    return int(m.sum())


def sample_points(pcd: o3d.geometry.PointCloud, n: int = 200_000) -> np.ndarray:
    if len(pcd.points) <= n:
        return np.asarray(pcd.points)
    idx = np.random.choice(len(pcd.points), n, replace=False)
    return np.asarray(pcd.points)[idx]


def log_cloud_stats(tag: str, pcd: o3d.geometry.PointCloud):
    if len(pcd.points) == 0:
        logger.info(f"{tag}: 0 pts")
        return
    P = np.asarray(pcd.points)
    c = P.mean(0)
    mn = P.min(0)
    mx = P.max(0)
    logger.info(
        f"{tag}: {len(P)} pts | center={_fmt(c)} | min={_fmt(mn)} | max={_fmt(mx)}"
    )


# ===================== TRANSFORMS: BASE <- TCP <- CAM ==========================
def build_T_base_tcp(p: Pose, unit_scale: float, euler_order: str) -> np.ndarray:
    R_tcp = euler_deg_to_R(p.rx, p.ry, p.rz, euler_order)
    t_tcp = np.array([p.x, p.y, p.z], dtype=np.float64) * unit_scale
    return make_T(R_tcp, t_tcp)


def get_T_tcp_cam(handeye_dir: str) -> np.ndarray:
    if handeye_dir == "tcp_cam":
        return make_T(HAND_EYE_R, HAND_EYE_T)  # TCP <- CAM
    elif handeye_dir == "cam_tcp_inv":
        T_cam_tcp = make_T(HAND_EYE_R, HAND_EYE_T)
        return np.linalg.inv(T_cam_tcp)  # TCP <- CAM
    else:
        raise ValueError("handeye_dir must be resolved")


def build_T_base_cam(
    p: Pose, unit_scale: float, euler_order: str, handeye_dir: str
) -> np.ndarray:
    T_base_tcp = build_T_base_tcp(p, unit_scale, euler_order)  # BASE <- TCP
    T_tcp_cam = get_T_tcp_cam(handeye_dir)  # TCP <- CAM
    return T_base_tcp @ T_tcp_cam  # BASE <- CAM


# ========================= AUTOTUNE ON FIRST FRAME ===========================
def autotune_extrinsics(
    pcd_cam: o3d.geometry.PointCloud, pose: Pose, aabb, *, report=True
):
    units = (
        [("meters", 1.0), ("millimeters", 0.001)]
        if POSE_UNIT_MODE == "auto"
        else [(POSE_UNIT_MODE, 1.0 if POSE_UNIT_MODE == "meters" else 0.001)]
    )
    orders = ["XYZ", "ZYX"] if POSE_EULER_ORDER == "auto" else [POSE_EULER_ORDER]
    hdirs = ["tcp_cam", "cam_tcp_inv"] if HAND_EYE_DIR == "auto" else [HAND_EYE_DIR]

    best = None
    P = sample_points(pcd_cam, 120_000)

    for u, scale in units:
        for order in orders:
            for hdir in hdirs:
                T = build_T_base_cam(pose, scale, order, hdir)
                Ptx = (T[:3, :3] @ P.T + T[:3, 3:4]).T
                hit = bbox_inliers_count(Ptx, aabb)
                if (best is None) or (hit > best["hit"]):
                    best = dict(
                        units=u, scale=scale, order=order, hdir=hdir, hit=hit, T=T
                    )

    if report:
        logger.info(
            f"[AUTOTUNE] chose units={best['units']} order={best['order']} handeye={best['hdir']} | inliers={best['hit']}"
        )
        logger.info(
            f"[AUTOTUNE] T_base_cam=\n{_fmt(best['T'][:3,:3])}\n t={_fmt(best['T'][:3,3])}"
        )
    return best["scale"], best["order"], best["hdir"]


# ============================= REGISTRATION ==================================
def optional_pairwise_refine(new_chunk, merged_ref) -> np.ndarray:
    """FGR -> robust ICP -> safe multi-scale Colored ICP."""
    if len(new_chunk.points) < 1000 or len(merged_ref.points) < 2000:
        return np.eye(4)
    # feature-level copies
    src = downsample(new_chunk, RD)
    tgt = downsample(merged_ref, RD)
    compute_normals(src, RN)
    compute_normals(tgt, RN)
    src_f = compute_fpfh(src, RF)
    tgt_f = compute_fpfh(tgt, RF)
    try:
        T0 = fgr(src, tgt, src_f, tgt_f, FGR_MAX_CORR_DIST)
        logger.info("[REFINE] FGR ok")
    except Exception as e:
        logger.warning(f"[REFINE] FGR fail: {e}")
        T0 = np.eye(4)
    # robust ICP (geometry)
    compute_normals(new_chunk, RN)
    compute_normals(merged_ref, RN)
    Ticp, fit, rmse = icp_pt2pl(
        new_chunk, merged_ref, T0, ICP_MAX_CORR_DIST, ICP_MAX_ITERS, ICP_TUKEY_K
    )
    # optional Colored ICP (photo+geo)
    if USE_COLORED_ICP:
        Ticp = colored_icp_multiscale(
            new_chunk,
            merged_ref,
            Ticp,
            COLORED_ICP_PYRAMID,
            COLORED_ICP_MAXIT,
            start_fitness=fit,
        )
    logger.info("[REFINE] ICP/ColoredICP ok")
    return Ticp


# ================================ PIPELINE ===================================
def merge_capture(root: Path, viz_mode: str = "all") -> o3d.geometry.PointCloud:
    """
    viz_mode: "all" (show intermediates) | "final" (only the final viewer).
    """
    assert viz_mode in ("all", "final")
    show_steps = viz_mode == "all"

    # intrinsics
    w, h, fx, fy, cx, cy = try_load_intrinsics_from_yaml(root)
    intr = o3d_intrinsics(w, h, fx, fy, cx, cy)

    # aabb
    _, aabb = crop_aabb(o3d.geometry.PointCloud(), BBOX_POINTS)
    aabb_ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        aabb
    ).paint_uniform_color((1, 0.1, 0.1))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=VIS_COORD_FRAME_SIZE)

    poses = load_poses(root / POSES_JSON)
    img_dir = root / IMG_DIR_NAME

    # preview cloud for refinement/overlay (PCD-based)
    merged_preview = o3d.geometry.PointCloud()

    # strategy: TSDF or pure PCD
    use_tsdf = MERGE_STRATEGY == "tsdf"
    if QUALITY_PRESET == "best":
        use_tsdf = True

    if use_tsdf:
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=TSDF_VOX, sdf_trunc=TSDF_TRUNC, color_type=TSDF_COLOR_TYPE
        )
        logger.info(
            f"[TSDF] Scalable volume: vox={TSDF_VOX:.4f} trunc={TSDF_TRUNC:.4f}"
        )

    # helper to load a pair
    def load_pair(stem):
        import cv2

        rgb_path = img_dir / f"{stem}_rgb.png"
        d_path = img_dir / f"{stem}_depth.npy"
        if not (rgb_path.exists() and d_path.exists()):
            logger.warning(f"[IO] missing {stem}")
            return None
        rgb = cv2.cvtColor(
            cv2.imread(str(rgb_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        depth = np.load(d_path)
        du = guess_depth_units(depth)
        depth = depth.astype(np.float32) * (0.001 if du == "millimeters" else 1.0)
        depth = prefilter_depth(depth)
        depth[~np.isfinite(depth)] = 0.0
        logger.info(f"[PAIR {stem}] rgb={rgb.shape} depth={depth.shape} units={du}")
        return rgb, depth

    # ---- first frame: build pcd_cam, AUTOTUNE extrinsics ----
    all_stems = sorted(poses.keys())
    if not all_stems:
        logger.warning("No poses found.")
        return o3d.geometry.PointCloud()

    first = all_stems[0]
    pair0 = load_pair(first)
    if pair0 is None:
        return o3d.geometry.PointCloud()
    rgb0, depth0 = pair0
    pcd0_cam = rgbd_to_pcd(rgb0, depth0, intr)
    log_cloud_stats(f"[S1 {first}] CAMERA", pcd0_cam)

    scale, order, hdir = autotune_extrinsics(pcd0_cam, poses[first], aabb)
    T0 = build_T_base_cam(poses[first], scale, order, hdir)
    pcd0_base = o3d.geometry.PointCloud(pcd0_cam).transform(T0.copy())
    log_cloud_stats(f"[S2 {first}] BASE (autotune)", pcd0_base)
    pcd0_crop = pcd0_base.crop(aabb)
    log_cloud_stats(f"[S3 {first}] CROP", pcd0_crop)
    if show_steps:
        o3d.visualization.draw_geometries(
            [pcd0_base.paint_uniform_color((0.2, 0.6, 1.0)), aabb_ls, axes],
            window_name=f"DIAG frame {first} after T (BASE)",
        )

    # integrate first frame (TSDF)
    if use_tsdf:
        extrinsic = np.linalg.inv(T0)  # TSDF expects camera extrinsic (CAM <- BASE)
        vol.integrate(make_rgbd(rgb0, depth0), intr, extrinsic)

    # add to preview (cropped)
    if len(pcd0_crop.points) > 0:
        pcd0_clean = downsample(pcd0_crop, FRAME_VOX)
        if REMOVE_OUTLIERS and len(pcd0_clean.points) > 500:
            pcd0_clean, _ = clean_outliers(pcd0_clean, OUTLIER_NN, OUTLIER_STD)
        merged_preview += pcd0_clean

    # ---- main loop over frames ----
    for i, stem in enumerate(all_stems):
        pair = load_pair(stem)
        if pair is None:
            continue
        rgb, depth = pair
        pcd_cam = rgbd_to_pcd(rgb, depth, intr)
        T = build_T_base_cam(poses[stem], scale, order, hdir)
        logger.info(f"[S2 {stem}] T_base_cam R=\n{_fmt(T[:3,:3])}\n t={_fmt(T[:3,3])}")

        # Transform to BASE
        pcd_base = o3d.geometry.PointCloud(pcd_cam).transform(T.copy())
        if show_steps and (i % VIS_EVERY_N == 0):
            o3d.visualization.draw_geometries(
                [pcd_base.paint_uniform_color((0.2, 0.6, 1.0)), aabb_ls, axes],
                window_name=f"S2 BASE - {stem}",
            )

        # Crop in BASE (ROI)
        pcd_crop = pcd_base.crop(aabb)
        logger.info(
            f"[S3 {stem}] Crop -> {len(pcd_crop.points)} pts (was {len(pcd_base.points)})"
        )
        if len(pcd_crop.points) < 200:
            logger.warning(f"[{stem}] skipped (tiny after crop)")
            # still integrate TSDF to avoid holes
            if use_tsdf:
                vol.integrate(make_rgbd(rgb, depth), intr, np.linalg.inv(T))
            continue

        # Clean/downsample
        pcd_clean = downsample(pcd_crop, FRAME_VOX)
        removed = 0
        if REMOVE_OUTLIERS and len(pcd_clean.points) > 500:
            pcd_clean, removed = clean_outliers(pcd_clean, OUTLIER_NN, OUTLIER_STD)
        logger.info(
            f"[S4 {stem}] Clean {len(pcd_crop.points)} -> {len(pcd_clean.points)} (removed {removed})"
        )

        # Pairwise refine against preview
        if APPLY_PAIRWISE_REFINEMENT and len(merged_preview.points) > 5000:
            if show_steps and (i % VIS_EVERY_N == 0):
                o3d.visualization.draw_geometries(
                    [
                        merged_preview.paint_uniform_color((0.75, 0.75, 0.75)),
                        pcd_clean.paint_uniform_color((0.2, 0.6, 1.0)),
                        aabb_ls,
                        axes,
                    ],
                    window_name=f"S5 Before refine - {stem}",
                )
            Tref = optional_pairwise_refine(pcd_clean, merged_preview)
            if not np.allclose(Tref, np.eye(4)):
                pcd_clean = o3d.geometry.PointCloud(pcd_clean).transform(Tref.copy())
                if show_steps and (i % VIS_EVERY_N == 0):
                    o3d.visualization.draw_geometries(
                        [
                            merged_preview.paint_uniform_color((0.75, 0.75, 0.75)),
                            pcd_clean.paint_uniform_color((0.1, 0.9, 0.3)),
                            aabb_ls,
                            axes,
                        ],
                        window_name=f"S6 After refine - {stem}",
                    )

        # Update preview accumulator
        merged_preview += pcd_clean
        if len(merged_preview.points) > 300_000:
            n0 = len(merged_preview.points)
            merged_preview = downsample(merged_preview, MERGE_VOX)
            logger.info(f"[ACC] preview compact {n0} -> {len(merged_preview.points)}")

        # Integrate TSDF (full frame) or keep PCD merge
        if use_tsdf:
            vol.integrate(make_rgbd(rgb, depth), intr, np.linalg.inv(T))

    # ---- finalize
    if use_tsdf:
        merged = vol.extract_point_cloud()
        logger.info(f"[TSDF] extracted cloud: {len(merged.points)} pts before crop")
        merged = merged.crop(aabb)  # enforce ROI again
    else:
        merged = merged_preview

    if len(merged.points) == 0:
        logger.warning("[FINAL] Empty merged cloud.")
        return merged

    n0 = len(merged.points)
    merged = downsample(merged, MERGE_VOX)
    logger.info(f"[FINAL] Downsample {n0} -> {len(merged.points)} @ {MERGE_VOX}")
    if REMOVE_OUTLIERS and len(merged.points) > 2000:
        merged, rm = clean_outliers(merged, OUTLIER_NN, OUTLIER_STD)
        logger.info(f"[FINAL] Outliers removed: {rm}")
    return merged


# =========================== FINAL EDIT/EXPORT ================================
def final_edit_and_export(
    cloud: o3d.geometry.PointCloud, root: Path, prefix: str = "final"
):
    """
    Opens an editing viewer, then saves picked indices, 3D coords, and helper geometries.
    Files go to <root>/debug/.
    """
    dbg = root / DEBUG_DIR_NAME
    dbg.mkdir(parents=True, exist_ok=True)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="FINAL - edit & save picks", width=1280, height=800)
    vis.add_geometry(cloud)
    vis.run()
    vis.destroy_window()

    picks = vis.get_picked_points()
    logger.info(
        f"[PICKS] picked {len(picks)} points: {picks[:10]}{'...' if len(picks)>10 else ''}"
    )

    P = np.asarray(cloud.points)
    coords = [P[i].tolist() for i in picks] if len(picks) else []
    import json as _json

    (dbg / f"{prefix}_picks.json").write_text(
        _json.dumps(
            {"count": len(picks), "indices": picks, "points_xyz": coords}, indent=2
        )
    )

    # markers/polyline export
    spheres, lines = [], []
    for i, pi in enumerate(picks):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=PICK_SPHERE_R)
        s.translate(P[pi])
        s.compute_vertex_normals()
        s.paint_uniform_color((1, 0.25, 0))
        spheres.append(s)
        if i > 0:
            lines.append([i - 1, i])

    markers = o3d.geometry.TriangleMesh()
    for s in spheres:
        markers += s
    o3d.io.write_triangle_mesh(str(dbg / f"{prefix}_pick_markers.ply"), markers)
    if len(picks) >= 2:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector([P[i] for i in picks])
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(lines))
        o3d.io.write_line_set(str(dbg / f"{prefix}_pick_polyline.ply"), ls)
    if len(picks) >= 3:
        Q = np.asarray(coords)
        c = Q.mean(0)
        _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
        n = Vt[-1] / np.linalg.norm(Vt[-1])
        d = -float(n @ c)
        (dbg / f"{prefix}_plane.json").write_text(
            _json.dumps(
                {"normal": n.tolist(), "d": d, "point_on_plane": c.tolist()}, indent=2
            )
        )


# =========================== SAVE FINAL CLOUD ================================
def save_final_cloud(
    cloud: o3d.geometry.PointCloud, root: Path, name: str = "final_merged.ply"
) -> Path:
    """
    Save final merged cloud under <root>/debug/<name>.
    Returns saved path.
    """
    out_dir = root / DEBUG_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    ok = o3d.io.write_point_cloud(str(out_path), cloud)
    if ok:
        logger.info(f"[SAVE] Final cloud saved: {out_path}")
    else:
        logger.warning(f"[SAVE] Failed to save cloud: {out_path}")
    return out_path


# ================================== MAIN =====================================
def main():
    logger.info(f"[START] Root={CAPTURE_ROOT}")
    cloud = merge_capture(CAPTURE_ROOT, viz_mode="final")  # "all" or "final"
    if len(cloud.points) == 0:
        print("Empty cloud - nothing to visualize.")
        return
    SAVE_FINAL_CLOUD = True
    if SAVE_FINAL_CLOUD:
        save_final_cloud(cloud, CAPTURE_ROOT, name="final_merged.ply")
    if CAPTURE_PICKS:
        final_edit_and_export(cloud, CAPTURE_ROOT, prefix="final")
    else:
        o3d.visualization.draw_geometries_with_editing(
            [cloud], window_name="FINAL - draw_geometries_with_editing"
        )


if __name__ == "__main__":
    main()
