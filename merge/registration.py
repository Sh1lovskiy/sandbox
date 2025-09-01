# merge/registration.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import open3d as o3d

from utils.helpers import SuppressO3DInfo, CaptureStderrToLogger
from utils.logger import Logger

LOG = Logger.get_logger("registration")


@dataclass(frozen=True)
class RefineParams:
    """Wrapper for pairwise refinement thresholds/iters."""

    rd: float
    rn: float
    rf: float
    fgr_max_corr: float
    icp_max_corr: float
    icp_max_iters: int
    icp_tukey_k: float
    use_colored_icp: bool
    cicp_pyramid: List[float]
    cicp_iters: List[int]
    cicp_enable_min_fitness: float
    cicp_continue_min_fitness: float
    cicp_fail_scale_up: float


def _downsample(
    pcd: o3d.geometry.PointCloud, vox: float
) -> o3d.geometry.PointCloud:
    return pcd.voxel_down_sample(vox) if vox and vox > 0 else pcd


def _ensure_normals(pcd: o3d.geometry.PointCloud, radius: float) -> None:
    if pcd.has_normals():
        return
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60)
    )
    pcd.orient_normals_consistent_tangent_plane(50)


def _compute_fpfh(pcd: o3d.geometry.PointCloud, radius: float):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )


def fast_global_registration(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    src_fpfh,
    tgt_fpfh,
    max_corr: float,
) -> np.ndarray:
    """Feature-based global alignment (FGR)."""
    rk = o3d.pipelines.registration
    opt = rk.FastGlobalRegistrationOption(
        maximum_correspondence_distance=max_corr, iteration_number=64
    )
    res = rk.registration_fgr_based_on_feature_matching(
        src, tgt, src_fpfh, tgt_fpfh, opt
    )
    return res.transformation


def _make_robust_kernel(tukey_k: float):
    """Return an Open3D robust kernel if available."""
    rk = o3d.pipelines.registration
    if hasattr(rk, "TukeyLoss"):
        return rk.TukeyLoss(k=float(tukey_k))
    if hasattr(rk, "RobustKernel") and hasattr(rk, "RobustKernelType"):
        return rk.RobustKernel(rk.RobustKernelType.Tukey, float(tukey_k))
    return None


def robust_icp_point_to_plane(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    init: np.ndarray,
    max_corr: float,
    iters: int,
    tukey_k: float,
) -> tuple[np.ndarray, float, float]:
    """Point-to-plane ICP with optional Tukey loss."""
    rk = o3d.pipelines.registration
    loss = _make_robust_kernel(tukey_k)
    est = (
        rk.TransformationEstimationPointToPlane(loss)
        if loss
        else rk.TransformationEstimationPointToPlane()
    )
    crit = rk.ICPConvergenceCriteria(max_iteration=iters)
    res = rk.registration_icp(src, tgt, max_corr, init, est, crit)
    LOG.info(f"fitness={res.fitness:.4f} rmse={res.inlier_rmse:.6f}")
    return res.transformation, float(res.fitness), float(res.inlier_rmse)


def _colored_icp_level(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    T: np.ndarray,
    dist: float,
    iters: int,
):
    rk = o3d.pipelines.registration
    return rk.registration_colored_icp(
        src,
        tgt,
        dist,
        T,
        rk.TransformationEstimationForColoredICP(),
        rk.ICPConvergenceCriteria(max_iteration=iters),
    )


def colored_icp_multiscale(
    src_in: o3d.geometry.PointCloud,
    tgt_in: o3d.geometry.PointCloud,
    init: np.ndarray,
    dists: List[float],
    iters: List[int],
    start_fitness: float,
    enable_min: float = 0.20,
    cont_min: float = 0.30,
    fail_scale: float = 2.0,
) -> np.ndarray:
    """Safe multi-scale Colored ICP with fitness gates."""
    if start_fitness < enable_min:
        LOG.info(
            f"[CICP] skipped (fitness {start_fitness:.3f} < {enable_min:.2f})"
        )
        return init

    src = _downsample(o3d.geometry.PointCloud(src_in), vox=0.006)
    tgt = _downsample(o3d.geometry.PointCloud(tgt_in), vox=0.006)
    _ensure_normals(src, dists[0] * 2.0)
    _ensure_normals(tgt, dists[0] * 2.0)

    if start_fitness < cont_min:
        dists = [d * fail_scale for d in dists]
        LOG.info(f"[CICP] inflate thresholds x{fail_scale:.1f} (weak start)")

    T = init.copy()
    last_T = T
    last_fit = start_fitness

    # Capture Open3D C++ stderr spam into our logger and suppress info noise.
    with SuppressO3DInfo(), CaptureStderrToLogger(LOG):
        for lvl, (dist, itn) in enumerate(zip(dists, iters)):
            try:
                res = _colored_icp_level(src, tgt, T, dist, itn)
            except RuntimeError as e:
                # Open3D may print to stderr; the context will redirect it.
                LOG.warning("[CICP L{lvl}] aborted: {e}")
                break

            T = res.transformation
            last_T = T
            last_fit = float(res.fitness)
            LOG.info(
                f"[CICP L{lvl}] thr={dist:.4f} it={itn} fit={res.fitness:.3f} rmse={res.inlier_rmse:.5f}"
            )
            if last_fit < cont_min:
                LOG.info(
                    f"[CICP] early stop at L{lvl} (fit {last_fit:.3f} < {cont_min:.2f})"
                )
                break

    return last_T


def pairwise_refine(
    chunk: o3d.geometry.PointCloud,
    reference: o3d.geometry.PointCloud,
    params: RefineParams,
) -> np.ndarray:
    """FGR -> robust ICP -> optional Colored ICP."""
    if len(chunk.points) < 1000 or len(reference.points) < 2000:
        return np.eye(4)

    src = _downsample(chunk, params.rd)
    tgt = _downsample(reference, params.rd)
    _ensure_normals(src, params.rn)
    _ensure_normals(tgt, params.rn)
    sf = _compute_fpfh(src, params.rf)
    tf = _compute_fpfh(tgt, params.rf)

    try:
        T0 = fast_global_registration(src, tgt, sf, tf, params.fgr_max_corr)
        LOG.info("[REFINE] FGR ok")
    except Exception as e:
        LOG.warning(f"[REFINE] FGR failed: {e}")
        T0 = np.eye(4)

    _ensure_normals(chunk, params.rn)
    _ensure_normals(reference, params.rn)
    Ticp, fit, _ = robust_icp_point_to_plane(
        chunk,
        reference,
        T0,
        params.icp_max_corr,
        params.icp_max_iters,
        params.icp_tukey_k,
    )

    if not params.use_colored_icp:
        return Ticp
    if fit < 0.2:
        LOG.warning(f"[REFINE] ICP fit too low ({fit:.3f}) -> skip refine")
        return np.eye(4)
    return colored_icp_multiscale(
        chunk,
        reference,
        Ticp,
        params.cicp_pyramid,
        params.cicp_iters,
        start_fitness=fit,
        enable_min=params.cicp_enable_min_fitness,
        cont_min=params.cicp_continue_min_fitness,
        fail_scale=params.cicp_fail_scale_up,
    )
