from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import open3d as o3d

from utils.config import HandEye, Pose, HAND_EYE, CAPTURE_ROOT
from utils.helpers import (
    guess_depth_units,
    prefilter_depth,
)
from utils.io import load_poses
from utils.logger import Logger

from .config import MAX_PREVIEW_POINTS, PipelineCfg
from .intrinsics import build_intrinsics
from .rgbd import rgbd_to_pcd, prefilter_depth_m
from .roi import crop_to_aabb, log_cloud_stats, make_aabb
from .registration import RefineParams, pairwise_refine
from .transforms import (
    autotune_extrinsics,
    build_T_base_cam,
    get_T_tcp_cam,
)
from .tsdf import build_tsdf, extract_cloud, integrate_frame
from .viz import draw_with_roi

LOG = Logger.get_logger("pipeline")


# ============================== HELPERS ======================================


def _load_pair(
    img_dir: Path, stem: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read <stem>_rgb.png + <stem>_depth.npy."""
    try:
        import cv2
    except Exception as e:
        LOG.error(f"OpenCV missing: {e}")
        return None

    rgb_path = img_dir / f"{stem}_rgb.png"
    d_path = img_dir / f"{stem}_depth.npy"
    if not (rgb_path.exists() and d_path.exists()):
        LOG.warning(f"[IO] missing pair for {stem}")
        return None

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth_raw = np.load(d_path)
    units = guess_depth_units(depth_raw, mode="auto")
    scale = 0.001 if units == "millimeters" else 1.0
    depth_m = depth_raw.astype(np.float32) * scale
    depth_m[~np.isfinite(depth_m)] = 0.0
    LOG.info(
        f"[PAIR {stem}] rgb={rgb.shape} depth={depth_raw.shape} units={units}"
    )
    return rgb, depth_m


def _refine_if_needed(
    cfg: PipelineCfg,
    pcd_clean: o3d.geometry.PointCloud,
    preview: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    """Optional pairwise refine using hybrid chain."""
    if len(preview.points) <= 5000:
        return pcd_clean
    params = RefineParams(
        rd=cfg.reg.rd,
        rn=cfg.reg.rn,
        rf=cfg.reg.rf,
        fgr_max_corr=cfg.reg.fgr_max_corr,
        icp_max_corr=cfg.reg.icp_max_corr,
        icp_max_iters=cfg.reg.icp_max_iters,
        icp_tukey_k=cfg.reg.icp_tukey_k,
        use_colored_icp=cfg.reg.use_colored_icp,
        cicp_pyramid=cfg.reg.cicp_pyramid,
        cicp_iters=cfg.reg.cicp_iters,
        cicp_enable_min_fitness=cfg.reg.cicp_enable_min_fitness,
        cicp_continue_min_fitness=cfg.reg.cicp_continue_min_fitness,
        cicp_fail_scale_up=cfg.reg.cicp_fail_scale_up,
    )
    Tref = pairwise_refine(pcd_clean, preview, params)
    if np.allclose(Tref, np.eye(4)):
        return pcd_clean
    return o3d.geometry.PointCloud(pcd_clean).transform(Tref.copy())


def _clean_and_downsample(
    pcd: o3d.geometry.PointCloud, cfg: PipelineCfg
) -> Tuple[o3d.geometry.PointCloud, int]:
    """Voxel-downsample and optional statistical outlier removal."""
    pc = pcd.voxel_down_sample(cfg.frame_vox) if cfg.frame_vox > 0 else pcd
    removed = 0
    if cfg.remove_outliers and len(pc.points) > 500:
        _, idx = pc.remove_statistical_outlier(
            nb_neighbors=cfg.outlier_nn, std_ratio=cfg.outlier_std
        )
        removed = len(pc.points) - len(idx)
        pc = pc.select_by_index(idx)
    return pc, removed


def _maybe_compact_preview(
    preview: o3d.geometry.PointCloud, cfg: PipelineCfg
) -> o3d.geometry.PointCloud:
    """Limit preview size to keep registration fast."""
    if len(preview.points) <= MAX_PREVIEW_POINTS:
        return preview
    n0 = len(preview.points)
    preview = preview.voxel_down_sample(cfg.merge_vox)
    LOG.info(f"[ACC] compact preview {n0} -> {len(preview.points)}")
    return preview


# ============================== PIPELINE =====================================


def merge_capture(root: Path, cfg: PipelineCfg) -> o3d.geometry.PointCloud:
    """Main merge routine: RGB-D -> BASE -> crop -> refine -> accumulate."""
    root = Path(root)
    intr = build_intrinsics(root)
    aabb = make_aabb(cfg.bbox_points)

    poses = load_poses(root / cfg.poses_json)
    stems = sorted(poses.keys())
    if not stems:
        LOG.warning(f"No poses found at {root / cfg.poses_json}")
        return o3d.geometry.PointCloud()

    img_dir = root / cfg.img_dir
    he = HAND_EYE  # from utils.config

    # Preview accumulator (PCD-based)
    preview = o3d.geometry.PointCloud()

    # TSDF volume if needed
    vol = None
    if cfg.use_tsdf():
        vol = build_tsdf(
            cfg.reg.rd * 0.4 if cfg.quality_preset == "best" else 0.004,
            cfg.reg.rd * 1.6 if cfg.quality_preset == "best" else 0.016,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    # --- First frame: autotune extrinsics
    rgb0, depth0 = _load_pair(img_dir, stems[0]) or (None, None)
    if rgb0 is None:
        return o3d.geometry.PointCloud()

    pcd0_cam = rgbd_to_pcd(rgb0, depth0, intr, depth_trunc=cfg.depth_trunc)
    log_cloud_stats(f"[S1 {stems[0]}] CAMERA", pcd0_cam)

    scale, order, hdir = autotune_extrinsics(
        pcd0_cam,
        poses[stems[0]],
        aabb,
        cfg.pose_unit_mode,
        cfg.euler_mode,
        cfg.he_dir_mode,
        he,
    )
    T0 = build_T_base_cam(
        poses[stems[0]], scale, order, HandEye(R=he.R, t=he.t, direction=hdir)
    )
    pcd0_base = o3d.geometry.PointCloud(pcd0_cam).transform(T0.copy())
    log_cloud_stats(f"[S2 {stems[0]}] BASE (autotune)", pcd0_base)
    pcd0_crop = crop_to_aabb(pcd0_base, aabb)
    log_cloud_stats(f"[S3 {stems[0]}] CROP", pcd0_crop)

    if cfg.viz_stages:
        draw_with_roi(
            pcd0_base.paint_uniform_color((0.2, 0.6, 1.0)),
            aabb,
            coord_frame_size=cfg.coord_frame_size,
            title=f"Diag BASE {stems[0]}",
        )

    # Integrate first frame
    if vol is not None:
        integrate_frame(vol, intr, rgb0, depth0, T0)

    # Seed preview
    if len(pcd0_crop.points) > 0:
        pcd0_clean, _ = _clean_and_downsample(pcd0_crop, cfg)
        preview += pcd0_clean

    # --- Main loop
    for i, stem in enumerate(stems):
        pair = _load_pair(img_dir, stem)
        if pair is None:
            continue
        rgb, depth = pair
        depth = prefilter_depth_m(depth, cfg)
        pcd_cam = rgbd_to_pcd(rgb, depth, intr, depth_trunc=cfg.depth_trunc)

        T = build_T_base_cam(
            poses[stem], scale, order, HandEye(R=he.R, t=he.t, direction=hdir)
        )
        pcd_base = o3d.geometry.PointCloud(pcd_cam).transform(T.copy())

        if cfg.viz_stages and (i % cfg.viz_every_n == 0):
            draw_with_roi(
                pcd_base.paint_uniform_color((0.2, 0.6, 1.0)),
                aabb,
                coord_frame_size=cfg.coord_frame_size,
                title=f"S2 BASE - {stem}",
            )

        pcd_crop = crop_to_aabb(pcd_base, aabb)
        LOG.info(
            f"[S3 {stem}] Crop -> {len(pcd_crop.points)} pts (was {len(pcd_base.points)})"
        )

        if len(pcd_crop.points) < 200:
            LOG.warning(f"[{stem}] skipped (tiny after crop)")
            if vol is not None:
                integrate_frame(vol, intr, rgb, depth, T)
            continue

        pcd_clean, removed = _clean_and_downsample(pcd_crop, cfg)
        LOG.info(
            f"[S4 {stem}] Clean {len(pcd_crop.points)} -> {len(pcd_clean.points)} (removed {removed})"
        )

        if len(preview.points) > 5000:
            pcd_clean = _refine_if_needed(cfg, pcd_clean, preview)

        preview += pcd_clean
        preview = _maybe_compact_preview(preview, cfg)

        if vol is not None:
            integrate_frame(vol, intr, rgb, depth, T)

    # --- Finalize
    if vol is not None:
        merged = extract_cloud(vol).crop(aabb)
    else:
        merged = preview

    if len(merged.points) == 0:
        LOG.warning("[FINAL] Empty cloud.")
        return merged

    n0 = len(merged.points)
    merged = merged.voxel_down_sample(cfg.merge_vox)
    LOG.info(
        f"[FINAL] Downsample {n0} -> {len(merged.points)} @ {cfg.merge_vox:.4f}"
    )

    if cfg.remove_outliers and len(merged.points) > 2000:
        _, idx = merged.remove_statistical_outlier(
            nb_neighbors=cfg.outlier_nn, std_ratio=cfg.outlier_std
        )
        removed = len(merged.points) - len(idx)
        merged = merged.select_by_index(idx)
        LOG.info(f"[FINAL] Outliers removed: {removed}")

    return merged
