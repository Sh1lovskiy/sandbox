from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import open3d as o3d

from utils import config as ucfg

# ============================== CONSTANTS ====================================

MAX_PREVIEW_POINTS = 300_000
DEBUG_DIR_NAME = "debug"

# ============================== CONFIG TYPES =================================


@dataclass(frozen=True)
class DepthFilterCfg:
    """Depth bilateral filter parameters (meters domain)."""

    use_bilateral: bool = True
    diameter_px: int = 5
    sigma_color_m: float = 0.03
    sigma_space_px: float = 3.0


@dataclass(frozen=True)
class RegistrationCfg:
    """Distances and iterations for the hybrid FGR -> ICP -> ColoredICP."""

    rd: float = 0.01  # downsample voxel for features
    rn: float = 0.02  # normal estimation radius
    rf: float = 0.05  # FPFH radius
    fgr_max_corr: float = 0.015
    icp_max_corr: float = 0.008
    icp_max_iters: int = 40
    icp_tukey_k: float = 0.03
    use_colored_icp: bool = True
    cicp_pyramid: List[float] = field(
        default_factory=lambda: [0.02, 0.01, 0.005]
    )
    cicp_iters: List[int] = field(default_factory=lambda: [30, 20, 15])
    cicp_enable_min_fitness: float = 0.20
    cicp_continue_min_fitness: float = 0.30
    cicp_fail_scale_up: float = 2.0


@dataclass(frozen=True)
class TSDFCfg:
    """TSDF volume parameters (Open3D scalable TSDF)."""

    voxel: float = 0.004
    trunc: float = 0.004 * 4.0
    color_type: o3d.pipelines.integration.TSDFVolumeColorType = (
        o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )


@dataclass(frozen=True)
class PipelineCfg:
    """Top-level knobs for the merge pipeline."""

    # I/O and paths (root from utils.config by default)
    capture_root: str = ucfg.CAPTURE_ROOT
    img_dir: str = ucfg.IMG_DIR_NAME
    poses_json: str = ucfg.POSES_JSON

    # Strategy
    quality_preset: Literal["fast", "best"] = "best"
    merge_strategy: Literal["pcd", "tsdf"] = "tsdf"

    # Per-frame / final processing
    frame_vox: float = 0.003
    merge_vox: float = 0.002
    remove_outliers: bool = True
    outlier_nn: int = 20
    outlier_std: float = 2.0

    # Depth
    depth_trunc: float = ucfg.DEPTH_TRUNC
    depth_filter: DepthFilterCfg = DepthFilterCfg()

    # Registration
    reg: RegistrationCfg = RegistrationCfg()

    # ROI
    bbox_points: np.ndarray = field(
        default_factory=lambda: ucfg.BBOX_POINTS.copy()
    )

    # Auto modes
    pose_unit_mode: Literal["auto", "meters", "millimeters"] = (
        ucfg.POSE_UNIT_MODE
    )
    euler_mode: Literal["auto", "XYZ", "ZYX"] = ucfg.POSE_EULER_ORDER
    he_dir_mode: Literal["auto", "tcp_cam", "cam_tcp_inv"] = ucfg.HAND_EYE_DIR

    # Visualization
    viz_stages: bool = True
    viz_every_n: int = 1
    coord_frame_size: float = 0.05
    point_size: float = 2.0
    open_final_viewer: bool = True

    # Picks / export
    capture_picks: bool = False
    pick_sphere_r: float = 0.004
    debug_dir_name: str = DEBUG_DIR_NAME

    # Derived switches
    def use_tsdf(self) -> bool:
        if self.quality_preset == "best":
            return True
        return self.merge_strategy == "tsdf"
