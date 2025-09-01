# vision/skeleton/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List, Tuple
import numpy as np

from utils import config as ucfg

SkelMode = Literal["vol3d", "plane2d"]


@dataclass(frozen=True)
class SmoothingCfg:
    enabled: bool = True
    iters: int = 2
    radius: float = 0.010
    min_nn: int = 20
    alpha_min: float = 0.15
    alpha_max: float = 1.00
    use_fpfh: bool = True
    fpfh_radius: float = 0.020
    fpfh_max_nn: int = 100
    fpfh_down_vox: float = 0.006


@dataclass(frozen=True)
class VoxelCfg:
    vsize: float = 0.0035
    pad: int = 3
    max_voxels: int = 60_000_000
    dilate_r: int = 1
    min_skel_voxels: int = 200


@dataclass(frozen=True)
class Skel3DCfg:
    resample_step: float = 0.10
    keep_tail_min_frac: float = 0.7


@dataclass(frozen=True)
class GraphCfg:
    node_merge_radius_m: float = 0.004
    min_branch_length_m: float = 0.01


@dataclass(frozen=True)
class NodeRefineCfg:
    bridge_width_est: float = 0.07
    search_radius: float = 0.03
    min_side_pts: int = 20
    normal_radius: float = 0.012
    normal_max_nn: int = 60
    merge_radius_m: float = 0.004


@dataclass(frozen=True)
class Skel2DCfg:
    ds_vox_seg: float = 0.0035
    dist_schedule: Tuple[float, ...] = (0.003, 0.005, 0.008, 0.012, 0.02)
    ransac_n: int = 3
    iters: int = 4000
    max_planes: int = 6
    min_plane_pts_abs: int = 500
    min_plane_pts_frac: float = 0.03
    grid_res: float = 0.002
    morph_close_k: int = 3
    method: Literal["skimage", "naive"] = "skimage"
    use_capture_poses: bool = True
    poses_json: str = ucfg.POSES_JSON


@dataclass(frozen=True)
class NodeRefineCfg:
    bridge_width_est: float = 0.07
    search_radius: float = 0.03
    min_side_pts: int = 20
    normal_radius: float = 0.007
    normal_max_nn: int = 60
    merge_radius_m: float = 0.005


@dataclass(frozen=True)
class CenterRefineCfg:
    search_r: float = 0.035
    min_side_pts: int = 20
    exp_width: float = 0.07
    straighten: bool = True
    straight_min_len: float = 0.08
    straight_lin_thr: float = 0.08
    resample_step: float = 0.10
    keep_tail_min_frac: float = 0.7


@dataclass(frozen=True)
class GroupingCfg:
    bins_deg: Tuple[float, float, float, float] = (15, 35, 55, 75)


@dataclass(frozen=True)
class NormalsCfg:
    angle_bins_deg: Tuple[float, float, float, float, float] = (
        15.0,
        35.0,
        55.0,
        75.0,
        90.0,
    )
    palette_rgb: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ] = (
        (1.0, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.7, 0.0),
        (0.0, 0.4, 1.0),
    )


@dataclass(frozen=True)
class RegionsCfg:
    # v1 bins per angle
    radius: float = 0.015
    normal_thr_deg: float = 18.0
    min_pts: int = 250
    bnd_k: int = 24
    bnd_frac: float = 0.75
    # v2 in-plane growth
    side_min_deg: float = 15.0
    grow_radius: float = 0.016
    ip_normal_thr_deg: float = 15.0
    min_region_pts: int = 180
    subfaces_k: int = 4


@dataclass(frozen=True)
class ViewerCfg:
    point_size: float = 2.0
    coord_frame_size: float = 0.05
    help_title: str = "1:Cloud 2:Skel 3:Graph 4:Centers 5:Members "
    # extended:
    help_tail: str = "6:Normals 7:Regions(v2) 8:Sides  Shift+LMB: Pick"
    window: Tuple[int, int] = (1280, 800)
    title_prefix: str = "Skeleton"
    side_regions_as_mesh: bool = False
    mesh_ball_pivot_r: float = 0.005
    mesh_decimate_target: int = 40_000
    side_regions_as_mesh: bool = True
    show_boundary_points: bool = True


@dataclass(frozen=True)
class SkelPipelineCfg:
    mode: SkelMode = "plane2d"
    capture_root: str = str(ucfg.CAPTURE_ROOT)
    input_cloud_path: str | None = None
    smoothing: SmoothingCfg = SmoothingCfg()
    vox: VoxelCfg = VoxelCfg()
    sk3d: Skel3DCfg = Skel3DCfg()
    sk2d: Skel2DCfg = Skel2DCfg()
    node_refine: NodeRefineCfg = NodeRefineCfg()
    center_refine: CenterRefineCfg = CenterRefineCfg()
    grouping: GroupingCfg = GroupingCfg()
    normals: NormalsCfg = NormalsCfg()
    regions: RegionsCfg = RegionsCfg()
    viewer: ViewerCfg = ViewerCfg()
    debug_dir_name: str = "debug"
