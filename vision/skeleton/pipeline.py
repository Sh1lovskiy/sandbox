# vision/skeleton/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d

from utils.logger import Logger

from .config import SkeletonCfg
from .geometry import (
    PlaneBasis,
    plane_basis_from_coeffs,
    resample_polyline_world,
    straighten_polylines,
)
from .nodes import (
    estimate_dominant_plane_normal,
    group_polylines_by_angle,
    merge_close_nodes_world,
    refine_nodes_by_normals,
)
from .planes import (
    morph_close,
    pick_nearest_plane_to_cameras,
    rasterize_points,
    segment_planes_iterative,
)
from .skeleton2d import skeleton_to_graph_2d, skeletonize_2d
from .skeleton3d import (
    make_lineset_from_world_polylines,
    make_nodes_mesh,
    node_vox_to_world,
    vox_path_to_world,
)
from .grouping import (
    build_side_surface_clouds,
    build_side_surface_regions,
    build_side_surface_regions_v2,
    make_normals_colored_cloud,
)
from .voxelize import (
    compute_voxel_grid_params,
    dilate_3d,
    skeleton_nodes_vox,
    skeleton_to_polylines_3d,
    voxelize_points,
)
from .refine import (
    _debug_dir,
    export_skeleton_model,
    save_cloud,
    save_lineset,
    save_mesh,
    save_normals_report,
)

LOG = Logger.get_logger("skel_pipe")


@dataclass(frozen=True)
class ViewerLayers:
    cloud: List[o3d.geometry.Geometry]
    skeleton: List[o3d.geometry.Geometry]
    graph: List[o3d.geometry.Geometry]
    centers: List[o3d.geometry.Geometry]
    members: List[o3d.geometry.Geometry]
    normals: List[o3d.geometry.Geometry]
    side_regions: _
