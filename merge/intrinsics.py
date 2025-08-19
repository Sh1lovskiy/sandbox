from __future__ import annotations

from pathlib import Path
from typing import Tuple

import open3d as o3d

from utils.config import CAMERA_FALLBACK, CameraDefaults
from utils.io import load_intrinsics_from_json
from utils.logger import Logger

LOG = Logger.get_logger("intrinsics")


def _tuple_to_intrinsics(
    t: Tuple[int, int, float, float, float, float],
) -> o3d.camera.PinholeCameraIntrinsic:
    """Create Open3D intrinsics from (w,h,fx,fy,cx,cy)."""
    w, h, fx, fy, cx, cy = t
    return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)


def build_intrinsics(root: Path) -> o3d.camera.PinholeCameraIntrinsic:
    """Prefer JSON intrinsics near capture root; fallback to defaults."""
    intr = load_intrinsics_from_json(root, CAMERA_FALLBACK)
    LOG.info(
        f"[INTR] w={intr[0]} h={intr[1]} fx={intr[2]:.3f} fy={intr[3]:.3f} cx={intr[4]:.3f} cy={intr[5]:.3f}"
    )
    return _tuple_to_intrinsics(intr)
