# utils/helpers.py
from __future__ import annotations

import math
import numpy as np

from .logger import Logger, CaptureStderrToLogger, SuppressO3DInfo

# ============================================================================ #
# Logger / numpy
# ============================================================================ #
logger = Logger.get_logger("helpers")
np.set_printoptions(suppress=True, precision=6, linewidth=180)


def setup_numpy_print(precision: int = 6, linewidth: int = 180) -> None:
    """Consistent numpy printing for debugging."""
    np.set_printoptions(suppress=True, precision=precision, linewidth=linewidth)


# ============================================================================ #
# Math: rotations, transforms, formatting
# ============================================================================ #
def euler_deg_to_R(
    rx: float, ry: float, rz: float, order: str = "XYZ"
) -> np.ndarray:
    """Euler angles in degrees -> 3x3 rotation matrix (right-multiplied)."""
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(rx), -math.sin(rx)],
            [0.0, math.sin(rx), math.cos(rx)],
        ]
    )
    Ry = np.array(
        [
            [math.cos(ry), 0.0, math.sin(ry)],
            [0.0, 1.0, 0.0],
            [-math.sin(ry), 0.0, math.cos(ry)],
        ]
    )
    Rz = np.array(
        [
            [math.cos(rz), -math.sin(rz), 0.0],
            [math.sin(rz), math.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    d = dict(X=Rx, Y=Ry, Z=Rz)
    R = np.eye(3)
    for ch in order:
        R = d[ch] @ R
    return R


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble 4x4 transform from R (3x3) and t (3,)."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def fmt_array(v) -> str:
    """Pretty numpy one-liner for logs."""
    return np.array2string(np.asarray(v), separator=", ")


# ============================================================================ #
# Depth utils
# ============================================================================ #
def guess_depth_units(depth: np.ndarray, mode: str = "auto") -> str:
    """
    Heuristic for raw depth arrays:
      - if mode is fixed -> return as is
      - otherwise p95 < 10  => 'meters', else 'millimeters'
    """
    if mode in ("meters", "millimeters"):
        return mode
    p95 = float(np.nanpercentile(depth, 95))
    return "meters" if p95 < 10 else "millimeters"


def prefilter_depth(
    depth_m: np.ndarray,
    use_bilateral: bool = True,
    diameter_px: int = 5,
    sigma_color_m: float = 0.03,
    sigma_space_px: float = 3.0,
) -> np.ndarray:
    """
    Bilateral filter on metric depth (meters). Requires OpenCV if enabled.
    Returns unchanged input on failure/unavailable.
    """
    if not use_bilateral:
        return depth_m
    try:
        import cv2  # lazy import

        d = depth_m.copy()
        d[~np.isfinite(d)] = 0.0
        d = cv2.bilateralFilter(d, diameter_px, sigma_color_m, sigma_space_px)
        d = np.where(d < 1e-6, 0.0, d).astype(np.float32)
        return d
    except Exception as e:
        logger.debug(f"[DEPTH] bilateral skipped: {e}")
        return depth_m


# ============================================================================ #
# Log sinks helpers re-exports
# ============================================================================ #
def suppress_o3d_info() -> SuppressO3DInfo:
    """
    Context manager: suppresses noisy console outputs from libs that print to
    stdout/stderr (e.g. Open3D). No dependency on Open3D here.
    """
    return SuppressO3DInfo()


def capture_native_stderr_to_logger():
    """
    Context manager: captures native (C/C++) stderr and funnels it to our logger.
    Useful to silence 3rd-party libs.
    """
    return CaptureStderrToLogger(logger)
