# vision/skeleton/voxelize.py
from __future__ import annotations
import numpy as np
from utils.logger import Logger

LOG = Logger.get_logger("vox")


def compute_voxel_grid_params(
    P: np.ndarray, vsize: float, pad: int, max_voxels: int
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (origin, vsize, dims[x,y,z]) that fits max_voxels."""
    mn = P.min(0)
    mx = P.max(0)
    span = mx - mn
    dims = np.maximum(1, np.ceil(span / vsize).astype(int)) + 1 + 2 * pad
    voxels = int(np.prod(dims))
    if voxels > max_voxels:
        scale = (voxels / max_voxels) ** (1 / 3)
        v_new = float(vsize * scale * 1.1)
        dims = np.maximum(1, np.ceil(span / v_new).astype(int)) + 1 + 2 * pad
        LOG.info(f"resize voxel {vsize:.4f}->{v_new:.4f} to fit {max_voxels}")
        vsize = v_new
    origin = mn - pad * vsize
    dims = np.maximum(1, np.ceil((mx - origin) / vsize).astype(int)) + 1
    return origin.astype(float), float(vsize), dims.astype(int)


def voxelize_points(
    P: np.ndarray, origin: np.ndarray, v: float, dims: np.ndarray
) -> np.ndarray:
    """Binary volume Z,Y,X with ones at occupied voxels."""
    I = np.floor((P - origin) / v).astype(int)
    valid = (
        (I[:, 0] >= 0)
        & (I[:, 1] >= 0)
        & (I[:, 2] >= 0)
        & (I[:, 0] < dims[0])
        & (I[:, 1] < dims[1])
        & (I[:, 2] < dims[2])
    )
    I = I[valid]
    vol = np.zeros((dims[2], dims[1], dims[0]), np.uint8)
    vol[I[:, 2], I[:, 1], I[:, 0]] = 1
    return vol


def dilate_3d(vol: np.ndarray, r: int) -> np.ndarray:
    """Ball dilation by r voxels; returns unchanged on failure."""
    if r <= 0:
        return vol
    try:
        from skimage.morphology import ball, binary_dilation

        return binary_dilation(vol.astype(bool), ball(int(r))).astype(np.uint8)
    except Exception as e:
        LOG.warning(f"dilation skipped: {e}")
        return vol
