# pcd_textio.py
"""
Tiny utility to:
1) Export a PLY point cloud into a compact *text* format (with normals & color),
2) Visualize the gzipped text format back.

I/O:
- Input PLY:  .data_captures/debug/final_merged.ply
- Output TXT: .data_captures/debug/final_merged.xyzrgbn.gz

Line schema (space-separated):
x y z nx ny nz r g b
- x,y,z: meters (quantized; see QUANT_* below)
- nx,ny,nz: normal components (quantized)
- r,g,b: 0-255

Usage:
- Set MODE = "export" to produce the gzipped text file.
- Set MODE = "view"   to visualize the gzipped text file.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d

from utils.logger import Logger
from utils.error_tracker import ErrorTracker

# -------------------------- Configuration ------------------------------------

MODE = "export"  # "export" | "view"

# Paths (project-relative; adjust if needed)
PLY_IN = Path(".data_captures/debug/final_merged.ply")
TXT_GZ = Path(".data_clouds/final_merged_xyzrgbn.gz")

# Quantization (keeps files small while preserving geometry well)
QUANTIZE = True
# Position quantization step (meters). 0.0005 = 0.5 mm
QUANT_POS_M = 0.0005
# Normal quantization step (unitless). 0.01 - 1% step
QUANT_NRM = 0.01

# Downsampling before export to further shrink (set to 0 to disable)
DOWNSAMPLE_VOX_M = 0.0  # e.g., 0.001 (1 mm) for heavy datasets

# Viewer options
WINDOW_TITLE = "xyzrgbn viewer"
POINT_SIZE = 2.0

log = Logger.get_logger("pcd.textio")


# -------------------------- Small helpers ------------------------------------


def _ensure_parents(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _quantize(arr: np.ndarray, step: float) -> np.ndarray:
    """Uniform quantization: round(arr/step)*step. Safe if step>0."""
    if not QUANTIZE or step <= 0:
        return arr
    return np.round(arr / step) * step


def _write_gz_lines(path: Path, lines: np.ndarray) -> None:
    """Write `lines` (array of strings) to a gz file efficiently."""
    _ensure_parents(path)
    with gzip.open(path, "wb") as f:
        data = ("\n".join(lines) + "\n").encode("utf-8")
        f.write(data)


def _read_gz_lines(path: Path) -> io.StringIO:
    """Return a text stream with file content (for fast numpy parsing)."""
    with gzip.open(path, "rb") as f:
        buf = f.read()
    return io.StringIO(buf.decode("utf-8"))


def _ensure_normals(pc: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Compute normals if missing."""
    if pc.has_normals() and len(pc.normals) == len(pc.points):
        return pc
    if len(pc.points) == 0:
        return pc

    bbox = pc.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())
    radius = max(1e-3, 0.01 * diag)
    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=60
        )
    )
    pc.orient_normals_consistent_tangent_plane(50)
    return pc


def _downsample(
    pc: o3d.geometry.PointCloud, vox: float
) -> o3d.geometry.PointCloud:
    if vox and vox > 0:
        return pc.voxel_down_sample(voxel_size=vox)
    return pc


# -------------------------- Export: PLY -> text.gz ----------------------------


def export_ply_to_text_gz(ply_in: Path, txt_gz_out: Path) -> None:
    """
    Convert PLY to gzipped text (xyzrgbn.gz).
    - Ensures normals are present (computes if missing).
    - Applies optional voxel downsample and quantization.
    - Saves as space-separated lines: x y z nx ny nz r g b
    """
    log.info(f"Loading PLY: {ply_in}")
    if not ply_in.exists():
        raise FileNotFoundError(f"Input PLY not found: {ply_in}")

    pc = o3d.io.read_point_cloud(str(ply_in))
    n_pts = len(pc.points)
    log.info(f"Loaded {n_pts} points")

    if n_pts == 0:
        raise RuntimeError("PLY has no points.")

    # Optional downsample to reduce size
    if DOWNSAMPLE_VOX_M and DOWNSAMPLE_VOX_M > 0:
        pc = _downsample(pc, DOWNSAMPLE_VOX_M)
        log.info(f"Downsampled -> {len(pc.points)} pts @ {DOWNSAMPLE_VOX_M} m")

    # Ensure normals
    pc = _ensure_normals(pc)

    P = np.asarray(pc.points, dtype=np.float32)  # (N, 3)
    N = np.asarray(pc.normals, dtype=np.float32)  # (N, 3)
    if pc.has_colors():
        C = np.asarray(pc.colors, dtype=np.float32)
        # Convert to uint8 0-255
        C = np.clip(np.round(C * 255.0), 0, 255).astype(np.uint8)
    else:
        C = np.full((len(P), 3), 200, dtype=np.uint8)

    # Quantize
    Pq = _quantize(P, QUANT_POS_M).astype(np.float32)
    Nq = _quantize(N, QUANT_NRM).astype(np.float32)

    # Format with minimal decimals (6 for pos, 3 for normals) to keep files small
    pos_txt = np.char.add(
        np.char.add(
            np.char.add(np.char.mod("%.6f", Pq[:, 0]), " "),
            np.char.add(np.char.mod("%.6f", Pq[:, 1]), " "),
        ),
        np.char.mod("%.6f", Pq[:, 2]),
    )

    nrm_txt = np.char.add(
        np.char.add(
            np.char.add(np.char.mod(" %.3f", Nq[:, 0]), " "),
            np.char.add(np.char.mod("%.3f", Nq[:, 1]), " "),
        ),
        np.char.mod(" %.3f", Nq[:, 2]),
    )

    col_txt = np.char.add(
        np.char.add(
            np.char.add(np.char.mod(" %d", C[:, 0]), " "),
            np.char.add(np.char.mod("%d", C[:, 1]), " "),
        ),
        np.char.mod("%d", C[:, 2]),
    )

    lines = np.char.add(np.char.add(pos_txt, nrm_txt), col_txt)

    log.info(f"Writing gz text: {txt_gz_out}")
    _write_gz_lines(txt_gz_out, lines)
    log.info("Done.")


# -------------------------- View: text.gz -> viewer ---------------------------


def load_text_gz_to_numpy(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load gz text (xyzrgbn.gz) into arrays.
    Returns:
        P: (N,3) float32 positions (meters)
        N: (N,3) float32 normals (unit)
        C: (N,3) float32 colors (0-1)
    """
    if not path.exists():
        raise FileNotFoundError(f"gz text not found: {path}")

    log.info(f"Loading gz text: {path}")
    stream = _read_gz_lines(path)

    # parse space-separated floats/ints quickly
    # each row: x y z nx ny nz r g b
    data = np.loadtxt(stream)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 9:
        raise ValueError(
            "Unexpected column count; expected 9 columns (x y z nx ny nz r g b)."
        )

    P = data[:, 0:3].astype(np.float32)
    N = data[:, 3:6].astype(np.float32)
    C = (data[:, 6:9].astype(np.float32) / 255.0).clip(0.0, 1.0)
    log.info(f"Loaded {len(P)} points from gz text")
    return P, N, C


def visualize_text_gz(path: Path) -> None:
    """Create an Open3D point cloud from gz text and visualize it."""
    P, N, C = load_text_gz_to_numpy(path)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.normals = o3d.utility.Vector3dVector(N)
    pc.colors = o3d.utility.Vector3dVector(C)

    # simple viewer window
    log.info("Opening viewer")
    o3d.visualization.draw_geometries(
        [pc],
        window_name=WINDOW_TITLE,
        point_show_normal=False,
        width=1280,
        height=800,
        left=100,
        top=80,
    )
    log.info("Closed.")


# ------------------------------- Entrypoint -----------------------------------


def main() -> None:
    """
    Set MODE to "export" to convert PLY -> gz text.
    Set MODE to "view"   to display gz text.
    """
    ErrorTracker.install_excepthook()
    if MODE == "export":
        export_ply_to_text_gz(PLY_IN, TXT_GZ)
    elif MODE == "view":
        visualize_text_gz(TXT_GZ)
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Fatal error: {e}")
        raise
