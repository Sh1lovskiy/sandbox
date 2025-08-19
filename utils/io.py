# utils/io.py
import json
from logging import Logger
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from utils.config import CameraDefaults, Pose
from utils.logger import Logger

logger = Logger.get_logger("io")
np.set_printoptions(suppress=True, precision=6, linewidth=180)


# ============================================================================ #
# I/O: poses, intrinsics (no Open3D logic here)
# ============================================================================ #
def load_poses(path: Path) -> Dict[str, Pose]:
    """
    Load poses.json -> dict[stem] = Pose.
    Returns empty dict if file not found.
    """
    if not path.exists():
        logger.warning(f"[POSES] not found: {path}")
        return {}
    data = json.loads(path.read_text())
    out: Dict[str, Pose] = {}
    for k, v in data.items():
        out[k] = Pose(
            float(v["x"]),
            float(v["y"]),
            float(v["z"]),
            float(v["rx"]),
            float(v["ry"]),
            float(v["rz"]),
        )
    logger.info(f"[POSES] loaded {len(out)} entries from {path.name}")
    return out


def load_intrinsics_from_json(
    root: Path, fallback: "CameraDefaults", name: str = "rs2_params.json"
) -> Tuple[int, int, float, float, float, float]:
    """
    Read intrinsics JSON next to capture root.
    Returns (w, h, fx, fy, cx, cy) for RGB (color) stream.
    Falls back to provided defaults if missing/bad.
    """

    p = root / name
    if p.exists():
        try:
            y = json.loads(p.read_text())
            color = y.get("intrinsics", {}).get("color", {})

            w = int(color.get("width", fallback.width))
            h = int(color.get("height", fallback.height))
            fx = float(color.get("fx", fallback.fx))
            fy = float(color.get("fy", fallback.fy))
            cx = float(color.get("ppx", fallback.cx))
            cy = float(color.get("ppy", fallback.cy))

            logger.info(
                f"[INTR JSON] {p.name}: w={w} h={h} fx={fx:.3f} fy={fy:.3f} "
                f"cx={cx:.3f} cy={cy:.3f}"
            )
            return (w, h, fx, fy, cx, cy)

        except Exception as e:
            logger.warning(f"[INTR JSON] parse failed: {e}")

    return (
        fallback.width,
        fallback.height,
        fallback.fx,
        fallback.fy,
        fallback.cx,
        fallback.cy,
    )
