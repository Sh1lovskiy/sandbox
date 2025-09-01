"""2D skeletonization via distance-transform watershed ridges."""
from __future__ import annotations

import cv2
import numpy as np

from utils.logger import Logger

LOG = Logger.get_logger("tk.skel2d")


def skeletonize(img: np.ndarray) -> np.ndarray:
    """Return a 1-px-wide skeleton from a binary raster image.

    The algorithm performs a small morphological close/dilate to connect sparse
    points, computes a distance transform, and extracts watershed ridges which
    approximate the medial axis.
    """
    mask = img.astype(np.uint8)
    LOG.info(f"raster points={int(mask.sum())}")
    if mask.sum() == 0:
        return mask

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, fg = cv2.threshold(dist, 0.4 * float(dist.max()), 255, 0)
    fg = np.uint8(fg)
    unk = cv2.subtract(mask * 255, fg)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unk == 255] = 0
    cv2.watershed(cv2.cvtColor(dist_norm, cv2.COLOR_GRAY2BGR), markers)
    sk = (markers == -1).astype(np.uint8)
    LOG.info(f"skeleton pixels={int(sk.sum())}")
    return sk
