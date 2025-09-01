# unwrap_viewer_truss.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Unwrapping + real 2D skeletonization for truss-like structures.

Keys:
  1 - original (cloud/mesh)
  2 - Cyl-4 projection + 2D contours (orange) + 2D skeleton (black)
  3 - Cube 3x2 projection (per-face colors, no overlap)
  4 - Sin projection (adaptive seam)
  5 - Hybrid (axial->Cyl-4, else->Cube)
  R - toggle source: point cloud ↔ Poisson mesh (with Taubin smoothing)
  B - toggle tile borders (2/3/5)
  S - save current projection (.ply)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import logging
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# --------------------- config ---------------------

PATH = Path(".data_captures/first/debug/final_merged.ply")


@dataclass(frozen=True)
class Cfg:
    epsilon: float = 1e-9
    gap: float = 0.08  # tile gap (relative to tile size)
    axial_deg: float = 60.0  # hybrid threshold to main axis
    face_tint: float = 0.85
    draw_borders: bool = True
    # cloud preprocess
    voxel: float = 0.0
    nb_rad: float = 0.006
    nb_min: int = 8
    # mesh build
    pois_depth: int = 9
    pois_scale: float = 1.1
    pois_linear_fit: bool = True
    pois_trim_q: float = 0.25
    taubin_iters: int = 3
    taubin_lambda: float = 0.5
    taubin_mu: float = -0.53
    sample_ratio: float = 1.0
    # 2D raster/skeleton
    target_px: int = 1200  # approx longest side in pixels
    min_branch_px: int = 8  # discard tiny skeleton branches
    # --- NEW: pre-skeletonization image processing ---
    blur_sigma_px: float = 1.2  # Gaussian sigma (px), 0=off
    open_px: int = 1  # morphological opening radius (px), 0=off
    close_px: int = 1  # morphological closing radius (px), 0=off


# --------------------- logging ---------------------


def _log() -> logging.Logger:
    log = logging.getLogger("unwrap")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | sin: %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


LOG = _log()


# --------------------- utils ---------------------

_FACE_COL = {
    "+X": np.array([0.95, 0.20, 0.20]),
    "-X": np.array([0.55, 0.10, 0.10]),
    "+Y": np.array([0.20, 0.95, 0.20]),
    "-Y": np.array([0.10, 0.55, 0.10]),
    "+Z": np.array([0.20, 0.40, 0.95]),
    "-Z": np.array([0.10, 0.20, 0.55]),
    "+U": np.array([0.90, 0.30, 0.30]),
    "-U": np.array([0.60, 0.15, 0.15]),
    "+V": np.array([0.30, 0.90, 0.30]),
    "-V": np.array([0.15, 0.60, 0.15]),
}


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    if not path.exists():
        raise FileNotFoundError(f"No file: {path}")
    pc = o3d.io.read_point_cloud(str(path))
    LOG.info(f"Loaded {path} | points={len(pc.points)}")
    return pc


def ensure_colors(pc: o3d.geometry.PointCloud) -> np.ndarray:
    if len(pc.colors):
        return np.asarray(pc.colors)
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return np.zeros((0, 3))
    z = pts[:, 2]
    t = (z - float(z.min())) / (float(z.ptp()) + 1e-9)
    col = np.stack([t, t, t], axis=1)
    pc.colors = o3d.utility.Vector3dVector(col)
    return col


def pca_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = points.mean(axis=0)
    X = points - c
    H = X.T @ X / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(H)
    R = V[:, np.argsort(w)[::-1]]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return c, R


def coord_frame_for(ref: o3d.geometry.Geometry) -> o3d.geometry.TriangleMesh:
    bbox = ref.get_axis_aligned_bounding_box()
    extent = float(np.linalg.norm(bbox.get_extent()))
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1 * extent)


def mix_face_colors(
    orig: np.ndarray, faces: List[str], tint: float
) -> np.ndarray:
    if len(orig) == 0:
        return orig
    fc = np.stack([_FACE_COL[f] for f in faces], axis=0)
    return (1.0 - tint) * orig + tint * fc


# -------- cloud preprocess & poisson mesh --------


def preprocess_cloud(
    pc: o3d.geometry.PointCloud, cfg: Cfg
) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud(pc)
    if cfg.voxel > 0:
        p = p.voxel_down_sample(cfg.voxel)
    if cfg.nb_rad > 0 and cfg.nb_min > 0:
        p, _ = p.remove_radius_outlier(nb_points=cfg.nb_min, radius=cfg.nb_rad)
    return p


def _estimate_orient_normals(
    pc: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud(pc)
    p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    p.orient_normals_consistent_tangent_plane(50)
    return p


def build_poisson_mesh(
    pc: o3d.geometry.PointCloud, cfg: Cfg
) -> o3d.geometry.TriangleMesh:
    p = preprocess_cloud(pc, cfg)
    p = _estimate_orient_normals(p)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        p,
        depth=cfg.pois_depth,
        scale=cfg.pois_scale,
        linear_fit=cfg.pois_linear_fit,
    )
    dens = np.asarray(dens)
    thr = float(np.quantile(dens, cfg.pois_trim_q))
    keep = dens >= thr
    mesh = mesh.select_by_index(np.where(keep)[0])
    mesh.remove_unreferenced_vertices()
    if cfg.taubin_iters > 0:
        mesh = mesh.filter_smooth_taubin(
            cfg.taubin_iters, cfg.taubin_lambda, cfg.taubin_mu
        )
    mesh.compute_vertex_normals()
    LOG.info(
        f"Mesh: Poisson depth={cfg.pois_depth}, verts={len(mesh.vertices)}, "
        f"tris={len(mesh.triangles)}"
    )
    return mesh


def sample_points_from_mesh(
    mesh: o3d.geometry.TriangleMesh, n_target: int
) -> o3d.geometry.PointCloud:
    n = max(int(n_target), 2000)
    pc = mesh.sample_points_uniformly(number_of_points=n)
    return pc


# --------------------- projections ---------------------


def proj_cyl4(points: np.ndarray, cfg: Cfg) -> Tuple[np.ndarray, Dict]:
    c, R = pca_frame(points)
    X = (points - c) @ R
    s = X[:, 0]
    u = X[:, 1]
    v = X[:, 2]
    theta = np.arctan2(v, u)
    r = np.hypot(u, v)
    R0 = np.quantile(r, 0.95)

    centers = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
    names = ["+U", "+V", "-U", "-V"]
    dist = np.abs(
        ((theta[:, None] - centers[None, :] + np.pi) % (2 * np.pi)) - np.pi
    )
    sec_idx = np.argmin(dist, axis=1)
    faces = [names[i] for i in sec_idx]

    theta_local = np.take(centers, sec_idx)
    dtheta = ((theta - theta_local + np.pi) % (2 * np.pi)) - np.pi
    width = np.pi / 4
    dtheta = np.clip(dtheta, -width, width)

    x2 = s
    y2 = R0 * dtheta
    P = np.column_stack((x2, y2, np.zeros_like(x2)))

    smin, smax = float(s.min()), float(s.max())
    base = max(smax - smin, 1e-6)
    gap = cfg.gap * base
    origins = {
        "+U": (0.0, 0.0),
        "+V": (base + gap, 0.0),
        "-U": (0.0, -(base + gap)),
        "-V": (base + gap, -(base + gap)),
    }
    sn = (s - (smin + smax) * 0.5) / max(base, 1e-6)
    P[:, 0] = sn * base
    for i, f in enumerate(faces):
        ox, oy = origins[f]
        P[i, 0] += ox
        P[i, 1] += oy

    LOG.info("Cyl-4: 2x2 tiles with sector separation")
    return P, {"c": c, "R": R, "X": X, "faces": faces, "base": base, "gap": gap}


def proj_sinusoidal(points: np.ndarray, cfg: Cfg) -> Tuple[np.ndarray, Dict]:
    c, R = pca_frame(points)
    X = (points - c) @ R
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.hypot(x, y) + cfg.epsilon)
    hist, edges = np.histogram(lon, bins=90, range=(-np.pi, np.pi))
    i_min = int(np.argmin(hist))
    seam = float(0.5 * (edges[i_min] + edges[i_min + 1]))
    lon_s = (lon - seam + np.pi) % (2 * np.pi) - np.pi
    xm = lon_s * np.cos(lat)
    ym = lat
    r_src = np.linalg.norm(points - c, axis=1)
    r_map = np.hypot(xm, ym)
    scale = np.quantile(r_src, 0.95) / max(
        np.quantile(r_map, 0.95), cfg.epsilon
    )
    P = np.column_stack((xm * scale, ym * scale, np.zeros_like(xm)))
    LOG.info(f"Sin: seam={seam:.3f}, scale={scale:.4f}")
    return P, {"c": c, "R": R, "X": X, "seam": seam, "scale": float(scale)}


def _cube_frames(
    R: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
    return {
        "+X": (ex, ey, ez),
        "-X": (-ex, ey, -ez),
        "+Y": (ey, ez, ex),
        "-Y": (-ey, ez, -ex),
        "+Z": (ez, ex, ey),
        "-Z": (-ez, ex, -ey),
    }


def _assign_cube_faces(X: np.ndarray, R: np.ndarray, eps: float) -> List[str]:
    ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
    D = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    proj = np.column_stack((np.abs(D @ ex), np.abs(D @ ey), np.abs(D @ ez)))
    idx = np.argmax(proj, axis=1)
    sgnx, sgny, sgnz = np.sign(D @ ex), np.sign(D @ ey), np.sign(D @ ez)
    faces: List[str] = []
    for i in range(len(X)):
        if idx[i] == 0:
            faces.append("+X" if sgnx[i] >= 0 else "-X")
        elif idx[i] == 1:
            faces.append("+Y" if sgny[i] >= 0 else "-Y")
        else:
            faces.append("+Z" if sgnz[i] >= 0 else "-Z")
    return faces


def proj_cube(points: np.ndarray, cfg: Cfg) -> Tuple[np.ndarray, Dict]:
    c, R = pca_frame(points)
    X = (points - c) @ R
    frames = _cube_frames(R)
    faces = _assign_cube_faces(X, R, cfg.epsilon)

    uv: Dict[str, List[Tuple[float, float]]] = {k: [] for k in frames}
    by_face: Dict[str, List[int]] = {k: [] for k in frames}
    for i, f in enumerate(faces):
        n, uax, vax = frames[f]
        xi = X[i]
        uv[f].append((float(xi @ uax), float(xi @ vax)))
        by_face[f].append(i)

    bounds = {}
    for f, ptsf in uv.items():
        if ptsf:
            U = np.array(ptsf)
            umin, vmin = U.min(axis=0)
            umax, vmax = U.max(axis=0)
        else:
            umin = vmin = umax = vmax = 0.0
        bounds[f] = (umin, vmin, umax, vmax)

    sizes = [max(b[2] - b[0], b[3] - b[1]) for b in bounds.values()]
    base = float(np.median(sizes)) if sizes else 1.0
    gap = cfg.gap * base

    layout = {
        (0, 0): "+X",
        (0, 1): "-X",
        (0, 2): "+Y",
        (1, 0): "-Y",
        (1, 1): "+Z",
        (1, 2): "-Z",
    }
    origins: Dict[str, Tuple[float, float]] = {}
    for r in (0, 1):
        for ccol in (0, 1, 2):
            f = layout[(r, ccol)]
            origins[f] = (ccol * (base + gap), -r * (base + gap))

    P = np.zeros((len(points), 3), dtype=np.float64)
    for f, idxs in by_face.items():
        if not idxs:
            continue
        umin, vmin, umax, vmax = bounds[f]
        w, h = max(umax - umin, 1e-9), max(vmax - vmin, 1e-9)
        scale = base / max(w, h)
        U = np.array([uv[f][k] for k in range(len(idxs))])
        Uc = (U - np.array([(umin + umax) * 0.5, (vmin + vmax) * 0.5])) * scale
        ox, oy = origins[f]
        P[idxs, 0] = Uc[:, 0] + ox
        P[idxs, 1] = Uc[:, 1] + oy

    LOG.info("Cube: 3x2 tiling + per-face scaling")
    return P, {
        "c": c,
        "R": R,
        "X": X,
        "faces": faces,
        "origins": origins,
        "base": base,
        "gap": gap,
    }


def proj_hybrid(points: np.ndarray, cfg: Cfg) -> Tuple[np.ndarray, Dict]:
    P_cyl, m_cyl = proj_cyl4(points, cfg)
    P_cub, m_cub = proj_cube(points, cfg)
    X = m_cyl["X"]
    norm = np.linalg.norm(X, axis=1) + cfg.epsilon
    cos_to_axis = np.abs(X[:, 0]) / norm
    mask_ax = (cos_to_axis >= np.cos(np.deg2rad(cfg.axial_deg))).astype(
        np.float64
    )
    P = np.zeros_like(P_cyl)
    P[:, :2] = (
        mask_ax[:, None] * P_cyl[:, :2]
        + (1.0 - mask_ax)[:, None] * P_cub[:, :2]
    )
    LOG.info(f"Hybrid: axial≤{cfg.axial_deg:.0f}°->Cyl-4, else->Cube")
    return P, {
        "faces": m_cub["faces"],
        "mask_ax": mask_ax,
        "cyl": m_cyl,
        "cube": m_cub,
    }


# --------------------- tile borders ---------------------


def _rect_lines(
    ox: float, oy: float, w: float, h: float
) -> o3d.geometry.LineSet:
    pts = np.array(
        [
            [ox - w / 2, oy - h / 2, 0.0],
            [ox + w / 2, oy - h / 2, 0.0],
            [ox + w / 2, oy + h / 2, 0.0],
            [ox - w / 2, oy + h / 2, 0.0],
        ],
        dtype=np.float64,
    )
    idx = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(idx),
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile([[0, 0, 0]], (4, 1)))
    return ls


def _tile_rects_for_cyl4(meta: Dict, cfg: Cfg) -> List[o3d.geometry.LineSet]:
    base, gap = meta["base"], meta["gap"]
    origins = {
        "+U": (0.0, 0.0),
        "+V": (base + gap, 0.0),
        "-U": (0.0, -(base + gap)),
        "-V": (base + gap, -(base + gap)),
    }
    return [_rect_lines(ox, oy, base, base) for ox, oy in origins.values()]


def _tile_rects_for_cube(meta: Dict) -> List[o3d.geometry.LineSet]:
    base, gap = meta["base"], meta["gap"]
    rects = []
    for r in (0, 1):
        for ccol in (0, 1, 2):
            ox = ccol * (base + gap)
            oy = -r * (base + gap)
            rects.append(_rect_lines(ox, oy, base, base))
    return rects


# --------------------- image ops: blur/open/close + raster/contours/skeleton ---------------------


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    r = max(1, int(round(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k /= k.sum() + 1e-12
    return k


def _gaussian_blur2d(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    k = _gaussian_kernel1d(sigma)
    pad = len(k) // 2
    # horizontal
    P = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(img)
    for i in range(img.shape[1]):
        out[:, i] = np.sum(P[:, i : i + len(k)] * k[None, :], axis=1)
    # vertical
    P = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
    out2 = np.zeros_like(out)
    for j in range(img.shape[0]):
        out2[j, :] = np.sum(P[j : j + len(k), :] * k[:, None], axis=0)
    return out2


def _morpho_binary(bin_img: np.ndarray, rad: int, op: str) -> np.ndarray:
    """Binary morphology on {0,255} using square structuring element (2*rad+1)."""
    if rad <= 0:
        return bin_img
    from numpy.lib.stride_tricks import sliding_window_view

    B = (bin_img > 0).astype(np.uint8)
    pad = ((rad, rad), (rad, rad))
    P = np.pad(B, pad, mode="constant", constant_values=0)

    win = sliding_window_view(P, (2 * rad + 1, 2 * rad + 1))
    if op == "dilate":
        res = (win.max(axis=(2, 3)) > 0).astype(np.uint8)
    elif op == "erode":
        res = (win.min(axis=(2, 3)) > 0).astype(np.uint8)
    else:
        raise ValueError("op must be 'dilate' or 'erode'")
    return (res * 255).astype(np.uint8)


def _opening(bin_img: np.ndarray, rad: int) -> np.ndarray:
    if rad <= 0:
        return bin_img
    return _morpho_binary(_morpho_binary(bin_img, rad, "erode"), rad, "dilate")


def _closing(bin_img: np.ndarray, rad: int) -> np.ndarray:
    if rad <= 0:
        return bin_img
    return _morpho_binary(_morpho_binary(bin_img, rad, "dilate"), rad, "erode")


def _otsu_threshold(img01: np.ndarray) -> float:
    hist, _ = np.histogram(img01, bins=256, range=(0.0, 1.0))
    prob = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-9)
    t_idx = int(np.nanargmax(sigma_b))
    return (t_idx + 0.5) / 256.0


def _estimate_mean_spacing(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return 1.0
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=min(8, len(xy)))
    # ignore self-dist (0) and take 10% quantile of neighbors
    vals = d[:, 1:].ravel()
    return float(np.quantile(vals, 0.10))


def _rasterize_points(
    P2: np.ndarray, cfg: Cfg
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """
    Density splatting (KDE-like): each 2D point contributes a Gaussian kernel.
    Returns binary image (uint8), origin (xmin,ymin), pixel_size.
    """
    if len(P2) == 0:
        return np.zeros((1, 1), np.uint8), (0.0, 0.0), 1.0

    xy = P2[:, :2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    w = float(xmax - xmin)
    h = float(ymax - ymin)
    if w <= 0 or h <= 0:
        return np.zeros((1, 1), np.uint8), (xmin, ymin), 1.0

    # pixel size so longest side ≈ target_px
    ps = max(w, h) / float(max(cfg.target_px, 64))
    nx = max(int(np.ceil(w / ps)) + 3, 64)
    ny = max(int(np.ceil(h / ps)) + 3, 64)

    # grid coords
    u = ((xy[:, 0] - xmin) / ps).astype(np.float32) + 1.0
    v = ((xy[:, 1] - ymin) / ps).astype(np.float32) + 1.0
    u = np.clip(u, 0, nx - 1)
    v = np.clip(v, 0, ny - 1)

    # estimate mean spacing in pixels (robust)
    mean_step_world = _estimate_mean_spacing(xy)
    sigma_px = max(mean_step_world / ps * 0.8, 0.8)

    # Gaussian splatting
    rad = int(max(2, round(3.0 * sigma_px)))
    img = np.zeros((ny, nx), dtype=np.float32)
    xs = np.arange(-rad, rad + 1)
    ys = np.arange(-rad, rad + 1)
    XX, YY = np.meshgrid(xs, ys)
    G = np.exp(-(XX**2 + YY**2) / (2.0 * sigma_px**2))

    for ui, vi in zip(u, v):
        cx, cy = int(round(ui)), int(round(vi))
        x0 = max(0, cx - rad)
        x1 = min(nx - 1, cx + rad)
        y0 = max(0, cy - rad)
        y1 = min(ny - 1, cy + rad)
        gx0 = x0 - (cx - rad)
        gx1 = gx0 + (x1 - x0)
        gy0 = y0 - (cy - rad)
        gy1 = gy0 + (y1 - y0)
        img[y0 : y1 + 1, x0 : x1 + 1] += G[gy0 : gy1 + 1, gx0 : gx1 + 1]

    # normalize to 0..1
    img = img / (img.max() + 1e-9)

    # --- NEW: Gaussian blur BEFORE threshold (smooth ragged densities) ---
    if cfg.blur_sigma_px > 0:
        img = _gaussian_blur2d(img, cfg.blur_sigma_px)

    # Otsu threshold -> binary
    thr = _otsu_threshold(img)
    bin_img = (img >= thr).astype(np.uint8) * 255

    # --- NEW: Opening then Closing to seal gaps and remove speckles ---
    if cfg.open_px > 0:
        bin_img = _opening(bin_img, cfg.open_px)
    if cfg.close_px > 0:
        bin_img = _closing(bin_img, cfg.close_px)

    return bin_img, (xmin, ymin), ps


def _binary_contour_lines(
    bin_img: np.ndarray, origin: Tuple[float, float], ps: float
) -> Optional[o3d.geometry.LineSet]:
    """
    Build simple contour polylines from binary image:
    - boundary pixels = foreground with at least one 4-neighbor background
    - connect boundary pixels with 8-neighborhood edges.
    """
    B = (bin_img > 0).astype(np.uint8)
    ny, nx = B.shape
    N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    N8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    boundary = np.zeros_like(B, dtype=bool)
    for y in range(ny):
        for x in range(nx):
            if B[y, x] == 0:
                continue
            for dy, dx in N4:
                yy, xx = y + dy, x + dx
                if yy < 0 or yy >= ny or xx < 0 or xx >= nx or B[yy, xx] == 0:
                    boundary[y, x] = True
                    break

    # map boundary pixels to indices
    idx_map = -np.ones_like(B, dtype=np.int32)
    coords = np.argwhere(boundary)
    pts = []
    xmin, ymin = origin
    for i, (y, x) in enumerate(coords):
        idx_map[y, x] = i
        pts.append([xmin + x * ps, ymin + y * ps, 0.0])

    lines = []
    for y, x in coords:
        i0 = idx_map[y, x]
        for dy, dx in N8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and boundary[yy, xx]:
                i1 = idx_map[yy, xx]
                if i1 >= 0 and i1 > i0:
                    lines.append([i0, i1])

    if not pts or not lines:
        return None
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(pts, float)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, np.int32)),
    )
    ls.paint_uniform_color([1.0, 0.5, 0.0])  # orange contours
    return ls


def _zhang_suen_thinning(img: np.ndarray) -> np.ndarray:
    """Zhang–Suen thinning on {0,255}."""
    I = (img > 0).astype(np.uint8)
    changed = True
    ny, nx = I.shape
    P = np.pad(I, ((1, 1), (1, 1)), mode="constant")

    def neighbors(y, x):
        p2 = P[y - 1, x]
        p3 = P[y - 1, x + 1]
        p4 = P[y, x + 1]
        p5 = P[y + 1, x + 1]
        p6 = P[y + 1, x]
        p7 = P[y + 1, x - 1]
        p8 = P[y, x - 1]
        p9 = P[y - 1, x - 1]
        return [p2, p3, p4, p5, p6, p7, p8, p9]

    while changed:
        changed = False
        to_del = []
        for y in range(1, ny + 1):
            for x in range(1, nx + 1):
                if P[y, x] == 0:
                    continue
                nb = neighbors(y, x)
                A = sum((nb[i] == 0 and nb[(i + 1) % 8] == 1) for i in range(8))
                B = sum(nb)
                if not (2 <= B <= 6 and A == 1):
                    continue
                if nb[0] * nb[2] * nb[4] != 0:
                    continue
                if nb[2] * nb[4] * nb[6] != 0:
                    continue
                to_del.append((y, x))
        if to_del:
            changed = True
            for y, x in to_del:
                P[y, x] = 0
        to_del = []
        for y in range(1, ny + 1):
            for x in range(1, nx + 1):
                if P[y, x] == 0:
                    continue
                nb = neighbors(y, x)
                A = sum((nb[i] == 0 and nb[(i + 1) % 8] == 1) for i in range(8))
                B = sum(nb)
                if not (2 <= B <= 6 and A == 1):
                    continue
                if nb[0] * nb[2] * nb[6] != 0:
                    continue
                if nb[0] * nb[4] * nb[6] != 0:
                    continue
                to_del.append((y, x))
        if to_del:
            changed = True
            for y, x in to_del:
                P[y, x] = 0

    S = P[1:-1, 1:-1].astype(np.uint8) * 255
    return S


def _skeleton_to_polylines(
    S: np.ndarray, origin: Tuple[float, float], ps: float, min_len_px: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorize 8-neighborhood skeleton to polylines.
    Returns (points3D, lines) where points lie in Z=0 projection space.
    """
    sk = (S > 0).astype(np.uint8)
    ny, nx = sk.shape
    offs = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]

    # degree map
    deg = np.zeros_like(sk, dtype=np.uint8)
    coords = np.argwhere(sk)
    for y, x in coords:
        nb = 0
        for dy, dx in offs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and sk[yy, xx]:
                nb += 1
        deg[y, x] = nb
    is_node = sk & (deg != 2)
    visited = np.zeros_like(sk, dtype=bool)

    pts: List[Tuple[float, float, float]] = []
    lines: List[Tuple[int, int]] = []
    xmin, ymin = origin

    def px_to_xyz(y: int, x: int) -> Tuple[float, float, float]:
        return (xmin + x * ps, ymin + y * ps, 0.0)

    for y0, x0 in np.argwhere(is_node):
        for dy, dx in offs:
            y1, x1 = y0 + dy, x0 + dx
            if (
                not (0 <= y1 < ny and 0 <= x1 < nx)
                or not sk[y1, x1]
                or visited[y1, x1]
            ):
                continue
            path = [(y0, x0)]
            y, x = y1, x1
            prev = (y0, x0)
            while True:
                path.append((y, x))
                visited[y, x] = True
                if is_node[y, x]:
                    break
                nxt = None
                for dyy, dxx in offs:
                    yy, xx = y + dyy, x + dxx
                    if (
                        0 <= yy < ny
                        and 0 <= xx < nx
                        and sk[yy, xx]
                        and (yy, xx) != prev
                    ):
                        nxt = (yy, xx)
                        break
                if nxt is None:
                    break
                prev = (y, x)
                y, x = nxt

            if len(path) < max(min_len_px, 2):
                continue
            start_idx = len(pts)
            for yy, xx in path:
                pts.append(px_to_xyz(yy, xx))
            for i in range(start_idx, len(pts) - 1):
                lines.append((i, i + 1))

    if not pts:
        return np.zeros((0, 3), float), np.zeros((0, 2), int)

    P = np.array(pts, dtype=float)
    L = np.array(lines, dtype=int)

    # remove single tiny segments
    if len(L):
        seg_len = np.linalg.norm(P[L[:, 1], :2] - P[L[:, 0], :2], axis=1)
        keep = seg_len >= min_len_px
        L = L[keep]

    return P, L


# --------------------- build & viewer ---------------------


def build_projection(pc_src: o3d.geometry.PointCloud, method: str, cfg: Cfg):
    pts = np.asarray(pc_src.points)
    skeleton_ls = None
    contours_ls = None

    if method == "cyl":
        P, meta = proj_cyl4(pts, cfg)
        faces = meta["faces"]
        borders = _tile_rects_for_cyl4(meta, cfg) if cfg.draw_borders else []
        # --- 2D raster -> blur/open/close -> contours + skeleton ---
        bin_img, origin, ps = _rasterize_points(P, cfg)
        cset = _binary_contour_lines(bin_img, origin, ps)
        if cset is not None:
            contours_ls = cset
        Sk = _zhang_suen_thinning(bin_img)
        sk_pts, sk_lines = _skeleton_to_polylines(
            Sk, origin, ps, cfg.min_branch_px
        )
        if len(sk_pts) and len(sk_lines):
            skeleton_ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(sk_pts),
                lines=o3d.utility.Vector2iVector(sk_lines),
            )
            skeleton_ls.paint_uniform_color([0, 0, 0])
    elif method == "cube":
        P, meta = proj_cube(pts, cfg)
        faces = meta["faces"]
        borders = _tile_rects_for_cube(meta) if cfg.draw_borders else []
    elif method == "sin":
        P, meta = proj_sinusoidal(pts, cfg)
        faces = None
        borders = []
    elif method == "hyb":
        P, meta = proj_hybrid(pts, cfg)
        faces = meta["faces"]
        borders = _tile_rects_for_cube(meta["cube"]) if cfg.draw_borders else []
    else:
        raise ValueError(f"Unknown method: {method}")

    pc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    base_colors = np.asarray(pc_src.colors)
    if faces is not None:
        pc2.colors = o3d.utility.Vector3dVector(
            mix_face_colors(base_colors, faces, cfg.face_tint)
        )
    else:
        pc2.colors = o3d.utility.Vector3dVector(base_colors)
    return pc2, meta, faces, borders, skeleton_ls, contours_ls


def recolor_original(
    pc: o3d.geometry.PointCloud, faces: List[str], tint: float
):
    out = o3d.geometry.PointCloud(pc)
    base = np.asarray(pc.colors)
    out.colors = o3d.utility.Vector3dVector(mix_face_colors(base, faces, tint))
    return out


class Viewer:
    def __init__(self, pc: o3d.geometry.PointCloud, cfg: Cfg):
        self.cfg = cfg

        # source state
        self.source_mode = "cloud"  # "cloud" | "mesh"
        self.cloud_raw = pc
        ensure_colors(self.cloud_raw)
        self.cloud_proc = preprocess_cloud(self.cloud_raw, cfg)
        self.mesh = None
        self.cloud_from_mesh = None

        # view/projection state
        self.method = "cyl"
        self.mode = 1
        self.pc_src = self.cloud_proc
        (
            self.pc_proj,
            self.meta,
            self.faces,
            self.borders,
            self.skel,
            self.contours,
        ) = build_projection(self.pc_src, self.method, self.cfg)

        self.show_borders = cfg.draw_borders
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

    def _ensure_mesh(self):
        if self.mesh is None:
            LOG.info("Building Poisson mesh…")
            self.mesh = build_poisson_mesh(self.cloud_proc, self.cfg)
        if self.cloud_from_mesh is None:
            n_target = int(len(self.cloud_proc.points) * self.cfg.sample_ratio)
            self.cloud_from_mesh = sample_points_from_mesh(self.mesh, n_target)
            ensure_colors(self.cloud_from_mesh)

    def _set_source(self, mode: str):
        if mode == "cloud":
            self.source_mode = "cloud"
            self.pc_src = self.cloud_proc
            LOG.info("Source: cloud")
        else:
            self._ensure_mesh()
            self.source_mode = "mesh"
            self.pc_src = self.cloud_from_mesh
            LOG.info("Source: mesh (sampled)")
        self._rebuild()

    def _rebuild(self):
        (
            self.pc_proj,
            self.meta,
            self.faces,
            self.borders,
            self.skel,
            self.contours,
        ) = build_projection(self.pc_src, self.method, self.cfg)
        self._update()

    def _update(self):
        self.vis.clear_geometries()
        if self.mode == 1:
            if self.method in ("cube", "hyb") and self.faces is not None:
                shown = recolor_original(
                    self.pc_src, self.faces, self.cfg.face_tint
                )
            else:
                shown = self.pc_src
            self.vis.add_geometry(shown, reset_bounding_box=True)
            LOG.info(f"View 1: original [{self.source_mode}]")
        else:
            self.vis.add_geometry(self.pc_proj, reset_bounding_box=True)
            self.vis.add_geometry(coord_frame_for(self.pc_proj))
            if self.show_borders and self.borders:
                for ls in self.borders:
                    self.vis.add_geometry(ls)
            if self.method == "cyl":
                if (
                    self.contours is not None
                    and len(self.contours.points) > 0
                    and len(self.contours.lines) > 0
                ):
                    self.vis.add_geometry(self.contours)
                if (
                    self.skel is not None
                    and len(self.skel.points) > 0
                    and len(self.skel.lines) > 0
                ):
                    self.vis.add_geometry(self.skel)
            LOG.info(
                f"View 2: projection [{self.method}] from [{self.source_mode}]"
            )

    def run(self):
        self.vis.create_window("Unwrap Viewer - Truss", width=1280, height=800)

        def k1(_):
            self.mode = 1
            self._update()
            return False

        def k2(_):
            self.mode = 2
            self.method = "cyl"
            self._rebuild()
            return False

        def k3(_):
            self.mode = 2
            self.method = "cube"
            self._rebuild()
            return False

        def k4(_):
            self.mode = 2
            self.method = "sin"
            self._rebuild()
            return False

        def k5(_):
            self.mode = 2
            self.method = "hyb"
            self._rebuild()
            return False

        def kR(_):
            new = "mesh" if self.source_mode == "cloud" else "cloud"
            self._set_source(new)
            return False

        def kB(_):
            self.show_borders = not self.show_borders
            LOG.info(f"Tile borders: {'ON' if self.show_borders else 'OFF'}")
            self._update()
            return False

        def kS(_):
            stem = PATH.with_suffix("")
            out = (
                stem.parent
                / f"{stem.name}_proj_{self.method}_{self.source_mode}.ply"
            )
            o3d.io.write_point_cloud(str(out), self.pc_proj)
            LOG.info(f"Saved {out}")
            return False

        self.vis.register_key_callback(ord("1"), k1)
        self.vis.register_key_callback(ord("2"), k2)
        self.vis.register_key_callback(ord("3"), k3)
        self.vis.register_key_callback(ord("4"), k4)
        self.vis.register_key_callback(ord("5"), k5)
        self.vis.register_key_callback(ord("R"), kR)
        self.vis.register_key_callback(ord("B"), kB)
        self.vis.register_key_callback(ord("S"), kS)

        self._update()
        self.vis.run()
        self.vis.destroy_window()


# --------------------- main ---------------------


def main():
    cfg = Cfg()
    pc = load_cloud(PATH)
    if len(pc.points) == 0:
        raise ValueError("Empty point cloud")
    Viewer(pc, cfg).run()


if __name__ == "__main__":
    main()
