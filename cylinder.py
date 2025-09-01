# -*- coding: utf-8 -*-
"""
skeletonization_final_corrected.py

Corrected Laplacian-based contraction that properly converges to center lines.
Uses PCA normal direction (smallest variance) for contraction, not main axis.
Based on the paper: "Skeletonization Quality Evaluation: Geometric Metrics for Point Cloud Analysis in Robotics"
"""

from __future__ import annotations
import sys
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import time
from pathlib import Path


def generate_two_cylinders(
    n_main: int = 800,
    n_branch: int = 600,
    main_height: float = 8.0,
    main_radius: float = 1.5,
    branch_height: float = 5.0,
    branch_radius: float = 1.2,
    angle_deg: float = 45.0,
) -> np.ndarray:
    """Generate two connected cylinders: main vertical + tilted branch."""
    rng = np.random.default_rng(42)
    z_main = rng.uniform(0, main_height, n_main)
    theta_main = rng.uniform(0, 2 * np.pi, n_main)
    x_main = main_radius * np.cos(theta_main)
    y_main = main_radius * np.sin(theta_main)
    main = np.column_stack([x_main, y_main, z_main])

    z_branch = rng.uniform(0, branch_height, n_branch)
    theta_branch = rng.uniform(0, 2 * np.pi, n_branch)
    x_branch = branch_radius * np.cos(theta_branch)
    y_branch = branch_radius * np.sin(theta_branch)
    branch = np.column_stack([x_branch, y_branch, z_branch])

    # Rotate and attach
    a = np.deg2rad(angle_deg)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(a), -np.sin(a)],
            [0.0, np.sin(a), np.cos(a)],
        ]
    )
    branch = branch @ Rx.T
    branch[:, 2] += main_height

    P = np.vstack([main, branch])
    P += 0.02 * rng.standard_normal(P.shape)  # light noise
    return P.astype(np.float64)


# --------------------------- utils ---------------------------
def _bbox_diag(P: np.ndarray) -> float:
    if len(P) == 0:
        return 0.0
    mn = P.min(axis=0)
    mx = P.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def _knn_indices_and_dists(
    P: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(P)
    d, j = tree.query(P, k=min(k + 1, len(P)))
    if d.ndim == 1:
        d = d[:, None]
        j = j[:, None]
    return j.astype(np.int64), d.astype(np.float64)


def _pca_main_axis(pts: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return eigenvalues and eigenvectors sorted by eigenvalues (largest first)."""
    if len(pts) < 3:
        return None, None

    mu = pts.mean(axis=0)
    X = pts - mu
    C = (X.T @ X) / max(1, len(pts) - 1)

    try:
        w, V = np.linalg.eigh(C)
    except np.linalg.LinAlgError:
        return None, None

    # Sort by eigenvalues descending
    idx = np.argsort(w)[::-1]
    w_sorted = w[idx]
    V_sorted = V[:, idx]

    # Normalize eigenvectors
    for i in range(V_sorted.shape[1]):
        norm = np.linalg.norm(V_sorted[:, i])
        if norm > 0:
            V_sorted[:, i] /= norm

    return w_sorted, V_sorted


def _gaussian_weights(d: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 1e-12:
        return np.ones_like(d, dtype=np.float64)
    w = np.exp(-(d**2) / (2.0 * sigma * sigma))
    return np.clip(w, 1e-12, None)


def _cluster_points(P: np.ndarray, radius: float) -> np.ndarray:
    if len(P) == 0:
        return P.copy()
    tree = cKDTree(P)
    used = np.zeros(len(P), dtype=bool)
    centers = []
    for i in range(len(P)):
        if used[i]:
            continue
        idx = tree.query_ball_point(P[i], radius)
        used[idx] = True
        centers.append(P[idx].mean(axis=0))
    return np.asarray(centers, dtype=np.float64)


def _mst_edges(points: np.ndarray, k: int = 8) -> np.ndarray:
    if len(points) <= 1:
        return np.empty((0, 2), np.int32)
    tree = cKDTree(points)
    rows, cols, data = [], [], []
    for i, p in enumerate(points):
        d, j = tree.query(p, k=min(k + 1, len(points)))
        for dj, jj in zip(d[1:], j[1:]):
            rows.append(i)
            cols.append(int(jj))
            data.append(float(dj))
    G = csr_matrix((data, (rows, cols)), shape=(len(points), len(points)))
    MST = minimum_spanning_tree(G).tocoo()
    return np.vstack([MST.row, MST.col]).T.astype(np.int32)


def _lineset_from_graph(
    points: np.ndarray, edges: np.ndarray
) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(edges.astype(np.int32))
    return ls


# --------------------------- metrics from paper ---------------------------
class SkeletonMetrics:
    """Implementation of metrics from the paper."""

    @staticmethod
    def boundedness(P_s: np.ndarray, P_o: np.ndarray, k: int = 15) -> float:
        """Calculate boundedness metric (Section 2.2)."""
        if len(P_s) == 0:
            return 0.0

        tree_o = cKDTree(P_o)
        bounded_count = 0

        for p in P_s:
            dists, indices = tree_o.query(p, k=min(k, len(P_o)))
            if len(dists) == 0:
                continue
            center = np.mean(P_o[indices], axis=0)
            if np.linalg.norm(p - center) < np.mean(dists):
                bounded_count += 1

        return bounded_count / len(P_s)

    @staticmethod
    def centeredness(P_s: np.ndarray, P_o: np.ndarray, k: int = 15) -> float:
        """Calculate centeredness metric (Section 2.3)."""
        if len(P_s) == 0:
            return 0.0

        tree_o = cKDTree(P_o)
        centeredness_vals = []

        for p in P_s:
            dists, indices = tree_o.query(p, k=min(k, len(P_o)))
            if len(indices) < 3:
                continue
            neighbors = P_o[indices]
            center = np.mean(neighbors, axis=0)
            c_val = 1.0 - (
                np.linalg.norm(p - center)
                / np.mean(np.linalg.norm(neighbors - center, axis=1))
            )
            centeredness_vals.append(max(0.0, min(1.0, c_val)))

        return np.mean(centeredness_vals) if centeredness_vals else 0.0

    @staticmethod
    def smoothness(P_s: np.ndarray, k: int = 10) -> float:
        """Calculate smoothness metric (Section 2.4)."""
        if len(P_s) < 3:
            return 0.0

        tree_s = cKDTree(P_s)
        smoothness_vals = []

        for i, p in enumerate(P_s):
            dists, indices = tree_s.query(p, k=min(k + 1, len(P_s)))
            if len(indices) < 3:
                continue
            neighbor_indices = indices[1:]
            neighbors = P_s[neighbor_indices]

            w, V = _pca_main_axis(np.vstack([p, neighbors]))
            if w is None or V is None or len(V.shape) < 2:
                continue

            tangent = V[:, 0]  # Direction of max variance
            smooth_val = 1.0
            for j in range(min(3, len(neighbor_indices))):
                diff = neighbors[j] - p
                norm = np.linalg.norm(diff)
                if norm < 1e-12:
                    continue
                neighbor_tangent = diff / norm
                angle = np.arccos(
                    np.clip(np.dot(tangent, neighbor_tangent), -1.0, 1.0)
                )
                smooth_val = min(smooth_val, 1.0 - angle / np.pi)
            smoothness_vals.append(smooth_val)

        return np.mean(smoothness_vals) if smoothness_vals else 0.0


# --------------------------- contraction core ---------------------------
@dataclass
class ContractParams:
    k: int = 14
    beta: float = 0.7  # contraction strength
    lam: float = 0.28  # Laplacian smoothing strength
    iters: int = 100
    sigma_scale: float = 0.6
    rebuild_knn_every: int = 1
    use_normal_contraction: bool = True  # Use normal (min var) direction


class ContractionSkeletonizer:
    def __init__(self, points: np.ndarray, params: ContractParams):
        P = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        self.P0 = P.copy()
        self.X = P.copy()
        self.params = params
        self.history: List[np.ndarray] = [self.X.copy()]
        self.metrics_history: List[Dict[str, float]] = []
        self._tree_orig = cKDTree(self.P0)

        for t in range(params.iters):
            self._step(t)
            self.history.append(self.X.copy())

            if t % 10 == 0:
                metrics = self._calculate_metrics()
                self.metrics_history.append(metrics)

    def _step(self, t: int) -> None:
        p = self.params
        N = len(self.X)
        if N == 0:
            return

        # Rebuild kNN graph
        if (t % max(1, p.rebuild_knn_every)) == 0 or t == 0:
            idx, dist = _knn_indices_and_dists(self.X, p.k)
            self._last_idx, self._last_dist = idx, dist
        else:
            idx, dist = self._last_idx, self._last_dist

        nn = dist[:, 1:]
        sigma = max(
            1e-8, p.sigma_scale * float(np.median(nn)) if nn.size else 0.0
        )

        X1 = self.X.copy()

        if p.use_normal_contraction:
            self._normal_contraction(X1, idx, p)
        else:
            self._medial_axis_contraction(X1, idx, p)

        # Laplacian smoothing
        X2 = X1.copy()
        if p.lam > 1e-12:
            for i in range(N):
                neigh_idx = idx[i, 1:]
                if len(neigh_idx) == 0:
                    continue
                w = _gaussian_weights(dist[i, 1:], sigma)
                nb_avg = (w[:, None] * X1[neigh_idx]).sum(axis=0) / max(
                    1e-12, np.sum(w)
                )
                X2[i] = (1.0 - p.lam) * X1[i] + p.lam * nb_avg

        self.X = X2

    def _normal_contraction(
        self, X1: np.ndarray, idx: np.ndarray, p: ContractParams
    ) -> None:
        """
        Move points along the direction of MINIMAL variance (normal direction)
        towards the inside of the shape.
        """
        N = len(X1)
        for i in range(N):
            neigh_idx = idx[i, 1:]
            neighbors = self.X[neigh_idx]
            if len(neighbors) < 3:
                continue

            # Compute PCA
            w, V = _pca_main_axis(neighbors)
            if w is None or V is None:
                continue

            # Use the direction of LEAST variance (3rd eigenvector)
            normal_dir = V[:, 2]  # Smallest eigenvalue direction

            # Determine inward direction:
            # We want to move toward the "center" of the shape, not outward.
            # So we check: is the current point "outside" relative to the neighborhood?
            mu = neighbors.mean(axis=0)
            to_center = mu - self.X[i]
            if np.dot(to_center, normal_dir) < 0:
                normal_dir = -normal_dir  # Flip to point inward

            # Move in the inward normal direction
            X1[i] += (
                p.beta * normal_dir * np.linalg.norm(self.X[i] - mu) * 0.1
            )  # scaled step

    def _medial_axis_contraction(
        self, X1: np.ndarray, idx: np.ndarray, p: ContractParams
    ) -> None:
        """Legacy: Move along main axis (largest variance)"""
        N = len(X1)
        for i in range(N):
            neigh_idx = idx[i, 1:]
            neighbors = self.X[neigh_idx]
            if len(neighbors) < 3:
                continue

            w, V = _pca_main_axis(neighbors)
            if w is None or V is None:
                continue

            u = V[:, 0]  # Direction of largest variance
            mu = neighbors.mean(axis=0)
            proj = (self.X[i] - mu) @ u
            target = mu + proj * u
            X1[i] = self.X[i] + p.beta * (target - self.X[i])

    def _calculate_metrics(self) -> Dict[str, float]:
        metrics = {}
        metrics['boundedness'] = SkeletonMetrics.boundedness(self.X, self.P0)
        metrics['centeredness'] = SkeletonMetrics.centeredness(self.X, self.P0)
        metrics['smoothness'] = SkeletonMetrics.smoothness(self.X)
        return metrics

    def final(self) -> np.ndarray:
        return self.history[-1]

    def get_final_metrics(self) -> Dict[str, float]:
        return self._calculate_metrics()


# --------------------------- visualization ---------------------------
class InteractiveViewer:
    def __init__(
        self,
        P_orig: np.ndarray,
        history: List[np.ndarray],
        graph_k: int = 8,
        cluster_radius: Optional[float] = None,
    ):
        self.P_orig = P_orig
        self.history = history
        self.step = 0
        self.graph_k = graph_k
        self.cluster_radius = cluster_radius or 0.02 * _bbox_diag(P_orig)
        self._graph_enabled = False
        self._graph_ls: Optional[o3d.geometry.LineSet] = None

        self.pcd_orig = o3d.geometry.PointCloud()
        self.pcd_orig.points = o3d.utility.Vector3dVector(P_orig)
        self.pcd_orig.paint_uniform_color([0.7, 0.7, 0.7])

        self.pcd_curr = o3d.geometry.PointCloud()
        self._update_curr_geometry()

    def _update_curr_geometry(self):
        P = self.history[self.step]
        self.pcd_curr.points = o3d.utility.Vector3dVector(P)
        tree = cKDTree(self.P_orig)
        dists, _ = tree.query(P, k=1)
        max_dist = np.max(dists) if len(dists) > 0 else 1.0
        colors = np.zeros((len(P), 3))
        colors[:, 0] = 1.0
        colors[:, 1] = 1.0 - np.clip(dists / max_dist, 0, 1)
        colors[:, 2] = np.clip(dists / max_dist, 0, 1)
        self.pcd_curr.colors = o3d.utility.Vector3dVector(colors)

    def _rebuild_graph(self) -> o3d.geometry.LineSet:
        P = self.history[self.step]
        nodes = _cluster_points(P, max(1e-9, self.cluster_radius))
        E = _mst_edges(nodes, k=self.graph_k)
        ls = _lineset_from_graph(nodes, E)
        ls.paint_uniform_color([0.0, 0.2, 1.0])
        return ls

    def _cb_next(self, vis):
        if self.step < len(self.history) - 1:
            self.step += 1
            self._update_curr_geometry()
            vis.update_geometry(self.pcd_curr)
            if self._graph_enabled:
                if self._graph_ls is not None:
                    vis.remove_geometry(
                        self._graph_ls, reset_bounding_box=False
                    )
                self._graph_ls = self._rebuild_graph()
                vis.add_geometry(self._graph_ls, reset_bounding_box=False)
            print(f"[INFO] Step {self.step}/{len(self.history)-1}")
        return False

    def _cb_prev(self, vis):
        if self.step > 0:
            self.step -= 1
            self._update_curr_geometry()
            vis.update_geometry(self.pcd_curr)
            if self._graph_enabled:
                if self._graph_ls is not None:
                    vis.remove_geometry(
                        self._graph_ls, reset_bounding_box=False
                    )
                self._graph_ls = self._rebuild_graph()
                vis.add_geometry(self._graph_ls, reset_bounding_box=False)
            print(f"[INFO] Step {self.step}/{len(self.history)-1}")
        return False

    def _cb_toggle_graph(self, vis):
        self._graph_enabled = not self._graph_enabled
        if self._graph_enabled:
            self._graph_ls = self._rebuild_graph()
            vis.add_geometry(self._graph_ls, reset_bounding_box=False)
            print("[INFO] Graph overlay: ON")
        else:
            if self._graph_ls is not None:
                vis.remove_geometry(self._graph_ls, reset_bounding_box=False)
                self._graph_ls = None
            print("[INFO] Graph overlay: OFF")
        return False

    def _cb_save(self, vis):
        P_final = self.history[-1]
        o3d.io.write_point_cloud(
            "final_skeletal_points.ply",
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_final)),
        )
        print("[INFO] Saved: final_skeletal_points.ply")

        nodes = _cluster_points(P_final, max(1e-9, self.cluster_radius))
        E = _mst_edges(nodes, k=self.graph_k)
        o3d.io.write_line_set(
            "final_skeleton_graph.ply", _lineset_from_graph(nodes, E)
        )
        print("[INFO] Saved: final_skeleton_graph.ply")
        return False

    def _cb_reset(self, vis):
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        vis.update_renderer()
        print("[INFO] View reset")
        return False

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(
            "Laplacian Contraction (Normal-based) [A/D: prev/next, G: graph, S: save]",
            width=1280,
            height=800,
        )
        vis.add_geometry(self.pcd_orig, reset_bounding_box=True)
        vis.add_geometry(self.pcd_curr, reset_bounding_box=False)

        print(
            "Controls:\n D : Next step\n A : Previous step\n G : Toggle graph overlay\n"
            " S : Save final skeletal points + graph (PLY)\n R : Reset view\n Q : Quit"
        )

        vis.register_key_callback(ord("D"), self._cb_next)
        vis.register_key_callback(ord("A"), self._cb_prev)
        vis.register_key_callback(ord("G"), self._cb_toggle_graph)
        vis.register_key_callback(ord("S"), self._cb_save)
        vis.register_key_callback(ord("R"), self._cb_reset)
        vis.register_key_callback(ord("Q"), lambda v: sys.exit(0))

        vis.run()
        vis.destroy_window()


def main():
    try:
        from main import _load_cloud
        from vision.skeleton.config import SkelPipelineCfg

        cfg = SkelPipelineCfg()
        pc = _load_cloud(cfg)
        P = np.asarray(pc.points, dtype=np.float64)
        print(
            f"[INFO] Loaded cloud: {len(P)} points, bbox diag={_bbox_diag(P):.3f}"
        )
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        print(f"[WARNING] Could not load from config: {e}")
        print("[INFO] Using generated cylinders instead")
        P = generate_two_cylinders(n_main=800, n_branch=600)
        print(
            f"[INFO] Generated cloud: {len(P)} points, bbox diag={_bbox_diag(P):.3f}"
        )

    params = ContractParams(
        k=14,
        beta=0.7,
        lam=0.28,
        iters=100,
        sigma_scale=0.6,
        use_normal_contraction=True,  # <-- теперь используем сжатие по нормали
    )
    print(f"[INFO] Params: {params}")
    start_time = time.time()
    skel = ContractionSkeletonizer(P, params)
    end_time = time.time()

    print(f"[INFO] Contraction finished in {end_time - start_time:.2f} seconds")
    print(f"[INFO] Steps stored: {len(skel.history)}")

    final_metrics = skel.get_final_metrics()
    print("[INFO] Final metrics:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")

    viewer = InteractiveViewer(
        P_orig=P,
        history=skel.history,
        graph_k=8,
        cluster_radius=0.02 * _bbox_diag(P),
    )
    viewer.run()


if __name__ == "__main__":
    main()
