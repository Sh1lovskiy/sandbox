"""Mesh reconstruction utilities."""
from __future__ import annotations

import open3d as o3d

from utils.logger import Logger
from graph import reconstruct_mesh

LOG = Logger.get_logger("tk.mesh")


def build_mesh(cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """Reconstruct mesh from point cloud using existing utility."""
    mesh = reconstruct_mesh(cloud)
    LOG.info(f"mesh: {len(mesh.vertices)}V {len(mesh.triangles)}T")
    return mesh
