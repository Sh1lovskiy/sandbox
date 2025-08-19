from __future__ import annotations

import open3d as o3d

from utils.logger import Logger

LOG = Logger.get_logger("viz")


def draw_with_roi(
    cloud: o3d.geometry.PointCloud,
    aabb: o3d.geometry.AxisAlignedBoundingBox,
    coord_frame_size: float = 0.05,
    title: str = "Viewer",
) -> None:
    """Simple viewer with ROI and axes."""
    aabb_ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    aabb_ls.paint_uniform_color((1.0, 0.1, 0.1))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coord_frame_size
    )
    o3d.visualization.draw_geometries([cloud, aabb_ls, axes], window_name=title)
