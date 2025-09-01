"""Configuration dataclasses for the truss pipeline."""
from dataclasses import dataclass


@dataclass
class RunConfig:
    cloud_path: str
    merge_node_radius: float
    raster_res_px: int
    save_tag: str
