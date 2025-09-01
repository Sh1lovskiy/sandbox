"""Entry point for the TrussKit pipeline."""
from trusskit.pipeline import run

CLOUD_PATH = "data_captures/first/debug/final_merged.ply"
MERGE_NODE_RADIUS = 0.004
RASTER_RES_PX = 1024
SAVE_TAG = "first"


if __name__ == "__main__":
    run(
        cloud_path=CLOUD_PATH,
        merge_node_radius=MERGE_NODE_RADIUS,
        raster_res_px=RASTER_RES_PX,
        save_tag=SAVE_TAG,
    )
