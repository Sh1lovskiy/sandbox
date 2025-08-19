from __future__ import annotations

from pathlib import Path

import open3d as o3d

from utils.error_tracker import ErrorTracker
from utils.logger import Logger

from .config import PipelineCfg
from .export import interactive_picks_and_save, save_cloud
from .pipeline import merge_capture

LOG = Logger.get_logger("main")


def run(cfg: PipelineCfg | None = None) -> Path | None:
    """
    Entry point: configure logging, install ErrorTracker, run pipeline.
    Returns saved path if the final cloud was written, else None.
    """
    Logger.configure()
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()
    ErrorTracker.install_keyboard_listener(stop_key="esc")

    cfg = cfg or PipelineCfg()
    root = Path(cfg.capture_root)
    LOG.info(f"[START] Root={root}")

    cloud = merge_capture(root, cfg)
    if len(cloud.points) == 0:
        LOG.warning("Empty cloud - nothing to visualize.")
        return None

    out_path = save_cloud(cloud, root, name="final_merged.ply")
    if cfg.capture_picks:
        interactive_picks_and_save(cloud, root, prefix="final")
    elif cfg.open_final_viewer:
        o3d.visualization.draw_geometries_with_editing(
            [cloud], window_name="FINAL - editing"
        )
    return out_path


def _main() -> None:
    """Module runner for `python -m merge_pipeline.main`."""
    run(PipelineCfg())


if __name__ == "__main__":
    _main()
