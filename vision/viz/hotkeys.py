# vision/viz/hotkeys.py
from __future__ import annotations
from typing import Callable, Dict
import open3d as o3d
from utils.logger import Logger

LOG = Logger.get_logger("hotkeys")
KeyFn = Callable[[], None]


def _wrap(
    vis: o3d.visualization.Visualizer, fn: KeyFn
) -> Callable[[o3d.visualization.Visualizer], bool]:

    def _cb(_: o3d.visualization.Visualizer) -> bool:
        try:
            fn()
        except Exception as e:
            LOG.error(f"hotkey callback failed: {e}")
        try:
            vis.update_renderer()
        except Exception:
            pass
        return True

    return _cb


def register(
    vis: o3d.visualization.VisualizerWithKeyCallback, mapping: Dict[str, KeyFn]
) -> None:
    for key, fn in mapping.items():
        try:
            vis.register_key_callback(ord(key), _wrap(vis, fn))
        except Exception as e:
            LOG.error(f"failed to register '{key}': {e}")
