"""Internal helpers for hysteresis interactive plotting."""

from .animation import start_animation, stop_animation
from .callbacks import (
    nearest_loop_index,
    on_component_changed,
    on_index_changed,
    on_loop_click,
    on_play_toggle,
    on_roi_changed,
    on_z_layer_changed,
)
from .presets import (
    apply_preset_state,
    collect_preset_state,
    get_presets_dir,
    list_presets,
    load_preset,
    save_preset,
)
from .rendering import draw_loop_panel, update_loop_cursor
from .snapshot import SnapshotCache, render_snapshot
from .state import HysteresisExplorerState
from .status import set_status
from .widgets import build_toolbar

__all__ = [
    "HysteresisExplorerState",
    "SnapshotCache",
    "build_toolbar",
    "draw_loop_panel",
    "update_loop_cursor",
    "render_snapshot",
    "on_loop_click",
    "on_index_changed",
    "on_component_changed",
    "on_z_layer_changed",
    "on_roi_changed",
    "on_play_toggle",
    "nearest_loop_index",
    "start_animation",
    "stop_animation",
    "set_status",
    "save_preset",
    "load_preset",
    "list_presets",
    "get_presets_dir",
    "collect_preset_state",
    "apply_preset_state",
]
