"""Interactive dispersion widget internals.

Public callers should use ``result.plot.interactive()`` or
``m.fft.dispersion.plot.interactive()``. This package keeps notebook widget
construction separate from the lightweight public controller.
"""

from .callbacks import on_canvas_click, on_display_change
from .presets import (
    apply_preset_state,
    collect_preset_state,
    get_presets_dir,
    list_presets,
    load_preset,
    save_preset,
)
from .rendering import draw_dispersion_panel, refresh_output_widget
from .state import DispersionExplorerState
from .status import set_status
from .widget import DispersionHeatmapWidget
from .widgets import build_toolbar

__all__ = [
    "DispersionExplorerState",
    "DispersionHeatmapWidget",
    "apply_preset_state",
    "build_toolbar",
    "collect_preset_state",
    "draw_dispersion_panel",
    "get_presets_dir",
    "list_presets",
    "load_preset",
    "on_canvas_click",
    "on_display_change",
    "refresh_output_widget",
    "save_preset",
    "set_status",
]
