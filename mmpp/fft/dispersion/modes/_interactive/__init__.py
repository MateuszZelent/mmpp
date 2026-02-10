"""
Internal submodules for InteractiveDispersionModes.

This package contains separated concerns for better code organization:
- widgets: ipywidgets creation & layout
- plotting: Matplotlib plot updates
- mode_extraction: Spatial mode extraction algorithms
"""

from .callbacks import (
    on_animate,
    on_save_animation,
    stop_animation,
)
from .filters import (
    build_compute_filters_config,
    build_live_filters_config,
)
from .layout import create_layout
from .mode_extraction import ModeExtractor
from .plotting import InteractivePlotter
from .presets import (
    apply_params,
    delete_preset,
    get_current_params,
    get_presets_dir,
    list_presets,
    load_preset,
    on_delete_preset,
    on_load_preset,
    on_refresh_presets,
    on_save_preset,
    refresh_preset_dropdown,
    save_preset,
)
from .state import (
    base_default_params,
    ensure_animation_state,
    ensure_runtime_state,
)
from .widgets import WidgetBuilder

__all__ = [
    "WidgetBuilder",
    "InteractivePlotter",
    "ModeExtractor",
    "on_animate",
    "on_save_animation",
    "stop_animation",
    "create_layout",
    "build_live_filters_config",
    "build_compute_filters_config",
    "base_default_params",
    "ensure_runtime_state",
    "ensure_animation_state",
    "get_presets_dir",
    "get_current_params",
    "apply_params",
    "save_preset",
    "load_preset",
    "delete_preset",
    "list_presets",
    "refresh_preset_dropdown",
    "on_save_preset",
    "on_load_preset",
    "on_delete_preset",
    "on_refresh_presets",
]
