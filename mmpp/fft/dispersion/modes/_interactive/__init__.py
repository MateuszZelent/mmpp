"""Lazy internal helpers for :class:`InteractiveDispersionModes`.

The widget builder and plotting helper are optional dependencies. Keeping this
namespace lazy lets the legacy mode controller be constructed for headless
export and cache workflows without importing the notebook stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "WidgetBuilder": (".widgets", "WidgetBuilder"),
    "InteractivePlotter": (".plotting", "InteractivePlotter"),
    "ModeExtractor": (".mode_extraction", "ModeExtractor"),
    "on_animate": (".callbacks", "on_animate"),
    "on_save_animation": (".callbacks", "on_save_animation"),
    "stop_animation": (".callbacks", "stop_animation"),
    "create_layout": (".layout", "create_layout"),
    "build_live_filters_config": (".filters", "build_live_filters_config"),
    "build_compute_filters_config": (".filters", "build_compute_filters_config"),
    "base_default_params": (".state", "base_default_params"),
    "ensure_runtime_state": (".state", "ensure_runtime_state"),
    "ensure_animation_state": (".state", "ensure_animation_state"),
    "get_presets_dir": (".presets", "get_presets_dir"),
    "get_current_params": (".presets", "get_current_params"),
    "apply_params": (".presets", "apply_params"),
    "save_preset": (".presets", "save_preset"),
    "load_preset": (".presets", "load_preset"),
    "delete_preset": (".presets", "delete_preset"),
    "list_presets": (".presets", "list_presets"),
    "refresh_preset_dropdown": (".presets", "refresh_preset_dropdown"),
    "on_save_preset": (".presets", "on_save_preset"),
    "on_load_preset": (".presets", "on_load_preset"),
    "on_delete_preset": (".presets", "on_delete_preset"),
    "on_refresh_presets": (".presets", "on_refresh_presets"),
}
__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load one internal helper on first use."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
