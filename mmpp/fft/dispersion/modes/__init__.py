"""
Interactive Dispersion Modes Analysis Module.

Exports are loaded lazily so headless dispersion paths can use
``result.modes`` without importing ipywidgets or Matplotlib.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Models
    "BrillouinZoneConfig": (".models", "BrillouinZoneConfig"),
    "DispersionMode": (".models", "DispersionMode"),
    "FoldedDispersionResult": (".models", "FoldedDispersionResult"),
    "ModeProfile": (".mode_profile", "ModeProfile"),
    # Core
    "BrillouinZoneFolding": (".folding", "BrillouinZoneFolding"),
    "BrillouinZoneDetector": (".detection", "BrillouinZoneDetector"),
    "InteractiveDispersionModes": (".interactive", "InteractiveDispersionModes"),
    # Animation
    "extract_amplitude_phase": (".animation", "extract_amplitude_phase"),
    "compute_spinwave_field": (".animation", "compute_spinwave_field"),
    "generate_animation_frames": (".animation", "generate_animation_frames"),
    "SpinWaveModeAnimator": (".animation", "SpinWaveModeAnimator"),
    "animate_mode_from_folding": (".animation", "animate_mode_from_folding"),
    # Bridge (new fluent API)
    "DispersionModesBridge": (".bridge", "DispersionModesBridge"),
    "DispersionModeResult": (".bridge", "DispersionModeResult"),
    "DispersionModePlotAccessor": (".bridge", "DispersionModePlotAccessor"),
    "DispersionModesPlotAccessor": (".bridge", "DispersionModesPlotAccessor"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load mode exports only when accessed."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
