"""Widget control helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from typing import Any

import numpy as np

from .filters import SpectrumFilterState, normalize_component_selection


def guess_layer_bounds(explorer: Any) -> tuple[int, int]:
    """Best-effort z-layer slider bounds."""
    try:
        if explorer.analyzer is not None and getattr(explorer.analyzer, "modes_path", None):
            modes_path = explorer.analyzer.modes_path
            shape = explorer.analyzer.zarr_file[modes_path].shape
            n_layers = int(shape[1])
            if n_layers > 0:
                return -n_layers, n_layers - 1
    except Exception:
        pass
    return -10, 10


def resolve_mode_rows(mode_view: str) -> list[str]:
    """Resolve mode view selector into row identifiers."""
    view = (mode_view or "all").lower()
    if view == "magnitude":
        return ["magnitude"]
    if view == "phase":
        return ["phase"]
    if view == "combined":
        return ["combined"]
    return ["magnitude", "phase", "combined"]


def read_controls(explorer: Any) -> None:
    """Read widget values into internal explorer state."""
    if not explorer._controls:
        return

    freq_min = float(explorer._controls["fmin"].value)
    freq_max = float(explorer._controls["fmax"].value)

    if freq_min > freq_max:
        freq_min, freq_max = freq_max, freq_min

    explorer._filter_state = SpectrumFilterState(
        freq_min=freq_min,
        freq_max=freq_max,
        smooth_filter=str(explorer._controls["smooth_filter"].value),
        smooth_window=int(explorer._controls["smooth_window"].value),
        smooth_sigma=float(explorer._controls["smooth_sigma"].value),
        baseline_mode=str(explorer._controls["baseline_mode"].value),
        clip_percentile_low=float(explorer._controls["clip_low"].value),
        clip_percentile_high=float(explorer._controls["clip_high"].value),
        soft_threshold_percentile=float(explorer._controls["soft_threshold"].value),
        normalize=bool(explorer._controls["normalize"].value),
        log_scale=bool(explorer._controls["log_scale"].value),
    )

    selected_components = list(explorer._controls["components"].value)
    explorer._current_components = normalize_component_selection(
        selected_components,
        available=explorer._available_components,
    )

    explorer._current_z_layer = int(explorer._controls["z_layer"].value)
    explorer._show_peaks = bool(explorer._controls["show_peaks"].value)
    explorer._peak_prominence = float(explorer._controls["peak_prom"].value)
    explorer._peak_distance = int(explorer._controls["peak_dist"].value)
    explorer._mode_row_types = resolve_mode_rows(str(explorer._controls["mode_view"].value))

    # Layout controls
    if "aspect" in explorer._controls:
        explorer._mode_aspect = str(explorer._controls["aspect"].value)
    if "layout" in explorer._controls:
        explorer._layout_mode = str(explorer._controls["layout"].value)


def refresh_freq_slider_bounds(explorer: Any) -> None:
    """Refresh frequency slider/play bounds after filtering changes."""
    if not explorer._controls:
        return

    slider = explorer._controls["freq_index"]
    play = explorer._controls["play"]

    explorer._internal_update = True
    try:
        max_idx = max(int(explorer._filtered_frequencies_ghz.size) - 1, 0)
        slider.max = max_idx
        play.max = max_idx
        idx = explorer._closest_freq_index(explorer._current_frequency_ghz)
        slider.value = idx
        play.value = idx
    finally:
        explorer._internal_update = False


__all__ = [
    "guess_layer_bounds",
    "resolve_mode_rows",
    "read_controls",
    "refresh_freq_slider_bounds",
]
