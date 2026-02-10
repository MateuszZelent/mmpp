"""Interaction helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from typing import Any

import numpy as np


def closest_freq_index(explorer: Any, freq_ghz: float | None) -> int:
    """Return closest filtered-frequency index for ``freq_ghz``."""
    if explorer._filtered_frequencies_ghz.size == 0:
        return 0
    if freq_ghz is None:
        return int(explorer._filtered_frequencies_ghz.size // 2)
    idx = int(np.argmin(np.abs(explorer._filtered_frequencies_ghz - float(freq_ghz))))
    return max(0, min(idx, explorer._filtered_frequencies_ghz.size - 1))


def update_frequency_selection(explorer: Any, redraw_canvas: bool = True) -> None:
    """Refresh selected-frequency visuals and mode maps."""
    explorer._draw_frequency_line()
    explorer._update_mode_plots()

    if redraw_canvas and explorer._fig is not None:
        explorer._fig.canvas.draw_idle()


def on_spectrum_click(explorer: Any, event: Any) -> None:
    """Handle click selection on the spectrum axis."""
    if explorer._ax_spectrum is None or event.inaxes != explorer._ax_spectrum:
        return
    if event.xdata is None:
        return

    clicked_freq_ghz = float(event.xdata) / explorer._get_freq_scale(explorer._freq_unit)

    if event.button == 3 and explorer._peaks:
        peak_freqs = np.array([p[0] for p in explorer._peaks], dtype=float)
        idx = int(np.argmin(np.abs(peak_freqs - clicked_freq_ghz)))
        selected = float(peak_freqs[idx])
    else:
        selected = clicked_freq_ghz

    explorer._current_frequency_ghz = selected

    if explorer._controls and "freq_index" in explorer._controls:
        idx = closest_freq_index(explorer, selected)
        explorer._internal_update = True
        try:
            explorer._controls["freq_index"].value = idx
            explorer._controls["play"].value = idx
        finally:
            explorer._internal_update = False

    update_frequency_selection(explorer, redraw_canvas=True)


__all__ = [
    "closest_freq_index",
    "update_frequency_selection",
    "on_spectrum_click",
]
