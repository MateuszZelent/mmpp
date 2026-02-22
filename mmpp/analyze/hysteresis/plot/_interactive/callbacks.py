"""Callbacks for hysteresis explorer interactions."""

from __future__ import annotations

import numpy as np

from .animation import start_animation, stop_animation


def nearest_loop_index(field: np.ndarray, mag: np.ndarray, x: float, y: float) -> int:
    """Return nearest loop point index to clicked coordinates."""
    field_arr = np.asarray(field, dtype=float)
    mag_arr = np.asarray(mag, dtype=float)

    # Axis normalization keeps behavior stable for differently scaled axes.
    x_span = max(float(np.nanmax(field_arr) - np.nanmin(field_arr)), 1e-12)
    y_span = max(float(np.nanmax(mag_arr) - np.nanmin(mag_arr)), 1e-12)

    dx = (field_arr - float(x)) / x_span
    dy = (mag_arr - float(y)) / y_span
    idx = int(np.argmin(dx * dx + dy * dy))
    return idx


def on_loop_click(explorer, event) -> None:
    """Click callback on loop panel."""
    if event.inaxes != explorer._ax_loop:
        return
    if event.xdata is None or event.ydata is None:
        return

    idx = nearest_loop_index(
        explorer.result.field,
        explorer.result.magnetization,
        float(event.xdata),
        float(event.ydata),
    )
    explorer._set_index(idx)


def on_index_changed(explorer, idx: int) -> None:
    """Slider callback for point index."""
    explorer._set_index(int(idx))


def on_component_changed(explorer, component: str) -> None:
    """Change right-panel component mode."""
    explorer.state.snapshot_component = str(component)
    explorer._update_snapshot(redraw=True)


def on_z_layer_changed(explorer, z_layer: int | str) -> None:
    """Change z-layer for snapshot view."""
    explorer.state.z_layer = z_layer
    if explorer._snapshot_cache is not None:
        explorer._snapshot_cache.clear()
    explorer._update_snapshot(redraw=True)


def on_roi_changed(explorer, roi: tuple[int, int, int, int] | None) -> None:
    """Change ROI for snapshot extraction."""
    explorer.state.roi = roi
    if explorer._snapshot_cache is not None:
        explorer._snapshot_cache.clear()
    explorer._update_snapshot(redraw=True)


def on_play_toggle(explorer, enabled: bool) -> None:
    """Play/pause callback."""
    if bool(enabled):
        start_animation(explorer)
    else:
        stop_animation(explorer)
