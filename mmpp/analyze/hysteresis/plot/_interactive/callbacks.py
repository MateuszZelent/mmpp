"""Callbacks for hysteresis explorer interactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from ...animation import create_animation
from .animation import start_animation, stop_animation
from .status import set_status


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
    debug = bool(getattr(explorer, "_debug_clicks", False))

    if event.inaxes != explorer._ax_loop:
        if debug:
            set_status(
                explorer, "click ignored: outside hysteresis axis", color="#7c3aed"
            )
        return
    if event.xdata is None or event.ydata is None:
        if debug:
            set_status(explorer, "click ignored: missing x/y data", color="#7c3aed")
        return

    click_x = float(event.xdata)
    click_y = float(event.ydata)
    idx = nearest_loop_index(
        explorer.result.field,
        explorer.result.magnetization,
        click_x,
        click_y,
    )
    if debug:
        set_status(
            explorer,
            (
                f"click detected: x={click_x:.5g}, y={click_y:.5g}, "
                f"nearest_idx={idx}, field={float(explorer.result.field[idx]):.5g}"
            ),
            color="#0369a1",
        )
    explorer._set_index(idx)


def on_index_changed(explorer, idx: int) -> None:
    """Slider callback for point index."""
    explorer._set_index(int(idx))


def nearest_field_index(
    field: np.ndarray, value: float, *, prefer_idx: int | None = None
) -> int:
    """Return nearest index for selected field value.

    For duplicated field values (ascending/descending branches), prefer the
    candidate closest to the currently selected index.
    """
    field_arr = np.asarray(field, dtype=float)
    diffs = np.abs(field_arr - float(value))
    if diffs.size == 0:
        return 0

    best = float(np.nanmin(diffs))
    tol = max(1e-12, best * 1e-9)
    candidates = np.flatnonzero(diffs <= (best + tol))
    if candidates.size == 0:
        return int(np.argmin(diffs))

    if prefer_idx is None:
        return int(candidates[0])

    ref = int(prefer_idx)
    local = int(np.argmin(np.abs(candidates - ref)))
    return int(candidates[local])


def on_field_changed(explorer, field_value: float) -> None:
    """Slider callback for field value selection."""
    idx = nearest_field_index(
        explorer.result.field,
        float(field_value),
        prefer_idx=int(explorer.state.current_idx),
    )
    explorer._set_index(idx)


def on_panel_widths_changed(explorer, loop_width: float, snapshot_width: float) -> None:
    """Update relative widths of loop and snapshot panels."""
    explorer._set_panel_widths(float(loop_width), float(snapshot_width), redraw=True)


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


def on_animation_speed_changed(explorer, speed: float) -> None:
    """Adjust playback speed multiplier."""
    value = float(max(0.1, min(10.0, speed)))
    explorer.state.animation_speed = value
    if explorer.state.is_animating:
        # Recreate animation with updated interval.
        start_animation(explorer)


def on_animation_fps_changed(explorer, fps: int) -> None:
    """Adjust base FPS used by online animation and export defaults."""
    explorer.result.config.animation_fps = int(max(1, fps))
    if explorer.state.is_animating:
        start_animation(explorer)


def on_trail_length_changed(explorer, trail_length: int) -> None:
    """Adjust trail length for marker history."""
    explorer.result.config.trail_length = int(max(1, trail_length))
    explorer._redraw_loop()


def _resolve_animation_target(path_text: str, fmt: str) -> Path:
    raw = str(path_text or "").strip()
    fmt_norm = str(fmt).strip().lower()
    if fmt_norm not in {"mp4", "gif"}:
        fmt_norm = "gif"

    if raw:
        target = Path(raw).expanduser()
        if target.suffix:
            return target
        return target.with_suffix(f".{fmt_norm}")
    return Path.cwd() / f"hysteresis_walk.{fmt_norm}"


def on_save_animation_clicked(explorer) -> None:
    """Save exported animation (MP4/GIF) from current explorer state."""
    controls: dict[str, Any] = getattr(explorer, "_controls", {})
    button = controls.get("save_animation")
    if button is None:
        return

    fps = (
        int(cast(Any, controls.get("anim_fps")).value)
        if controls.get("anim_fps") is not None
        else int(explorer.result.config.animation_fps)
    )
    trail = (
        int(cast(Any, controls.get("anim_trail")).value)
        if controls.get("anim_trail") is not None
        else int(explorer.result.config.trail_length)
    )
    fmt = (
        str(cast(Any, controls.get("anim_format")).value)
        if controls.get("anim_format") is not None
        else "gif"
    )
    path_text = (
        str(cast(Any, controls.get("anim_path")).value)
        if controls.get("anim_path") is not None
        else ""
    )
    use_snapshot = (
        bool(cast(Any, controls.get("anim_snapshot")).value)
        if controls.get("anim_snapshot") is not None
        else True
    )

    target = _resolve_animation_target(path_text, fmt)
    target.parent.mkdir(parents=True, exist_ok=True)

    old_description = str(button.description)
    old_style = str(button.button_style)
    button.description = "Saving..."
    button.button_style = "warning"
    button.disabled = True

    set_status(explorer, f"Saving animation to {target} ...", color="#0F766E")
    try:
        out = create_animation(
            explorer.result,
            save_path=target,
            fps=fps,
            show_arrow=bool(explorer.state.show_flags.get("arrow", True)),
            trail_length=trail,
            snapshot=use_snapshot,
            snapshot_component=str(explorer.state.snapshot_component),
            z_layer=explorer.state.z_layer,
            roi=explorer.state.roi,
            loop_width=float(explorer.state.loop_panel_weight),
            snapshot_width=float(explorer.state.snapshot_panel_weight),
            show_hc=bool(explorer.state.show_flags.get("hc", True)),
            show_mr=bool(explorer.state.show_flags.get("mr", True)),
            show_ms=bool(explorer.state.show_flags.get("ms", False)),
            show_branch_colors=bool(
                explorer.state.show_flags.get("branch_colors", True)
            ),
            bitrate="auto",
            dpi=explorer.result.config.dpi,
        )
        set_status(explorer, f"Animation saved: {out}", color="#0F766E")
    except Exception as exc:
        set_status(explorer, f"Animation export failed: {exc}", color="crimson")
    finally:
        button.disabled = False
        button.description = old_description
        button.button_style = old_style
