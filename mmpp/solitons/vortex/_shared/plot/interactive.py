"""Interactive trajectory viewer with matplotlib fallback controls."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation as mpl_animation
from matplotlib.widgets import Button, Slider

from .....ui.snapshot import SnapshotCache, render_snapshot


def _resolve_snapshot_context(trajectory):
    meta = dict(getattr(trajectory, "metadata", {}) or {})
    job_result = meta.get("job_result")
    dataset = meta.get("dataset")
    slice_info = meta.get("slice_info")

    if job_result is None or dataset is None:
        return None

    attrs = getattr(job_result, "attrs", {})
    dx = float(attrs.get("dx", 1e-9)) if hasattr(attrs, "get") else 1e-9
    dy = float(attrs.get("dy", 1e-9)) if hasattr(attrs, "get") else 1e-9
    frame_index = np.arange(trajectory.time.size, dtype=int)
    return {
        "job_result": job_result,
        "dataset": str(dataset),
        "slice_info": slice_info,
        "dx": dx,
        "dy": dy,
        "frame_index": frame_index,
    }


def trajectory_interactive(
    trajectory,
    *,
    snapshot: bool = False,
    snapshot_component: str = "snapshot",
    z_layer: int | str = 0,
    roi: tuple[int, int, int, int] | None = None,
    toolbar: str | bool = "auto",
    fps: int = 30,
    trail_length: int = 60,
    **kwargs,
):
    """Interactive trajectory view with slider, click-select and play/pause.

    Controls are implemented with matplotlib widgets, so this works both in a
    notebook and as a plain matplotlib fallback without ipywidgets.
    """
    _ = toolbar, kwargs
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("trajectory must contain at least one sample")

    context = _resolve_snapshot_context(trajectory)
    show_snapshot = bool(snapshot and context is not None)

    if show_snapshot:
        fig = plt.figure(figsize=(10.8, 5.2), dpi=110)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.20)
        ax_orbit = fig.add_subplot(gs[0, 0])
        ax_snapshot = fig.add_subplot(gs[0, 1])
        cache = SnapshotCache(
            context["job_result"],
            dset=context["dataset"],
            slice_info=context["slice_info"],
            max_cached=50,
        )
    else:
        fig, ax_orbit = plt.subplots(figsize=(6.2, 5.2), dpi=110)
        ax_snapshot = None
        cache = None

    ax_orbit.plot(x, y, color="#1d4ed8", lw=1.6, alpha=0.85)
    (trail_line,) = ax_orbit.plot([], [], color="#0ea5e9", lw=2.2, alpha=0.85)
    marker = ax_orbit.scatter(
        [x[0]], [y[0]], s=70, color="#f59e0b", edgecolors="black", zorder=5
    )
    ax_orbit.set_title("Core trajectory")
    ax_orbit.set_xlabel("X [m]")
    ax_orbit.set_ylabel("Y [m]")
    ax_orbit.set_aspect("equal")
    ax_orbit.grid(True, alpha=0.25)

    slider_ax = fig.add_axes((0.16, 0.035, 0.52, 0.035))
    play_ax = fig.add_axes((0.71, 0.028, 0.12, 0.05))
    frame_slider = Slider(
        slider_ax,
        "Frame",
        valmin=0,
        valmax=max(int(x.size - 1), 0),
        valinit=0,
        valstep=1,
    )
    play_btn = Button(play_ax, "Play", color="#e5e7eb", hovercolor="#d1d5db")

    state = {"index": 0, "playing": False}

    def _update_visuals(frame_idx: int) -> None:
        idx = int(np.clip(int(frame_idx), 0, x.size - 1))
        state["index"] = idx
        marker.set_offsets(np.array([[x[idx], y[idx]]], dtype=float))

        trail = max(int(trail_length), 1)
        start = max(0, idx - trail)
        trail_line.set_data(x[start : idx + 1], y[start : idx + 1])

        if ax_snapshot is not None and cache is not None:
            frame_map = context["frame_index"] if context is not None else None
            mapped_idx = int(frame_map[idx]) if frame_map is not None else idx
            frame = cache.get_frame(
                mapped_idx,
                component=snapshot_component,
                z_layer=z_layer,
                roi=roi,
            )
            render_snapshot(
                ax_snapshot,
                frame,
                component=snapshot_component,
                dx=float(context["dx"]) if context is not None else 1.0,
                dy=float(context["dy"]) if context is not None else 1.0,
            )

        if int(round(frame_slider.val)) != idx:
            frame_slider.eventson = False
            frame_slider.set_val(idx)
            frame_slider.eventson = True
        fig.canvas.draw_idle()

    def _on_slider(val):
        _update_visuals(int(round(float(val))))

    def _toggle_play(_event):
        state["playing"] = not state["playing"]
        play_btn.label.set_text("Pause" if state["playing"] else "Play")
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes is not ax_orbit:
            return
        if event.xdata is None or event.ydata is None:
            return
        dist2 = (x - float(event.xdata)) ** 2 + (y - float(event.ydata)) ** 2
        _update_visuals(int(np.argmin(dist2)))

    def _animate(_frame):
        if not state["playing"]:
            return ()
        _update_visuals((state["index"] + 1) % int(x.size))
        return ()

    frame_slider.on_changed(_on_slider)
    play_btn.on_clicked(_toggle_play)
    fig.canvas.mpl_connect("button_press_event", _on_click)

    anim = mpl_animation.FuncAnimation(
        fig,
        _animate,
        interval=1000.0 / max(int(fps), 1),
        blit=False,
        cache_frame_data=False,
    )
    # keep widgets/animation alive for interactive backends
    fig._mmpp_interactive = {  # type: ignore[attr-defined]
        "slider": frame_slider,
        "play_button": play_btn,
        "animation": anim,
        "state": state,
    }
    _update_visuals(0)
    return fig


__all__ = ["trajectory_interactive"]
