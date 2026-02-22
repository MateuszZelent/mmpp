"""Animation helpers for hysteresis walkthrough export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation as mpl_animation

    FuncAnimation = mpl_animation.FuncAnimation
    FFMpegWriter = mpl_animation.FFMpegWriter
    PillowWriter = mpl_animation.PillowWriter
    _HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    FuncAnimation = FFMpegWriter = PillowWriter = None  # type: ignore[assignment]
    _HAS_MPL = False

from .plot._interactive.snapshot import SnapshotCache, render_snapshot


def create_animation(
    result,
    *,
    save_path: str | Path | None = None,
    fps: int | None = None,
    show_arrow: bool = True,
    trail_length: int | None = None,
    snapshot: bool = True,
    bitrate: str | int = "auto",
    dpi: int | None = None,
):
    """Create online animation or export to MP4/GIF."""
    if not _HAS_MPL:
        raise ImportError("Matplotlib animation support is required")

    field = np.asarray(result.field, dtype=float)
    magnetization = np.asarray(result.magnetization, dtype=float)
    n_points = int(field.size)
    if n_points < 2:
        raise ValueError("Need at least 2 points for animation")

    fps_value = int(fps if fps is not None else result.config.animation_fps)
    trail = int(trail_length if trail_length is not None else result.config.trail_length)

    have_snapshot = (
        bool(snapshot)
        and result.metadata.get("job_result") is not None
        and result.metadata.get("dataset") is not None
        and result.frame_index is not None
    )

    if have_snapshot:
        fig = plt.figure(figsize=result.config.figsize, dpi=dpi or result.config.dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.2)
        ax_loop = fig.add_subplot(gs[0, 0])
        ax_snap = fig.add_subplot(gs[0, 1])
        cache = SnapshotCache(
            result.metadata["job_result"],
            dset=str(result.metadata["dataset"]),
            slice_info=result.metadata.get("slice_info"),
            max_cached=50,
        )
    else:
        fig, ax_loop = plt.subplots(figsize=result.config.figsize, dpi=dpi or result.config.dpi)
        ax_snap = None
        cache = None

    ax_loop.plot(field, magnetization, lw=1.8, color="#2563eb", alpha=0.9)
    marker = ax_loop.scatter([field[0]], [magnetization[0]], s=90, color="#111827", zorder=6)
    trail_line, = ax_loop.plot([], [], color="#0ea5e9", lw=2.0, alpha=result.config.trail_alpha_decay)
    arrow = ax_loop.annotate(
        "",
        xy=(field[1], magnetization[1]),
        xytext=(field[0], magnetization[0]),
        arrowprops={"arrowstyle": "->", "color": "#111827", "lw": 1.4, "alpha": 0.85},
    )
    ax_loop.set_xlabel(f"Field [{result.metadata.get('field_unit', 'input')}]")
    ax_loop.set_ylabel("Magnetization")
    ax_loop.set_title("Hysteresis animation")
    ax_loop.grid(True, alpha=0.25)

    def _update(idx):
        i = int(idx) % n_points
        marker.set_offsets(np.array([[field[i], magnetization[i]]], dtype=float))

        start = max(0, i - trail)
        trail_line.set_data(field[start : i + 1], magnetization[start : i + 1])

        if show_arrow:
            j = (i + 1) % n_points
            arrow.set_visible(True)
            arrow.xy = (field[j], magnetization[j])
            arrow.set_position((field[i], magnetization[i]))
        else:
            arrow.set_visible(False)

        if cache is not None and ax_snap is not None:
            frame_idx = int(result.frame_index[i]) if result.frame_index is not None else i
            frame = cache.get_frame(
                frame_idx,
                component=result.config.snapshot_component,
                z_layer=result.config.z_layer,
                roi=result.metadata.get("roi"),
            )
            attrs = getattr(result.metadata["job_result"], "attrs", {})
            dx = float(attrs.get("dx", 1e-9)) if hasattr(attrs, "get") else 1e-9
            dy = float(attrs.get("dy", 1e-9)) if hasattr(attrs, "get") else 1e-9
            render_snapshot(
                ax_snap,
                frame,
                component=result.config.snapshot_component,
                dx=dx,
                dy=dy,
                cmap=result.config.colormap_magnitude,
            )
        return []

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_points,
        interval=1000.0 / max(1, fps_value),
        repeat=True,
        blit=False,
    )

    if save_path is None:
        return anim

    target = Path(save_path)
    suffix = target.suffix.lower()
    if suffix == ".mp4":
        bitrate_value = 2000 if bitrate == "auto" else int(bitrate)
        writer = FFMpegWriter(fps=fps_value, bitrate=bitrate_value)
    elif suffix == ".gif":
        writer = PillowWriter(fps=fps_value)
    else:
        raise ValueError("save_path extension must be .mp4 or .gif")

    anim.save(target, writer=writer, dpi=dpi or result.config.dpi)
    return target


__all__ = ["create_animation"]
