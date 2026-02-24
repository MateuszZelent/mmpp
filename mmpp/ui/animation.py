"""Shared animation helpers for loop and snapshot explorers."""

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

from .snapshot import SnapshotCache, render_snapshot


def _resolve_loop_axis_labels(result) -> tuple[str, str]:
    meta = getattr(result, "metadata", {}) or {}

    field_unit = str(meta.get("field_unit", "input")).strip()
    raw_field = str(meta.get("field_column") or meta.get("field_source") or "B").strip().lower()
    field_symbol = "H" if raw_field.startswith("h") else "B"
    field_suffix = raw_field[-1] if raw_field else ""
    if field_suffix in {"x", "y", "z"}:
        xlabel = rf"${field_symbol}_{field_suffix}$ ({field_unit})"
    else:
        xlabel = rf"${field_symbol}$ ({field_unit})"

    column = meta.get("magnetization_column")
    if isinstance(column, str) and column.strip():
        value = column.strip().lower()
        if value in {"mx", "my", "mz"}:
            ylabel = rf"$m_{value[-1]}$"
        elif value in {"m_full", "norm", "|m|", "magnitude"}:
            ylabel = r"$|m|$"
        else:
            ylabel = column.strip()
    else:
        component = str(meta.get("component", "")).strip().lower()
        if component in {"x", "y", "z"}:
            ylabel = rf"$m_{component}$"
        elif component in {"norm", "|m|", "magnitude"}:
            ylabel = r"$|m|$"
        else:
            ylabel = r"$M$"

    return xlabel, ylabel


def create_animation(
    result,
    *,
    save_path: str | Path | None = None,
    fps: int | None = None,
    show_arrow: bool = True,
    trail_length: int | None = None,
    snapshot: bool = True,
    snapshot_component: str | None = None,
    z_layer: int | str | None = None,
    roi: tuple[int, int, int, int] | None = None,
    loop_width: float | None = None,
    snapshot_width: float | None = None,
    show_hc: bool | None = None,
    show_mr: bool | None = None,
    show_ms: bool | None = None,
    show_branch_colors: bool | None = None,
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
    component = str(
        snapshot_component if snapshot_component is not None else result.config.snapshot_component
    )
    z_layer_value: int | str = result.config.z_layer if z_layer is None else z_layer
    roi_value = result.metadata.get("roi") if roi is None else roi
    loop_weight = float(max(0.2, loop_width if loop_width is not None else 1.15))
    snap_weight = float(max(0.2, snapshot_width if snapshot_width is not None else 1.0))
    show_hc_val = bool(result.config.show_hc if show_hc is None else show_hc)
    show_mr_val = bool(result.config.show_mr if show_mr is None else show_mr)
    show_ms_val = bool(result.config.show_ms if show_ms is None else show_ms)
    show_branch_colors_val = bool(
        result.config.show_branch_colors if show_branch_colors is None else show_branch_colors
    )

    source_type = str(result.metadata.get("source_type", ""))
    frame_keys = result.metadata.get("frame_keys")
    zarr_group = result.metadata.get("zarr_group")
    have_snapshot = (
        bool(snapshot)
        and result.frame_index is not None
        and (
            (
                result.metadata.get("job_result") is not None
                and result.metadata.get("dataset") is not None
            )
            or (source_type == "zarr_keys" and frame_keys is not None and zarr_group is not None)
        )
    )

    if have_snapshot:
        fig = plt.figure(figsize=result.config.figsize, dpi=dpi or result.config.dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[loop_weight, snap_weight], wspace=0.2)
        ax_loop = fig.add_subplot(gs[0, 0])
        ax_snap = fig.add_subplot(gs[0, 1])
        if source_type == "zarr_keys" and frame_keys is not None and zarr_group is not None:
            cache = SnapshotCache(
                result.metadata.get("job_result"),
                frame_keys=frame_keys,
                zarr_group=zarr_group,
                max_cached=50,
            )
        else:
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

    if result.branches:
        for branch in result.branches:
            x = field[branch.slice]
            y = magnetization[branch.slice]
            if x.size == 0:
                continue
            if show_branch_colors_val:
                color = (
                    result.config.branch_colors[0]
                    if branch.name == "ascending"
                    else result.config.branch_colors[1]
                )
            else:
                color = "#3b82f6"
            ax_loop.plot(x, y, lw=1.0, color=color, alpha=0.35, zorder=2)
            ax_loop.scatter(x, y, s=26, color=color, alpha=0.9, linewidths=0.0, zorder=4)
    else:
        ax_loop.plot(field, magnetization, lw=1.0, color="#2563eb", alpha=0.35, zorder=2)
        ax_loop.scatter(field, magnetization, s=26, color="#2563eb", alpha=0.9, linewidths=0.0, zorder=4)

    marker = ax_loop.scatter(
        [field[0]],
        [magnetization[0]],
        s=140,
        color="#f59e0b",
        edgecolors="#111827",
        linewidths=1.2,
        zorder=7,
    )
    trail_line, = ax_loop.plot([], [], color="#0ea5e9", lw=2.0, alpha=result.config.trail_alpha_decay)
    arrow = ax_loop.annotate(
        "",
        xy=(field[1], magnetization[1]),
        xytext=(field[0], magnetization[0]),
        arrowprops={"arrowstyle": "->", "color": "#f59e0b", "lw": 1.5, "alpha": 0.9},
    )

    if show_hc_val:
        hc = result.metrics.coercive_field
        for value in (hc.hc_minus, hc.hc_plus):
            if np.isfinite(value):
                ax_loop.axvline(float(value), ls="--", lw=1.1, color="#f97316", alpha=0.75)
    if show_mr_val:
        mr = result.metrics.remanence
        for value in (mr.mr_minus, mr.mr_plus):
            if np.isfinite(value):
                ax_loop.axhline(float(value), ls=":", lw=1.1, color="#a855f7", alpha=0.75)
    if show_ms_val:
        ms = result.metrics.saturation_points
        if np.isfinite(ms.ms_positive):
            ax_loop.scatter([ms.hs_positive], [ms.ms_positive], color="#22c55e", s=42, zorder=6)
        if np.isfinite(ms.ms_negative):
            ax_loop.scatter([ms.hs_negative], [ms.ms_negative], color="#ef4444", s=42, zorder=6)

    xlabel, ylabel = _resolve_loop_axis_labels(result)
    ax_loop.set_xlabel(xlabel)
    ax_loop.set_ylabel(ylabel)
    ax_loop.set_title("")
    ax_loop.grid(True, alpha=0.25)

    def _update(idx):
        i = int(idx) % n_points
        marker.set_offsets(np.array([[field[i], magnetization[i]]], dtype=float))

        start = max(0, i - trail)
        trail_line.set_data(field[start : i + 1], magnetization[start : i + 1])

        if show_arrow and i < (n_points - 1):
            j = i + 1
            arrow.set_visible(True)
            arrow.xy = (field[j], magnetization[j])
            arrow.set_position((field[i], magnetization[i]))
        else:
            arrow.set_visible(False)

        if cache is not None and ax_snap is not None:
            frame_idx = int(result.frame_index[i]) if result.frame_index is not None else i
            frame = cache.get_frame(
                frame_idx,
                component=component,
                z_layer=z_layer_value,
                roi=roi_value,
            )
            job_result = result.metadata.get("job_result")
            attrs = getattr(job_result, "attrs", {}) if job_result is not None else {}
            dx = float(attrs.get("dx", 1e-9)) if hasattr(attrs, "get") else 1e-9
            dy = float(attrs.get("dy", 1e-9)) if hasattr(attrs, "get") else 1e-9
            render_snapshot(
                ax_snap,
                frame,
                component=component,
                field_value=float(field[i]),
                field_unit=str(result.metadata.get("field_unit", "input")),
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
