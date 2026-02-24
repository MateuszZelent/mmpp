"""Rendering helpers for interactive hysteresis explorer."""

from __future__ import annotations

import numpy as np


def _field_axis_label_from_metadata(meta: dict) -> str:
    unit = str(meta.get("field_unit", "input")).strip()

    raw = str(meta.get("field_column") or meta.get("field_source") or "B").strip().lower()
    symbol = "H" if raw.startswith("h") else "B"
    suffix = raw[-1] if raw else ""
    if suffix in {"x", "y", "z"}:
        var = rf"${symbol}_{suffix}$"
    else:
        var = rf"${symbol}$"
    return f"{var} ({unit})"


def _magnetization_label_from_metadata(meta: dict) -> str:
    column = meta.get("magnetization_column")
    if isinstance(column, str) and column.strip():
        c = column.strip()
        cl = c.lower()
        if cl in {"mx", "my", "mz"}:
            return rf"$m_{cl[-1]}$"
        if cl in {"m_full", "norm", "|m|", "magnitude"}:
            return r"$|m|$"
        return c

    component = str(meta.get("component", "")).strip().lower()
    if component in {"x", "y", "z"}:
        return rf"$m_{component}$"
    if component in {"norm", "|m|", "magnitude"}:
        return r"$|m|$"

    label = meta.get("magnetization_label")
    if isinstance(label, str) and label.strip():
        return label
    return r"$M$"


def resolve_loop_axis_labels(result) -> tuple[str, str]:
    """Resolve loop-axis labels from result metadata using math-style notation."""
    meta = getattr(result, "metadata", {}) or {}
    return _field_axis_label_from_metadata(meta), _magnetization_label_from_metadata(meta)


def _branch_color(explorer, branch_name: str) -> str:
    if not explorer.state.show_flags.get("branch_colors", True):
        return "#3b82f6"
    asc, desc = explorer.result.config.branch_colors
    return asc if branch_name == "ascending" else desc


def _field_axis_label(explorer) -> str:
    meta = getattr(explorer.result, "metadata", {}) or {}
    return _field_axis_label_from_metadata(meta)


def _magnetization_label(explorer) -> str:
    meta = getattr(explorer.result, "metadata", {}) or {}
    return _magnetization_label_from_metadata(meta)


def draw_loop_panel(explorer) -> None:
    """Render static loop traces and metric guides."""
    ax = explorer._ax_loop
    ax.clear()
    explorer._loop_points = []

    field = np.asarray(explorer.result.field, dtype=float)
    mag = np.asarray(explorer.result.magnetization, dtype=float)

    for branch in explorer.result.branches:
        x = field[branch.slice]
        y = mag[branch.slice]
        if x.size == 0:
            continue
        color = _branch_color(explorer, branch.name)
        ax.plot(
            x,
            y,
            color=color,
            lw=1.0,
            alpha=0.35,
            zorder=2,
        )
        points = ax.scatter(
            x,
            y,
            s=26,
            color=color,
            alpha=0.9,
            linewidths=0.0,
            zorder=4,
        )
        explorer._loop_points.append(points)

    if explorer.state.show_flags.get("hc", True):
        hc = explorer.result.metrics.coercive_field
        for h in (hc.hc_minus, hc.hc_plus):
            if np.isfinite(h):
                ax.axvline(float(h), ls="--", lw=1.1, color="#f97316", alpha=0.75)

    if explorer.state.show_flags.get("mr", True):
        mr = explorer.result.metrics.remanence
        for m in (mr.mr_minus, mr.mr_plus):
            if np.isfinite(m):
                ax.axhline(float(m), ls=":", lw=1.1, color="#a855f7", alpha=0.75)

    if explorer.state.show_flags.get("ms", False):
        ms = explorer.result.metrics.saturation_points
        if np.isfinite(ms.ms_positive):
            ax.scatter([ms.hs_positive], [ms.ms_positive], color="#22c55e", s=42, zorder=6)
        if np.isfinite(ms.ms_negative):
            ax.scatter([ms.hs_negative], [ms.ms_negative], color="#ef4444", s=42, zorder=6)

    ax.set_xlabel(_field_axis_label(explorer))
    ax.set_ylabel(_magnetization_label(explorer))
    ax.set_title("")
    ax.grid(True, alpha=0.25)

    explorer._loop_trail, = ax.plot([], [], color="#0ea5e9", lw=2.0, alpha=0.35)
    explorer._loop_marker = ax.scatter(
        [],
        [],
        s=140,
        color="#f59e0b",
        edgecolors="#111827",
        linewidths=1.2,
        zorder=7,
    )
    explorer._loop_arrow = ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "color": "#f59e0b", "lw": 1.5, "alpha": 0.9},
    )


def update_loop_cursor(explorer, *, redraw: bool = True) -> None:
    """Update loop marker, directional arrow and trail for current index."""
    field = np.asarray(explorer.result.field, dtype=float)
    mag = np.asarray(explorer.result.magnetization, dtype=float)
    n_points = field.size
    if n_points == 0:
        return

    idx = int(np.clip(explorer.state.current_idx, 0, n_points - 1))
    explorer.state.current_idx = idx

    x = float(field[idx])
    y = float(mag[idx])
    explorer.state.field_value = x
    explorer.state.magnetization_value = y

    explorer._loop_marker.set_offsets(np.array([[x, y]], dtype=float))

    if explorer.state.show_flags.get("trail", True):
        trail_len = int(max(1, explorer.result.config.trail_length))
        start = max(0, idx - trail_len)
        explorer._loop_trail.set_data(field[start : idx + 1], mag[start : idx + 1])
        explorer._loop_trail.set_alpha(float(explorer.result.config.trail_alpha_decay))
    else:
        explorer._loop_trail.set_data([], [])

    if explorer.state.show_flags.get("arrow", True) and n_points >= 2 and idx < (n_points - 1):
        next_idx = idx + 1
        explorer._loop_arrow.set_visible(True)
        explorer._loop_arrow.xy = (float(field[next_idx]), float(mag[next_idx]))
        explorer._loop_arrow.set_position((x, y))
    else:
        explorer._loop_arrow.set_visible(False)

    if redraw and explorer._fig is not None:
        explorer._fig.canvas.draw_idle()
