"""Rendering helpers for interactive hysteresis explorer."""

from __future__ import annotations

import numpy as np


def _branch_color(explorer, branch_name: str) -> str:
    if not explorer.state.show_flags.get("branch_colors", True):
        return "#3b82f6"
    asc, desc = explorer.result.config.branch_colors
    return asc if branch_name == "ascending" else desc


def draw_loop_panel(explorer) -> None:
    """Render static loop traces and metric guides."""
    ax = explorer._ax_loop
    ax.clear()

    field = np.asarray(explorer.result.field, dtype=float)
    mag = np.asarray(explorer.result.magnetization, dtype=float)

    for branch in explorer.result.branches:
        x = field[branch.slice]
        y = mag[branch.slice]
        if x.size == 0:
            continue
        ax.plot(
            x,
            y,
            color=_branch_color(explorer, branch.name),
            lw=1.8,
            alpha=0.9,
        )

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

    ax.set_xlabel(f"Field [{explorer.result.metadata.get('field_unit', 'input')}]")
    ax.set_ylabel("Magnetization")
    ax.set_title("Hysteresis loop")
    ax.grid(True, alpha=0.25)

    explorer._loop_trail, = ax.plot([], [], color="#0ea5e9", lw=2.0, alpha=0.35)
    explorer._loop_marker = ax.scatter([], [], s=90, color="#111827", zorder=7)
    explorer._loop_arrow = ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "color": "#111827", "lw": 1.4, "alpha": 0.8},
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

    if explorer.state.show_flags.get("arrow", True) and n_points >= 2:
        next_idx = (idx + 1) % n_points
        explorer._loop_arrow.set_visible(True)
        explorer._loop_arrow.xy = (float(field[next_idx]), float(mag[next_idx]))
        explorer._loop_arrow.set_position((x, y))
    else:
        explorer._loop_arrow.set_visible(False)

    if redraw and explorer._fig is not None:
        explorer._fig.canvas.draw_idle()
