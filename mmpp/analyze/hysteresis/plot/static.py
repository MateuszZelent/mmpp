"""Static plotting helpers for hysteresis loops."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False


def plot_loop(
    result,
    *,
    ax: Any | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    show_markers: bool = False,
    show_hc: bool | None = None,
    show_mr: bool | None = None,
    show_ms: bool | None = None,
    show_branch_colors: bool | None = None,
    title: str | None = None,
):
    """Plot static hysteresis loop with optional metric markers."""
    if not _HAS_MPL:
        raise ImportError("Matplotlib is required for plotting")

    cfg = result.config
    show_hc = cfg.show_hc if show_hc is None else bool(show_hc)
    show_mr = cfg.show_mr if show_mr is None else bool(show_mr)
    show_ms = cfg.show_ms if show_ms is None else bool(show_ms)
    show_branch_colors = (
        cfg.show_branch_colors if show_branch_colors is None else bool(show_branch_colors)
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or cfg.figsize,
            dpi=cfg.dpi if dpi is None else dpi,
        )
    else:
        fig = ax.figure
        if dpi is not None:
            fig.set_dpi(dpi)

    field = np.asarray(result.field, dtype=float)
    mag = np.asarray(result.magnetization, dtype=float)

    c_asc, c_desc = cfg.branch_colors
    for branch in result.branches:
        color = c_asc if branch.name == "ascending" else c_desc
        if not show_branch_colors:
            color = "#3b82f6"
        x = field[branch.slice]
        y = mag[branch.slice]
        if x.size == 0:
            continue
        label = branch.name if branch is result.branches[0] or branch.name not in {"ascending", "descending"} else None
        ax.plot(x, y, color=color, lw=1.8, alpha=0.9, label=label)

    if show_markers and field.size:
        ax.scatter(
            [field[0], field[-1]],
            [mag[0], mag[-1]],
            s=float(cfg.marker_size) * 4,
            color=["#16a34a", "#dc2626"],
            zorder=5,
            label="start/end",
        )

    annotations: dict[str, Any] = {}

    if show_hc:
        hc = result.metrics.coercive_field
        hc_vals = [float(hc.hc_minus), float(hc.hc_plus)]
        hc_vals = [v for v in hc_vals if np.isfinite(v)]
        for h in hc_vals:
            ax.axvline(h, color="#f97316", ls="--", lw=1.2, alpha=0.8)
        annotations["hc"] = hc

    if show_mr:
        mr = result.metrics.remanence
        mr_vals = [float(mr.mr_minus), float(mr.mr_plus)]
        mr_vals = [v for v in mr_vals if np.isfinite(v)]
        for m in mr_vals:
            ax.axhline(m, color="#a855f7", ls=":", lw=1.2, alpha=0.75)
        annotations["mr"] = mr

    if show_ms:
        ms = result.metrics.saturation_points
        if np.isfinite(ms.ms_positive):
            ax.scatter(
                [ms.hs_positive],
                [ms.ms_positive],
                color="#22c55e",
                s=float(cfg.marker_size) * 5,
                zorder=6,
            )
        if np.isfinite(ms.ms_negative):
            ax.scatter(
                [ms.hs_negative],
                [ms.ms_negative],
                color="#ef4444",
                s=float(cfg.marker_size) * 5,
                zorder=6,
            )
        annotations["ms"] = ms

    field_unit = result.metadata.get("field_unit", "input")
    component = result.metadata.get("component")
    key_prefix = result.metadata.get("key_prefix", "")
    # build axis labels from metadata
    field_label = key_prefix.rstrip("-_ ") if key_prefix else "Field"
    ax.set_xlabel(f"{field_label} [{field_unit}]")
    if component:
        comp_map = {"x": "mₓ", "y": "m_y", "z": "m_z", "norm": "|m|"}
        mag_label = f"⟨{comp_map.get(component, component)}⟩  (spatial mean)"
    else:
        mag_label = "Magnetization"
    ax.set_ylabel(mag_label)
    ax.set_title(title or "Hysteresis Loop")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # deduplicate labels while preserving order
        seen = set()
        uniq_h = []
        uniq_l = []
        for handle, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            uniq_h.append(handle)
            uniq_l.append(label)
        ax.legend(uniq_h, uniq_l, frameon=False, fontsize=9)

    fig.tight_layout()
    return fig, ax, annotations
