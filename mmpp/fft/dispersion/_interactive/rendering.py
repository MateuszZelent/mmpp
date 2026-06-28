"""Rendering helpers for interactive dispersion explorer."""

from __future__ import annotations

from typing import Any


def _scaled_k_axis(k_axis: Any, kscale: str) -> tuple[Any, str]:
    import numpy as np

    if kscale == "rad_um":
        return k_axis / 1e6, "k [rad/um]"
    if kscale in {"cycles_m", "meter"}:
        return k_axis / (2 * np.pi), "k [1/m]"
    return k_axis, "k [rad/m]"


def _norm(explorer: Any, spectrum: Any) -> Any:
    if not bool(explorer.state.lognorm):
        return None
    try:
        import numpy as np
        from matplotlib.colors import LogNorm

        positive_values = spectrum[spectrum > 0]
        if positive_values.size:
            return LogNorm(
                vmin=float(np.min(positive_values)),
                vmax=float(np.max(positive_values)),
            )
    except Exception:
        return None
    return None


def draw_dispersion_panel(explorer: Any) -> None:
    """Render ``S(k, f)`` into the explorer axes."""
    if explorer.axes is None or explorer.figure is None:
        return

    import numpy as np

    ax = explorer.axes
    ax.clear()

    spectrum, k_axis, f_axis = explorer.result.frequency_view(
        positive_frequencies=bool(explorer.state.positive_frequencies),
        analysis_source=str(explorer.state.source),
    )
    spectrum = np.asarray(spectrum, dtype=float)
    k_axis = np.asarray(k_axis, dtype=float)
    f_axis = np.asarray(f_axis, dtype=float)

    f_mask = np.ones_like(f_axis, dtype=bool)
    if explorer.state.positive_frequencies:
        f_mask &= f_axis >= 0
    f_mask &= f_axis >= float(explorer.state.fmin_ghz) * 1e9
    f_mask &= f_axis <= float(explorer.state.fmax_ghz) * 1e9
    if np.any(f_mask):
        spectrum = spectrum[:, f_mask]
        f_axis = f_axis[f_mask]

    k_plot, k_label = _scaled_k_axis(k_axis, str(explorer.state.kscale))
    f_plot = f_axis / 1e9
    if k_plot.size == 0 or f_plot.size == 0:
        ax.text(0.5, 0.5, "No dispersion data in selected range", ha="center")
        return

    image = ax.imshow(
        spectrum.T,
        aspect="auto",
        origin="lower",
        extent=(
            float(k_plot[0]),
            float(k_plot[-1]),
            float(f_plot[0]),
            float(f_plot[-1]),
        ),
        cmap=str(explorer.state.cmap),
        norm=_norm(explorer, spectrum),
    )
    explorer._image = image

    if explorer.state.show_flags and explorer.state.show_flags.get("grid", True):
        if hasattr(ax, "grid"):
            ax.grid(True, alpha=0.22, linestyle=":")
    if (
        explorer.state.show_flags
        and explorer.state.show_flags.get("selection", True)
        and explorer.state.selected_k is not None
        and explorer.state.selected_f is not None
    ):
        k_sel = float(explorer.state.selected_k)
        f_sel = float(explorer.state.selected_f) / 1e9
        if explorer.state.kscale == "rad_um":
            k_sel /= 1e6
        elif explorer.state.kscale in {"cycles_m", "meter"}:
            k_sel /= 2 * np.pi
        if hasattr(ax, "plot"):
            ax.plot(
                [k_sel],
                [f_sel],
                "o",
                color="#ef4444",
                markerfacecolor="none",
                markeredgewidth=1.8,
                markersize=8,
            )

    ax.set_title(f"Dispersion S(k, f) - {explorer.state.source}")
    ax.set_xlabel(k_label)
    ax.set_ylabel("Frequency [GHz]")
    explorer.figure.canvas.draw_idle()


def refresh_output_widget(explorer: Any) -> None:
    """Render current figure into output widget, matching hysteresis explorer."""
    output = explorer.controls.get("output") if explorer.controls else None
    if output is None:
        return
    if not hasattr(output, "clear_output") or not hasattr(output, "append_display_data"):
        return
    output.clear_output(wait=False)
    if explorer.figure is not None:
        explorer.figure.canvas.draw()
        output.append_display_data(explorer.figure)
