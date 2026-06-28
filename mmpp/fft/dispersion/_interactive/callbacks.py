"""Callbacks for interactive dispersion explorer."""

from __future__ import annotations

from typing import Any

from .rendering import draw_dispersion_panel, refresh_output_widget
from .status import set_status


def on_display_change(explorer: Any) -> None:
    """Apply current widget values to state and redraw."""
    controls = explorer.controls
    if not controls:
        return
    explorer.state.fmin_ghz = float(controls["fmin"].value)
    explorer.state.fmax_ghz = float(controls["fmax"].value)
    explorer.state.source = str(controls["source"].value)
    explorer.state.kscale = str(controls["kscale"].value)
    explorer.state.cmap = str(controls["cmap"].value)
    explorer.state.positive_frequencies = bool(controls["positive"].value)
    explorer.state.lognorm = bool(controls["lognorm"].value)
    for key in ["grid", "selection", "notes"]:
        if key in controls and explorer.state.show_flags is not None:
            explorer.state.show_flags[key] = bool(controls[key].value)
    draw_dispersion_panel(explorer)
    refresh_output_widget(explorer)
    set_status(
        explorer,
        (
            f"range={explorer.state.fmin_ghz:.4g}.."
            f"{explorer.state.fmax_ghz:.4g} GHz, "
            f"source={explorer.state.source}, k={explorer.state.kscale}"
        ),
        color="#0F766E",
    )


def on_canvas_click(explorer: Any, event: Any) -> None:
    """Record selected (k, f) point from a Matplotlib click."""
    if event is None or event.inaxes is not explorer.axes:
        return
    if event.xdata is None or event.ydata is None:
        return

    import numpy as np

    k_value = float(event.xdata)
    if explorer.state.kscale == "rad_um":
        k_value *= 1e6
    elif explorer.state.kscale in {"cycles_m", "meter"}:
        k_value *= 2 * np.pi
    f_value = float(event.ydata) * 1e9

    explorer.state.selected_k = k_value
    explorer.state.selected_f = f_value
    explorer.state.selected_power = None

    try:
        spectrum, k_axis, f_axis = explorer.result.frequency_view(
            positive_frequencies=bool(explorer.state.positive_frequencies),
            analysis_source=str(explorer.state.source),
        )
        ik = int(np.argmin(np.abs(np.asarray(k_axis) - k_value)))
        jf = int(np.argmin(np.abs(np.asarray(f_axis) - f_value)))
        explorer.state.selected_power = float(spectrum[ik, jf])
    except Exception:
        pass

    if "selection_info" in explorer.controls:
        power = explorer.state.selected_power
        power_text = "?" if power is None else f"{power:.4g}"
        explorer.controls["selection_info"].value = (
            "<small>"
            f"k={k_value / 1e6:.4g} rad/um, "
            f"f={f_value / 1e9:.4g} GHz, S={power_text}"
            "</small>"
        )
    draw_dispersion_panel(explorer)
    refresh_output_widget(explorer)
    set_status(
        explorer,
        f"selected k={k_value / 1e6:.4g} rad/um, f={f_value / 1e9:.4g} GHz",
        color="#0369a1",
    )
