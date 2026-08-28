"""Callbacks for interactive dispersion explorer."""

from __future__ import annotations

from typing import Any

from .rendering import draw_dispersion_panel, refresh_output_widget
from .status import set_status

_ANALYTICAL_MATERIAL_KEYS = ("B", "Ms", "Aex", "d", "Ku", "Kc1", "Kc2", "phi_ani", "g")
_ANALYTICAL_MAPPED_MATERIAL_KEYS = {"phi": "analytical_phi", "D": "analytical_D"}


def _positive_int(value: Any, default: int) -> int:
    """Convert widget numeric values to a positive integer."""
    try:
        converted = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(1, converted)


def _optional_float(value: Any) -> float | None:
    """Convert optional widget numeric values to float."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    converted = _optional_float(value)
    return float(default) if converted is None else float(converted)


def sync_analytical_options(explorer: Any) -> None:
    """Mirror analytical overlay state into renderer options."""
    analytical = dict(getattr(explorer.state, "analytical", None) or {})
    normalized = {
        "enabled": bool(analytical.get("enabled", False)),
        "model": str(analytical.get("model") or "kalinikos"),
        "sw_config": str(analytical.get("sw_config") or "DE"),
        "n_modes": _positive_int(analytical.get("n_modes"), 1),
        "k_points": _positive_int(analytical.get("k_points"), 500),
    }
    for key in _ANALYTICAL_MATERIAL_KEYS:
        value = _optional_float(analytical.get(key))
        if value is not None:
            normalized[key] = value
    for key in _ANALYTICAL_MAPPED_MATERIAL_KEYS:
        value = _optional_float(analytical.get(key))
        if value is not None:
            normalized[key] = value
    explorer.state.analytical = normalized

    options = getattr(explorer, "options", None)
    if not isinstance(options, dict):
        return
    options["analytical"] = normalized["sw_config"] if normalized["enabled"] else False
    options["analytical_model"] = normalized["model"]
    options["analytical_n_modes"] = normalized["n_modes"]
    options["analytical_k_points"] = normalized["k_points"]
    for key in _ANALYTICAL_MATERIAL_KEYS:
        if key in normalized:
            options[key] = normalized[key]
    for key, target_key in _ANALYTICAL_MAPPED_MATERIAL_KEYS.items():
        if key in normalized:
            options[target_key] = normalized[key]


def _update_analytical_state_from_controls(explorer: Any) -> None:
    controls = explorer.controls
    analytical_keys = {
        "analytical_enabled",
        "analytical_model",
        "analytical_sw_config",
        "analytical_n_modes",
        "analytical_k_points",
        "analytical_B",
        "analytical_Ms",
        "analytical_Aex",
        "analytical_d",
        "analytical_Ku",
        "analytical_Kc1",
        "analytical_Kc2",
        "analytical_phi",
        "analytical_phi_ani",
        "analytical_D",
        "analytical_g",
    }
    if not any(key in controls for key in analytical_keys):
        return

    analytical = dict(getattr(explorer.state, "analytical", None) or {})
    if "analytical_enabled" in controls:
        analytical["enabled"] = bool(controls["analytical_enabled"].value)
    if "analytical_model" in controls:
        analytical["model"] = str(controls["analytical_model"].value)
    if "analytical_sw_config" in controls:
        analytical["sw_config"] = str(controls["analytical_sw_config"].value)
    if "analytical_n_modes" in controls:
        analytical["n_modes"] = _positive_int(controls["analytical_n_modes"].value, 1)
    if "analytical_k_points" in controls:
        analytical["k_points"] = _positive_int(
            controls["analytical_k_points"].value, 500
        )
    for key in _ANALYTICAL_MATERIAL_KEYS:
        control_key = f"analytical_{key}"
        if control_key in controls:
            value = _optional_float(controls[control_key].value)
            if value is not None:
                analytical[key] = value
    for key in _ANALYTICAL_MAPPED_MATERIAL_KEYS:
        control_key = f"analytical_{key}"
        if control_key in controls:
            value = _optional_float(controls[control_key].value)
            if value is not None:
                analytical[key] = value
    explorer.state.analytical = analytical


def _update_live_filter_state_from_controls(explorer: Any) -> None:
    """Mirror live-filter controls into the shared viewer state."""
    controls = explorer.controls
    live: dict[str, Any] = {}

    if controls.get("filter_snr_enabled") is not None and bool(
        controls["filter_snr_enabled"].value
    ):
        live["snr_filter"] = {
            "enabled": True,
            "threshold_snr": _float_or_default(
                controls["filter_snr_threshold"].value,
                3.0,
            ),
            "method": "percentile",
            "noise_percentile": 5.0,
        }

    if controls.get("filter_gaussian_enabled") is not None and bool(
        controls["filter_gaussian_enabled"].value
    ):
        live["gaussian_morph"] = {
            "enabled": True,
            "sigma_f": _float_or_default(
                controls["filter_gaussian_sigma_f"].value,
                1.0,
            ),
            "sigma_k": _float_or_default(
                controls["filter_gaussian_sigma_k"].value,
                1.0,
            ),
            "threshold_std": _float_or_default(
                controls["filter_gaussian_threshold"].value,
                1.5,
            ),
            "opening_size": 3,
        }

    if controls.get("filter_percentile_enabled") is not None and bool(
        controls["filter_percentile_enabled"].value
    ):
        live["percentile_autoscale"] = {
            "enabled": True,
            "low_percentile": _float_or_default(
                controls["filter_percentile_low"].value,
                2.0,
            ),
            "high_percentile": _float_or_default(
                controls["filter_percentile_high"].value,
                99.0,
            ),
        }

    if controls.get("filter_soft_enabled") is not None and bool(
        controls["filter_soft_enabled"].value
    ):
        live["soft_threshold"] = {
            "enabled": True,
            "threshold_percentile": _float_or_default(
                controls["filter_soft_percentile"].value,
                50.0,
            ),
            "smoothness": _float_or_default(
                controls["filter_soft_smoothness"].value,
                5.0,
            ),
        }

    if controls.get("filter_log_enabled") is not None and bool(
        controls["filter_log_enabled"].value
    ):
        live["log_transform"] = {
            "enabled": True,
            "method": str(controls["filter_log_method"].value),
            "scale": 1.0,
            "floor_percentile": 1.0,
        }

    if controls.get("filter_gamma_enabled") is not None and bool(
        controls["filter_gamma_enabled"].value
    ):
        live["gamma"] = {
            "enabled": True,
            "gamma": _float_or_default(controls["filter_gamma_value"].value, 0.5),
        }

    explorer.state.live_filters = live or None
    options = getattr(explorer, "options", None)
    if isinstance(options, dict):
        if live:
            options["live_filters"] = live
        else:
            options.pop("live_filters", None)


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
    if "mode_type" in controls:
        explorer.state.mode_type = str(controls["mode_type"].value)
        options = getattr(explorer, "options", None)
        if isinstance(options, dict):
            options["mode_type"] = explorer.state.mode_type
    for key in ["grid", "selection", "notes"]:
        if key in controls and explorer.state.show_flags is not None:
            explorer.state.show_flags[key] = bool(controls[key].value)
    _update_analytical_state_from_controls(explorer)
    sync_analytical_options(explorer)
    _update_live_filter_state_from_controls(explorer)
    if hasattr(explorer, "refresh_auxiliary_panels"):
        explorer.refresh_auxiliary_panels()
    if bool(getattr(explorer, "options", {}).get("auto_render", True)):
        if hasattr(explorer, "ensure_figure"):
            explorer.ensure_figure()
        draw_dispersion_panel(explorer)
        refresh_output_widget(explorer)
        render_note = "rendered"
    else:
        render_note = "press Render / refresh dispersion to update plot"
    set_status(
        explorer,
        (
            f"range={explorer.state.fmin_ghz:.4g}.."
            f"{explorer.state.fmax_ghz:.4g} GHz, "
            f"source={explorer.state.source}, k={explorer.state.kscale}; "
            f"{render_note}"
        ),
        color="#0F766E",
    )


def _selected_mode_request_from_explorer(explorer: Any) -> dict[str, Any]:
    """Build a mode reconstruction request from the current heatmap selection."""
    selected_k = getattr(explorer.state, "selected_k", None)
    selected_f = getattr(explorer.state, "selected_f", None)
    request = {
        "available": False,
        "k_rad_um": None,
        "f_ghz": None,
        "z_layer": 0,
        "component": getattr(explorer.result, "component", None),
        "reason": "",
    }
    controls = getattr(explorer, "controls", {})
    if "mode_z_layer" in controls:
        try:
            request["z_layer"] = max(0, int(float(controls["mode_z_layer"].value)))
        except (TypeError, ValueError):
            request["z_layer"] = 0
    if "mode_component" in controls:
        request["component"] = str(controls["mode_component"].value)
    if selected_k is None or selected_f is None:
        request["reason"] = "Select a point on S(k, f) first."
        return request
    request["k_rad_um"] = float(selected_k) / 1e6
    request["f_ghz"] = float(selected_f) / 1e9
    if getattr(explorer.result, "S_complex", None) is None:
        request["reason"] = "Mode reconstruction requires S_complex."
        return request
    request["available"] = True
    return request


def _selected_mode_type(explorer: Any) -> str:
    controls = getattr(explorer, "controls", {})
    if "mode_type" in controls:
        return str(getattr(controls["mode_type"], "value", "abs") or "abs")
    state = getattr(explorer, "state", None)
    if state is not None:
        return str(getattr(state, "mode_type", None) or "abs")
    options = getattr(explorer, "options", {})
    if isinstance(options, dict):
        return str(options.get("mode_type") or "abs")
    return "abs"


def _render_extracted_mode(explorer: Any, mode: Any) -> bool:
    """Render the extracted mode in the main output panel."""
    output = explorer.controls.get("output") if explorer.controls else None
    if output is None:
        return False
    if not hasattr(output, "clear_output") or not hasattr(
        output, "append_display_data"
    ):
        return False

    mode_type = _selected_mode_type(explorer)
    cmap = "hsv" if mode_type == "phase" else "RdBu_r"
    title = (
        f"Mode {mode_type} @ k={float(mode.k_rad_um):.4g} rad/um, "
        f"f={float(mode.f_ghz):.4g} GHz"
    )
    previous = getattr(explorer, "_mode_figure", None)
    if previous is not None:
        try:
            import matplotlib.pyplot as plt

            plt.close(previous)
        except Exception:
            pass
    output.clear_output(wait=False)
    try:
        fig, _ax = mode.plot.imshow(mode_type=mode_type, cmap=cmap, title=title)
        explorer._mode_figure = fig
        output.append_display_data(fig)
        return True
    except Exception:
        output.append_display_data(mode)
        return False


def on_mode_extract(explorer: Any) -> None:
    """Extract a mode for the currently selected dispersion point."""
    request = _selected_mode_request_from_explorer(explorer)
    if not request["available"]:
        if "mode_info" in explorer.controls:
            explorer.controls["mode_info"].value = f"<small>{request['reason']}</small>"
        set_status(explorer, str(request["reason"]), color="crimson")
        return

    try:
        mode = explorer.result.modes.at(
            k_rad_um=float(request["k_rad_um"]),
            f_ghz=float(request["f_ghz"]),
            z_layer=int(request["z_layer"]),
            component=request["component"],
        )
    except Exception as exc:
        if "mode_info" in explorer.controls:
            explorer.controls[
                "mode_info"
            ].value = (
                f"<small>Mode extraction failed: {type(exc).__name__}: {exc}</small>"
            )
        set_status(explorer, f"Mode extraction failed: {exc}", color="crimson")
        return

    explorer.last_mode = mode
    rendered = _render_extracted_mode(explorer, mode)
    if "mode_info" in explorer.controls:
        explorer.controls["mode_info"].value = (
            "<small>"
            f"k={float(request['k_rad_um']):.4g} rad/um, "
            f"f={float(request['f_ghz']):.4g} GHz, "
            f"component={request['component']}, "
            f"z_layer={int(request['z_layer'])}, "
            f"mode_type={_selected_mode_type(explorer)}"
            "</small>"
        )
    set_status(
        explorer,
        (
            f"mode extracted at k={float(request['k_rad_um']):.4g} rad/um, "
            f"f={float(request['f_ghz']):.4g} GHz"
            + (", visualized in main panel" if rendered else ", metadata shown")
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
    if hasattr(explorer, "refresh_auxiliary_panels"):
        explorer.refresh_auxiliary_panels()
    set_status(
        explorer,
        f"selected k={k_value / 1e6:.4g} rad/um, f={f_value / 1e9:.4g} GHz",
        color="#0369a1",
    )
