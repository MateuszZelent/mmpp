"""Widget toolbar for interactive dispersion explorer."""

from __future__ import annotations

import json
from html import escape
from typing import Any

from .callbacks import on_display_change, on_mode_extract, sync_analytical_options
from .frequency import normalize_frequency_window_ghz
from .presets import list_presets, load_preset, save_preset
from .rendering import draw_dispersion_panel, refresh_output_widget
from .status import set_status
from .._json import json_safe


def _layout(widgets: Any, **kwargs: Any) -> Any:
    layout_cls = getattr(widgets, "Layout", None)
    return layout_cls(**kwargs) if layout_cls is not None else None


def _maybe_layout(widgets: Any, **kwargs: Any) -> dict[str, Any]:
    layout = _layout(widgets, **kwargs)
    return {"layout": layout} if layout is not None else {}


def _make_button(
    widgets: Any,
    description: str,
    *,
    width: str = "100%",
    color: str = "#0f766e",
    button_style: str = "primary",
    tooltip: str | None = None,
) -> Any:
    """Create a high-contrast action button that stays visible in notebooks."""
    button_cls = getattr(widgets, "Button", None)
    if button_cls is None:
        return widgets.HTML(
            value=(
                "<div style='box-sizing:border-box;width:100%;min-height:34px;"
                "margin:3px 0;padding:7px 10px;border-radius:6px;"
                "background:#334155;color:#fff;font-weight:700;text-align:center;'>"
                f"{escape(description)}"
                "</div>"
            )
        )
    button = button_cls(
        description=description,
        button_style=button_style,
        tooltip=tooltip or description,
        **_maybe_layout(widgets, width=width, height="34px", margin="3px 0px"),
    )
    try:
        button.style.button_color = color
        button.style.text_color = "#ffffff"
        button.style.font_weight = "700"
    except Exception:
        pass
    return button


def _render_current_dispersion(explorer: Any) -> None:
    """Render the current heatmap with visible status and error reporting."""
    try:
        set_status(explorer, "Rendering dispersion heatmap...", color="#334155")
        if hasattr(explorer, "render"):
            explorer.render()
        else:
            if hasattr(explorer, "ensure_figure"):
                explorer.ensure_figure()
            draw_dispersion_panel(explorer)
            refresh_output_widget(explorer)
        explorer._has_rendered_dispersion = True
        _refresh_auxiliary_panels(explorer, update_status=False)
        set_status(explorer, "Dispersion heatmap rendered", color="#0F766E")
    except Exception as exc:
        set_status(
            explorer,
            f"Dispersion render failed: {type(exc).__name__}: {exc}",
            color="crimson",
        )


def _show_dispersion_placeholder(explorer: Any) -> None:
    """Show a lightweight placeholder before the first Matplotlib render."""
    output = explorer.controls.get("output") if explorer.controls else None
    placeholder = explorer.controls.get("output_placeholder") if explorer.controls else None
    if output is None or placeholder is None:
        return
    if not hasattr(output, "clear_output") or not hasattr(output, "append_display_data"):
        return
    output.clear_output(wait=False)
    output.append_display_data(placeholder)


def _selection_payload(explorer: Any) -> dict[str, Any]:
    """Return current selected point in JSON-friendly display and raw units."""
    selected_k = getattr(explorer.state, "selected_k", None)
    selected_f = getattr(explorer.state, "selected_f", None)
    selected_power = getattr(explorer.state, "selected_power", None)
    payload: dict[str, Any] = {}
    if selected_k is not None:
        k_rad_per_m = float(selected_k)
        payload["k_rad_per_m"] = k_rad_per_m
        payload["k_rad_um"] = k_rad_per_m / 1e6
    if selected_f is not None:
        f_hz = float(selected_f)
        payload["f_hz"] = f_hz
        payload["f_ghz"] = f_hz / 1e9
    if selected_power is not None:
        payload["power"] = float(selected_power)
    return payload


def _mode_request_payload(explorer: Any, selection: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight mode request matching the public viewer shape."""
    request = {
        "available": False,
        "k_rad_um": selection.get("k_rad_um"),
        "f_ghz": selection.get("f_ghz"),
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
    if request["k_rad_um"] is None or request["f_ghz"] is None:
        request["reason"] = "Select a dispersion point with k and f first."
        return request
    if getattr(explorer.result, "S_complex", None) is None:
        request["reason"] = "Mode reconstruction requires S_complex."
        return request
    request["available"] = True
    return request


def _axis_range(values: Any, *, scale: float = 1.0) -> tuple[float | None, float | None]:
    try:
        converted = [float(value) / scale for value in values]
    except Exception:
        return None, None
    if not converted:
        return None, None
    return min(converted), max(converted)


def _analysis_summary_payload(explorer: Any) -> dict[str, Any]:
    """Return a lightweight analysis summary without touching Matplotlib."""
    result = explorer.result
    k_min, k_max = _axis_range(getattr(result, "k_axis", []), scale=1e6)
    f_min, f_max = _axis_range(getattr(result, "f_axis", []), scale=1e9)
    selection = _selection_payload(explorer)
    live_filters = getattr(explorer.state, "live_filters", None) or {}
    return {
        "shape": list(getattr(result, "shape", [])),
        "axis": getattr(result, "axis", None),
        "component": getattr(result, "component", None),
        "k_range_rad_um": [k_min, k_max],
        "f_range_ghz": [f_min, f_max],
        "display_window_ghz": [
            float(getattr(explorer.state, "fmin_ghz", 0.0)),
            float(getattr(explorer.state, "fmax_ghz", 0.0)),
        ],
        "selection": selection,
        "live_filters": sorted(str(key) for key in live_filters.keys()),
        "live_filter_error": getattr(explorer, "_last_filter_error", ""),
        "has_complex_modes": getattr(result, "S_complex", None) is not None,
        "rendered": bool(getattr(explorer, "_has_rendered_dispersion", False)),
        "backend": (
            explorer.diagnostics().get("backend")
            if hasattr(explorer, "diagnostics")
            else "unknown"
        ),
    }


def _refresh_analysis_summary(explorer: Any, *, update_status: bool = True) -> None:
    """Refresh lightweight analysis diagnostics in the Analysis tab."""
    payload = json_safe(_analysis_summary_payload(explorer))
    rows = "".join(
        "<tr>"
        f"<td style='padding:2px 8px;color:#475569;font-weight:600;'>{escape(str(key))}</td>"
        f"<td style='padding:2px 8px;font-family:monospace;color:#0f172a;'>{escape(str(value))}</td>"
        "</tr>"
        for key, value in payload.items()
    )
    if "analysis_summary" in explorer.controls:
        explorer.controls["analysis_summary"].value = (
            "<table style='border-collapse:collapse;font-size:12px;'>"
            f"{rows}"
            "</table>"
        )
    if update_status:
        set_status(explorer, "Analysis summary refreshed", color="#0F766E")


def _export_snapshot(explorer: Any, *, update_status: bool = True) -> None:
    """Show a compact JSON snapshot of current viewer state."""
    selection = _selection_payload(explorer)
    payload = {
        "viewer": explorer.collect_preset(),
        "selection": selection,
        "mode_request": _mode_request_payload(explorer, selection),
        "diagnostics": explorer.diagnostics() if hasattr(explorer, "diagnostics") else {},
    }
    text = json.dumps(json_safe(payload), indent=2, sort_keys=True)
    if "export_snapshot" in explorer.controls:
        explorer.controls["export_snapshot"].value = (
            "<pre style='max-height:260px;overflow:auto;font-size:11px;"
            "line-height:1.25;background:#0f172a;color:#dbeafe;"
            "padding:8px;border-radius:6px;'>"
            f"{escape(text)}"
            "</pre>"
        )
    if update_status:
        set_status(explorer, "Export snapshot refreshed", color="#0F766E")


def _refresh_auxiliary_panels(explorer: Any, *, update_status: bool = False) -> None:
    """Refresh passive analysis/export panels without forcing a Matplotlib render."""
    _refresh_analysis_summary(explorer, update_status=update_status)
    _export_snapshot(explorer, update_status=update_status)


def build_toolbar(
    explorer: Any,
    widgets_module: Any,
    *,
    render_initial: bool = False,
) -> None:
    """Build ipywidgets toolbar controls and wire callbacks."""
    widgets = widgets_module
    controls: dict[str, Any] = {}

    fmin, fmax = normalize_frequency_window_ghz(
        explorer.options,
        getattr(explorer.result, "f_axis", None),
    )
    explorer.state.fmin_ghz = float(fmin)
    explorer.state.fmax_ghz = float(fmax)

    controls["fmin"] = widgets.FloatText(
        value=float(explorer.state.fmin_ghz),
        description="f min [GHz]",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["fmax"] = widgets.FloatText(
        value=float(explorer.state.fmax_ghz),
        description="f max [GHz]",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["source"] = widgets.Dropdown(
        options=["display", "raw"],
        value=str(explorer.state.source),
        description="source",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["render_dispersion"] = _make_button(
        widgets,
        "Render heatmap",
        color="#0f766e",
        button_style="success",
        tooltip="Render or refresh S(k, f) with the current display settings.",
    )
    controls["kscale"] = widgets.Dropdown(
        options=[
            ("rad/um", "rad_um"),
            ("rad/m", "rad"),
            ("1/m", "cycles_m"),
        ],
        value=str(explorer.state.kscale),
        description="k",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["cmap"] = widgets.Dropdown(
        options=["viridis", "cmc.davos", "plasma", "cividis", "turbo", "inferno", "magma"],
        value=str(explorer.state.cmap),
        description="cmap",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["positive"] = widgets.Checkbox(
        value=bool(explorer.state.positive_frequencies),
        description="f >= 0",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    controls["lognorm"] = widgets.Checkbox(
        value=bool(explorer.state.lognorm),
        description="log scale",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    live_filters = dict(getattr(explorer.state, "live_filters", None) or {})
    snr_cfg = dict(live_filters.get("snr_filter") or {})
    gaussian_cfg = dict(live_filters.get("gaussian_morph") or {})
    percentile_cfg = dict(live_filters.get("percentile_autoscale") or {})
    soft_cfg = dict(live_filters.get("soft_threshold") or {})
    log_cfg = dict(live_filters.get("log_transform") or {})
    gamma_cfg = dict(live_filters.get("gamma") or {})
    controls["filter_snr_enabled"] = widgets.Checkbox(
        value=bool(snr_cfg.get("enabled", False)),
        description="SNR filter",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_snr_threshold"] = widgets.FloatText(
        value=float(snr_cfg.get("threshold_snr", 3.0)),
        description="SNR",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_gaussian_enabled"] = widgets.Checkbox(
        value=bool(gaussian_cfg.get("enabled", False)),
        description="gaussian enhance",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_gaussian_sigma_f"] = widgets.FloatText(
        value=float(gaussian_cfg.get("sigma_f", 1.0)),
        description="sigma f",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_gaussian_sigma_k"] = widgets.FloatText(
        value=float(gaussian_cfg.get("sigma_k", 1.0)),
        description="sigma k",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_gaussian_threshold"] = widgets.FloatText(
        value=float(gaussian_cfg.get("threshold_std", 1.5)),
        description="threshold",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_percentile_enabled"] = widgets.Checkbox(
        value=bool(percentile_cfg.get("enabled", False)),
        description="percentile autoscale",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_percentile_low"] = widgets.FloatText(
        value=float(percentile_cfg.get("low_percentile", 2.0)),
        description="low %",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_percentile_high"] = widgets.FloatText(
        value=float(percentile_cfg.get("high_percentile", 99.0)),
        description="high %",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_soft_enabled"] = widgets.Checkbox(
        value=bool(soft_cfg.get("enabled", False)),
        description="soft threshold",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_soft_percentile"] = widgets.FloatText(
        value=float(soft_cfg.get("threshold_percentile", 50.0)),
        description="percentile",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_soft_smoothness"] = widgets.FloatText(
        value=float(soft_cfg.get("smoothness", 5.0)),
        description="smooth",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_log_enabled"] = widgets.Checkbox(
        value=bool(log_cfg.get("enabled", False)),
        description="log transform",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    log_method_value = str(log_cfg.get("method", "log1p"))
    log_method_options = ["log1p", "log10", "asinh", "log"]
    if log_method_value not in log_method_options:
        log_method_options.append(log_method_value)
    controls["filter_log_method"] = widgets.Dropdown(
        options=log_method_options,
        value=log_method_value,
        description="log",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_gamma_enabled"] = widgets.Checkbox(
        value=bool(gamma_cfg.get("enabled", False)),
        description="gamma",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_gamma_value"] = widgets.FloatText(
        value=float(gamma_cfg.get("gamma", 0.5)),
        description="gamma",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["filter_info"] = widgets.HTML(
        value=(
            "<small>Live filters are non-destructive and render from cached "
            "S(k, f); compute-stage filters still belong to "
            "<code>disp.filters(...)</code>.</small>"
        )
    )
    controls["grid"] = widgets.Checkbox(
        value=bool((explorer.state.show_flags or {}).get("grid", True)),
        description="grid",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    controls["selection"] = widgets.Checkbox(
        value=bool((explorer.state.show_flags or {}).get("selection", True)),
        description="selection",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    controls["notes"] = widgets.Checkbox(
        value=bool((explorer.state.show_flags or {}).get("notes", True)),
        description="notes",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    analytical = dict(explorer.state.analytical or {})
    sw_config_options = ["DE", "BV", "FV"]
    sw_config_value = str(analytical.get("sw_config") or "DE")
    if sw_config_value not in sw_config_options:
        sw_config_options.append(sw_config_value)
    model_options = [
        "kalinikos",
        "damon_eshbach",
        "backward_volume",
        "forward_volume",
        "bottcher",
        "kim",
        "cortes_ortuno",
    ]
    model_value = str(analytical.get("model") or "kalinikos")
    if model_value not in model_options:
        model_options.append(model_value)
    controls["analytical_enabled"] = widgets.Checkbox(
        value=bool(analytical.get("enabled", False)),
        description="analytical overlay",
        indent=False,
        **_maybe_layout(widgets, width="100%"),
    )
    controls["analytical_sw_config"] = widgets.Dropdown(
        options=sw_config_options,
        value=sw_config_value,
        description="geometry",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["analytical_model"] = widgets.Dropdown(
        options=model_options,
        value=model_value,
        description="model",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["analytical_n_modes"] = widgets.FloatText(
        value=float(analytical.get("n_modes") or 1),
        description="branches",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["analytical_k_points"] = widgets.FloatText(
        value=float(analytical.get("k_points") or 500),
        description="k points",
        **_maybe_layout(widgets, width="100%"),
    )
    material_defaults = {
        "B": ("B [T]", None),
        "Ms": ("Ms [A/m]", None),
        "Aex": ("Aex [J/m]", None),
        "d": ("d [m]", None),
        "phi": ("phi [rad]", None),
        "D": ("D [J/m2]", None),
    }
    material_text_cls = getattr(widgets, "Text", None)
    material_cls = material_text_cls if material_text_cls is not None else widgets.FloatText
    for key, (description, default_value) in material_defaults.items():
        value = analytical.get(key, default_value)
        controls[f"analytical_{key}"] = material_cls(
            value="" if value is None else str(value),
            description=description,
            **_maybe_layout(widgets, width="100%"),
        )
    controls["selection_info"] = widgets.HTML(value="<small>No point selected</small>")
    mode_components = list(explorer.options.get("mode_components") or [])
    result_component = getattr(explorer.result, "component", None)
    if result_component and result_component not in mode_components:
        mode_components.insert(0, str(result_component))
    if not mode_components:
        mode_components = ["perp", "x", "y", "z", "+", "-"]
    controls["mode_component"] = widgets.Dropdown(
        options=mode_components,
        value=mode_components[0],
        description="component",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["mode_z_layer"] = widgets.FloatText(
        value=float(explorer.options.get("z_layer", 0)),
        description="z layer",
        **_maybe_layout(widgets, width="100%"),
    )
    mode_type_options = ["abs", "real", "imag", "phase"]
    mode_type_value = str(getattr(explorer.state, "mode_type", None) or "abs")
    if mode_type_value not in mode_type_options:
        mode_type_options.append(mode_type_value)
    controls["mode_type"] = widgets.Dropdown(
        options=mode_type_options,
        value=mode_type_value,
        description="view",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["mode_extract"] = _make_button(
        widgets,
        "Extract mode",
        color="#15803d",
        button_style="success",
        tooltip="Extract the dispersion mode selected on the heatmap.",
    )
    controls["mode_show_dispersion"] = _make_button(
        widgets,
        "Show dispersion",
        color="#2563eb",
        button_style="info",
        tooltip="Return to the dispersion heatmap output.",
    )
    controls["mode_info"] = widgets.HTML(
        value="<small>Select a point on S(k, f), then extract a mode.</small>"
    )
    controls["status"] = widgets.HTML(value="")
    controls["status_log"] = widgets.HTML(
        value=(
            "<div style='max-height:150px;overflow:auto;font-family:monospace;"
            "font-size:11px;line-height:1.25;border:1px solid #e5e7eb;"
            "padding:4px;background:#f8fafc;'></div>"
        )
    )
    controls["output"] = widgets.Output(
        **_maybe_layout(widgets, border="1px solid #e2e8f0", width="100%")
    )
    initial_render_enabled = bool(
        getattr(explorer, "options", {}).get("initial_render", True)
    )
    placeholder_body = (
        "Initial heatmap render is starting after the toolbar appears."
        if initial_render_enabled
        else (
            "Press <b>Render / refresh dispersion</b> to draw S(k, f). "
            "Manual first render is enabled for this viewer."
        )
    )
    controls["output_placeholder"] = widgets.HTML(
        value=(
            "<div style='padding:18px;font-family:monospace;color:#334155;'>"
            "<b>Dispersion viewer ready.</b><br>"
            f"{placeholder_body}"
            "</div>"
        )
    )
    controls["export_refresh"] = _make_button(
        widgets,
        "Refresh export",
        color="#334155",
        button_style="",
        tooltip="Refresh the JSON-like export snapshot.",
    )
    controls["export_snapshot"] = widgets.HTML(
        value="<small>Press Refresh export snapshot to inspect state.</small>"
    )
    controls["analysis_refresh"] = _make_button(
        widgets,
        "Refresh summary",
        color="#334155",
        button_style="",
        tooltip="Refresh the analysis summary panel.",
    )
    controls["analysis_summary"] = widgets.HTML(
        value="<small>Press Refresh analysis summary to inspect current data.</small>"
    )

    text_cls = getattr(widgets, "Text", None)
    dropdown_cls = getattr(widgets, "Dropdown")
    controls["preset_name"] = (
        text_cls(
            value="",
            placeholder="name",
            description="save",
            **_maybe_layout(widgets, width="100%"),
        )
        if text_cls is not None
        else widgets.HTML(value="")
    )
    controls["preset_select"] = dropdown_cls(
        options=["-- load --"] + list_presets(explorer),
        value="-- load --",
        description="preset",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["preset_save"] = _make_button(
        widgets,
        "Save preset",
        width="49%",
        color="#475569",
        button_style="",
    )
    controls["preset_load"] = _make_button(
        widgets,
        "Load preset",
        width="49%",
        color="#475569",
        button_style="",
    )

    explorer.controls = controls
    explorer.refresh_auxiliary_panels = lambda: _refresh_auxiliary_panels(
        explorer,
        update_status=False,
    )

    for key in [
        "fmin",
        "fmax",
        "source",
        "kscale",
        "cmap",
        "positive",
        "lognorm",
        "grid",
        "selection",
        "notes",
        "analytical_enabled",
        "analytical_sw_config",
        "analytical_model",
        "analytical_n_modes",
        "analytical_k_points",
        "analytical_B",
        "analytical_Ms",
        "analytical_Aex",
        "analytical_d",
        "analytical_phi",
        "analytical_D",
        "filter_snr_enabled",
        "filter_snr_threshold",
        "filter_gaussian_enabled",
        "filter_gaussian_sigma_f",
        "filter_gaussian_sigma_k",
        "filter_gaussian_threshold",
        "filter_percentile_enabled",
        "filter_percentile_low",
        "filter_percentile_high",
        "filter_soft_enabled",
        "filter_soft_percentile",
        "filter_soft_smoothness",
        "filter_log_enabled",
        "filter_log_method",
        "filter_gamma_enabled",
        "filter_gamma_value",
        "mode_type",
    ]:
        if hasattr(controls[key], "observe"):
            controls[key].observe(lambda _change=None: on_display_change(explorer), names="value")

    if hasattr(controls["preset_save"], "on_click"):
        controls["preset_save"].on_click(lambda _btn: _save_current_preset(explorer))
    if hasattr(controls["preset_load"], "on_click"):
        controls["preset_load"].on_click(lambda _btn: _load_selected_preset(explorer))
    if hasattr(controls["render_dispersion"], "on_click"):
        controls["render_dispersion"].on_click(lambda _btn: _render_current_dispersion(explorer))
    if hasattr(controls["mode_extract"], "on_click"):
        controls["mode_extract"].on_click(lambda _btn: on_mode_extract(explorer))
    if hasattr(controls["mode_show_dispersion"], "on_click"):
        controls["mode_show_dispersion"].on_click(lambda _btn: _render_current_dispersion(explorer))
    if hasattr(controls["export_refresh"], "on_click"):
        controls["export_refresh"].on_click(
            lambda _btn: _export_snapshot(explorer, update_status=True)
        )
    if hasattr(controls["analysis_refresh"], "on_click"):
        controls["analysis_refresh"].on_click(
            lambda _btn: _refresh_analysis_summary(explorer, update_status=True)
        )

    display_tab = widgets.VBox(
        [
            controls["render_dispersion"],
            controls["fmin"],
            controls["fmax"],
            controls["source"],
            controls["kscale"],
            controls["cmap"],
            controls["positive"],
            controls["lognorm"],
        ],
        **_maybe_layout(widgets, width="100%"),
    )
    overlays_tab = widgets.VBox(
        [
            controls["grid"],
            controls["selection"],
            controls["notes"],
            controls["selection_info"],
        ],
        **_maybe_layout(widgets, width="100%"),
    )
    analytical_tab = widgets.VBox(
        [
            controls["analytical_enabled"],
            controls["analytical_sw_config"],
            controls["analytical_model"],
            controls["analytical_n_modes"],
            controls["analytical_k_points"],
            controls["analytical_B"],
            controls["analytical_Ms"],
            controls["analytical_Aex"],
            controls["analytical_d"],
            controls["analytical_phi"],
            controls["analytical_D"],
        ],
        **_maybe_layout(widgets, width="100%"),
    )
    filters_tab = widgets.VBox(
        [
            controls["filter_info"],
            controls["filter_snr_enabled"],
            controls["filter_snr_threshold"],
            controls["filter_gaussian_enabled"],
            controls["filter_gaussian_sigma_f"],
            controls["filter_gaussian_sigma_k"],
            controls["filter_gaussian_threshold"],
            controls["filter_percentile_enabled"],
            controls["filter_percentile_low"],
            controls["filter_percentile_high"],
            controls["filter_soft_enabled"],
            controls["filter_soft_percentile"],
            controls["filter_soft_smoothness"],
            controls["filter_log_enabled"],
            controls["filter_log_method"],
            controls["filter_gamma_enabled"],
            controls["filter_gamma_value"],
        ],
        **_maybe_layout(widgets, width="100%"),
    )
    modes_tab = widgets.VBox(
        [
            controls["mode_component"],
            controls["mode_z_layer"],
            controls["mode_type"],
            controls["mode_extract"],
            controls["mode_show_dispersion"],
            controls["mode_info"],
        ],
        **_maybe_layout(widgets, width="100%"),
    )
    preset_box = widgets.VBox(
        [
            controls["preset_select"],
            controls["preset_name"],
            widgets.HBox([controls["preset_save"], controls["preset_load"]]),
        ],
        **_maybe_layout(widgets, width="100%"),
    )
    export_tab = widgets.VBox(
        [
            controls["export_refresh"],
            controls["export_snapshot"],
        ],
        **_maybe_layout(widgets, width="100%"),
    )
    analysis_tab = widgets.VBox(
        [
            controls["analysis_refresh"],
            controls["analysis_summary"],
        ],
        **_maybe_layout(widgets, width="100%"),
    )

    tab_cls = getattr(widgets, "Tab", None)
    if tab_cls is not None:
        tabs = tab_cls(
            children=[
                display_tab,
                overlays_tab,
                analytical_tab,
                filters_tab,
                modes_tab,
                analysis_tab,
                export_tab,
            ],
            selected_index=0,
            **_maybe_layout(widgets, width="100%"),
        )
        tabs.set_title(0, "Display")
        tabs.set_title(1, "Overlays")
        tabs.set_title(2, "Analytical")
        tabs.set_title(3, "Filters")
        tabs.set_title(4, "Modes")
        tabs.set_title(5, "Analysis")
        tabs.set_title(6, "Export")
    else:
        tabs = widgets.VBox(
            [
                display_tab,
                overlays_tab,
                analytical_tab,
                filters_tab,
                modes_tab,
                analysis_tab,
                export_tab,
            ]
        )
    controls["tabs"] = tabs

    control_panel = widgets.VBox(
        [
            widgets.HTML("<b>Dispersion Toolbar v3</b>"),
            preset_box,
            tabs,
            controls["status"],
            controls["status_log"],
        ],
        **_maybe_layout(
            widgets,
            width="320px",
            min_width="320px",
            flex="0 0 320px",
            max_width="100%",
            border="1px solid #ddd",
            padding="8px",
        ),
    )
    right_panel = widgets.VBox(
        [controls["output"]],
        **_maybe_layout(widgets, flex="1 1 420px", width="100%", min_width="0"),
    )
    explorer.widget = widgets.HBox(
        [control_panel, right_panel],
        **_maybe_layout(
            widgets,
            width="100%",
            align_items="stretch",
            flex_flow="row wrap",
            gap="8px",
        ),
    )

    sync_analytical_options(explorer)
    if render_initial:
        set_status(
            explorer,
            "Interactive toolbar ready; initial render scheduled",
            color="#334155",
        )
    else:
        _show_dispersion_placeholder(explorer)
        set_status(
            explorer,
            "Interactive toolbar ready; press Render / refresh dispersion to draw the heatmap",
            color="#334155",
        )


def _save_current_preset(explorer: Any) -> None:
    try:
        name = getattr(explorer.controls["preset_name"], "value", "")
        path = save_preset(explorer, name)
        explorer.controls["preset_select"].options = ["-- load --"] + list_presets(explorer)
        set_status(explorer, f"Preset saved: {path.name}", color="#0F766E")
    except Exception as exc:
        set_status(explorer, f"Preset save failed: {exc}", color="crimson")


def _load_selected_preset(explorer: Any) -> None:
    selected = str(getattr(explorer.controls["preset_select"], "value", "-- load --"))
    if selected == "-- load --":
        return
    try:
        load_preset(explorer, selected)
        explorer.apply_state_to_controls()
        _render_current_dispersion(explorer)
        set_status(explorer, f"Preset loaded: {selected}", color="#0F766E")
    except Exception as exc:
        set_status(explorer, f"Preset load failed: {exc}", color="crimson")
