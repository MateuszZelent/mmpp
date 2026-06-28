"""Widget toolbar for interactive dispersion explorer."""

from __future__ import annotations

from typing import Any

from .callbacks import on_display_change
from .presets import list_presets, load_preset, save_preset
from .rendering import draw_dispersion_panel, refresh_output_widget
from .status import set_status


def _layout(widgets: Any, **kwargs: Any) -> Any:
    layout_cls = getattr(widgets, "Layout", None)
    return layout_cls(**kwargs) if layout_cls is not None else None


def _maybe_layout(widgets: Any, **kwargs: Any) -> dict[str, Any]:
    layout = _layout(widgets, **kwargs)
    return {"layout": layout} if layout is not None else {}


def build_toolbar(explorer: Any, widgets_module: Any) -> None:
    """Build ipywidgets toolbar controls and wire callbacks."""
    widgets = widgets_module
    controls: dict[str, Any] = {}

    f_axis = getattr(explorer.result, "f_axis", None)
    if f_axis is not None and len(f_axis):
        f_pos = [float(v) / 1e9 for v in f_axis if float(v) >= 0.0]
        fmin = min(f_pos) if f_pos else 0.0
        fmax = max(f_pos) if f_pos else 1.0
    else:
        fmin, fmax = 0.0, 1.0
    fmax = float(explorer.options.get("fmax") or fmax or 1.0)
    explorer.state.fmin_ghz = float(explorer.options.get("fmin", fmin))
    explorer.state.fmax_ghz = fmax

    controls["fmin"] = widgets.FloatText(
        value=float(explorer.state.fmin_ghz),
        description="f min",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["fmax"] = widgets.FloatText(
        value=float(explorer.state.fmax_ghz),
        description="f max",
        **_maybe_layout(widgets, width="100%"),
    )
    controls["source"] = widgets.Dropdown(
        options=["display", "raw"],
        value=str(explorer.state.source),
        description="source",
        **_maybe_layout(widgets, width="100%"),
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
        options=["viridis", "plasma", "cividis", "turbo", "inferno", "magma"],
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
    controls["selection_info"] = widgets.HTML(value="<small>No point selected</small>")
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

    text_cls = getattr(widgets, "Text", None)
    button_cls = getattr(widgets, "Button", None)
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
    controls["preset_save"] = (
        button_cls(description="Save preset", **_maybe_layout(widgets, width="49%"))
        if button_cls is not None
        else widgets.HTML(value="")
    )
    controls["preset_load"] = (
        button_cls(description="Load preset", **_maybe_layout(widgets, width="49%"))
        if button_cls is not None
        else widgets.HTML(value="")
    )

    explorer.controls = controls

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
    ]:
        if hasattr(controls[key], "observe"):
            controls[key].observe(lambda _change=None: on_display_change(explorer), names="value")

    if hasattr(controls["preset_save"], "on_click"):
        controls["preset_save"].on_click(lambda _btn: _save_current_preset(explorer))
    if hasattr(controls["preset_load"], "on_click"):
        controls["preset_load"].on_click(lambda _btn: _load_selected_preset(explorer))

    display_tab = widgets.VBox(
        [
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
    preset_box = widgets.VBox(
        [
            controls["preset_select"],
            controls["preset_name"],
            widgets.HBox([controls["preset_save"], controls["preset_load"]]),
        ],
        **_maybe_layout(widgets, width="100%"),
    )

    tab_cls = getattr(widgets, "Tab", None)
    if tab_cls is not None:
        tabs = tab_cls(
            children=[display_tab, overlays_tab],
            selected_index=0,
            **_maybe_layout(widgets, width="100%"),
        )
        tabs.set_title(0, "Display")
        tabs.set_title(1, "Overlays")
    else:
        tabs = widgets.VBox([display_tab, overlays_tab])
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
            border="1px solid #ddd",
            padding="8px",
        ),
    )
    right_panel = widgets.VBox(
        [controls["output"]],
        **_maybe_layout(widgets, flex="1 1 auto", width="auto", min_width="760px"),
    )
    explorer.widget = widgets.HBox(
        [control_panel, right_panel],
        **_maybe_layout(widgets, width="100%", align_items="stretch"),
    )

    draw_dispersion_panel(explorer)
    refresh_output_widget(explorer)
    set_status(explorer, "Interactive toolbar ready", color="#0F766E")


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
        draw_dispersion_panel(explorer)
        refresh_output_widget(explorer)
        set_status(explorer, f"Preset loaded: {selected}", color="#0F766E")
    except Exception as exc:
        set_status(explorer, f"Preset load failed: {exc}", color="crimson")
