"""Widget toolbar for interactive hysteresis explorer."""

from __future__ import annotations

from typing import Any

from .callbacks import (
    on_animation_fps_changed,
    on_animation_speed_changed,
    on_component_changed,
    on_field_changed,
    on_index_changed,
    on_panel_widths_changed,
    on_play_toggle,
    on_roi_changed,
    on_save_animation_clicked,
    on_trail_length_changed,
    on_z_layer_changed,
)
from .presets import list_presets, load_preset, save_preset
from .status import set_status


def _parse_roi_text(raw: str) -> tuple[int, int, int, int] | None:
    text = str(raw).strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must have exactly four comma-separated integers")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def build_toolbar(explorer: Any, widgets_module: Any) -> None:
    """Build ipywidgets toolbar controls and wire callbacks."""
    widgets = widgets_module

    n_points = int(explorer.result.field.size)
    z_min, z_max = explorer._z_bounds
    field_arr = explorer.result.field
    field_min = float(field_arr.min()) if n_points else 0.0
    field_max = float(field_arr.max()) if n_points else 1.0
    field_span = max(abs(field_max - field_min), 1e-12)
    field_step = field_span / max(1, n_points - 1)
    controls: dict[str, Any] = {}

    controls["index"] = widgets.IntSlider(
        value=int(explorer.state.current_idx),
        min=0,
        max=max(0, n_points - 1),
        step=1,
        description="idx:",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )
    controls["field"] = widgets.FloatSlider(
        value=float(field_arr[int(explorer.state.current_idx)]) if n_points else 0.0,
        min=field_min,
        max=field_max,
        step=float(field_step),
        description="field:",
        readout_format=".5g",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["play"] = widgets.ToggleButton(
        value=False,
        description="Play",
        button_style="",
        layout=widgets.Layout(width="48%"),
    )
    controls["animation_speed"] = widgets.FloatSlider(
        value=float(explorer.state.animation_speed),
        min=0.25,
        max=4.0,
        step=0.05,
        description="speed:",
        continuous_update=False,
        layout=widgets.Layout(width="50%"),
        style={"description_width": "55px"},
    )
    controls["debug_clicks"] = widgets.Checkbox(
        value=bool(getattr(explorer, "_debug_clicks", False)),
        description="debug clicks",
        indent=False,
        layout=widgets.Layout(width="100%"),
    )
    controls["loop_width"] = widgets.FloatSlider(
        value=float(explorer.state.loop_panel_weight),
        min=0.2,
        max=3.0,
        step=0.05,
        description="loop w:",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["snapshot_width"] = widgets.FloatSlider(
        value=float(explorer.state.snapshot_panel_weight),
        min=0.2,
        max=3.0,
        step=0.05,
        description="snap w:",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )

    for key, label in [
        ("hc", "Hc"),
        ("mr", "Mr"),
        ("ms", "Ms"),
        ("arrow", "arrow"),
        ("branch_colors", "branches"),
        ("trail", "trail"),
    ]:
        controls[key] = widgets.Checkbox(
            value=bool(explorer.state.show_flags.get(key, False)),
            description=label,
            indent=False,
            layout=widgets.Layout(width="100%"),
        )

    controls["component"] = widgets.Dropdown(
        options=["snapshot", "x", "y", "z", "norm"],
        value=str(explorer.state.snapshot_component),
        description="view:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["z_layer"] = widgets.IntSlider(
        value=int(explorer.state.z_layer)
        if str(explorer.state.z_layer) != "all"
        else 0,
        min=int(z_min),
        max=int(z_max),
        step=1,
        description="z:",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["roi"] = widgets.Text(
        value=""
        if explorer.state.roi is None
        else ",".join(str(v) for v in explorer.state.roi),
        placeholder="x0,x1,y0,y1",
        description="roi:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["roi_apply"] = widgets.Button(
        description="Apply ROI",
        layout=widgets.Layout(width="100%"),
        button_style="",
    )

    controls["anim_fps"] = widgets.IntSlider(
        value=int(explorer.result.config.animation_fps),
        min=1,
        max=60,
        step=1,
        description="fps:",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["anim_trail"] = widgets.IntSlider(
        value=int(explorer.result.config.trail_length),
        min=1,
        max=200,
        step=1,
        description="trail:",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["anim_snapshot"] = widgets.Checkbox(
        value=True,
        description="include snapshot panel",
        indent=False,
        layout=widgets.Layout(width="100%"),
    )
    controls["anim_format"] = widgets.Dropdown(
        options=[("GIF", "gif"), ("MP4", "mp4")],
        value="gif",
        description="format:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["anim_path"] = widgets.Text(
        value="",
        placeholder="hysteresis_walk.gif",
        description="path:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["save_animation"] = widgets.Button(
        description="Save animation",
        button_style="warning",
        layout=widgets.Layout(width="100%"),
    )

    controls["preset_name"] = widgets.Text(
        value="",
        placeholder="name",
        description="save:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["preset_save"] = widgets.Button(
        description="Save preset",
        layout=widgets.Layout(width="49%"),
    )
    controls["preset_select"] = widgets.Dropdown(
        options=["-- load --"] + list_presets(explorer),
        value="-- load --",
        description="preset:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["preset_load"] = widgets.Button(
        description="Load preset",
        layout=widgets.Layout(width="49%"),
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
        layout=widgets.Layout(border="1px solid #e2e8f0", width="100%")
    )

    explorer._controls = controls

    def _syncing() -> bool:
        return bool(getattr(explorer, "_syncing_controls", False))

    def _on_index(change):
        if _syncing():
            return
        if change.get("name") == "value":
            on_index_changed(explorer, int(change["new"]))

    controls["index"].observe(_on_index, names="value")

    def _on_field(change):
        if _syncing():
            return
        if change.get("name") == "value":
            on_field_changed(explorer, float(change["new"]))

    controls["field"].observe(_on_field, names="value")

    def _on_play(change):
        if _syncing():
            return
        if change.get("name") != "value":
            return
        enabled = bool(change["new"])
        controls["play"].description = "Pause" if enabled else "Play"
        controls["play"].button_style = "danger" if enabled else ""
        on_play_toggle(explorer, enabled)

    controls["play"].observe(_on_play, names="value")

    def _on_speed(change):
        if _syncing():
            return
        if change.get("name") == "value":
            on_animation_speed_changed(explorer, float(change["new"]))

    controls["animation_speed"].observe(_on_speed, names="value")

    def _on_debug(change):
        if _syncing():
            return
        if change.get("name") != "value":
            return
        enabled = bool(change["new"])
        explorer._debug_clicks = enabled
        if enabled:
            set_status(explorer, "click debug enabled", color="#0369a1")
        else:
            set_status(explorer, "click debug disabled", color="#334155")

    controls["debug_clicks"].observe(_on_debug, names="value")

    def _on_panel_width(_change):
        if _syncing():
            return
        on_panel_widths_changed(
            explorer,
            float(controls["loop_width"].value),
            float(controls["snapshot_width"].value),
        )

    controls["loop_width"].observe(_on_panel_width, names="value")
    controls["snapshot_width"].observe(_on_panel_width, names="value")

    def _on_component(change):
        if _syncing():
            return
        if change.get("name") == "value":
            on_component_changed(explorer, str(change["new"]))

    controls["component"].observe(_on_component, names="value")

    def _on_z(change):
        if _syncing():
            return
        if change.get("name") == "value":
            on_z_layer_changed(explorer, int(change["new"]))

    controls["z_layer"].observe(_on_z, names="value")

    def _on_roi_apply(_btn):
        try:
            roi = _parse_roi_text(controls["roi"].value)
            on_roi_changed(explorer, roi)
            set_status(explorer, f"ROI updated: {roi}", color="#0F766E")
        except Exception as exc:
            set_status(explorer, f"Invalid ROI: {exc}", color="crimson")

    controls["roi_apply"].on_click(_on_roi_apply)

    def _flag_handler(flag_name: str):
        def _on_flag(change):
            if _syncing():
                return
            if change.get("name") != "value":
                return
            explorer.state.show_flags[flag_name] = bool(change["new"])
            explorer._redraw_loop()

        return _on_flag

    for key in ["hc", "mr", "ms", "arrow", "branch_colors", "trail"]:
        controls[key].observe(_flag_handler(key), names="value")

    def _on_anim_fps(change):
        if _syncing():
            return
        if change.get("name") == "value":
            on_animation_fps_changed(explorer, int(change["new"]))

    controls["anim_fps"].observe(_on_anim_fps, names="value")

    def _on_trail(change):
        if _syncing():
            return
        if change.get("name") == "value":
            on_trail_length_changed(explorer, int(change["new"]))

    controls["anim_trail"].observe(_on_trail, names="value")

    controls["save_animation"].on_click(
        lambda _btn: on_save_animation_clicked(explorer)
    )

    def _on_preset_save(_btn):
        try:
            path = save_preset(explorer, controls["preset_name"].value)
            controls["preset_select"].options = ["-- load --"] + list_presets(explorer)
            set_status(explorer, f"Preset saved: {path.name}", color="#0F766E")
        except Exception as exc:
            set_status(explorer, f"Preset save failed: {exc}", color="crimson")

    controls["preset_save"].on_click(_on_preset_save)

    def _on_preset_load(_btn):
        selected = str(controls["preset_select"].value)
        if selected == "-- load --":
            return
        try:
            load_preset(explorer, selected)
            explorer._apply_state_to_controls()
            explorer._set_index(explorer.state.current_idx)
            set_status(explorer, f"Preset loaded: {selected}", color="#0F766E")
        except Exception as exc:
            set_status(explorer, f"Preset load failed: {exc}", color="crimson")

    controls["preset_load"].on_click(_on_preset_load)

    display_tab = widgets.VBox(
        [
            controls["field"],
            widgets.HBox([controls["play"], controls["animation_speed"]]),
            widgets.HBox([controls["loop_width"], controls["snapshot_width"]]),
            controls["debug_clicks"],
            controls["hc"],
            controls["mr"],
            controls["ms"],
            controls["arrow"],
            controls["branch_colors"],
            controls["trail"],
        ],
        layout=widgets.Layout(width="100%"),
    )

    snapshot_tab = widgets.VBox(
        [
            controls["component"],
            controls["z_layer"],
            controls["roi"],
            controls["roi_apply"],
        ],
        layout=widgets.Layout(width="100%"),
    )

    animation_tab = widgets.VBox(
        [
            controls["anim_fps"],
            controls["anim_trail"],
            controls["anim_snapshot"],
            controls["anim_format"],
            controls["anim_path"],
            controls["save_animation"],
        ],
        layout=widgets.Layout(width="100%"),
    )

    tabs = widgets.Tab(
        children=[display_tab, snapshot_tab, animation_tab],
        selected_index=0,
        layout=widgets.Layout(width="100%"),
    )
    tabs.set_title(0, "Display")
    tabs.set_title(1, "Snapshot")
    tabs.set_title(2, "Animation")
    controls["tabs"] = tabs

    preset_box = widgets.VBox(
        [
            controls["preset_select"],
            controls["preset_name"],
            widgets.HBox([controls["preset_save"], controls["preset_load"]]),
        ],
        layout=widgets.Layout(width="100%"),
    )

    control_panel = widgets.VBox(
        [
            widgets.HTML("<b>Hysteresis Toolbar v3</b>"),
            preset_box,
            tabs,
            controls["status"],
            controls["status_log"],
        ],
        layout=widgets.Layout(
            width="320px",
            min_width="320px",
            flex="0 0 320px",
            border="1px solid #ddd",
            padding="8px",
        ),
    )

    right_panel = widgets.VBox(
        [controls["output"]],
        layout=widgets.Layout(flex="1 1 auto", width="auto", min_width="760px"),
    )

    explorer._widget_root = widgets.HBox(
        [control_panel, right_panel],
        layout=widgets.Layout(width="100%", align_items="stretch"),
    )
