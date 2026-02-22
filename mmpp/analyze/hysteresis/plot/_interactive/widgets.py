"""Widget toolbar for interactive hysteresis explorer."""

from __future__ import annotations

from typing import Any

from .callbacks import (
    on_component_changed,
    on_index_changed,
    on_play_toggle,
    on_roi_changed,
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

    controls: dict[str, Any] = {}

    controls["index"] = widgets.IntSlider(
        value=int(explorer.state.current_idx),
        min=0,
        max=max(0, n_points - 1),
        step=1,
        description="idx:",
        continuous_update=False,
        layout=widgets.Layout(width="98%"),
    )

    controls["play"] = widgets.ToggleButton(
        value=False,
        description="Play",
        button_style="",
        layout=widgets.Layout(width="90px"),
    )

    controls["component"] = widgets.Dropdown(
        options=["snapshot", "x", "y", "z", "norm"],
        value=str(explorer.state.snapshot_component),
        description="view:",
        layout=widgets.Layout(width="220px"),
    )

    controls["z_layer"] = widgets.IntSlider(
        value=int(explorer.state.z_layer) if str(explorer.state.z_layer) != "all" else 0,
        min=int(z_min),
        max=int(z_max),
        step=1,
        description="z:",
        continuous_update=False,
        layout=widgets.Layout(width="220px"),
    )

    controls["roi"] = widgets.Text(
        value="" if explorer.state.roi is None else ",".join(str(v) for v in explorer.state.roi),
        placeholder="x0,x1,y0,y1",
        description="roi:",
        layout=widgets.Layout(width="240px"),
    )
    controls["roi_apply"] = widgets.Button(
        description="Apply ROI",
        layout=widgets.Layout(width="100px"),
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
            layout=widgets.Layout(width="92px"),
        )

    controls["preset_name"] = widgets.Text(
        value="",
        placeholder="name",
        description="preset:",
        layout=widgets.Layout(width="220px"),
    )
    controls["preset_save"] = widgets.Button(
        description="Save",
        layout=widgets.Layout(width="70px"),
    )
    controls["preset_select"] = widgets.Dropdown(
        options=["-- load --"] + list_presets(explorer),
        value="-- load --",
        description="load:",
        layout=widgets.Layout(width="220px"),
    )
    controls["preset_load"] = widgets.Button(
        description="Load",
        layout=widgets.Layout(width="70px"),
    )

    controls["status"] = widgets.HTML(value="")
    controls["status_log"] = widgets.HTML(value="")

    controls["output"] = widgets.Output(
        layout=widgets.Layout(border="1px solid #e2e8f0", width="100%")
    )

    explorer._controls = controls

    def _on_index(change):
        if change.get("name") == "value":
            on_index_changed(explorer, int(change["new"]))

    controls["index"].observe(_on_index, names="value")

    def _on_play(change):
        if change.get("name") == "value":
            enabled = bool(change["new"])
            controls["play"].description = "Pause" if enabled else "Play"
            on_play_toggle(explorer, enabled)

    controls["play"].observe(_on_play, names="value")

    def _on_component(change):
        if change.get("name") == "value":
            on_component_changed(explorer, str(change["new"]))

    controls["component"].observe(_on_component, names="value")

    def _on_z(change):
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
            if change.get("name") != "value":
                return
            explorer.state.show_flags[flag_name] = bool(change["new"])
            explorer._redraw_loop()

        return _on_flag

    for key in ["hc", "mr", "ms", "arrow", "branch_colors", "trail"]:
        controls[key].observe(_flag_handler(key), names="value")

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

    # Layout
    row1 = widgets.HBox([controls["index"], controls["play"]])
    row2 = widgets.HBox([controls["component"], controls["z_layer"], controls["roi"], controls["roi_apply"]])
    row3 = widgets.HBox([
        controls["hc"],
        controls["mr"],
        controls["ms"],
        controls["arrow"],
        controls["branch_colors"],
        controls["trail"],
    ])
    row4 = widgets.HBox([
        controls["preset_name"],
        controls["preset_save"],
        controls["preset_select"],
        controls["preset_load"],
    ])

    toolbar = widgets.VBox([
        row1,
        row2,
        row3,
        row4,
        controls["status"],
        controls["status_log"],
    ])

    explorer._widget_root = widgets.VBox([toolbar, controls["output"]])
