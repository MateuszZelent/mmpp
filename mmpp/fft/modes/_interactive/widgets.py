"""Widget-construction helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from typing import Any

import numpy as np

from .filters import TOPOLOGICAL_COMPONENT_NAMES


def build_toolbar(explorer: Any, widgets_module: Any) -> None:
    """Build ipywidgets toolbar UI."""
    widgets = widgets_module

    fmin = float(np.nanmin(explorer._raw_frequencies_ghz))
    fmax = float(np.nanmax(explorer._raw_frequencies_ghz))

    z_min, z_max = explorer._guess_layer_bounds()

    controls: dict[str, Any] = {}
    component_labels = {
        "x": "m_x",
        "y": "m_y",
        "z": "m_z",
        "+": "m+ (RCP)",
        "-": "m- (LCP)",
        "rho": "m_rho",
        "phi": "m_phi",
    }
    component_keys: list[str] = []
    for comp in explorer._available_components:
        key = str(comp).strip().lower()
        if key and key not in component_keys:
            component_keys.append(key)
    for comp in TOPOLOGICAL_COMPONENT_NAMES:
        if comp not in component_keys:
            component_keys.append(comp)

    controls["components"] = widgets.SelectMultiple(
        options=[
            (component_labels.get(name, f"m_{name}"), name) for name in component_keys
        ],
        value=tuple(explorer._current_components),
        description="Comp:",
        layout=widgets.Layout(width="100%", height="90px"),
        style={"description_width": "55px"},
    )
    controls["z_layer"] = widgets.IntSlider(
        value=explorer._current_z_layer,
        min=z_min,
        max=z_max,
        step=1,
        description="z:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )

    controls["fmin"] = widgets.FloatSlider(
        value=explorer._filter_state.freq_min,
        min=fmin,
        max=fmax,
        step=max((fmax - fmin) / 400.0, 1e-4),
        description="f min:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )
    controls["fmax"] = widgets.FloatSlider(
        value=explorer._filter_state.freq_max,
        min=fmin,
        max=fmax,
        step=max((fmax - fmin) / 400.0, 1e-4),
        description="f max:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )

    controls["smooth_filter"] = widgets.Dropdown(
        options=[
            ("none", "none"),
            ("moving average", "moving_average"),
            ("gaussian", "gaussian"),
            ("savitzky-golay", "savgol"),
        ],
        value=explorer._filter_state.smooth_filter,
        description="smooth:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["smooth_window"] = widgets.IntSlider(
        value=explorer._filter_state.smooth_window,
        min=3,
        max=61,
        step=2,
        description="window:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )
    controls["smooth_sigma"] = widgets.FloatSlider(
        value=explorer._filter_state.smooth_sigma,
        min=0.0,
        max=8.0,
        step=0.1,
        description="sigma:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )
    controls["baseline_mode"] = widgets.Dropdown(
        options=[
            ("none", "none"),
            ("mean", "mean"),
            ("median", "median"),
            ("linear", "linear"),
        ],
        value=explorer._filter_state.baseline_mode,
        description="baseline:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["clip_low"] = widgets.FloatSlider(
        value=explorer._filter_state.clip_percentile_low,
        min=0.0,
        max=50.0,
        step=0.5,
        description="clip lo:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )
    controls["clip_high"] = widgets.FloatSlider(
        value=explorer._filter_state.clip_percentile_high,
        min=50.0,
        max=100.0,
        step=0.5,
        description="clip hi:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )
    controls["soft_threshold"] = widgets.FloatSlider(
        value=explorer._filter_state.soft_threshold_percentile,
        min=0.0,
        max=100.0,
        step=1.0,
        description="soft thr:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )

    controls["normalize"] = widgets.Checkbox(
        value=explorer._filter_state.normalize,
        description="normalize",
        layout=widgets.Layout(width="100%"),
    )
    controls["log_scale"] = widgets.Checkbox(
        value=explorer._filter_state.log_scale,
        description="log10",
        layout=widgets.Layout(width="100%"),
    )
    controls["show_peaks"] = widgets.Checkbox(
        value=explorer._show_peaks,
        description="show peaks",
        layout=widgets.Layout(width="100%"),
    )
    controls["peak_prom"] = widgets.FloatSlider(
        value=explorer._peak_prominence,
        min=0.0,
        max=1.0,
        step=0.01,
        description="prom:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )
    controls["peak_dist"] = widgets.IntSlider(
        value=explorer._peak_distance,
        min=1,
        max=200,
        step=1,
        description="dist:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )

    controls["mode_view"] = widgets.Dropdown(
        options=[
            ("all", "all"),
            ("magnitude", "magnitude"),
            ("phase", "phase"),
            ("combined", "combined"),
        ],
        value=(
            "all"
            if len(explorer._mode_row_types) > 1
            else explorer._mode_row_types[0]
        ),
        description="rows:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )

    controls["cmap_mag"] = widgets.Dropdown(
        options=["viridis", "inferno", "plasma", "cividis", "magma"],
        value="viridis",
        description="cmap |m|:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["cmap_phase"] = widgets.Dropdown(
        options=["twilight", "twilight_shifted", "hsv", "RdBu_r", "seismic"],
        value="twilight",
        description="cmap ph:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["cmap_combined"] = widgets.Dropdown(
        options=["RdBu_r", "coolwarm", "seismic", "PiYG", "PRGn"],
        value="RdBu_r",
        description="cmap cmb:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["aspect"] = widgets.Dropdown(
        options=["equal", "auto", "0.5", "1.0", "2.0"],
        value=explorer._mode_aspect if explorer._mode_aspect in ["equal", "auto"] else "equal",
        description="aspect:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["layout"] = widgets.Dropdown(
        options=[
            ("auto", "auto"),
            ("vertical", "vertical"),
            ("horizontal", "horizontal"),
        ],
        value=(
            explorer._layout_mode
            if explorer._layout_mode in {"auto", "vertical", "horizontal"}
            else "auto"
        ),
        description="layout:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )

    freq_vals = np.asarray(explorer._filtered_frequencies_ghz, dtype=float)
    if freq_vals.size:
        freq_min = float(np.nanmin(freq_vals))
        freq_max = float(np.nanmax(freq_vals))
        current_freq = (
            float(explorer._current_frequency_ghz)
            if explorer._current_frequency_ghz is not None
            else freq_min
        )
        current_freq = float(np.clip(current_freq, freq_min, freq_max))
    else:
        freq_min = 0.0
        freq_max = 0.0
        current_freq = 0.0

    freq_step = max((freq_max - freq_min) / 2000.0, 1e-4)
    controls["freq_index"] = widgets.FloatSlider(
        value=current_freq,
        min=freq_min,
        max=freq_max,
        step=freq_step,
        description="freq [GHz]:",
        readout_format=".4f",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "70px"},
        continuous_update=False,
    )

    n_anim_frames = 60
    controls["phase_index"] = widgets.IntSlider(
        value=0,
        min=0,
        max=n_anim_frames - 1,
        step=1,
        description="φ:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "30px"},
        continuous_update=True,
    )
    controls["play"] = widgets.Play(
        value=0,
        min=0,
        max=n_anim_frames - 1,
        step=1,
        interval=42,
        description="phase",
        disabled=False,
    )

    controls["anim_frames"] = widgets.IntSlider(
        value=180,
        min=20,
        max=600,
        step=10,
        description="frames:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )
    controls["anim_fps"] = widgets.IntSlider(
        value=24,
        min=5,
        max=60,
        step=1,
        description="fps:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
        continuous_update=False,
    )
    controls["anim_format"] = widgets.Dropdown(
        options=[("GIF (animated)", "gif"), ("MP4 (video)", "mp4")],
        value="gif",
        description="format:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["save_animation"] = widgets.Button(
        description="💾 Save Mode",
        button_style="warning",
        layout=widgets.Layout(width="49%"),
    )
    controls["animate"] = widgets.Button(
        description="🎬 Animate",
        button_style="warning",
        layout=widgets.Layout(width="49%"),
    )
    controls["mode_type"] = widgets.Dropdown(
        options=[
            ("Real (oscillating)", "real"),
            ("Imaginary", "imag"),
            ("Amplitude |M|", "abs"),
            ("Phase φ", "phase"),
            ("Combined Re[M]", "combined"),
        ],
        value="combined",
        description="viz:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )

    controls["refresh"] = widgets.Button(
        description="Refresh",
        button_style="success",
        layout=widgets.Layout(width="48%"),
    )
    controls["reset"] = widgets.Button(
        description="Reset",
        button_style="",
        layout=widgets.Layout(width="48%"),
    )

    controls["status"] = widgets.HTML(
        value="<small>Left-click spectrum: exact frequency. Right-click or Shift+click: nearest peak.</small>",
    )
    controls["preset_select"] = widgets.Dropdown(
        options=[("-- load preset --", "")],
        value="",
        description="preset:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["preset_name"] = widgets.Text(
        value="",
        placeholder="name...",
        description="save:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "55px"},
    )
    controls["preset_save"] = widgets.Button(
        description="Save preset",
        button_style="",
        layout=widgets.Layout(width="49%"),
    )
    controls["preset_delete"] = widgets.Button(
        description="Delete preset",
        button_style="",
        layout=widgets.Layout(width="49%"),
    )

    observe_keys = [
        "components",
        "z_layer",
        "fmin",
        "fmax",
        "smooth_filter",
        "smooth_window",
        "smooth_sigma",
        "baseline_mode",
        "clip_low",
        "clip_high",
        "soft_threshold",
        "normalize",
        "log_scale",
        "show_peaks",
        "peak_prom",
        "peak_dist",
        "mode_view",
        "aspect",
        "layout",
        "cmap_mag",
        "cmap_phase",
        "cmap_combined",
    ]
    for key in observe_keys:
        controls[key].observe(explorer._on_controls_changed, names="value")

    controls["freq_index"].observe(explorer._on_frequency_index_changed, names="value")
    controls["refresh"].on_click(explorer._on_refresh_clicked)
    controls["reset"].on_click(explorer._on_reset_clicked)
    controls["save_animation"].on_click(explorer._on_save_animation_clicked)
    controls["animate"].on_click(explorer._on_animate_clicked)
    controls["mode_type"].observe(explorer._on_mode_type_changed, names="value")
    controls["phase_index"].observe(explorer._on_phase_index_changed, names="value")

    def _on_play_value_changed(change: Any) -> None:
        if change.get("name") != "value":
            return
        phase_control = controls.get("phase_index")
        if phase_control is None:
            return
        new_val = int(change.get("new", 0))
        new_val = int(np.clip(new_val, phase_control.min, phase_control.max))
        if int(phase_control.value) != new_val:
            explorer._internal_update = True
            try:
                phase_control.value = new_val
            finally:
                explorer._internal_update = False
        explorer._on_phase_index_changed({"name": "value", "new": new_val})

    controls["play"].observe(_on_play_value_changed, names="value")
    controls["preset_save"].on_click(explorer._on_save_preset_clicked)
    controls["preset_delete"].on_click(explorer._on_delete_preset_clicked)
    controls["preset_select"].observe(explorer._on_load_preset_changed, names="value")

    explorer._widget_output = widgets.Output(
        layout=widgets.Layout(width="100%", height="auto")
    )

    preset_box = widgets.VBox(
        [
            controls["preset_select"],
            controls["preset_name"],
            widgets.HBox([controls["preset_save"], controls["preset_delete"]]),
        ]
    )

    sections = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    controls["components"],
                    controls["z_layer"],
                    controls["mode_view"],
                    controls["aspect"],
                    controls["layout"],
                    controls["cmap_mag"],
                    controls["cmap_phase"],
                    controls["cmap_combined"],
                ]
            ),
            widgets.VBox(
                [
                    controls["fmin"],
                    controls["fmax"],
                    controls["normalize"],
                    controls["log_scale"],
                    controls["show_peaks"],
                    controls["peak_prom"],
                    controls["peak_dist"],
                ]
            ),
            widgets.VBox(
                [
                    controls["smooth_filter"],
                    controls["smooth_window"],
                    controls["smooth_sigma"],
                    controls["baseline_mode"],
                    controls["clip_low"],
                    controls["clip_high"],
                    controls["soft_threshold"],
                ]
            ),
            widgets.VBox(
                [
                    controls["freq_index"],
                    widgets.HBox([controls["play"], controls["phase_index"]]),
                    controls["mode_type"],
                    controls["anim_frames"],
                    controls["anim_fps"],
                    controls["anim_format"],
                    widgets.HBox([controls["save_animation"], controls["animate"]]),
                ]
            ),
        ],
        selected_index=0,
        layout=widgets.Layout(width="100%"),
    )
    sections.set_title(0, "Display")
    sections.set_title(1, "Spectrum")
    sections.set_title(2, "Filters")
    sections.set_title(3, "Animation")

    control_panel = widgets.VBox(
        [
            widgets.HTML("<b>FMR Spectrum Toolbar v3</b>"),
            preset_box,
            sections,
            widgets.HBox([controls["refresh"], controls["reset"]]),
            controls["status"],
        ],
        layout=widgets.Layout(
            width="315px",
            min_width="315px",
            flex="0 0 315px",
            border="1px solid #ddd",
            padding="8px",
        ),
    )

    right_panel = widgets.VBox(
        [explorer._widget_output],
        layout=widgets.Layout(flex="1 1 auto", width="auto", min_width="760px"),
    )

    explorer._widget_root = widgets.HBox(
        [control_panel, right_panel],
        layout=widgets.Layout(width="100%", align_items="stretch"),
    )

    explorer._controls = controls
    explorer._refresh_preset_options()


__all__ = ["build_toolbar"]
