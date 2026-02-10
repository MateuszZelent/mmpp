"""Layout helpers for interactive dispersion modes."""

from __future__ import annotations

from typing import Any


def create_layout(explorer: Any, widgets_module: Any) -> Any:
    """Create layout with controls on left, stacked plots on right."""
    widgets = widgets_module

    live_filters_box = widgets.VBox(
        [
            explorer.w_live_snr_enabled,
            explorer.w_live_snr_threshold,
            explorer.w_live_gaussian_enabled,
            explorer.w_live_sigma_f,
            explorer.w_live_sigma_k,
            explorer.w_live_gaussian_threshold_std,
            explorer.w_live_wiener_enabled,
            explorer.w_live_wiener_window,
            explorer.w_live_bandpass_enabled,
            explorer.w_live_kmin,
            explorer.w_live_kmax,
            widgets.HTML("<small>Bandpass uses current f min/f max sliders.</small>"),
        ],
        layout=widgets.Layout(width="100%"),
    )

    compute_filters_box = widgets.VBox(
        [
            explorer.w_pre_remove_static,
            explorer.w_pre_remove_average,
            explorer.w_pre_hann_time,
            explorer.w_pre_hann_space,
            explorer.w_pre_envelope_enabled,
            explorer.w_pre_envelope_threshold_std,
            explorer.w_pre_envelope_margin,
            explorer.w_pre_wavelet_enabled,
            explorer.w_pre_wavelet_level,
            explorer.w_pre_equalize_enabled,
            explorer.w_pre_compression_enabled,
            explorer.w_pre_compression_alpha,
            explorer.w_pre_welch_enabled,
            explorer.w_pre_welch_segments,
            explorer.w_pre_welch_overlap,
            explorer.w_recompute,
        ],
        layout=widgets.Layout(width="100%"),
    )

    # Enhancement filters (non-destructive, image-like processing)
    enhancement_filters_box = widgets.VBox(
        [
            widgets.HTML("<small><b>Dynamic Range</b></small>"),
            explorer.w_live_log_enabled,
            explorer.w_live_log_method,
            explorer.w_live_gamma_enabled,
            explorer.w_live_gamma_value,
            explorer.w_live_percentile_enabled,
            explorer.w_live_percentile_low,
            explorer.w_live_percentile_high,
            widgets.HTML("<hr style='margin:5px'>"),
            widgets.HTML("<small><b>Contrast Enhancement</b></small>"),
            explorer.w_live_clahe_enabled,
            explorer.w_live_clahe_clip,
            explorer.w_live_clahe_tile,
            explorer.w_live_lcn_enabled,
            explorer.w_live_lcn_sigma,
            widgets.HTML("<hr style='margin:5px'>"),
            widgets.HTML("<small><b>Edge Enhancement</b></small>"),
            explorer.w_live_unsharp_enabled,
            explorer.w_live_unsharp_sigma,
            explorer.w_live_unsharp_alpha,
            widgets.HTML("<hr style='margin:5px'>"),
            widgets.HTML("<small><b>Noise Suppression (soft)</b></small>"),
            explorer.w_live_soft_threshold_enabled,
            explorer.w_live_soft_percentile,
            explorer.w_live_soft_smoothness,
            widgets.HTML("<small style='color:#888'>These filters enhance visibility without destroying data.</small>"),
        ],
        layout=widgets.Layout(width="100%"),
    )

    filters_accordion = widgets.Accordion(
        children=[enhancement_filters_box, live_filters_box, compute_filters_box],
        selected_index=0,  # Enhancement filters open by default
        layout=widgets.Layout(width="95%"),
    )
    filters_accordion.set_title(0, "✨ Enhancement (fast)")
    filters_accordion.set_title(1, "🔧 Classic post-filters")
    filters_accordion.set_title(2, "♻️ Compute filters (recompute)")


    # Preset controls at the very top
    preset_load_box = widgets.HBox(
        [explorer.w_preset_load, explorer.w_preset_refresh_btn],
        layout=widgets.Layout(width="100%")
    )

    preset_save_box = widgets.HBox(
        [explorer.w_preset_name, explorer.w_preset_save_btn],
        layout=widgets.Layout(width="100%")
    )

    preset_controls = widgets.VBox(
        [
            widgets.HTML("<small style='color:#666'><b>📁 Presets</b></small>"),
            preset_load_box,
            preset_save_box,
            explorer.w_preset_delete_btn,
        ],
        layout=widgets.Layout(width="100%", padding="3px")
    )

    # === TAB 1: Dispersion Parameters ===
    tab_dispersion = widgets.VBox(
        [
            widgets.HTML("<small><b>Lattice & BZ</b></small>"),
            explorer.w_lattice,
            explorer.w_auto_detect,
            explorer.w_show_bz_lines,
            explorer.w_show_system_info,
            widgets.HTML("<hr style='margin:5px'>"),
            widgets.HTML("<small><b>Frequency Range</b></small>"),
            explorer.w_fmin,
            explorer.w_fmax,
            widgets.HTML("<hr style='margin:5px'>"),
            widgets.HTML("<small><b>Display</b></small>"),
            explorer.w_cmap_disp,
        ],
        layout=widgets.Layout(width="100%", padding="5px")
    )

    # === TAB 2: Mode Parameters ===
    tab_modes = widgets.VBox(
        [
            widgets.HTML("<small><b>BZ Mask Settings</b></small>"),
            explorer.w_n_bz_mask,
            explorer.w_k_direction,
            explorer.w_k_margin,
            explorer.w_f_margin,
            explorer.w_neighbor_reduce,
            widgets.HTML("<hr style='margin:5px'>"),
            widgets.HTML("<small><b>Visualization</b></small>"),
            explorer.w_mode_type,
            explorer.w_mode_x_periods,
            explorer.w_cmap_mode,
        ],
        layout=widgets.Layout(width="100%", padding="5px")
    )

    # === TAB 3: Actions & Animation ===
    tab_actions = widgets.VBox(
        [
            widgets.HTML("<small><b>Plot Controls</b></small>"),
            explorer.w_update,
            explorer.w_reset_zoom,
            widgets.HTML("<hr style='margin:5px'>"),
            widgets.HTML("<small><b>Animation</b></small>"),
            explorer.w_animate,
            explorer.w_anim_frames,
            explorer.w_anim_fps,
            explorer.w_anim_save_mode,
            explorer.w_anim_file_format,
            explorer.w_save_animation,
        ],
        layout=widgets.Layout(width="100%", padding="5px")
    )

    # === TAB 4: Filters ===
    tab_filters = widgets.VBox(
        [
            filters_accordion,
        ],
        layout=widgets.Layout(width="100%", padding="5px")
    )

    # Create tabs
    tabs = widgets.Tab(
        children=[tab_dispersion, tab_modes, tab_actions, tab_filters],
        layout=widgets.Layout(width="100%")
    )
    tabs.set_title(0, "📊 Dispersion")
    tabs.set_title(1, "🎯 Modes")
    tabs.set_title(2, "⚡ Actions")
    tabs.set_title(3, "🔧 Filters")

    # Left panel: controls
    left_panel = widgets.VBox(
        [
            widgets.HTML("<b>🌊 BZ Mode Analysis</b>"),
            widgets.HTML("<hr style='margin:3px'>"),
            preset_controls,
            widgets.HTML("<hr style='margin:3px'>"),
            tabs,
            widgets.HTML("<hr style='margin:5px'>"),
            explorer.w_info,
            widgets.HTML("<hr style='margin:3px'>"),
            widgets.HTML("<small><b>Selected Mode</b></small>"),
            explorer.w_mode_info,
        ],
        layout=widgets.Layout(
            width="320px",
            padding="5px",
            border="1px solid #ddd",
        ),
    )

    # Right panel: stacked plots
    right_panel = widgets.VBox(
        [
            explorer._output,
        ],
        layout=widgets.Layout(
            width="calc(100% - 340px)",
            min_width="760px",
        ),
    )

    # Main layout
    main = widgets.HBox(
        [
            left_panel,
            right_panel,
        ],
        layout=widgets.Layout(
            width="100%",
        ),
    )

    return main
