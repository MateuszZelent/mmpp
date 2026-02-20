"""Internal interactive-spectrum helpers."""

from .callbacks import (
    on_animate_clicked,
    on_mode_type_changed,
    on_phase_index_changed,
    on_save_animation_clicked,
    stop_animation,
)
from .compat import plot as plot_compat
from .controls import (
    guess_layer_bounds,
    read_controls,
    refresh_freq_slider_bounds,
    resolve_mode_rows,
)
from .data import (
    initialize_frequency,
    load_spectrum_data,
    recompute_filtered_spectrum,
)
from .filters import (
    _COMPONENT_INDEX,
    COMPONENT_LABELS,
    COMPONENT_NAMES,
    SpectrumFilterState,
    _component_from_label,
    _to_ghz,
    _to_power,
    apply_spectrum_filters,
    collapse_spectrum_components,
    detect_spectrum_peaks,
    normalize_component_selection,
    normalize_spectrum_component_selection,
)
from .interactions import (
    closest_freq_index,
    on_spectrum_click,
    update_frequency_selection,
)
from .mode_layout import (
    apply_mode_colorbars,
    finalize_mode_figure,
    reset_mode_colorbars,
)
from .mode_plots import update_mode_plots
from .presets import (
    apply_preset_state,
    collect_preset_state,
    get_presets_dir,
    list_presets,
    on_delete_preset_clicked,
    on_load_preset_changed,
    on_save_preset_clicked,
    refresh_preset_options,
)
from .rendering import (
    create_figure,
    draw_frequency_line,
    draw_spectrum,
    render_figure,
)
from .status import set_status, update_status_text
from .widgets import build_toolbar

__all__ = [
    "COMPONENT_LABELS",
    "COMPONENT_NAMES",
    "_COMPONENT_INDEX",
    "SpectrumFilterState",
    "_component_from_label",
    "_to_ghz",
    "_to_power",
    "apply_spectrum_filters",
    "collapse_spectrum_components",
    "detect_spectrum_peaks",
    "normalize_component_selection",
    "normalize_spectrum_component_selection",
    "guess_layer_bounds",
    "load_spectrum_data",
    "recompute_filtered_spectrum",
    "initialize_frequency",
    "read_controls",
    "refresh_freq_slider_bounds",
    "resolve_mode_rows",
    "closest_freq_index",
    "update_frequency_selection",
    "on_spectrum_click",
    "plot_compat",
    "get_presets_dir",
    "list_presets",
    "collect_preset_state",
    "apply_preset_state",
    "refresh_preset_options",
    "on_save_preset_clicked",
    "on_load_preset_changed",
    "on_delete_preset_clicked",
    "on_save_animation_clicked",
    "on_animate_clicked",
    "stop_animation",
    "on_mode_type_changed",
    "on_phase_index_changed",
    "build_toolbar",
    "update_mode_plots",
    "reset_mode_colorbars",
    "apply_mode_colorbars",
    "finalize_mode_figure",
    "set_status",
    "update_status_text",
    "render_figure",
    "create_figure",
    "draw_spectrum",
    "draw_frequency_line",
]
