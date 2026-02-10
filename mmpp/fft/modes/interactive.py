"""Interactive FMR spectrum explorer.

This module provides a refactored interactive UI for FMR spectrum analysis.
It supports two operation modes:

- Toolbar mode (ipywidgets): interactive controls, filtering, and sweep animation
- Classic matplotlib mode: click-to-select spectrum with mode panels
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import logging
import numpy as np

log = logging.getLogger("mmpp.fft.modes")

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.gridspec import GridSpec

    _HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - optional dependency
    Figure = Any
    Axes = Any
    _HAS_MATPLOTLIB = False

try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    _HAS_WIDGETS = True
except ImportError:  # pragma: no cover - optional dependency
    widgets = None  # type: ignore[assignment]
    clear_output = display = None  # type: ignore[assignment]
    _HAS_WIDGETS = False

from ._interactive import (
    COMPONENT_NAMES,
    SpectrumFilterState,
    apply_preset_state,
    closest_freq_index,
    collect_preset_state,
    build_toolbar,
    create_figure,
    draw_frequency_line,
    draw_spectrum,
    guess_layer_bounds,
    get_presets_dir,
    initialize_frequency,
    list_presets,
    load_spectrum_data,
    on_animate_clicked,
    on_delete_preset_clicked,
    on_load_preset_changed,
    on_spectrum_click,
    on_mode_type_changed,
    on_phase_index_changed,
    on_save_animation_clicked,
    on_save_preset_clicked,
    plot_compat,
    read_controls,
    render_figure,
    refresh_freq_slider_bounds,
    refresh_preset_options,
    resolve_mode_rows,
    recompute_filtered_spectrum,
    set_status,
    stop_animation,
    update_mode_plots,
    update_status_text,
    update_frequency_selection,
    normalize_component_selection,
)


class InteractiveSpectrum:
    """Interactive FMR spectrum explorer with optional toolbar UI."""

    def __init__(
        self,
        data_loader: Any = None,
        spectrum_result: Any = None,
        component_label: Optional[str] = None,
        analyzer: Any = None,
        dpi: int = 100,
        figsize: Tuple[float, float] = (16.0, 10.0),
    ):
        if not _HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for interactive spectrum")

        self.data_loader = data_loader
        self.spectrum_result = spectrum_result
        self._component_label = component_label
        self.analyzer = analyzer

        self.dpi = int(dpi)
        self.figsize = tuple(figsize)

        # Spectrum state
        self._raw_frequencies_ghz: np.ndarray = np.array([], dtype=float)
        self._raw_component_power: dict[str, np.ndarray] = {}
        self._available_components: list[str] = []

        self._filtered_frequencies_ghz: np.ndarray = np.array([], dtype=float)
        self._filtered_component_power: dict[str, np.ndarray] = {}
        self._peaks: list[tuple[float, float]] = []

        # Visualization state
        self._fig: Optional[Figure] = None
        self._ax_spectrum: Optional[Axes] = None
        self._mode_axes: Optional[np.ndarray] = None
        self._mode_row_types: list[str] = ["magnitude", "phase", "combined"]
        self._mode_colorbars: list[Any] = []
        self._frequency_line: Any = None
        self._current_frequency_ghz: Optional[float] = None
        self._current_components: list[str] = ["x", "y", "z"]
        self._current_z_layer: int = -1
        self._freq_unit: str = "GHz"
        self._title: Optional[str] = None
        self._show_peaks: bool = True
        self._filter_state = SpectrumFilterState(0.0, 1.0)

        # Toolbar widgets/state
        self._toolbar_enabled = False
        self._widget_root: Any = None
        self._widget_output: Any = None
        self._controls: dict[str, Any] = {}
        self._internal_update = False
        self._presets_dir: Optional[Path] = None
        self._is_saving_animation = False
        
        # Layout configuration
        self._mode_aspect: str = "equal"
        self._xlim: Optional[Tuple[float, float]] = None
        self._ylim: Optional[Tuple[float, float]] = None
        self._layout_mode: str = "vertical"  # "vertical" or "horizontal"
        
        # Animation state (matching dispersion module pattern)
        self._animation: Any = None
        self._is_animating: bool = False
        self._geometry_contour: Optional[np.ndarray] = None  # For overlay on mode plots
        self._mode_type: str = "combined"  # real, imag, abs, phase, combined, ampl_phase

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def show(
        self,
        components: Optional[Sequence[Union[int, str]]] = None,
        z_layer: int = -1,
        log_scale: bool = False,
        normalize: bool = True,
        freq_unit: str = "GHz",
        show_peaks: bool = True,
        title: Optional[str] = None,
        initial_frequency: Optional[float] = None,
        toolbar: bool = True,
        smooth_filter: str = "none",
        smooth_window: int = 7,
        smooth_sigma: float = 1.0,
        baseline_mode: str = "none",
        clip_percentile_low: float = 0.0,
        clip_percentile_high: float = 100.0,
        soft_threshold_percentile: float = 0.0,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        peak_prominence: float = 0.05,
        peak_distance: int = 5,
        mode_view: str = "all",
        show: bool = True,
        # New layout parameters
        aspect: str = "equal",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        layout: str = "vertical",
        **_ignored: Any,
    ) -> Any:
        """Create interactive spectrum with mode visualization.

        Parameters mirror previous API and extend it with toolbar/filter options.
        """
        self._load_spectrum_data()

        self._current_z_layer = int(z_layer)
        self._freq_unit = str(freq_unit)
        self._title = title
        self._show_peaks = bool(show_peaks)
        self._mode_row_types = self._resolve_mode_rows(mode_view)
        self._current_components = normalize_component_selection(
            components,
            available=self._available_components or COMPONENT_NAMES,
        )
        
        # Store layout configuration
        self._mode_aspect = str(aspect)
        self._xlim = tuple(xlim) if xlim else None
        self._ylim = tuple(ylim) if ylim else None
        self._layout_mode = str(layout)

        data_fmin = float(np.nanmin(self._raw_frequencies_ghz))
        data_fmax = float(np.nanmax(self._raw_frequencies_ghz))
        init_fmin = data_fmin if freq_min is None else float(freq_min)
        init_fmax = data_fmax if freq_max is None else float(freq_max)
        init_fmin = float(np.clip(init_fmin, data_fmin, data_fmax))
        init_fmax = float(np.clip(init_fmax, data_fmin, data_fmax))
        if init_fmin > init_fmax:
            init_fmin, init_fmax = init_fmax, init_fmin

        self._filter_state = SpectrumFilterState(
            freq_min=init_fmin,
            freq_max=init_fmax,
            smooth_filter=str(smooth_filter),
            smooth_window=int(smooth_window),
            smooth_sigma=float(smooth_sigma),
            baseline_mode=str(baseline_mode),
            clip_percentile_low=float(clip_percentile_low),
            clip_percentile_high=float(clip_percentile_high),
            soft_threshold_percentile=float(soft_threshold_percentile),
            normalize=bool(normalize),
            log_scale=bool(log_scale),
        )

        self._peak_prominence = float(peak_prominence)
        self._peak_distance = int(peak_distance)

        self._recompute_filtered_spectrum()
        self._initialize_frequency(initial_frequency)

        if toolbar and _HAS_WIDGETS:
            self._toolbar_enabled = True
            self._build_toolbar()
            self._render_figure()
            if show:
                display(self._widget_root)
                return None  # Avoid double display in Jupyter (display + auto-return)
            return self._widget_root

        self._toolbar_enabled = False
        self._render_figure()
        if show:
            plt.show()
            return None  # Avoid double display in Jupyter (plt.show + auto-return)
        return self._fig

    # ---------------------------------------------------------------------
    # Data processing
    # ---------------------------------------------------------------------
    def _load_spectrum_data(self) -> None:
        """Load and normalize spectrum data from available source."""
        load_spectrum_data(self)

    def _recompute_filtered_spectrum(self) -> None:
        """Recompute filtered traces and peak list from current filter state."""
        recompute_filtered_spectrum(self)

    def _initialize_frequency(self, initial_frequency: Optional[float]) -> None:
        """Set current frequency based on initial request, peaks, or center."""
        initialize_frequency(self, initial_frequency)

    # ---------------------------------------------------------------------
    # Presets
    # ---------------------------------------------------------------------
    def _get_presets_dir(self) -> Path:
        """Return project-local presets directory."""
        return get_presets_dir(self)

    def _list_presets(self) -> list[str]:
        """List available interactive toolbar presets."""
        return list_presets(self)

    def _collect_preset_state(self) -> dict[str, Any]:
        """Collect serializable state from current controls."""
        return collect_preset_state(self)

    def _apply_preset_state(self, payload: dict[str, Any]) -> None:
        """Apply preset payload to widgets/state."""
        apply_preset_state(self, payload)

    def _refresh_preset_options(self) -> None:
        """Refresh preset dropdown options."""
        refresh_preset_options(self)

    def _on_save_preset_clicked(self, _btn: Any) -> None:
        """Persist current toolbar config as a preset."""
        on_save_preset_clicked(self, _btn)

    def _on_load_preset_changed(self, change: Any) -> None:
        """Load selected preset and apply values to toolbar."""
        on_load_preset_changed(self, change)

    def _on_delete_preset_clicked(self, _btn: Any) -> None:
        """Delete selected preset file."""
        on_delete_preset_clicked(self, _btn)

    # ---------------------------------------------------------------------
    # Widget toolbar
    # ---------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        """Build ipywidgets toolbar UI."""
        if not _HAS_WIDGETS:
            raise RuntimeError("ipywidgets is required for toolbar mode")
        build_toolbar(self, widgets_module=widgets)

    def _guess_layer_bounds(self) -> tuple[int, int]:
        """Best-effort z-layer slider bounds."""
        return guess_layer_bounds(self)

    def _on_controls_changed(self, _change: Any) -> None:
        if self._internal_update:
            return

        self._read_controls()
        self._recompute_filtered_spectrum()

        # Clamp currently selected frequency to filtered range.
        if self._filtered_frequencies_ghz.size:
            idx = self._closest_freq_index(self._current_frequency_ghz)
            self._current_frequency_ghz = float(self._filtered_frequencies_ghz[idx])

        self._refresh_freq_slider_bounds()
        self._render_figure()

    def _on_frequency_index_changed(self, change: Any) -> None:
        if self._internal_update:
            return
        if change.get("name") != "value":
            return

        if self._filtered_frequencies_ghz.size == 0:
            return

        idx = int(change["new"])
        idx = max(0, min(idx, self._filtered_frequencies_ghz.size - 1))
        self._current_frequency_ghz = float(self._filtered_frequencies_ghz[idx])
        self._update_frequency_selection(redraw_canvas=True)

    def _on_refresh_clicked(self, _btn: Any) -> None:
        self._read_controls()
        self._recompute_filtered_spectrum()
        self._refresh_freq_slider_bounds()
        self._render_figure()

    def _on_reset_clicked(self, _btn: Any) -> None:
        if not self._controls:
            return

        self._internal_update = True
        try:
            fmin = float(np.nanmin(self._raw_frequencies_ghz))
            fmax = float(np.nanmax(self._raw_frequencies_ghz))
            self._controls["fmin"].value = fmin
            self._controls["fmax"].value = fmax
            self._controls["smooth_filter"].value = "none"
            self._controls["smooth_window"].value = 7
            self._controls["smooth_sigma"].value = 1.0
            self._controls["baseline_mode"].value = "none"
            self._controls["clip_low"].value = 0.0
            self._controls["clip_high"].value = 100.0
            self._controls["soft_threshold"].value = 0.0
            self._controls["normalize"].value = True
            self._controls["log_scale"].value = False
            self._controls["show_peaks"].value = True
            self._controls["peak_prom"].value = 0.05
            self._controls["peak_dist"].value = 5
            self._controls["mode_view"].value = "all"
            self._controls["components"].value = tuple(self._available_components)
            self._controls["z_layer"].value = -1
        finally:
            self._internal_update = False

        self._on_refresh_clicked(_btn)

    def _on_save_animation_clicked(self, _btn: Any) -> None:
        """Save phase oscillation animation of selected mode."""
        on_save_animation_clicked(self, _btn)

    def _on_animate_clicked(self, _btn: Any) -> None:
        """Toggle live animation of selected mode."""
        on_animate_clicked(self, _btn)

    def _stop_animation(self) -> None:
        """Stop any running animation."""
        stop_animation(self)

    def _on_mode_type_changed(self, change: Any) -> None:
        """Handle mode visualization type change."""
        on_mode_type_changed(self, change)

    def _on_phase_index_changed(self, change: Any) -> None:
        """Handle phase index slider updates for static phase preview."""
        on_phase_index_changed(self, change)

    def _read_controls(self) -> None:
        """Read widget values into internal state."""
        read_controls(self)

    def _refresh_freq_slider_bounds(self) -> None:
        refresh_freq_slider_bounds(self)

    # ---------------------------------------------------------------------
    # Figure rendering and interaction
    # ---------------------------------------------------------------------
    def _resolve_mode_rows(self, mode_view: str) -> list[str]:
        return resolve_mode_rows(mode_view)

    def _render_figure(self) -> None:
        """Render spectrum + mode figure (in output widget or directly)."""
        render_figure(
            self,
            clear_output_fn=clear_output,
            plt_module=plt,
            grid_spec_cls=GridSpec,
        )

    def _create_figure(self, n_rows: int, n_components: int) -> None:
        """Create matplotlib figure and axes layout."""
        create_figure(
            self,
            n_rows=n_rows,
            n_components=n_components,
            plt_module=plt,
            grid_spec_cls=GridSpec,
        )

    def _draw_spectrum(self) -> None:
        """Draw filtered spectrum traces and peak markers."""
        draw_spectrum(self)

    def _draw_frequency_line(self) -> None:
        """Draw or update current frequency indicator line."""
        draw_frequency_line(self)

    def _load_mode(self, frequency_ghz: float, z_layer: int) -> tuple[np.ndarray, float, tuple[float, float, float, float]]:
        """Load mode array and metadata at selected frequency."""
        if self.analyzer is not None:
            mode_data = self.analyzer.get_mode(frequency_ghz, z_layer)
            mode_array = np.asarray(mode_data.mode_array)
            extent = tuple(mode_data.extent)
            actual = float(mode_data.frequency)
            return mode_array, actual, extent

        if self.data_loader is not None:
            mode_array, actual, _meta = self.data_loader.load_mode_at_frequency(frequency_ghz, z_layer)
            arr = np.asarray(mode_array)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            ny, nx = arr.shape[:2]
            extent = (0.0, float(nx), 0.0, float(ny))
            return arr, float(actual), extent

        raise RuntimeError("No analyzer/data loader available for mode visualization")

    def _update_mode_plots(self) -> None:
        """Render mode maps for selected frequency."""
        update_mode_plots(self)

    def _update_status_text(self) -> None:
        update_status_text(self, logger=log)

    def _set_status(self, message: str, color: str = "#334155") -> None:
        """Set status message in toolbar or fallback to logger."""
        set_status(self, message, color=color, logger=log)

    def _update_frequency_selection(self, redraw_canvas: bool = True) -> None:
        """Update vertical line and mode maps after frequency change."""
        update_frequency_selection(self, redraw_canvas=redraw_canvas)

    def _on_click(self, event: Any) -> None:
        """Handle spectrum click interactions."""
        on_spectrum_click(self, event)

    def _closest_freq_index(self, freq_ghz: Optional[float]) -> int:
        return closest_freq_index(self, freq_ghz)

    def _cleanup_figure_connections(self) -> None:
        """Clean up previous figure resources before re-render."""
        if self._fig is None:
            return
        try:
            plt.close(self._fig)
        except Exception:
            pass

    @staticmethod
    def _get_freq_scale(freq_unit: str) -> float:
        """Convert GHz to display unit scaling factor."""
        mapping = {
            "hz": 1e9,
            "khz": 1e6,
            "mhz": 1e3,
            "ghz": 1.0,
            "thz": 1e-3,
        }
        return float(mapping.get(str(freq_unit).lower(), 1.0))


# Backward-compatible alias

def plot(
    data_loader: Any,
    log_scale: bool = False,
    normalize: bool = True,
    freq_unit: str = "GHz",
    show_peaks: bool = True,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    smooth_filter: str = "none",
    smooth_window: int = 7,
    smooth_sigma: float = 1.0,
    baseline_mode: str = "none",
    clip_percentile_low: float = 0.0,
    clip_percentile_high: float = 100.0,
    soft_threshold_percentile: float = 0.0,
    peak_prominence: float = 0.05,
    peak_distance: int = 5,
    title: Optional[str] = None,
    dpi: int = 100,
    figsize: Tuple[float, float] = (12.0, 6.0),
) -> Figure:
    """Simple static spectrum plot compatibility helper."""
    return plot_compat(
        data_loader=data_loader,
        log_scale=log_scale,
        normalize=normalize,
        freq_unit=freq_unit,
        show_peaks=show_peaks,
        freq_min=freq_min,
        freq_max=freq_max,
        smooth_filter=smooth_filter,
        smooth_window=smooth_window,
        smooth_sigma=smooth_sigma,
        baseline_mode=baseline_mode,
        clip_percentile_low=clip_percentile_low,
        clip_percentile_high=clip_percentile_high,
        soft_threshold_percentile=soft_threshold_percentile,
        peak_prominence=peak_prominence,
        peak_distance=peak_distance,
        title=title,
        dpi=dpi,
        figsize=figsize,
    )
