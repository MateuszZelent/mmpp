"""Interactive FMR spectrum explorer.

This module provides a refactored interactive UI for FMR spectrum analysis.
It supports two operation modes:

- Toolbar mode (ipywidgets): interactive controls, filtering, and sweep animation
- Classic matplotlib mode: click-to-select spectrum with mode panels
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import logging
import numpy as np

log = logging.getLogger("mmpp.fft.modes")

# Component labels
COMPONENT_LABELS = [r"$m_x$", r"$m_y$", r"$m_z$"]
COMPONENT_NAMES = ["x", "y", "z"]
_COMPONENT_INDEX = {name: idx for idx, name in enumerate(COMPONENT_NAMES)}

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

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:  # pragma: no cover - optional dependency
    gaussian_filter1d = None

try:
    from scipy.signal import find_peaks as scipy_find_peaks
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - optional dependency
    scipy_find_peaks = None
    savgol_filter = None


@dataclass
class SpectrumFilterState:
    """Runtime filter state for spectrum processing."""

    freq_min: float
    freq_max: float
    smooth_filter: str = "none"
    smooth_window: int = 7
    smooth_sigma: float = 1.0
    baseline_mode: str = "none"
    clip_percentile_low: float = 0.0
    clip_percentile_high: float = 100.0
    soft_threshold_percentile: float = 0.0
    normalize: bool = True
    log_scale: bool = False


def _component_from_label(label: Optional[str]) -> Optional[str]:
    """Infer component key from a label like '$m_x$' or 'my'."""
    if not label:
        return None
    text = str(label).lower().replace("$", "")
    if "x" in text:
        return "x"
    if "y" in text:
        return "y"
    if "z" in text:
        return "z"
    return None


def _to_ghz(frequencies: np.ndarray) -> np.ndarray:
    """Convert frequencies to GHz when input appears to be in Hz."""
    freqs = np.asarray(frequencies, dtype=float)
    if freqs.size == 0:
        return freqs

    max_abs = float(np.nanmax(np.abs(freqs)))
    if max_abs > 1e6:
        return freqs / 1e9
    return freqs


def _to_power(spectrum: np.ndarray) -> np.ndarray:
    """Convert complex or amplitude spectrum to non-negative power-like data."""
    spec = np.asarray(spectrum)
    if spec.size == 0:
        return spec.astype(float)

    if np.iscomplexobj(spec):
        return np.abs(spec) ** 2

    spec = np.asarray(spec, dtype=float)
    if np.nanmin(spec) < 0:
        return np.abs(spec)
    return spec


def normalize_component_selection(
    components: Optional[Sequence[Union[int, str]]],
    available: Optional[Sequence[str]] = None,
) -> list[str]:
    """Normalize mixed component input to canonical ['x', 'y', 'z'] subset."""
    if components is None:
        normalized = ["x", "y", "z"]
    else:
        normalized = []
        for comp in components:
            key: Optional[str] = None
            if isinstance(comp, int):
                if 0 <= comp <= 2:
                    key = COMPONENT_NAMES[comp]
            elif isinstance(comp, str):
                text = comp.strip().lower().replace("_", "")
                if text.startswith("m") and len(text) > 1:
                    text = text[1:]
                if text in _COMPONENT_INDEX:
                    key = text

            if key is not None and key not in normalized:
                normalized.append(key)

    if available:
        allowed = [c for c in normalized if c in available]
        if allowed:
            return allowed
        return [available[0]]

    return normalized or ["z"]


def collapse_spectrum_components(
    spectrum_power: np.ndarray,
    component_hint: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """Collapse arbitrary spectrum shapes into per-component 1D traces."""
    spec = np.asarray(spectrum_power)
    if spec.ndim == 0:
        return {component_hint or "z": np.asarray([float(spec)])}

    if spec.ndim == 1:
        key = component_hint or "z"
        return {key: spec.astype(float, copy=False)}

    if spec.shape[-1] <= 3:
        if spec.ndim == 2:
            traces = spec
        else:
            spatial_axes = tuple(range(1, spec.ndim - 1))
            traces = np.mean(spec, axis=spatial_axes)

        out: dict[str, np.ndarray] = {}
        n_comp = min(traces.shape[-1], 3)
        for idx in range(n_comp):
            out[COMPONENT_NAMES[idx]] = np.asarray(traces[:, idx], dtype=float)
        return out

    # Generic high-dimensional fallback: average over non-frequency axes.
    reduced = np.mean(spec, axis=tuple(range(1, spec.ndim)))
    key = component_hint or "z"
    return {key: np.asarray(reduced, dtype=float)}


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average smoothing."""
    if window <= 1 or values.size < 3:
        return values

    window = max(1, int(window))
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: values.size]


def _apply_smoothing(
    values: np.ndarray,
    smooth_filter: str,
    smooth_window: int,
    smooth_sigma: float,
) -> np.ndarray:
    """Apply selected smoothing filter."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    mode = (smooth_filter or "none").lower()
    if mode == "none":
        return arr

    if mode == "moving_average":
        window = max(1, int(smooth_window))
        if window % 2 == 0:
            window += 1
        return _moving_average(arr, window)

    if mode == "gaussian":
        sigma = max(0.0, float(smooth_sigma))
        if sigma == 0.0:
            return arr
        if gaussian_filter1d is None:
            # Fallback approximation when scipy is unavailable.
            window = max(3, int(round(4 * sigma)) | 1)
            return _moving_average(arr, window)
        return gaussian_filter1d(arr, sigma=sigma, mode="nearest")

    if mode == "savgol":
        window = max(3, int(smooth_window))
        if window % 2 == 0:
            window += 1
        if window >= arr.size:
            window = max(3, arr.size - (1 - arr.size % 2))
        if window < 3:
            return arr
        if savgol_filter is None:
            return _moving_average(arr, window)
        polyorder = 2 if window > 3 else 1
        return savgol_filter(arr, window_length=window, polyorder=polyorder, mode="interp")

    return arr


def _remove_baseline(values: np.ndarray, baseline_mode: str) -> np.ndarray:
    """Apply baseline correction to 1D spectrum trace."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    mode = (baseline_mode or "none").lower()
    if mode == "none":
        return arr

    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)

    out = arr.copy()
    baseline = float(np.nanmedian(out[finite]))

    if mode == "mean":
        baseline = float(np.nanmean(out[finite]))
        out = out - baseline
    elif mode == "median":
        baseline = float(np.nanmedian(out[finite]))
        out = out - baseline
    elif mode == "linear":
        x = np.arange(out.size, dtype=float)[finite]
        y = out[finite]
        if x.size >= 2:
            try:
                coeff = np.polyfit(x, y, deg=1)
                trend = np.polyval(coeff, np.arange(out.size, dtype=float))
                out = out - trend
            except Exception:
                out = out - baseline
        else:
            out = out - baseline
    else:
        out = out - baseline

    # Keep the spectrum in non-negative domain after detrending.
    min_val = float(np.nanmin(out))
    if np.isfinite(min_val) and min_val < 0:
        out = out - min_val
    return out


def _apply_percentile_clip(
    values: np.ndarray,
    clip_percentile_low: float,
    clip_percentile_high: float,
) -> np.ndarray:
    """Clip values to robust percentile range and re-zero output."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    low = float(np.clip(min(clip_percentile_low, clip_percentile_high), 0.0, 100.0))
    high = float(np.clip(max(clip_percentile_low, clip_percentile_high), 0.0, 100.0))
    if low <= 0.0 and high >= 100.0:
        return arr

    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)

    lo_val = float(np.nanpercentile(arr[finite], low))
    hi_val = float(np.nanpercentile(arr[finite], high))
    if hi_val < lo_val:
        hi_val = lo_val

    clipped = np.clip(arr, lo_val, hi_val)
    if lo_val != 0.0:
        clipped = clipped - lo_val
    return clipped


def _apply_soft_threshold(values: np.ndarray, percentile: float) -> np.ndarray:
    """Suppress weak components using soft thresholding."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    pct = float(np.clip(percentile, 0.0, 100.0))
    if pct <= 0.0:
        return arr

    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)

    threshold = float(np.nanpercentile(arr[finite], pct))
    if not np.isfinite(threshold) or threshold <= 0:
        return arr

    # Soft-threshold keeps stronger peaks but attenuates low-level noise.
    return np.sign(arr) * np.maximum(np.abs(arr) - threshold, 0.0)


def apply_spectrum_filters(
    frequencies_ghz: np.ndarray,
    component_power: dict[str, np.ndarray],
    filters: SpectrumFilterState,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply frequency range, smoothing, normalization and optional log transform."""
    freqs = np.asarray(frequencies_ghz, dtype=float)
    if freqs.size == 0:
        return freqs, {key: np.asarray(val, dtype=float) for key, val in component_power.items()}

    fmin = float(min(filters.freq_min, filters.freq_max))
    fmax = float(max(filters.freq_min, filters.freq_max))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        mask = np.ones_like(freqs, dtype=bool)

    filtered_freqs = freqs[mask]
    output: dict[str, np.ndarray] = {}

    for comp, values in component_power.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] != freqs.shape[0]:
            length = min(arr.shape[0], freqs.shape[0])
            arr = arr[:length]
            local_mask = mask[:length]
            local_freqs = freqs[:length]
        else:
            local_mask = mask
            local_freqs = freqs

        sub = arr[local_mask]
        if sub.size == 0:
            sub = np.asarray([0.0], dtype=float)
            local_freqs = np.asarray([fmin], dtype=float)
            filtered_freqs = local_freqs

        sub = _apply_smoothing(
            sub,
            smooth_filter=filters.smooth_filter,
            smooth_window=filters.smooth_window,
            smooth_sigma=filters.smooth_sigma,
        )
        sub = _remove_baseline(sub, filters.baseline_mode)
        sub = _apply_percentile_clip(
            sub,
            clip_percentile_low=filters.clip_percentile_low,
            clip_percentile_high=filters.clip_percentile_high,
        )
        sub = _apply_soft_threshold(sub, filters.soft_threshold_percentile)

        finite = np.isfinite(sub)
        if np.any(finite):
            sub = np.where(finite, sub, np.nanmedian(sub[finite]))
        else:
            sub = np.zeros_like(sub)

        min_val = float(np.nanmin(sub)) if sub.size else 0.0
        if np.isfinite(min_val) and min_val < 0:
            sub = sub - min_val

        if filters.normalize and sub.size:
            vmax = float(np.nanmax(sub))
            if vmax > 0:
                sub = sub / vmax

        if filters.log_scale:
            sub = np.log10(np.clip(sub, 1e-12, None))

        output[comp] = np.asarray(sub, dtype=float)

        if filtered_freqs.shape[0] != local_freqs[local_mask].shape[0]:
            filtered_freqs = local_freqs[local_mask]

    return filtered_freqs, output


def detect_spectrum_peaks(
    frequencies_ghz: np.ndarray,
    spectrum_1d: np.ndarray,
    min_prominence: float = 0.05,
    min_distance: int = 5,
) -> list[tuple[float, float]]:
    """Detect local peaks and return list of ``(frequency, amplitude)`` tuples."""
    freqs = np.asarray(frequencies_ghz, dtype=float)
    values = np.asarray(spectrum_1d, dtype=float)

    if freqs.size < 3 or values.size < 3 or freqs.shape[0] != values.shape[0]:
        return []

    finite = np.isfinite(values)
    if not np.any(finite):
        return []

    data = values.copy()
    data[~finite] = np.nanmin(data[finite])

    if scipy_find_peaks is not None:
        max_val = float(np.nanmax(data))
        prominence = float(min_prominence)
        if 0 < prominence < 1 and max_val > 1:
            prominence = prominence * max_val

        try:
            idx, _ = scipy_find_peaks(
                data,
                prominence=max(prominence, 0.0),
                distance=max(int(min_distance), 1),
            )
        except Exception:
            idx = np.array([], dtype=int)
    else:
        idx = []
        threshold = np.nanmin(data) + float(min_prominence) * (
            np.nanmax(data) - np.nanmin(data)
        )
        for i in range(1, data.size - 1):
            if (
                data[i] > data[i - 1]
                and data[i] > data[i + 1]
                and data[i] >= threshold
            ):
                idx.append(i)
        idx = np.asarray(idx, dtype=int)

    peaks = [(float(freqs[i]), float(data[i])) for i in idx]
    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks


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
        freqs: Optional[np.ndarray] = None
        spectrum: Optional[np.ndarray] = None
        component_hint = _component_from_label(self._component_label)

        if self.spectrum_result is not None:
            freqs = np.asarray(getattr(self.spectrum_result, "frequencies", []), dtype=float)
            if hasattr(self.spectrum_result, "power"):
                spectrum = np.asarray(self.spectrum_result.power)
            elif hasattr(self.spectrum_result, "spectrum"):
                spectrum = _to_power(np.asarray(self.spectrum_result.spectrum))
            else:
                spectrum = None

            hint_from_result = _component_from_label(
                getattr(self.spectrum_result, "component_label", None)
            )
            component_hint = hint_from_result or component_hint

        if (freqs is None or spectrum is None) and self.data_loader is not None:
            loaded_freqs, loaded_spectrum, loaded_label = self.data_loader.load_spectrum()
            freqs = np.asarray(loaded_freqs, dtype=float)
            spectrum = _to_power(np.asarray(loaded_spectrum))
            component_hint = _component_from_label(loaded_label) or component_hint

        if (freqs is None or spectrum is None) and self.analyzer is not None:
            if getattr(self.analyzer, "frequencies", None) is not None:
                freqs = np.asarray(self.analyzer.frequencies, dtype=float)
            if getattr(self.analyzer, "spectrum", None) is not None:
                spectrum = _to_power(np.asarray(self.analyzer.spectrum))

        if freqs is None or spectrum is None:
            raise ValueError(
                "No spectrum data available. Provide spectrum_result, data_loader, or analyzer."
            )

        freqs_ghz = _to_ghz(freqs)
        component_power = collapse_spectrum_components(_to_power(spectrum), component_hint)

        # Align component traces to frequency axis length.
        trimmed: dict[str, np.ndarray] = {}
        for comp, values in component_power.items():
            arr = np.asarray(values, dtype=float)
            length = min(arr.shape[0], freqs_ghz.shape[0])
            if length == 0:
                continue
            trimmed[comp] = arr[:length]

        if not trimmed:
            raise ValueError("Spectrum data is empty after preprocessing")

        min_len = min(trace.shape[0] for trace in trimmed.values())
        self._raw_frequencies_ghz = freqs_ghz[:min_len]
        self._raw_component_power = {k: v[:min_len] for k, v in trimmed.items()}
        self._available_components = list(self._raw_component_power.keys())

    def _recompute_filtered_spectrum(self) -> None:
        """Recompute filtered traces and peak list from current filter state."""
        self._filtered_frequencies_ghz, filtered = apply_spectrum_filters(
            self._raw_frequencies_ghz,
            self._raw_component_power,
            self._filter_state,
        )

        selected = {
            comp: values
            for comp, values in filtered.items()
            if comp in self._current_components
        }
        if not selected:
            fallback_key = self._available_components[0]
            selected = {fallback_key: filtered[fallback_key]}
            self._current_components = [fallback_key]

        self._filtered_component_power = selected

        if self._show_peaks:
            stacked = np.vstack(list(selected.values()))
            avg_trace = np.mean(stacked, axis=0)
            self._peaks = detect_spectrum_peaks(
                self._filtered_frequencies_ghz,
                avg_trace,
                min_prominence=self._peak_prominence,
                min_distance=self._peak_distance,
            )
        else:
            self._peaks = []

    def _initialize_frequency(self, initial_frequency: Optional[float]) -> None:
        """Set current frequency based on initial request, peaks, or center."""
        if initial_frequency is not None:
            self._current_frequency_ghz = float(initial_frequency)
            return

        if self._peaks:
            self._current_frequency_ghz = float(self._peaks[0][0])
            return

        if self._filtered_frequencies_ghz.size:
            center = self._filtered_frequencies_ghz.size // 2
            self._current_frequency_ghz = float(self._filtered_frequencies_ghz[center])
            return

        self._current_frequency_ghz = None

    # ---------------------------------------------------------------------
    # Presets
    # ---------------------------------------------------------------------
    def _get_presets_dir(self) -> Path:
        """Return project-local presets directory."""
        if self._presets_dir is None:
            self._presets_dir = Path.cwd() / ".mmpp_presets"
            self._presets_dir.mkdir(parents=True, exist_ok=True)
        return self._presets_dir

    def _list_presets(self) -> list[str]:
        """List available interactive toolbar presets."""
        preset_dir = self._get_presets_dir()
        names = []
        for file_path in sorted(preset_dir.glob("fmr_*.json")):
            name = file_path.stem.removeprefix("fmr_")
            if name:
                names.append(name)
        return names

    def _collect_preset_state(self) -> dict[str, Any]:
        """Collect serializable state from current controls."""
        return {
            "components": list(self._current_components),
            "z_layer": int(self._current_z_layer),
            "freq_min": float(self._filter_state.freq_min),
            "freq_max": float(self._filter_state.freq_max),
            "smooth_filter": str(self._filter_state.smooth_filter),
            "smooth_window": int(self._filter_state.smooth_window),
            "smooth_sigma": float(self._filter_state.smooth_sigma),
            "baseline_mode": str(self._filter_state.baseline_mode),
            "clip_percentile_low": float(self._filter_state.clip_percentile_low),
            "clip_percentile_high": float(self._filter_state.clip_percentile_high),
            "soft_threshold_percentile": float(self._filter_state.soft_threshold_percentile),
            "normalize": bool(self._filter_state.normalize),
            "log_scale": bool(self._filter_state.log_scale),
            "show_peaks": bool(self._show_peaks),
            "peak_prominence": float(self._peak_prominence),
            "peak_distance": int(self._peak_distance),
            "mode_view": "all" if len(self._mode_row_types) > 1 else self._mode_row_types[0],
            "cmap_mag": str(self._controls.get("cmap_mag").value) if self._controls.get("cmap_mag") else "viridis",
            "cmap_phase": str(self._controls.get("cmap_phase").value) if self._controls.get("cmap_phase") else "twilight",
            "cmap_combined": str(self._controls.get("cmap_combined").value) if self._controls.get("cmap_combined") else "RdBu_r",
            "freq_unit": str(self._freq_unit),
        }

    def _apply_preset_state(self, payload: dict[str, Any]) -> None:
        """Apply preset payload to widgets/state."""
        if not self._controls:
            return

        self._internal_update = True
        try:
            components = normalize_component_selection(
                payload.get("components"),
                available=self._available_components or COMPONENT_NAMES,
            )
            self._controls["components"].value = tuple(components)
            z_control = self._controls["z_layer"]
            z_val = int(payload.get("z_layer", self._current_z_layer))
            self._controls["z_layer"].value = int(np.clip(z_val, z_control.min, z_control.max))

            fmin_control = self._controls["fmin"]
            fmax_control = self._controls["fmax"]
            fmin = float(payload.get("freq_min", self._filter_state.freq_min))
            fmax = float(payload.get("freq_max", self._filter_state.freq_max))
            self._controls["fmin"].value = float(np.clip(fmin, fmin_control.min, fmin_control.max))
            self._controls["fmax"].value = float(np.clip(fmax, fmax_control.min, fmax_control.max))

            smooth_filter = str(payload.get("smooth_filter", self._filter_state.smooth_filter))
            if smooth_filter not in [opt[1] for opt in self._controls["smooth_filter"].options]:
                smooth_filter = "none"
            self._controls["smooth_filter"].value = smooth_filter

            smooth_window = self._controls["smooth_window"]
            self._controls["smooth_window"].value = int(
                np.clip(
                    int(payload.get("smooth_window", self._filter_state.smooth_window)),
                    smooth_window.min,
                    smooth_window.max,
                )
            )

            smooth_sigma = self._controls["smooth_sigma"]
            self._controls["smooth_sigma"].value = float(
                np.clip(
                    float(payload.get("smooth_sigma", self._filter_state.smooth_sigma)),
                    smooth_sigma.min,
                    smooth_sigma.max,
                )
            )

            baseline_mode = str(payload.get("baseline_mode", self._filter_state.baseline_mode))
            if baseline_mode not in [opt[1] for opt in self._controls["baseline_mode"].options]:
                baseline_mode = "none"
            self._controls["baseline_mode"].value = baseline_mode
            clip_low = self._controls["clip_low"]
            clip_high = self._controls["clip_high"]
            soft_thr = self._controls["soft_threshold"]
            self._controls["clip_low"].value = float(
                np.clip(
                    float(payload.get("clip_percentile_low", self._filter_state.clip_percentile_low)),
                    clip_low.min,
                    clip_low.max,
                )
            )
            self._controls["clip_high"].value = float(
                np.clip(
                    float(payload.get("clip_percentile_high", self._filter_state.clip_percentile_high)),
                    clip_high.min,
                    clip_high.max,
                )
            )
            self._controls["soft_threshold"].value = float(
                np.clip(
                    float(
                        payload.get(
                            "soft_threshold_percentile",
                            self._filter_state.soft_threshold_percentile,
                        )
                    ),
                    soft_thr.min,
                    soft_thr.max,
                )
            )

            self._controls["normalize"].value = bool(payload.get("normalize", self._filter_state.normalize))
            self._controls["log_scale"].value = bool(payload.get("log_scale", self._filter_state.log_scale))
            self._controls["show_peaks"].value = bool(payload.get("show_peaks", self._show_peaks))
            peak_prom = self._controls["peak_prom"]
            self._controls["peak_prom"].value = float(
                np.clip(
                    float(payload.get("peak_prominence", self._peak_prominence)),
                    peak_prom.min,
                    peak_prom.max,
                )
            )
            peak_dist = self._controls["peak_dist"]
            self._controls["peak_dist"].value = int(
                np.clip(
                    int(payload.get("peak_distance", self._peak_distance)),
                    peak_dist.min,
                    peak_dist.max,
                )
            )

            mode_view = str(payload.get("mode_view", "all"))
            if mode_view not in [opt[1] for opt in self._controls["mode_view"].options]:
                mode_view = "all"
            self._controls["mode_view"].value = mode_view

            cmap_mag = str(payload.get("cmap_mag", "viridis"))
            if cmap_mag not in self._controls["cmap_mag"].options:
                cmap_mag = "viridis"
            self._controls["cmap_mag"].value = cmap_mag

            cmap_phase = str(payload.get("cmap_phase", "twilight"))
            if cmap_phase not in self._controls["cmap_phase"].options:
                cmap_phase = "twilight"
            self._controls["cmap_phase"].value = cmap_phase

            cmap_combined = str(payload.get("cmap_combined", "RdBu_r"))
            if cmap_combined not in self._controls["cmap_combined"].options:
                cmap_combined = "RdBu_r"
            self._controls["cmap_combined"].value = cmap_combined
        finally:
            self._internal_update = False

        self._read_controls()
        self._recompute_filtered_spectrum()
        self._refresh_freq_slider_bounds()
        self._render_figure()

    def _refresh_preset_options(self) -> None:
        """Refresh preset dropdown options."""
        if "preset_select" not in self._controls:
            return
        options = [("-- load preset --", "")] + [(name, name) for name in self._list_presets()]
        current = self._controls["preset_select"].value
        self._controls["preset_select"].options = options
        if current not in [opt[1] for opt in options]:
            self._controls["preset_select"].value = ""

    def _on_save_preset_clicked(self, _btn: Any) -> None:
        """Persist current toolbar config as a preset."""
        if not self._controls:
            return

        name = str(self._controls["preset_name"].value).strip()
        if not name:
            self._set_status("Preset name required", color="darkorange")
            return

        safe_name = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_")).strip("_-")
        if not safe_name:
            self._set_status("Preset name contains invalid characters", color="crimson")
            return

        payload = self._collect_preset_state()
        payload["saved_at"] = datetime.now().isoformat()

        preset_path = self._get_presets_dir() / f"fmr_{safe_name}.json"
        try:
            preset_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self._set_status(f"Failed to save preset: {exc}", color="crimson")
            return

        self._controls["preset_name"].value = ""
        self._refresh_preset_options()
        self._controls["preset_select"].value = safe_name
        self._set_status(f"Preset saved: {preset_path.name}", color="seagreen")

    def _on_load_preset_changed(self, change: Any) -> None:
        """Load selected preset and apply values to toolbar."""
        if change.get("name") != "value":
            return
        name = str(change.get("new") or "").strip()
        if not name:
            return

        preset_path = self._get_presets_dir() / f"fmr_{name}.json"
        if not preset_path.exists():
            self._set_status(f"Preset not found: {name}", color="crimson")
            self._refresh_preset_options()
            return

        try:
            payload = json.loads(preset_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._set_status(f"Failed to load preset: {exc}", color="crimson")
            return

        self._apply_preset_state(payload)
        self._set_status(f"Preset loaded: {name}", color="seagreen")

    def _on_delete_preset_clicked(self, _btn: Any) -> None:
        """Delete selected preset file."""
        if not self._controls:
            return

        name = str(self._controls["preset_select"].value or "").strip()
        if not name:
            self._set_status("Select preset to delete", color="darkorange")
            return

        preset_path = self._get_presets_dir() / f"fmr_{name}.json"
        if not preset_path.exists():
            self._set_status(f"Preset not found: {name}", color="crimson")
            self._refresh_preset_options()
            return

        try:
            preset_path.unlink()
        except Exception as exc:
            self._set_status(f"Failed to delete preset: {exc}", color="crimson")
            return

        self._refresh_preset_options()
        self._set_status(f"Preset deleted: {name}", color="seagreen")

    # ---------------------------------------------------------------------
    # Widget toolbar
    # ---------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        """Build ipywidgets toolbar UI."""
        if not _HAS_WIDGETS:
            raise RuntimeError("ipywidgets is required for toolbar mode")

        fmin = float(np.nanmin(self._raw_frequencies_ghz))
        fmax = float(np.nanmax(self._raw_frequencies_ghz))

        z_min, z_max = self._guess_layer_bounds()

        controls: dict[str, Any] = {}
        controls["components"] = widgets.SelectMultiple(
            options=[(f"m_{name}", name) for name in self._available_components],
            value=tuple(self._current_components),
            description="Comp:",
            layout=widgets.Layout(width="100%", height="90px"),
            style={"description_width": "55px"},
        )
        controls["z_layer"] = widgets.IntSlider(
            value=self._current_z_layer,
            min=z_min,
            max=z_max,
            step=1,
            description="z:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
            continuous_update=False,
        )

        controls["fmin"] = widgets.FloatSlider(
            value=self._filter_state.freq_min,
            min=fmin,
            max=fmax,
            step=max((fmax - fmin) / 400.0, 1e-4),
            description="f min:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
            continuous_update=False,
        )
        controls["fmax"] = widgets.FloatSlider(
            value=self._filter_state.freq_max,
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
            value=self._filter_state.smooth_filter,
            description="smooth:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
        )
        controls["smooth_window"] = widgets.IntSlider(
            value=self._filter_state.smooth_window,
            min=3,
            max=61,
            step=2,
            description="window:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
            continuous_update=False,
        )
        controls["smooth_sigma"] = widgets.FloatSlider(
            value=self._filter_state.smooth_sigma,
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
            value=self._filter_state.baseline_mode,
            description="baseline:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
        )
        controls["clip_low"] = widgets.FloatSlider(
            value=self._filter_state.clip_percentile_low,
            min=0.0,
            max=50.0,
            step=0.5,
            description="clip lo:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
            continuous_update=False,
        )
        controls["clip_high"] = widgets.FloatSlider(
            value=self._filter_state.clip_percentile_high,
            min=50.0,
            max=100.0,
            step=0.5,
            description="clip hi:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
            continuous_update=False,
        )
        controls["soft_threshold"] = widgets.FloatSlider(
            value=self._filter_state.soft_threshold_percentile,
            min=0.0,
            max=100.0,
            step=1.0,
            description="soft thr:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
            continuous_update=False,
        )

        controls["normalize"] = widgets.Checkbox(
            value=self._filter_state.normalize,
            description="normalize",
            layout=widgets.Layout(width="100%"),
        )
        controls["log_scale"] = widgets.Checkbox(
            value=self._filter_state.log_scale,
            description="log10",
            layout=widgets.Layout(width="100%"),
        )
        controls["show_peaks"] = widgets.Checkbox(
            value=self._show_peaks,
            description="show peaks",
            layout=widgets.Layout(width="100%"),
        )
        controls["peak_prom"] = widgets.FloatSlider(
            value=self._peak_prominence,
            min=0.0,
            max=1.0,
            step=0.01,
            description="prom:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
            continuous_update=False,
        )
        controls["peak_dist"] = widgets.IntSlider(
            value=self._peak_distance,
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
            value="all" if len(self._mode_row_types) > 1 else self._mode_row_types[0],
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
            value=self._mode_aspect if self._mode_aspect in ["equal", "auto"] else "equal",
            description="aspect:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
        )
        controls["layout"] = widgets.Dropdown(
            options=["vertical", "horizontal"],
            value=self._layout_mode,
            description="layout:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
        )

        controls["freq_index"] = widgets.IntSlider(
            value=max(self._closest_freq_index(self._current_frequency_ghz), 0),
            min=0,
            max=max(int(self._filtered_frequencies_ghz.size) - 1, 0),
            step=1,
            description="freq:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "55px"},
            continuous_update=False,
        )
        # Phase animation slider (0-359 degrees / frames)
        n_anim_frames = 60  # Default frame count for phase animation
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
            interval=42,  # ~24 fps
            description="phase",
            disabled=False,
        )
        # Link Play to phase_index (NOT freq_index!)
        widgets.jslink((controls["play"], "value"), (controls["phase_index"], "value"))
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
            options=[("gif", "gif"), ("mp4", "mp4")],
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
            value="<small>Left-click spectrum to select frequency, right-click to snap to nearest peak.</small>",
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

        # Callbacks
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
            "cmap_mag",
            "cmap_phase",
            "cmap_combined",
        ]
        for key in observe_keys:
            controls[key].observe(self._on_controls_changed, names="value")

        controls["freq_index"].observe(self._on_frequency_index_changed, names="value")
        controls["refresh"].on_click(self._on_refresh_clicked)
        controls["reset"].on_click(self._on_reset_clicked)
        controls["save_animation"].on_click(self._on_save_animation_clicked)
        controls["animate"].on_click(self._on_animate_clicked)
        controls["mode_type"].observe(self._on_mode_type_changed, names="value")
        controls["phase_index"].observe(self._on_phase_index_changed, names="value")
        controls["preset_save"].on_click(self._on_save_preset_clicked)
        controls["preset_delete"].on_click(self._on_delete_preset_clicked)
        controls["preset_select"].observe(self._on_load_preset_changed, names="value")

        self._widget_output = widgets.Output(
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
                        controls["freq_index"],  # Frequency selection (separate)
                        widgets.HBox([controls["play"], controls["phase_index"]]),  # Phase animation
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
                widgets.HTML("<b>FMR Spectrum Toolbar</b>"),
                preset_box,
                sections,
                widgets.HBox([controls["refresh"], controls["reset"]]),
                controls["status"],
            ],
            layout=widgets.Layout(width="330px", border="1px solid #ddd", padding="8px"),
        )

        right_panel = widgets.VBox(
            [self._widget_output],
            layout=widgets.Layout(width="calc(100% - 350px)", min_width="680px"),
        )

        self._widget_root = widgets.HBox(
            [control_panel, right_panel],
            layout=widgets.Layout(width="100%"),
        )

        self._controls = controls
        self._refresh_preset_options()

    def _guess_layer_bounds(self) -> tuple[int, int]:
        """Best-effort z-layer slider bounds."""
        try:
            if self.analyzer is not None and getattr(self.analyzer, "modes_path", None):
                modes_path = self.analyzer.modes_path
                shape = self.analyzer.zarr_file[modes_path].shape
                n_layers = int(shape[1])
                if n_layers > 0:
                    return -n_layers, n_layers - 1
        except Exception:
            pass
        return -10, 10

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
        """Save phase oscillation animation of the FMR mode at selected frequency.
        
        Animates mode through one full period (0-360° phase) at the currently
        selected frequency, similar to dispersion module animation.
        Uses: mode * exp(-i*omega*t) for time evolution.
        """
        if self._is_saving_animation:
            return
        if self._fig is None or self._current_frequency_ghz is None:
            self._set_status("No mode selected to animate", color="crimson")
            return
        if "save_animation" not in self._controls:
            return

        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
            try:
                from matplotlib.animation import FFMpegWriter
            except Exception:  # pragma: no cover - optional backend
                FFMpegWriter = None  # type: ignore[assignment]
        except Exception as exc:  # pragma: no cover - optional backend
            self._set_status(f"Animation backend unavailable: {exc}", color="crimson")
            return

        n_frames = max(2, int(self._controls["anim_frames"].value))
        fps = max(1, int(self._controls["anim_fps"].value))
        fmt = str(self._controls["anim_format"].value).lower()

        button = self._controls["save_animation"]
        old_desc = button.description

        self._is_saving_animation = True
        button.disabled = True
        button.description = "Loading mode..."
        self._set_status("Loading mode data...", color="#0F766E")

        try:
            # ============================================================
            # LOAD COMPLEX MODE AT SELECTED FREQUENCY
            # ============================================================
            freq_ghz = self._current_frequency_ghz
            mode_array, actual_freq, extent = self._load_mode(freq_ghz, self._current_z_layer)
            
            # mode_array is (ny, nx, n_components) complex
            # actual_freq is in GHz
            freq_hz = actual_freq * 1e9  # Convert to Hz
            omega = 2 * np.pi * freq_hz  # Angular frequency
            period_s = 1.0 / freq_hz  # One full oscillation period
            
            # Time array for one complete cycle (0 to 2π phase)
            time_array = np.linspace(0, period_s, n_frames, endpoint=False)
            
            button.description = "Pre-computing..."
            self._set_status(f"Pre-computing {n_frames} frames...", color="#0F766E")
            
            # ============================================================
            # PRE-COMPUTE ALL FRAMES (mode * exp(-i*omega*t))
            # ============================================================
            precomputed_frames = []
            for i, t in enumerate(time_array):
                # Phase evolution: multiply by exp(-i*omega*t)
                phase_factor = np.exp(-1j * omega * t)
                mode_at_t = mode_array * phase_factor  # Still complex
                precomputed_frames.append(mode_at_t)
                
                if i % max(1, n_frames // 10) == 0:
                    button.description = f"Frame {i+1}/{n_frames}"
            
            button.description = "Rendering..."
            self._set_status("Rendering animation...", color="#0F766E")
            
            # ============================================================
            # SETUP FIGURE FOR ANIMATION (get image references)
            # ============================================================
            mode_images = []
            mode_titles = []
            
            if self._mode_axes is not None:
                for row_idx, row_type in enumerate(self._mode_row_types):
                    row_images = []
                    row_titles = []
                    for col_idx in range(self._mode_axes.shape[1]):
                        ax = self._mode_axes[row_idx, col_idx]
                        # Find the imshow AxesImage
                        for child in ax.get_children():
                            from matplotlib.image import AxesImage
                            if isinstance(child, AxesImage):
                                row_images.append(child)
                                break
                        else:
                            row_images.append(None)
                        row_titles.append(ax.title)
                    mode_images.append(row_images)
                    mode_titles.append(row_titles)
            
            # ============================================================
            # ANIMATION UPDATE FUNCTION
            # ============================================================
            def _update_frame(frame_idx: int) -> list[Any]:
                mode_at_t = precomputed_frames[frame_idx]
                t = time_array[frame_idx]
                phase_deg = (t / period_s) * 360  # Phase in degrees
                
                artists = []
                
                for row_idx, row_type in enumerate(self._mode_row_types):
                    for col_idx, comp in enumerate(self._current_components):
                        if row_idx >= len(mode_images) or col_idx >= len(mode_images[row_idx]):
                            continue
                        img = mode_images[row_idx][col_idx]
                        if img is None:
                            continue
                        
                        comp_idx = _COMPONENT_INDEX.get(comp, 0)
                        if comp_idx >= mode_at_t.shape[2]:
                            continue
                        
                        comp_data = mode_at_t[:, :, comp_idx]
                        
                        # Select visualization based on row type
                        if row_type == "magnitude":
                            # Magnitude is constant over time (envelope)
                            plot_data = np.abs(comp_data)
                        elif row_type == "phase":
                            # Phase evolves linearly with time
                            plot_data = np.angle(comp_data)
                        else:  # combined - this shows the oscillation!
                            # Real part shows the actual oscillating magnetization
                            plot_data = np.real(comp_data)
                        
                        img.set_data(plot_data)
                        artists.append(img)
                        
                        # Update title with phase info
                        if row_idx == 0 and row_idx < len(mode_titles):
                            title_obj = mode_titles[row_idx][col_idx]
                            title_obj.set_text(f"m_{comp} @ {actual_freq:.3f} GHz (φ={phase_deg:.0f}°)")
                            artists.append(title_obj)
                
                return artists

            # ============================================================
            # CREATE ANIMATION WITH BLIT=TRUE
            # ============================================================
            animation = FuncAnimation(
                self._fig,
                _update_frame,
                frames=n_frames,
                interval=1000.0 / float(fps),
                blit=True,
                repeat=False,
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path.cwd() / f"fmr_mode_{actual_freq:.2f}GHz_{timestamp}.{fmt}"

            if fmt == "mp4":
                if FFMpegWriter is None:
                    raise RuntimeError("FFmpeg writer unavailable; select GIF format")
                writer = FFMpegWriter(fps=fps, bitrate=3000)
            else:
                writer = PillowWriter(fps=fps)

            animation.save(str(output_path), writer=writer, dpi=self.dpi)
            size_mb = output_path.stat().st_size / (1024 * 1024)
            self._set_status(
                f"Saved: {output_path.name} ({size_mb:.1f} MB)",
                color="seagreen",
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._set_status(f"Animation failed: {exc}", color="crimson")
        finally:
            button.disabled = False
            button.description = old_desc
            self._is_saving_animation = False

    def _on_animate_clicked(self, _btn: Any) -> None:
        """Toggle live animation of the selected mode (phase oscillation).
        
        Matches dispersion module's _on_animate() pattern.
        """
        if self._fig is None or self._current_frequency_ghz is None:
            self._set_status("No mode selected to animate", color="crimson")
            return
        
        # Toggle animation on/off
        if self._is_animating:
            self._stop_animation()
            if "animate" in self._controls:
                self._controls["animate"].description = "🎬 Animate"
                self._controls["animate"].button_style = "warning"
            self._set_status("Animation stopped", color="seagreen")
            # Restore static view
            self._update_mode_plots()
            return
        
        try:
            from matplotlib.animation import FuncAnimation
            
            # Get parameters
            freq_ghz = self._current_frequency_ghz
            mode_array, actual_freq, extent = self._load_mode(freq_ghz, self._current_z_layer)
            
            freq_hz = actual_freq * 1e9
            omega = 2 * np.pi * freq_hz
            period_s = 1.0 / freq_hz
            
            n_frames = int(self._controls.get("anim_frames", {}).value if hasattr(self._controls.get("anim_frames"), "value") else 60)
            fps = int(self._controls.get("anim_fps", {}).value if hasattr(self._controls.get("anim_fps"), "value") else 24)
            
            # Time array for one complete cycle
            time_array = np.linspace(0, period_s, n_frames, endpoint=False)
            
            # Pre-compute all frames
            precomputed_frames = []
            for t in time_array:
                phase_factor = np.exp(-1j * omega * t)
                mode_at_t = mode_array * phase_factor
                precomputed_frames.append(mode_at_t)
            
            # Get image references from mode axes
            mode_images = []
            mode_titles = []
            
            if self._mode_axes is not None:
                for row_idx, row_type in enumerate(self._mode_row_types):
                    row_images = []
                    row_titles = []
                    for col_idx in range(self._mode_axes.shape[1]):
                        ax = self._mode_axes[row_idx, col_idx]
                        for child in ax.get_children():
                            from matplotlib.image import AxesImage
                            if isinstance(child, AxesImage):
                                row_images.append(child)
                                break
                        else:
                            row_images.append(None)
                        row_titles.append(ax.title)
                    mode_images.append(row_images)
                    mode_titles.append(row_titles)
            
            # Get selected mode type
            mode_type = self._mode_type
            
            def _update_frame(frame_idx: int) -> list[Any]:
                mode_at_t = precomputed_frames[frame_idx]
                t = time_array[frame_idx]
                phase_deg = (t / period_s) * 360
                t_ns = t * 1e9
                
                artists = []
                
                for row_idx, row_type in enumerate(self._mode_row_types):
                    for col_idx, comp in enumerate(self._current_components):
                        if row_idx >= len(mode_images) or col_idx >= len(mode_images[row_idx]):
                            continue
                        img = mode_images[row_idx][col_idx]
                        if img is None:
                            continue
                        
                        comp_idx = _COMPONENT_INDEX.get(comp, 0)
                        if comp_idx >= mode_at_t.shape[2]:
                            continue
                        
                        comp_data = mode_at_t[:, :, comp_idx]
                        
                        # Use mode type for first row, row_type for others
                        viz_type = mode_type if row_idx == 0 else row_type
                        
                        if viz_type in ["magnitude", "abs"]:
                            plot_data = np.abs(comp_data)
                        elif viz_type == "phase":
                            plot_data = np.angle(comp_data)
                        elif viz_type == "real":
                            plot_data = np.real(comp_data)
                        elif viz_type == "imag":
                            plot_data = np.imag(comp_data)
                        else:  # combined
                            plot_data = np.real(comp_data)
                        
                        img.set_data(plot_data)
                        artists.append(img)
                        
                        if row_idx == 0 and row_idx < len(mode_titles):
                            title_obj = mode_titles[row_idx][col_idx]
                            title_obj.set_text(f"m_{comp} @ {actual_freq:.3f} GHz | t={t_ns:.2f}ns | φ={phase_deg:.0f}°")
                            artists.append(title_obj)
                
                return artists
            
            # Create animation
            self._animation = FuncAnimation(
                self._fig,
                _update_frame,
                frames=n_frames,
                interval=1000.0 / float(fps),
                blit=True,
                repeat=True,  # Loop continuously for live preview
            )
            
            self._is_animating = True
            if "animate" in self._controls:
                self._controls["animate"].description = "⏸️ Stop"
                self._controls["animate"].button_style = "danger"
            
            self._set_status(
                f"Animating: {n_frames} frames, T={period_s*1e9:.2f}ns (1 period)",
                color="seagreen",
            )
            
            # Redraw
            self._fig.canvas.draw_idle()
            
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._set_status(f"Animation error: {exc}", color="crimson")
            self._is_animating = False
            if "animate" in self._controls:
                self._controls["animate"].description = "🎬 Animate"
                self._controls["animate"].button_style = "warning"
    
    def _stop_animation(self) -> None:
        """Stop any running animation."""
        if self._animation is not None:
            try:
                self._animation.event_source.stop()
            except Exception:
                pass
            self._animation = None
        self._is_animating = False
    
    def _on_mode_type_changed(self, change: Any) -> None:
        """Handle mode visualization type change."""
        if self._internal_update:
            return
        new_type = change.get("new", "combined")
        self._mode_type = new_type
        
        # If animating, restart with new mode type
        if self._is_animating:
            self._stop_animation()
            self._on_animate_clicked(None)
        else:
            # Update static view
            self._update_mode_plots()

    def _on_phase_index_changed(self, change: Any) -> None:
        """Handle phase index slider change - animate FMR mode through phase.
        
        This is the core of phase animation: multiplies mode by exp(-i*omega*t).
        Includes zoom preservation and stable colorbar limits.
        """
        if self._internal_update:
            return
        if self._fig is None or self._current_frequency_ghz is None:
            return
        if self._mode_axes is None:
            return
        
        phase_idx = change.get("new", 0)
        n_frames = 60  # Matches the phase_index slider max
        
        # Calculate phase for this frame (0 to 2π)
        phase_rad = (phase_idx / n_frames) * 2 * np.pi
        phase_deg = (phase_idx / n_frames) * 360
        
        try:
            # Load current mode (complex)
            mode_array, actual_freq, extent = self._load_mode(
                self._current_frequency_ghz, 
                self._current_z_layer
            )
            
            # Pre-compute fixed vmin/vmax from amplitude (prevents flickering)
            # Same for all frames since amplitude is constant
            max_amplitude = float(np.nanmax(np.abs(mode_array)))
            if max_amplitude <= 0:
                max_amplitude = 1.0
            
            # Apply phase evolution: mode * exp(-i * phase)
            phase_factor = np.exp(-1j * phase_rad)
            mode_at_phase = mode_array * phase_factor
            
            # Get mode type
            mode_type = self._mode_type
            
            print(f"DEBUG: Starting loop with row_types={self._mode_row_types}, components={self._current_components}")
            print(f"DEBUG: mode_axes shape = {self._mode_axes.shape if hasattr(self._mode_axes, 'shape') else 'no shape'}")
            print(f"DEBUG: mode_at_phase shape = {mode_at_phase.shape}")
            
            # Update each subplot
            for row_idx, row_type in enumerate(self._mode_row_types):
                for col_idx, comp in enumerate(self._current_components):
                    print(f"DEBUG: Processing row={row_idx}, col={col_idx}, row_type={row_type}, comp={comp}")
                    if row_idx >= self._mode_axes.shape[0] or col_idx >= self._mode_axes.shape[1]:
                        print(f"  → Skipped: out of bounds")
                        continue
                    
                    ax = self._mode_axes[row_idx, col_idx]
                    
                    # ZOOM PRESERVATION: Save current view limits before update
                    xlim_saved = ax.get_xlim()
                    ylim_saved = ax.get_ylim()
                    
                    # Find the AxesImage
                    img = None
                    for child in ax.get_children():
                        from matplotlib.image import AxesImage
                        if isinstance(child, AxesImage):
                            img = child
                            break
                    
                    if img is None:
                        print(f"  → Skipped: no AxesImage found in ax")
                        continue
                    
                    comp_idx = _COMPONENT_INDEX.get(comp, 0)
                    if comp_idx >= mode_at_phase.shape[2]:
                        print(f"  → Skipped: comp_idx={comp_idx} >= shape[2]={mode_at_phase.shape[2]}")
                        continue
                    
                    comp_data = mode_at_phase[:, :, comp_idx]
                    comp_amplitude = float(np.nanmax(np.abs(comp_data)))
                    if comp_amplitude <= 0:
                        comp_amplitude = 1.0
                    
                    # Use mode_type for visualization + FIXED CLIM
                    if row_type == "magnitude" or mode_type == "abs":
                        plot_data = np.abs(comp_data)
                        # Amplitude is constant, use fixed range
                        img.set_clim(0, comp_amplitude)
                    elif row_type == "phase":
                        plot_data = np.angle(comp_data)
                        # Phase is always -π to π
                        img.set_clim(-np.pi, np.pi)
                    elif mode_type == "real" or row_type == "combined":
                        plot_data = np.real(comp_data)
                        # Symmetric range based on amplitude
                        img.set_clim(-comp_amplitude, comp_amplitude)
                    elif mode_type == "imag":
                        plot_data = np.imag(comp_data)
                        img.set_clim(-comp_amplitude, comp_amplitude)
                    else:
                        plot_data = np.real(comp_data)
                        img.set_clim(-comp_amplitude, comp_amplitude)
                    
                    img.set_data(plot_data)
                    
                    # ZOOM PRESERVATION: Restore view limits
                    ax.set_xlim(xlim_saved)
                    ax.set_ylim(ylim_saved)
                    
                    # Update title with phase info
                    if row_idx == 0:
                        freq_hz = actual_freq * 1e9
                        t_ns = (phase_idx / n_frames) * (1.0 / freq_hz) * 1e9
                        ax.set_title(f"m_{comp} @ {actual_freq:.3f} GHz | t={t_ns:.2f}ns | φ={phase_deg:.0f}°", fontsize=10)
            
            # Redraw
            if self._fig is not None:
                self._fig.canvas.draw_idle()
                
        except Exception as exc:
            import traceback
            traceback.print_exc()

    def _read_controls(self) -> None:
        """Read widget values into internal state."""
        if not self._controls:
            return

        freq_min = float(self._controls["fmin"].value)
        freq_max = float(self._controls["fmax"].value)

        if freq_min > freq_max:
            freq_min, freq_max = freq_max, freq_min

        self._filter_state = SpectrumFilterState(
            freq_min=freq_min,
            freq_max=freq_max,
            smooth_filter=str(self._controls["smooth_filter"].value),
            smooth_window=int(self._controls["smooth_window"].value),
            smooth_sigma=float(self._controls["smooth_sigma"].value),
            baseline_mode=str(self._controls["baseline_mode"].value),
            clip_percentile_low=float(self._controls["clip_low"].value),
            clip_percentile_high=float(self._controls["clip_high"].value),
            soft_threshold_percentile=float(self._controls["soft_threshold"].value),
            normalize=bool(self._controls["normalize"].value),
            log_scale=bool(self._controls["log_scale"].value),
        )

        selected_components = list(self._controls["components"].value)
        self._current_components = normalize_component_selection(
            selected_components,
            available=self._available_components,
        )

        self._current_z_layer = int(self._controls["z_layer"].value)
        self._show_peaks = bool(self._controls["show_peaks"].value)
        self._peak_prominence = float(self._controls["peak_prom"].value)
        self._peak_distance = int(self._controls["peak_dist"].value)
        self._mode_row_types = self._resolve_mode_rows(str(self._controls["mode_view"].value))
        
        # Layout controls
        if "aspect" in self._controls:
            self._mode_aspect = str(self._controls["aspect"].value)
        if "layout" in self._controls:
            self._layout_mode = str(self._controls["layout"].value)

    def _refresh_freq_slider_bounds(self) -> None:
        if not self._controls:
            return

        slider = self._controls["freq_index"]
        play = self._controls["play"]

        self._internal_update = True
        try:
            max_idx = max(int(self._filtered_frequencies_ghz.size) - 1, 0)
            slider.max = max_idx
            play.max = max_idx
            idx = self._closest_freq_index(self._current_frequency_ghz)
            slider.value = idx
            play.value = idx
        finally:
            self._internal_update = False

    # ---------------------------------------------------------------------
    # Figure rendering and interaction
    # ---------------------------------------------------------------------
    def _resolve_mode_rows(self, mode_view: str) -> list[str]:
        view = (mode_view or "all").lower()
        if view == "magnitude":
            return ["magnitude"]
        if view == "phase":
            return ["phase"]
        if view == "combined":
            return ["combined"]
        return ["magnitude", "phase", "combined"]

    def _render_figure(self) -> None:
        """Render spectrum + mode figure (in output widget or directly)."""
        n_components = max(len(self._current_components), 1)
        n_rows = max(len(self._mode_row_types), 1)

        if self._toolbar_enabled and self._widget_output is not None:
            with self._widget_output:
                clear_output(wait=True)
                self._create_figure(n_rows=n_rows, n_components=n_components)
                self._draw_spectrum()
                self._update_mode_plots()
                # Show figure in widget output (this doesn't cause double display
                # because output is captured by the widget, not returned to Jupyter)
                plt.show()
        else:
            self._create_figure(n_rows=n_rows, n_components=n_components)
            self._draw_spectrum()
            self._update_mode_plots()

    def _create_figure(self, n_rows: int, n_components: int) -> None:
        """Create matplotlib figure and axes layout.
        
        Supports two layout modes:
        - "vertical": spectrum on top (row 0), modes below in 1-3 columns
        - "horizontal": spectrum on left, modes on right (original layout)
        """
        self._cleanup_figure_connections()
        
        # Disable interactive mode during figure creation to prevent
        # duplicate display (matches dispersion module pattern)
        plt.ioff()
        
        try:
            if self._layout_mode == "vertical":
                # Vertical layout: spectrum takes first row, modes below
                total_rows = 1 + n_rows  # 1 for spectrum + n_rows for modes
                self._fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=False)
                gs = GridSpec(
                    total_rows,
                    n_components,
                    figure=self._fig,
                    height_ratios=[1.2] + [1.0] * n_rows,
                )
                
                # Spectrum spans full top row
                self._ax_spectrum = self._fig.add_subplot(gs[0, :])
                
                # Mode axes in rows 1+ (each row is magnitude/phase/combined)
                axes = []
                for row in range(n_rows):
                    row_axes = []
                    for col in range(n_components):
                        row_axes.append(self._fig.add_subplot(gs[row + 1, col]))
                    axes.append(row_axes)
                self._mode_axes = np.asarray(axes, dtype=object)
            else:
                # Horizontal layout (original): spectrum on left, modes on right
                self._fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=False)
                gs = GridSpec(
                    n_rows,
                    n_components + 1,
                    figure=self._fig,
                    width_ratios=[1.6] + [1.0] * n_components,
                )

                self._ax_spectrum = self._fig.add_subplot(gs[:, 0])

                axes = []
                for row in range(n_rows):
                    row_axes = []
                    for col in range(n_components):
                        row_axes.append(self._fig.add_subplot(gs[row, col + 1]))
                    axes.append(row_axes)
                self._mode_axes = np.asarray(axes, dtype=object)

            # Reconnect click handler.
            if self._fig is not None:
                self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        finally:
            # Always re-enable interactive mode after figure creation
            plt.ion()

    def _draw_spectrum(self) -> None:
        """Draw filtered spectrum traces and peak markers."""
        if self._ax_spectrum is None:
            return

        ax = self._ax_spectrum
        ax.clear()

        if self._filtered_frequencies_ghz.size == 0:
            ax.text(0.5, 0.5, "No spectrum data", ha="center", va="center", transform=ax.transAxes)
            return

        freq_scale = self._get_freq_scale(self._freq_unit)
        freqs_plot = self._filtered_frequencies_ghz * freq_scale

        color_map = {"x": "#E76F51", "y": "#2A9D8F", "z": "#457B9D"}

        for comp in self._current_components:
            trace = self._filtered_component_power.get(comp)
            if trace is None or trace.size == 0:
                continue
            ax.plot(
                freqs_plot,
                trace,
                color=color_map.get(comp, "#4C78A8"),
                linewidth=1.8,
                alpha=0.95,
                label=COMPONENT_LABELS[_COMPONENT_INDEX.get(comp, 2)],
            )

        if self._show_peaks and self._peaks:
            for freq_ghz, amp in self._peaks:
                x_val = freq_ghz * freq_scale
                ax.plot(
                    [x_val],
                    [amp],
                    marker="o",
                    markersize=5,
                    color="#D62828",
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=6,
                )

        self._draw_frequency_line()

        label = self._title or "FMR Spectrum"
        ax.set_title(f"{label} (click to select frequency)")
        ax.set_xlabel(f"Frequency ({self._freq_unit})")
        ax.set_ylabel("log10(Power)" if self._filter_state.log_scale else "Power")
        ax.grid(True, alpha=0.25, linestyle="--")
        if len(self._current_components) > 1:
            ax.legend(loc="upper right", frameon=True, framealpha=0.9)

        ax.text(
            0.02,
            0.02,
            "left click: select, right click: snap to peak",
            transform=ax.transAxes,
            fontsize=8,
            alpha=0.75,
            va="bottom",
        )

    def _draw_frequency_line(self) -> None:
        """Draw or update current frequency indicator line."""
        if self._ax_spectrum is None or self._current_frequency_ghz is None:
            return

        scale = self._get_freq_scale(self._freq_unit)
        x_value = self._current_frequency_ghz * scale

        # Remove previous line if any.
        if self._frequency_line is not None:
            try:
                self._frequency_line.remove()
            except Exception:
                pass

        self._frequency_line = self._ax_spectrum.axvline(
            x_value,
            color="#D62828",
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
        )

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
        if self._mode_axes is None or self._current_frequency_ghz is None:
            return

        for cbar in self._mode_colorbars:
            try:
                cbar.remove()
            except Exception:
                pass
        self._mode_colorbars = []

        try:
            mode_array, actual_freq, extent = self._load_mode(
                self._current_frequency_ghz,
                self._current_z_layer,
            )
            self._current_frequency_ghz = actual_freq
        except Exception as exc:
            for ax in self._mode_axes.flatten():
                ax.clear()
                ax.text(
                    0.5,
                    0.5,
                    f"Mode load error:\n{exc}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="crimson",
                )
            if self._fig is not None:
                self._fig.canvas.draw_idle()
            return

        if mode_array.ndim == 2:
            mode_array = mode_array[:, :, np.newaxis]

        cmap_mag = self._controls.get("cmap_mag", None)
        cmap_phase = self._controls.get("cmap_phase", None)
        cmap_combined = self._controls.get("cmap_combined", None)

        cmap_mag_name = str(cmap_mag.value) if cmap_mag is not None else "viridis"
        cmap_phase_name = str(cmap_phase.value) if cmap_phase is not None else "twilight"
        cmap_combined_name = str(cmap_combined.value) if cmap_combined is not None else "RdBu_r"

        row_images: list[Any] = [None] * len(self._mode_row_types)

        for row_idx, row_type in enumerate(self._mode_row_types):
            for col_idx, comp in enumerate(self._current_components):
                ax = self._mode_axes[row_idx, col_idx]
                ax.clear()

                comp_idx = _COMPONENT_INDEX.get(comp)
                if comp_idx is None or comp_idx >= mode_array.shape[-1]:
                    ax.text(0.5, 0.5, f"No m_{comp}", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                comp_data = mode_array[:, :, comp_idx]
                magnitude = np.abs(comp_data)
                phase = np.angle(comp_data)

                if row_type == "magnitude":
                    plot_data = magnitude
                    cmap_name = cmap_mag_name
                    vmin = None
                    vmax = None
                    row_title = "|m|"
                elif row_type == "phase":
                    plot_data = phase
                    cmap_name = cmap_phase_name
                    vmin = -np.pi
                    vmax = np.pi
                    row_title = "phase"
                else:
                    plot_data = magnitude * np.cos(phase)
                    vmax_val = float(np.nanmax(np.abs(plot_data))) if plot_data.size else 1.0
                    if vmax_val <= 0:
                        vmax_val = 1.0
                    cmap_name = cmap_combined_name
                    vmin = -vmax_val
                    vmax = vmax_val
                    row_title = "combined"

                img = ax.imshow(
                    plot_data,
                    origin="lower",
                    extent=extent,
                    aspect=self._mode_aspect,
                    cmap=cmap_name,
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                )
                
                # Apply xlim/ylim if specified (crop large structures)
                if self._xlim:
                    ax.set_xlim(*self._xlim)
                if self._ylim:
                    ax.set_ylim(*self._ylim)

                if row_images[row_idx] is None:
                    row_images[row_idx] = img

                if row_idx == 0:
                    ax.set_title(f"m_{comp} @ {actual_freq:.3f} GHz", fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(row_title, fontsize=9)
                
                # Add axis labels with units (matching dispersion style)
                # Only show x-axis label on bottom row
                if row_idx == len(self._mode_row_types) - 1:
                    ax.set_xlabel("x [μm]", fontsize=9)
                else:
                    ax.set_xlabel("")
                
                # Y-axis label only on first column (already set as row_title above)
                # Enable tick labels with proper formatting
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.2, linestyle=":")
                
                # Add geometry contour overlay if available (matching dispersion pattern)
                if self._geometry_contour is not None:
                    try:
                        geom = self._geometry_contour
                        # Create coordinate arrays that match the extent
                        geom_y = np.linspace(extent[2], extent[3], geom.shape[0])
                        geom_x = np.linspace(extent[0], extent[1], geom.shape[1])
                        # White contour for visibility on dark backgrounds
                        ax.contour(geom_x, geom_y, geom, levels=[0.5], colors=['white'], linewidths=[1.5])
                        # Black outline for visibility on light backgrounds
                        ax.contour(geom_x, geom_y, geom, levels=[0.5], colors=['black'], linewidths=[0.5])
                    except Exception:
                        pass  # Skip if contour fails

        for row_idx, img in enumerate(row_images):
            if img is None:
                continue
            try:
                cbar = self._fig.colorbar(
                    img,
                    ax=list(self._mode_axes[row_idx, :]),
                    fraction=0.035,
                    pad=0.02,
                )
                self._mode_colorbars.append(cbar)
            except Exception:
                continue

        if self._fig is not None:
            self._fig.suptitle(
                f"FMR modes at {self._current_frequency_ghz:.3f} GHz (z={self._current_z_layer})",
                fontsize=12,
            )
            # Use tight_layout with rect to avoid suptitle overlap
            # Suppress warning for complex GridSpec layouts
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
                try:
                    self._fig.tight_layout(rect=[0, 0, 1, 0.97])
                except Exception:
                    pass  # Fallback: skip tight_layout if it fails
            
            self._fig.canvas.draw_idle()

        self._update_status_text()

    def _update_status_text(self) -> None:
        if not self._controls:
            return
        n_peaks = len(self._peaks)
        if self._current_frequency_ghz is None:
            freq_text = "n/a"
        else:
            freq_text = f"{self._current_frequency_ghz:.3f} GHz"
        self._set_status(
            f"f={freq_text}, "
            f"components={','.join(self._current_components)}, "
            f"peaks={n_peaks}",
            color="#334155",
        )

    def _set_status(self, message: str, color: str = "#334155") -> None:
        """Set status message in toolbar or fallback to logger."""
        if self._controls and "status" in self._controls:
            self._controls["status"].value = (
                f"<small style='color:{color}'>{message}</small>"
            )
        else:
            log.info(message)

    def _update_frequency_selection(self, redraw_canvas: bool = True) -> None:
        """Update vertical line and mode maps after frequency change."""
        self._draw_frequency_line()
        self._update_mode_plots()

        if redraw_canvas and self._fig is not None:
            self._fig.canvas.draw_idle()

    def _on_click(self, event: Any) -> None:
        """Handle spectrum click interactions."""
        if self._ax_spectrum is None or event.inaxes != self._ax_spectrum:
            return
        if event.xdata is None:
            return

        clicked_freq_ghz = float(event.xdata) / self._get_freq_scale(self._freq_unit)

        if event.button == 3 and self._peaks:
            peak_freqs = np.array([p[0] for p in self._peaks], dtype=float)
            idx = int(np.argmin(np.abs(peak_freqs - clicked_freq_ghz)))
            selected = float(peak_freqs[idx])
        else:
            selected = clicked_freq_ghz

        self._current_frequency_ghz = selected

        if self._controls and "freq_index" in self._controls:
            idx = self._closest_freq_index(selected)
            self._internal_update = True
            try:
                self._controls["freq_index"].value = idx
                self._controls["play"].value = idx
            finally:
                self._internal_update = False

        self._update_frequency_selection(redraw_canvas=True)

    def _closest_freq_index(self, freq_ghz: Optional[float]) -> int:
        if self._filtered_frequencies_ghz.size == 0:
            return 0
        if freq_ghz is None:
            return int(self._filtered_frequencies_ghz.size // 2)
        idx = int(np.argmin(np.abs(self._filtered_frequencies_ghz - float(freq_ghz))))
        return max(0, min(idx, self._filtered_frequencies_ghz.size - 1))

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
    if not _HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    frequencies, spectrum, component_label = data_loader.load_spectrum()
    freqs_ghz = _to_ghz(np.asarray(frequencies, dtype=float))
    power = collapse_spectrum_components(_to_power(np.asarray(spectrum)), _component_from_label(component_label))

    data_fmin = float(np.nanmin(freqs_ghz))
    data_fmax = float(np.nanmax(freqs_ghz))
    init_fmin = data_fmin if freq_min is None else float(freq_min)
    init_fmax = data_fmax if freq_max is None else float(freq_max)
    init_fmin = float(np.clip(init_fmin, data_fmin, data_fmax))
    init_fmax = float(np.clip(init_fmax, data_fmin, data_fmax))
    if init_fmin > init_fmax:
        init_fmin, init_fmax = init_fmax, init_fmin

    state = SpectrumFilterState(
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
    freqs_filtered, traces = apply_spectrum_filters(freqs_ghz, power, state)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    scale = InteractiveSpectrum._get_freq_scale(freq_unit)
    x = freqs_filtered * scale

    color_map = {"x": "#E76F51", "y": "#2A9D8F", "z": "#457B9D"}
    for comp, trace in traces.items():
        ax.plot(x, trace, linewidth=1.6, color=color_map.get(comp, "#4C78A8"), label=f"m_{comp}")

    if show_peaks:
        stacked = np.vstack(list(traces.values()))
        peaks = detect_spectrum_peaks(
            freqs_filtered,
            np.mean(stacked, axis=0),
            min_prominence=float(peak_prominence),
            min_distance=int(peak_distance),
        )
        for freq_ghz, amp in peaks:
            ax.plot(freq_ghz * scale, amp, "o", color="#D62828", markersize=4)

    ax.set_xlabel(f"Frequency ({freq_unit})")
    ax.set_ylabel("log10(Power)" if log_scale else "Power")
    ax.set_title(title or "FFT Power Spectrum")
    ax.grid(True, alpha=0.25, linestyle="--")
    if len(traces) > 1:
        ax.legend(loc="upper right")

    # Apply tight_layout with error handling
    try:
        fig.tight_layout()
    except Exception:
        pass  # Skip if layout adjustment fails
    
    return fig
