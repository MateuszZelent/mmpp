"""Transmission analysis core utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple
import math
import time
import os

import numpy as np
from tqdm.auto import tqdm

# Try to use scipy.fft (faster) with fallback to numpy.fft
# Can be disabled via environment variable MMPP_USE_NUMPY_FFT=1
_FORCE_NUMPY = os.environ.get('MMPP_USE_NUMPY_FFT', '').lower() in ('1', 'true', 'yes')

if _FORCE_NUMPY:
    scipy_fft = None
    _USE_SCIPY_FFT = False
    print("🔧 Forced numpy.fft via MMPP_USE_NUMPY_FFT environment variable")
else:
    try:
        from scipy import fft as scipy_fft
        _USE_SCIPY_FFT = True
    except ImportError:
        scipy_fft = None
        _USE_SCIPY_FFT = False

# Try to use joblib for parallel processing
try:
    from joblib import Parallel, delayed
    _USE_JOBLIB = True
except ImportError:
    _USE_JOBLIB = False

from ..compute_fft import FFTCompute, FILTER_TYPES, WINDOW_TYPES

from ...cli.logging_config import get_mmpp_logger


log = get_mmpp_logger("mmpp.fft.transmission")


TransmissionMethod = Literal["power_ratio", "circular", "cpsd"]
AverageMode = Literal["mean", "median", "edge_taper", "none"]
NormalizeMode = Literal["reference", "max", "none"]
ReferenceStatistic = Literal["mean", "median", "max"]
YIntegrationMode = Literal["sum_m", "sum_fft", "none"]
FFTEngine = Literal["scipy", "numpy", "auto"]
SpatialWindowMode = Literal["pre_fft", "post_fft"]


@dataclass
class TransmissionConfig:
    """Configuration parameters for transmission analysis.
    
    All processing steps are optional - can be disabled to match raw FFT behavior.
    """

    dataset_name: Optional[str] = None
    z_layer: int = -1
    method: TransmissionMethod = "power_ratio"
    
    # Temporal preprocessing (can be disabled with None)
    window_function: Optional[WINDOW_TYPES] = "hann"  # None = no windowing
    filter_type: Optional[FILTER_TYPES] = "remove_mean"  # None = no filtering
    
    # Spatial averaging controls
    spatial_window: int = 5  # Set to 1 for no spatial averaging
    spatial_step: int = 1
    spatial_window_mode: SpatialWindowMode = "post_fft"  # "pre_fft" = sum neighbors before FFT (slower, local), "post_fft" = extract from full FFT (faster)
    average_mode: AverageMode = "mean"  # "none" = no y/z averaging
    edge_taper_power: float = 1.5
    y_integration_mode: YIntegrationMode = "sum_fft"  # "sum_m" = sum before FFT, "sum_fft" = sum |FFT|, "none" = no y-sum
    
    # Component selection
    component_weights: Tuple[float, float, float] = (1.0, 1.0, 0.1)
    enable_circular_components: bool = False
    
    # Normalization (can be disabled)
    normalize: NormalizeMode = "reference"  # "none" = raw power
    reference_window: Optional[Tuple[int, int]] = None
    reference_statistic: ReferenceStatistic = "mean"
    
    # Other options
    tmax: Optional[int] = None
    keep_complex_fft: bool = False
    store_component_maps: bool = False
    engine: FFTEngine = "auto"  # "scipy" (fastest), "numpy" (fallback), "auto" (use scipy if available)
    raw_fft_output: bool = False  # If True, skip all post-FFT processing and return raw full_spectrum
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ensure_valid(self) -> None:
        """Validate configuration values."""
        if self.spatial_window <= 0:
            raise ValueError("spatial_window must be > 0")
        if self.spatial_step <= 0:
            raise ValueError("spatial_step must be > 0")
        if self.reference_window is not None:
            start, stop = self.reference_window
            if stop < start:
                raise ValueError("reference_window stop must be >= start")
        if self.component_weights is not None:
            weights = np.atleast_1d(self.component_weights)
            if weights.size not in (1, 3):
                raise ValueError(
                    "component_weights must contain either one entry (for a sliced component) "
                    "or three entries for (mx, my, mz)"
                )


@dataclass
class TransmissionResult:
    """Result of a transmission analysis."""

    frequencies: np.ndarray
    x_positions: np.ndarray  # In nm if dx available, otherwise cell indices
    transmission: np.ndarray
    power_map: np.ndarray
    reference_power: np.ndarray
    config: TransmissionConfig
    dx: Optional[float] = None  # Cell size in meters (None if not available)
    metadata: Dict[str, Any] = field(default_factory=dict)
    power_plus: Optional[np.ndarray] = None
    power_minus: Optional[np.ndarray] = None
    transverse_power: Optional[np.ndarray] = None
    longitudinal_power: Optional[np.ndarray] = None
    # Optional lightweight complex-spectrum summary when keep_complex_fft is True
    complex_spectra_summary: Optional[np.ndarray] = None

    def plot_transmission(self, plot_config=None, **kwargs):
        """Render a frequency-position transmission map.

        Accepts a `plot_config` which may be a mapping (dict) or a
        :class:`TransmissionPlotConfig`. Any additional plotting kwargs
        (e.g., dpi, ax) are forwarded to the plotter.
        """
        # Import here to avoid circular imports at module import time
        from .plot import TransmissionPlotter, TransmissionPlotConfig

        # Convert dict -> TransmissionPlotConfig when needed (same behaviour as FFT interface)
        if plot_config is not None and isinstance(plot_config, dict):
            plot_config = TransmissionPlotConfig(**plot_config)

        plotter = TransmissionPlotter(self)
        return plotter.plot(config=plot_config, **kwargs)

    def overlay_transmission(self, **kwargs):
        """Overlay experimental transmission data on a plot.

        This is a convenience wrapper around the standalone
        :func:`~mmpp.fft.transmission.overlay_transmission` function.

        All keyword arguments are forwarded to the underlying function.
        The `ax` keyword argument is required.

        Parameters
        ----------
        **kwargs
            Keyword arguments for :func:`~mmpp.fft.transmission.overlay_transmission`.
            See its docstring for details (e.g., `ax`, `d`, `p`, `base_path`).

        Returns
        -------
        matplotlib.lines.Line2D
        """
        from .experimental import overlay_transmission as overlay_transmission_func

        # Pass the result object itself to the overlay function so it can access
        # simulation data for normalization.
        return overlay_transmission_func(sim_result=self, **kwargs)

    def plot_transmission_crosssection(
        self,
        x: float,
        freq_unit: str = "GHz",
        trim_0f: Optional[int] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        flip: bool = False,
        log_scale: bool = False,
        ax=None,
        mark_on_ax=None,
        find_minima: Optional[dict] = None,
        x_width: Optional[float] = None,
        **kwargs
    ):
        """Plot 1D transmission cross-section at specific x position.

        Parameters
        ----------
        x : float
            Target X-position. When dx is known, values greater than 1 are
            interpreted as nanometers, while values ≤ 1 are treated as meters
            for backward compatibility with earlier releases. When dx is not
            available, ``x`` is interpreted as a cell index.
        freq_unit : str, optional
            Frequency unit ("Hz", "kHz", "MHz", "GHz"), default "GHz"
        trim_0f : int, optional
            Number of lowest frequency points to remove
        fmin : float, optional
            Minimum frequency to display (in `freq_unit` units).
        fmax : float, optional
            Maximum frequency to display (in `freq_unit` units).
        flip : bool, optional
            If True, frequency is on Y-axis and transmission on X-axis.
            If False (default), frequency on X-axis and transmission on Y-axis.
            Use flip=True to match vertical frequency axis with dispersion plots.
        log_scale : bool, optional
            If True, use logarithmic scale for transmission axis.
            When flip=False (default): X-axis (frequency) is linear, Y-axis (transmission) is log.
            When flip=True: X-axis (transmission) is log, Y-axis (frequency) is linear.
            Default is False (linear scale).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        mark_on_ax : matplotlib.axes.Axes, optional
            If provided, draws a vertical line on this axes object at the crosssection position.
            Useful for marking the crosssection location on a transmission heatmap.
        find_minima : dict, optional
            If provided, finds local minima in the transmission data and marks them.
            Dictionary can contain:
            - 'height': maximum height for minima (default: median of data)
            - 'distance': minimum distance between minima in points (default: 5)
            - 'prominence': minimum prominence of minima (default: None)
            - 'width': minimum width of minima (default: None)
            - 'freq_range': (fmin, fmax) tuple in freq_unit - only search in this range (default: None = all)
            - 'threshold': float - only find minima below this transmission value (e.g., 0.5 for T<50%)
            - 'label_minima': bool, whether to add text labels with frequency for each minimum (default: True)
            - 'label_rounding': int, number of decimal places for frequency labels (e.g., 2). Overrides 'label_format'.
            - 'label_format': str, format string for the label (default: '{:.2f}').
            - 'mark': bool, whether to mark minima on plot (default: True)
            - 'color': color for minima markers (default: 'cyan')
            - 'marker': marker style (default: 'o')
            - 'markersize': marker size (default: 8)
            Returns minima frequencies as third output.
            Example: {'freq_range': (1.0, 3.0), 'threshold': 0.3, 'distance': 10, 'label_rounding': 3}
        x_width : float, optional
            Width of the spatial averaging window around the target x position (in nanometers if dx available, otherwise in indices).
            If provided, the transmission cross-section will be averaged over the range [x - x_width/2, x + x_width/2].
            For example, x_width=500 will average ±250 nm around the specified x position.
            Default is None (no averaging, single x position).
        **kwargs
            Additional matplotlib plot kwargs (color, linewidth, label, etc.)

        Returns
        -------
        fig, ax : matplotlib objects
            Matplotlib figure and axes objects
        minima_freqs : list or None
            If find_minima is provided, returns list of frequencies (in freq_unit) where minima occur.
            Otherwise returns None.
        """
        # Import here to avoid circular imports
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        from .plot import FREQ_SCALE

        # Interpret requested x in the same units as x_positions
        if self.dx is not None:
            if x <= 1.0:
                target_x = x * 1e9  # meters → nanometers (legacy behaviour)
            else:
                target_x = x  # assume already in nanometers
        else:
            target_x = x  # indices

        # Find closest x-position index
        x_idx = np.argmin(np.abs(self.x_positions - target_x))
        actual_x = self.x_positions[x_idx]

        # Get transmission slice at this x (or averaged over x_width)
        if x_width is not None and x_width > 0:
            # Interpret x_width units (same as x interpretation logic)
            if self.dx is not None:
                # x_width already in nanometers (or convert if needed)
                width_in_nm = x_width
            else:
                # x_width in indices
                width_in_nm = x_width
            
            # Average over spatial range [x - x_width/2, x + x_width/2]
            half_width = width_in_nm / 2.0
            x_min = target_x - half_width
            x_max = target_x + half_width
            
            # Find indices within range
            mask = (self.x_positions >= x_min) & (self.x_positions <= x_max)
            num_points = np.sum(mask)
            
            if num_points == 0:
                # Fallback: no points in range - use single closest point
                # This happens when x_width is smaller than dx spacing
                import warnings
                warnings.warn(
                    f"x_width={x_width} nm is too small (no points in range). "
                    f"Using single point at x={actual_x:.1f} nm. "
                    f"Try x_width >= {self.dx * 1e9 if self.dx else 1:.1f} nm.",
                    UserWarning
                )
                transmission_slice = self.transmission[:, x_idx]
            elif num_points == 1:
                # Exactly one point in range - extract it directly
                transmission_slice = self.transmission[:, mask].flatten()
            else:
                # Multiple points - average transmission over all x positions in range
                transmission_slice = self.transmission[:, mask].mean(axis=1)
                # Update actual_x to reflect the center of the averaging range
                actual_x = self.x_positions[mask].mean()
        else:
            # Single x position (no averaging)
            transmission_slice = self.transmission[:, x_idx]

        # Convert frequencies to requested unit
        if freq_unit not in FREQ_SCALE:
            raise ValueError(f"Unsupported frequency unit: {freq_unit}. Use: {list(FREQ_SCALE.keys())}")
        freq_scale = FREQ_SCALE[freq_unit]
        freqs = self.frequencies * freq_scale

        # Apply trim_0f if specified
        trim_idx = 0
        if trim_0f is not None and trim_0f > 0:
            trim_idx = min(trim_0f, len(freqs) - 1)
            freqs = freqs[trim_idx:]
            transmission_slice = transmission_slice[trim_idx:]

        # Apply fmin/fmax if specified
        if fmin is not None:
            mask = freqs >= fmin
            freqs = freqs[mask]
            transmission_slice = transmission_slice[mask]

        if fmax is not None:
            mask = freqs <= fmax
            freqs = freqs[mask]
            transmission_slice = transmission_slice[mask]



        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 6)), dpi=kwargs.pop("dpi", 100))
        else:
            fig = ax.figure

        # Remove log_scale from kwargs if passed (it's not a valid plot() kwarg)
        # User might mistakenly pass it as **kwargs
        kwargs.pop("norm", None)  # Remove 'norm' if present (not valid for plot())
        kwargs.pop("log_scale", None)  # Remove 'log_scale' if passed twice
        
        # Default plot kwargs
        plot_kwargs = {
            "linewidth": 2,
            "color": "C0",
        }
        plot_kwargs.update(kwargs)

        # Plot - choose orientation based on flip parameter
        if flip:
            # Frequency on Y-axis (vertical), Transmission on X-axis (horizontal)
            ax.plot(transmission_slice, freqs, **plot_kwargs)
            ax.set_xlabel("Transmission T(f)", fontsize=12)
            ax.set_ylabel(f"Frequency ({freq_unit})", fontsize=12)
            # Apply log scale to transmission axis (X-axis when flipped)
            if log_scale:
                ax.set_xscale('log')
        else:
            # Frequency on X-axis (horizontal), Transmission on Y-axis (vertical) - default
            ax.plot(freqs, transmission_slice, **plot_kwargs)
            ax.set_xlabel(f"Frequency ({freq_unit})", fontsize=12)
            ax.set_ylabel("Transmission T(f)", fontsize=12)
            # Apply log scale to transmission axis (Y-axis when not flipped)
            if log_scale:
                ax.set_yscale('log')

        # Title
        if self.dx is not None:
            position_label = f"{actual_x:.1f} nm"
        else:
            position_label = f"cell {actual_x:.1f}"

        # Add width info to title if averaging
        width_info = ""
        if x_width is not None and x_width > 0:
            if self.dx is not None:
                width_info = f" (±{x_width/2:.1f} nm avg)"
            else:
                width_info = f" (±{x_width/2:.1f} cells avg)"

        ax.set_title(
            f"Transmission Cross-section at x = {position_label}{width_info}"
            + (f" (trimmed {trim_idx} pts)" if trim_idx > 0 else ""),
            fontsize=13,
            fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        
        # Mark crosssection position on another axes if requested
        if mark_on_ax is not None:
            # Get color from plot kwargs if available, otherwise use default
            mark_color = plot_kwargs.get('color', 'C0')
            mark_on_ax.axvline(
                actual_x, 
                color=mark_color, 
                linestyle='--', 
                linewidth=2, 
                alpha=0.7,
                label=f'Crosssection at {position_label}'
            )
        
        # Find minima if requested
        minima_freqs = None
        if find_minima is not None:
            try:
                from scipy.signal import find_peaks
                
                # Default parameters
                minima_params = {
                    'height': None,  # Will be set to median if None
                    'distance': 5,
                    'prominence': None,
                    'width': None,
                    'mark': True,
                    'color': 'cyan',
                    'label_minima': True,
                    'label_rounding': None,
                    'label_format': '{:.2f}',
                    'marker': 'o',
                    'markersize': 8,
                    'freq_range': None,  # (fmin, fmax) in freq_unit - search only in this range
                    'threshold': None,  # Only minima below this transmission value (e.g., 0.5 for T<50%)
                }
                minima_params.update(find_minima)

                # If label_rounding is provided, it overrides label_format
                if minima_params.get('label_rounding') is not None:
                    try:
                        rounding_places = int(minima_params['label_rounding'])
                        minima_params['label_format'] = f'{{:.{rounding_places}f}}'
                    except (ValueError, TypeError):
                        import warnings
                        warnings.warn(f"Invalid value for 'label_rounding': {minima_params['label_rounding']}. Using default format.")
                
                # Create frequency mask if freq_range is specified
                freq_mask = np.ones(len(freqs), dtype=bool)
                if minima_params['freq_range'] is not None:
                    freq_min, freq_max = minima_params['freq_range']
                    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
                
                # Apply threshold if specified (only find minima below this value)
                if minima_params['threshold'] is not None:
                    freq_mask &= (transmission_slice <= minima_params['threshold'])
                
                # For minima, we need to invert the signal
                inverted_transmission = -transmission_slice.copy()
                
                # Mask out regions we don't want to search
                inverted_transmission[~freq_mask] = np.inf  # Won't be detected as peaks
                
                # Set default height to median if not provided
                if minima_params['height'] is None:
                    # Use median of valid (masked) region
                    valid_transmission = transmission_slice[freq_mask]
                    if len(valid_transmission) > 0:
                        minima_params['height'] = -np.median(valid_transmission)
                    else:
                        minima_params['height'] = -np.median(transmission_slice)
                
                # Find peaks in inverted signal (= minima in original)
                peak_kwargs = {
                    'height': minima_params['height'],
                    'distance': minima_params['distance'],
                }
                if minima_params['prominence'] is not None:
                    peak_kwargs['prominence'] = minima_params['prominence']
                if minima_params['width'] is not None:
                    peak_kwargs['width'] = minima_params['width']
                
                minima_indices, properties = find_peaks(inverted_transmission, **peak_kwargs)
                minima_freqs = freqs[minima_indices].tolist()
                minima_values = transmission_slice[minima_indices]
                
                # Mark minima on plot if requested
                if minima_params['mark'] and len(minima_indices) > 0:
                    if flip:
                        # Frequency on Y-axis, transmission on X-axis
                        ax.plot(
                            minima_values, 
                            minima_freqs,
                            minima_params['marker'],
                            color=minima_params['color'],
                            markersize=minima_params['markersize'],
                            markeredgecolor='white',
                            markeredgewidth=1.5,
                            label=f'Minima ({len(minima_indices)} found)',
                            zorder=10
                        )
                    else:
                        # Frequency on X-axis, transmission on Y-axis
                        ax.plot(
                            minima_freqs, 
                            minima_values,
                            minima_params['marker'],
                            color=minima_params['color'],
                            markersize=minima_params['markersize'],
                            markeredgecolor='white',
                            markeredgewidth=1.5,
                            label=f'Minima ({len(minima_indices)} found)',
                            zorder=10
                        )
                    # Add text labels for each minimum if requested
                    if minima_params['label_minima']:
                        for freq, val in zip(minima_freqs, minima_values):
                            label_text = minima_params['label_format'].format(freq)
                            if flip:
                                # Text to the right of the point
                                ax.text(val, freq, f' {label_text}', 
                                        ha='left', va='center', 
                                        color=minima_params['color'], fontsize=9)
                            else:
                                # Text above the point
                                ax.text(freq, val, label_text, 
                                        ha='center', va='bottom', 
                                        color=minima_params['color'], fontsize=9)
                
            except ImportError:
                import warnings
                warnings.warn("scipy is required for find_minima functionality. Install with: pip install scipy")
                minima_freqs = None

        ax.legend(loc='best', framealpha=0.8)
        if find_minima is not None:
            return fig, ax, minima_freqs
        else:
            return fig, ax


def _compute_hann_weights(length: int, power: float) -> np.ndarray:
    """Create taper weights for averaging across the Y direction."""

    if length <= 1:
        return np.ones((length,), dtype=float)

    window = np.hanning(length)
    window = np.clip(window, 1e-6, None)
    if power != 1.0:
        window = window**power
    window /= window.sum()
    return window


def _aggregate_spatial(
    power: np.ndarray,
    mode: AverageMode,
    edge_taper_power: float,
) -> np.ndarray:
    """Reduce spatial dimensions (z, window_x) of the local power map.
    
    NOTE: Y dimension is already summed before calling this function (physically correct for transmission).
    
    Parameters
    ----------
    power : np.ndarray
        Power array with shape (freq, z, window) - y already summed!
    mode : AverageMode
        "mean" - simple average
        "median" - median (robust to outliers)
        "edge_taper" - weighted average with Hann window
        "none" - no averaging, take only first slice (z=0), mean over window
    """

    if power.ndim != 3:
        raise ValueError(f"Expected power array with shape (freq, z, window), got {power.shape}")

    freq_axis = 0
    z_axis = 1
    x_axis = 2

    if mode == "none":
        # Raw mode: no averaging - take only z=0, average over window_x
        # After slicing [:, 0, :], we have (freq, window_x), so axis=1 for window_x
        return power[:, 0, :].mean(axis=1)

    if mode == "mean":
        return power.mean(axis=(z_axis, x_axis))

    if mode == "median":
        return np.median(power, axis=(z_axis, x_axis))

    if mode == "edge_taper":
        n_z, n_w = power.shape[1:]
        weights_z = np.ones((n_z,), dtype=float)
        weights_z /= weights_z.sum() if weights_z.sum() > 0 else 1.0
        weights_w = np.ones((n_w,), dtype=float)
        weights_w /= weights_w.sum() if weights_w.sum() > 0 else 1.0

        combined = weights_z[:, None] * weights_w[None, :]
        weighted = power * combined[None, ...]
        normalization = combined.sum()
        if normalization <= 0:
            normalization = 1.0
        return weighted.sum(axis=(z_axis, x_axis)) / normalization

    raise ValueError(f"Unsupported averaging mode: {mode}")


class TransmissionCompute:
    """Compute transmission profiles for FFT datasets."""

    def __init__(self, fft_compute: FFTCompute, job_result: Any):
        self._fft_compute = fft_compute
        self._job_result = job_result

    def _get_dx(self, dataset_name: str) -> Optional[float]:
        """Resolve spatial cell size (dx) in **meters** using job[0]-style access."""

        dx_attr_names = [
            "dx",
            "Dx",
            "cellsize_x",
            "cell_size_x",
            "gridsize_x",
            "grid_size_x",
        ]

        def _normalize_dx(value: Any, source: str) -> Optional[float]:
            if value is None:
                return None
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) == 0:
                    return None
                value = value[0]
            try:
                raw = float(value)
            except (TypeError, ValueError):
                log.debug("_get_dx: %s is not a numeric dx (value=%r)", source, value)
                return None

            if not math.isfinite(raw) or raw <= 0:
                log.debug("_get_dx: Ignoring invalid dx=%r from %s", raw, source)
                return None

            if raw <= 1.0:
                dx_m = raw
                unit = "m"
            else:
                dx_m = raw / 1e9
                unit = "nm"

            log.debug(
                "_get_dx: Candidate %.6e %s → %.6e m from %s",
                raw,
                unit,
                dx_m,
                source,
            )
            return dx_m

        def _from_attrs(attrs: Any, source: str) -> Optional[float]:
            if attrs is None:
                return None
            mapping: Mapping[Any, Any]
            if isinstance(attrs, Mapping):
                mapping = attrs  # type: ignore[assignment]
            else:
                try:
                    keys = attrs.keys()  # type: ignore[attr-defined]
                except AttributeError:
                    return None
                mapping = {key: attrs[key] for key in keys}  # type: ignore[index]

            for key in dx_attr_names:
                if key in mapping:
                    dx_m = _normalize_dx(mapping[key], f"{source}['{key}']")
                    if dx_m is not None:
                        return dx_m
            return None

        def _from_object(obj: Any, source: str) -> Optional[float]:
            if obj is None:
                return None

            # Direct attributes (obj.dx, obj.cellsize_x, ...)
            for attr in dx_attr_names:
                try:
                    value = getattr(obj, attr)
                except AttributeError:
                    continue
                dx_m = _normalize_dx(value, f"{source}.{attr}")
                if dx_m is not None:
                    log.debug("_get_dx: ✅ Using dx from %s.%s", source, attr)
                    return dx_m

            # Associated attribute containers
            dx_m = _from_attrs(getattr(obj, "attrs", None), f"{source}.attrs")
            if dx_m is not None:
                log.debug("_get_dx: ✅ Using dx from %s.attrs", source)
                return dx_m

            dx_m = _from_attrs(getattr(obj, "attributes", None), f"{source}.attributes")
            if dx_m is not None:
                log.debug("_get_dx: ✅ Using dx from %s.attributes", source)
                return dx_m

            metadata = getattr(obj, "metadata", None)
            if isinstance(metadata, Mapping):
                dx_m = _from_attrs(metadata, f"{source}.metadata")
                if dx_m is not None:
                    log.debug("_get_dx: ✅ Using dx from %s.metadata", source)
                    return dx_m

            return None

        try:
            log.debug("_get_dx: Resolving dx for dataset '%s'", dataset_name)
            log.debug("_get_dx: job_result type = %s", type(self._job_result))

            dataset_obj = None
            if hasattr(self._job_result, "__getitem__"):
                try:
                    dataset_obj = self._job_result[dataset_name]
                    log.debug(
                        "_get_dx: job_result['%s'] → %s",
                        dataset_name,
                        type(dataset_obj),
                    )
                except (KeyError, TypeError, AttributeError, NameError, AssertionError) as exc:
                    log.debug(
                        "_get_dx: Unable to access job_result['%s']: %s",
                        dataset_name,
                        exc,
                    )

            dx_m = _from_object(dataset_obj, f"job_result['{dataset_name}']")
            if dx_m is not None:
                return dx_m

            if dataset_obj is not None:
                base_array = getattr(dataset_obj, "zarr_array", None)
                dx_m = _from_object(base_array, f"job_result['{dataset_name}'].zarr_array")
                if dx_m is not None:
                    return dx_m

            try:
                z_dataset = self._job_result.z[dataset_name]
            except Exception as exc:  # pragma: no cover - defensive
                log.debug(
                    "_get_dx: job_result.z['%s'] unavailable: %s",
                    dataset_name,
                    exc,
                )
            else:
                dx_m = _from_object(z_dataset, f"job_result.z['{dataset_name}']")
                if dx_m is not None:
                    return dx_m

            dx_m = _from_object(self._job_result, "job_result")
            if dx_m is not None:
                return dx_m

            dx_m = _from_attrs(getattr(self._job_result, "attributes", None), "job_result.attributes")
            if dx_m is not None:
                return dx_m

            log.warning(
                "Could not find dx for dataset '%s'; falling back to cell indices",
                dataset_name,
            )
            return None

        except Exception as exc:  # pragma: no cover - defensive logging
            import traceback

            log.warning("Error getting dx for '%s': %s", dataset_name, exc)
            log.debug("_get_dx traceback:\n%s", traceback.format_exc())
            return None

    def _prepare_data(
        self,
        config: TransmissionConfig,
        slice_info: Optional[Any] = None,
    ) -> tuple[np.ndarray, float]:
        dataset = config.dataset_name or self._job_result.get_largest_m_dataset()

        data, dt = self._fft_compute.load_data_from_zarr(
            self._job_result.path,
            dataset,
            z_layer=config.z_layer,
            tmax=config.tmax,
            slice_info=slice_info,
        )

        # Check if component was pre-selected via slicing
        component_was_selected = False
        if slice_info is not None and isinstance(slice_info, tuple):
            non_ellipsis_slices = [s for s in slice_info if s is not Ellipsis]
            if non_ellipsis_slices and isinstance(non_ellipsis_slices[-1], (int, np.integer)):
                component_was_selected = True
                log.debug("Component was pre-selected via slicing - will add component axis")

        # Normalize to 5D: (t, z, y, x, comp)
        if data.ndim == 5:
            # Already 5D (t, z, y, x, comp)
            pass
        elif data.ndim == 4:
            # Could be (t, y, x, comp) or (t, z, y, x) with component pre-selected
            if component_was_selected:
                # (t, z, y, x) → add component axis → (t, z, y, x, 1)
                data = data[..., np.newaxis]
                log.debug("Added component axis to (t,z,y,x) data → (t,z,y,x,1)")
            else:
                # (t, y, x, comp) → add z axis → (t, 1, y, x, comp)
                data = data[:, np.newaxis, ...]
                log.debug("Added z-axis to (t,y,x,comp) data → (t,1,y,x,comp)")
        elif data.ndim == 3:
            # Could be (t, y, x) with component pre-selected
            if component_was_selected:
                # (t, y, x) → add z and component axes → (t, 1, y, x, 1)
                data = data[:, np.newaxis, :, :, np.newaxis]
                log.debug("Added z and component axes to (t,y,x) data → (t,1,y,x,1)")
            else:
                # Ambiguous - assume (t, y, comp), add z and x
                data = data[:, np.newaxis, :, np.newaxis, :]
                log.debug("Added z and x axes to (t,y,comp) data → (t,1,y,1,comp)")
        else:
            raise ValueError(
                f"Transmission analysis requires 3D, 4D or 5D datasets, got {data.ndim}D with shape {data.shape}"
            )

        # Validate component dimension
        if data.shape[-1] < 1:
            raise ValueError(
                f"Expected at least 1 magnetization component in last dimension, got {data.shape[-1]}"
            )

        return data, dt

    def compute(self, config: TransmissionConfig, slice_info: Optional[Any] = None) -> TransmissionResult:
        config.ensure_valid()

        dataset = config.dataset_name or self._job_result.get_largest_m_dataset()
        data, dt = self._prepare_data(config, slice_info=slice_info)

        # Check if component was pre-selected via slicing
        component_was_selected = False
        if slice_info is not None and isinstance(slice_info, tuple):
            non_ellipsis_slices = [s for s in slice_info if s is not Ellipsis]
            if non_ellipsis_slices and isinstance(non_ellipsis_slices[-1], (int, np.integer)):
                component_was_selected = True

        # Debug: basic metadata about data being processed
        log.debug(
            "Transmission compute: dataset=%s, data.shape=%s, dt=%s, component_pre_selected=%s",
            dataset,
            getattr(data, 'shape', None),
            dt,
            component_was_selected,
        )

        n_time, n_z, n_y, n_x, n_comp = data.shape

        # Get cell size (dx) for spatial positions
        dx_m = self._get_dx(dataset)
        if dx_m is not None:
            dx_nm = dx_m * 1e9
            log.debug(
                "Using dx=%.6e m (%.3f nm) for spatial positions",
                dx_m,
                dx_nm,
            )
        else:
            dx_nm = None
            log.warning("dx not found, x_positions will be in cell indices")

        # 🔍 DEBUG: Print raw data stats
        print(f"📊 RAW data.shape: {data.shape}, min: {data.min():.8e}, max: {data.max():.8e}")
        print(f"⚙️  config.filter_type = {config.filter_type}")
        print(f"⚙️  config.window_function = {config.window_function}")

        # Apply filtering (optional - can be None)
        if config.filter_type is not None:
            if isinstance(config.filter_type, list):
                log.debug(f"Applying sequential filters: {config.filter_type}")
            else:
                log.debug(f"Applying filter: {config.filter_type}")
            filtered = self._fft_compute.apply_filter(data, config.filter_type)
            print(f"🔧 FILTERED data: min: {filtered.min():.8e}, max: {filtered.max():.8e}")
        else:
            filtered = data
            log.debug("Skipping temporal filtering (filter_type=None)")
            print(f"⏭️  SKIPPED filtering (filter_type=None)")
        
        # Apply windowing (optional - can be None)
        if config.window_function is not None:
            windowed = self._fft_compute.apply_window(filtered, config.window_function)
            print(f"🪟 WINDOWED data: min: {windowed.min():.8e}, max: {windowed.max():.8e}")
        else:
            windowed = filtered
            log.debug("Skipping temporal windowing (window_function=None)")
            print(f"⏭️  SKIPPED windowing (window_function=None)")

        window_size = min(config.spatial_window, n_x)
        step = config.spatial_step

        window_starts = list(range(0, max(n_x - window_size + 1, 1), step))
        if not window_starts:
            window_starts = [0]

        # Calculate x_centers in cell indices first
        x_centers_idx = np.array(
            [start + (window_size - 1) / 2.0 for start in window_starts],
            dtype=float,
        )
        
        # Convert to nanometers if dx is available
        if dx_nm is not None:
            x_centers = x_centers_idx * dx_nm
            log.debug(
                "Converted x_positions to nanometers (first 3 values: %s)",
                x_centers[:3],
            )
        else:
            x_centers = x_centers_idx
            log.debug("Using x_positions as cell indices")

        n_windows = len(window_starts)
        n_freq = n_time // 2 + 1

        # Generate frequency array (same for both scipy and numpy)
        freqs = np.fft.rfftfreq(n_time, d=dt)

        power_map = np.zeros((n_freq, n_windows), dtype=float)
        transverse_map = (
            np.zeros((n_freq, n_windows), dtype=float)
            if config.store_component_maps
            else None
        )
        longitudinal_map = (
            np.zeros((n_freq, n_windows), dtype=float)
            if config.store_component_maps and n_comp > 2
            else None
        )

        power_plus = (
            np.zeros((n_freq, n_windows), dtype=float)
            if config.enable_circular_components
            else None
        )
        power_minus = (
            np.zeros((n_freq, n_windows), dtype=float)
            if config.enable_circular_components
            else None
        )

        # Normalize / broadcast component_weights defensively
        # 🔑 SPECIAL CASE: If component was pre-selected via slicing (n_comp=1),
        # user's component_weights=(0,0,1) would be trimmed to [0], giving zero transmission!
        # Solution: When n_comp=1 AND component was pre-selected, override to (1,)
        if component_was_selected and n_comp == 1:
            component_weights = np.array([1.0], dtype=float)
            log.info(
                "Component pre-selected via slicing → auto-setting component_weights=(1,) "
                "(ignoring user-provided weights %s)", 
                config.component_weights
            )
        else:
            component_weights = np.asarray(config.component_weights, dtype=float)
            if component_weights.ndim == 0:
                component_weights = np.full((n_comp,), float(component_weights), dtype=float)
            elif component_weights.size < n_comp:
                # If fewer weights provided, repeat last value to match n_comp
                last = float(component_weights[-1]) if component_weights.size > 0 else 1.0
                component_weights = np.concatenate(
                    [component_weights, np.full((n_comp - component_weights.size,), last, dtype=float)]
                )
            elif component_weights.size > n_comp:
                component_weights = component_weights[:n_comp]
            log.debug("Component weights after broadcast/trim: %s", component_weights)

        # Prepare lightweight complex-spectrum accumulator if requested
        complex_accum = None
        if config.keep_complex_fft:
            # Accumulate mean complex amplitude per frequency & component across windows
            complex_accum = np.zeros((n_freq, n_comp), dtype=np.complex128)
            log.debug(
                "keep_complex_fft=True: storing lightweight complex-spectrum summary (avg over windows)."
            )

        # 🚀 OPTIMIZATION: Compute FFT ONCE for entire dataset instead of in loop
        log.debug("Computing FFT for full dataset (t=%d, z=%d, y=%d, x=%d, comp=%d)...", 
                  n_time, n_z, n_y, n_x, n_comp)
        t_fft_start = time.time()

        # 🔍 DEBUG: Verify data before FFT
        print(f"windowed.shape: {windowed.shape}, min: {windowed.min():.8e}, max: {windowed.max():.8e}")
        print(f"⚙️  y_integration_mode = {config.y_integration_mode}")
        print(f"⚙️  spatial_window_mode = {config.spatial_window_mode}")
        print(f"⚙️  engine = {config.engine}")

        # 🔑 Determine which FFT engine to use based on config.engine parameter
        if config.engine == "scipy":
            if not _USE_SCIPY_FFT or scipy_fft is None:
                raise ValueError("engine='scipy' requested but scipy is not available. Install scipy or use engine='numpy'")
            use_scipy = True
            engine_name = "scipy.fft"
        elif config.engine == "numpy":
            use_scipy = False
            engine_name = "numpy.fft"
        else:  # "auto"
            use_scipy = _USE_SCIPY_FFT
            engine_name = "scipy.fft" if use_scipy else "numpy.fft"
        
        log.info(f"Using FFT engine: {engine_name}")
        print(f"🔧 FFT engine: {engine_name}")

        # 🔑 SPATIAL WINDOW MODE: Choose between pre-FFT (local, slower) or post-FFT (global, faster)
        if config.spatial_window_mode == "pre_fft":
            # 🐢 SLOW PATH: Apply spatial windows BEFORE FFT (physically correct for local transmission)
            # This computes separate FFT for each spatial window position
            log.info("⚠️  Spatial window mode: PRE_FFT (computing separate FFT for each window - SLOW but local)")
            print(f"⚠️  PRE_FFT mode: Will compute {n_windows} separate FFTs (slower)")
            
            # Pre-allocate result arrays
            power_map = np.zeros((n_freq, n_windows), dtype=float)
            full_spectrum = None  # Won't have single full_spectrum in this mode
            
            # 🔑 Process each window separately
            for win_idx, start in tqdm(enumerate(window_starts),
                                       total=n_windows,
                                       desc="Computing FFT per window (pre_fft mode)",
                                       unit="win"):
                end = min(start + window_size, n_x)
                window_slice = slice(start, end)
                
                # Extract window from time-domain data: (t, z, y, window_x, comp)
                window_data = windowed[:, :, :, window_slice, :]
                
                # Apply y-integration if requested (BEFORE FFT!)
                if config.y_integration_mode == "sum_m":
                    # Sum over y: (t, z, y, window_x, comp) → (t, z, window_x, comp)
                    window_data = window_data.sum(axis=2)
                elif config.y_integration_mode == "none":
                    # Keep y dimension
                    pass
                # Note: "sum_fft" doesn't make sense in pre_fft mode (would need FFT first)
                # so we treat it same as "sum_m"
                elif config.y_integration_mode == "sum_fft":
                    log.warning("y_integration_mode='sum_fft' with spatial_window_mode='pre_fft' → using 'sum_m' instead")
                    window_data = window_data.sum(axis=2)
                
                # Sum over spatial window if window_size > 1
                # Shape after y-sum: (t, z, window_x, comp) or (t, z, y, window_x, comp)
                # We want to sum over the window_x axis
                if config.y_integration_mode in ("sum_m", "sum_fft"):
                    # (t, z, window_x, comp) → sum over window_x → (t, z, comp)
                    window_data_summed = window_data.sum(axis=2)
                else:
                    # (t, z, y, window_x, comp) → sum over window_x → (t, z, y, comp)
                    window_data_summed = window_data.sum(axis=3)
                
                # Compute FFT for this window
                if use_scipy:
                    window_spectrum = scipy_fft.rfft(window_data_summed, axis=0)
                else:
                    window_spectrum = np.fft.rfft(window_data_summed, axis=0)
                
                # Compute power from all components
                # Shape: (freq, z, comp) or (freq, z, y, comp)
                power_components = None
                for comp_idx in range(n_comp):
                    if component_weights[comp_idx] != 0:
                        if config.y_integration_mode in ("sum_m", "sum_fft"):
                            comp_fft = window_spectrum[..., comp_idx]  # (freq, z)
                        else:
                            comp_fft = window_spectrum[..., comp_idx]  # (freq, z, y)
                        comp_power = np.abs(comp_fft) * component_weights[comp_idx]
                        if power_components is None:
                            power_components = comp_power
                        else:
                            power_components += comp_power
                
                # Aggregate spatially (over z, and possibly y)
                if config.y_integration_mode in ("sum_m", "sum_fft"):
                    # (freq, z) → aggregate over z
                    if config.average_mode == "none":
                        aggregated = power_components[:, 0]  # Just take z=0
                    elif config.average_mode == "mean":
                        aggregated = power_components.mean(axis=1)
                    else:
                        # Fallback to mean
                        aggregated = power_components.mean(axis=1)
                else:
                    # (freq, z, y) → aggregate over z and y
                    if config.average_mode == "none":
                        aggregated = power_components[:, 0, :].mean(axis=1)  # z=0, mean over y
                    elif config.average_mode == "mean":
                        aggregated = power_components.mean(axis=(1, 2))
                    else:
                        aggregated = power_components.mean(axis=(1, 2))
                
                power_map[:, win_idx] = aggregated
            
            print(f"✅ PRE_FFT complete: computed {n_windows} FFTs")
            t_fft_end = time.time()
            log.info("PRE_FFT mode completed in %.3fs for %d windows", 
                     t_fft_end - t_fft_start, n_windows)
            
            # Skip the post-FFT processing section entirely
            use_post_fft_processing = False
            
        else:  # "post_fft" - current fast implementation
            # 🚀 FAST PATH: Compute FFT once, then extract windows (current implementation)
            log.info("🚀 Spatial window mode: POST_FFT (computing FFT once, then extracting windows - FAST)")
            use_post_fft_processing = True
            
            # 🔑 Y-AXIS INTEGRATION: Handle different methods for summing across y-dimension
            if config.y_integration_mode == "sum_m":
                # Method 1: Sum magnetization data along y BEFORE FFT
                # windowed shape: (t, z, y, x, comp) → sum over y → (t, z, x, comp)
                log.info("Y-integration: sum_m (summing magnetization before FFT)")
                windowed_integrated = windowed.sum(axis=2)  # Sum over y (axis=2)
                print(f"🔧 SUM_M: summed over y-axis → shape {windowed_integrated.shape}")
                
                if use_scipy:
                    full_spectrum = scipy_fft.rfft(windowed_integrated, axis=0)
                else:
                    full_spectrum = np.fft.rfft(windowed_integrated, axis=0)
                
                # 🔍 DEBUG: Verify FFT output
                abs_spectrum = np.abs(full_spectrum)
                print(f"✅ FFT complete: full_spectrum.shape = {full_spectrum.shape}")
                print(f"   |FFT| min: {abs_spectrum.min():.8e}, max: {abs_spectrum.max():.8e}")
            
            elif config.y_integration_mode == "sum_fft":
                # Method 2: Compute FFT first, THEN sum complex FFT along y (preserve phase!)
                log.info("Y-integration: sum_fft (computing FFT first, then summing complex values)")
                
                if use_scipy:
                    full_spectrum_raw = scipy_fft.rfft(windowed, axis=0)
                else:
                    full_spectrum_raw = np.fft.rfft(windowed, axis=0)
                
                print(f"✅ FFT complete (raw): shape = {full_spectrum_raw.shape}")
                
                # Sum complex FFT along y-axis: (freq, z, y, x, comp) → (freq, z, x, comp)
                # ⚠️ IMPORTANT: Sum complex values, NOT absolute values - preserves phase!
                full_spectrum = np.sum(full_spectrum_raw, axis=2)  # Sum over y (axis=2)
                print(f"🔧 SUM_FFT: summed complex FFT over y-axis → shape {full_spectrum.shape}")
                abs_spectrum = np.abs(full_spectrum)
                print(f"   |sum(complex FFT)| min: {abs_spectrum.min():.8e}, max: {abs_spectrum.max():.8e}")
            
            else:  # "none"
                # No y-integration: keep full 5D spectrum
                log.info("Y-integration: none (keeping full 5D spectrum)")
                
                if use_scipy:
                    full_spectrum = scipy_fft.rfft(windowed, axis=0)
                else:
                    full_spectrum = np.fft.rfft(windowed, axis=0)
                
                # 🔍 DEBUG: Verify FFT output
                abs_spectrum = np.abs(full_spectrum)
                print(f"✅ FFT complete: full_spectrum.shape = {full_spectrum.shape}")
                print(f"   |FFT| min: {abs_spectrum.min():.8e}, max: {abs_spectrum.max():.8e}")


            t_fft_end = time.time()
            log.info("FFT completed in %.3fs (shape: %s → %s)", 
                     t_fft_end - t_fft_start, windowed.shape, full_spectrum.shape)

            # 🚀 RAW FFT OUTPUT MODE: Skip all post-processing and return raw spectrum
            if config.raw_fft_output:
                log.info("⚡ RAW FFT OUTPUT MODE: Skipping all post-processing, returning full_spectrum directly")
                print(f"⚡ RAW FFT OUTPUT: Returning full_spectrum shape={full_spectrum.shape}")
                
                # Create minimal result with raw FFT spectrum
                # Note: transmission and power_map will contain the raw complex spectrum
                # User should access result.power_map or result.transmission to get full_spectrum
                metadata = {
                    "dataset": dataset,
                    "z_layer": config.z_layer,
                    "time_step": dt,
                    "raw_fft_output": True,
                    "fft_shape": full_spectrum.shape,
                }
                if dx_m is not None:
                    metadata["dx_m"] = dx_m
                    metadata["dx_nm"] = dx_nm
                    metadata["x_unit"] = "nm"
                else:
                    metadata["x_unit"] = "index"
                
                # Return full_spectrum as both transmission and power_map
                # Shape depends on y_integration_mode:
                # - sum_m/sum_fft: (freq, z, x, comp)
                # - none: (freq, z, y, x, comp)
                return TransmissionResult(
                    frequencies=freqs,
                    x_positions=np.arange(n_x, dtype=float) * (dx_nm if dx_nm else 1.0),  # All X positions
                    transmission=full_spectrum,  # Raw complex FFT
                    power_map=np.abs(full_spectrum),  # FFT magnitude (not squared)
                    reference_power=np.ones(n_freq, dtype=float),  # Dummy reference
                    config=config,
                    dx=dx_m,
                    metadata=metadata,
                )

            # Now extract windows from pre-computed FFT (much faster!)
            # 🚀 OPTIMIZATION 2: Vectorize or parallelize window processing
            
            # 🔑 Determine spectrum shape based on y_integration_mode
            # - sum_m or sum_fft: (freq, z, x, comp) - y already integrated
            # - none: (freq, z, y, x, comp) - full 5D
            y_already_integrated = config.y_integration_mode in ("sum_m", "sum_fft")
            
            # Decide on processing strategy
            use_parallel = _USE_JOBLIB and n_windows > 100  # Only parallelize for many windows
            use_vectorized = (config.average_mode == "none" and 
                          not config.enable_circular_components and 
                          not config.store_component_maps and
                          not use_parallel)
            
            # 🚀 ULTRA-OPTIMIZATION for average_mode='none' with sliding_window_view
            use_sliding_window = (use_vectorized and 
                                 config.spatial_step == 1 and 
                                 hasattr(np.lib.stride_tricks, 'sliding_window_view'))
            
            if use_sliding_window:
                # 🔥 FASTEST PATH: Zero Python loops - pure NumPy vectorization!
                log.info("Using sliding_window_view optimization (step=1, average_mode='none')")
                log.info("Processing %d windows with vectorized operations (no progress bar - too fast!)", n_windows)
                t_process_start = time.time()
                
                if y_already_integrated:
                    # Y already summed: (freq, z, x, comp) → extract z=0 → (freq, x, comp)
                    relevant_spectrum = full_spectrum[:, 0, :, :]  # Extract z=0
                    print(f"🔍 Extracted z=0: {full_spectrum.shape} → {relevant_spectrum.shape}")
                    log.debug("Y already integrated, extracted z=0: %s → %s", full_spectrum.shape, relevant_spectrum.shape)
                else:
                    # Y not summed yet: (freq, z, y, x, comp) → sum over y, extract z=0 → (freq, x, comp)
                    relevant_spectrum = full_spectrum[:, 0, :, :, :].sum(axis=1)  # Sum over y (axis=1)
                    print(f"🔍 Summed over y at z=0: {full_spectrum.shape} → {relevant_spectrum.shape}")
                    log.debug("Summed spectrum over y-dimension: %s → %s", full_spectrum.shape, relevant_spectrum.shape)
                
                print(f"🔍 relevant_spectrum stats: min={np.abs(relevant_spectrum).min():.8e}, max={np.abs(relevant_spectrum).max():.8e}")
                
                # Create sliding window view - NO COPIES, just strides!
                # sliding_window_view adds new axis at the END!
                # Input:  (n_freq, n_x, n_comp)
                # Output: (n_freq, n_windows, n_comp, window_size) ← window_size at END!
                windowed_view = np.lib.stride_tricks.sliding_window_view(
                    relevant_spectrum, 
                    window_shape=window_size, 
                    axis=1  # Slide along x-axis
                )
                # windowed_view shape: (n_freq, n_windows, n_comp, window_size)
                print(f"🔍 windowed_view.shape: {windowed_view.shape} (window_size={window_size})")
                
                # Compute power for ALL windows - iterate only over active components
                # Initialize with zeros - shape (n_freq, n_windows, window_size)
                power_all_windows = np.zeros((n_freq, n_windows, window_size), dtype=float)
                
                # Add contribution from each component with non-zero weight
                for comp_idx in range(n_comp):
                    if component_weights[comp_idx] != 0:
                        # Extract component: (n_freq, n_windows, window_size)
                        comp_fft_all = windowed_view[:, :, comp_idx, :]
                        power_all_windows += np.abs(comp_fft_all) * component_weights[comp_idx]
                
                print(f"🔍 power_all_windows stats BEFORE mean: min={power_all_windows.min():.8e}, max={power_all_windows.max():.8e}")
                
                # Mean over window_size dimension - NO LOOP!
                # power_all_windows shape: (n_freq, n_windows, window_size)
                power_map = power_all_windows.mean(axis=2)  # Result: (n_freq, n_windows)
                
                print(f"🔍 power_map stats AFTER mean: min={power_map.min():.8e}, max={power_map.max():.8e}")
                
                t_process_end = time.time()
                log.info("Sliding window vectorization: %.3fs for %d windows (%.1f µs/window)", 
                          t_process_end - t_process_start, n_windows,
                          (t_process_end - t_process_start) * 1e6 / n_windows)
                          
            elif use_vectorized:
                # 🔥 OPTIMIZED PATH: Loop with reduced dimensions (for step != 1)
                log.debug("Using optimized vectorized processing (average_mode='none', step=%d)", 
                         config.spatial_step)
                t_process_start = time.time()
                
                if y_already_integrated:
                    # Y already summed: (freq, z, x, comp) → extract z=0 → (freq, x, comp)
                    relevant_spectrum = full_spectrum[:, 0, :, :]  # Extract z=0
                    log.debug("Y already integrated, extracted z=0: %s → %s", full_spectrum.shape, relevant_spectrum.shape)
                else:
                    # Y not summed yet: (freq, z, y, x, comp) → sum over y, extract z=0 → (freq, x, comp)
                    relevant_spectrum = full_spectrum[:, 0, :, :, :].sum(axis=1)  # Sum over y (axis=1)
                    log.debug("Summed spectrum over y-dimension: %s → %s", full_spectrum.shape, relevant_spectrum.shape)
                
                # Now loop with much smaller slicing operations
                for win_idx, start in tqdm(enumerate(window_starts), 
                                           total=n_windows,
                                           desc="Processing windows",
                                           unit="win",
                                           disable=n_windows < 10):  # Disable for very few windows
                    end = min(start + window_size, n_x)
                    
                    # Slice from reduced 3D array (n_freq, n_x, n_comp)
                    # instead of 5D array - much faster!
                    spectrum_slice = relevant_spectrum[:, start:end, :]  # (n_freq, window_len, n_comp)
                    
                    # Compute power - iterate only over active components
                    # Initialize with zeros
                    power_components = np.zeros((n_freq, end - start), dtype=float)
                    
                    # Add contribution from each component with non-zero weight
                    for comp_idx in range(n_comp):
                        if component_weights[comp_idx] != 0:
                            comp_fft = spectrum_slice[..., comp_idx]  # (n_freq, window_len)
                            power_components += np.abs(comp_fft) * component_weights[comp_idx]
                    
                    # Fast aggregation: mean over window dimension
                    # power_components shape: (n_freq, window_len)
                    power_map[:, win_idx] = power_components.mean(axis=1)
                
                t_process_end = time.time()
                log.info("Optimized vectorized processing: %.3fs for %d windows (%.1f µs/window)", 
                          t_process_end - t_process_start, n_windows,
                          (t_process_end - t_process_start) * 1e6 / n_windows)
                          
            elif use_parallel:
                # Parallel path: use joblib to process windows in parallel
                log.info("Using parallel processing with joblib (%d windows, %d CPUs)", 
                         n_windows, -1)  # -1 = use all CPUs
                t_process_start = time.time()
                
                def process_window(win_idx: int, start: int):
                    """Process single window - can run in parallel."""
                    end = min(start + window_size, n_x)
                    window_slice = slice(start, end)
                    
                    if y_already_integrated:
                        # Y already summed: (freq, z, x, comp) → extract window
                        # Result: (freq, z, window_x, comp)
                        spectrum = full_spectrum[:, :, window_slice, :]
                    else:
                        # Extract: (n_freq, n_z, n_y, window_x, n_comp)
                        spectrum = full_spectrum[:, :, :, window_slice, :]
                        # Sum over y dimension (integrate across width) - physically correct!
                        # Result: (n_freq, n_z, window_x, n_comp)
                        spectrum = spectrum.sum(axis=2)
                    
                mx_fft = spectrum[..., 0]
                my_fft = spectrum[..., 1]
                power_components = np.abs(mx_fft) * component_weights[0]
                power_components += np.abs(my_fft) * component_weights[1]
                
                if n_comp > 2:
                    mz_fft = spectrum[..., 2]
                    power_components += np.abs(mz_fft) * component_weights[2]
                
                aggregated = _aggregate_spatial(
                    power_components,
                    config.average_mode,
                    config.edge_taper_power,
                )
                
                results = {'power': aggregated}
                    
                if transverse_map is not None:
                    results['transverse'] = _aggregate_spatial(
                        np.abs(mx_fft) + np.abs(my_fft),
                        config.average_mode,
                        config.edge_taper_power,
                    )
                
                if longitudinal_map is not None and n_comp > 2:
                    results['longitudinal'] = _aggregate_spatial(
                        np.abs(mz_fft),
                        config.average_mode,
                        config.edge_taper_power,
                    )
                
                if config.enable_circular_components:
                    m_plus = (mx_fft + 1j * my_fft) / np.sqrt(2.0)
                    m_minus = (mx_fft - 1j * my_fft) / np.sqrt(2.0)
                    results['power_plus'] = _aggregate_spatial(
                        np.abs(m_plus),
                        config.average_mode,
                        config.edge_taper_power,
                    )
                    results['power_minus'] = _aggregate_spatial(
                        np.abs(m_minus),
                        config.average_mode,
                        config.edge_taper_power,
                    )
                
                return win_idx, results
                
                # Process windows in parallel
                results_list = Parallel(n_jobs=-1, backend='threading')(
                    delayed(process_window)(win_idx, start)
                    for win_idx, start in enumerate(window_starts)
                )
                
                # Collect results
                for win_idx, results in results_list:
                    power_map[:, win_idx] = results['power']
                    if transverse_map is not None and 'transverse' in results:
                        transverse_map[:, win_idx] = results['transverse']
                    if longitudinal_map is not None and 'longitudinal' in results:
                        longitudinal_map[:, win_idx] = results['longitudinal']
                    if config.enable_circular_components:
                        if power_plus is not None:
                            power_plus[:, win_idx] = results.get('power_plus', 0)
                        if power_minus is not None:
                            power_minus[:, win_idx] = results.get('power_minus', 0)
                
                t_process_end = time.time()
                log.info("Parallel processing: %.3fs for %d windows", 
                          t_process_end - t_process_start, n_windows)
            
            else:
                # Standard path: use _aggregate_spatial for each window (serial)
                log.debug("Using standard serial processing (average_mode='%s')", config.average_mode)
                t_process_start = time.time()
                
                for win_idx, start in tqdm(enumerate(window_starts),
                                           total=n_windows,
                                           desc="Processing windows",
                                           unit="win",
                                           disable=n_windows < 10):  # Disable for very few windows
                    end = min(start + window_size, n_x)
                    window_slice = slice(start, end)
                    
                    if y_already_integrated:
                        # Y already summed: (freq, z, x, comp) → extract window
                        # Result: (freq, z, window_x, comp)
                        spectrum = full_spectrum[:, :, window_slice, :]
                    else:
                        # Extract window from pre-computed FFT spectrum
                        # Shape: (n_freq, n_z, n_y, window_x, n_comp)
                        spectrum = full_spectrum[:, :, :, window_slice, :]
                        # Sum over y dimension (integrate across width) - physically correct!
                        # Result: (n_freq, n_z, window_x, n_comp)
                        spectrum = spectrum.sum(axis=2)

                    # Compute power - iterate only over active components
                    power_components = None
                    for comp_idx in range(n_comp):
                        if component_weights[comp_idx] != 0:
                            comp_fft = spectrum[..., comp_idx]
                            comp_power = np.abs(comp_fft) * component_weights[comp_idx]
                            if power_components is None:
                                power_components = comp_power
                            else:
                                power_components += comp_power
                    
                    # Handle case where no components are active (shouldn't happen but be safe)
                    # Note: y dimension already summed, so shape is (n_freq, n_z, window_x)
                    if power_components is None:
                        power_components = np.zeros((n_freq, n_z, end - start), dtype=float)

                    # Store longitudinal component map if requested
                    if longitudinal_map is not None and n_comp > 2 and component_weights[2] != 0:
                        mz_fft = spectrum[..., 2]
                        longitudinal_map[:, win_idx] = _aggregate_spatial(
                            np.abs(mz_fft),
                            config.average_mode,
                            config.edge_taper_power,
                        )

                    # Accumulate lightweight complex-spectrum summary per component (mean across z,window)
                    # Note: y dimension already summed
                    if complex_accum is not None:
                        for comp_idx in range(n_comp):
                            comp_spec = spectrum[..., comp_idx]
                            # mean over z and the window dimension (note: block may be smaller than window_size at edges)
                            comp_mean = comp_spec.mean(axis=(1, 2))  # axes: z, window_x
                            complex_accum[:, comp_idx] += comp_mean

                    aggregated = _aggregate_spatial(
                        power_components,
                        config.average_mode,
                        config.edge_taper_power,
                    )

                    power_map[:, win_idx] = aggregated

                    # Store transverse component map if requested (mx + my)
                    if transverse_map is not None:
                        transverse_power = None
                        if n_comp > 0 and component_weights[0] != 0:  # mx
                            mx_fft = spectrum[..., 0]
                            transverse_power = np.abs(mx_fft)
                        if n_comp > 1 and component_weights[1] != 0:  # my
                            my_fft = spectrum[..., 1]
                            my_power = np.abs(my_fft)
                            if transverse_power is None:
                                transverse_power = my_power
                            else:
                                transverse_power += my_power
                        
                        if transverse_power is not None:
                            transverse_map[:, win_idx] = _aggregate_spatial(
                                transverse_power,
                                config.average_mode,
                                config.edge_taper_power,
                            )

                    # Store circular components if requested
                    if config.enable_circular_components and power_plus is not None and power_minus is not None:
                        # Need mx and my for circular components
                        if n_comp > 1:
                            mx_fft = spectrum[..., 0]
                            my_fft = spectrum[..., 1]
                            m_plus = (mx_fft + 1j * my_fft) / np.sqrt(2.0)
                            m_minus = (mx_fft - 1j * my_fft) / np.sqrt(2.0)
                            power_plus[:, win_idx] = _aggregate_spatial(
                                np.abs(m_plus),
                                config.average_mode,
                                config.edge_taper_power,
                            )
                            power_minus[:, win_idx] = _aggregate_spatial(
                                np.abs(m_minus),
                                config.average_mode,
                                config.edge_taper_power,
                            )

                # End of serial window processing loop
                t_process_end = time.time()
                log.debug("Serial processing: %.3fs for %d windows", 
                          t_process_end - t_process_start, n_windows)

        reference_mask = self._select_reference_windows(
            x_centers,
            window_size,
            config.reference_window,
        )

        if not np.any(reference_mask):
            reference_mask[0] = True

        reference_values = self._compute_reference(
            power_map,
            reference_mask,
            config.reference_statistic,
        )

        if config.normalize == "reference":
            denom = np.where(reference_values <= 0, 1.0, reference_values)
            transmission = power_map / denom[:, None]
        elif config.normalize == "max":
            denom = np.max(power_map, axis=1, keepdims=True)
            denom = np.where(denom <= 0, 1.0, denom)
            transmission = power_map / denom
        else:
            transmission = power_map.copy()

        metadata = {
            "dataset": dataset,
            "z_layer": config.z_layer,
            "window_size": window_size,
            "window_step": step,
            "time_step": dt,
            "method": config.method,
        }
        metadata.update(config.metadata)
        if dx_m is not None:
            metadata.setdefault("dx_m", dx_m)
            metadata.setdefault("dx_nm", dx_nm)
            metadata.setdefault("x_unit", "nm")
        else:
            metadata.setdefault("x_unit", "index")

        result = TransmissionResult(
            frequencies=freqs,
            x_positions=x_centers,
            transmission=transmission,
            power_map=power_map,
            reference_power=reference_values,
            config=config,
            dx=dx_m,  # Store dx in meters to match job[0].dx
            metadata=metadata,
            power_plus=power_plus,
            power_minus=power_minus,
            transverse_power=transverse_map,
            longitudinal_power=longitudinal_map,
        )

        # Finalize and attach complex-spectrum summary if requested
        if complex_accum is not None:
            # Average over windows
            complex_summary = complex_accum / float(n_windows)
            result.complex_spectra_summary = complex_summary
            log.debug("Attached complex_spectra_summary shape=%s", getattr(complex_summary, 'shape', None))

        log.debug("Transmission compute complete: transmission.shape=%s", transmission.shape)

        return result

    @staticmethod
    def _select_reference_windows(
        x_centers: np.ndarray,
        window_size: int,
        reference_window: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        mask = np.zeros_like(x_centers, dtype=bool)
        if reference_window is None:
            if x_centers.size:
                mask[0] = True
            return mask

        start, stop = reference_window
        mask = (x_centers >= start) & (x_centers <= stop)
        if not np.any(mask):
            # If provided reference range does not intersect any center, warn and fall back to first window
            log.warning(
                "Reference window %s does not intersect x_centers range [%s, %s]; falling back to first window.",
                reference_window,
                x_centers[0] if x_centers.size else None,
                x_centers[-1] if x_centers.size else None,
            )
            if x_centers.size:
                mask[0] = True
        return mask

    @staticmethod
    def _compute_reference(
        power_map: np.ndarray,
        reference_mask: np.ndarray,
        statistic: ReferenceStatistic,
    ) -> np.ndarray:
        ref_columns = power_map[:, reference_mask]
        if ref_columns.ndim == 1:
            ref_columns = ref_columns[:, None]
        if ref_columns.size == 0:
            return np.ones((power_map.shape[0],), dtype=float)

        if statistic == "mean":
            return np.mean(ref_columns, axis=1)
        if statistic == "median":
            return np.median(ref_columns, axis=1)
        if statistic == "max":
            return np.max(ref_columns, axis=1)

        raise ValueError(f"Unsupported reference statistic: {statistic}")


__all__ = [
    "TransmissionConfig",
    "TransmissionCompute",
    "TransmissionResult",
]
