"""Transmission analysis core utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple, Union
import math
import time
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

# Try to use scipy.fft (faster) with fallback to numpy.fft
# Can be disabled via environment variable MMPP_USE_NUMPY_FFT=1
_FORCE_NUMPY = os.environ.get("MMPP_USE_NUMPY_FFT", "").lower() in ("1", "true", "yes")

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


def _report_progress(
    callback: Optional[Callable[[float, str], None]],
    current: int,
    total: int,
    stage: str,
) -> None:
    """Report progress to callback if provided."""
    if callback is not None:
        try:
            # Defensive: ensure current and total are numeric
            current_num = int(current) if current is not None else 0
            total_num = int(total) if total is not None else 1
            progress = (current_num / total_num) * 100.0 if total_num > 0 else 0.0
            callback(progress, stage)
        except Exception as e:
            log.debug(
                "Progress callback failed: %s (current=%r, total=%r, stage=%r)",
                e,
                current,
                total,
                stage,
            )


TransmissionMethod = Literal["power_ratio", "circular", "cpsd"]
AverageMode = Literal["mean", "median", "edge_taper", "none"]
NormalizeMode = Literal["reference", "max", "none"]
ReferenceStatistic = Literal["mean", "median", "max"]
YIntegrationMode = Literal["sum_m", "sum_fft", "none"]
FFTEngine = Literal["scipy", "numpy", "auto"]
SpatialWindowMode = Literal["pre_fft", "post_fft"]
ReconstructionMode = Literal["real_signal", "phasor"]


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
    spatial_window: int = 5  # Set to 1 for no spatial averaging (0/False are treated as 1)
    spatial_step: int = 1
    spatial_window_mode: SpatialWindowMode = (
        "post_fft"  # "pre_fft" = sum neighbors before FFT (slower, local), "post_fft" = extract from full FFT (faster)
    )
    average_mode: AverageMode = "mean"  # "none" = no y/z averaging
    edge_taper_power: float = 1.5
    y_integration_mode: YIntegrationMode = (
        "sum_fft"  # "sum_m" = sum before FFT, "sum_fft" = sum complex FFT along y (phase-preserving), "none" = no y-sum
    )

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
    engine: FFTEngine = (
        "auto"  # "scipy" (fastest), "numpy" (fallback), "auto" (use scipy if available)
    )
    raw_fft_output: bool = (
        False  # If True, skip all post-FFT processing and return raw full_spectrum
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Progress reporting callback
    # Signature: callback(progress: float, stage: str) where progress is 0-100%
    progress_callback: Optional[Callable[[float, str], None]] = None

    def ensure_valid(self) -> None:
        """Validate configuration values."""
        # Backward-compatible aliases for "no x-averaging".
        # Many users intuitively pass False/0 to disable window averaging.
        if isinstance(self.spatial_window, bool):
            if not self.spatial_window:
                log.debug("spatial_window=False interpreted as 1 (no x-averaging)")
            self.spatial_window = 1
        else:
            try:
                spatial_window_int = int(self.spatial_window)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"spatial_window must be an integer-like value, got {self.spatial_window!r}"
                ) from exc

            if spatial_window_int == 0:
                log.debug("spatial_window=0 interpreted as 1 (no x-averaging)")
                spatial_window_int = 1
            if spatial_window_int < 0:
                raise ValueError("spatial_window must be >= 0 (0 is treated as 1)")
            self.spatial_window = spatial_window_int

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
            if not np.all(np.isfinite(weights)):
                raise ValueError("component_weights must be finite numbers")
            if np.any(weights < 0):
                raise ValueError(
                    "component_weights must be non-negative (negative values produce non-physical power)"
                )
            if not np.any(weights > 0):
                raise ValueError("At least one component weight must be > 0")
        if self.method == "cpsd" and self.spatial_window_mode != "post_fft":
            raise ValueError(
                "method='cpsd' requires spatial_window_mode='post_fft' (window axis is needed)"
            )
        if self.raw_fft_output and self.spatial_window_mode != "post_fft":
            log.warning(
                "raw_fft_output=True requires spatial_window_mode='post_fft'; "
                "overriding spatial_window_mode=%r to 'post_fft'",
                self.spatial_window_mode,
            )
            self.spatial_window_mode = "post_fft"


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
        disable_averaging: bool = False,
        normalize: bool = False,
        use_power_map: bool = False,
        verbose: bool = False,
        legend: Union[bool, dict] = False,
        **kwargs,
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
            - 'markersize': marker size (default: 12)
            - 'markeredgewidth': edge width for markers (default: 2.0)
            - 'markeredgecolor': edge color for markers (default: 'white')
            - 'label_fontsize': font size for frequency labels (default: 11)
            - 'label_fontweight': font weight for labels (default: 'bold')
            - 'label_bbox': whether to add background box to labels (default: True)
            - 'label_bbox_alpha': transparency of label background (default: 0.7)
            - 'label_bbox_color': color of label background (default: 'black')
            Returns minima frequencies as third output.
            Example: {'freq_range': (1.0, 3.0), 'threshold': 0.3, 'distance': 10, 'label_rounding': 3}
        x_width : float, optional
            Width of the spatial averaging window around the target x position (in nanometers if dx available, otherwise in indices).
            If provided, the transmission cross-section will be averaged over the range [x - x_width/2, x + x_width/2].
            For example, x_width=500 will average ±250 nm around the specified x position.
            Default is None (no averaging, single x position).
            IGNORED if disable_averaging=True.
        disable_averaging : bool, optional
            If True, forces single-point extraction (no averaging) regardless of x_width value.
            Use this flag to guarantee exact single-point behavior. Default is False.
        normalize : bool, optional
            If True, normalizes the transmission cross-section so that the maximum value is 1.
            This is useful for comparing transmission profiles with different amplitudes.
            Default is False (no normalization).
        use_power_map : bool, optional
            If True, use raw ``power_map`` instead of ``transmission`` as source data.
            This is useful when ``result.config.normalize`` was set to ``"reference"`` or
            ``"max"`` during computation and you want unnormalized amplitudes.
            Default is False.
        verbose : bool, optional
            If True, prints detailed diagnostic information about x position selection,
            averaging behavior, and data extraction. Default is False.
        legend : bool or dict, optional
            Controls legend display and styling.
            - False (default): No legend displayed
            - True: Display legend with automatic label "x = {value} µm"
            - dict: Display legend with custom options. Supported keys:
                - 'label': str - Custom label text (overrides automatic)
                - 'loc': str - Legend location ('best', 'upper right', 'lower left', etc.)
                - 'framealpha': float - Legend background transparency (0-1)
                - 'fontsize': int/str - Font size for legend text
                - 'frameon': bool - Whether to draw legend frame
                - 'fancybox': bool - Rounded corners on frame
                - 'shadow': bool - Draw shadow behind legend
                - 'ncol': int - Number of columns
                - 'title': str - Legend title
                - 'title_fontsize': int/str - Title font size
            Example: legend={'label': 'Sample A', 'loc': 'upper right', 'fontsize': 12}
        **kwargs
            Additional matplotlib plot kwargs (color, linewidth, label, etc.)
            Note: If 'label' is provided in kwargs, it will be used for the legend.

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
        import logging

        log = logging.getLogger(__name__)

        source_data = self.power_map if use_power_map else self.transmission
        source_label_base = "Power P(f)" if use_power_map else "Transmission T(f)"
        result_norm_mode = str(getattr(self.config, "normalize", "none")).lower()
        if not use_power_map and not normalize and result_norm_mode != "none":
            msg = (
                "TransmissionResult data was already normalized during compute() "
                f"(config.normalize='{result_norm_mode}'). "
                "plot_transmission_crosssection(normalize=False) disables only additional "
                "per-curve max normalization. Use use_power_map=True or recompute with "
                "normalize='none' for raw amplitudes."
            )
            log.warning(msg)
            if verbose:
                print(f"[VERBOSE] WARNING: {msg}")

        # Interpret requested x in the same units as x_positions
        if self.dx is not None:
            if x <= 1.0:
                target_x = x * 1e9  # meters → nanometers (legacy behaviour)
                if verbose:
                    print(f"[VERBOSE] Input x={x} m converted to {target_x:.1f} nm")
            else:
                target_x = x  # assume already in nanometers
                if verbose:
                    print(f"[VERBOSE] Input x={x} nm (no conversion)")
        else:
            target_x = x  # indices
            if verbose:
                print(f"[VERBOSE] Input x={x} (cell index, dx not available)")

        if verbose:
            print(f"[VERBOSE] dx = {self.dx * 1e9 if self.dx else 'N/A'} nm")
            print(
                f"[VERBOSE] x_positions range: {self.x_positions.min():.1f} to {self.x_positions.max():.1f}"
            )
            print(f"[VERBOSE] x_positions shape: {self.x_positions.shape}")
            print(f"[VERBOSE] source data shape: {source_data.shape}")

        # Find closest x-position index
        x_idx = np.argmin(np.abs(self.x_positions - target_x))
        actual_x = self.x_positions[x_idx]

        if verbose:
            print(f"[VERBOSE] Target x position: {target_x:.1f} nm")
            print(f"[VERBOSE] Closest x_idx: {x_idx}")
            print(f"[VERBOSE] Actual x at index: {actual_x:.1f} nm")
            print(f"[VERBOSE] disable_averaging: {disable_averaging}")
            print(f"[VERBOSE] x_width: {x_width}")

        # Get transmission slice at this x (or averaged over x_width)
        if disable_averaging:
            # FORCE single point extraction - no averaging
            if verbose:
                print(f"[VERBOSE] MODE: Single point (disable_averaging=True)")
                print(f"[VERBOSE] Extracting source_data[:, {x_idx}]")
            transmission_slice = source_data[:, x_idx]

        elif x_width is not None and x_width > 0:
            # Averaging mode
            if verbose:
                print(f"[VERBOSE] MODE: Averaging with x_width={x_width} nm")

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

            if verbose:
                print(f"[VERBOSE] Averaging range: [{x_min:.1f}, {x_max:.1f}] nm")

            # Find indices within range
            mask = (self.x_positions >= x_min) & (self.x_positions <= x_max)
            num_points = np.sum(mask)

            if verbose:
                print(f"[VERBOSE] Points in range: {num_points}")
                if num_points > 0:
                    indices = np.where(mask)[0]
                    print(
                        f"[VERBOSE] Indices to average: {indices[:10]}{'...' if len(indices) > 10 else ''}"
                    )

            if num_points == 0:
                # Fallback: no points in range - use single closest point
                # This happens when x_width is smaller than dx spacing
                import warnings

                msg = (
                    f"x_width={x_width} nm is too small (no points in range). "
                    f"Using single point at x={actual_x:.1f} nm. "
                    f"Try x_width >= {self.dx * 1e9 if self.dx else 1:.1f} nm."
                )
                if verbose:
                    print(f"[VERBOSE] WARNING: {msg}")
                warnings.warn(msg, UserWarning)
                transmission_slice = source_data[:, x_idx]
            elif num_points == 1:
                # Exactly one point in range - extract it directly
                if verbose:
                    print(f"[VERBOSE] Exactly 1 point in range, extracting directly")
                transmission_slice = source_data[:, mask].flatten()
            else:
                # Multiple points - average transmission over all x positions in range
                if verbose:
                    print(f"[VERBOSE] Averaging over {num_points} points")
                transmission_slice = source_data[:, mask].mean(axis=1)
                # Update actual_x to reflect the center of the averaging range
                actual_x = self.x_positions[mask].mean()
                if verbose:
                    print(
                        f"[VERBOSE] New actual_x (center of averaged range): {actual_x:.1f} nm"
                    )
        else:
            # Single x position (no averaging)
            if verbose:
                print(f"[VERBOSE] MODE: Single point (x_width not specified)")
                print(f"[VERBOSE] Extracting source_data[:, {x_idx}]")
            transmission_slice = source_data[:, x_idx]

        if verbose:
            print(f"[VERBOSE] transmission_slice shape: {transmission_slice.shape}")
            print(
                f"[VERBOSE] transmission_slice stats: min={transmission_slice.min():.4f}, max={transmission_slice.max():.4f}, mean={transmission_slice.mean():.4f}"
            )

        # Convert frequencies to requested unit
        if freq_unit not in FREQ_SCALE:
            raise ValueError(
                f"Unsupported frequency unit: {freq_unit}. Use: {list(FREQ_SCALE.keys())}"
            )
        freq_scale = FREQ_SCALE[freq_unit]
        freqs = self.frequencies * freq_scale

        if verbose:
            print(f"[VERBOSE] Frequency unit: {freq_unit}, scale: {freq_scale}")
            print(
                f"[VERBOSE] Initial freqs range: {freqs.min():.3f} to {freqs.max():.3f} {freq_unit}"
            )
            print(f"[VERBOSE] Initial data points: {len(freqs)}")

        # Apply trim_0f if specified
        trim_idx = 0
        if trim_0f is not None and trim_0f > 0:
            trim_idx = min(trim_0f, len(freqs) - 1)
            if verbose:
                print(f"[VERBOSE] Trimming first {trim_idx} frequency points")
            freqs = freqs[trim_idx:]
            transmission_slice = transmission_slice[trim_idx:]
            if verbose:
                print(
                    f"[VERBOSE] After trim: {len(freqs)} points, freq range: {freqs.min():.3f} to {freqs.max():.3f} {freq_unit}"
                )

        # Apply fmin/fmax if specified
        if fmin is not None:
            mask = freqs >= fmin
            points_before = len(freqs)
            freqs = freqs[mask]
            transmission_slice = transmission_slice[mask]
            if verbose:
                print(
                    f"[VERBOSE] Applying fmin={fmin} {freq_unit}: {points_before} -> {len(freqs)} points"
                )

        if fmax is not None:
            mask = freqs <= fmax
            points_before = len(freqs)
            freqs = freqs[mask]
            transmission_slice = transmission_slice[mask]
            if verbose:
                print(
                    f"[VERBOSE] Applying fmax={fmax} {freq_unit}: {points_before} -> {len(freqs)} points"
                )

        # Apply normalization if requested
        if normalize:
            max_val = transmission_slice.max()
            if max_val > 0:
                if verbose:
                    print(
                        f"[VERBOSE] Normalizing: dividing by max value = {max_val:.4f}"
                    )
                transmission_slice = transmission_slice / max_val
            else:
                if verbose:
                    print(
                        f"[VERBOSE] WARNING: Cannot normalize - max value is {max_val}"
                    )

        if verbose:
            print(f"[VERBOSE] Final data for plotting: {len(freqs)} points")
            print(
                f"[VERBOSE] Final freq range: {freqs.min():.3f} to {freqs.max():.3f} {freq_unit}"
            )
            print(
                f"[VERBOSE] Final transmission range: {transmission_slice.min():.4f} to {transmission_slice.max():.4f}"
            )

        # Create figure if needed
        _dpi = kwargs.pop("dpi", 100)
        if ax is None:
            fig, ax = plt.subplots(
                figsize=kwargs.pop("figsize", (10, 6)), dpi=_dpi
            )
        else:
            fig = ax.figure
            fig.set_dpi(_dpi)

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

        # Handle legend parameter - can be bool or dict
        show_legend = False
        legend_kwargs = {"loc": "best", "framealpha": 0.8}  # Default legend options

        if isinstance(legend, dict):
            show_legend = True
            # Extract custom label from legend dict if provided
            if "label" in legend:
                plot_kwargs["label"] = legend.pop("label")
            # Remaining legend dict entries are legend style options
            legend_kwargs.update(legend)
        elif legend:
            show_legend = True

        # Set automatic label if legend enabled and no custom label provided
        if show_legend and "label" not in plot_kwargs:
            # Convert actual_x (in nm) to µm for display
            x_um = actual_x / 1000.0
            plot_kwargs["label"] = f"x = {x_um:.1f} µm"

        # Prepare axis labels
        transmission_label = (
            f"Normalized {source_label_base}" if normalize else source_label_base
        )

        # Plot - choose orientation based on flip parameter
        if flip:
            # Frequency on Y-axis (vertical), Transmission on X-axis (horizontal)
            ax.plot(transmission_slice, freqs, **plot_kwargs)
            ax.set_xlabel(transmission_label, fontsize=12)
            ax.set_ylabel(f"Frequency ({freq_unit})", fontsize=12)
            # Apply log scale to transmission axis (X-axis when flipped)
            if log_scale:
                ax.set_xscale("log")
        else:
            # Frequency on X-axis (horizontal), Transmission on Y-axis (vertical) - default
            ax.plot(freqs, transmission_slice, **plot_kwargs)
            ax.set_xlabel(f"Frequency ({freq_unit})", fontsize=12)
            ax.set_ylabel(transmission_label, fontsize=12)
            # Apply log scale to transmission axis (Y-axis when not flipped)
            if log_scale:
                ax.set_yscale("log")

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

        plot_title = "Transmission Cross-section" if not use_power_map else "Power Cross-section"
        ax.set_title(
            f"{plot_title} at x = {position_label}{width_info}"
            + (f" (trimmed {trim_idx} pts)" if trim_idx > 0 else ""),
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Mark crosssection position on another axes if requested
        if mark_on_ax is not None:
            # Get color from plot kwargs if available, otherwise use default
            mark_color = plot_kwargs.get("color", "C0")
            mark_on_ax.axvline(
                actual_x,
                color=mark_color,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"Crosssection at {position_label}",
            )

        # Find minima if requested
        minima_freqs = None
        if find_minima is not None:
            try:
                from scipy.signal import find_peaks

                # Default parameters
                minima_params = {
                    "height": None,  # Will be set to median if None
                    "distance": 5,
                    "prominence": None,
                    "width": None,
                    "mark": True,
                    "color": "cyan",
                    "label_minima": True,
                    "label_rounding": None,
                    "label_format": "{:.2f}",
                    "marker": "o",
                    "markersize": 12,  # Larger for publication visibility
                    "markeredgewidth": 2.0,  # Thicker edge
                    "markeredgecolor": "white",  # White edge for contrast
                    "freq_range": None,  # (fmin, fmax) in freq_unit - search only in this range
                    "threshold": None,  # Only minima below this transmission value (e.g., 0.5 for T<50%)
                    # Publication label styling
                    "label_fontsize": 11,
                    "label_fontweight": "bold",
                    "label_bbox": True,  # Add background box to labels
                    "label_bbox_alpha": 0.7,
                    "label_bbox_color": "black",
                    "label_offset": (
                        0.02,
                        0.0,
                    ),  # Offset from marker (in axes fraction)
                }
                minima_params.update(find_minima)

                # If label_rounding is provided, it overrides label_format
                if minima_params.get("label_rounding") is not None:
                    try:
                        rounding_places = int(minima_params["label_rounding"])
                        minima_params["label_format"] = f"{{:.{rounding_places}f}}"
                    except (ValueError, TypeError):
                        import warnings

                        warnings.warn(
                            f"Invalid value for 'label_rounding': {minima_params['label_rounding']}. Using default format."
                        )

                # Create frequency mask if freq_range is specified
                freq_mask = np.ones(len(freqs), dtype=bool)
                if minima_params["freq_range"] is not None:
                    freq_min, freq_max = minima_params["freq_range"]
                    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)

                # Apply threshold if specified (only find minima below this value)
                if minima_params["threshold"] is not None:
                    freq_mask &= transmission_slice <= minima_params["threshold"]

                # For minima, we need to invert the signal
                inverted_transmission = -transmission_slice.copy()

                # Mask out regions we don't want to search
                inverted_transmission[~freq_mask] = np.inf  # Won't be detected as peaks

                # Set default height to median if not provided
                if minima_params["height"] is None:
                    # Use median of valid (masked) region
                    valid_transmission = transmission_slice[freq_mask]
                    if len(valid_transmission) > 0:
                        minima_params["height"] = -np.median(valid_transmission)
                    else:
                        minima_params["height"] = -np.median(transmission_slice)

                # Find peaks in inverted signal (= minima in original)
                peak_kwargs = {
                    "height": minima_params["height"],
                    "distance": minima_params["distance"],
                }
                if minima_params["prominence"] is not None:
                    peak_kwargs["prominence"] = minima_params["prominence"]
                if minima_params["width"] is not None:
                    peak_kwargs["width"] = minima_params["width"]

                minima_indices, properties = find_peaks(
                    inverted_transmission, **peak_kwargs
                )
                minima_freqs = freqs[minima_indices].tolist()
                minima_values = transmission_slice[minima_indices]

                # Mark minima on plot if requested
                if minima_params["mark"] and len(minima_indices) > 0:
                    marker_kwargs = {
                        "marker": minima_params["marker"],
                        "color": minima_params["color"],
                        "markersize": minima_params["markersize"],
                        "markeredgecolor": minima_params.get(
                            "markeredgecolor", "white"
                        ),
                        "markeredgewidth": minima_params.get("markeredgewidth", 2.0),
                        "linestyle": "none",  # No connecting line
                        "label": f"Minima ({len(minima_indices)} found)",
                        "zorder": 15,
                    }

                    if flip:
                        # Frequency on Y-axis, transmission on X-axis
                        ax.plot(minima_values, minima_freqs, **marker_kwargs)
                    else:
                        # Frequency on X-axis, transmission on Y-axis
                        ax.plot(minima_freqs, minima_values, **marker_kwargs)

                    # Add text labels for each minimum if requested
                    if minima_params["label_minima"]:
                        for freq, val in zip(minima_freqs, minima_values):
                            label_text = minima_params["label_format"].format(freq)

                            # Build text kwargs
                            text_kwargs = {
                                "fontsize": minima_params.get("label_fontsize", 11),
                                "fontweight": minima_params.get(
                                    "label_fontweight", "bold"
                                ),
                                "color": minima_params["color"],
                                "zorder": 16,
                            }

                            # Add background box for better visibility
                            if minima_params.get("label_bbox", True):
                                text_kwargs["bbox"] = dict(
                                    boxstyle="round,pad=0.3",
                                    facecolor=minima_params.get(
                                        "label_bbox_color", "black"
                                    ),
                                    alpha=minima_params.get("label_bbox_alpha", 0.7),
                                    edgecolor="none",
                                )

                            if flip:
                                # Text to the right of the point (horizontal layout)
                                offset_x = minima_params.get(
                                    "label_offset", (0.02, 0.0)
                                )[0]
                                ax.text(
                                    val
                                    + offset_x * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                                    freq,
                                    f"{label_text} GHz",
                                    ha="left",
                                    va="center",
                                    **text_kwargs,
                                )
                            else:
                                # Text above the point (vertical layout)
                                offset_y = minima_params.get(
                                    "label_offset", (0.0, 0.02)
                                )[1]
                                ax.text(
                                    freq,
                                    val
                                    + offset_y * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                                    f"{label_text}",
                                    ha="center",
                                    va="bottom",
                                    **text_kwargs,
                                )

            except ImportError:
                import warnings

                warnings.warn(
                    "scipy is required for find_minima functionality. Install with: pip install scipy"
                )
                minima_freqs = None

        minima_count = 0 if minima_freqs is None else len(minima_freqs)
        if show_legend or minima_count > 0:
            ax.legend(
                loc=legend_kwargs.get("loc", "best"),
                framealpha=legend_kwargs.get("framealpha", 0.8),
                **{k: v for k, v in legend_kwargs.items() if k not in {"loc", "framealpha"}},
            )
        if find_minima is not None:
            return fig, ax, minima_freqs
        else:
            return fig, ax

    @staticmethod
    def _resolve_component_index(component: Union[int, str], n_comp: int) -> int:
        """Resolve component alias/index with single-component fallback."""
        if isinstance(component, str):
            comp_alias = {
                "x": 0,
                "mx": 0,
                "0": 0,
                "y": 1,
                "my": 1,
                "1": 1,
                "z": 2,
                "mz": 2,
                "2": 2,
            }
            key = component.lower().strip()
            if key not in comp_alias:
                raise ValueError(
                    "Unsupported component string. Use one of: "
                    "'x', 'y', 'z', 'mx', 'my', 'mz' or integer index."
                )
            comp_idx = int(comp_alias[key])
        else:
            comp_idx = int(component)

        if comp_idx < 0 or comp_idx >= n_comp:
            if n_comp == 1:
                log.warning(
                    "Requested component=%r resolved to index=%d but n_comp=1; using component index 0.",
                    component,
                    comp_idx,
                )
                return 0
            raise ValueError(f"component index {comp_idx} out of range for n_comp={n_comp}")
        return comp_idx

    @staticmethod
    def _single_bin_snapshot(
        spectrum_k: np.ndarray,
        *,
        k_idx: int,
        n_time: int,
        t_idx: int,
        phase_offset_rad: float = 0.0,
        reconstruction: ReconstructionMode = "real_signal",
    ) -> np.ndarray:
        """Evaluate one rFFT bin contribution at a single time index.

        This avoids allocating full ``zeros_like(spectrum)`` and full inverse FFT volume.
        """
        phase = np.exp(
            1j * (2.0 * np.pi * float(k_idx) * float(t_idx) / float(n_time) + float(phase_offset_rad))
        )
        if k_idx == 0 or (n_time % 2 == 0 and k_idx == n_time // 2):
            scale = 1.0 / float(n_time)
        else:
            scale = 2.0 / float(n_time)
        phasor = spectrum_k * phase * scale
        if reconstruction == "real_signal":
            return np.real(phasor)
        if reconstruction == "phasor":
            return phasor
        raise ValueError("reconstruction must be one of: 'real_signal', 'phasor'")

    def _reconstruct_mode_xy(
        self,
        spectrum: np.ndarray,
        *,
        k_idx: int,
        n_time: int,
        t_idx: int,
        z_idx: int,
        comp_idx: int,
        y_slice: Optional[slice],
        copy_y: int,
        phase_offset_rad: float = 0.0,
        reconstruction: ReconstructionMode = "real_signal",
    ) -> tuple[np.ndarray, int]:
        """Reconstruct selected mode snapshot in XY view with optional Y tiling."""
        if spectrum.ndim == 5:
            y_selector = slice(None) if y_slice is None else y_slice
            xy_k = np.asarray(spectrum[k_idx, z_idx, y_selector, :, comp_idx])
            if xy_k.ndim == 1:
                xy_k = xy_k[np.newaxis, :]
        else:
            xy_k = np.asarray(spectrum[k_idx, z_idx, :, comp_idx])[np.newaxis, :]

        xy = self._single_bin_snapshot(
            xy_k,
            k_idx=k_idx,
            n_time=n_time,
            t_idx=t_idx,
            phase_offset_rad=phase_offset_rad,
            reconstruction=reconstruction,
        )
        y_block = int(xy.shape[0])
        if copy_y > 1:
            xy = np.tile(xy, (copy_y, 1))
        return np.asarray(xy), y_block

    def visualize_mode(
        self,
        f: Optional[float] = None,
        *,
        k: Optional[int] = None,
        freq_unit: str = "GHz",
        t_show: int = 0,
        phase_deg: float = 0.0,
        reconstruction: ReconstructionMode = "real_signal",
        z_layer: int = 0,
        component: Union[int, str] = "z",
        y_slice: Optional[slice] = None,
        copy_y: int = 1,
        mode: str = "real",
        vlim_scale: float = 0.1,
        ax=None,
        figsize: tuple[float, float] = (13.0, 4.5),
        dpi: int = 100,
        aspect: str = "auto",
        origin: str = "lower",
        cmap: Optional[str] = None,
        colorbar: bool = True,
        interpolation: str = "nearest",
        x_unit: str = "nm",
        x_lines: Optional[Sequence[float]] = None,
        x_lines_in_index: bool = False,
        y_lines: Optional[Sequence[float]] = None,
        y_spans: Optional[Sequence[tuple[float, float]]] = None,
        y_span_color: str = "cyan",
        y_span_alpha: float = 0.15,
        flip_x: bool = True,
        separator_lines: bool = True,
        separator_style: str = "--",
        separator_color: str = "black",
        separator_linewidth: float = 1.0,
        **imshow_kwargs,
    ):
        """Visualize reconstructed spin-wave mode for one selected frequency.

        The method expects complex raw FFT output produced with
        ``raw_fft_output=True`` in transmission compute.

        Parameters
        ----------
        f : float, optional
            Target frequency in ``freq_unit`` units (e.g. GHz).
            Provide either ``f`` or ``k``.
        k : int, optional
            Direct frequency-bin index in rFFT array.
        freq_unit : str, optional
            Frequency unit for ``f`` ("Hz", "kHz", "MHz", "GHz"), default "GHz".
        t_show : int, optional
            Time index used after inverse FFT reconstruction, default 0.
        phase_deg : float, optional
            Global phase shift applied before snapshot extraction [degrees].
            Useful for continuous phase stepping without changing ``t_show``.
        reconstruction : {"real_signal", "phasor"}, optional
            ``real_signal`` reconstructs real-valued time-domain contribution
            (imaginary part is numerically ~0). ``phasor`` keeps complex
            quadrature so ``mode='imag'`` shows the 90° component.
        z_layer : int, optional
            Z-layer index, default 0.
        component : int or str, optional
            Magnetization component selector ("x"/"y"/"z" or index), default "z".
        y_slice : slice, optional
            Slice over Y axis before plotting.
        copy_y : int, optional
            Repeat the selected Y block vertically for easier visual inspection.
        mode : str, optional
            Rendered value: "real", "imag", "abs", or "phase".
        vlim_scale : float, optional
            Symmetric clipping factor for "real"/"imag" modes.
            ``vlim = max(abs(data)) * vlim_scale``.
        ax : matplotlib.axes.Axes, optional
            Existing axes. If None, create a new figure/axes.
        figsize : tuple, optional
            Figure size used when ``ax is None``.
        dpi : int, optional
            Figure DPI used when ``ax is None``.
        x_unit : str, optional
            X-axis unit ("index", "m", "um", "nm"), default "nm".
        x_lines : sequence of float, optional
            Vertical reference lines. Values are interpreted in ``x_unit``
            unless ``x_lines_in_index=True``.

        Returns
        -------
        fig, ax, dict
            Figure, axes and metadata dictionary with selected bin/frequency and
            reconstructed 2D array under ``meta["xy"]``.
        """
        import matplotlib.pyplot as plt

        if (f is None) == (k is None):
            raise ValueError("Provide exactly one selector: either f=... or k=...")

        freqs = np.asarray(self.frequencies, dtype=float)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("TransmissionResult.frequencies must be a non-empty 1D array")

        spectrum = np.asarray(self.transmission)
        if not np.iscomplexobj(spectrum):
            raise ValueError(
                "visualize_mode requires complex raw FFT data. "
                "Recompute transmission with raw_fft_output=True."
            )
        if spectrum.ndim not in (4, 5):
            raise ValueError(
                "Expected raw spectrum shape (freq,z,x,comp) or (freq,z,y,x,comp); "
                f"got {spectrum.shape}"
            )

        if k is None:
            unit_scales = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
            freq_scale = unit_scales.get(str(freq_unit).lower())
            if freq_scale is None:
                raise ValueError(
                    f"Unsupported freq_unit={freq_unit!r}. Use one of {list(unit_scales)}."
                )
            target_hz = float(f) * freq_scale
            k = int(np.argmin(np.abs(freqs - target_hz)))
        else:
            k = int(k)
            target_hz = float(freqs[k]) if 0 <= k < freqs.size else float("nan")

        if k < 0 or k >= freqs.size:
            raise ValueError(f"k={k} out of range [0, {freqs.size - 1}]")

        n_time_meta = self.metadata.get("n_time")
        try:
            n_time = int(n_time_meta)
        except (TypeError, ValueError):
            n_time = int(2 * (freqs.size - 1))
        if n_time <= 0:
            raise ValueError("Unable to infer valid n_time for inverse FFT reconstruction")

        n_z = int(spectrum.shape[1])
        z_idx = int(z_layer)
        if z_idx < 0:
            z_idx += n_z
        if z_idx < 0 or z_idx >= n_z:
            raise ValueError(f"z_layer={z_layer} out of range for n_z={n_z}")

        n_comp = int(spectrum.shape[-1])
        comp_idx = self._resolve_component_index(component, n_comp)

        t_idx = int(np.clip(int(t_show), 0, n_time - 1))
        copy_y_int = int(copy_y)
        if copy_y_int < 1:
            raise ValueError("copy_y must be >= 1")
        phase_offset_rad = np.deg2rad(float(phase_deg))

        xy_complex_vis, y_block = self._reconstruct_mode_xy(
            spectrum,
            k_idx=int(k),
            n_time=n_time,
            t_idx=t_idx,
            z_idx=z_idx,
            comp_idx=comp_idx,
            y_slice=y_slice,
            copy_y=copy_y_int,
            phase_offset_rad=phase_offset_rad,
            reconstruction=reconstruction,
        )

        mode_key = str(mode).lower().strip()
        if mode_key == "real":
            xy_vis = np.real(xy_complex_vis)
        elif mode_key in ("imag", "imaginary"):
            xy_vis = np.imag(xy_complex_vis)
        elif mode_key in ("abs", "magnitude"):
            xy_vis = np.abs(xy_complex_vis)
        elif mode_key == "phase":
            xy_vis = np.angle(xy_complex_vis)
        else:
            raise ValueError("mode must be one of: 'real', 'imag', 'abs', 'phase'")

        x_unit_key = str(x_unit).lower().strip()
        if self.dx is None or x_unit_key in ("index", "idx", "cell", "cells"):
            x_step = 1.0
            x_unit_label = "index"
        else:
            x_scales = {"m": 1.0, "um": 1e6, "nm": 1e9}
            if x_unit_key not in x_scales:
                raise ValueError("x_unit must be one of: 'index', 'm', 'um', 'nm'")
            x_step = float(self.dx) * float(x_scales[x_unit_key])
            x_unit_label = x_unit_key

        n_x = int(xy_vis.shape[1])
        extent = [0.0, n_x * x_step, 0.0, float(xy_vis.shape[0])]

        vmin = imshow_kwargs.pop("vmin", None)
        vmax = imshow_kwargs.pop("vmax", None)
        if vmin is None or vmax is None:
            peak = float(np.max(np.abs(xy_vis))) if xy_vis.size > 0 else 0.0
            if mode_key in ("real", "imag", "imaginary"):
                vlim = peak * float(vlim_scale)
                if not np.isfinite(vlim) or vlim <= 0:
                    vlim = peak if peak > 0 else 1e-12
                default_vmin, default_vmax = -vlim, +vlim
            elif mode_key == "phase":
                default_vmin, default_vmax = -np.pi, np.pi
            else:
                default_vmin, default_vmax = 0.0, peak if peak > 0 else 1e-12
            if vmin is None:
                vmin = default_vmin
            if vmax is None:
                vmax = default_vmax

        if cmap is None:
            if mode_key in ("abs", "magnitude"):
                cmap = "inferno"
            elif mode_key == "phase":
                cmap = "twilight"
            else:
                cmap = "coolwarm"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure

        img = ax.imshow(
            xy_vis,
            aspect=aspect,
            origin=origin,
            cmap=cmap,
            interpolation=interpolation,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            **imshow_kwargs,
        )

        unit_scales_out = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
        out_freq_scale = unit_scales_out.get(str(freq_unit).lower(), 1e9)
        actual_freq_hz = float(freqs[k])
        actual_freq_unit = actual_freq_hz / out_freq_scale

        ax.set_title(
            f"Mode @ f={actual_freq_unit:.4g} {freq_unit} (k={k}, t={t_idx}, z={z_idx})"
        )
        ax.set_xlabel(f"x [{x_unit_label}]")
        ax.set_ylabel("y [index]")

        if copy_y_int > 1 and separator_lines and y_block > 0:
            for i in range(1, copy_y_int):
                y_sep = i * y_block
                ax.hlines(
                    y=y_sep,
                    xmin=extent[0],
                    xmax=extent[1],
                    colors=separator_color,
                    linestyles=separator_style,
                    linewidth=separator_linewidth,
                )

        if x_lines is not None:
            for x_line in x_lines:
                x_val = float(x_line)
                if x_lines_in_index:
                    x_val = x_val * x_step
                ax.axvline(x_val, color="red", linewidth=1.1, alpha=0.9)

        if y_lines is not None:
            for y_line in y_lines:
                ax.axhline(float(y_line), color="cyan", linewidth=1.1, linestyle="--", alpha=0.9)

        if y_spans is not None:
            for y0, y1 in y_spans:
                ax.axhspan(
                    float(y0),
                    float(y1),
                    color=y_span_color,
                    alpha=float(y_span_alpha),
                )

        if flip_x:
            ax.invert_xaxis()

        if colorbar:
            cbar_label = {
                "real": "Re(m)",
                "imag": "Im(m)",
                "imaginary": "Im(m)",
                "abs": "|m|",
                "magnitude": "|m|",
                "phase": "arg(m) [rad]",
            }[mode_key]
            fig.colorbar(img, ax=ax, label=cbar_label)

        meta = {
            "k": int(k),
            "frequency_hz": actual_freq_hz,
            "requested_frequency_hz": float(target_hz),
            "frequency_value": float(actual_freq_unit),
            "frequency_unit": str(freq_unit),
            "time_index": int(t_idx),
            "phase_shift_deg": float(phase_deg),
            "reconstruction": str(reconstruction),
            "n_time": int(n_time),
            "z_index": int(z_idx),
            "component_index": int(comp_idx),
            "component": str(component),
            "x_unit": x_unit_label,
            "x_step": float(x_step),
            "mode": mode_key,
            "xy": xy_vis,
            "xy_complex": xy_complex_vis,
            "extent": extent,
        }
        return fig, ax, meta

    def visualize_modes(
        self,
        frequencies: Sequence[float],
        *,
        freq_unit: str = "GHz",
        ncols: int = 3,
        figsize: Optional[tuple[float, float]] = None,
        dpi: int = 100,
        colorbar: bool = False,
        **kwargs,
    ):
        """Plot reconstructed mode maps for multiple selected frequencies."""
        import matplotlib.pyplot as plt

        freq_list = [float(v) for v in frequencies]
        if not freq_list:
            raise ValueError("frequencies must contain at least one value")

        ncols_int = max(1, int(ncols))
        nrows = int(math.ceil(len(freq_list) / float(ncols_int)))
        if figsize is None:
            figsize = (5.6 * ncols_int, 4.1 * nrows)

        fig, axes = plt.subplots(nrows, ncols_int, figsize=figsize, dpi=dpi, squeeze=False)
        axes_flat = axes.reshape(-1)
        used_axes = []
        metas = []

        for idx, freq_val in enumerate(freq_list):
            axis = axes_flat[idx]
            _, _, meta = self.visualize_mode(
                f=freq_val,
                freq_unit=freq_unit,
                ax=axis,
                colorbar=colorbar,
                **kwargs,
            )
            used_axes.append(axis)
            metas.append(meta)

        for idx in range(len(freq_list), axes_flat.size):
            fig.delaxes(axes_flat[idx])

        fig.tight_layout()
        return fig, used_axes, metas

    def animate_mode(
        self,
        *,
        animate: Literal["k", "t"] = "t",
        f: Optional[float] = None,
        k: Optional[int] = None,
        freq_unit: str = "GHz",
        t_show: int = 0,
        frames: Optional[Union[Sequence[int], slice]] = None,
        k_frames: Optional[Union[Sequence[int], slice]] = None,
        t_frames: Optional[Union[Sequence[int], slice]] = None,
        phase_deg: float = 0.0,
        reconstruction: ReconstructionMode = "real_signal",
        z_layer: int = 0,
        component: Union[int, str] = "z",
        y_slice: Optional[slice] = None,
        copy_y: int = 1,
        mode: str = "real",
        vlim_scale: float = 0.1,
        ax=None,
        figsize: tuple[float, float] = (13.0, 4.5),
        dpi: int = 100,
        aspect: str = "auto",
        origin: str = "lower",
        cmap: Optional[str] = None,
        colorbar: bool = True,
        interpolation: str = "nearest",
        x_unit: str = "nm",
        x_lines: Optional[Sequence[float]] = None,
        x_lines_in_index: bool = False,
        y_lines: Optional[Sequence[float]] = None,
        y_spans: Optional[Sequence[tuple[float, float]]] = None,
        y_span_color: str = "cyan",
        y_span_alpha: float = 0.15,
        flip_x: bool = True,
        separator_lines: bool = True,
        separator_style: str = "--",
        separator_color: str = "black",
        separator_linewidth: float = 1.0,
        interval: int = 100,
        fps: int = 15,
        repeat: bool = True,
        color_scale: Literal["global", "frame"] = "global",
        saveas: Optional[Union[str, Path]] = None,
        writer: Optional[str] = None,
        animation_save_kwargs: Optional[dict[str, Any]] = None,
        show_progress: bool = False,
        **imshow_kwargs,
    ):
        """Animate reconstructed transmission mode over ``k`` or ``t``.

        Parameters
        ----------
        animate : {"k", "t"}, optional
            Animation axis. ``"k"`` animates frequency bins at fixed time.
            ``"t"`` animates time at fixed frequency.
        frames, k_frames, t_frames : sequence/slice, optional
            Explicit frame indices. ``frames`` is generic for the selected axis.
            If omitted:
            - animate="k": all frequency bins are used
            - animate="t": all time indices are used
        saveas : str or Path, optional
            Output animation path (e.g. ``"mode.mp4"`` or ``"mode.gif"``).
            If omitted, animation object is returned without saving.
        """
        import matplotlib.pyplot as plt
        from matplotlib import animation as mpl_animation

        freqs = np.asarray(self.frequencies, dtype=float)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("TransmissionResult.frequencies must be a non-empty 1D array")

        spectrum = np.asarray(self.transmission)
        if not np.iscomplexobj(spectrum):
            raise ValueError(
                "animate_mode requires complex raw FFT data. "
                "Recompute transmission with raw_fft_output=True."
            )
        if spectrum.ndim not in (4, 5):
            raise ValueError(
                "Expected raw spectrum shape (freq,z,x,comp) or (freq,z,y,x,comp); "
                f"got {spectrum.shape}"
            )

        n_time_meta = self.metadata.get("n_time")
        try:
            n_time = int(n_time_meta)
        except (TypeError, ValueError):
            n_time = int(2 * (freqs.size - 1))
        if n_time <= 0:
            raise ValueError("Unable to infer valid n_time for inverse FFT reconstruction")

        n_z = int(spectrum.shape[1])
        z_idx = int(z_layer)
        if z_idx < 0:
            z_idx += n_z
        if z_idx < 0 or z_idx >= n_z:
            raise ValueError(f"z_layer={z_layer} out of range for n_z={n_z}")

        n_comp = int(spectrum.shape[-1])
        comp_idx = self._resolve_component_index(component, n_comp)

        copy_y_int = int(copy_y)
        if copy_y_int < 1:
            raise ValueError("copy_y must be >= 1")

        phase_offset_rad = np.deg2rad(float(phase_deg))
        mode_key = str(mode).lower().strip()
        if mode_key not in ("real", "imag", "imaginary", "abs", "magnitude", "phase"):
            raise ValueError("mode must be one of: 'real', 'imag', 'abs', 'phase'")

        unit_scales = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
        freq_scale = unit_scales.get(str(freq_unit).lower())
        if freq_scale is None:
            raise ValueError(
                f"Unsupported freq_unit={freq_unit!r}. Use one of {list(unit_scales)}."
            )

        x_unit_key = str(x_unit).lower().strip()
        if self.dx is None or x_unit_key in ("index", "idx", "cell", "cells"):
            x_step = 1.0
            x_unit_label = "index"
        else:
            x_scales = {"m": 1.0, "um": 1e6, "nm": 1e9}
            if x_unit_key not in x_scales:
                raise ValueError("x_unit must be one of: 'index', 'm', 'um', 'nm'")
            x_step = float(self.dx) * float(x_scales[x_unit_key])
            x_unit_label = x_unit_key

        out_freq_scale = unit_scales.get(str(freq_unit).lower(), 1e9)
        animate_key = str(animate).lower().strip()
        if animate_key not in {"k", "t"}:
            raise ValueError("animate must be either 'k' or 't'")

        def _materialize_indices(
            spec: Optional[Union[Sequence[int], slice]],
            *,
            upper: int,
            name: str,
        ) -> Optional[list[int]]:
            if spec is None:
                return None
            if isinstance(spec, slice):
                start, stop, step = spec.indices(upper)
                out = list(range(start, stop, step))
            else:
                out = [int(v) for v in spec]
            for idx in out:
                if idx < 0 or idx >= upper:
                    raise ValueError(f"{name} contains {idx}, allowed range [0, {upper - 1}]")
            return out

        generic_frames = _materialize_indices(
            frames,
            upper=freqs.size if animate_key == "k" else n_time,
            name="frames",
        )

        frame_pairs: list[tuple[int, int]] = []
        if animate_key == "k":
            k_list = (
                generic_frames
                or _materialize_indices(k_frames, upper=freqs.size, name="k_frames")
            )
            if k_list is None:
                if k is not None:
                    k_val = int(k)
                    if k_val < 0 or k_val >= freqs.size:
                        raise ValueError(f"k={k_val} out of range [0, {freqs.size - 1}]")
                    k_list = [k_val]
                elif f is not None:
                    k_list = [int(np.argmin(np.abs(freqs - float(f) * float(freq_scale))))]
                else:
                    k_list = list(range(freqs.size))

            t_fixed = int(np.clip(int(t_show), 0, n_time - 1))
            frame_pairs = [(int(k_idx), t_fixed) for k_idx in k_list]
        else:
            if (f is None) == (k is None):
                raise ValueError("For animate='t', provide exactly one selector: either f=... or k=...")

            if k is None:
                k_fixed = int(np.argmin(np.abs(freqs - float(f) * float(freq_scale))))
            else:
                k_fixed = int(k)
            if k_fixed < 0 or k_fixed >= freqs.size:
                raise ValueError(f"k={k_fixed} out of range [0, {freqs.size - 1}]")

            t_list = (
                generic_frames
                or _materialize_indices(t_frames, upper=n_time, name="t_frames")
                or list(range(n_time))
            )
            frame_pairs = [(k_fixed, int(t_idx)) for t_idx in t_list]

        if not frame_pairs:
            raise ValueError("No animation frames selected")

        def _frame_data(k_idx: int, t_idx: int) -> tuple[np.ndarray, np.ndarray, int]:
            xy_complex, y_block = self._reconstruct_mode_xy(
                spectrum,
                k_idx=int(k_idx),
                n_time=n_time,
                t_idx=int(t_idx),
                z_idx=z_idx,
                comp_idx=comp_idx,
                y_slice=y_slice,
                copy_y=copy_y_int,
                phase_offset_rad=phase_offset_rad,
                reconstruction=reconstruction,
            )

            if mode_key == "real":
                xy_vis = np.real(xy_complex)
            elif mode_key in ("imag", "imaginary"):
                xy_vis = np.imag(xy_complex)
            elif mode_key in ("abs", "magnitude"):
                xy_vis = np.abs(xy_complex)
            else:
                xy_vis = np.angle(xy_complex)
            return np.asarray(xy_complex), np.asarray(xy_vis), int(y_block)

        def _default_clim(arr: np.ndarray) -> tuple[float, float]:
            peak = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
            if mode_key in ("real", "imag", "imaginary"):
                vlim = peak * float(vlim_scale)
                if not np.isfinite(vlim) or vlim <= 0:
                    vlim = peak if peak > 0 else 1e-12
                return -vlim, +vlim
            if mode_key == "phase":
                return -np.pi, np.pi
            return 0.0, peak if peak > 0 else 1e-12

        _, xy_vis0, y_block0 = _frame_data(*frame_pairs[0])
        n_x = int(xy_vis0.shape[1])
        extent = [0.0, n_x * x_step, 0.0, float(xy_vis0.shape[0])]

        plot_kwargs = dict(imshow_kwargs)
        user_vmin = plot_kwargs.pop("vmin", None)
        user_vmax = plot_kwargs.pop("vmax", None)
        color_scale_key = str(color_scale).lower().strip()
        if color_scale_key not in ("global", "frame"):
            raise ValueError("color_scale must be one of: 'global', 'frame'")

        if user_vmin is None or user_vmax is None:
            if color_scale_key == "global":
                if mode_key == "phase":
                    default_vmin, default_vmax = -np.pi, np.pi
                else:
                    peak = float(np.max(np.abs(xy_vis0))) if xy_vis0.size > 0 else 0.0
                    iter_pairs = frame_pairs[1:]
                    if show_progress and len(frame_pairs) > 2:
                        iter_pairs = tqdm(iter_pairs, desc="Computing animation color scale")
                    for k_idx, t_idx in iter_pairs:
                        _, frame_vis, _ = _frame_data(k_idx, t_idx)
                        if frame_vis.size > 0:
                            peak = max(peak, float(np.max(np.abs(frame_vis))))
                    if mode_key in ("real", "imag", "imaginary"):
                        vlim = peak * float(vlim_scale)
                        if not np.isfinite(vlim) or vlim <= 0:
                            vlim = peak if peak > 0 else 1e-12
                        default_vmin, default_vmax = -vlim, +vlim
                    else:
                        default_vmin, default_vmax = 0.0, peak if peak > 0 else 1e-12
            else:
                default_vmin, default_vmax = _default_clim(xy_vis0)
            vmin = default_vmin if user_vmin is None else float(user_vmin)
            vmax = default_vmax if user_vmax is None else float(user_vmax)
        else:
            vmin = float(user_vmin)
            vmax = float(user_vmax)

        if cmap is None:
            if mode_key in ("abs", "magnitude"):
                cmap = "inferno"
            elif mode_key == "phase":
                cmap = "twilight"
            else:
                cmap = "coolwarm"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure

        img = ax.imshow(
            xy_vis0,
            aspect=aspect,
            origin=origin,
            cmap=cmap,
            interpolation=interpolation,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            **plot_kwargs,
        )

        ax.set_xlabel(f"x [{x_unit_label}]")
        ax.set_ylabel("y [index]")

        if copy_y_int > 1 and separator_lines and y_block0 > 0:
            for i in range(1, copy_y_int):
                y_sep = i * y_block0
                ax.hlines(
                    y=y_sep,
                    xmin=extent[0],
                    xmax=extent[1],
                    colors=separator_color,
                    linestyles=separator_style,
                    linewidth=separator_linewidth,
                )

        if x_lines is not None:
            for x_line in x_lines:
                x_val = float(x_line) * x_step if x_lines_in_index else float(x_line)
                ax.axvline(x_val, color="red", linewidth=1.1, alpha=0.9)

        if y_lines is not None:
            for y_line in y_lines:
                ax.axhline(float(y_line), color="cyan", linewidth=1.1, linestyle="--", alpha=0.9)

        if y_spans is not None:
            for y0, y1 in y_spans:
                ax.axhspan(
                    float(y0),
                    float(y1),
                    color=y_span_color,
                    alpha=float(y_span_alpha),
                )

        if flip_x:
            ax.invert_xaxis()

        if colorbar:
            cbar_label = {
                "real": "Re(m)",
                "imag": "Im(m)",
                "imaginary": "Im(m)",
                "abs": "|m|",
                "magnitude": "|m|",
                "phase": "arg(m) [rad]",
            }[mode_key]
            fig.colorbar(img, ax=ax, label=cbar_label)

        def _set_title(k_idx: int, t_idx: int, frame_idx: int) -> None:
            f_disp = float(freqs[k_idx]) / float(out_freq_scale)
            ax.set_title(
                f"Mode @ f={f_disp:.4g} {freq_unit} (k={k_idx}, t={t_idx}, z={z_idx}) "
                f"[{frame_idx + 1}/{len(frame_pairs)}]"
            )

        def _update(frame_idx: int):
            k_idx, t_idx = frame_pairs[int(frame_idx)]
            _, xy_vis, _ = _frame_data(k_idx, t_idx)
            img.set_data(xy_vis)
            if color_scale_key == "frame" and (user_vmin is None or user_vmax is None):
                auto_vmin, auto_vmax = _default_clim(xy_vis)
                img.set_clim(
                    auto_vmin if user_vmin is None else float(user_vmin),
                    auto_vmax if user_vmax is None else float(user_vmax),
                )
            _set_title(k_idx, t_idx, int(frame_idx))
            return (img,)

        _update(0)
        animation = mpl_animation.FuncAnimation(
            fig,
            _update,
            frames=len(frame_pairs),
            interval=int(interval),
            repeat=bool(repeat),
            blit=False,
        )

        save_path: Optional[Path] = None
        if saveas is not None:
            save_path = Path(saveas)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            writer_name = writer
            if writer_name is None:
                ext = save_path.suffix.lower()
                if ext == ".gif":
                    writer_name = "pillow"
                elif ext in {".mp4", ".m4v", ".mov"}:
                    writer_name = "ffmpeg"
                else:
                    raise ValueError(
                        "Cannot infer writer from extension. Use .gif/.mp4 or pass writer=..."
                    )
            save_kwargs = dict(animation_save_kwargs or {})
            save_kwargs.setdefault("fps", int(fps))
            save_kwargs.setdefault("dpi", int(dpi))
            animation.save(str(save_path), writer=writer_name, **save_kwargs)

        frame_values = [int(pair[0] if animate_key == "k" else pair[1]) for pair in frame_pairs]
        meta = {
            "animate": animate_key,
            "frame_count": int(len(frame_pairs)),
            "frame_values": np.asarray(frame_values, dtype=int),
            "frame_pairs": np.asarray(frame_pairs, dtype=int),
            "frequency_unit": str(freq_unit),
            "phase_shift_deg": float(phase_deg),
            "reconstruction": str(reconstruction),
            "z_index": int(z_idx),
            "component_index": int(comp_idx),
            "x_unit": x_unit_label,
            "x_step": float(x_step),
            "mode": mode_key,
            "saved_to": None if save_path is None else str(save_path),
        }
        return fig, ax, animation, meta

    def save_mode_visualizations(
        self,
        output_dir: Union[str, Path],
        *,
        f: Optional[Union[float, Sequence[float]]] = None,
        k: Optional[Union[int, Sequence[int]]] = None,
        freq_unit: str = "GHz",
        filename_template: str = "mode_k{k:05d}_f{frequency:.6f}_{unit}.{ext}",
        image_format: str = "png",
        dpi: Optional[int] = None,
        overwrite: bool = True,
        close_figures: bool = True,
        show_progress: bool = True,
        savefig_kwargs: Optional[dict[str, Any]] = None,
        **visualize_kwargs,
    ) -> list[Path]:
        """Save per-frequency mode visualizations to ``output_dir``.

        By default (``f=None`` and ``k=None``), saves all available frequency bins.
        """
        import matplotlib.pyplot as plt

        if "ax" in visualize_kwargs:
            raise ValueError("save_mode_visualizations does not accept explicit ax=...")
        if (f is not None) and (k is not None):
            raise ValueError("Provide at most one selector: f=... or k=...")

        freqs = np.asarray(self.frequencies, dtype=float)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("TransmissionResult.frequencies must be a non-empty 1D array")

        unit_scales = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
        freq_scale = unit_scales.get(str(freq_unit).lower())
        if freq_scale is None:
            raise ValueError(
                f"Unsupported freq_unit={freq_unit!r}. Use one of {list(unit_scales)}."
            )

        selected_bins: list[int] = []
        if f is not None:
            freq_values = [float(f)] if np.isscalar(f) else [float(v) for v in f]
            for f_val in freq_values:
                hz = float(f_val) * float(freq_scale)
                selected_bins.append(int(np.argmin(np.abs(freqs - hz))))
        elif k is not None:
            k_values = [int(k)] if np.isscalar(k) else [int(v) for v in k]
            for k_val in k_values:
                if k_val < 0 or k_val >= freqs.size:
                    raise ValueError(f"k={k_val} out of range [0, {freqs.size - 1}]")
                selected_bins.append(int(k_val))
        else:
            selected_bins = list(range(freqs.size))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fmt = str(image_format).lower().strip(".")
        fig_save_kwargs = dict(savefig_kwargs or {})
        if dpi is not None:
            fig_save_kwargs.setdefault("dpi", int(dpi))

        iterator = selected_bins
        if show_progress and len(selected_bins) > 1:
            iterator = tqdm(selected_bins, desc="Saving transmission mode visualizations")

        saved_paths: list[Path] = []
        for k_idx in iterator:
            fig, _, meta = self.visualize_mode(
                k=int(k_idx),
                freq_unit=freq_unit,
                **visualize_kwargs,
            )

            file_name = filename_template.format(
                k=int(meta["k"]),
                frequency=float(meta["frequency_value"]),
                frequency_hz=float(meta["frequency_hz"]),
                unit=str(meta["frequency_unit"]).lower(),
                mode=str(meta["mode"]),
                t=int(meta["time_index"]),
                phase_deg=float(meta.get("phase_shift_deg", 0.0)),
                reconstruction=str(meta.get("reconstruction", "real_signal")),
                z=int(meta["z_index"]),
                component=int(meta["component_index"]),
                ext=fmt,
            )

            file_path = out_dir / file_name
            if file_path.exists() and not overwrite:
                if close_figures:
                    plt.close(fig)
                raise FileExistsError(f"File already exists: {file_path}")

            fig.savefig(file_path, format=fmt, **fig_save_kwargs)
            saved_paths.append(file_path)

            if close_figures:
                plt.close(fig)

        return saved_paths

    def calculate_modes(
        self,
        f: Optional[Union[float, Sequence[float]]] = None,
        *,
        k: Optional[Union[int, Sequence[int]]] = None,
        freq_unit: str = "GHz",
        t_show: int = 0,
        phase_deg: float = 0.0,
        reconstruction: ReconstructionMode = "real_signal",
        z_layer: int = 0,
        component: Union[int, str] = "z",
        y_slice: Optional[slice] = None,
        copy_y: int = 1,
    ) -> "TransmissionModesResult":
        """Precompute reconstructed mode maps for selected frequencies/bins.

        Parameters
        ----------
        f : float or sequence of float, optional
            Frequency/frequencies in ``freq_unit`` units.
        k : int or sequence of int, optional
            Direct rFFT bin index/indices.
        freq_unit : str, optional
            Unit used for ``f`` and for visualization labels.
        t_show, phase_deg, reconstruction, z_layer, component, y_slice, copy_y
            Same semantics as :meth:`visualize_mode`.

        Returns
        -------
        TransmissionModesResult
            Container with precomputed complex mode maps and visualization helpers.
        """
        if (f is None) == (k is None):
            raise ValueError("Provide exactly one selector: either f=... or k=...")

        freqs = np.asarray(self.frequencies, dtype=float)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("TransmissionResult.frequencies must be a non-empty 1D array")

        spectrum = np.asarray(self.transmission)
        if not np.iscomplexobj(spectrum):
            raise ValueError(
                "calculate_modes requires complex raw FFT data. "
                "Recompute transmission with raw_fft_output=True."
            )
        if spectrum.ndim not in (4, 5):
            raise ValueError(
                "Expected raw spectrum shape (freq,z,x,comp) or (freq,z,y,x,comp); "
                f"got {spectrum.shape}"
            )

        unit_scales = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
        freq_scale = unit_scales.get(str(freq_unit).lower())
        if freq_scale is None:
            raise ValueError(
                f"Unsupported freq_unit={freq_unit!r}. Use one of {list(unit_scales)}."
            )

        selected_bins: list[int] = []
        requested_hz: list[float] = []
        if f is not None:
            freq_values = [float(f)] if np.isscalar(f) else [float(v) for v in f]
            for f_val in freq_values:
                hz = f_val * float(freq_scale)
                selected_bins.append(int(np.argmin(np.abs(freqs - hz))))
                requested_hz.append(float(hz))
        else:
            k_values = [int(k)] if np.isscalar(k) else [int(v) for v in k]
            for k_val in k_values:
                if k_val < 0 or k_val >= freqs.size:
                    raise ValueError(f"k={k_val} out of range [0, {freqs.size - 1}]")
                selected_bins.append(int(k_val))
                requested_hz.append(float(freqs[k_val]))

        n_time_meta = self.metadata.get("n_time")
        try:
            n_time = int(n_time_meta)
        except (TypeError, ValueError):
            n_time = int(2 * (freqs.size - 1))
        if n_time <= 0:
            raise ValueError("Unable to infer valid n_time for inverse FFT reconstruction")

        n_z = int(spectrum.shape[1])
        z_idx = int(z_layer)
        if z_idx < 0:
            z_idx += n_z
        if z_idx < 0 or z_idx >= n_z:
            raise ValueError(f"z_layer={z_layer} out of range for n_z={n_z}")

        n_comp = int(spectrum.shape[-1])
        comp_idx = self._resolve_component_index(component, n_comp)

        t_idx = int(np.clip(int(t_show), 0, n_time - 1))
        copy_y_int = int(copy_y)
        if copy_y_int < 1:
            raise ValueError("copy_y must be >= 1")
        phase_offset_rad = np.deg2rad(float(phase_deg))

        precomputed: list[dict[str, Any]] = []

        for k_idx, req_hz in zip(selected_bins, requested_hz):
            xy_complex, y_block = self._reconstruct_mode_xy(
                spectrum,
                k_idx=int(k_idx),
                n_time=n_time,
                t_idx=t_idx,
                z_idx=z_idx,
                comp_idx=comp_idx,
                y_slice=y_slice,
                copy_y=copy_y_int,
                phase_offset_rad=phase_offset_rad,
                reconstruction=reconstruction,
            )

            precomputed.append(
                {
                    "k": int(k_idx),
                    "frequency_hz": float(freqs[k_idx]),
                    "requested_frequency_hz": float(req_hz),
                    "frequency_value": float(freqs[k_idx] / freq_scale),
                    "frequency_unit": str(freq_unit),
                    "time_index": int(t_idx),
                    "phase_shift_deg": float(phase_deg),
                    "reconstruction": str(reconstruction),
                    "n_time": int(n_time),
                    "z_index": int(z_idx),
                    "component_index": int(comp_idx),
                    "component": str(component),
                    "xy_complex": np.asarray(xy_complex),
                    "y_block": int(y_block),
                    "copy_y": int(copy_y_int),
                }
            )

        return TransmissionModesResult(
            modes=precomputed,
            dx=self.dx,
            freq_unit=str(freq_unit),
        )


@dataclass
class TransmissionModesResult:
    """Precomputed transmission mode maps for selected frequencies."""

    modes: list[dict[str, Any]]
    dx: Optional[float] = None
    freq_unit: str = "GHz"

    def __len__(self) -> int:
        return len(self.modes)

    def __repr__(self) -> str:
        return f"TransmissionModesResult(n_modes={len(self.modes)}, freq_unit={self.freq_unit!r})"

    @property
    def frequencies_hz(self) -> np.ndarray:
        return np.asarray([float(m["frequency_hz"]) for m in self.modes], dtype=float)

    def _mode_to_display(self, xy_complex: np.ndarray, mode: str) -> np.ndarray:
        mode_key = str(mode).lower().strip()
        if mode_key == "real":
            return np.real(xy_complex)
        if mode_key in ("imag", "imaginary"):
            return np.imag(xy_complex)
        if mode_key in ("abs", "magnitude"):
            return np.abs(xy_complex)
        if mode_key == "phase":
            return np.angle(xy_complex)
        raise ValueError("mode must be one of: 'real', 'imag', 'abs', 'phase'")

    def _x_step(self, x_unit: str) -> tuple[float, str]:
        x_key = str(x_unit).lower().strip()
        if self.dx is None or x_key in ("index", "idx", "cell", "cells"):
            return 1.0, "index"
        scales = {"m": 1.0, "um": 1e6, "nm": 1e9}
        if x_key not in scales:
            raise ValueError("x_unit must be one of: 'index', 'm', 'um', 'nm'")
        return float(self.dx) * float(scales[x_key]), x_key

    def _select_indices(
        self,
        *,
        index: Optional[int],
        f: Optional[float],
        k: Optional[int],
        freq_unit: str,
    ) -> list[int]:
        if index is not None:
            idx = int(index)
            if idx < 0 or idx >= len(self.modes):
                raise ValueError(f"index={idx} out of range [0, {len(self.modes) - 1}]")
            return [idx]

        if k is not None:
            k_val = int(k)
            for i, meta in enumerate(self.modes):
                if int(meta["k"]) == k_val:
                    return [i]
            raise ValueError(f"Requested k={k_val} is not precomputed")

        if f is not None:
            unit_scales = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
            scale = unit_scales.get(str(freq_unit).lower())
            if scale is None:
                raise ValueError(
                    f"Unsupported freq_unit={freq_unit!r}. Use one of {list(unit_scales)}."
                )
            target_hz = float(f) * float(scale)
            arr = np.asarray([float(m["frequency_hz"]) for m in self.modes], dtype=float)
            return [int(np.argmin(np.abs(arr - target_hz)))]

        if len(self.modes) == 1:
            return [0]
        return list(range(len(self.modes)))

    def visualize(
        self,
        *,
        index: Optional[int] = None,
        f: Optional[float] = None,
        k: Optional[int] = None,
        freq_unit: Optional[str] = None,
        mode: str = "real",
        ncols: int = 3,
        figsize: Optional[tuple[float, float]] = None,
        dpi: int = 100,
        aspect: str = "auto",
        origin: str = "lower",
        cmap: Optional[str] = None,
        colorbar: bool = True,
        interpolation: str = "nearest",
        x_unit: str = "nm",
        x_lines: Optional[Sequence[float]] = None,
        x_lines_in_index: bool = False,
        y_lines: Optional[Sequence[float]] = None,
        y_spans: Optional[Sequence[tuple[float, float]]] = None,
        y_span_color: str = "cyan",
        y_span_alpha: float = 0.15,
        flip_x: bool = True,
        separator_lines: bool = True,
        separator_style: str = "--",
        separator_color: str = "black",
        separator_linewidth: float = 1.0,
        vlim_scale: float = 0.1,
        **imshow_kwargs,
    ):
        """Visualize one or many precomputed mode maps."""
        import matplotlib.pyplot as plt

        if not self.modes:
            raise ValueError("No precomputed modes available")

        unit_out = str(freq_unit or self.freq_unit)
        selected = self._select_indices(index=index, f=f, k=k, freq_unit=unit_out)
        x_step, x_label = self._x_step(x_unit)
        mode_key = str(mode).lower().strip()

        if cmap is None:
            if mode_key in ("abs", "magnitude"):
                cmap = "inferno"
            elif mode_key == "phase":
                cmap = "twilight"
            else:
                cmap = "coolwarm"

        def _plot_one(ax, meta: dict[str, Any]):
            xy_complex = np.asarray(meta["xy_complex"])
            xy_vis = self._mode_to_display(xy_complex, mode_key)

            n_x = int(xy_vis.shape[1])
            extent = [0.0, n_x * x_step, 0.0, float(xy_vis.shape[0])]

            plot_kwargs = dict(imshow_kwargs)
            vmin = plot_kwargs.pop("vmin", None)
            vmax = plot_kwargs.pop("vmax", None)
            if vmin is None or vmax is None:
                peak = float(np.max(np.abs(xy_vis))) if xy_vis.size > 0 else 0.0
                if mode_key in ("real", "imag", "imaginary"):
                    vlim = peak * float(vlim_scale)
                    if not np.isfinite(vlim) or vlim <= 0:
                        vlim = peak if peak > 0 else 1e-12
                    default_vmin, default_vmax = -vlim, +vlim
                elif mode_key == "phase":
                    default_vmin, default_vmax = -np.pi, np.pi
                else:
                    default_vmin, default_vmax = 0.0, peak if peak > 0 else 1e-12
                if vmin is None:
                    vmin = default_vmin
                if vmax is None:
                    vmax = default_vmax

            img = ax.imshow(
                xy_vis,
                aspect=aspect,
                origin=origin,
                cmap=cmap,
                interpolation=interpolation,
                extent=extent,
                vmin=vmin,
                vmax=vmax,
                **plot_kwargs,
            )

            out_scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}.get(
                unit_out.lower(),
                1e9,
            )
            f_disp = float(meta["frequency_hz"]) / float(out_scale)
            ax.set_title(f"Mode @ f={f_disp:.4g} {unit_out} (k={meta['k']}, t={meta['time_index']})")
            ax.set_xlabel(f"x [{x_label}]")
            ax.set_ylabel("y [index]")

            if x_lines is not None:
                for x_line in x_lines:
                    xv = float(x_line) * x_step if x_lines_in_index else float(x_line)
                    ax.axvline(xv, color="red", linewidth=1.1, alpha=0.9)

            if y_lines is not None:
                for y_line in y_lines:
                    ax.axhline(float(y_line), color="cyan", linewidth=1.1, linestyle="--", alpha=0.9)

            if y_spans is not None:
                for y0, y1 in y_spans:
                    ax.axhspan(
                        float(y0),
                        float(y1),
                        color=y_span_color,
                        alpha=float(y_span_alpha),
                    )

            if flip_x:
                ax.invert_xaxis()

            if (
                separator_lines
                and int(meta.get("copy_y", 1)) > 1
                and int(meta.get("y_block", 0)) > 0
            ):
                for i in range(1, int(meta["copy_y"])):
                    y_sep = i * int(meta["y_block"])
                    ax.hlines(
                        y=y_sep,
                        xmin=extent[0],
                        xmax=extent[1],
                        colors=separator_color,
                        linestyles=separator_style,
                        linewidth=separator_linewidth,
                    )

            if colorbar:
                cbar_label = {
                    "real": "Re(m)",
                    "imag": "Im(m)",
                    "imaginary": "Im(m)",
                    "abs": "|m|",
                    "magnitude": "|m|",
                    "phase": "arg(m) [rad]",
                }[mode_key]
                ax.figure.colorbar(img, ax=ax, label=cbar_label)

            result_meta = dict(meta)
            result_meta["mode"] = mode_key
            result_meta["xy"] = xy_vis
            result_meta["x_unit"] = x_label
            result_meta["x_step"] = float(x_step)
            result_meta["extent"] = extent
            return result_meta

        if len(selected) == 1:
            if figsize is None:
                figsize = (13.0, 4.5)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            meta = _plot_one(ax, self.modes[selected[0]])
            fig.tight_layout()
            return fig, ax, meta

        ncols_int = max(1, int(ncols))
        nrows = int(math.ceil(len(selected) / float(ncols_int)))
        if figsize is None:
            figsize = (5.6 * ncols_int, 4.1 * nrows)
        fig, axes = plt.subplots(nrows, ncols_int, figsize=figsize, dpi=dpi, squeeze=False)
        axes_flat = axes.reshape(-1)
        used_axes = []
        metas = []
        for i, idx in enumerate(selected):
            ax = axes_flat[i]
            metas.append(_plot_one(ax, self.modes[idx]))
            used_axes.append(ax)
        for i in range(len(selected), axes_flat.size):
            fig.delaxes(axes_flat[i])
        fig.tight_layout()
        return fig, used_axes, metas

    def save_visualizations(
        self,
        output_dir: Union[str, Path],
        *,
        index: Optional[int] = None,
        f: Optional[float] = None,
        k: Optional[int] = None,
        freq_unit: Optional[str] = None,
        filename_template: str = "mode_k{k:05d}_f{frequency:.6f}_{unit}.{ext}",
        image_format: str = "png",
        dpi: Optional[int] = None,
        overwrite: bool = True,
        close_figures: bool = True,
        show_progress: bool = True,
        savefig_kwargs: Optional[dict[str, Any]] = None,
        **visualize_kwargs,
    ) -> list[Path]:
        """Save one image per precomputed mode to ``output_dir``."""
        import matplotlib.pyplot as plt

        if "ax" in visualize_kwargs:
            raise ValueError("save_visualizations does not accept explicit ax=...")

        unit_out = str(freq_unit or self.freq_unit)
        selected = self._select_indices(index=index, f=f, k=k, freq_unit=unit_out)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_scale = float({"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}.get(unit_out.lower(), 1e9))

        fmt = str(image_format).lower().strip(".")
        fig_save_kwargs = dict(savefig_kwargs or {})
        if dpi is not None:
            fig_save_kwargs.setdefault("dpi", int(dpi))

        iterator = selected
        if show_progress and len(selected) > 1:
            iterator = tqdm(selected, desc="Saving precomputed transmission mode visualizations")

        saved_paths: list[Path] = []
        for mode_idx in iterator:
            fig, _, meta = self.visualize(
                index=int(mode_idx),
                freq_unit=unit_out,
                **visualize_kwargs,
            )
            file_name = filename_template.format(
                k=int(meta["k"]),
                frequency=float(meta["frequency_hz"]) / out_scale,
                frequency_hz=float(meta["frequency_hz"]),
                unit=unit_out.lower(),
                mode=str(meta["mode"]),
                t=int(meta["time_index"]),
                phase_deg=float(meta.get("phase_shift_deg", 0.0)),
                reconstruction=str(meta.get("reconstruction", "real_signal")),
                z=int(meta["z_index"]),
                component=int(meta["component_index"]),
                ext=fmt,
            )

            file_path = out_dir / file_name
            if file_path.exists() and not overwrite:
                if close_figures:
                    plt.close(fig)
                raise FileExistsError(f"File already exists: {file_path}")
            fig.savefig(file_path, format=fmt, **fig_save_kwargs)
            saved_paths.append(file_path)
            if close_figures:
                plt.close(fig)

        return saved_paths


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


def _edge_taper_weighted_average(
    data: np.ndarray,
    *,
    axes: tuple[int, ...],
    taper_power: float,
) -> np.ndarray:
    """Apply separable Hann taper over selected axes and reduce them."""
    weighted = np.asarray(data, dtype=float)
    norm = 1.0
    for axis in axes:
        axis_len = weighted.shape[axis]
        axis_weights = _compute_hann_weights(axis_len, taper_power)
        shape = [1] * weighted.ndim
        shape[axis] = axis_len
        weighted = weighted * axis_weights.reshape(shape)
        norm *= axis_weights.sum()

    if norm <= 0:
        norm = 1.0
    return weighted.sum(axis=axes) / norm


def _apply_transmission_method(
    spectrum: np.ndarray,
    *,
    component_weights: np.ndarray,
    method: TransmissionMethod,
    window_axis: Optional[int],
) -> np.ndarray:
    """Convert complex spectrum to real-valued transmission metric."""
    if spectrum.ndim < 2:
        raise ValueError(f"Spectrum must have at least 2 dims (..., comp), got {spectrum.shape}")

    n_comp = spectrum.shape[-1]
    if component_weights.size < n_comp:
        raise ValueError(
            f"Insufficient component_weights size={component_weights.size} for n_comp={n_comp}"
        )

    if method == "power_ratio":
        metric = np.zeros(spectrum.shape[:-1], dtype=float)
        for comp_idx in range(n_comp):
            w = float(component_weights[comp_idx])
            if w == 0.0:
                continue
            metric += np.abs(spectrum[..., comp_idx]) * w
        return metric

    if method == "circular":
        if n_comp < 2:
            raise ValueError("method='circular' requires at least 2 components (mx,my)")
        mx = spectrum[..., 0] * float(component_weights[0])
        my = spectrum[..., 1] * float(component_weights[1])
        m_plus = (mx + 1j * my) / np.sqrt(2.0)
        m_minus = (mx - 1j * my) / np.sqrt(2.0)
        metric = 0.5 * (np.abs(m_plus) + np.abs(m_minus))
        # Keep optional longitudinal/extra contributions explicit via weights.
        for comp_idx in range(2, n_comp):
            w = float(component_weights[comp_idx])
            if w == 0.0:
                continue
            metric += np.abs(spectrum[..., comp_idx]) * w
        return metric

    if method == "cpsd":
        if window_axis is None:
            raise ValueError(
                "method='cpsd' requires per-window spectrum. Use spatial_window_mode='post_fft'."
            )
        metric = np.zeros(spectrum.shape[:-1], dtype=float)
        for comp_idx in range(n_comp):
            w = float(component_weights[comp_idx])
            if w == 0.0:
                continue
            comp_spec = spectrum[..., comp_idx]
            ref = np.take(comp_spec, indices=0, axis=window_axis)
            ref = np.expand_dims(ref, axis=window_axis)
            metric += np.abs(comp_spec * np.conj(ref)) * w
        return metric

    raise ValueError(f"Unsupported transmission method: {method}")


def _aggregate_pre_fft(
    metric: np.ndarray,
    mode: AverageMode,
    edge_taper_power: float,
) -> np.ndarray:
    """Aggregate pre-FFT metric (freq,z) or (freq,z,y) to 1D freq profile."""
    if metric.ndim == 2:
        # (freq, z)
        if mode == "none":
            return metric[:, 0]
        if mode == "mean":
            return metric.mean(axis=1)
        if mode == "median":
            return np.median(metric, axis=1)
        if mode == "edge_taper":
            return _edge_taper_weighted_average(metric, axes=(1,), taper_power=edge_taper_power)
        raise ValueError(f"Unsupported average mode: {mode}")

    if metric.ndim == 3:
        # (freq, z, y)
        if mode == "none":
            return metric[:, 0, 0]
        if mode == "mean":
            return metric.mean(axis=(1, 2))
        if mode == "median":
            return np.median(metric, axis=(1, 2))
        if mode == "edge_taper":
            return _edge_taper_weighted_average(metric, axes=(1, 2), taper_power=edge_taper_power)
        raise ValueError(f"Unsupported average mode: {mode}")

    raise ValueError(f"Expected pre-FFT metric with ndim 2 or 3, got shape {metric.shape}")


def _aggregate_spatial(
    power: np.ndarray,
    mode: AverageMode,
    edge_taper_power: float,
) -> np.ndarray:
    """Reduce local power map to 1D frequency profile.

    Supported inputs:
    - ``(freq, z, window_x)`` when y is already integrated
    - ``(freq, z, y, window_x)`` when y is retained

    Parameters
    ----------
    power : np.ndarray
        Power array with shape (freq, z, window) or (freq, z, y, window)
    mode : AverageMode
        "mean" - simple average
        "median" - median (robust to outliers)
        "edge_taper" - weighted average with Hann window
        "none" - take z=0 (and y=0 when present), mean over window
    """

    arr = np.asarray(power, dtype=float)
    if arr.ndim == 3:
        # (freq, z, window)
        if mode == "none":
            return arr[:, 0, :].mean(axis=1)
        if mode == "mean":
            return arr.mean(axis=(1, 2))
        if mode == "median":
            return np.median(arr, axis=(1, 2))
        if mode == "edge_taper":
            return _edge_taper_weighted_average(
                arr,
                axes=(1, 2),
                taper_power=edge_taper_power,
            )
        raise ValueError(f"Unsupported averaging mode: {mode}")

    if arr.ndim == 4:
        # (freq, z, y, window)
        if mode == "none":
            return arr[:, 0, 0, :].mean(axis=1)
        if mode == "mean":
            return arr.mean(axis=(1, 2, 3))
        if mode == "median":
            return np.median(arr, axis=(1, 2, 3))
        if mode == "edge_taper":
            return _edge_taper_weighted_average(
                arr,
                axes=(1, 2, 3),
                taper_power=edge_taper_power,
            )
        raise ValueError(f"Unsupported averaging mode: {mode}")

    raise ValueError(f"Expected power array ndim 3 or 4, got shape {arr.shape}")


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
                except (
                    KeyError,
                    TypeError,
                    AttributeError,
                    NameError,
                    AssertionError,
                ) as exc:
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
                dx_m = _from_object(
                    base_array, f"job_result['{dataset_name}'].zarr_array"
                )
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

            dx_m = _from_attrs(
                getattr(self._job_result, "attributes", None), "job_result.attributes"
            )
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
            if non_ellipsis_slices and isinstance(
                non_ellipsis_slices[-1], (int, np.integer)
            ):
                component_was_selected = True
                log.debug(
                    "Component was pre-selected via slicing - will add component axis"
                )

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

    def compute(
        self, config: TransmissionConfig, slice_info: Optional[Any] = None
    ) -> TransmissionResult:
        import gc  # Garbage collector for memory management

        config.ensure_valid()

        dataset = config.dataset_name or self._job_result.get_largest_m_dataset()
        data, dt = self._prepare_data(config, slice_info=slice_info)

        # Check if component was pre-selected via slicing
        component_was_selected = False
        if slice_info is not None and isinstance(slice_info, tuple):
            non_ellipsis_slices = [s for s in slice_info if s is not Ellipsis]
            if non_ellipsis_slices and isinstance(
                non_ellipsis_slices[-1], (int, np.integer)
            ):
                component_was_selected = True

        # Debug: basic metadata about data being processed
        log.debug(
            "Transmission compute: dataset=%s, data.shape=%s, dt=%s, component_pre_selected=%s",
            dataset,
            getattr(data, "shape", None),
            dt,
            component_was_selected,
        )

        n_time, n_z, n_y, n_x, n_comp = data.shape
        if config.method == "circular" and n_comp < 2:
            raise ValueError(
                "method='circular' requires at least 2 components (mx,my); "
                f"got n_comp={n_comp}"
            )

        # 🐛 CRITICAL DEBUG: Log dimensional interpretation
        log.info(
            "📊 Data dimensions: n_time=%d, n_z=%d, n_y=%d, n_x=%d, n_comp=%d",
            n_time,
            n_z,
            n_y,
            n_x,
            n_comp,
        )

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

        # Apply filtering (optional - can be None)
        # Use in-place operations where possible to save memory
        if config.filter_type is not None:
            if isinstance(config.filter_type, list):
                log.debug(f"Applying sequential filters: {config.filter_type}")
            else:
                log.debug(f"Applying filter: {config.filter_type}")
            filtered = self._fft_compute.apply_filter(data, config.filter_type)
            # Free original data if filtered is a new array
            if filtered is not data:
                del data
                gc.collect()
            log.debug(
                f"Filtered data: min={filtered.min():.8e}, max={filtered.max():.8e}"
            )
        else:
            filtered = data
            log.debug("Skipping temporal filtering (filter_type=None)")

        # Apply windowing (optional - can be None)
        if config.window_function is not None:
            windowed = self._fft_compute.apply_window(filtered, config.window_function)
            # Free filtered data if windowed is a new array
            if windowed is not filtered:
                del filtered
                gc.collect()
            log.debug(
                f"Windowed data: min={windowed.min():.8e}, max={windowed.max():.8e}"
            )
        else:
            windowed = filtered
            log.debug("Skipping temporal windowing (window_function=None)")

        window_size = min(config.spatial_window, n_x)
        step = config.spatial_step

        window_starts = list(range(0, max(n_x - window_size + 1, 1), step))
        if not window_starts:
            window_starts = [0]

        # 🐛 CRITICAL DEBUG: Log window calculation
        log.info(
            "🪟 Window calc: n_x=%d, spatial_window=%d → window_size=%d, step=%d → %d windows",
            n_x,
            config.spatial_window,
            window_size,
            step,
            len(window_starts),
        )

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
                config.component_weights,
            )
        else:
            component_weights = np.asarray(config.component_weights, dtype=float)
            if component_weights.ndim == 0:
                component_weights = np.full(
                    (n_comp,), float(component_weights), dtype=float
                )
            elif component_weights.size < n_comp:
                # If fewer weights provided, repeat last value to match n_comp
                last = (
                    float(component_weights[-1]) if component_weights.size > 0 else 1.0
                )
                component_weights = np.concatenate(
                    [
                        component_weights,
                        np.full((n_comp - component_weights.size,), last, dtype=float),
                    ]
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
        log.debug(
            "Computing FFT for full dataset (t=%d, z=%d, y=%d, x=%d, comp=%d)...",
            n_time,
            n_z,
            n_y,
            n_x,
            n_comp,
        )
        t_fft_start = time.time()

        log.debug(
            f"y_integration_mode={config.y_integration_mode}, spatial_window_mode={config.spatial_window_mode}, engine={config.engine}"
        )

        # 🔑 Determine which FFT engine to use based on config.engine parameter
        if config.engine == "scipy":
            if not _USE_SCIPY_FFT or scipy_fft is None:
                raise ValueError(
                    "engine='scipy' requested but scipy is not available. Install scipy or use engine='numpy'"
                )
            use_scipy = True
            engine_name = "scipy.fft"
        elif config.engine == "numpy":
            use_scipy = False
            engine_name = "numpy.fft"
        else:  # "auto"
            use_scipy = _USE_SCIPY_FFT
            engine_name = "scipy.fft" if use_scipy else "numpy.fft"

        log.debug(f"Using FFT engine: {engine_name}")

        # 🔑 SPATIAL WINDOW MODE: Choose between pre-FFT (local, slower) or post-FFT (global, faster)
        if config.spatial_window_mode == "pre_fft":
            # 🐢 SLOW PATH: Apply spatial windows BEFORE FFT (physically correct for local transmission)
            # This computes separate FFT for each spatial window position
            log.info(
                "Spatial window mode: PRE_FFT (computing separate FFT for each window - SLOW but local)"
            )

            # Pre-allocate result arrays
            power_map = np.zeros((n_freq, n_windows), dtype=float)
            full_spectrum = None  # Won't have single full_spectrum in this mode

            # Decide whether to parallelize pre-FFT window processing
            use_parallel_pre = _USE_JOBLIB and n_windows > 100

            if use_parallel_pre:
                log.info(
                    "Using parallel pre_fft processing with joblib (%d windows, %d CPUs)",
                    n_windows,
                    -1,  # -1 = all CPUs
                )

                def process_window_pre_fft(win_idx: int, start: int):
                    """Process single spatial window in pre_fft mode (can run in parallel)."""
                    end = min(start + window_size, n_x)
                    window_slice = slice(start, end)

                    # Extract window from time-domain data: (t, z, y, window_x, comp)
                    window_data = windowed[:, :, :, window_slice, :]

                    # Apply y-integration if requested (BEFORE FFT!)
                    if config.y_integration_mode == "sum_m":
                        # Sum over y: (t, z, y, window_x, comp) → (t, z, window_x, comp)
                        window_data_local = window_data.sum(axis=2)
                    elif config.y_integration_mode == "none":
                        window_data_local = window_data
                    else:  # "sum_fft" treated as "sum_m" in pre_fft mode
                        log.warning(
                            "y_integration_mode='sum_fft' with spatial_window_mode='pre_fft' "
                            "→ using 'sum_m' instead"
                        )
                        window_data_local = window_data.sum(axis=2)

                    # Sum over spatial window if window_size > 1
                    # Shape after y-sum: (t, z, window_x, comp) or (t, z, y, window_x, comp)
                    if config.y_integration_mode in ("sum_m", "sum_fft"):
                        # (t, z, window_x, comp) → sum over window_x → (t, z, comp)
                        window_data_summed = window_data_local.sum(axis=2)
                    else:
                        # (t, z, y, window_x, comp) → sum over window_x → (t, z, y, comp)
                        window_data_summed = window_data_local.sum(axis=3)

                    # Compute FFT for this window
                    if use_scipy:
                        window_spectrum = scipy_fft.rfft(window_data_summed, axis=0)
                    else:
                        window_spectrum = np.fft.rfft(window_data_summed, axis=0)

                    # Apply selected transmission metric, then aggregate spatial axes.
                    metric_local = _apply_transmission_method(
                        window_spectrum,
                        component_weights=component_weights,
                        method=config.method,
                        window_axis=None,
                    )
                    aggregated_local = _aggregate_pre_fft(
                        metric_local,
                        config.average_mode,
                        config.edge_taper_power,
                    )

                    return win_idx, aggregated_local

                # Run windows in parallel (threading backend to share memory)
                # Use joblib's verbose parameter for progress indication
                results_pre = Parallel(n_jobs=-1, backend="threading", verbose=10)(
                    delayed(process_window_pre_fft)(win_idx, start)
                    for win_idx, start in enumerate(window_starts)
                )

                # Collect results
                for win_idx, aggregated_local in results_pre:
                    power_map[:, win_idx] = aggregated_local

            else:
                # 🔑 Process each window separately (serial)
                for win_idx, start in tqdm(
                    enumerate(window_starts),
                    total=n_windows,
                    desc="Computing FFT per window (pre_fft mode)",
                    unit="win",
                ):
                    # Report progress via callback
                    _report_progress(
                        config.progress_callback, win_idx, n_windows, "pre_fft"
                    )

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
                        log.warning(
                            "y_integration_mode='sum_fft' with spatial_window_mode='pre_fft' "
                            "→ using 'sum_m' instead"
                        )
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

                    # Apply selected transmission metric, then aggregate spatial axes.
                    metric = _apply_transmission_method(
                        window_spectrum,
                        component_weights=component_weights,
                        method=config.method,
                        window_axis=None,
                    )
                    aggregated = _aggregate_pre_fft(
                        metric,
                        config.average_mode,
                        config.edge_taper_power,
                    )

                    power_map[:, win_idx] = aggregated

            t_fft_end = time.time()
            log.info(
                "PRE_FFT mode completed in %.3fs for %d windows",
                t_fft_end - t_fft_start,
                n_windows,
            )

            # Skip the post-FFT processing section entirely
            use_post_fft_processing = False

        else:  # "post_fft" - current fast implementation
            # 🚀 FAST PATH: Compute FFT once, then extract windows (current implementation)
            log.info(
                "🚀 Spatial window mode: POST_FFT (computing FFT once, then extracting windows - FAST)"
            )
            use_post_fft_processing = True

            # 🔑 Y-AXIS INTEGRATION: Handle different methods for summing across y-dimension
            if config.y_integration_mode == "sum_m":
                # Method 1: Sum magnetization data along y BEFORE FFT
                # windowed shape: (t, z, y, x, comp) → sum over y → (t, z, x, comp)
                log.debug("Y-integration: sum_m (summing magnetization before FFT)")
                windowed_integrated = windowed.sum(axis=2)  # Sum over y (axis=2)

                # Free windowed data - no longer needed
                del windowed
                gc.collect()

                if use_scipy:
                    full_spectrum = scipy_fft.rfft(windowed_integrated, axis=0)
                else:
                    full_spectrum = np.fft.rfft(windowed_integrated, axis=0)

                # Free integrated data
                del windowed_integrated
                gc.collect()

                log.debug(f"FFT complete: full_spectrum.shape = {full_spectrum.shape}")

            elif config.y_integration_mode == "sum_fft":
                # Method 2: Compute FFT first, THEN sum complex FFT along y (preserve phase!)
                log.debug(
                    "Y-integration: sum_fft (computing FFT first, then summing complex values)"
                )

                if use_scipy:
                    full_spectrum_raw = scipy_fft.rfft(windowed, axis=0)
                else:
                    full_spectrum_raw = np.fft.rfft(windowed, axis=0)

                # Free windowed data immediately after FFT
                del windowed
                gc.collect()

                log.debug(f"FFT complete (raw): shape = {full_spectrum_raw.shape}")

                # Sum complex FFT along y-axis: (freq, z, y, x, comp) → (freq, z, x, comp)
                # ⚠️ IMPORTANT: Sum complex values, NOT absolute values - preserves phase!
                full_spectrum = np.sum(full_spectrum_raw, axis=2)  # Sum over y (axis=2)

                # Free raw spectrum
                del full_spectrum_raw
                gc.collect()

                log.debug(
                    f"SUM_FFT: summed complex FFT over y-axis → shape {full_spectrum.shape}"
                )

            else:  # "none"
                # No y-integration: keep full 5D spectrum
                log.debug("Y-integration: none (keeping full 5D spectrum)")

                if use_scipy:
                    full_spectrum = scipy_fft.rfft(windowed, axis=0)
                else:
                    full_spectrum = np.fft.rfft(windowed, axis=0)

                # Free windowed data
                del windowed
                gc.collect()

                log.debug(f"FFT complete: full_spectrum.shape = {full_spectrum.shape}")

            t_fft_end = time.time()
            log.info(
                "FFT completed in %.3fs (shape: %s)",
                t_fft_end - t_fft_start,
                full_spectrum.shape,
            )

            # 🚀 RAW FFT OUTPUT MODE: Skip all post-processing and return raw spectrum
            if config.raw_fft_output:
                log.info(
                    "RAW FFT OUTPUT MODE: Skipping all post-processing, returning full_spectrum directly"
                )

                # Create minimal result with raw FFT spectrum
                # Note: transmission and power_map will contain the raw complex spectrum
                # User should access result.power_map or result.transmission to get full_spectrum
                metadata = {
                    "dataset": dataset,
                    "z_layer": config.z_layer,
                    "time_step": dt,
                    "n_time": int(n_time),
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
                    x_positions=np.arange(n_x, dtype=float)
                    * (dx_nm if dx_nm else 1.0),  # All X positions
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
            use_parallel = (
                _USE_JOBLIB and n_windows > 100
            )  # Only parallelize for many windows
            use_vectorized = (
                config.average_mode == "none"
                and y_already_integrated
                and config.method == "power_ratio"
                and not config.enable_circular_components
                and not config.store_component_maps
                and not use_parallel
            )

            # 🚀 ULTRA-OPTIMIZATION for average_mode='none' with sliding_window_view
            use_sliding_window = (
                use_vectorized
                and config.spatial_step == 1
                and hasattr(np.lib.stride_tricks, "sliding_window_view")
            )

            if use_sliding_window:
                # 🔥 FASTEST PATH: Zero Python loops - pure NumPy vectorization!
                log.info(
                    "Using sliding_window_view optimization (step=1, average_mode='none')"
                )
                log.info(
                    "Processing %d windows with vectorized operations (no progress bar - too fast!)",
                    n_windows,
                )
                t_process_start = time.time()

                if y_already_integrated:
                    # Y already summed: (freq, z, x, comp) → extract z=0 → (freq, x, comp)
                    relevant_spectrum = full_spectrum[:, 0, :, :]  # Extract z=0
                    log.debug(
                        "Y already integrated, extracted z=0: %s → %s",
                        full_spectrum.shape,
                        relevant_spectrum.shape,
                    )
                else:
                    # Y not summed yet: (freq, z, y, x, comp) → sum over y, extract z=0 → (freq, x, comp)
                    relevant_spectrum = full_spectrum[:, 0, :, :, :].sum(
                        axis=1
                    )  # Sum over y (axis=1)
                    log.debug(
                        "Summed spectrum over y-dimension: %s → %s",
                        full_spectrum.shape,
                        relevant_spectrum.shape,
                    )

                # Create sliding window view - NO COPIES, just strides!
                # sliding_window_view adds new axis at the END!
                # Input:  (n_freq, n_x, n_comp)
                # Output: (n_freq, n_windows, n_comp, window_size) ← window_size at END!
                windowed_view = np.lib.stride_tricks.sliding_window_view(
                    relevant_spectrum,
                    window_shape=window_size,
                    axis=1,  # Slide along x-axis
                )
                # windowed_view shape: (n_freq, n_windows, n_comp, window_size)

                # Compute power for ALL windows - iterate only over active components
                # Initialize with zeros - shape (n_freq, n_windows, window_size)
                power_all_windows = np.zeros(
                    (n_freq, n_windows, window_size), dtype=float
                )

                # Add contribution from each component with non-zero weight
                for comp_idx in range(n_comp):
                    if component_weights[comp_idx] != 0:
                        # Extract component: (n_freq, n_windows, window_size)
                        comp_fft_all = windowed_view[:, :, comp_idx, :]
                        power_all_windows += (
                            np.abs(comp_fft_all) * component_weights[comp_idx]
                        )

                # Mean over window_size dimension - NO LOOP!
                # power_all_windows shape: (n_freq, n_windows, window_size)
                power_map = power_all_windows.mean(
                    axis=2
                )  # Result: (n_freq, n_windows)

                # Clean up intermediate arrays
                del power_all_windows, windowed_view, relevant_spectrum
                gc.collect()

                # Report 100% progress for vectorized path (no loop)
                _report_progress(
                    config.progress_callback, n_windows, n_windows, "sliding_window"
                )

                t_process_end = time.time()
                log.info(
                    "Sliding window vectorization: %.3fs for %d windows (%.1f µs/window)",
                    t_process_end - t_process_start,
                    n_windows,
                    (t_process_end - t_process_start) * 1e6 / n_windows,
                )

            elif use_vectorized:
                # 🔥 OPTIMIZED PATH: Loop with reduced dimensions (for step != 1)
                log.debug(
                    "Using optimized vectorized processing (average_mode='none', step=%d)",
                    config.spatial_step,
                )
                t_process_start = time.time()

                if y_already_integrated:
                    # Y already summed: (freq, z, x, comp) → extract z=0 → (freq, x, comp)
                    relevant_spectrum = full_spectrum[:, 0, :, :]  # Extract z=0
                    log.debug(
                        "Y already integrated, extracted z=0: %s → %s",
                        full_spectrum.shape,
                        relevant_spectrum.shape,
                    )
                else:
                    # Y not summed yet: (freq, z, y, x, comp) → sum over y, extract z=0 → (freq, x, comp)
                    relevant_spectrum = full_spectrum[:, 0, :, :, :].sum(
                        axis=1
                    )  # Sum over y (axis=1)
                    log.debug(
                        "Summed spectrum over y-dimension: %s → %s",
                        full_spectrum.shape,
                        relevant_spectrum.shape,
                    )

                # Now loop with much smaller slicing operations
                for win_idx, start in tqdm(
                    enumerate(window_starts),
                    total=n_windows,
                    desc="Processing windows",
                    unit="win",
                    disable=n_windows < 10,
                ):  # Disable for very few windows
                    # Report progress via callback
                    _report_progress(
                        config.progress_callback,
                        win_idx,
                        n_windows,
                        "post_fft_vectorized",
                    )

                    end = min(start + window_size, n_x)

                    # Slice from reduced 3D array (n_freq, n_x, n_comp)
                    # instead of 5D array - much faster!
                    spectrum_slice = relevant_spectrum[
                        :, start:end, :
                    ]  # (n_freq, window_len, n_comp)

                    # Compute power - iterate only over active components
                    # Initialize with zeros
                    power_components = np.zeros((n_freq, end - start), dtype=float)

                    # Add contribution from each component with non-zero weight
                    for comp_idx in range(n_comp):
                        if component_weights[comp_idx] != 0:
                            comp_fft = spectrum_slice[
                                ..., comp_idx
                            ]  # (n_freq, window_len)
                            power_components += (
                                np.abs(comp_fft) * component_weights[comp_idx]
                            )

                    # Fast aggregation: mean over window dimension
                    # power_components shape: (n_freq, window_len)
                    power_map[:, win_idx] = power_components.mean(axis=1)

                t_process_end = time.time()
                log.info(
                    "Optimized vectorized processing: %.3fs for %d windows (%.1f µs/window)",
                    t_process_end - t_process_start,
                    n_windows,
                    (t_process_end - t_process_start) * 1e6 / n_windows,
                )

            elif use_parallel:
                # Parallel path: use joblib to process windows in parallel
                log.info(
                    "Using parallel processing with joblib (%d windows, %d CPUs)",
                    n_windows,
                    -1,
                )  # -1 = use all CPUs
                t_process_start = time.time()

                def process_window(win_idx: int, start: int):
                    """Process single window - can run in parallel."""
                    end = min(start + window_size, n_x)
                    window_slice = slice(start, end)

                    if y_already_integrated:
                        # Y already summed: (freq, z, x, comp) → extract window
                        # Result: (freq, z, window_x, comp)
                        spectrum = full_spectrum[:, :, window_slice, :]
                        window_axis = 2
                    else:
                        # Keep y dimension for y_integration_mode='none':
                        # (freq, z, y, x, comp) -> (freq, z, y, window_x, comp)
                        spectrum = full_spectrum[:, :, :, window_slice, :]
                        window_axis = 3

                    metric = _apply_transmission_method(
                        spectrum,
                        component_weights=component_weights,
                        method=config.method,
                        window_axis=window_axis,
                    )

                    aggregated = _aggregate_spatial(
                        metric,
                        config.average_mode,
                        config.edge_taper_power,
                    )

                    results = {"power": aggregated}

                    mx_fft = spectrum[..., 0] if n_comp > 0 else None
                    my_fft = spectrum[..., 1] if n_comp > 1 else None
                    mz_fft = spectrum[..., 2] if n_comp > 2 else None

                    if transverse_map is not None:
                        transverse_metric = None
                        if mx_fft is not None:
                            transverse_metric = np.abs(mx_fft)
                        if my_fft is not None:
                            my_metric = np.abs(my_fft)
                            if transverse_metric is None:
                                transverse_metric = my_metric
                            else:
                                transverse_metric += my_metric
                        if transverse_metric is None:
                            transverse_metric = np.zeros_like(metric)
                        results["transverse"] = _aggregate_spatial(
                            transverse_metric,
                            config.average_mode,
                            config.edge_taper_power,
                        )

                    if longitudinal_map is not None and mz_fft is not None:
                        results["longitudinal"] = _aggregate_spatial(
                            np.abs(mz_fft),
                            config.average_mode,
                            config.edge_taper_power,
                        )

                    if config.enable_circular_components:
                        if mx_fft is not None and my_fft is not None:
                            m_plus = (mx_fft + 1j * my_fft) / np.sqrt(2.0)
                            m_minus = (mx_fft - 1j * my_fft) / np.sqrt(2.0)
                            results["power_plus"] = _aggregate_spatial(
                                np.abs(m_plus),
                                config.average_mode,
                                config.edge_taper_power,
                            )
                            results["power_minus"] = _aggregate_spatial(
                                np.abs(m_minus),
                                config.average_mode,
                                config.edge_taper_power,
                            )
                        else:
                            zeros_metric = np.zeros_like(metric)
                            results["power_plus"] = _aggregate_spatial(
                                zeros_metric,
                                config.average_mode,
                                config.edge_taper_power,
                            )
                            results["power_minus"] = _aggregate_spatial(
                                zeros_metric,
                                config.average_mode,
                                config.edge_taper_power,
                            )

                    return win_idx, results

                # Process windows in parallel
                # Use joblib's verbose parameter for progress indication
                results_list = Parallel(n_jobs=-1, backend="threading", verbose=10)(
                    delayed(process_window)(win_idx, start)
                    for win_idx, start in enumerate(window_starts)
                )

                # Collect results
                for win_idx, results in results_list:
                    power_map[:, win_idx] = results["power"]
                    if transverse_map is not None and "transverse" in results:
                        transverse_map[:, win_idx] = results["transverse"]
                    if longitudinal_map is not None and "longitudinal" in results:
                        longitudinal_map[:, win_idx] = results["longitudinal"]
                    if config.enable_circular_components:
                        if power_plus is not None:
                            power_plus[:, win_idx] = results.get("power_plus", 0)
                        if power_minus is not None:
                            power_minus[:, win_idx] = results.get("power_minus", 0)

                # Report 100% progress for parallel path (no granular tracking)
                _report_progress(
                    config.progress_callback, n_windows, n_windows, "parallel"
                )

                t_process_end = time.time()
                log.info(
                    "Parallel processing: %.3fs for %d windows",
                    t_process_end - t_process_start,
                    n_windows,
                )

            else:
                # Standard path: use _aggregate_spatial for each window (serial)
                log.debug(
                    "Using standard serial processing (average_mode='%s')",
                    config.average_mode,
                )
                t_process_start = time.time()

                for win_idx, start in tqdm(
                    enumerate(window_starts),
                    total=n_windows,
                    desc="Processing windows",
                    unit="win",
                    disable=n_windows < 10,
                ):  # Disable for very few windows
                    # Report progress via callback
                    _report_progress(
                        config.progress_callback, win_idx, n_windows, "post_fft_serial"
                    )

                    end = min(start + window_size, n_x)
                    window_slice = slice(start, end)

                    if y_already_integrated:
                        # Y already summed: (freq, z, x, comp) → extract window
                        # Result: (freq, z, window_x, comp)
                        spectrum = full_spectrum[:, :, window_slice, :]
                        window_axis = 2
                    else:
                        # Extract window from pre-computed FFT spectrum
                        # Shape: (n_freq, n_z, n_y, window_x, n_comp)
                        spectrum = full_spectrum[:, :, :, window_slice, :]
                        window_axis = 3

                    metric = _apply_transmission_method(
                        spectrum,
                        component_weights=component_weights,
                        method=config.method,
                        window_axis=window_axis,
                    )

                    # Store longitudinal component map if requested
                    if (
                        longitudinal_map is not None
                        and n_comp > 2
                        and component_weights[2] != 0
                    ):
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
                            # Mean over all spatial dims (z,[y],window_x), keep frequency axis.
                            spatial_axes = tuple(range(1, comp_spec.ndim))
                            comp_mean = comp_spec.mean(axis=spatial_axes)
                            complex_accum[:, comp_idx] += comp_mean

                    aggregated = _aggregate_spatial(
                        metric,
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
                    if (
                        config.enable_circular_components
                        and power_plus is not None
                        and power_minus is not None
                    ):
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
                log.debug(
                    "Serial processing: %.3fs for %d windows",
                    t_process_end - t_process_start,
                    n_windows,
                )

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
            "n_time": int(n_time),
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
            log.debug(
                "Attached complex_spectra_summary shape=%s",
                getattr(complex_summary, "shape", None),
            )

        log.debug(
            "Transmission compute complete: transmission.shape=%s", transmission.shape
        )

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
    "TransmissionModesResult",
]
