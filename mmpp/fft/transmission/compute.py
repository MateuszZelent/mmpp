"""Transmission analysis core utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple
import time

import numpy as np

# Try to use scipy.fft (faster) with fallback to numpy.fft
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
    average_mode: AverageMode = "mean"  # "none" = no y/z averaging
    edge_taper_power: float = 1.5
    
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
        if self.component_weights is not None and len(self.component_weights) != 3:
            raise ValueError("component_weights must contain three entries for (mx, my, mz)")


@dataclass
class TransmissionResult:
    """Result of a transmission analysis."""

    frequencies: np.ndarray
    x_positions: np.ndarray
    transmission: np.ndarray
    power_map: np.ndarray
    reference_power: np.ndarray
    config: TransmissionConfig
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

    def plot_transmission_crosssection(
        self,
        x: float,
        freq_unit: str = "GHz",
        trim_0f: Optional[int] = None,
        flip: bool = False,
        ax=None,
        **kwargs
    ):
        """Plot 1D transmission cross-section at specific x position.

        Parameters
        ----------
        x : float
            X-position in meters (e.g., 2000e-9 for 2000 nm)
        freq_unit : str, optional
            Frequency unit ("Hz", "kHz", "MHz", "GHz"), default "GHz"
        trim_0f : int, optional
            Number of lowest frequency points to remove
        flip : bool, optional
            If True, frequency is on Y-axis and transmission on X-axis.
            If False (default), frequency on X-axis and transmission on Y-axis.
            Use flip=True to match vertical frequency axis with dispersion plots.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional matplotlib plot kwargs (color, linewidth, label, etc.)

        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects
        """
        # Import here to avoid circular imports
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        from .plot import FREQ_SCALE

        # Find closest x-position index
        x_idx = np.argmin(np.abs(self.x_positions - x))
        actual_x = self.x_positions[x_idx]

        # Get transmission slice at this x
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

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 6)), dpi=kwargs.pop("dpi", 100))
        else:
            fig = ax.figure

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
        else:
            # Frequency on X-axis (horizontal), Transmission on Y-axis (vertical) - default
            ax.plot(freqs, transmission_slice, **plot_kwargs)
            ax.set_xlabel(f"Frequency ({freq_unit})", fontsize=12)
            ax.set_ylabel("Transmission T(f)", fontsize=12)

        # Title
        ax.set_title(
            f"Transmission Cross-section at x = {actual_x*1e9:.1f} nm"
            + (f" (trimmed {trim_idx} pts)" if trim_idx > 0 else ""),
            fontsize=13,
            fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

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
    """Reduce spatial dimensions (z, y, window_x) of the local power map.
    
    Parameters
    ----------
    mode : AverageMode
        "mean" - simple average
        "median" - median (robust to outliers)
        "edge_taper" - weighted average with Hann window
        "none" - no averaging, take only first slice (z=0, y=0)
    """

    if power.ndim != 4:
        raise ValueError("Expected power array with shape (freq, z, y, window)")

    freq_axis = 0
    z_axis = 1
    y_axis = 2
    x_axis = 3

    if mode == "none":
        # Raw mode: no averaging - take only z=0, y=0, average over window_x
        # After slicing [:, 0, 0, :], we have (freq, window_x), so axis=1 for window_x
        return power[:, 0, 0, :].mean(axis=1)

    if mode == "mean":
        return power.mean(axis=(z_axis, y_axis, x_axis))

    if mode == "median":
        return np.median(power, axis=(z_axis, y_axis, x_axis))

    if mode == "edge_taper":
        n_z, n_y, n_w = power.shape[1:]
        weights_y = _compute_hann_weights(n_y, edge_taper_power)
        weights_z = np.ones((n_z,), dtype=float)
        weights_z /= weights_z.sum() if weights_z.sum() > 0 else 1.0
        weights_w = np.ones((n_w,), dtype=float)
        weights_w /= weights_w.sum() if weights_w.sum() > 0 else 1.0

        combined = (
            weights_z[:, None, None]
            * weights_y[None, :, None]
            * weights_w[None, None, :]
        )
        weighted = power * combined[None, ...]
        normalization = combined.sum()
        if normalization <= 0:
            normalization = 1.0
        return weighted.sum(axis=(z_axis, y_axis, x_axis)) / normalization

    raise ValueError(f"Unsupported averaging mode: {mode}")


class TransmissionCompute:
    """Compute transmission profiles for FFT datasets."""

    def __init__(self, fft_compute: FFTCompute, job_result: Any):
        self._fft_compute = fft_compute
        self._job_result = job_result

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

        if data.ndim == 4:  # (t, y, x, comp)
            data = data[:, np.newaxis, ...]
        elif data.ndim == 5:
            pass
        else:
            raise ValueError(
                "Transmission analysis requires 4D (t,y,x,c) or 5D (t,z,y,x,c) datasets"
            )

        if data.shape[-1] < 2:
            raise ValueError(
                "Expected vector magnetization components (>=2) in the last dimension"
            )

        return data, dt

    def compute(self, config: TransmissionConfig, slice_info: Optional[Any] = None) -> TransmissionResult:
        config.ensure_valid()

        dataset = config.dataset_name or self._job_result.get_largest_m_dataset()
        data, dt = self._prepare_data(config, slice_info=slice_info)

        # Debug: basic metadata about data being processed
        log.debug(
            "Transmission compute: dataset=%s, data.shape=%s, dt=%s",
            dataset,
            getattr(data, 'shape', None),
            dt,
        )

        n_time, n_z, n_y, n_x, n_comp = data.shape

        # Apply filtering (optional - can be None)
        if config.filter_type is not None:
            filtered = self._fft_compute.apply_filter(data, config.filter_type)
        else:
            filtered = data
            log.debug("Skipping temporal filtering (filter_type=None)")
        
        # Apply windowing (optional - can be None)
        if config.window_function is not None:
            windowed = self._fft_compute.apply_window(filtered, config.window_function)
        else:
            windowed = filtered
            log.debug("Skipping temporal windowing (window_function=None)")

        window_size = min(config.spatial_window, n_x)
        step = config.spatial_step

        window_starts = list(range(0, max(n_x - window_size + 1, 1), step))
        if not window_starts:
            window_starts = [0]

        x_centers = np.array(
            [start + (window_size - 1) / 2.0 for start in window_starts],
            dtype=float,
        )

        n_windows = len(window_starts)
        n_freq = n_time // 2 + 1

        # Use scipy.fft if available (faster than numpy.fft)
        if _USE_SCIPY_FFT:
            freqs = scipy_fft.rfftfreq(n_time, d=dt)
            log.debug("Using scipy.fft.rfft (optimized)")
        else:
            freqs = np.fft.rfftfreq(n_time, d=dt)
            log.debug("Using numpy.fft.rfft (fallback)")

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
        
        if _USE_SCIPY_FFT:
            # scipy.fft.rfft is typically 2-3x faster than numpy.fft.rfft
            full_spectrum = scipy_fft.rfft(windowed, axis=0)
        else:
            full_spectrum = np.fft.rfft(windowed, axis=0)
        
        t_fft_end = time.time()
        log.info("FFT completed in %.3fs (shape: %s → %s)", 
                 t_fft_end - t_fft_start, windowed.shape, full_spectrum.shape)

        # Now extract windows from pre-computed FFT (much faster!)
        # Now extract windows from pre-computed FFT (much faster!)
        # 🚀 OPTIMIZATION 2: Vectorize or parallelize window processing
        
        # Decide on processing strategy
        use_parallel = _USE_JOBLIB and n_windows > 100  # Only parallelize for many windows
        use_vectorized = (config.average_mode == "none" and 
                          not config.enable_circular_components and 
                          not config.store_component_maps and
                          not use_parallel)
        
        if use_vectorized:
            # Fast path: no averaging, no extra components - fully vectorize!
            log.debug("Using vectorized window processing (average_mode='none')")
            t_process_start = time.time()
            
            # Pre-compute power for all windows at once
            for win_idx, start in enumerate(window_starts):
                end = min(start + window_size, n_x)
                window_slice = slice(start, end)
                spectrum = full_spectrum[:, :, :, window_slice, :]
                
                # Extract components
                mx_fft = spectrum[..., 0]
                my_fft = spectrum[..., 1]
                power_components = np.abs(mx_fft) ** 2 * component_weights[0]
                power_components += np.abs(my_fft) ** 2 * component_weights[1]
                
                if n_comp > 2:
                    mz_fft = spectrum[..., 2]
                    power_components += np.abs(mz_fft) ** 2 * component_weights[2]
                
                # Fast aggregation: average_mode="none" → just take [0,0,:].mean
                # power_components shape: (freq, z, y, window_x)
                power_map[:, win_idx] = power_components[:, 0, 0, :].mean(axis=1)
            
            t_process_end = time.time()
            log.info("Vectorized processing: %.3fs for %d windows", 
                      t_process_end - t_process_start, n_windows)
                      
        elif use_parallel:
            # Parallel path: use joblib to process windows in parallel
            log.info("Using parallel processing with joblib (%d windows, %d CPUs)", 
                     n_windows, -1)  # -1 = use all CPUs
            t_process_start = time.time()
            
            def process_window(win_idx: int, start: int):
                """Process single window - can run in parallel."""
                end = min(start + window_size, n_x)
                window_slice = slice(start, end)
                spectrum = full_spectrum[:, :, :, window_slice, :]
                
                mx_fft = spectrum[..., 0]
                my_fft = spectrum[..., 1]
                power_components = np.abs(mx_fft) ** 2 * component_weights[0]
                power_components += np.abs(my_fft) ** 2 * component_weights[1]
                
                if n_comp > 2:
                    mz_fft = spectrum[..., 2]
                    power_components += np.abs(mz_fft) ** 2 * component_weights[2]
                
                aggregated = _aggregate_spatial(
                    power_components,
                    config.average_mode,
                    config.edge_taper_power,
                )
                
                results = {'power': aggregated}
                
                if transverse_map is not None:
                    results['transverse'] = _aggregate_spatial(
                        np.abs(mx_fft) ** 2 + np.abs(my_fft) ** 2,
                        config.average_mode,
                        config.edge_taper_power,
                    )
                
                if longitudinal_map is not None and n_comp > 2:
                    results['longitudinal'] = _aggregate_spatial(
                        np.abs(mz_fft) ** 2,
                        config.average_mode,
                        config.edge_taper_power,
                    )
                
                if config.enable_circular_components:
                    m_plus = (mx_fft + 1j * my_fft) / np.sqrt(2.0)
                    m_minus = (mx_fft - 1j * my_fft) / np.sqrt(2.0)
                    results['power_plus'] = _aggregate_spatial(
                        np.abs(m_plus) ** 2,
                        config.average_mode,
                        config.edge_taper_power,
                    )
                    results['power_minus'] = _aggregate_spatial(
                        np.abs(m_minus) ** 2,
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
            
            for win_idx, start in enumerate(window_starts):
                end = min(start + window_size, n_x)
                window_slice = slice(start, end)
                
                # Extract window from pre-computed FFT spectrum
                spectrum = full_spectrum[:, :, :, window_slice, :]

                mx_fft = spectrum[..., 0]
                my_fft = spectrum[..., 1]
                power_components = np.abs(mx_fft) ** 2 * component_weights[0]
                power_components += np.abs(my_fft) ** 2 * component_weights[1]

                if n_comp > 2:
                    mz_fft = spectrum[..., 2]
                    power_components += np.abs(mz_fft) ** 2 * component_weights[2]
                    if longitudinal_map is not None:
                        longitudinal_map[:, win_idx] = _aggregate_spatial(
                            np.abs(mz_fft) ** 2,
                            config.average_mode,
                            config.edge_taper_power,
                        )

                # Accumulate lightweight complex-spectrum summary per component (mean across z,y,window)
                if complex_accum is not None:
                    for comp_idx in range(n_comp):
                        comp_spec = spectrum[..., comp_idx]
                        # mean over z, y, and the window dimension (note: block may be smaller than window_size at edges)
                        comp_mean = comp_spec.mean(axis=(1, 2, 3))
                        complex_accum[:, comp_idx] += comp_mean

                aggregated = _aggregate_spatial(
                    power_components,
                    config.average_mode,
                    config.edge_taper_power,
                )

                power_map[:, win_idx] = aggregated

                if transverse_map is not None:
                    transverse_map[:, win_idx] = _aggregate_spatial(
                        np.abs(mx_fft) ** 2 + np.abs(my_fft) ** 2,
                        config.average_mode,
                        config.edge_taper_power,
                    )

                if config.enable_circular_components and power_plus is not None and power_minus is not None:
                    m_plus = (mx_fft + 1j * my_fft) / np.sqrt(2.0)
                    m_minus = (mx_fft - 1j * my_fft) / np.sqrt(2.0)
                    power_plus[:, win_idx] = _aggregate_spatial(
                        np.abs(m_plus) ** 2,
                        config.average_mode,
                        config.edge_taper_power,
                    )
                    power_minus[:, win_idx] = _aggregate_spatial(
                        np.abs(m_minus) ** 2,
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

        result = TransmissionResult(
            frequencies=freqs,
            x_positions=x_centers,
            transmission=transmission,
            power_map=power_map,
            reference_power=reference_values,
            config=config,
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
