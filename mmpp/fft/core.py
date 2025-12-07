"""
FFT Core Module

Main FFT class providing unified interface for FFT analysis.
"""

from typing import Any, Optional

import numpy as np

# Import from our own modules
from .compute_fft import FFTCompute, FFTComputeResult
from .plot import FFTPlotter
from .transmission.interface import FFTTransmissionInterface
from ..cli.logging_config import get_mmpp_logger

# Get logger for FFT core
log = get_mmpp_logger("mmpp.fft")

# Import mode visualization capabilities
try:
    from .modes import FFTModeInterface, FMRModeAnalyzer, ModeVisualizationConfig
    from .modes.interface import FFTModeInterfaceNew  # New refactored interface

    MODES_AVAILABLE = True
except ImportError:
    MODES_AVAILABLE = False

try:
    from .dispersion import (
        SpinWaveAnalyzer,
        DispersionConfig,
        FFTDispersionInterface,
        find_peaks_1d,
    )

    DISPERSION_AVAILABLE = True
except ImportError:
    DISPERSION_AVAILABLE = False
    find_peaks_1d = None  # type: ignore

# Optional matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.colors import to_rgba
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Axes = Any  # type: ignore
    to_rgba = None  # type: ignore


def generate_pastel_colors(n: int):
    """Generate n pastel colors using the Accent colormap.
    
    Parameters
    ----------
    n : int
        Number of colors to generate
        
    Returns
    -------
    list
        List of RGBA color tuples
        
    Examples
    --------
    >>> colors = generate_pastel_colors(3)  # For mx, my, mz
    >>> ax.plot(x, y, color=colors[0])
    """
    if not MATPLOTLIB_AVAILABLE:
        # Fallback colors if matplotlib not available
        return [(0.4, 0.6, 0.8, 1.0)] * n
    
    colors = plt.cm.Accent(np.linspace(0, 1, max(n, 3)))
    return [to_rgba(c) for c in colors[:n]]



class SpectrumHelper:
    """Callable helper wrapper for FFT.spectrum() with rich display.
    
    When accessed as property (job.fft.spectrum), displays helpful usage info.
    When called (job.fft.spectrum(...)), delegates to actual spectrum method.
    """
    
    def __init__(self, fft_instance):
        self._fft = fft_instance
        self._spectrum_method = fft_instance._spectrum_impl
    
    def __call__(self, *args, **kwargs):
        """Delegate to actual spectrum method."""
        return self._spectrum_method(*args, **kwargs)
    
    def __repr__(self):
        return self._rich_display()
    
    def _repr_html_(self):
        """For Jupyter notebook display."""
        return None  # Let rich handle it
    
    def _rich_display(self) -> str:
        """Generate rich help display for spectrum method."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich.syntax import Syntax
            from io import StringIO
            
            capture = StringIO()
            console = Console(file=capture, force_terminal=True, width=100)
            
            # Title
            title = Text()
            title.append("📊 FFT Spectrum Analysis\n", style="bold blue")
            title.append(f"Path: {self._fft.job_result.path}", style="dim")
            
            console.print(Panel(title, border_style="blue"))
            
            # Parameters table
            params_table = Table(show_header=True, header_style="bold green")
            params_table.add_column("Parameter", style="yellow")
            params_table.add_column("Description", style="white")
            params_table.add_column("Default", style="cyan")
            
            params = [
                ("dset", "Dataset name", "'m'"),
                ("z_layer", "Z-layer index", "-1"),
                ("tmin/tmax", "Time range (indices)", "None"),
                ("fmin/fmax", "Frequency filter (Hz)", "None"),
                ("find_peaks", "Peak detection config", "None"),
                ("force", "Force recalculation", "False"),
                ("save", "Save to zarr", "False"),
            ]
            for p, d, v in params:
                params_table.add_row(p, d, v)
            
            console.print(params_table)
            console.print("")
            
            # Examples
            example_code = '''# Basic spectrum
result = job[0].fft.spectrum()
freqs, spec = result  # Tuple unpacking

# With time slicing using slice notation
result = job[0].m[:200,...,1].fft.spectrum()

# Or with tmin/tmax parameters
result = job[0].fft.spectrum(tmin=0, tmax=200)

# Fluent plotting API
job[0].fft.spectrum(find_peaks={'min_prominence': 0.1}).plot_spectrum(
    freq_unit="GHz",
    log_scale=True,
    dpi=150
)

# Access properties
result.power       # |FFT|²
result.magnitude   # |FFT|
result.frequencies
result.peaks_info  # If find_peaks was used'''
            
            syntax = Syntax(example_code, "python", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title="[bold magenta]Usage Examples[/bold magenta]", border_style="magenta"))
            
            return capture.getvalue()
        except ImportError:
            return "FFT.spectrum(...) - Call with parameters to compute FFT spectrum. Use help(job[0].fft.spectrum) for details."


class SpectrumResult:
    """Result of FFT spectrum computation with fluent plotting API.
    
    Provides both tuple-like access (frequencies, spectrum) and 
    method chaining for plotting.
    
    Examples
    --------
    >>> # Fluent API
    >>> job[0].m[:100,...].fft.spectrum().plot_spectrum(log_scale=True)
    >>> 
    >>> # Tuple unpacking still works
    >>> freqs, spec = job[0].fft.spectrum()
    """
    
    def __init__(
        self,
        frequencies: np.ndarray,
        spectrum: np.ndarray,
        peaks_info: Optional[dict] = None,
        component_label: Optional[str] = None,
    ):
        self.frequencies = frequencies
        self.spectrum = spectrum
        self.peaks_info = peaks_info
        self.component_label = component_label
        self._single_component = False  # Set to True if user selected specific component
    
    @property
    def power(self) -> np.ndarray:
        """Power spectrum |FFT|²"""
        return np.abs(self.spectrum) ** 2
    
    @property
    def magnitude(self) -> np.ndarray:
        """Magnitude spectrum |FFT|"""
        return np.abs(self.spectrum)
    
    def __iter__(self):
        """Enable tuple unpacking: freqs, spec = result"""
        yield self.frequencies
        yield self.spectrum
        if self.peaks_info is not None:
            yield self.peaks_info
    
    def __getitem__(self, index: int):
        """Enable indexed access: result[0] for frequencies, result[1] for spectrum"""
        items = [self.frequencies, self.spectrum]
        if self.peaks_info is not None:
            items.append(self.peaks_info)
        return items[index]
    
    def __len__(self):
        """Length for tuple-like behavior"""
        return 3 if self.peaks_info is not None else 2
    
    def __repr__(self):
        label_info = f", label='{self.component_label}'" if self.component_label else ""
        return (
            f"SpectrumResult(frequencies={len(self.frequencies)}, "
            f"spectrum_shape={self.spectrum.shape}, "
            f"peaks={len(self.peaks_info['indices']) if self.peaks_info else 'None'}"
            f"{label_info})"
        )
    
    def plot_spectrum(
        self,
        ax: Optional[Any] = None,
        freq_unit: str = "GHz",
        log_scale: bool = True,
        normalize: bool = False,
        show_peaks: bool = True,
        title: Optional[str] = None,
        dpi: Optional[int] = None,
        **kwargs,
    ):
        """Plot power spectrum.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on (creates new figure if None)
        freq_unit : str
            Frequency unit: "Hz", "kHz", "MHz", "GHz", "THz"
        log_scale : bool
            Use logarithmic Y-scale (default: True)
        normalize : bool
            Normalize to maximum value (default: False)
        show_peaks : bool
            Show detected peaks if available (default: True)
        title : str, optional
            Custom plot title
        dpi : int, optional
            Resolution in dots per inch (default: None)
        **kwargs
            Additional matplotlib plot arguments
            
        Returns
        -------
        Tuple[Figure, Axes]
            Matplotlib figure and axes
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")
        
        # Frequency scaling
        freq_scales = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
        freq_scale = freq_scales.get(freq_unit, 1e9)
        freqs_display = self.frequencies / freq_scale
        
        # Power spectrum
        power = self.power
        if normalize:
            power = power / np.max(power)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
        else:
            fig = ax.figure
            if dpi is not None:
                fig.set_dpi(dpi)
        
        # Handle multidimensional power spectrum
        # If we have (Freqs, ..., 3), we want to plot 3 lines (mx, my, mz)
        # BUT: if user selected specific component via slicing, only plot that component
        if power.ndim > 1 and power.shape[-1] == 3 and not getattr(self, '_single_component', False):
            # Multiple components available and NOT single-component selection
            # Check if last dimension is 3 (components)
            # Average over any spatial dimensions in between (Freqs, [Space], 3) -> (Freqs, 3)
            if power.ndim > 2:
                spatial_axes = tuple(range(1, power.ndim - 1))
                power_to_plot = np.mean(power, axis=spatial_axes)
            else:
                power_to_plot = power

            # Plot with component labels and pastel colors
            component_labels = [r"$m_x$", r"$m_y$", r"$m_z$"]
            colors = generate_pastel_colors(3)
            for i in range(3):
                ax.plot(freqs_display, power_to_plot[:, i], 
                       label=component_labels[i], color=colors[i], **kwargs)
            ax.legend()
        elif power.ndim > 1:
            # Treat as multiple spatial points or other dimensions
            # Just flatten the rest or average?
            # Default behavior: average everything else to get 1D
            spatial_axes = tuple(range(1, power.ndim))
            power_to_plot = np.mean(power, axis=spatial_axes)
            
            # Use component label if available and not overridden
            if "label" not in kwargs and self.component_label:
                kwargs["label"] = self.component_label
            elif "label" not in kwargs:
                kwargs["label"] = "Average Power"
            
            ax.plot(freqs_display, power_to_plot, **kwargs)
            ax.legend()
        else:
            # 1D case
            # Use component label if available and not overridden
            if "label" not in kwargs and self.component_label:
                kwargs["label"] = self.component_label
            
            ax.plot(freqs_display, power, **kwargs)
            if "label" in kwargs:
                ax.legend()

        ax.set_xlabel(f"Frequency ({freq_unit})")
        
        # Professional Y-axis label with exponent
        if not normalize and not log_scale:
            # Get max value to determine exponent
            if "power_to_plot" in dir():
                max_val = np.max(power_to_plot)
            else:
                max_val = np.max(power)
            
            if max_val > 0:
                exponent = int(np.floor(np.log10(max_val)))
                if abs(exponent) >= 2:
                    # Scale data and update label
                    scale_factor = 10 ** exponent
                    for line in ax.get_lines():
                        ydata = line.get_ydata()
                        line.set_ydata(ydata / scale_factor)
                    ax.set_ylabel(f"Power (×10$^{{{exponent}}}$ arb. u.)")
                    ax.relim()
                    ax.autoscale_view()
                else:
                    ax.set_ylabel("Power (arb. u.)")
            else:
                ax.set_ylabel("Power (arb. u.)")
        elif normalize:
            ax.set_ylabel("Power (normalized)")
        else:
            ax.set_ylabel("Power (arb. u.)")
        
        if log_scale:
            ax.set_yscale("log")
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title("FFT Power Spectrum")
        
        # Professional peak markers
        if show_peaks and self.peaks_info is not None and len(self.peaks_info["indices"]) > 0:
            peak_freqs = self.peaks_info["frequencies"] / freq_scale
            peak_powers = self.peaks_info["amplitudes"] ** 2
            if normalize:
                peak_powers = peak_powers / np.max(self.power)
            
            # Scale peak powers if we scaled the data
            if not normalize and not log_scale and 'scale_factor' in dir():
                peak_powers = peak_powers / scale_factor
            
            # Plot peaks as subtle markers (not scatter)
            ax.plot(peak_freqs, peak_powers, 'o', 
                   color='#E74C3C', markersize=6, markeredgecolor='white', 
                   markeredgewidth=1.5, zorder=5, label='Peaks')
            
            # Find top 3 peaks for annotation
            sorted_indices = np.argsort(peak_powers)[::-1]
            n_annotate = min(3, len(peak_freqs))
            
            for i in range(n_annotate):
                idx = sorted_indices[i]
                freq = peak_freqs[idx]
                power_val = peak_powers[idx]
                
                # Only add vertical line for the highest peak
                if i == 0:
                    # Draw line from y=0 to the peak value
                    ax.vlines(x=freq, ymin=0, ymax=power_val, 
                             color='#E74C3C', linestyle=':', alpha=0.6, linewidth=1.2)
                
                # Format frequency label
                freq_text = f"{freq:.2f}" if 0.01 < freq < 100 else f"{freq:.2e}"
                
                # Add annotation with arrow
                ax.annotate(
                    f"{freq_text} {freq_unit}",
                    xy=(freq, power_val),
                    xytext=(8, 8 + i * 12),  # Offset to avoid overlap
                    textcoords='offset points',
                    fontsize=9,
                    color='#2C3E50',
                    fontweight='medium',
                    arrowprops=dict(
                        arrowstyle='-',
                        color='#E74C3C',
                        alpha=0.6,
                        lw=0.8
                    ) if i == 0 else None,
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='#E74C3C',
                        alpha=0.9,
                        linewidth=0.8
                    ) if i == 0 else None,
                    zorder=10
                )
        
        # Style improvements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(frameon=True, fancybox=True, shadow=False, 
                 framealpha=0.9, edgecolor='lightgray', fontsize=9)
        
        fig.tight_layout()
        
        return fig, ax


class FFT:
    """
    Main FFT analysis class providing numpy.fft-like interface.

    This class aggregates FFT computation and plotting capabilities
    for MMPP job results.
    """

    # Feature availability flags
    MODES_AVAILABLE = MODES_AVAILABLE
    DISPERSION_AVAILABLE = DISPERSION_AVAILABLE

    def __init__(self, job_result, mmpp_instance: Optional[Any] = None):
        """
        Initialize FFT analyzer for a job result.

        Parameters:
        -----------
        job_result : ZarrJobResult
            Job result to analyze
        mmpp_instance : MMPP, optional
            Reference to parent MMPP instance
        """
        self.job_result = job_result
        self.mmpp = mmpp_instance

        # Initialize compute engine with debug mode from parent MMPP if available
        debug_mode = getattr(mmpp_instance, "debug", False) if mmpp_instance else False
        self._compute = FFTCompute(debug=debug_mode)

        # Initialize plotter (lazy loaded)
        self._plotter = None

        # Transmission interface (lazy)
        self._transmission_interface = None

        # Cache for FFT results
        self._cache = {}

    @property
    def plotter(self) -> FFTPlotter:
        """Get plotter instance (lazy initialization)."""
        if self._plotter is None:
            self._plotter = FFTPlotter([self.job_result], self.mmpp)
        return self._plotter

    @property
    def transmission(self) -> FFTTransmissionInterface:
        """Transmission analysis helper."""

        if self._transmission_interface is None:
            self._transmission_interface = FFTTransmissionInterface(
                self,
                self._compute,
                self.job_result,
            )
        return self._transmission_interface

    def _format_slice_identifier(self, slice_info: Optional[Any]) -> str:
        """Create a deterministic identifier for slice_info for caching/saving."""
        if slice_info is None:
            return "slice=None"

        def format_item(item: Any) -> str:
            if isinstance(item, slice):
                return f"{item.start}:{item.stop}:{item.step}"
            if item is Ellipsis:
                return "..."
            if isinstance(item, tuple):
                return "(" + ",".join(format_item(sub) for sub in item) + ")"
            if isinstance(item, (int, np.integer)):
                return str(int(item))
            return repr(item)

        slice_tuple = slice_info if isinstance(slice_info, tuple) else (slice_info,)
        formatted = ",".join(format_item(part) for part in slice_tuple)
        return f"slice={formatted}"

    def _get_cache_key(
        self,
        dataset_name: str,
        z_layer: int,
        method: int,
        slice_identifier: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate cache key for FFT results."""
        # Normalize z_layer for consistent cache keys
        # For cache purposes, we use the raw z_layer value since the actual normalization
        # happens in calculate_fft_data and we want consistent caching behavior
        key_parts = [dataset_name, str(z_layer), str(method)]
        if slice_identifier:
            key_parts.append(slice_identifier)
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)

    def _compute_fft(
        self,
        dataset_name: Optional[str] = None,
        z_layer: int = -1,
        method: int = 1,
        use_cache: bool = True,
        save: bool = False,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
        **kwargs,
    ) -> FFTComputeResult:
        """
        Compute FFT with caching and optional saving.

        Parameters:
        -----------
        dataset_name : str, optional
            Dataset name (default: auto-select largest m dataset)
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        use_cache : bool, optional
            Use memory cache (default: True)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        slice_info : Any, optional
            Optional slicing (e.g., [:1000, ..., 0]) applied before FFT and used in caching
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        FFTComputeResult
            FFT computation result
        """
        # Auto-select largest m dataset if none specified
        if dataset_name is None:
            dataset_name = self.job_result.get_largest_m_dataset()

        if not isinstance(dataset_name, str):
            dataset_name = str(dataset_name)

        slice_identifier = self._format_slice_identifier(slice_info)
        cache_key = self._get_cache_key(
            dataset_name, z_layer, method, slice_identifier=slice_identifier, **kwargs
        )

        # Check memory cache only if not forcing and not saving
        if use_cache and not force and not save and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self._compute.calculate_fft_data(
                self.job_result.path,
                dataset_name,
                z_layer,
                method,
                save=save,
                force=force,
                save_dataset_name=save_dataset_name,
                slice_info=slice_info,
                slice_identifier=(
                    None if slice_identifier == "slice=None" else slice_identifier
                ),
                **kwargs,
            )
        except OSError as e:
            if "directory not empty" in str(e).lower():
                print(
                    "Warning: FFT directory already exists and is not empty. Use force=True to overwrite."
                )
            raise

        # Cache result only if not forcing
        if use_cache and not force:
            self._cache[cache_key] = result

        return result

    @property
    def spectrum(self) -> SpectrumHelper:
        """Get spectrum helper (shows help when accessed, callable to compute).
        
        When accessed directly in notebook, shows usage help.
        When called, computes FFT spectrum.
        
        Examples
        --------
        >>> job[0].fft.spectrum  # Shows help
        >>> job[0].fft.spectrum()  # Computes spectrum
        """
        return SpectrumHelper(self)

    def _spectrum_impl(
        self,
        dset: str = "m",
        z_layer: int = -1,
        method: int = 1,
        save: bool = False,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
        tmin: Optional[int] = None,
        tmax: Optional[int] = None,
        find_peaks: Optional[dict] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        **kwargs,
    ):
        """
        Compute FFT spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        slice_info : Any, optional
            Slicing info for data selection (e.g., (slice(0,100), ...))
            If tmin/tmax are provided, they take precedence for time slicing.
        tmin : int, optional
            Start time index for slicing (default: None, use all)
            Creates slice_info=(slice(tmin, tmax), ...) internally.
        tmax : int, optional
            End time index for slicing (default: None, use all)
        find_peaks : dict, optional
            If provided, detect peaks in the spectrum. Dictionary with parameters:
            - 'min_prominence': float, minimum peak prominence (default: 0.0)
            Example: find_peaks={'min_prominence': 0.1}
        fmin : float, optional
            Minimum frequency in Hz to include in result (default: None, include all)
        fmax : float, optional
            Maximum frequency in Hz to include in result (default: None, include all)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] or tuple[np.ndarray, np.ndarray, dict]
            If find_peaks is None: (frequencies, complex FFT spectrum)
            If find_peaks is provided: (frequencies, complex FFT spectrum, peaks_info)
            where peaks_info is a dict with 'indices', 'frequencies', and 'amplitudes'
        
        Examples:
        ---------
        >>> # Basic usage
        >>> freqs, spec = job[0].fft.spectrum()
        >>> 
        >>> # With time slicing
        >>> freqs, spec = job[0].fft.spectrum(tmin=0, tmax=1000)
        >>> 
        >>> # Equivalent using slice notation
        >>> freqs, spec = job[0].m[:1000,...].fft.spectrum()
        """
        # If tmin or tmax specified, create slice_info for time dimension
        if tmin is not None or tmax is not None:
            slice_info = (slice(tmin, tmax), ...)
        
        result = self._compute_fft(
            dset,
            z_layer,
            method,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            slice_info=slice_info,
            **kwargs,
        )

        frequencies = result.frequencies
        spectrum = result.spectrum

        # Apply frequency range filtering
        if fmin is not None or fmax is not None:
            freq_mask = np.ones(len(frequencies), dtype=bool)
            if fmin is not None:
                freq_mask &= frequencies >= fmin
            if fmax is not None:
                freq_mask &= frequencies <= fmax
            frequencies = frequencies[freq_mask]
            spectrum = spectrum[freq_mask] if spectrum.ndim == 1 else spectrum[freq_mask, ...]

        peaks_info = None
        if find_peaks is not None:
            # Find peaks in spectrum
            if find_peaks_1d is None:
                log.warning(
                    "Peak finding requested but dispersion module not available. Install required dependencies."
                )
            else:
                # Extract parameters
                min_prominence = find_peaks.get("min_prominence", 0.0)

                # Use absolute value of spectrum for peak detection
                spectrum_abs = np.abs(spectrum)

                # For peak finding, handle multidimensional spectrum by spatial averaging
                if spectrum_abs.ndim > 1:
                    # Spatial axes are 1, 2, ...
                    spatial_axes = tuple(range(1, spectrum_abs.ndim))
                    spectrum_for_peaks = np.mean(spectrum_abs, axis=spatial_axes)
                    log.debug(f"Averaged spectrum over axes {spatial_axes} for peak finding")
                else:
                    spectrum_for_peaks = spectrum_abs

                # Find peaks
                peak_indices = find_peaks_1d(spectrum_for_peaks, min_prominence=min_prominence)

                # Create peaks info dictionary
                # Use amplitudes from the spectrum used for peak finding
                peaks_info = {
                    "indices": peak_indices,
                    "frequencies": frequencies[peak_indices],
                    "amplitudes": spectrum_for_peaks[peak_indices],
                }

                log.info(f"Found {len(peak_indices)} peaks with prominence >= {min_prominence}")

        # Try to determine if user selected a specific component
        component_selected = False
        component_label = None
        if slice_info is not None:
             # Look for component selection in the last dimension
             # Typically slice_info is a tuple of slices/indices
             if isinstance(slice_info, tuple) and len(slice_info) > 0:
                 last_idx = slice_info[-1]
                 if isinstance(last_idx, int):
                     # User explicitly selected a specific component
                     component_selected = True
                     if last_idx == 0:
                         component_label = r"$m_x$"
                     elif last_idx == 1:
                         component_label = r"$m_y$"
                     elif last_idx == 2:
                         component_label = r"$m_z$"
        
        # Mark the spectrum result to indicate single-component selection
        result = SpectrumResult(frequencies, spectrum, peaks_info, component_label=component_label)
        result._single_component = component_selected
        return result

    def frequencies(
        self,
        dset: str = "m",
        z_layer: int = -1,
        method: int = 1,
        save: bool = False,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Get frequency array for FFT.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Frequency array
        """
        # Try to compute frequencies efficiently without loading data
        try:
            log.debug(f"Attempting fast frequency calculation for dataset '{dset}'")
            frequencies = self._compute_frequencies_fast(
                dset, slice_info=slice_info, **kwargs
            )
            log.debug(
                f"✓ Fast frequency calculation successful (shape: {frequencies.shape})"
            )
            return frequencies
        except Exception as e:
            # Fallback to full FFT computation
            log.debug(
                f"Fast frequency calculation failed: {e}, falling back to full FFT"
            )
            result = self._compute_fft(
                dset,
                z_layer,
                method,
                save=save,
                force=force,
                save_dataset_name=save_dataset_name,
                slice_info=slice_info,
                **kwargs,
            )
            return result.frequencies

    def _compute_frequencies_fast(
        self,
        dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute frequency array without loading full dataset.
        Only reads metadata (dt and shape) from zarr.
        """
        # Auto-select largest m dataset if none specified
        if dataset_name is None:
            dataset_name = self.job_result.get_largest_m_dataset()

        if not isinstance(dataset_name, str):
            dataset_name = str(dataset_name)

        def _extract_dt(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                if hasattr(value, "item"):
                    value = value.item()
                return float(value)
            except Exception:
                return None

        # Get dataset metadata without materializing the data
        try:
            data_set = None
            zarr_group = None

            # Prefer existing job_result handle to avoid reopening the store
            try:
                zarr_group = self.job_result.z
                data_set = self.job_result.get_dset(dataset_name)
            except Exception as access_error:
                log.debug(
                    f"Fast frequency metadata lookup via job_result failed for {dataset_name}: {access_error}"
                )

            if data_set is None:
                import zarr

                zarr_group = zarr.open(self.job_result.path, mode="r")
                if dataset_name not in zarr_group:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found in {self.job_result.path}"
                    )
                data_set = zarr_group[dataset_name]

            data_shape = getattr(data_set, "shape", None)
            if not data_shape:
                raise ValueError(
                    f"Could not determine shape for dataset {dataset_name}"
                )

            n_timesteps = data_shape[0]
            n_timesteps = self._apply_time_slice_length(n_timesteps, slice_info)
            if n_timesteps <= 0:
                raise ValueError(f"Dataset {dataset_name} has no time dimension")

            # Respect optional tmax override if provided
            tmax = kwargs.get("tmax")
            if tmax is not None:
                try:
                    tmax_int = int(tmax)
                    if tmax_int > 0:
                        n_timesteps = min(n_timesteps, tmax_int)
                except (TypeError, ValueError):
                    log.debug(f"Ignoring invalid tmax value: {tmax}")

            # Get dt from dataset's smart .dt property (handles t_sampl, time arrays, etc.)
            # Falls back to manual attrs check if wrapper not available
            dt = None
            if hasattr(data_set, 'dt'):
                try:
                    dt = data_set.dt
                    log.debug(f"Using dt from data_set.dt property: {dt}")
                except AttributeError:
                    pass  # Fall through to manual checks
            
            if dt is None and hasattr(data_set, "attrs"):
                dataset_attrs = getattr(data_set, "attrs", {})
                for key in ("t_sampl", "dt"):
                    dt = _extract_dt(dataset_attrs.get(key))
                    if dt:
                        break

            if dt is None and zarr_group is not None and hasattr(zarr_group, "attrs"):
                for key in ("t_sampl", "dt"):
                    dt = _extract_dt(zarr_group.attrs.get(key))
                    if dt:
                        break

            if dt is None:
                dt = 1e-12
                log.warning(
                    f"t_sampl not found in metadata for {dataset_name}, using default dt={dt}"
                )

            # Determine FFT length (same logic as in compute_fft)
            fft_length = n_timesteps

            zero_padding = kwargs.get("zero_padding", self._compute.config.zero_padding)
            nfft = kwargs.get("nfft", self._compute.config.nfft)

            if nfft is not None:
                fft_length = nfft
            elif zero_padding:
                next_power_two = 1 << (n_timesteps - 1).bit_length()
                if next_power_two > n_timesteps:
                    fft_length = next_power_two

            # Compute frequencies
            frequencies = np.fft.rfftfreq(fft_length, dt)
            return frequencies

        except Exception as e:
            raise RuntimeError(
                f"Failed to compute frequencies from metadata: {e}"
            ) from e

    def _apply_time_slice_length(
        self, n_timesteps: int, slice_info: Optional[Any]
    ) -> int:
        """Estimate resulting time length after applying slice info."""
        if slice_info is None or n_timesteps <= 0:
            return n_timesteps

        slice_tuple = slice_info if isinstance(slice_info, tuple) else (slice_info,)
        if not slice_tuple:
            return n_timesteps

        first = slice_tuple[0]
        if first is Ellipsis or first is None:
            return n_timesteps

        if isinstance(first, slice):
            start, stop, step = first.indices(n_timesteps)
            if step == 0:
                return n_timesteps
            length = max(0, (stop - start + (step - 1)) // step)
            return max(0, min(length, n_timesteps))

        if isinstance(first, (int, np.integer)):
            return 1

        return n_timesteps

    def power(
        self,
        dset: str = "m",
        z_layer: int = -1,
        method: int = 1,
        save: bool = False,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute power spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Power spectrum (|FFT|^2)
        """
        result = self.spectrum(
            dset,
            z_layer,
            method,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            slice_info=slice_info,
            **kwargs,
        )
        spectrum = result[1]  # Extract spectrum from tuple
        return np.abs(spectrum) ** 2

    def phase(
        self,
        dset: str = "m",
        z_layer: int = -1,
        method: int = 1,
        slice_info: Optional[Any] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute phase spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Phase spectrum
        """
        result = self.spectrum(dset, z_layer, method, slice_info=slice_info, **kwargs)
        spectrum = result[1]  # Extract spectrum from tuple
        return np.angle(spectrum)

    def magnitude(
        self,
        dset: str = "m",
        z_layer: int = -1,
        method: int = 1,
        slice_info: Optional[Any] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute magnitude spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        slice_info : Any, optional
            Optional slicing applied before FFT
        \\*\\*kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Magnitude spectrum (\\|FFT\\|)
        """
        result = self.spectrum(dset, z_layer, method, slice_info=slice_info, **kwargs)
        spectrum = result[1]  # Extract spectrum from tuple
        return np.abs(spectrum)

    def plot_spectrum(
        self,
        dset: str = "m",
        ax: Optional[Any] = None,
        method: int = 1,
        z_layer: int = -1,
        log_scale: bool = True,
        normalize: bool = False,
        save: bool = True,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        **kwargs,
    ) -> tuple[Any, Any]:
        """
        Plot power spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure.
        method : int, optional
            FFT method (default: 1)
        z_layer : int, optional
            Z-layer (default: -1)
        log_scale : bool, optional
            Use logarithmic scale (default: True)
        normalize : bool, optional
            Normalize spectrum (default: False)
        save : bool, optional
            Save FFT result to zarr file (default: True)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional plotting options

        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        return self.plotter.power_spectrum(
            dataset_name=dset,
            ax=ax,
            method=method,
            z_layer=z_layer,
            log_scale=log_scale,
            normalize=normalize,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            **kwargs,
        )

    def clear_cache(self):
        """Clear FFT computation cache."""
        self._cache.clear()

    def __repr__(self) -> str:
        """Rich documentation display for FFT interface."""
        try:
            return self._rich_fft_display()
        except Exception:
            return self._basic_fft_display()

    def _rich_fft_display(self) -> str:
        """Create rich documentation display with panels and proper styling."""
        try:
            import io

            from rich.columns import Columns
            from rich.console import Console
            from rich.panel import Panel
            from rich.syntax import Syntax
            from rich.table import Table
            from rich.text import Text

            console = Console(file=io.StringIO(), width=120, force_terminal=True)

            # Get basic info
            path = self.job_result.path
            cache_size = len(self._cache)
            has_modes = MODES_AVAILABLE

            # Summary panel content
            summary_text = Text()
            summary_text.append("🔬 MMPP FFT Analysis Interface\n", style="bold cyan")
            summary_text.append(f"📁 Job Path: {path}\n", style="dim")
            summary_text.append(f"💾 Cache Entries: {cache_size}\n", style="dim")
            summary_text.append(
                f"🎯 Mode Analysis: {'✓ Available' if has_modes else '✗ Unavailable'}\n",
                style="green" if has_modes else "red",
            )

            # Core methods panel content
            core_methods_text = Text()
            core_methods_text.append("🔧 Core FFT Methods:\n", style="bold yellow")
            methods = [
                ("spectrum()", "Get (freqs, complex FFT) tuple"),
                ("frequencies()", "Get frequency array"),
                ("power()", "Get power spectrum |FFT|²"),
                ("magnitude()", "Get magnitude |FFT|"),
                ("phase()", "Get phase spectrum"),
                ("spectrum(tmin=,tmax=)", "FFT with time slicing"),
                ("plot_spectrum(ax=)", "Plot on existing axes"),
                ("transmission", "Transmission analysis interface"),
                ("clear_cache()", "Clear computation cache"),
            ]

            for method, desc in methods:
                core_methods_text.append("  • ", style="dim")
                core_methods_text.append(method, style="code")
                core_methods_text.append(f" - {desc}\n", style="dim")

            # Plotting methods panel content
            plotting_methods_text = Text()
            plotting_methods_text.append("📈 Plotting Toolkit:\n", style="bold magenta")
            plotting_methods = [
                ("plot_spectrum(log_scale=True)", "Quick-look power spectrum"),
                ("plotter.power_spectrum(normalize=True)", "Overlay multiple results"),
                (
                    "plotter.power_spectrum(save_path='fft.png')",
                    "Export publication figure",
                ),
                ("plot_modes(frequency=..., z_layer=-1)", "Static mode grid"),
                ("modes.save_modes_animation()", "Animated mode evolution"),
            ]

            for method, desc in plotting_methods:
                plotting_methods_text.append("  • ", style="dim")
                plotting_methods_text.append(method, style="code")
                plotting_methods_text.append(f" - {desc}\n", style="dim")

            # Mode methods panel content (if available)
            if has_modes:
                mode_methods_text = Text()
                mode_methods_text.append(
                    "🌊 Mode Analysis Methods:\n", style="bold blue"
                )
                mode_methods = [
                    ("modes", "Access mode interface"),
                    ("[index]", "Index-based mode access"),
                    ("plot_modes(frequency)", "Plot modes at frequency"),
                    ("interactive_spectrum()", "Interactive spectrum+modes"),
                ]

                for method, desc in mode_methods:
                    mode_methods_text.append("  • ", style="dim")
                    mode_methods_text.append(method, style="code")
                    mode_methods_text.append(f" - {desc}\n", style="dim")
            else:
                mode_methods_text = Text()
                mode_methods_text.append(
                    "🌊 Mode Analysis: Not Available\n", style="bold red"
                )
                mode_methods_text.append(
                    "Install mode visualization dependencies to enable", style="dim"
                )

            # Batch operations panel content
            batch_methods_text = Text()
            batch_methods_text.append("📦 Batch Operations (job[:].fft):\n", style="bold green")
            batch_methods = [
                ("spectrum.compute_all()", "Compute all spectra in batch"),
                ("transmission.compute_all()", "Compute all transmissions"),
                ("batch[i].plot()", "Plot individual spectrum"),
                ("batch.plot_heatmap(param)", "2D heatmap vs parameter"),
                ("batch.save(path)", "Save batch to zarr/pickle"),
            ]

            for method, desc in batch_methods:
                batch_methods_text.append("  • ", style="dim")
                batch_methods_text.append(method, style="code")
                batch_methods_text.append(f" - {desc}\n", style="dim")

            # Parameters table
            params_table = Table(show_header=False, box=None, padding=(0, 1))
            params_table.add_column("Parameter", style="bold yellow")
            params_table.add_column("Description", style="white")
            params_table.add_column("Values", style="cyan")

            params = [
                (
                    "dset",
                    "Dataset name",
                    "Auto-selected or explicit: 'm', 'm_x11', 'm_y11'",
                ),
                ("z_layer", "Z-layer index", "-1 (top), 0 (bottom), 1, 2, ..."),
                ("method", "FFT method", "1 (default), 2"),
                ("ax", "Existing matplotlib axes", "None (create new) or Axes"),
                ("fmin/fmax", "Frequency range filter", "None or float (Hz)"),
                ("tmin/tmax", "Time index range", "None or int (start:stop)"),
                ("slice_info", "Advanced slicing", "tuple e.g. (slice(0,100), ..., 1)"),
                ("find_peaks", "Peak detection", "None or {'min_prominence': 0.1}"),
                ("save", "Save to zarr", "True/False"),
                ("force", "Force recalculation", "True/False"),
                ("zero_padding", "Pad to power-of-two", "True/False (default: True)"),
                ("nfft", "Manual FFT length", "int or None (auto)"),
                ("filter_type", "Preprocessing filter", "remove_mean, savgol_smooth, high_pass, band_pass"),
                ("window", "Window function", "hann (default), flattop, nuttall, blackman"),
                ("dpi", "Plot resolution", "int (e.g., 100, 300)"),
                ("log_scale", "Logarithmic Y-scale", "True (default) / False"),
                ("normalize", "Normalize power", "True/False (default: False)"),
                ("show_peaks", "Show peak markers", "True (default) / False"),
                ("freq_unit", "Frequency display unit", "Hz, kHz, MHz, GHz (default), THz"),
            ]

            for param, desc, values in params:
                params_table.add_row(param, desc, values)

            example_code = '''# Basic FFT operations
freqs, spectrum = job[0].fft.spectrum()
power = job[0].fft.power()

# Time slicing - two equivalent ways:
freqs, spec = job[0].fft.spectrum(tmin=0, tmax=1000)
freqs, spec = job[0].m[:1000,...].fft.spectrum()  # slice notation

# Frequency range filtering
freqs, spec = job[0].fft.spectrum(fmin=5e9, fmax=25e9)

# Advanced filters
job[0].fft.spectrum(filter_type="savgol_smooth")

# Batch spectrum
batch = job[:].fft.spectrum.compute_all(
    extract_parameters=["B0"],
)
batch[0].plot()                    # Plot single spectrum
batch.plot_heatmap("B0")           # 2D heatmap vs B0'''

            syntax = Syntax(
                example_code, "python", theme="monokai", background_color="default"
            )

            # Build panels
            with console.capture() as capture:
                # Main summary panel
                console.print(
                    Panel.fit(
                        summary_text,
                        title="[bold cyan]MMPP FFT Interface[/bold cyan]",
                        border_style="cyan",
                    )
                )
                console.print("")

                # Method panels side by side
                console.print(
                    Columns(
                        [
                            Panel.fit(
                                core_methods_text,
                                title="[bold yellow]Core Methods[/bold yellow]",
                                border_style="yellow",
                            ),
                            Panel.fit(
                                plotting_methods_text,
                                title="[bold magenta]Plotting[/bold magenta]",
                                border_style="magenta",
                            ),
                            Panel.fit(
                                mode_methods_text,
                                title="[bold blue]Mode Methods[/bold blue]",
                                border_style="blue" if has_modes else "red",
                            ),
                            Panel.fit(
                                batch_methods_text,
                                title="[bold green]Batch Operations[/bold green]",
                                border_style="green",
                            ),
                        ]
                    )
                )
                console.print("")

                # Parameters panel
                console.print(
                    Panel.fit(
                        params_table,
                        title="[bold green]Common Parameters[/bold green]",
                        border_style="green",
                    )
                )
                console.print("")

                # Examples panel
                console.print(
                    Panel.fit(
                        syntax,
                        title="[bold magenta]Usage Examples[/bold magenta]",
                        border_style="magenta",
                    )
                )

            return capture.get()

        except Exception:
            # Fallback to basic text display if rich fails
            return self._basic_fft_display_enhanced()

    def _basic_fft_display(self) -> str:
        """Fallback basic display if rich display fails."""
        return f"FFT(path='{self.job_result.path}', cache_entries={len(self._cache)})"

    def _basic_fft_display_enhanced(self) -> str:
        """Enhanced fallback display with more details if rich display fails."""
        path = self.job_result.path
        cache_size = len(self._cache)
        has_modes = MODES_AVAILABLE

        output = []
        output.append("=" * 70)
        output.append("🔬 MMPP FFT Analysis Interface")
        output.append("=" * 70)
        output.append(f"📁 Job Path: {path}")
        output.append(f"💾 Cache Entries: {cache_size}")
        output.append(
            f"🎯 Mode Analysis: {'✓ Available' if has_modes else '✗ Unavailable'}"
        )
        output.append("")

        # Core FFT Methods
        output.append("🔧 CORE FFT METHODS:")
        output.append("─" * 50)
        methods = [
            (
                "spectrum()",
                "Get (freqs, complex FFT spectrum)",
                "freqs, spectrum = job[0].fft.spectrum('m', z_layer=-1)",
            ),
            ("frequencies()", "Get frequency array", "job[0].fft.frequencies()"),
            ("power()", "Get power spectrum |FFT|²", "job[0].fft.power()"),
            ("magnitude()", "Get magnitude |FFT|", "job[0].fft.magnitude()"),
            ("phase()", "Get phase spectrum", "job[0].fft.phase()"),
            (
                "plot_spectrum()",
                "Plot power spectrum",
                "fig, ax = job[0].fft.plot_spectrum()",
            ),
            ("clear_cache()", "Clear computation cache", "job[0].fft.clear_cache()"),
        ]

        for method, desc, example in methods:
            output.append(f"  • {method:<15} {desc}")
            output.append(f"    └─ {example}")

        output.append("")

        # Plotting toolkit
        output.append("📈 PLOTTING TOOLKIT:")
        output.append("─" * 50)
        plot_methods = [
            (
                "plot_spectrum(log_scale=True)",
                "Quick-look spectrum",
                "job[0].fft.plot_spectrum(log_scale=True)",
            ),
            (
                "plotter.power_spectrum(normalize=True)",
                "Overlay multiple jobs",
                "job[0].fft.plotter.power_spectrum(normalize=True)",
            ),
            (
                "plotter.power_spectrum(save_path='fft.png')",
                "Export PNG/ publication",
                "job[0].fft.plotter.power_spectrum(save_path='fft.png')",
            ),
            (
                "plot_modes(frequency, z_layer)",
                "Static mode panels",
                "job[0].fft.plot_modes(frequency=10.4, z_layer=-1)",
            ),
            (
                "modes.save_modes_animation()",
                "Animated mode evolution",
                "job[0].fft.modes.save_modes_animation(frequency=10.4)",
            ),
        ]

        for method, desc, example in plot_methods:
            output.append(f"  • {method:<40} {desc}")
            output.append(f"    └─ {example}")

        output.append("")

        # Mode Analysis (if available)
        if has_modes:
            output.append("🌊 MODE ANALYSIS METHODS:")
            output.append("─" * 50)
            mode_methods = [
                (
                    "modes",
                    "Access mode interface",
                    "job[0].fft.modes.interactive_spectrum()",
                ),
                (
                    "[index]",
                    "Index-based mode access",
                    "job[0].fft[0][200].plot_modes()",
                ),
                (
                    "plot_modes()",
                    "Plot modes at frequency",
                    "job[0].fft.plot_modes(frequency=1.5)",
                ),
                (
                    "interactive_spectrum()",
                    "Interactive spectrum+modes",
                    "job[0].fft.interactive_spectrum()",
                ),
            ]

            for method, desc, example in mode_methods:
                output.append(f"  • {method:<20} {desc}")
                output.append(f"    └─ {example}")
        else:
            output.append("🌊 MODE ANALYSIS: Not Available")
            output.append("   Install mode visualization dependencies to enable")

        output.append("")

        # Common Parameters
        output.append("⚙️  COMMON PARAMETERS:")
        output.append("─" * 50)
        params = [
            ("dset", "Dataset name", "'m', 'm_x11', 'm_y11'"),
            ("z_layer", "Z-layer index", "-1 (top), 0 (bottom), 1, 2, ..."),
            ("method", "FFT method", "1 (default), 2"),
            ("save", "Save to zarr", "True/False"),
            ("force", "Force recalculation", "True/False"),
            ("zero_padding", "Pad to power-of-two length", "True/False"),
            ("nfft", "Manual FFT length", "int or None"),
        ]

        for param, desc, values in params:
            output.append(f"  • {param:<12} {desc:<20} {values}")

        output.append("")

        # Quick Examples
        output.append("🚀 QUICK START EXAMPLES:")
        output.append("─" * 50)
        examples = [
            "# Basic FFT operations",
            "power = job[0].fft.power('m')",
            "freqs = job[0].fft.frequencies()",
            "freqs_fft, spectrum = job[0].fft.spectrum(save=True, force=True)",
            "",
            "# Plotting",
            "fig, ax = job[0].fft.plot_spectrum(log_scale=True)",
            "job[0].fft.plotter.power_spectrum(save_path='fft_publication.png')",
            "",
            "# Mode analysis (if available)",
            "job[0].fft.modes.interactive_spectrum()",
            "job[0].fft[0][200].plot_modes()  # Elegant syntax",
            "job[0].fft.plot_modes(frequency=1.5)",
            "",
            "# Advanced usage",
            "job[0].fft.plotter.power_spectrum(normalize=True)",
            "job[0].fft.modes.save_modes_animation(frequency=10.4, save_path='mode.gif')",
            "help(job[0].fft.spectrum)  # Detailed documentation",
        ]

        for example in examples:
            output.append(f"  {example}")

        output.append("")
        output.append("=" * 70)
        output.append("📖 For detailed docs: help(job[0].fft.spectrum)")
        output.append("🔧 Clear cache: job[0].fft.clear_cache()")
        output.append("=" * 70)

        return "\n".join(output)

    @property
    def modes(self) -> "FFTModeInterfaceNew":
        """
        Get mode visualization interface.
        
        Supports slice propagation for component selection.

        Returns:
        --------
        FFTModeInterfaceNew
            Interface for mode operations with slice support

        Examples:
        ---------
        >>> job[0].fft.modes.interactive_spectrum()
        >>> job[0].m[:200,...,1].fft.modes.interactive_spectrum(dpi=150)  # my only
        >>> job[0].fft.modes.plot_modes(frequency=1.5)
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        if not hasattr(self, "_mode_interface_new"):
            self._mode_interface_new = FFTModeInterfaceNew(0, self)
        return self._mode_interface_new

    @property
    def dispersion(self) -> "FFTDispersionInterface":
        """
        Get spin-wave dispersion analysis interface.

        Returns:
        --------
        FFTDispersionInterface
            Interface for dispersion operations

        Examples:
        ---------
        >>> job[0].fft.dispersion.plot_dispersion()
        >>> job[0].fft.dispersion.compute_1d(axis="x")
        >>> job[0].m_layer.fft.dispersion.plot_branch()
        """
        if not DISPERSION_AVAILABLE:
            raise ImportError(
                "Dispersion analysis not available. Check dispersion module import."
            )

        if not hasattr(self, "_dispersion_interface"):
            self._dispersion_interface = FFTDispersionInterface(self)
        return self._dispersion_interface

    def __getitem__(self, index: int) -> "FFTModeInterface":
        """
        Get FFT result by index for mode operations.

        Parameters:
        -----------
        index : int
            FFT result index (usually 0 for latest)

        Returns:
        --------
        FFTModeInterface
            Interface for mode operations at specific FFT result

        Examples:
        ---------
        >>> job[0].fft[0].interactive_spectrum()
        >>> job[0].fft[0][200].plot_modes()
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        return FFTModeInterface(index, self)

    def plot_modes(
        self, frequency: float, dset: str = "m", z_layer: int = 0, **kwargs
    ) -> tuple[Any, Any]:
        """
        Plot FMR modes at specific frequency.

        Parameters:
        -----------
        frequency : float
            Frequency in GHz
        dset : str
            Dataset name
        z_layer : int
            Z-layer index
        **kwargs
            Additional arguments for mode plotting

        Returns:
        --------
        Tuple[Figure, np.ndarray]
            Matplotlib figure and axes
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        # Create temporary mode analyzer
        debug_mode = getattr(self.mmpp, "debug", False) if self.mmpp else False
        log_level = getattr(self.mmpp, "log_level", None) if self.mmpp else None
        analyzer = FMRModeAnalyzer(
            self.job_result.path,
            dataset_name=dset,
            debug=debug_mode,
            log_level=log_level,
        )
        return analyzer.plot_modes(frequency=frequency, z_layer=z_layer, **kwargs)

    def interactive_spectrum(self, dset: str = "m", **kwargs) -> Any:
        """
        Create interactive spectrum plot with mode visualization.

        Parameters:
        -----------
        dset : str
            Dataset name
        **kwargs
            Additional arguments for interactive plotting

        Returns:
        --------
        Figure
            Interactive matplotlib figure
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        # Create temporary mode analyzer
        debug_mode = getattr(self.mmpp, "debug", False) if self.mmpp else False
        log_level = getattr(self.mmpp, "log_level", None) if self.mmpp else None
        analyzer = FMRModeAnalyzer(
            self.job_result.path,
            dataset_name=dset,
            debug=debug_mode,
            log_level=log_level,
        )
        return analyzer.interactive_spectrum(**kwargs)
