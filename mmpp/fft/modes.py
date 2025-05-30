"""
FMR Mode Visualization Module

Professional implementation for visualizing FMR modes with interactive spectrum.
Provides both programmatic and interactive interfaces for mode analysis.
"""

from typing import Optional, Dict, List, Union, Any, Tuple, Callable
import numpy as np
from dataclasses import dataclass
import warnings
from datetime import datetime

# Import shared logging configuration
from ..logging_config import setup_mmpp_logging, get_mmpp_logger

# Get logger for FMR modes
log = get_mmpp_logger("mmpp.fft.modes")

# Import dependencies with error handling
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    log.warning("Matplotlib not available - mode visualization disabled")

try:
    import scipy.signal
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    log.warning("SciPy not available - peak detection features limited")

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    log.error("Zarr not available - mode analysis disabled")


@dataclass
class ModeVisualizationConfig:
    """Configuration for mode visualization."""
    # Figure settings
    figsize: Tuple[float, float] = (16, 10)
    dpi: int = 100
    
    # Spectrum settings
    spectrum_log_scale: bool = False
    spectrum_normalize: bool = True
    peak_threshold: float = 0.1
    peak_min_distance: int = 5
    
    # Mode visualization settings
    show_magnitude: bool = True
    show_phase: bool = True
    show_combined: bool = True
    colormap_magnitude: str = "inferno"
    colormap_phase: str = "hsv"
    interpolation: str = "nearest"
    
    # Frequency range for analysis
    f_min: float = 0.0
    f_max: float = 40.0
    
    # Layout settings
    spectrum_width_ratio: float = 0.4
    modes_width_ratio: float = 0.6
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.f_min >= self.f_max:
            raise ValueError(f"f_min ({self.f_min}) must be less than f_max ({self.f_max})")
        
        if self.peak_threshold < 0 or self.peak_threshold > 1:
            raise ValueError(f"peak_threshold must be between 0 and 1, got {self.peak_threshold}")
            
        if self.peak_min_distance < 1:
            raise ValueError(f"peak_min_distance must be >= 1, got {self.peak_min_distance}")
            
        if self.spectrum_width_ratio <= 0 or self.modes_width_ratio <= 0:
            raise ValueError("Width ratios must be positive")
            
        if self.dpi < 50 or self.dpi > 500:
            log.warning(f"Unusual DPI value: {self.dpi}")
            
        # Validate colormaps
        try:
            import matplotlib.pyplot as plt
            plt.get_cmap(self.colormap_magnitude)
            plt.get_cmap(self.colormap_phase)
        except Exception as e:
            log.warning(f"Colormap validation failed: {e}")


@dataclass 
class Peak:
    """Peak data structure."""
    idx: int
    freq: float
    amplitude: float


class FMRModeData:
    """Container for FMR mode data at a specific frequency."""
    
    def __init__(self, frequency: float, mode_array: np.ndarray, 
                 extent: Optional[Tuple[float, float, float, float]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize FMR mode data.
        
        Parameters:
        -----------
        frequency : float
            Frequency in GHz
        mode_array : np.ndarray
            Complex mode array with shape (ny, nx, 3) for spatial x-y and magnetization components
        extent : tuple, optional
            Spatial extent [x_min, x_max, y_min, y_max] in nm
        metadata : dict, optional
            Additional metadata
        """
        self.frequency = frequency
        self.mode_array = mode_array
        self.extent = extent or (0, mode_array.shape[1], 0, mode_array.shape[0])
        self.metadata = metadata or {}
        
        # Validate input
        if not isinstance(mode_array, np.ndarray):
            raise TypeError("mode_array must be numpy array")
        if mode_array.ndim != 3 or mode_array.shape[2] != 3:
            raise ValueError("mode_array must have shape (ny, nx, 3)")
    
    @property
    def magnitude(self) -> np.ndarray:
        """Get magnitude of mode for each component."""
        return np.abs(self.mode_array)
    
    @property 
    def phase(self) -> np.ndarray:
        """Get phase of mode for each component."""
        return np.angle(self.mode_array)
    
    @property
    def total_magnitude(self) -> np.ndarray:
        """Get total magnitude across all components."""
        return np.sqrt(np.sum(self.magnitude**2, axis=2))
    
    def get_component(self, component: Union[int, str]) -> np.ndarray:
        """
        Get specific magnetization component.
        
        Parameters:
        -----------
        component : int or str
            Component index (0, 1, 2) or name ('x', 'y', 'z', 'mx', 'my', 'mz')
            
        Returns:
        --------
        np.ndarray
            Complex mode array for specified component
        """
        component_map = {'x': 0, 'y': 1, 'z': 2, 'mx': 0, 'my': 1, 'mz': 2}
        
        if isinstance(component, str):
            if component.lower() not in component_map:
                raise ValueError(f"Unknown component '{component}'. Use 'x', 'y', 'z' or 0, 1, 2")
            component = component_map[component.lower()]
        
        if not 0 <= component <= 2:
            raise ValueError(f"Component index must be 0, 1, or 2, got {component}")
            
        return self.mode_array[:, :, component]


class FMRModeAnalyzer:
    """
    Professional FMR mode analyzer with interactive visualization.
    
    Provides both programmatic access to mode data and interactive
    spectrum visualization for frequency selection.
    """
    
    def __init__(self, zarr_path: str, dataset_name: str = "m", 
                 config: Optional[ModeVisualizationConfig] = None,
                 debug: bool = False):
        """
        Initialize FMR mode analyzer.
        
        Parameters:
        -----------
        zarr_path : str
            Path to zarr file containing mode data
        dataset_name : str, optional
            Base dataset name (default: "m") 
        config : ModeVisualizationConfig, optional
            Visualization configuration
        debug : bool, optional
            Enable debug logging
        """
        if not ZARR_AVAILABLE:
            raise ImportError("Zarr is required for mode analysis")
            
        self.zarr_path = zarr_path
        self.dataset_name = dataset_name
        self.config = config or ModeVisualizationConfig()
        
        # Set up logging
        setup_mmpp_logging(debug=debug, logger_name="mmpp.fft.modes")
        if debug:
            log.debug("FMR mode analyzer debug logging enabled")
        
        # Load data
        self._load_data()
        
        # Interactive state
        self._current_frequency = None
        self._interactive_fig = None
        self._frequency_line = None
        self._mode_axes = None
        
        # Mode data cache (LRU cache with max 10 entries)
        self._mode_cache = {}
        self._cache_order = []
        self._max_cache_size = 10
        
    @property
    def modes_available(self) -> bool:
        """Check if mode data is available."""
        return (self.modes_path is not None and 
                self.freqs_path is not None and 
                self.spectrum_path is not None)
    
    def _get_zarr_paths(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Unified path resolution for zarr datasets.
        
        Returns:
        --------
        Tuple[str, str, str]
            (modes_path, freqs_path, spectrum_path) or None if not found
        """
        # Possible base paths for modes/frequencies - consistent order
        base_paths = [f"modes/{self.dataset_name}", f"tmodes/{self.dataset_name}"]
        
        modes_path = None
        freqs_path = None
        
        # Find first existing base path
        for base_path in base_paths:
            if f"{base_path}/arr" in self.zarr_file and f"{base_path}/freqs" in self.zarr_file:
                modes_path = f"{base_path}/arr"
                freqs_path = f"{base_path}/freqs"
                break
        
        # If not found together, try separately (for backward compatibility)
        if modes_path is None:
            for base_path in base_paths:
                if f"{base_path}/arr" in self.zarr_file:
                    modes_path = f"{base_path}/arr"
                    break
                    
        if freqs_path is None:
            for base_path in base_paths:
                if f"{base_path}/freqs" in self.zarr_file:
                    freqs_path = f"{base_path}/freqs"
                    break
        
        # Find spectrum
        spectrum_path = None
        spectrum_candidates = [f"fft/{self.dataset_name}/spec", f"fft/{self.dataset_name}/sum"]
        for path in spectrum_candidates:
            if path in self.zarr_file:
                spectrum_path = path
                break
        
        return modes_path, freqs_path, spectrum_path
        
    def _load_data(self) -> None:
        """Load mode and spectrum data from zarr file."""
        try:
            self.zarr_file = zarr.open(self.zarr_path, mode='r')
            log.info(f"Opened zarr file: {self.zarr_path}")
        except Exception as e:
            log.error(f"Failed to open zarr file {self.zarr_path}: {e}")
            raise
            
        # Use unified path resolution
        self.modes_path, self.freqs_path, self.spectrum_path = self._get_zarr_paths()
        
        if not self.modes_path:
            log.debug(f"No mode data found for dataset '{self.dataset_name}'. "
                     "Modes will need to be computed.")
        
        if not self.freqs_path:
            log.debug(f"No frequency data found for dataset '{self.dataset_name}'. "
                     "Frequencies will be computed with modes.")
                           
        if not self.spectrum_path:
            log.warning(f"No spectrum data found for dataset '{self.dataset_name}'. "
                       f"Expected paths: fft/{self.dataset_name}/spec or fft/{self.dataset_name}/sum")
            
        # Load frequency array if available
        if self.freqs_path:
            self.frequencies = np.array(self.zarr_file[self.freqs_path])
            log.info(f"Loaded frequencies: {len(self.frequencies)} points, "
                    f"range {self.frequencies[0]:.3f} - {self.frequencies[-1]:.3f} GHz")
        else:
            self.frequencies = None
            log.debug("No frequency data loaded - will be computed with modes")
        
        # Load spectrum if available
        if self.spectrum_path:
            self.spectrum = np.array(self.zarr_file[self.spectrum_path])
            if self.spectrum.ndim > 1:
                # Take first component if multi-component
                self.spectrum = self.spectrum[:, 0] if self.spectrum.shape[1] == 3 else np.sum(self.spectrum, axis=tuple(range(1, self.spectrum.ndim)))
            log.info(f"Loaded spectrum data: shape {self.spectrum.shape}")
        else:
            self.spectrum = None
            
        # Get spatial information
        self._get_spatial_info()
        
    def _get_spatial_info(self) -> None:
        """Extract spatial information from zarr metadata."""
        # Try to get spatial resolution from attributes
        self.dx = 1.0  # Default spatial resolution in nm
        self.dy = 1.0
        
        # Look for spatial attributes in various locations
        attrs_to_check = [
            self.zarr_file.attrs,
            self.zarr_file[self.dataset_name].attrs if self.dataset_name in self.zarr_file else {},
        ]
        
        for attrs in attrs_to_check:
            if 'dx' in attrs:
                self.dx = float(attrs['dx']) * 1e9  # Convert to nm
            if 'dy' in attrs:
                self.dy = float(attrs['dy']) * 1e9  # Convert to nm
                
        log.debug(f"Spatial resolution: dx={self.dx:.3f} nm, dy={self.dy:.3f} nm")
        
    def _detect_peaks(self, spectrum: np.ndarray, frequencies: np.ndarray) -> List[Peak]:
        """
        Detect peaks in spectrum.
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Power spectrum data
        frequencies : np.ndarray
            Frequency array in GHz
            
        Returns:
        --------
        List[Peak]
            List of detected peaks
        """
        if not SCIPY_AVAILABLE:
            log.warning("SciPy not available, using simple peak detection")
            # Simple peak detection without scipy
            peaks = []
            for i in range(1, len(spectrum) - 1):
                if (spectrum[i] > spectrum[i-1] and 
                    spectrum[i] > spectrum[i+1] and
                    spectrum[i] > self.config.peak_threshold * np.max(spectrum)):
                    peaks.append(Peak(
                        idx=i,
                        freq=frequencies[i],
                        amplitude=spectrum[i]
                    ))
            return peaks
        
        try:
            # Normalize spectrum for peak detection
            norm_spectrum = spectrum / np.max(spectrum)
            
            # Find peaks using scipy
            peak_indices, properties = find_peaks(
                norm_spectrum,
                height=self.config.peak_threshold,
                distance=self.config.peak_min_distance
            )
            
            # Create peak objects
            peaks = []
            for idx in peak_indices:
                peaks.append(Peak(
                    idx=int(idx),
                    freq=frequencies[idx],
                    amplitude=spectrum[idx]
                ))
            
            # Sort by amplitude (descending)
            peaks.sort(key=lambda p: p.amplitude, reverse=True)
            
            log.debug(f"Detected {len(peaks)} peaks")
            return peaks
            
        except Exception as e:
            log.error(f"Peak detection failed: {e}")
            return []
        
    def get_mode(self, frequency: float, z_layer: int = 0) -> FMRModeData:
        """
        Get mode data at specified frequency.
        
        Parameters:
        -----------
        frequency : float
            Frequency in GHz
        z_layer : int, optional
            Z-layer index (default: 0)
            
        Returns:
        --------
        FMRModeData
            Mode data at specified frequency
            
        Raises:
        -------
        ValueError
            If frequency or z_layer is out of range
        RuntimeError
            If mode data is not available
        """
        if self.frequencies is None:
            raise RuntimeError("No frequency data available. Run compute_modes() first.")
            
        if self.modes_path is None:
            raise RuntimeError("No mode data available. Run compute_modes() first.")
        
        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))
        actual_freq = self.frequencies[freq_idx]
        
        if abs(actual_freq - frequency) > 0.1:
            log.warning(f"Requested frequency {frequency:.3f} GHz not found, "
                       f"using closest: {actual_freq:.3f} GHz")
        
        # Validate z_layer bounds
        mode_shape = self.zarr_file[self.modes_path].shape
        if z_layer >= mode_shape[1]:
            raise ValueError(f"z_layer {z_layer} out of range. Available layers: 0-{mode_shape[1]-1}")
        
        # Load mode data for this frequency with bounds checking
        try:
            mode_data = self.zarr_file[self.modes_path][freq_idx, z_layer, :, :, :]
        except IndexError as e:
            raise ValueError(f"Invalid indices: freq_idx={freq_idx}, z_layer={z_layer}. {e}")
        
        # Create spatial extent
        ny, nx = mode_data.shape[:2]
        extent = (0, nx * self.dx, 0, ny * self.dy)
        
        # Metadata
        metadata = {
            'frequency_index': freq_idx,
            'requested_frequency': frequency,
            'actual_frequency': actual_freq,
            'z_layer': z_layer,
            'spatial_resolution': (self.dx, self.dy),
            'mode_shape': mode_shape
        }
        
        # Update cache
        self._update_cache(frequency, z_layer, FMRModeData(actual_freq, mode_data, extent, metadata))
        
        return FMRModeData(actual_freq, mode_data, extent, metadata)
    
    def _update_cache(self, frequency: float, z_layer: int, mode_data: FMRModeData) -> None:
        """Update mode data cache."""
        key = (frequency, z_layer)
        if key in self._mode_cache:
            # Move to end to indicate recent use
            self._cache_order.remove(key)
        elif len(self._mode_cache) >= self._max_cache_size:
            # Remove least recently used item
            oldest_key = self._cache_order.pop(0)
            del self._mode_cache[oldest_key]
        
        # Update cache with new mode data
        self._mode_cache[key] = mode_data
        self._cache_order.append(key)
    
    def find_peaks(self, threshold: Optional[float] = None, 
                   min_distance: Optional[int] = None,
                   component: int = 0) -> List[Peak]:
        """
        Find peaks in the spectrum.
        
        Parameters:
        -----------
        threshold : float, optional
            Peak detection threshold (default: from config)
        min_distance : int, optional
            Minimum distance between peaks (default: from config)
        component : int, optional
            Spectrum component to analyze (default: 0)
            
        Returns:
        --------
        List[Peak]
            List of detected peaks
        """
        if self.spectrum is None:
            log.warning("No spectrum data available for peak detection")
            return []
            
        threshold = threshold or self.config.peak_threshold
        min_distance = min_distance or self.config.peak_min_distance
        
        # Normalize spectrum for peak detection
        spectrum = self.spectrum.copy()
        if self.config.spectrum_normalize:
            spectrum = spectrum / np.max(spectrum)
        
        # Filter frequency range
        freq_mask = (self.frequencies >= self.config.f_min) & (self.frequencies <= self.config.f_max)
        freqs_filtered = self.frequencies[freq_mask]
        spectrum_filtered = spectrum[freq_mask]
        
        # Detect peaks
        peaks = self._detect_peaks(spectrum_filtered, freqs_filtered)
        
        # Convert to Peak objects with proper index mapping
        peaks_converted = []
        for peak in peaks:
            # Safely map back to original index
            try:
                orig_indices = np.where(freq_mask)[0]
                if peak.idx < len(orig_indices):
                    orig_idx = orig_indices[peak.idx]
                    peaks_converted.append(Peak(orig_idx, peak.freq, peak.amplitude))
                else:
                    log.warning(f"Peak index {peak.idx} out of range for filtered array")
            except IndexError as e:
                log.warning(f"Index mapping error for peak {peak.idx}: {e}")
                continue
            
        log.info(f"Found {len(peaks_converted)} peaks in frequency range "
                f"{self.config.f_min}-{self.config.f_max} GHz")
        
        return peaks_converted
        
    def plot_modes(self, frequency: float, z_layer: int = 0,
                   components: Optional[List[Union[int, str]]] = None,
                   save_path: Optional[str] = None) -> Tuple[Figure, np.ndarray]:
        """
        Plot mode visualization for a specific frequency.
        
        Parameters:
        -----------
        frequency : float
            Frequency in GHz
        z_layer : int, optional
            Z-layer index (default: 0)
        components : list, optional
            List of components to plot (default: ['x', 'y', 'z'])
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        Tuple[Figure, np.ndarray]
            Matplotlib figure and axes array
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for plotting")
            
        components = components or ['x', 'y', 'z']
        mode_data = self.get_mode(frequency, z_layer)
        
        # Create figure with subplots
        n_components = len(components)
        n_rows = 3 if self.config.show_magnitude and self.config.show_phase and self.config.show_combined else 2
        
        fig, axes = plt.subplots(n_rows, n_components, 
                               figsize=(4*n_components, 3*n_rows),
                               dpi=self.config.dpi)
        
        if n_components == 1:
            axes = axes.reshape(-1, 1)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
            
        # Plot each component
        for i, comp in enumerate(components):
            comp_data = mode_data.get_component(comp)
            magnitude = np.abs(comp_data)
            phase = np.angle(comp_data)
            
            row = 0
            
            # Magnitude plot
            if self.config.show_magnitude:
                im1 = axes[row, i].imshow(magnitude, 
                                        cmap=self.config.colormap_magnitude,
                                        extent=mode_data.extent,
                                        aspect='equal',
                                        interpolation=self.config.interpolation,
                                        origin='lower')
                axes[row, i].set_title(f'|m_{comp}| @ {frequency:.3f} GHz')
                axes[row, i].set_xlabel('x (nm)')
                if i == 0:
                    axes[row, i].set_ylabel('y (nm)')
                plt.colorbar(im1, ax=axes[row, i], shrink=0.8)
                row += 1
            
            # Phase plot  
            if self.config.show_phase:
                im2 = axes[row, i].imshow(phase,
                                        cmap=self.config.colormap_phase,
                                        extent=mode_data.extent,
                                        aspect='equal',
                                        interpolation=self.config.interpolation,
                                        vmin=-np.pi, vmax=np.pi,
                                        origin='lower')
                axes[row, i].set_title(f'arg(m_{comp}) @ {frequency:.3f} GHz')
                axes[row, i].set_xlabel('x (nm)')
                if i == 0:
                    axes[row, i].set_ylabel('y (nm)')
                plt.colorbar(im2, ax=axes[row, i], shrink=0.8)
                row += 1
            
            # Combined plot (phase with magnitude as alpha)
            if self.config.show_combined:
                alpha = magnitude / np.max(magnitude)
                im3 = axes[row, i].imshow(phase,
                                        cmap=self.config.colormap_phase,
                                        extent=mode_data.extent,
                                        aspect='equal',
                                        interpolation=self.config.interpolation,
                                        alpha=alpha,
                                        vmin=-np.pi, vmax=np.pi,
                                        origin='lower')
                axes[row, i].set_title(f'm_{comp} (phase+mag) @ {frequency:.3f} GHz')
                axes[row, i].set_xlabel('x (nm)')
                if i == 0:
                    axes[row, i].set_ylabel('y (nm)')
                plt.colorbar(im3, ax=axes[row, i], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            log.info(f"Saved mode plot to {save_path}")
            
        return fig, axes
        
    def interactive_spectrum(self, components: Optional[List[Union[int, str]]] = None,
                           z_layer: int = 0) -> Figure:
        """
        Create interactive spectrum plot with mode visualization.
        
        Click on spectrum to select frequency and visualize corresponding mode.
        Right-click to snap to nearest peak.
        
        Parameters:
        -----------
        components : list, optional
            List of components to plot (default: ['x', 'y', 'z'])
        z_layer : int, optional
            Z-layer index (default: 0)
            
        Returns:
        --------
        Figure
            Interactive matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for interactive plotting")
            
        if self.spectrum is None:
            raise ValueError("No spectrum data available for interactive mode")
            
        components = components or ['x', 'y', 'z']
        n_components = len(components)
        
        # Validate number of components for layout
        if n_components > 5:
            raise ValueError(f"Too many components ({n_components}). Maximum supported: 5")
        
        # Create figure with custom layout
        try:
            import matplotlib
            matplotlib.use("module://ipympl", force=True)
        except Exception as e:
            log.error(f"Failed to enable interactive mode: {e}")
            raise ImportError("Matplotlib widget backend is required for interactive plotting")
        self._interactive_fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Create grid layout: spectrum on left, modes on right
        # Use dynamic number of rows (3 for all visualization types)
        n_vis_types = sum([self.config.show_magnitude, self.config.show_phase, self.config.show_combined])
        if n_vis_types == 0:
            raise ValueError("At least one visualization type must be enabled")
            
        gs = gridspec.GridSpec(n_vis_types, n_components + 1, 
                             width_ratios=[self.config.spectrum_width_ratio] + 
                                        [self.config.modes_width_ratio/n_components] * n_components,
                             height_ratios=[1] * n_vis_types)
        
        # Spectrum plot spans all rows in first column
        ax_spectrum = self._interactive_fig.add_subplot(gs[:, 0])
        
        # Mode plots in remaining columns - dynamic based on enabled visualizations
        self._mode_axes = np.array([[self._interactive_fig.add_subplot(gs[row, col+1]) 
                                   for col in range(n_components)] 
                                  for row in range(n_vis_types)])
        
        # Plot spectrum
        freq_mask = (self.frequencies >= self.config.f_min) & (self.frequencies <= self.config.f_max)
        freqs_plot = self.frequencies[freq_mask]
        spectrum_plot = self.spectrum[freq_mask]
        
        if self.config.spectrum_normalize:
            spectrum_plot = spectrum_plot / np.max(spectrum_plot)
            
        if self.config.spectrum_log_scale:
            spectrum_plot = np.log10(spectrum_plot + 1e-10)
            ax_spectrum.set_ylabel('log₁₀(Power Spectrum)')
        else:
            ax_spectrum.set_ylabel('Power Spectrum')
            
        ax_spectrum.plot(freqs_plot, spectrum_plot, 'b-', linewidth=1.5)
        ax_spectrum.set_xlabel('Frequency (GHz)')
        ax_spectrum.set_title('FMR Spectrum (Click to select frequency)')
        ax_spectrum.grid(True, alpha=0.3)
        
        # Find and mark peaks
        peaks = self.find_peaks()
        for peak in peaks:
            if self.config.f_min <= peak.freq <= self.config.f_max:
                y_val = spectrum_plot[np.argmin(np.abs(freqs_plot - peak.freq))]
                ax_spectrum.plot(peak.freq, y_val, 'ro', markersize=4)
                ax_spectrum.text(peak.freq, y_val + 0.05 * np.max(spectrum_plot),
                               f'{peak.freq:.2f}', rotation=90, ha='center', va='bottom',
                               fontsize=8)
        
        # Initial frequency line
        init_freq = peaks[0].freq if peaks else freqs_plot[len(freqs_plot)//2]
        self._frequency_line = ax_spectrum.axvline(init_freq, color='red', 
                                                  linestyle='--', linewidth=2, alpha=0.8)
        
        # Plot initial mode
        self._current_frequency = init_freq
        self._update_mode_plots(components, z_layer)
        
        # Set up click handler with proper cleanup
        def on_click(event):
            if event.inaxes == ax_spectrum and event.xdata is not None:
                if event.button == 3:  # Right click - snap to peak
                    if peaks:
                        peak_freqs = [p.freq for p in peaks]
                        closest_peak_freq = peak_freqs[np.argmin(np.abs(np.array(peak_freqs) - event.xdata))]
                        selected_freq = closest_peak_freq
                    else:
                        selected_freq = event.xdata
                else:  # Left click - exact frequency
                    selected_freq = event.xdata
                
                # Update frequency line and mode plots
                self._frequency_line.set_xdata([selected_freq, selected_freq])
                self._current_frequency = selected_freq
                self._update_mode_plots(components, z_layer)
                self._interactive_fig.canvas.draw()
        
        # Store event connection for cleanup
        self._click_connection = self._interactive_fig.canvas.mpl_connect('button_press_event', on_click)
        
        # Add cleanup method to figure
        def cleanup():
            if hasattr(self, '_click_connection') and self._click_connection:
                self._interactive_fig.canvas.mpl_disconnect(self._click_connection)
                self._click_connection = None
                log.debug("Interactive plot event handlers cleaned up")
        
        # Store cleanup function for later use
        self._interactive_fig._mmpp_cleanup = cleanup
        
        plt.tight_layout()
        log.info("Interactive spectrum plot created. Click to select frequency, right-click to snap to peaks.")
        
        return self._interactive_fig
        
    def _update_mode_plots(self, components: List[Union[int, str]], z_layer: int) -> None:
        """Update mode plots for current frequency."""
        if self._mode_axes is None or self._current_frequency is None:
            return
            
        # Clear all axes
        for ax_row in self._mode_axes:
            for ax in ax_row:
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Get mode data
        try:
            mode_data = self.get_mode(self._current_frequency, z_layer)
        except Exception as e:
            log.error(f"Failed to get mode data: {e}")
            # Show error message on plots instead of leaving them empty
            for ax_row in self._mode_axes:
                for ax in ax_row:
                    ax.text(0.5, 0.5, f"Error loading mode data:\n{str(e)}", 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, color='red', wrap=True)
            return
        
        # Plot each component
        for i, comp in enumerate(components):
            try:
                comp_data = mode_data.get_component(comp)
                magnitude = np.abs(comp_data)
                phase = np.angle(comp_data)
                
                # Magnitude (row 0)
                im1 = self._mode_axes[0, i].imshow(magnitude,
                                                 cmap=self.config.colormap_magnitude,
                                                 extent=mode_data.extent,
                                                 aspect='equal',
                                                 interpolation=self.config.interpolation,
                                                 origin='lower')
                self._mode_axes[0, i].set_title(f'|m_{comp}|')
                
                # Phase (row 1)  
                im2 = self._mode_axes[1, i].imshow(phase,
                                                 cmap=self.config.colormap_phase,
                                                 extent=mode_data.extent,
                                                 aspect='equal',
                                                 interpolation=self.config.interpolation,
                                                 vmin=-np.pi, vmax=np.pi,
                                                 origin='lower')
                self._mode_axes[1, i].set_title(f'arg(m_{comp})')
                
                # Combined (row 2)
                alpha = magnitude / np.max(magnitude) if np.max(magnitude) > 0 else magnitude
                im3 = self._mode_axes[2, i].imshow(phase,
                                                 cmap=self.config.colormap_phase,
                                                 extent=mode_data.extent,
                                                 aspect='equal',
                                                 interpolation=self.config.interpolation,
                                                 alpha=alpha,
                                                 vmin=-np.pi, vmax=np.pi,
                                                 origin='lower')
                self._mode_axes[2, i].set_title(f'm_{comp} combined')
                
            except Exception as e:
                log.error(f"Failed to plot component {comp}: {e}")
                continue
        
        # Add frequency info
        self._interactive_fig.suptitle(f'FMR Modes at {self._current_frequency:.3f} GHz', 
                                     fontsize=14, fontweight='bold')

    def compute_modes(self, z_slice: slice = slice(None), 
                     window: bool = True, 
                     save: bool = True,
                     force: bool = False) -> None:
        """
        Compute FMR modes from magnetization data.
        
        Parameters:
        -----------
        z_slice : slice
            Z-layer slice to process
        window : bool
            Apply Hanning window
        save : bool
            Save results to zarr
        force : bool
            Force recomputation even if data exists
        """
        if not force and f"modes/{self.dataset_name}/arr" in self.zarr_file:
            log.info("Mode data already exists, use force=True to recompute")
            return
            
        log.info(f"Computing FMR modes for dataset {self.dataset_name}")
        
        # Remove existing data if force=True
        if force:
            try:
                # Open in write mode for deletion
                zarr_write = zarr.open(self.zarr_path, mode='a')
                if f"modes/{self.dataset_name}" in zarr_write:
                    del zarr_write[f"modes/{self.dataset_name}"]
                    log.info(f"Removed existing modes data for {self.dataset_name}")
                if f"fft/{self.dataset_name}" in zarr_write:
                    del zarr_write[f"fft/{self.dataset_name}"]
                    log.info(f"Removed existing FFT data for {self.dataset_name}")
                zarr_write.close()
                # Important: Reopen in read mode and reload data paths
                self.zarr_file = zarr.open(self.zarr_path, mode='r')
                self._load_data()  # Reload paths after deletion
            except Exception as e:
                log.warning(f"Could not remove existing data: {e}")
                # Continue anyway - might be permission issue
        
        # Load magnetization data
        if self.dataset_name not in self.zarr_file:
            raise ValueError(f"Dataset {self.dataset_name} not found in zarr")
            
        dset = self.zarr_file[self.dataset_name]
        
        # Get time array
        try:
            t_array = dset.attrs["t"][:]
            dt = (t_array[-1] - t_array[0]) / len(t_array)
        except:
            # Fallback to dt from zarr attrs
            dt = float(self.zarr_file.attrs.get("dt", 1e-12))
            t_array = np.arange(dset.shape[0]) * dt
        
        # Calculate frequencies
        freqs = np.fft.rfftfreq(len(t_array), dt) * 1e-9  # Convert to GHz
        
        # Load and process data
        log.info(f"Loading magnetization data: {dset.shape}")
        arr = np.asarray(dset[:, z_slice])
        log.info(f"Loading magnetization data finished")

        # Remove DC component
        arr = arr - arr.mean(axis=0)[None, ...]
        
        # Apply window function
        if window:
            window_func = np.hanning(arr.shape[0])
            for i in range(arr.ndim - 1):
                window_func = window_func[:, None]
            arr = arr * window_func
            log.info("Applied Hanning window")
        
        # Compute FFT
        log.info("Computing FFT...")
        fft_result = np.fft.rfft(arr, axis=0)
        log.info("Computing FFT finished.")

        # Save results
        if save:
            log.info("Saving mode data...")
            
            # Open in write mode
            zarr_write = zarr.open(self.zarr_path, mode='a')
            
            # Create groups
            modes_group = zarr_write.require_group(f"modes/{self.dataset_name}")
            fft_group = zarr_write.require_group(f"fft/{self.dataset_name}")
            
            # Save frequencies
            modes_group.array("freqs", freqs, chunks=False, overwrite=True)
            fft_group.array("freqs", freqs, chunks=False, overwrite=True)
            
            # Save complex modes (chunked only on first dimension)
            chunks = (1,) + fft_result.shape[1:]
            modes_group.array("arr", fft_result, 
                             dtype=np.complex64, 
                             chunks=chunks, 
                             overwrite=True)
            
            # Save power spectrum
            power_spec = np.abs(fft_result)
            fft_group.array("spec", np.max(power_spec, axis=(1, 2, 3)), 
                           chunks=False, overwrite=True)
            fft_group.array("sum", np.sum(power_spec, axis=(1, 2, 3)), 
                           chunks=False, overwrite=True)
            
            # Save metadata
            modes_group.attrs["computed_at"] = str(datetime.now())
            modes_group.attrs["window_applied"] = window
            modes_group.attrs["z_slice"] = str(z_slice)
            modes_group.attrs["dt"] = dt
            
            # zarr groups don't have close() method, just let it go out of scope
            log.info("✅ Mode computation completed and saved")
        
        # Reload data
        self.zarr_file = zarr.open(self.zarr_path, mode='r')
        self._load_data()

    def save_modes_animation(self, frequency_range: Tuple[float, float],
                            save_path: str,
                            fps: int = 10,
                            z_layer: int = 0,
                            component: Union[str, int] = 'z') -> None:
        """
        Save animation of modes across frequency range.
        
        Parameters:
        -----------
        frequency_range : tuple
            (f_min, f_max) in GHz
        save_path : str
            Output file path (.gif or .mp4)
        fps : int
            Frames per second
        z_layer : int
            Z-layer to animate
        component : str or int
            Component to animate
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for animations")
            
        try:
            from matplotlib.animation import FuncAnimation
            
            # Get frequency indices
            f_min, f_max = frequency_range
            freq_mask = (self.frequencies >= f_min) & (self.frequencies <= f_max)
            freq_indices = np.where(freq_mask)[0]
            
            if len(freq_indices) == 0:
                raise ValueError("No frequencies found in specified range")
            
            # Setup figure
            fig, ax = plt.subplots(figsize=(8, 6), dpi=self.config.dpi)
            
            def animate(frame):
                ax.clear()
                freq_idx = freq_indices[frame]
                frequency = self.frequencies[freq_idx]
                
                # Get mode data
                mode_data = self.get_mode(frequency, z_layer)
                comp_data = mode_data.get_component(component)
                
                # Plot magnitude
                magnitude = np.abs(comp_data)
                im = ax.imshow(magnitude, 
                              cmap=self.config.colormap_magnitude,
                              extent=mode_data.extent,
                              aspect='equal',
                              origin='lower',
                              interpolation=self.config.interpolation)
                
                ax.set_title(f'|m_{component}| @ {frequency:.3f} GHz')
                ax.set_xlabel('x (nm)')
                ax.set_ylabel('y (nm)')
                
                return [im]
            
            # Create animation
            anim = FuncAnimation(fig, animate, frames=len(freq_indices),
                               interval=1000/fps, blit=False, repeat=True)
            
            # Save animation
            anim.save(save_path, fps=fps, dpi=self.config.dpi)
            log.info(f"Saved animation to {save_path}")
            
        except ImportError:
            log.error("Animation requires matplotlib.animation")
            raise
        except Exception as e:
            log.error(f"Failed to create animation: {e}")
            raise


class FFTModeInterface:
    """
    Enhanced FFT interface with mode visualization capabilities.
    
    Provides elegant syntax like: op[0].fft[0][200].plot_modes()
    """
    
    def __init__(self, fft_result_index: int, parent_fft):
        """Initialize mode interface for specific FFT result."""
        self.fft_result_index = fft_result_index
        self.parent_fft = parent_fft
        self._mode_analyzer = None
        
    def __getitem__(self, frequency_index: int) -> "FrequencyModeInterface":
        """Get mode interface for specific frequency index."""
        return FrequencyModeInterface(frequency_index, self)
        
    @property 
    def mode_analyzer(self) -> FMRModeAnalyzer:
        """Get or create mode analyzer (lazy initialization)."""
        if self._mode_analyzer is None:
            # Get zarr path from parent FFT
            zarr_path = self.parent_fft.job_result.path
            debug_mode = getattr(self.parent_fft.mmpp, 'debug', False) if self.parent_fft.mmpp else False
            self._mode_analyzer = FMRModeAnalyzer(zarr_path, debug=debug_mode)
            
        return self._mode_analyzer
        
    def interactive_spectrum(self, dset: str = None, **kwargs) -> Figure:
        """Create interactive spectrum plot."""
        # If dset is specified, create a new analyzer for that dataset
        if dset is not None and dset != self.mode_analyzer.dataset_name:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = getattr(self.parent_fft.mmpp, 'debug', False) if self.parent_fft.mmpp else False
            temp_analyzer = FMRModeAnalyzer(zarr_path, dataset_name=dset, debug=debug_mode)
            
            # Check if modes exist, if not compute them
            if not temp_analyzer.modes_available:
                log.info(f"Computing modes for dataset '{dset}'...")
                temp_analyzer.compute_modes(save=True)
                
            return temp_analyzer.interactive_spectrum(**kwargs)
        else:
            # Use default analyzer
            if not self.mode_analyzer.modes_available:
                log.info(f"Computing modes for dataset '{self.mode_analyzer.dataset_name}'...")
                self.mode_analyzer.compute_modes(save=True)
                
            return self.mode_analyzer.interactive_spectrum(**kwargs)
        
    def compute_modes(self, dset: str = None, **kwargs) -> None:
        """Compute modes for specified dataset."""
        if dset is not None:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = getattr(self.parent_fft.mmpp, 'debug', False) if self.parent_fft.mmpp else False
            temp_analyzer = FMRModeAnalyzer(zarr_path, dataset_name=dset, debug=debug_mode)
            temp_analyzer.compute_modes(**kwargs)
        else:
            self.mode_analyzer.compute_modes(**kwargs)
        
    def plot_modes(self, frequency: float, dset: str = None, **kwargs) -> Tuple[Figure, np.ndarray]:
        """Plot modes at specified frequency."""
        # If dset is specified, create a new analyzer for that dataset
        if dset is not None and dset != self.mode_analyzer.dataset_name:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = getattr(self.parent_fft.mmpp, 'debug', False) if self.parent_fft.mmpp else False
            temp_analyzer = FMRModeAnalyzer(zarr_path, dataset_name=dset, debug=debug_mode)
            
            # Check if modes exist, if not compute them
            if not temp_analyzer.modes_available:
                log.info(f"Computing modes for dataset '{dset}'...")
                temp_analyzer.compute_modes(save=True)
                
            return temp_analyzer.plot_modes(frequency, **kwargs)
        else:
            # Use default analyzer
            if not self.mode_analyzer.modes_available:
                log.info(f"Computing modes for dataset '{self.mode_analyzer.dataset_name}'...")
                self.mode_analyzer.compute_modes(save=True)
                
            return self.mode_analyzer.plot_modes(frequency, **kwargs)
        

class FrequencyModeInterface:
    """Interface for mode operations at a specific frequency."""
    
    def __init__(self, frequency_index: int, parent_mode_interface):
        """Initialize frequency-specific mode interface."""
        self.frequency_index = frequency_index
        self.parent = parent_mode_interface
        
    @property
    def frequency(self) -> float:
        """Get frequency value for this index."""
        return self.parent.mode_analyzer.frequencies[self.frequency_index]
        
    def plot_modes(self, **kwargs) -> Tuple[Figure, np.ndarray]:
        """Plot modes at this frequency."""
        return self.parent.mode_analyzer.plot_modes(self.frequency, **kwargs)
        
    def get_mode(self, **kwargs) -> FMRModeData:
        """Get mode data at this frequency.""" 
        return self.parent.mode_analyzer.get_mode(self.frequency, **kwargs)
