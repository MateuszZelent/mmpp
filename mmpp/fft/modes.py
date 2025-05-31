"""
FMR Mode Visualization Module

Professional implementation for visualizing FMR modes with interactive spectrum.
Provides both programmatic and interactive interfaces for mode analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from typing import Optional, Dict, List, Union, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime

# Import shared logging configuration
from ..logging_config import setup_mmpp_logging, get_mmpp_logger

# Get logger for FMR modes
log = get_mmpp_logger("mmpp.fft.modes")

# Import electromagnetic analysis module
try:
    from .electromagnetic_analysis import (
        ElectromagneticAnalysisConfig,
        PoyntingVectorAnalysis,
        RadiationPatternAnalysis,
        QFactorAnalysis,
        analyze_electromagnetic_properties,
        create_comprehensive_em_report,
    )

    EM_ANALYSIS_AVAILABLE = True
except ImportError:
    EM_ANALYSIS_AVAILABLE = False
    log.warning("Electromagnetic analysis module not available")

# Import styling functions from plotting module
try:
    from ..plotting import setup_custom_fonts, load_paper_style, apply_custom_colors

    STYLING_AVAILABLE = True
except ImportError:
    STYLING_AVAILABLE = False
    log.warning("Styling functions not available - using default matplotlib styling")

# Import electromagnetic analysis module
try:
    from .electromagnetic_analysis import (
        ElectromagneticAnalysisConfig,
        PoyntingVectorAnalysis,
        RadiationPatternAnalysis,
        QFactorAnalysis,
        analyze_electromagnetic_properties,
        create_comprehensive_em_report,
    )

    EM_ANALYSIS_AVAILABLE = True
except ImportError:
    EM_ANALYSIS_AVAILABLE = False
    log.warning("Electromagnetic analysis module not available")

# Import dependencies with error handling
try:
    import zarr

    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    log.error("Zarr not available - mode analysis disabled")

try:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    log.warning("Matplotlib not available - mode visualization disabled")

try:
    from scipy.signal import find_peaks

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    log.warning("SciPy not available - peak detection features limited")

# Check for animation support
try:
    from matplotlib.animation import FuncAnimation, PillowWriter

    ANIMATION_AVAILABLE = True

    # Check for FFmpeg support
    try:
        from matplotlib.animation import FFMpegWriter

        FFMPEG_AVAILABLE = True
        log.debug("FFmpeg available for MP4 animations")
    except ImportError:
        FFMPEG_AVAILABLE = False
        log.debug("FFmpeg not available - MP4 animations will fallback to GIF")

except ImportError:
    ANIMATION_AVAILABLE = False
    FFMPEG_AVAILABLE = False
    log.warning("Animation support not available")

# Check for scientific colormaps
try:
    import cmcrameri.cm as cmc

    CMCRAMERI_AVAILABLE = True
    log.debug("cmcrameri colormaps available")
except ImportError:
    CMCRAMERI_AVAILABLE = False
    log.debug("cmcrameri not available - using standard matplotlib colormaps")

try:
    import cmocean

    CMOCEAN_AVAILABLE = True
    log.debug("cmocean colormaps available")
except ImportError:
    CMOCEAN_AVAILABLE = False
    log.debug("cmocean not available - using standard matplotlib colormaps")


class MidpointNormalize(mcolors.Normalize):
    """
    Matplotlib normalization class with symmetric colormap around midpoint.

    Useful for data that has meaningful zero point (like magnetic field components)
    where you want symmetric color scaling around zero.
    """

    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        """
        Initialize normalization.

        Parameters:
        -----------
        vmin : float, optional
            Minimum value for normalization
        vmax : float, optional
            Maximum value for normalization
        midpoint : float, optional
            Value that should map to center of colormap (default: 0)
        clip : bool, optional
            Whether to clip values outside [vmin, vmax]
        """
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """Apply normalization to values."""
        # Handle the case where vmin, vmax, or midpoint could be None
        if self.vmin is None or self.vmax is None:
            return mcolors.Normalize.__call__(self, value, clip)

        # Calculate normalized positions for min, mid, max
        normalized_min = max(
            0,
            1
            / 2
            * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))),
        )
        normalized_max = min(
            1,
            1
            / 2
            * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))),
        )
        normalized_mid = 0.5

        # Interpolate
        x = [self.vmin, self.midpoint, self.vmax]
        y = [normalized_min, normalized_mid, normalized_max]

        return np.ma.masked_array(np.interp(value, x, y))


def setup_animation_styling(
    use_paper_style: bool = True, use_custom_fonts: bool = True
) -> bool:
    """
    Setup styling for FMR mode animations using MMPP paper style.

    Parameters:
    -----------
    use_paper_style : bool, optional
        Whether to apply paper.mplstyle styling (default: True)
    use_custom_fonts : bool, optional
        Whether to setup custom fonts (default: True)

    Returns:
    --------
    bool
        True if styling was successfully applied
    """
    if not STYLING_AVAILABLE:
        log.warning(
            "Styling functions not available - using default matplotlib styling"
        )
        return False

    try:
        success = True

        # Setup custom fonts if requested
        if use_custom_fonts:
            font_success = setup_custom_fonts(verbose=False)
            if not font_success:
                log.warning("Custom font setup failed - using default fonts")
                success = False

        # Load paper style if requested
        if use_paper_style:
            style_success = load_paper_style(verbose=False)
            if style_success:
                log.debug("✓ Applied paper.mplstyle to mode animations")
            else:
                log.warning("Paper style loading failed - using default style")
                success = False

        # Apply custom colors for better visibility
        custom_colors = {
            "text": "#2E2E2E",  # Dark gray for text
            "axes": "#2E2E2E",  # Dark gray for axes
            "grid": "#CCCCCC",  # Light gray for grid
        }
        apply_custom_colors(custom_colors)

        return success

    except Exception as e:
        log.warning(f"Animation styling setup failed: {e}")
        return False


def check_ffmpeg_available() -> bool:
    """
    Check if ffmpeg is available for MP4 animations.

    Returns:
    --------
    bool
        True if ffmpeg is available
    """
    try:
        import subprocess

        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return False


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
    colormap_magnitude: str = "cmc.berlin"  # cmcrameri berlin for amplitude data
    colormap_phase: str = "cmc.romaO"  # cmcrameri romaO for phase data
    colormap_animation: str = (
        "balance"  # cmocean.cm.balance for animations, RdBu_r fallback
    )
    interpolation: str = "nearest"
    use_midpoint_norm: bool = False  # Use MidpointNormalize for diverging data
    animation_time_steps: int = 60  # Number of time steps for one full phase cycle

    # Frequency range for analysis
    f_min: float = 0.0
    f_max: float = 40.0

    # Layout settings
    spectrum_width_ratio: float = 0.4
    modes_width_ratio: float = 0.6

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.f_min >= self.f_max:
            raise ValueError(
                f"f_min ({self.f_min}) must be less than f_max ({self.f_max})"
            )

        if self.peak_threshold < 0 or self.peak_threshold > 1:
            raise ValueError(
                f"peak_threshold must be between 0 and 1, got {self.peak_threshold}"
            )

        if self.peak_min_distance < 1:
            raise ValueError(
                f"peak_min_distance must be >= 1, got {self.peak_min_distance}"
            )

        if self.spectrum_width_ratio <= 0 or self.modes_width_ratio <= 0:
            raise ValueError("Width ratios must be positive")

        if self.dpi < 50 or self.dpi > 500:
            log.warning(f"Unusual DPI value: {self.dpi}")

        # Validate colormaps
        try:
            self._resolve_colormap(self.colormap_magnitude)
            self._resolve_colormap(self.colormap_phase)
        except Exception as e:
            log.warning(f"Colormap validation failed: {e}")

    def _resolve_colormap(self, cmap_name: str):
        """
        Resolve colormap from various sources (cmcrameri, cmocean, matplotlib).

        Parameters:
        -----------
        cmap_name : str
            Name of the colormap

        Returns:
        --------
        matplotlib colormap object
        """
        # Try cmcrameri first (scientific colormaps)
        if CMCRAMERI_AVAILABLE:
            try:
                return getattr(cmc, cmap_name)
            except AttributeError:
                pass

        # Try cmocean (oceanographic colormaps)
        if CMOCEAN_AVAILABLE:
            try:
                import cmocean

                return getattr(cmocean.cm, cmap_name)
            except AttributeError:
                pass

        # Fallback to matplotlib
        import matplotlib.pyplot as plt

        return plt.get_cmap(cmap_name)


@dataclass
class Peak:
    """Peak data structure."""

    idx: int
    freq: float
    amplitude: float


class FMRModeData:
    """Container for FMR mode data at a specific frequency."""

    def __init__(
        self,
        frequency: float,
        mode_array: np.ndarray,
        extent: Optional[Tuple[float, float, float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
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
        component_map = {"x": 0, "y": 1, "z": 2, "mx": 0, "my": 1, "mz": 2}

        if isinstance(component, str):
            if component.lower() not in component_map:
                raise ValueError(
                    f"Unknown component '{component}'. Use 'x', 'y', 'z' or 0, 1, 2"
                )
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

    def __init__(
        self,
        zarr_path: str,
        dataset_name: str = "m",
        config: Optional[ModeVisualizationConfig] = None,
        debug: bool = False,
    ):
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
        return (
            self.modes_path is not None
            and self.freqs_path is not None
            and self.spectrum_path is not None
        )

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
            if (
                f"{base_path}/arr" in self.zarr_file
                and f"{base_path}/freqs" in self.zarr_file
            ):
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
        spectrum_candidates = [
            f"fft/{self.dataset_name}/spec",
            f"fft/{self.dataset_name}/sum",
        ]
        for path in spectrum_candidates:
            if path in self.zarr_file:
                spectrum_path = path
                break

        return modes_path, freqs_path, spectrum_path

    def _load_data(self) -> None:
        """Load mode and spectrum data from zarr file."""
        try:
            self.zarr_file = zarr.open(self.zarr_path, mode="r")
            log.info(f"Opened zarr file: {self.zarr_path}")
        except Exception as e:
            log.error(f"Failed to open zarr file {self.zarr_path}: {e}")
            raise

        # Use unified path resolution
        self.modes_path, self.freqs_path, self.spectrum_path = self._get_zarr_paths()

        if not self.modes_path:
            log.debug(
                f"No mode data found for dataset '{self.dataset_name}'. "
                "Modes will need to be computed."
            )

        if not self.freqs_path:
            log.debug(
                f"No frequency data found for dataset '{self.dataset_name}'. "
                "Frequencies will be computed with modes."
            )

        if not self.spectrum_path:
            log.warning(
                f"No spectrum data found for dataset '{self.dataset_name}'. "
                f"Expected paths: fft/{self.dataset_name}/spec or fft/{self.dataset_name}/sum"
            )

        # Load frequency array if available
        if self.freqs_path:
            self.frequencies = np.array(self.zarr_file[self.freqs_path])
            log.info(
                f"Loaded frequencies: {len(self.frequencies)} points, "
                f"range {self.frequencies[0]:.3f} - {self.frequencies[-1]:.3f} GHz"
            )
        else:
            self.frequencies = None
            log.debug("No frequency data loaded - will be computed with modes")

        # Load spectrum if available
        if self.spectrum_path:
            self.spectrum = np.array(self.zarr_file[self.spectrum_path])
            if self.spectrum.ndim > 1:
                # Take first component if multi-component
                self.spectrum = (
                    self.spectrum[:, 0]
                    if self.spectrum.shape[1] == 3
                    else np.sum(self.spectrum, axis=tuple(range(1, self.spectrum.ndim)))
                )
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
            (
                self.zarr_file[self.dataset_name].attrs
                if self.dataset_name in self.zarr_file
                else {}
            ),
        ]

        for attrs in attrs_to_check:
            if "dx" in attrs:
                self.dx = float(attrs["dx"]) * 1e9  # Convert to nm
            if "dy" in attrs:
                self.dy = float(attrs["dy"]) * 1e9  # Convert to nm

        log.debug(f"Spatial resolution: dx={self.dx:.3f} nm, dy={self.dy:.3f} nm")

    def _detect_peaks(
        self, spectrum: np.ndarray, frequencies: np.ndarray
    ) -> List[Peak]:
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
                if (
                    spectrum[i] > spectrum[i - 1]
                    and spectrum[i] > spectrum[i + 1]
                    and spectrum[i] > self.config.peak_threshold * np.max(spectrum)
                ):
                    peaks.append(
                        Peak(idx=i, freq=frequencies[i], amplitude=spectrum[i])
                    )
            return peaks

        try:
            # Normalize spectrum for peak detection
            norm_spectrum = spectrum / np.max(spectrum)

            # Find peaks using scipy
            peak_indices, properties = find_peaks(
                norm_spectrum,
                height=self.config.peak_threshold,
                distance=self.config.peak_min_distance,
            )

            # Create peak objects
            peaks = []
            for idx in peak_indices:
                peaks.append(
                    Peak(idx=int(idx), freq=frequencies[idx], amplitude=spectrum[idx])
                )

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
            raise RuntimeError(
                "No frequency data available. Run compute_modes() first."
            )

        if self.modes_path is None:
            raise RuntimeError("No mode data available. Run compute_modes() first.")

        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))
        actual_freq = self.frequencies[freq_idx]

        if abs(actual_freq - frequency) > 0.1:
            log.warning(
                f"Requested frequency {frequency:.3f} GHz not found, "
                f"using closest: {actual_freq:.3f} GHz"
            )

        # Validate z_layer bounds
        mode_shape = self.zarr_file[self.modes_path].shape
        if z_layer >= mode_shape[1]:
            raise ValueError(
                f"z_layer {z_layer} out of range. Available layers: 0-{mode_shape[1]-1}"
            )

        # Load mode data for this frequency with bounds checking
        try:
            mode_data = self.zarr_file[self.modes_path][freq_idx, z_layer, :, :, :]
        except IndexError as e:
            raise ValueError(
                f"Invalid indices: freq_idx={freq_idx}, z_layer={z_layer}. {e}"
            )

        # Create spatial extent
        ny, nx = mode_data.shape[:2]
        extent = (0, nx * self.dx, 0, ny * self.dy)

        # Metadata
        metadata = {
            "frequency_index": freq_idx,
            "requested_frequency": frequency,
            "actual_frequency": actual_freq,
            "z_layer": z_layer,
            "spatial_resolution": (self.dx, self.dy),
            "mode_shape": mode_shape,
        }

        # Update cache
        self._update_cache(
            frequency, z_layer, FMRModeData(actual_freq, mode_data, extent, metadata)
        )

        return FMRModeData(actual_freq, mode_data, extent, metadata)

    def _update_cache(
        self, frequency: float, z_layer: int, mode_data: FMRModeData
    ) -> None:
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

    def find_peaks(
        self,
        threshold: Optional[float] = None,
        min_distance: Optional[int] = None,
        component: int = 0,
    ) -> List[Peak]:
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
        freq_mask = (self.frequencies >= self.config.f_min) & (
            self.frequencies <= self.config.f_max
        )
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
                    log.warning(
                        f"Peak index {peak.idx} out of range for filtered array"
                    )
            except IndexError as e:
                log.warning(f"Index mapping error for peak {peak.idx}: {e}")
                continue

        log.info(
            f"Found {len(peaks_converted)} peaks in frequency range "
            f"{self.config.f_min}-{self.config.f_max} GHz"
        )

        return peaks_converted

    def plot_modes(
        self,
        frequency: float,
        z_layer: int = 0,
        components: Optional[List[Union[int, str]]] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[Figure, np.ndarray]:
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

        # Setup professional styling for mode plots
        setup_animation_styling(use_paper_style=True, use_custom_fonts=True)

        components = components or ["x", "y", "z"]
        mode_data = self.get_mode(frequency, z_layer)

        # Create figure with subplots
        n_components = len(components)
        n_rows = (
            3
            if self.config.show_magnitude
            and self.config.show_phase
            and self.config.show_combined
            else 2
        )

        fig, axes = plt.subplots(
            n_rows,
            n_components,
            figsize=(4 * n_components, 3 * n_rows),
            dpi=self.config.dpi,
        )

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
                im1 = axes[row, i].imshow(
                    magnitude,
                    cmap=self.config._resolve_colormap(self.config.colormap_magnitude),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=self.config.interpolation,
                    origin="lower",
                )
                axes[row, i].set_title(f"|m_{comp}| @ {frequency:.3f} GHz")
                axes[row, i].set_xlabel("x (nm)")
                if i == 0:
                    axes[row, i].set_ylabel("y (nm)")
                plt.colorbar(im1, ax=axes[row, i], shrink=0.8)
                row += 1

            # Phase plot
            if self.config.show_phase:
                im2 = axes[row, i].imshow(
                    phase,
                    cmap=self.config._resolve_colormap(self.config.colormap_phase),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=self.config.interpolation,
                    vmin=-np.pi,
                    vmax=np.pi,
                    origin="lower",
                )
                axes[row, i].set_title(f"arg(m_{comp}) @ {frequency:.3f} GHz")
                axes[row, i].set_xlabel("x (nm)")
                if i == 0:
                    axes[row, i].set_ylabel("y (nm)")
                plt.colorbar(im2, ax=axes[row, i], shrink=0.8)
                row += 1

            # Combined plot (phase with magnitude as alpha)
            if self.config.show_combined:
                # Create combined visualization: magnitude * cos(phase) for real part
                combined_data = magnitude * np.cos(phase)  # Real part
                # Alternative: combined_data = magnitude * np.sin(phase)  # Imaginary part

                im3 = axes[row, i].imshow(
                    combined_data,
                    cmap=self.config._resolve_colormap(self.config.colormap_phase),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=self.config.interpolation,
                    origin="lower",
                )
                axes[row, i].set_title(
                    f"m_{comp} combined (mag×cos(φ)) @ {frequency:.3f} GHz"
                )
                axes[row, i].set_xlabel("x (nm)")
                if i == 0:
                    axes[row, i].set_ylabel("y (nm)")
                plt.colorbar(im3, ax=axes[row, i], shrink=0.8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            log.info(f"Saved mode plot to {save_path}")

        return fig, axes

    def interactive_spectrum(
        self,
        components: Optional[List[Union[int, str]]] = None,
        z_layer: int = 0,
        method: int = 1,
        show: bool = True,
        **kwargs,
    ) -> Figure:
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
        method : int, optional
            Visualization method (default: 1)
            1 = Standard interactive plot
            2 = Alternative layout (if implemented)
        show : bool, optional
            Whether to automatically display the figure (default: True)
        \\*\\*kwargs : dict
            Additional keyword arguments:
            - figsize : tuple, optional
                Figure size (width, height) in inches (default: from config)
            - dpi : int, optional
                Figure resolution in dots per inch (default: from config)
            - cmap : str, optional
                Colormap for all mode visualizations (overrides config colormaps)
                Examples: 'viridis', 'inferno', 'plasma', 'cividis', 'balance'
            - acmap : str, optional
                Colormap specifically for amplitude/magnitude plots (overrides cmap for magnitude)
                Examples: 'viridis', 'inferno', 'plasma', 'hot'
            - pcmap : str, optional
                Colormap specifically for phase plots (overrides cmap for phase)
                Examples: 'hsv', 'twilight', 'twilight_shifted', 'phase'

        Returns:
        --------
        Figure
            Interactive matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for interactive plotting")

        if self.spectrum is None:
            raise ValueError("No spectrum data available for interactive mode")

        # Apply paper style for consistent visualization
        if STYLING_AVAILABLE:
            try:
                load_paper_style(verbose=False)
                log.debug("Applied paper style to interactive spectrum")
            except Exception as e:
                log.warning(f"Could not apply paper style: {e}")

        # Handle method parameter
        if method not in [1, 2]:
            log.warning(f"Unknown method {method}, using default method 1")
            method = 1

        components = components or ["x", "y", "z"]
        n_components = len(components)

        # Extract parameters from kwargs
        figsize = kwargs.get("figsize", self.config.figsize)
        dpi = kwargs.get("dpi", self.config.dpi)
        cmap = kwargs.get("cmap", None)
        acmap = kwargs.get("acmap", None)  # Amplitude/magnitude colormap
        pcmap = kwargs.get("pcmap", None)  # Phase colormap

        # Update colormaps if provided
        if cmap or acmap or pcmap:
            # Create a temporary copy of config with updated colormaps
            import copy

            temp_config = copy.deepcopy(self.config)

            # If cmap is provided, use it for all types unless specifically overridden
            if cmap:
                temp_config.colormap_magnitude = cmap
                temp_config.colormap_phase = cmap

            # Override with specific colormaps if provided
            if acmap:
                temp_config.colormap_magnitude = acmap
            if pcmap:
                temp_config.colormap_phase = pcmap

            self.config = temp_config

        # Update figure settings from kwargs
        self.config.figsize = figsize
        self.config.dpi = dpi

        # Validate number of components for layout
        if n_components > 5:
            raise ValueError(
                f"Too many components ({n_components}). Maximum supported: 5"
            )

        # Create figure with custom layout
        # Try to use interactive backend if available, but don't force it
        try:
            import matplotlib

            current_backend = matplotlib.get_backend()
            if (
                "ipympl" not in current_backend.lower()
                and "widget" not in current_backend.lower()
            ):
                log.info(
                    f"Current backend: {current_backend}. Interactive features may be limited."
                )
        except Exception as e:
            log.warning(f"Could not check matplotlib backend: {e}")

        self._interactive_fig = plt.figure(figsize=figsize, dpi=dpi)

        # Create grid layout: spectrum on left, modes on right
        # Use dynamic number of rows (3 for all visualization types)
        n_vis_types = sum(
            [
                self.config.show_magnitude,
                self.config.show_phase,
                self.config.show_combined,
            ]
        )
        if n_vis_types == 0:
            raise ValueError("At least one visualization type must be enabled")

        gs = gridspec.GridSpec(
            n_vis_types,
            n_components + 1,
            width_ratios=[self.config.spectrum_width_ratio]
            + [self.config.modes_width_ratio / n_components] * n_components,
            height_ratios=[1] * n_vis_types,
        )

        # Spectrum plot spans all rows in first column
        ax_spectrum = self._interactive_fig.add_subplot(gs[:, 0])

        # Mode plots in remaining columns - dynamic based on enabled visualizations
        self._mode_axes = np.array(
            [
                [
                    self._interactive_fig.add_subplot(gs[row, col + 1])
                    for col in range(n_components)
                ]
                for row in range(n_vis_types)
            ]
        )

        # Plot spectrum
        freq_mask = (self.frequencies >= self.config.f_min) & (
            self.frequencies <= self.config.f_max
        )
        freqs_plot = self.frequencies[freq_mask]
        spectrum_plot = self.spectrum[freq_mask]

        if self.config.spectrum_normalize:
            spectrum_plot = spectrum_plot / np.max(spectrum_plot)

        if self.config.spectrum_log_scale:
            spectrum_plot = np.log10(spectrum_plot + 1e-10)
            ax_spectrum.set_ylabel("log₁₀(Power Spectrum)")
        else:
            ax_spectrum.set_ylabel("Power Spectrum")

        ax_spectrum.plot(freqs_plot, spectrum_plot, "b-", linewidth=1.5)
        ax_spectrum.set_xlabel("Frequency (GHz)")
        ax_spectrum.set_title("FMR Spectrum (Click to select frequency)")
        ax_spectrum.grid(True, alpha=0.3)

        # Find and mark peaks
        peaks = self.find_peaks()
        for peak in peaks:
            if self.config.f_min <= peak.freq <= self.config.f_max:
                y_val = spectrum_plot[np.argmin(np.abs(freqs_plot - peak.freq))]
                ax_spectrum.plot(peak.freq, y_val, "ro", markersize=4)
                ax_spectrum.text(
                    peak.freq,
                    y_val + 0.05 * np.max(spectrum_plot),
                    f"{peak.freq:.2f}",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Initial frequency line
        init_freq = peaks[0].freq if peaks else freqs_plot[len(freqs_plot) // 2]
        self._frequency_line = ax_spectrum.axvline(
            init_freq, color="red", linestyle="--", linewidth=2, alpha=0.8
        )

        # Plot initial mode
        self._current_frequency = init_freq
        self._update_mode_plots(components, z_layer)

        # Set up click handler with proper cleanup
        def on_click(event):
            if event.inaxes == ax_spectrum and event.xdata is not None:
                if event.button == 3:  # Right click - snap to peak
                    if peaks:
                        peak_freqs = [p.freq for p in peaks]
                        closest_peak_freq = peak_freqs[
                            np.argmin(np.abs(np.array(peak_freqs) - event.xdata))
                        ]
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
        self._click_connection = self._interactive_fig.canvas.mpl_connect(
            "button_press_event", on_click
        )

        # Add cleanup method to figure
        def cleanup():
            if hasattr(self, "_click_connection") and self._click_connection:
                self._interactive_fig.canvas.mpl_disconnect(self._click_connection)
                self._click_connection = None
                log.debug("Interactive plot event handlers cleaned up")

        # Store cleanup function for later use
        self._interactive_fig._mmpp_cleanup = cleanup

        plt.tight_layout()
        log.info(
            "Interactive spectrum plot created. Click to select frequency, right-click to snap to peaks."
        )

        # Control figure display to avoid double showing
        if show:
            plt.show()
            return None  # Don't return figure to avoid Jupyter auto-display
        else:
            return self._interactive_fig

    def _update_mode_plots(
        self, components: List[Union[int, str]], z_layer: int
    ) -> None:
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
                    ax.text(
                        0.5,
                        0.5,
                        f"Error loading mode data:\n{str(e)}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        color="red",
                        wrap=True,
                    )
            return

        # Plot each component
        for i, comp in enumerate(components):
            try:
                comp_data = mode_data.get_component(comp)
                magnitude = np.abs(comp_data)
                phase = np.angle(comp_data)

                row_idx = 0

                # Magnitude plot (if enabled)
                if self.config.show_magnitude:
                    im1 = self._mode_axes[row_idx, i].imshow(
                        magnitude,
                        cmap=self.config._resolve_colormap(
                            self.config.colormap_magnitude
                        ),
                        extent=mode_data.extent,
                        aspect="equal",
                        interpolation=self.config.interpolation,
                        origin="lower",
                    )
                    self._mode_axes[row_idx, i].set_title(f"|m_{comp}|")
                    row_idx += 1

                # Phase plot (if enabled)
                if self.config.show_phase:
                    im2 = self._mode_axes[row_idx, i].imshow(
                        phase,
                        cmap=self.config._resolve_colormap(self.config.colormap_phase),
                        extent=mode_data.extent,
                        aspect="equal",
                        interpolation=self.config.interpolation,
                        vmin=-np.pi,
                        vmax=np.pi,
                        origin="lower",
                    )
                    self._mode_axes[row_idx, i].set_title(f"arg(m_{comp})")
                    row_idx += 1

                # Combined plot (if enabled)
                if self.config.show_combined:
                    # Create combined visualization: magnitude * cos(phase) for real part
                    # or magnitude * sin(phase) for imaginary part
                    # This shows the actual complex amplitude with sign
                    combined_data = magnitude * np.cos(phase)  # Real part
                    # Alternative: combined_data = magnitude * np.sin(phase)  # Imaginary part

                    im3 = self._mode_axes[row_idx, i].imshow(
                        combined_data,
                        cmap=self.config._resolve_colormap(self.config.colormap_phase),
                        extent=mode_data.extent,
                        aspect="equal",
                        interpolation=self.config.interpolation,
                        origin="lower",
                    )
                    self._mode_axes[row_idx, i].set_title(
                        f"m_{comp} combined (mag×cos(φ))"
                    )

            except Exception as e:
                log.error(f"Failed to plot component {comp}: {e}")
                continue

        # Add frequency info
        self._interactive_fig.suptitle(
            f"FMR Modes at {self._current_frequency:.3f} GHz",
            fontsize=14,
            fontweight="bold",
        )

    def compute_modes(
        self,
        z_slice: slice = slice(None),
        window: bool = True,
        save: bool = True,
        force: bool = False,
    ) -> None:
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
                zarr_write = zarr.open(self.zarr_path, mode="a")
                if f"modes/{self.dataset_name}" in zarr_write:
                    del zarr_write[f"modes/{self.dataset_name}"]
                    log.info(f"Removed existing modes data for {self.dataset_name}")
                if f"fft/{self.dataset_name}" in zarr_write:
                    del zarr_write[f"fft/{self.dataset_name}"]
                    log.info(f"Removed existing FFT data for {self.dataset_name}")
                zarr_write.close()
                # Important: Reopen in read mode and reload data paths
                self.zarr_file = zarr.open(self.zarr_path, mode="r")
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
            zarr_write = zarr.open(self.zarr_path, mode="a")

            # Create groups
            modes_group = zarr_write.require_group(f"modes/{self.dataset_name}")
            fft_group = zarr_write.require_group(f"fft/{self.dataset_name}")

            # Save frequencies
            modes_group.array("freqs", freqs, chunks=False, overwrite=True)
            fft_group.array("freqs", freqs, chunks=False, overwrite=True)

            # Save complex modes (chunked only on first dimension)
            chunks = (1,) + fft_result.shape[1:]
            modes_group.array(
                "arr", fft_result, dtype=np.complex64, chunks=chunks, overwrite=True
            )

            # Save power spectrum
            power_spec = np.abs(fft_result)
            fft_group.array(
                "spec", np.max(power_spec, axis=(1, 2, 3)), chunks=False, overwrite=True
            )
            fft_group.array(
                "sum", np.sum(power_spec, axis=(1, 2, 3)), chunks=False, overwrite=True
            )

            # Save metadata
            modes_group.attrs["computed_at"] = str(datetime.now())
            modes_group.attrs["window_applied"] = window
            modes_group.attrs["z_slice"] = str(z_slice)
            modes_group.attrs["dt"] = dt

            # zarr groups don't have close() method, just let it go out of scope
            log.info("✅ Mode computation completed and saved")

        # Reload data
        self.zarr_file = zarr.open(self.zarr_path, mode="r")
        self._load_data()

    def save_modes_animation(
        self,
        frequency_range: Tuple[float, float] = None,
        frequency: float = None,
        save_path: str = "mode_animation.gif",
        fps: int = 15,
        z_layer: int = 0,
        component: Union[str, int] = "z",
        animation_type: str = "temporal",
        colormap: str = None,
        use_midpoint_norm: bool = None,
        figsize: Tuple[float, float] = None,
    ) -> None:
        """
        Save animation of FMR modes.

        Parameters:
        -----------
        frequency_range : tuple, optional
            (f_min, f_max) in GHz for frequency sweep animation
        frequency : float, optional
            Single frequency for temporal animation (in GHz)
        save_path : str
            Output file path (.gif or .mp4)
        fps : int
            Frames per second (default: 15)
        z_layer : int
            Z-layer to animate (default: 0)
        component : str or int
            Component to animate (default: 'z')
        animation_type : str
            Type of animation:
            - 'temporal': Real part of mode oscillating in time at fixed frequency
            - 'frequency': Mode amplitude across frequency range
            - 'phase': Phase evolution at fixed frequency
        colormap : str, optional
            Colormap name. If None, uses config defaults
        use_midpoint_norm : bool, optional
            Use symmetric normalization around zero (default: from config)
        figsize : tuple, optional
            Figure size (width, height)
        """
        if not MATPLOTLIB_AVAILABLE or not ANIMATION_AVAILABLE:
            raise ImportError(
                "Matplotlib and animation support are required for animations"
            )

        # Setup professional styling for animations
        setup_animation_styling(use_paper_style=True, use_custom_fonts=True)

        try:
            from matplotlib.animation import FuncAnimation
            import matplotlib.cm as cm

            # Parameter validation
            if frequency_range is None and frequency is None:
                raise ValueError(
                    "Either frequency_range or frequency must be specified"
                )

            if frequency_range is not None and frequency is not None:
                raise ValueError(
                    "Specify either frequency_range OR frequency, not both"
                )

            # Set defaults with intelligent choices for animation type
            figsize = figsize or (10, 8)
            colormap = colormap or self.config.colormap_animation

            # Auto-enable MidpointNormalize for temporal animations if not explicitly set
            if use_midpoint_norm is None:
                if animation_type == "temporal":
                    use_midpoint_norm = (
                        True  # Temporal animations benefit from symmetric normalization
                    )
                else:
                    use_midpoint_norm = self.config.use_midpoint_norm

            # Choose better colormap for temporal oscillations if default 'balance'
            if animation_type == "temporal" and colormap == "balance":
                log.info(
                    "Using diverging colormap 'balance' - perfect for oscillating modes"
                )

            # Try to get cmocean colormaps for better scientific visualization
            try:
                import cmocean

                if colormap == "balance":
                    cmap = cmocean.cm.balance  # Perfect for data with +/- symmetry
                elif colormap == "diff":
                    cmap = cmocean.cm.diff  # Another good diverging colormap
                elif colormap == "curl":
                    cmap = cmocean.cm.curl  # Good for circular/phase data
                elif colormap == "delta":
                    cmap = cmocean.cm.delta  # Good for deviations from mean
                elif colormap == "tarn":
                    cmap = cmocean.cm.tarn  # Good for complex data
                else:
                    # Try as regular matplotlib colormap
                    cmap = plt.get_cmap(colormap)
            except ImportError:
                log.warning("cmocean not available, using matplotlib colormaps")
                if colormap == "balance":
                    cmap = plt.get_cmap("RdBu_r")  # Best fallback for balance
                elif colormap == "diff":
                    cmap = plt.get_cmap("RdBu")  # Alternative diverging
                elif colormap == "curl" or colormap == "tarn":
                    cmap = plt.get_cmap("RdYlBu_r")  # Complex fallback
                elif colormap == "delta":
                    cmap = plt.get_cmap("PuOr_r")  # Another diverging option
                else:
                    cmap = plt.get_cmap(colormap)

            # Setup figure
            fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)

            if animation_type == "temporal" and frequency is not None:
                # Temporal animation: Real part oscillating in time (true physical dynamics)
                # Shows how the mode would actually oscillate at this frequency
                log.info(f"Creating temporal animation at {frequency:.3f} GHz")

                # Get mode data
                mode_data = self.get_mode(frequency, z_layer)
                comp_data = mode_data.get_component(component)

                # Get amplitude and phase - this is the complex mode from FFT
                amplitude = np.abs(comp_data)
                phase = np.angle(comp_data)

                # Setup normalization - MidpointNormalize is perfect for oscillating data
                if use_midpoint_norm:
                    vmax = np.max(amplitude)
                    norm = MidpointNormalize(vmin=-vmax, vmax=vmax, midpoint=0)
                else:
                    vmax = np.max(amplitude)
                    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

                # Time steps for one full oscillation period
                time_steps = np.linspace(0, 2 * np.pi, self.config.animation_time_steps)

                # Create initial plot for colorbar setup
                t = time_steps[0]
                real_part = amplitude * np.cos(phase + t)
                im = ax.imshow(
                    real_part,
                    cmap=cmap,
                    norm=norm,
                    extent=mode_data.extent,
                    aspect="equal",
                    origin="lower",
                    interpolation=self.config.interpolation,
                )

                # Create colorbar once
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(f"Magnetization (arb. units)")

                # Set labels once (they won't change)
                ax.set_xlabel("x (nm)")
                ax.set_ylabel("y (nm)")

                def animate_temporal(frame):
                    # Calculate real part at this time step: Re[A * e^(i*φ) * e^(i*ω*t)]
                    # This is exactly: amplitude * cos(phase + ωt) where ωt = time_steps[frame]
                    t = time_steps[frame]
                    real_part = amplitude * np.cos(phase + t)

                    # Update image data instead of recreating
                    im.set_array(real_part)

                    # Show fraction of period completed
                    period_fraction = t / (2 * np.pi)
                    ax.set_title(
                        f"Re[m_{component}] @ {frequency:.3f} GHz (t = {period_fraction:.2f}T)"
                    )

                    return [im]

                # Create animation
                anim = FuncAnimation(
                    fig,
                    animate_temporal,
                    frames=len(time_steps),
                    interval=1000 / fps,
                    blit=True,
                    repeat=True,
                )

            elif animation_type == "frequency" and frequency_range is not None:
                # Frequency sweep animation
                f_min, f_max = frequency_range
                freq_mask = (self.frequencies >= f_min) & (self.frequencies <= f_max)
                freq_indices = np.where(freq_mask)[0]

                if len(freq_indices) == 0:
                    raise ValueError("No frequencies found in specified range")

                log.info(
                    f"Creating frequency sweep animation: {f_min:.3f} - {f_max:.3f} GHz"
                )

                # Pre-calculate all mode data for consistent normalization
                all_amplitudes = []
                for freq_idx in freq_indices:
                    freq = self.frequencies[freq_idx]
                    mode_data = self.get_mode(freq, z_layer)
                    comp_data = mode_data.get_component(component)
                    all_amplitudes.append(np.abs(comp_data))

                # Global normalization
                global_max = np.max([np.max(amp) for amp in all_amplitudes])
                norm = plt.Normalize(vmin=0, vmax=global_max)

                # Create initial plot for colorbar setup
                mode_data = self.get_mode(self.frequencies[freq_indices[0]], z_layer)
                im = ax.imshow(
                    all_amplitudes[0],
                    cmap=cmap,
                    norm=norm,
                    extent=mode_data.extent,
                    aspect="equal",
                    origin="lower",
                    interpolation=self.config.interpolation,
                )

                # Create colorbar once
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(f"|Magnetization| (arb. units)")

                # Set labels once (they won't change)
                ax.set_xlabel("x (nm)")
                ax.set_ylabel("y (nm)")

                def animate_frequency(frame):
                    freq_idx = freq_indices[frame]
                    frequency = self.frequencies[freq_idx]
                    amplitude = all_amplitudes[frame]

                    # Update image data instead of recreating
                    im.set_array(amplitude)

                    ax.set_title(f"|m_{component}| @ {frequency:.3f} GHz")

                    return [im]

                # Create animation
                anim = FuncAnimation(
                    fig,
                    animate_frequency,
                    frames=len(freq_indices),
                    interval=1000 / fps,
                    blit=True,
                    repeat=True,
                )

            elif animation_type == "phase" and frequency is not None:
                # Phase evolution animation
                log.info(f"Creating phase animation at {frequency:.3f} GHz")

                mode_data = self.get_mode(frequency, z_layer)
                comp_data = mode_data.get_component(component)

                amplitude = np.abs(comp_data)
                phase = np.angle(comp_data)

                # Phase steps
                phase_steps = np.linspace(
                    0, 2 * np.pi, self.config.animation_time_steps
                )

                # Create initial plot for colorbar setup
                current_phase = (phase + phase_steps[0]) % (2 * np.pi)
                im = ax.imshow(
                    current_phase,
                    cmap="hsv",  # HSV is perfect for phase
                    vmin=0,
                    vmax=2 * np.pi,
                    extent=mode_data.extent,
                    aspect="equal",
                    origin="lower",
                    interpolation=self.config.interpolation,
                )

                # Create colorbar once
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Phase (rad)")
                cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
                cbar.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"])

                # Set labels once (they won't change)
                ax.set_xlabel("x (nm)")
                ax.set_ylabel("y (nm)")

                def animate_phase(frame):
                    # Add phase offset
                    current_phase = (phase + phase_steps[frame]) % (2 * np.pi)

                    # Update image data instead of recreating
                    im.set_array(current_phase)

                    ax.set_title(
                        f"Phase[m_{component}] @ {frequency:.3f} GHz (φ offset = {phase_steps[frame]:.2f})"
                    )

                    return [im]

                # Create animation
                anim = FuncAnimation(
                    fig,
                    animate_phase,
                    frames=len(phase_steps),
                    interval=1000 / fps,
                    blit=True,
                    repeat=True,
                )

            else:
                raise ValueError(
                    f"Invalid animation_type '{animation_type}' for given parameters"
                )

            # Save animation
            plt.tight_layout()

            # Choose writer based on file extension with fallback support
            if save_path.endswith(".mp4"):
                # Check if ffmpeg is actually available on the system
                if FFMPEG_AVAILABLE and check_ffmpeg_available():
                    writer = "ffmpeg"
                    log.info("Using FFmpeg writer for MP4 format")
                else:
                    log.warning(
                        "FFmpeg not available on system, converting to GIF format"
                    )
                    save_path = save_path.replace(".mp4", ".gif")
                    writer = "pillow"
            elif save_path.endswith(".gif"):
                writer = "pillow"
            else:
                writer = "pillow"  # Default to GIF
                if not save_path.endswith(".gif"):
                    save_path += ".gif"

            log.info(f"Saving animation to {save_path} (this may take a while...)")

            try:
                # For MP4, use higher quality settings
                if writer == "ffmpeg" and save_path.endswith(".mp4"):
                    anim.save(
                        save_path,
                        writer=writer,
                        fps=fps,
                        dpi=self.config.dpi // 2,
                        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
                    )
                else:
                    anim.save(
                        save_path, writer=writer, fps=fps, dpi=self.config.dpi // 2
                    )

                log.info(f"✅ Animation saved successfully!")

            except Exception as save_error:
                log.error(f"Failed to save animation with {writer}: {save_error}")

                # Fallback to GIF if MP4 fails
                if writer == "ffmpeg" and save_path.endswith(".mp4"):
                    log.info("Attempting fallback to GIF format...")
                    fallback_path = save_path.replace(".mp4", ".gif")
                    try:
                        anim.save(
                            fallback_path,
                            writer="pillow",
                            fps=fps,
                            dpi=self.config.dpi // 2,
                        )
                        log.info(f"✅ Animation saved as GIF: {fallback_path}")
                    except Exception as gif_error:
                        log.error(f"Fallback to GIF also failed: {gif_error}")
                        raise save_error
                else:
                    raise save_error

            plt.close(fig)

        except ImportError as e:
            log.error(f"Animation requires additional packages: {e}")
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

    def __repr__(self) -> str:
        """Rich representation of the FFT mode interface."""
        try:
            from rich.console import Console
            from rich.text import Text

            RICH_AVAILABLE = True
        except ImportError:
            RICH_AVAILABLE = False

        if (
            RICH_AVAILABLE
            and self.parent_fft.mmpp
            and getattr(self.parent_fft.mmpp, "_interactive_mode", False)
        ):
            return self._rich_modes_display()
        else:
            return self._basic_modes_display()

    def _rich_modes_display(self) -> str:
        """Generate rich display for FFT modes interface."""
        try:
            from rich.console import Console
            from rich.text import Text
            from rich.panel import Panel
            from rich.columns import Columns

            console = Console()

            summary_text = Text()
            summary_text.append("🎯 MMPP FFT Mode Analyzer\n", style="bold cyan")
            summary_text.append(
                f"📁 Dataset: {getattr(self.mode_analyzer, 'dataset_name', 'Not loaded')}\n",
                style="dim",
            )
            summary_text.append(
                f"🌊 Modes available: {'Yes' if getattr(self.mode_analyzer, 'modes_available', False) else 'No'}\n",
                style="dim",
            )
            summary_text.append(
                f"📊 Z-layers: {getattr(self.mode_analyzer, 'n_z_layers', 'Unknown')}\n",
                style="dim",
            )

            methods_text = Text()
            methods_text.append("🔧 Available methods:\n", style="bold yellow")
            methods = [
                (
                    "interactive_spectrum(dset=None, **kwargs)",
                    "Interactive spectrum with modes",
                ),
                (
                    "plot_modes(frequency, dset=None, **kwargs)",
                    "Plot mode at specific frequency",
                ),
                ("save_modes_animation(**kwargs)", "Create mode animations"),
                ("compute_modes(dset=None, **kwargs)", "Compute/recompute modes"),
                ("[freq_index].plot_modes(**kwargs)", "Plot modes at frequency index"),
            ]

            for method, description in methods:
                methods_text.append("  • ", style="dim")
                methods_text.append(method, style="code")
                methods_text.append(f" - {description}\n", style="dim")

            examples_text = Text()
            examples_text.append("💡 Usage examples:\n", style="bold green")
            examples = [
                "modes.interactive_spectrum(dset='m_z11')",
                "modes.plot_modes(frequency=1.5, dset='m_z11')",
                "modes.save_modes_animation(frequency=1.5, animation_type='temporal')",
                "modes[0][150].plot_modes()  # freq index 0, freq point 150",
                "modes.compute_modes(dset='m_z5-8')",
            ]

            for example in examples:
                examples_text.append(f"  {example}\n", style="code")

            try:
                with console.capture() as capture:
                    console.print(
                        Panel.fit(
                            summary_text,
                            title="[bold blue]MMPP FFT Modes[/bold blue]",
                            border_style="blue",
                        )
                    )
                    console.print("")
                    console.print(
                        Columns(
                            [
                                Panel.fit(
                                    methods_text,
                                    title="[bold yellow]Methods[/bold yellow]",
                                    border_style="yellow",
                                ),
                                Panel.fit(
                                    examples_text,
                                    title="[bold green]Examples[/bold green]",
                                    border_style="green",
                                ),
                            ]
                        )
                    )
                return capture.get()
            except Exception:
                pass

            return (
                str(summary_text) + "\n" + str(methods_text) + "\n" + str(examples_text)
            )

        except Exception:
            return self._basic_modes_display()

    def _basic_modes_display(self) -> str:
        """Generate basic display for FFT modes interface."""
        return f"""
MMPP FFT Mode Analyzer:
======================
🎯 Advanced FMR mode visualization and analysis
📁 Dataset: {getattr(self.mode_analyzer, 'dataset_name', 'Not loaded')}
🌊 Modes available: {'Yes' if getattr(self.mode_analyzer, 'modes_available', False) else 'No'}
📊 Z-layers: {getattr(self.mode_analyzer, 'n_z_layers', 'Unknown')}

🔧 Main methods:
  • interactive_spectrum(dset=None, **kwargs) - Interactive spectrum with modes
  • plot_modes(frequency, dset=None, **kwargs) - Plot mode at specific frequency  
  • save_modes_animation(**kwargs) - Create mode animations
  • compute_modes(dset=None, **kwargs) - Compute/recompute modes
  • [freq_index].plot_modes(**kwargs) - Plot modes at frequency index

💡 Animation examples:
  • modes.save_modes_animation(frequency=1.5, animation_type='temporal')
  • modes.save_modes_animation(frequency_range=(1.0, 3.0), animation_type='frequency')
  
🎬 Animation types: 'temporal', 'frequency', 'phase'
🎨 Supports MP4 (ffmpeg) and GIF (pillow) output formats
"""

    @property
    def mode_analyzer(self) -> FMRModeAnalyzer:
        """Get or create mode analyzer (lazy initialization)."""
        if self._mode_analyzer is None:
            # Get zarr path from parent FFT
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            self._mode_analyzer = FMRModeAnalyzer(zarr_path, debug=debug_mode)

        return self._mode_analyzer

    def interactive_spectrum(self, dset: str = None, **kwargs) -> Figure:
        """Create interactive spectrum plot."""
        # If dset is specified, create a new analyzer for that dataset
        if dset is not None and dset != self.mode_analyzer.dataset_name:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode
            )

            # Check if modes exist, if not compute them
            if not temp_analyzer.modes_available:
                log.info(f"Computing modes for dataset '{dset}'...")
                temp_analyzer.compute_modes(save=True)

            return temp_analyzer.interactive_spectrum(**kwargs)
        else:
            # Use default analyzer
            if not self.mode_analyzer.modes_available:
                log.info(
                    f"Computing modes for dataset '{self.mode_analyzer.dataset_name}'..."
                )
                self.mode_analyzer.compute_modes(save=True)

            return self.mode_analyzer.interactive_spectrum(**kwargs)

    def compute_modes(self, dset: str = None, **kwargs) -> None:
        """Compute modes for specified dataset."""
        if dset is not None:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode
            )
            temp_analyzer.compute_modes(**kwargs)
        else:
            self.mode_analyzer.compute_modes(**kwargs)

    def plot_modes(
        self, frequency: float, dset: str = None, **kwargs
    ) -> Tuple[Figure, np.ndarray]:
        """Plot modes at specified frequency."""
        # If dset is specified, create a new analyzer for that dataset
        if dset is not None and dset != self.mode_analyzer.dataset_name:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode
            )

            # Check if modes exist, if not compute them
            if not temp_analyzer.modes_available:
                log.info(f"Computing modes for dataset '{dset}'...")
                temp_analyzer.compute_modes(save=True)

            return temp_analyzer.plot_modes(frequency, **kwargs)
        else:
            # Use default analyzer
            if not self.mode_analyzer.modes_available:
                log.info(
                    f"Computing modes for dataset '{self.mode_analyzer.dataset_name}'..."
                )
                self.mode_analyzer.compute_modes(save=True)

            return self.mode_analyzer.plot_modes(frequency, **kwargs)

    def save_modes_animation(
        self,
        frequency_range: Tuple[float, float] = None,
        frequency: float = None,
        save_path: str = "mode_animation.gif",
        dset: str = None,
        fps: int = 15,
        z_layer: int = 0,
        component: Union[str, int] = "z",
        animation_type: str = "temporal",
        **kwargs,
    ) -> None:
        """
        Save animation of FMR modes.

        Parameters:
        -----------
        frequency_range : tuple, optional
            (f_min, f_max) in GHz for frequency sweep animation
        frequency : float, optional
            Single frequency for temporal animation (in GHz)
        save_path : str
            Output file path (.gif or .mp4)
        dset : str, optional
            Dataset name. If None, uses default analyzer
        fps : int
            Frames per second (default: 15)
        z_layer : int
            Z-layer to animate (default: 0)
        component : str or int
            Component to animate (default: 'z')
        animation_type : str
            Type of animation ('temporal', 'frequency', 'phase')
        **kwargs
            Additional arguments passed to FMRModeAnalyzer.save_modes_animation
        """
        # If dset is specified, create a new analyzer for that dataset
        if dset is not None and dset != self.mode_analyzer.dataset_name:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode
            )

            # Check if modes exist, if not compute them
            if not temp_analyzer.modes_available:
                log.info(f"Computing modes for dataset '{dset}'...")
                temp_analyzer.compute_modes(save=True)

            return temp_analyzer.save_modes_animation(
                frequency_range=frequency_range,
                frequency=frequency,
                save_path=save_path,
                fps=fps,
                z_layer=z_layer,
                component=component,
                animation_type=animation_type,
                **kwargs,
            )
        else:
            # Use default analyzer
            if not self.mode_analyzer.modes_available:
                log.info(
                    f"Computing modes for dataset '{self.mode_analyzer.dataset_name}'..."
                )
                self.mode_analyzer.compute_modes(save=True)

            return self.mode_analyzer.save_modes_animation(
                frequency_range=frequency_range,
                frequency=frequency,
                save_path=save_path,
                fps=fps,
                z_layer=z_layer,
                component=component,
                animation_type=animation_type,
                **kwargs,
            )


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

    def __repr__(self) -> str:
        """Rich string representation of FrequencyModeInterface."""
        try:
            # Try rich display first
            return self._rich_frequency_display()
        except ImportError:
            # Fallback to basic display
            return self._basic_frequency_display()

    def _rich_frequency_display(self) -> str:
        """Rich display with styling and detailed information."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich.table import Table
        from rich.syntax import Syntax
        import io

        console = Console(file=io.StringIO(), width=100, force_terminal=True)

        # Main header
        header = Text("FrequencyModeInterface", style="bold cyan")

        # Frequency information table
        freq_table = Table(show_header=False, box=None, padding=(0, 1))
        freq_table.add_column("Property", style="bold yellow")
        freq_table.add_column("Value", style="white")

        freq_table.add_row("🎯 Frequency Index", f"{self.frequency_index}")
        freq_table.add_row("⚡ Frequency Value", f"{self.frequency:.2e} Hz")
        freq_table.add_row(
            "📊 Parent Modes",
            f"{len(self.parent.mode_analyzer.frequencies)} frequencies",
        )

        # Available methods
        methods_text = Text("Available Methods:", style="bold green")
        methods_list = [
            "• plot_modes(**kwargs) → Tuple[Figure, np.ndarray]",
            "• get_mode(**kwargs) → FMRModeData",
            "• frequency → float (property)",
        ]
        methods_content = "\n".join(methods_list)

        # Usage examples
        example_code = f"""# Access frequency-specific operations
freq_interface = modes[{self.frequency_index}]

# Plot modes at this frequency
fig, axes = freq_interface.plot_modes()

# Get mode data
mode_data = freq_interface.get_mode()

# Check frequency value
print(f"Frequency: {{freq_interface.frequency:.2e}} Hz")"""

        syntax = Syntax(
            example_code, "python", theme="monokai", background_color="default"
        )

        # Build the panel content
        content_parts = [
            freq_table,
            "",
            methods_text,
            Text(methods_content),
            "",
            Text("Usage Examples:", style="bold blue"),
            syntax,
        ]

        panel = Panel(
            "\n".join(str(part) for part in content_parts),
            title=str(header),
            border_style="cyan",
            width=98,
        )

        console.print(panel)
        return console.file.getvalue()

    def _basic_frequency_display(self) -> str:
        """Basic fallback display without rich formatting."""
        return (
            f"FrequencyModeInterface(frequency_index={self.frequency_index}, "
            f"frequency={self.frequency:.2e} Hz)\n"
            f"Methods: plot_modes(), get_mode(), frequency (property)\n"
            f"Parent analyzer has {len(self.parent.mode_analyzer.frequencies)} frequencies"
        )
