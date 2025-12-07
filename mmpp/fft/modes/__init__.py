"""
FMR Mode Visualization Module

Professional implementation for visualizing FMR modes with interactive spectrum.
Provides both programmatic and interactive interfaces for mode analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

import math

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Import shared logging configuration
from ...cli.logging_config import get_mmpp_logger, setup_mmpp_logging

# Get logger for FMR modes
log = get_mmpp_logger("mmpp.fft.modes")

# Import electromagnetic analysis module

from .ffmpeg_utils import (
    _create_ffmpeg_writer,
    _ensure_ffmpeg_available,
    check_ffmpeg_available,
    check_ffmpeg_installation,
    install_ffmpeg,
    install_ffmpeg_simple,
)
from .compat import (
    ANIMATION_AVAILABLE,
    AXES_GRID_AVAILABLE,
    CMCRAMERI_AVAILABLE,
    CMOCEAN_AVAILABLE,
    FFMPEG_AVAILABLE,
    MATPLOTLIB_AVAILABLE,
    PYZFN_AVAILABLE,
    SCIPY_AVAILABLE,
    ZARR_AVAILABLE,
    AnchoredSizeBar,
    Axes,
    Figure,
    FuncAnimation,
    MouseEvent,
    PillowWriter,
    Pyzfn,
    cmocean,
    cmc,
    find_peaks,
    zarr,
)
from .styling import STYLING_AVAILABLE, MidpointNormalize, setup_animation_styling

# Import refactored utilities
from .utils.peak_detection import detect_peaks
from .utils.scalebar import calculate_optimal_length, format_scalebar_label
from .analyzer.cache import ModeCache

# Import animation functions
from .visualization.animation import (
    save_modes_animation as _save_modes_animation,
    toggle_mode_animation as _toggle_mode_animation,
    stop_mode_animation as _stop_mode_animation,
    save_animated_view as _save_animated_view,
    start_mode_animation as _start_mode_animation,
)

# Import static plotting functions
from .visualization.static_plots import (
    plot_modes as _plot_modes,
    update_single_mode_plot as _update_single_mode_plot,
)

# Import interactive spectrum functions
from .visualization.interactive import (
    interactive_spectrum as _interactive_spectrum,
    update_mode_plots as _update_mode_plots,
)

# Import mode analysis functions
from .analyzer.mode_analysis import (
    characterize_mode as _characterize_mode,
    characterize_vortex_mode as _characterize_vortex_mode,
    print_characterization_details as _print_characterization_details,
)

from ..mode_characterization import (
    ModeCharacterAnalyzer,
    ModeCharacteristicConfig,
    ModeCharacterizationResult,
)
from ..metrics import (
    PeakWidth,
    compute_half_width_at_half_max,
    format_width_value,
    normalize_peak_width_option,
)


# FFmpeg installation utilities
@dataclass
class ModeVisualizationConfig:
    """Configuration for mode visualization."""

    # Figure settings
    figsize: tuple[float, float] = (16, 10)
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

    # Publication-style annotations
    show_scalebar: bool = True
    scalebar_length_nm: Optional[float] = None  # Auto-computed when None
    scalebar_location: str = "lower right"
    scalebar_pad: float = 0.3
    scalebar_color: str = "white"
    scalebar_fontsize: int = 9
    scalebar_frame: bool = False
    scalebar_height_fraction: float = 0.01
    scale_units: str = "nm"

    colorbar_fraction: float = 0.04  # Proper colorbar width
    colorbar_pad: float = 0.01  # Small padding for close positioning
    colorbar_ticklabel_size: int = 9  # Larger tick labels
    colorbar_label_size: int = 10  # Larger labels
    colorbar_labels: dict[str, str] = field(
        default_factory=lambda: {
            "magnitude": "Magnetization |m|",
            "phase": "Phase (rad)",
            "combined": "Re(m) × cos(φ)",
        }
    )

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

        if self.scalebar_length_nm is not None and self.scalebar_length_nm <= 0:
            raise ValueError("scalebar_length_nm must be positive when provided")

        if self.scalebar_height_fraction <= 0 or self.scalebar_height_fraction > 0.1:
            raise ValueError(
                "scalebar_height_fraction should be within (0, 0.1] for sensible display"
            )

        if self.colorbar_fraction <= 0 or self.colorbar_pad < 0:
            raise ValueError("colorbar_fraction must be > 0 and colorbar_pad >= 0")

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
        extent: Optional[tuple[float, float, float, float]] = None,
        metadata: Optional[dict[str, Any]] = None,
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
        dataset_name: Optional[str] = None,
        config: Optional[ModeVisualizationConfig] = None,
        mode_character_config: Optional[ModeCharacteristicConfig] = None,
        debug: bool = False,
        log_level: Optional[Union[str, int]] = None,
    ):
        """
        Initialize FMR mode analyzer.

        Parameters:
        -----------
        zarr_path : str
            Path to zarr file containing mode data
        dataset_name : str, optional
            Base dataset name (default: auto-select largest m dataset)
        config : ModeVisualizationConfig, optional
            Visualization configuration
        mode_character_config : ModeCharacteristicConfig, optional
            Mode characterization configuration
        debug : bool, optional
            Enable debug logging (default: False)
        log_level : str or int, optional
            Set specific log level. Can be string ("DEBUG", "INFO", "WARNING", "ERROR")
            or integer constant (logging.DEBUG, logging.INFO, etc.).
            If provided, overrides debug parameter.
            Default: None (uses debug flag - DEBUG if True, INFO if False)
        """
        if not ZARR_AVAILABLE:
            raise ImportError("Zarr is required for mode analysis")

        # Auto-select largest m dataset if none specified
        if dataset_name is None:
            from ...plotting import _find_largest_m_dataset

            dataset_name = _find_largest_m_dataset(zarr_path)

        self.zarr_path = zarr_path
        self.dataset_name = dataset_name
        self.config = config or ModeVisualizationConfig()
        self._character_analyzer = ModeCharacterAnalyzer(mode_character_config)

        # Set up logging with flexible level control
        import logging

        # Convert string log level to integer if needed
        numeric_level = None
        if log_level is not None:
            if isinstance(log_level, str):
                numeric_level = getattr(logging, log_level.upper(), None)
                if numeric_level is None:
                    raise ValueError(
                        f"Invalid log level: {log_level}. Use DEBUG, INFO, WARNING, or ERROR"
                    )
            else:
                numeric_level = log_level

        setup_mmpp_logging(
            debug=debug, logger_name="mmpp.fft.modes", level=numeric_level
        )

        if debug or (numeric_level is not None and numeric_level <= logging.DEBUG):
            log.debug("FMR mode analyzer debug logging enabled")

        # Load data
        self._load_data()

        # Interactive state
        self._current_frequency = None
        self._interactive_fig = None
        self._frequency_line = None
        self._mode_axes = None
        self._row_colorbars: list[Any] = []
        self._fwhm_artists: list[Any] = []
        self._last_fwhm = None

        # Animation state tracking
        self._mode_animations = {}  # Dict to track active animations per axis
        self._animated_axes = set()  # Set of axes currently being animated

        # Mode data cache using refactored ModeCache
        self._mode_cache = ModeCache(maxsize=128)

    @property
    def modes_available(self) -> bool:
        """Check if mode data is available."""
        return (
            self.modes_path is not None
            and self.freqs_path is not None
            and self.spectrum_path is not None
        )

    @property
    def last_fwhm(self) -> Optional[PeakWidth]:
        """Return the most recently computed half-width at half-maximum."""

        return getattr(self, "_last_fwhm", None)

    def _list_available_datasets(self) -> list[str]:
        """Enumerate top-level datasets available in the zarr archive."""

        try:
            root = zarr.open(self.zarr_path, mode="r")
            keys = set(root.group_keys()) | set(root.array_keys())
            return sorted({key.split("/")[0] for key in keys})
        except Exception as exc:
            log.debug("Unable to list datasets in %s: %s", self.zarr_path, exc)
            return []

    def _get_zarr_paths(self) -> tuple[Optional[str], Optional[str], Optional[str]]:
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

        # Find spectrum - try multiple locations for consistency with plot_spectrum
        spectrum_path = None
        spectrum_candidates = [
            # Standard FFT locations (consistent with plot_spectrum)
            f"fft/{self.dataset_name}_z-1_m1/spectrum",  # Most common case
            f"fft/{self.dataset_name}_z0_m1/spectrum",
            f"fft/{self.dataset_name}/spectrum",
            # Legacy locations (from compute_modes)
            f"fft/{self.dataset_name}/spec",
            f"fft/{self.dataset_name}/sum",
            # Try other z_layers and methods
            *[f"fft/{self.dataset_name}_z{z}_m1/spectrum" for z in range(-5, 10)],
        ]

        for path in spectrum_candidates:
            if path in self.zarr_file:
                spectrum_path = path
                log.debug(f"Found spectrum at: {spectrum_path}")
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

        # Load spectrum - prioritize fresh modes data over potentially stale FFT data
        self.spectrum = None

        # First try modes data (most up-to-date)
        modes_power_sum_path = f"modes/{self.dataset_name}/power_sum"
        modes_power_max_path = f"modes/{self.dataset_name}/power_max"

        log.debug(f"Looking for fresh modes spectrum at: {modes_power_sum_path}")
        if modes_power_sum_path in self.zarr_file:
            self.spectrum = np.array(self.zarr_file[modes_power_sum_path])
            if np.iscomplexobj(self.spectrum):
                self.spectrum = np.abs(self.spectrum)
            log.info(
                f"Using fresh modes power_sum as spectrum: shape {self.spectrum.shape}"
            )
        elif modes_power_max_path in self.zarr_file:
            log.debug(
                f"power_sum not found, trying power_max at: {modes_power_max_path}"
            )
            self.spectrum = np.array(self.zarr_file[modes_power_max_path])
            if np.iscomplexobj(self.spectrum):
                self.spectrum = np.abs(self.spectrum)
            log.info(
                f"Using fresh modes power_max as spectrum: shape {self.spectrum.shape}"
            )
        elif self.spectrum_path:
            # Fallback to FFT spectrum (may be stale)
            log.warning(
                f"No fresh modes spectrum found, falling back to FFT spectrum (may be outdated)"
            )
            self.spectrum = np.array(self.zarr_file[self.spectrum_path])
            if self.spectrum.ndim > 1:
                # Take first component if multi-component
                self.spectrum = (
                    self.spectrum[:, 0]
                    if self.spectrum.shape[1] == 3
                    else np.sum(self.spectrum, axis=tuple(range(1, self.spectrum.ndim)))
                )
            if np.iscomplexobj(self.spectrum):
                self.spectrum = np.abs(self.spectrum)
            log.info(f"Loaded FFT spectrum data: shape {self.spectrum.shape}")
        else:
            log.error(f"No spectrum data found - neither modes nor FFT data available")
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
    ) -> list[Peak]:
        """
        Detect peaks in spectrum using refactored utilities.

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
        # Use refactored peak detection
        return detect_peaks(
            spectrum=spectrum,
            frequencies=frequencies,
            threshold=self.config.peak_threshold,
            min_distance=self.config.peak_min_distance,
            use_scipy=True,
        )

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

        # Validate and normalize z_layer bounds
        mode_shape = self.zarr_file[self.modes_path].shape
        n_layers = mode_shape[1]

        # Handle negative indexing (like Python lists)
        if z_layer < 0:
            z_layer = n_layers + z_layer
            log.debug(f"Converted negative z_layer to {z_layer}")

        if z_layer < 0 or z_layer >= n_layers:
            raise ValueError(
                f"z_layer {z_layer} out of range. Available layers: 0-{n_layers - 1} (or negative: -{n_layers} to -1)"
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

    def characterize_mode(
        self,
        frequency: float,
        z_layer: int = 0,
        *,
        core_position: Optional[tuple[float, float]] = None,
        analysis_radius: Optional[float] = None,
        config: Optional[ModeCharacteristicConfig] = None,
        verbose: bool = False,
    ) -> ModeCharacterizationResult:
        """Classify the mode at ``frequency`` into gyration/breathing/azimuthal families - see analyzer.mode_analysis for details."""
        return _characterize_mode(
            self, frequency, z_layer,
            core_position=core_position,
            analysis_radius=analysis_radius,
            config=config,
            verbose=verbose,
        )

    def characterize_vortex_mode(
        self,
        frequency: float,
        z_layer: int = 0,
        *,
        core_position: Optional[tuple[float, float]] = None,
        R_dot: Optional[float] = None,
        config: Optional[ModeCharacteristicConfig] = None,
        verbose: bool = False,
    ) -> "VortexModeResult":
        """Advanced vortex/skyrmion mode classification - see analyzer.mode_analysis for details."""
        return _characterize_vortex_mode(
            self, frequency, z_layer,
            core_position=core_position,
            R_dot=R_dot,
            config=config,
            verbose=verbose,
        )

    def _print_characterization_details(
        self, result: ModeCharacterizationResult, frequency: float, z_layer: int
    ) -> None:
        """Print detailed characterization analysis results."""
        _print_characterization_details(self, result, frequency, z_layer)

    def _update_cache(
        self, frequency: float, z_layer: int, mode_data: FMRModeData
    ) -> None:
        """Update mode data cache using refactored ModeCache."""
        self._mode_cache.put(frequency, z_layer, mode_data)

    def find_peaks(
        self,
        threshold: Optional[float] = None,
        min_distance: Optional[int] = None,
        component: int = 0,
        spectrum: Optional[np.ndarray] = None,
        frequencies: Optional[np.ndarray] = None,
    ) -> list[Peak]:
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
        spectrum : np.ndarray, optional
            Spectrum data to use (default: self.spectrum)
        frequencies : np.ndarray, optional
            Frequency data to use (default: self.frequencies)

        Returns:
        --------
        List[Peak]
            List of detected peaks
        """
        # Use provided spectrum/frequencies or fallback to instance data
        spectrum_data = spectrum if spectrum is not None else self.spectrum
        freq_data = frequencies if frequencies is not None else self.frequencies

        if spectrum_data is None:
            log.warning("No spectrum data available for peak detection")
            return []

        threshold = threshold or self.config.peak_threshold
        min_distance = min_distance or self.config.peak_min_distance

        # Normalize spectrum for peak detection
        spectrum_work = spectrum_data.copy()
        if self.config.spectrum_normalize:
            spectrum_work = spectrum_work / np.max(spectrum_work)

        # Filter frequency range
        freq_mask = (freq_data >= self.config.f_min) & (freq_data <= self.config.f_max)
        freqs_filtered = freq_data[freq_mask]
        spectrum_filtered = spectrum_work[freq_mask]

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
        components: Optional[list[Union[int, str]]] = None,
        save_path: Optional[str] = None,
    ) -> tuple[Figure, np.ndarray]:
        """Plot mode visualization for a specific frequency - see visualization.static_plots.plot_modes for details."""
        return _plot_modes(self, frequency, z_layer, components, save_path)

    def interactive_spectrum(
        self,
        components: Optional[list[Union[int, str]]] = None,
        z_layer: int = 0,
        method: int = 1,
        show: bool = True,
        force: bool = False,
        use_fft_spectrum: bool = True,
        saveanim: Union[bool, str, None] = None,
        auto_animate: bool = False,
        auto_save: bool = False,
        spectrum_result: Any = None,  # NEW: Inject spectrum from FFT.spectrum()
        **kwargs,
    ) -> Figure:
        """Create interactive spectrum plot with mode visualization - see visualization.interactive for details."""
        return _interactive_spectrum(
            self,
            components=components,
            z_layer=z_layer,
            method=method,
            show=show,
            force=force,
            use_fft_spectrum=use_fft_spectrum,
            saveanim=saveanim,
            auto_animate=auto_animate,
            auto_save=auto_save,
            spectrum_result=spectrum_result,  # Pass injected spectrum
            **kwargs,
        )

    def _update_mode_plots(
        self, components: list[Union[int, str]], z_layer: int
    ) -> None:
        """Update mode plots for current frequency."""
        _update_mode_plots(self, components, z_layer)
    
    # Alias for backward compatibility
    interactive_spectrum_old = interactive_spectrum

    def _toggle_mode_animation(
        self,
        ax: Any,
        row_idx: int,
        col_idx: int,
        component: Union[str, int],
        z_layer: int,
    ) -> None:
        """Toggle between static mode plot and in-place animation."""
        _toggle_mode_animation(self, ax, row_idx, col_idx, component, z_layer)

    def _stop_mode_animation(self, axis_key: tuple[int, int]) -> None:
        """Stop animation for specific axis."""
        _stop_mode_animation(self, axis_key)

    def _save_animated_view(self, save_path: str, z_layer: int = 0) -> None:
        """Save current animated view to video file."""
        _save_animated_view(self, save_path, z_layer)

    def _start_mode_animation(
        self,
        ax: Any,
        row_idx: int,
        col_idx: int,
        component: Union[str, int],
        z_layer: int,
    ) -> None:
        """Start in-place animation for specific mode axis."""
        _start_mode_animation(self, ax, row_idx, col_idx, component, z_layer)

    def _update_single_mode_plot(
        self,
        ax: Any,
        row_idx: int,
        col_idx: int,
        component: Union[str, int],
        z_layer: int,
    ) -> None:
        """Update single mode plot (used when stopping animation)."""
        _update_single_mode_plot(self, ax, row_idx, col_idx, component, z_layer)

    def _add_scale_bar(
        self, ax: Any, extent: tuple[float, float, float, float]
    ) -> None:
        """Add a publication-style scale bar to the supplied axis."""
        _add_scale_bar(self, ax, extent)

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
            available = self._list_available_datasets()
            suggestion = (
                f" Available datasets: {', '.join(available)}" if available else ""
            )
            raise ValueError(
                f"Dataset '{self.dataset_name}' not found in zarr file '{self.zarr_path}'.{suggestion}"
            )

        dset = self.zarr_file[self.dataset_name]

        # Determine sampling interval dt
        dt: Optional[float] = None
        t_array: Optional[np.ndarray] = None
        try:
            raw_t = dset.attrs["t"][:]
            t_array = np.asarray(raw_t, dtype=float)
            if t_array.size > 1:
                diffs = np.diff(t_array)
                positive_diffs = diffs[diffs > 0]
                if positive_diffs.size:
                    dt = float(np.mean(positive_diffs))
                else:
                    dt = float(np.median(np.abs(diffs)))
                if not np.isfinite(dt) or dt <= 0:
                    raise ValueError("Invalid timestep derived from t attribute")
            else:
                raise ValueError("Insufficient time samples in attribute")
        except Exception as exc:
            log.debug(
                "Falling back to alternative dt sources for dataset %s: %s",
                self.dataset_name,
                exc,
            )
            t_array = None
            dt = None

        def _extract_dt(candidate: Any) -> Optional[float]:
            if candidate is None:
                return None
            try:
                value = float(np.asarray(candidate).item())
                if np.isfinite(value) and value > 0:
                    return value
            except Exception:
                return None
            return None

        if dt is None:
            for attrs in (
                getattr(dset, "attrs", {}),
                getattr(self.zarr_file, "attrs", {}),
            ):
                for key in ("t_sampl", "dt"):
                    dt_candidate = _extract_dt(attrs.get(key))
                    if dt_candidate is not None:
                        dt = dt_candidate
                        break
                if dt is not None:
                    break

        if dt is None and PYZFN_AVAILABLE:
            try:
                pyz_job = Pyzfn(self.zarr_path)
                dt_candidate = _extract_dt(pyz_job.attrs.get("t_sampl", None))
                if dt_candidate is not None:
                    dt = dt_candidate
            except Exception as exc:
                log.debug("Could not retrieve t_sampl via Pyzfn: %s", exc)

        if dt is None:
            dt = 1e-12
            log.warning(
                "Falling back to default timestep 1e-12 s for dataset %s."
                " Check zarr metadata for t or t_sampl attributes.",
                self.dataset_name,
            )

        if t_array is None:
            num_samples = dset.shape[0]
            t_array = np.arange(num_samples, dtype=float) * dt
        else:
            num_samples = t_array.size

        # Calculate frequencies using number of time samples
        if num_samples < 2:
            raise ValueError("Mode computation requires at least two time samples")

        freqs = np.fft.rfftfreq(num_samples, dt) * 1e-9  # Convert to GHz

        # Load and process data
        log.info(f"Loading magnetization data: {dset.shape}")
        arr = np.asarray(dset[:, z_slice])
        log.info("Loading magnetization data finished")

        # Remove DC component
        arr = arr - arr.mean(axis=0)[None, ...]

        # Apply window function
        if window:
            window_func = np.hanning(arr.shape[0])
            for _i in range(arr.ndim - 1):
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

            # Remove existing data if force=True to avoid conflicts
            if force:
                if f"modes/{self.dataset_name}" in zarr_write:
                    del zarr_write[f"modes/{self.dataset_name}"]
                if f"fft/{self.dataset_name}" in zarr_write:
                    del zarr_write[f"fft/{self.dataset_name}"]

            # Create groups
            modes_group = zarr_write.require_group(f"modes/{self.dataset_name}")
            # Don't create fft_group here - let plot_spectrum/calculate_fft_data handle it
            # This avoids conflicts with standard FFT data format

            # Save frequencies in modes group only
            modes_group.array(
                "freqs",
                data=freqs,
                shape=freqs.shape,
                dtype=freqs.dtype,
                chunks=freqs.shape,  # Use the data shape as chunk size
                overwrite=True,
            )

            # Save complex modes (chunked only on first dimension)
            chunks = (1,) + fft_result.shape[1:]
            modes_group.array(
                "arr",
                data=fft_result.astype(np.complex64, copy=False),
                shape=fft_result.shape,
                dtype=np.complex64,
                chunks=chunks,
                overwrite=True,
            )

            # Save power spectrum summary in modes group
            power_spec = np.abs(fft_result)
            reduction_axes = (
                tuple(range(1, power_spec.ndim)) if power_spec.ndim > 1 else None
            )
            power_max = (
                np.max(power_spec, axis=reduction_axes)
                if reduction_axes
                else np.max(power_spec, keepdims=False)
            )
            power_sum = (
                np.sum(power_spec, axis=reduction_axes)
                if reduction_axes
                else np.sum(power_spec, keepdims=False)
            )
            modes_group.array(
                "power_max",
                data=power_max.astype(np.float32, copy=False),
                shape=power_max.shape,
                dtype=np.float32,
                chunks=power_max.shape,  # Use data shape as chunk size
                overwrite=True,
            )
            modes_group.array(
                "power_sum",
                data=power_sum.astype(np.float32, copy=False),
                shape=power_sum.shape,
                dtype=np.float32,
                chunks=power_sum.shape,  # Use data shape as chunk size
                overwrite=True,
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
        frequency_range: tuple[float, float] = None,
        frequency: float = None,
        save_path: str = "mode_animation.gif",
        fps: int = 15,
        z_layer: int = 0,
        component: Union[str, int] = "z",
        animation_type: str = "temporal",
        colormap: str = None,
        use_midpoint_norm: bool = None,
        figsize: tuple[float, float] = None,
    ) -> None:
        """Save animation of FMR modes - see visualization.animation.save_modes_animation for details."""
        _save_modes_animation(
            self,
            frequency_range=frequency_range,
            frequency=frequency,
            save_path=save_path,
            fps=fps,
            z_layer=z_layer,
            component=component,
            animation_type=animation_type,
            colormap=colormap,
            use_midpoint_norm=use_midpoint_norm,
            figsize=figsize,
        )

    def install_ffmpeg(self) -> Optional[str]:
        """
        Install FFmpeg for MP4 animation support.

        This method ensures FFmpeg is available for high-quality video
        animation export. If FFmpeg is not found on the system, it will
        be automatically downloaded and installed.

        Returns:
        --------
        str or None
            Path to ffmpeg executable if successful, None if failed

        Examples:
        ---------
        >>> analyzer = FMRModeAnalyzer(zarr_file, dataset_name)
        >>> ffmpeg_path = analyzer.install_ffmpeg()
        """
        log.info("🔧 Installing FFmpeg for animation support...")
        return install_ffmpeg(force=False, verbose=True)


class FFTModeInterface:
    """
    Enhanced FFT interface with mode visualization capabilities.

    Provides elegant syntax like: job[0].fft[0][200].plot_modes()
    """

    def __init__(self, fft_result_index: int, parent_fft):
        """Initialize mode interface for specific FFT result."""
        self.fft_result_index = fft_result_index
        self.parent_fft = parent_fft
        self._mode_analyzer = None

    def __getitem__(self, frequency_index: int) -> "FrequencyModeInterface":
        """Get mode interface for specific frequency index."""
        return FrequencyModeInterface(frequency_index, self)

    def characterize_mode(
        self,
        frequency: float,
        verbose: bool = False,
        **kwargs,
    ) -> ModeCharacterizationResult:
        """
        Characterize the mode at a given frequency (GHz).

        Parameters:
        -----------
        frequency : float
            Frequency to analyze [GHz]
        verbose : bool, optional
            Show detailed calculation results and classification criteria (default: False)
        **kwargs
            Additional arguments passed to mode analyzer

        Returns:
        --------
        ModeCharacterizationResult
            Classification result with detailed metrics
        """

        return self.mode_analyzer.characterize_mode(
            frequency, verbose=verbose, **kwargs
        )

    def __repr__(self) -> str:
        """Rich representation of the FFT mode interface."""
        try:
            from rich.console import Console
            from rich.text import Text
            return self._rich_modes_display()
        except ImportError:
            return self._basic_modes_display()

    def _rich_modes_display(self) -> str:
        """Generate rich display for FFT modes interface."""
        try:
            from rich.columns import Columns
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text

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
                (
                    "characterize_mode(frequency, **kwargs)",
                    "Return structured mode classification",
                ),
                ("save_modes_animation(**kwargs)", "Create mode animations"),
                ("install_ffmpeg()", "Install FFmpeg for MP4 animations"),
                ("compute_modes(dset=None, **kwargs)", "Compute/recompute modes"),
                ("[freq_index].plot_modes(**kwargs)", "Plot modes at frequency index"),
                (
                    "[freq_index].characterize(**kwargs)",
                    "Get mode labels at frequency index",
                ),
            ]

            for method, description in methods:
                methods_text.append("  • ", style="dim")
                methods_text.append(method, style="code")
                methods_text.append(f" - {description}\n", style="dim")

            examples_text = Text()
            examples_text.append("💡 Usage examples:\n", style="bold green")
            examples = [
                "modes.interactive_spectrum(dset='m')",
                "modes.plot_modes(frequency=1.5, dset='m')",
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
📁 Dataset: {getattr(self.mode_analyzer, "dataset_name", "Not loaded")}
🌊 Modes available: {"Yes" if getattr(self.mode_analyzer, "modes_available", False) else "No"}
📊 Z-layers: {getattr(self.mode_analyzer, "n_z_layers", "Unknown")}

🔧 Main methods:
  • interactive_spectrum(dset=None, **kwargs) - Interactive spectrum with modes
  • plot_modes(frequency, dset=None, **kwargs) - Plot mode at specific frequency
  • characterize_mode(frequency, **kwargs) - Structured mode classification
  • save_modes_animation(**kwargs) - Create mode animations
  • compute_modes(dset=None, **kwargs) - Compute/recompute modes
  • [freq_index].plot_modes(**kwargs) - Plot modes at frequency index
  • [freq_index].characterize(**kwargs) - Analyze mode at frequency index

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
            # Check if parent has log_level attribute
            log_level = (
                getattr(self.parent_fft.mmpp, "log_level", None)
                if self.parent_fft.mmpp
                else None
            )
            
            # Use injected dataset context from DatasetSpecificFFT if available
            dataset_name = getattr(self, "_dataset_context", None)
            
            self._mode_analyzer = FMRModeAnalyzer(
                zarr_path, 
                dataset_name=dataset_name,  # Use context if available, else auto-detect
                debug=debug_mode, 
                log_level=log_level
            )

        return self._mode_analyzer

    def interactive_spectrum_old(
        self, dset: str = None, force: bool = False, **kwargs
    ) -> Figure:
        """Create interactive spectrum plot (ORIGINAL IMPLEMENTATION).
        
        This is the original, full-featured implementation with all capabilities:
        - Click to select frequency
        - Right-click to snap to peak
        - Double-click to toggle animations
        - Press 'c' to characterize mode
        - Press 's' to save animation
        - Press 'h' for help
        
        Use this if the new interactive_spectrum() doesn't work properly.
        """
        # If dset is specified, create a new analyzer for that dataset
        if dset is not None and dset != self.mode_analyzer.dataset_name:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            # Check if parent has log_level attribute
            log_level = (
                getattr(self.parent_fft.mmpp, "log_level", None)
                if self.parent_fft.mmpp
                else None
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode, log_level=log_level
            )

            # Check if modes exist or force recomputation
            if not temp_analyzer.modes_available or force:
                log.info(f"Computing modes for dataset '{dset}' (force={force})...")
                temp_analyzer.compute_modes(save=True, force=force)

            return temp_analyzer.interactive_spectrum(**kwargs)
        else:
            # Use default analyzer
            if not self.mode_analyzer.modes_available or force:
                log.info(
                    f"Computing modes for dataset '{self.mode_analyzer.dataset_name}' (force={force})..."
                )
                self.mode_analyzer.compute_modes(save=True, force=force)

            return self.mode_analyzer.interactive_spectrum(**kwargs)
    
    def interactive_spectrum(
        self, dset: str = None, force: bool = False, **kwargs
    ) -> Figure:
        """Create interactive spectrum plot.
        
        This delegates to the original interactive_spectrum implementation.
        If you experience issues, try interactive_spectrum_old() directly.
        """
        return self.interactive_spectrum_old(dset=dset, force=force, **kwargs)

    def compute_modes(self, dset: str = None, **kwargs) -> None:
        """Compute modes for specified dataset."""
        if dset is not None:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            # Check if parent has log_level attribute
            log_level = (
                getattr(self.parent_fft.mmpp, "log_level", None)
                if self.parent_fft.mmpp
                else None
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode, log_level=log_level
            )
            temp_analyzer.compute_modes(**kwargs)
        else:
            self.mode_analyzer.compute_modes(**kwargs)

    def plot_modes(
        self, frequency: float, dset: str = None, **kwargs
    ) -> tuple[Figure, np.ndarray]:
        """Plot modes at specified frequency."""
        # If dset is specified, create a new analyzer for that dataset
        if dset is not None and dset != self.mode_analyzer.dataset_name:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            # Check if parent has log_level attribute
            log_level = (
                getattr(self.parent_fft.mmpp, "log_level", None)
                if self.parent_fft.mmpp
                else None
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode, log_level=log_level
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
        frequency_range: tuple[float, float] = None,
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
            # Check if parent has log_level attribute
            log_level = (
                getattr(self.parent_fft.mmpp, "log_level", None)
                if self.parent_fft.mmpp
                else None
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode, log_level=log_level
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

    def install_ffmpeg(self) -> str:
        """
        Install FFmpeg for MP4 animation support.

        This method ensures FFmpeg is available for high-quality video
        animation export. If FFmpeg is not found on the system, it will
        be automatically downloaded and installed.

        Returns:
        --------
        str or None
            Path to ffmpeg executable if successful, None if failed

        Example:
        --------
        >>> job = load_zarr("data.zarr")
        >>> ffmpeg_path = job[0].fft.modes.install_ffmpeg()
        >>> if ffmpeg_path:
        ...     job[0].fft.modes.save_modes_animation("animation.mp4")
        """
        return install_ffmpeg()


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

    def plot_modes(self, **kwargs) -> tuple[Figure, np.ndarray]:
        """Plot modes at this frequency."""
        return self.parent.mode_analyzer.plot_modes(self.frequency, **kwargs)

    def get_mode(self, **kwargs) -> FMRModeData:
        """Get mode data at this frequency."""
        return self.parent.mode_analyzer.get_mode(self.frequency, **kwargs)

    def characterize(self, **kwargs) -> ModeCharacterizationResult:
        """Return automatic mode classification for this frequency."""

        return self.parent.mode_analyzer.characterize_mode(self.frequency, **kwargs)

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
        import io

        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.table import Table
        from rich.text import Text

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
            "• characterize(**kwargs) → ModeCharacterizationResult",
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

# Automatic classification
characterization = freq_interface.characterize()
print(characterization.primary_class)

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
            f"Methods: plot_modes(), get_mode(), characterize(), frequency (property)\n"
            f"Parent analyzer has {len(self.parent.mode_analyzer.frequencies)} frequencies"
        )
