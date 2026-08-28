"""
FMR Mode Visualization Module

Professional implementation for visualizing FMR modes with interactive spectrum.
Provides both programmatic and interactive interfaces for mode analysis.
"""

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np

if TYPE_CHECKING:
    from ..vortex_classifier import VortexModeResult

# Import shared logging configuration
from ...cli.logging_config import get_mmpp_logger, setup_mmpp_logging
from .material_mask import masked_spatial, resolve_material_mask

# Get logger for FMR modes
log = get_mmpp_logger("mmpp.fft.modes")


def _mode_extent_nm(
    geometry: Any,
    *,
    nx: int,
    ny: int,
    dx_nm: float,
    dy_nm: float,
) -> tuple[float, float, float, float]:
    """Resolve the public mode extent, whose coordinate unit is nanometres."""
    geometry_axes = getattr(geometry, "axes", {})
    geometry_x = geometry_axes.get("x")
    geometry_y = geometry_axes.get("y")
    if geometry_x is not None and geometry_y is not None:
        metres_to_nm = 1e9
        return (
            float(geometry_x.min_m) * metres_to_nm,
            float(geometry_x.max_m) * metres_to_nm,
            float(geometry_y.min_m) * metres_to_nm,
            float(geometry_y.max_m) * metres_to_nm,
        )
    return (0.0, nx * dx_nm, 0.0, ny * dy_nm)


def _select_mode_time_axis(
    raw_time: Any,
    *,
    total_samples: int,
    view_slice: Any,
    time_slice: slice,
    expected_samples: int,
) -> np.ndarray | None:
    """Select time metadata only when it exactly describes the active view."""
    time_axis = np.asarray(raw_time, dtype=float).reshape(-1)
    view_key = view_slice if isinstance(view_slice, tuple) else (view_slice,)
    view_time = view_key[0] if view_key and view_key[0] is not None else None

    if isinstance(view_time, slice):
        candidate = time_axis[view_time]
    elif view_time is None or view_time is Ellipsis:
        candidate = time_axis
    else:
        return None

    if candidate.size != total_samples:
        if time_axis.size != total_samples:
            return None
        candidate = time_axis

    selected = np.asarray(candidate[time_slice], dtype=float)
    return selected if selected.size == expected_samples else None


def _uniform_mode_dt(time_axis: np.ndarray) -> float:
    """Return dt for a strictly increasing, uniformly sampled mode time axis."""
    values = np.asarray(time_axis, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError("Mode FFT requires at least two time-axis samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("Mode time axis contains non-finite values")
    deltas = np.diff(values)
    if np.any(deltas <= 0):
        raise ValueError("Mode time axis must be strictly increasing")
    dt = float(np.mean(deltas))
    tolerance = max(abs(dt) * 1e-6, np.finfo(float).eps * 10)
    if np.max(np.abs(deltas - dt)) > tolerance:
        raise ValueError(
            "Mode FFT requires a uniformly sampled time axis; resample the data "
            "before computing modes"
        )
    return dt


def _normalize_mode_input_shape(
    data: np.ndarray,
    *,
    component_index: int | None,
) -> np.ndarray:
    """Normalize selected magnetization to canonical ``(t,z,y,x,c)``."""
    values = np.asarray(data)
    if component_index is not None:
        if component_index not in {0, 1, 2}:
            raise ValueError(f"Invalid selected component index: {component_index}")
        component_axis_is_present = values.ndim in {4, 5} and values.shape[-1] == 1
        if not component_axis_is_present:
            values = values[..., np.newaxis]
    if values.ndim == 4:
        values = values[:, np.newaxis, ...]
    if values.ndim != 5 or values.shape[-1] not in {1, 3}:
        raise ValueError(
            "Mode input must resolve to (t,z,y,x,c) with one or three "
            f"components, got shape {values.shape}"
        )
    return values


def _mode_power_paths(
    mode_group: str,
    *,
    dataset_name: str,
    include_legacy: bool,
) -> list[tuple[str, str]]:
    """Return ordered, view-safe mode-spectrum cache candidates."""
    candidates = [
        (f"{mode_group}/power_sum", "power_sum"),
        (f"{mode_group}/power_max", "power_max"),
    ]
    legacy_group = f"modes/{dataset_name}"
    if include_legacy and legacy_group != mode_group:
        candidates.extend(
            [
                (f"{legacy_group}/power_sum", "power_sum"),
                (f"{legacy_group}/power_max", "power_max"),
            ]
        )
    return candidates


def _mode_power_summaries(fft_result: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return maximum and integrated squared FFT magnitude per frequency."""
    power = np.abs(np.asarray(fft_result)) ** 2
    reduction_axes = tuple(range(1, power.ndim)) if power.ndim > 1 else None
    if reduction_axes:
        return np.max(power, axis=reduction_axes), np.sum(power, axis=reduction_axes)
    return power.copy(), power.copy()


def _mode_power_cache_is_squared(group: Any) -> bool:
    """Return whether a modes group declares the current power definition."""
    attrs = getattr(group, "attrs", {})
    return attrs.get("power_definition") == "abs_fft_squared"


def _select_peak_spectrum_component(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    component: int,
) -> np.ndarray:
    """Return one frequency trace from a scalar or component spectrum."""
    values = np.asarray(spectrum)
    freqs = np.asarray(frequencies)
    if freqs.ndim != 1:
        raise ValueError("Peak-detection frequencies must be one-dimensional")
    if values.ndim not in {1, 2} or values.shape[0] != freqs.size:
        raise ValueError(
            "Peak spectrum must have shape (frequency,) or "
            f"(frequency, component), got {values.shape} for {freqs.size} frequencies"
        )
    if values.ndim == 1:
        return values
    if not isinstance(component, (int, np.integer)):
        raise TypeError("component must be an integer index")
    if component < 0 or component >= values.shape[1]:
        raise ValueError(
            f"component {component} out of range for {values.shape[1]} traces"
        )
    return values[:, int(component)]


def _write_zarr_array(
    group: Any,
    name: str,
    data: Any,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: Any = None,
    chunks: tuple[int, ...] | None = None,
    overwrite: bool = True,
) -> Any:
    """Write an array with a Zarr v2/v3 compatible group API."""
    array_data = (
        np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)
    )
    array_shape = shape or array_data.shape
    array_dtype = dtype or array_data.dtype

    if hasattr(group, "array"):
        return group.array(
            name,
            data=array_data,
            shape=array_shape,
            dtype=array_dtype,
            chunks=chunks,
            overwrite=overwrite,
        )

    if overwrite:
        try:
            if name in group:
                del group[name]
        except Exception:
            pass

    if hasattr(group, "create_array"):
        try:
            return group.create_array(
                name,
                data=array_data,
                shape=array_shape,
                dtype=array_dtype,
                chunks=chunks,
                overwrite=overwrite,
            )
        except (TypeError, ValueError):
            return group.create_array(
                name,
                data=array_data,
                chunks=chunks,
                overwrite=overwrite,
            )

    return group.create_dataset(
        name,
        data=array_data,
        shape=array_shape,
        dtype=array_dtype,
        chunks=chunks,
        overwrite=overwrite,
    )


# Import electromagnetic analysis module

from ..metrics import (
    PeakWidth,
    compute_half_width_at_half_max,
    format_width_value,
    normalize_peak_width_option,
)
from ..mode_characterization import (
    ModeCharacterAnalyzer,
    ModeCharacteristicConfig,
    ModeCharacterizationResult,
)
from .analyzer.cache import ModeCache

# Import mode analysis functions
from .analyzer.mode_analysis import (
    characterize_mode as _characterize_mode,
)
from .analyzer.mode_analysis import (
    characterize_vortex_mode as _characterize_vortex_mode,
)
from .analyzer.mode_analysis import (
    print_characterization_details as _print_characterization_details,
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
    cmc,
    cmocean,
    zarr,
)
from .ffmpeg_utils import (
    _create_ffmpeg_writer,
    _ensure_ffmpeg_available,
    check_ffmpeg_available,
    check_ffmpeg_installation,
    install_ffmpeg,
    install_ffmpeg_simple,
)
from .style import STYLING_AVAILABLE, MidpointNormalize, setup_animation_styling

# Import refactored utilities
from .utils.peak_detection import detect_peaks
from .utils.scalebar import calculate_optimal_length, format_scalebar_label
from .visualization.animation import (
    save_animated_view as _save_animated_view,
)

# Import animation functions
from .visualization.animation import (
    save_modes_animation as _save_modes_animation,
)
from .visualization.animation import (
    start_column_animation as _start_column_animation,
)
from .visualization.animation import (
    start_mode_animation as _start_mode_animation,
)
from .visualization.animation import (
    stop_column_animation as _stop_column_animation,
)
from .visualization.animation import (
    stop_mode_animation as _stop_mode_animation,
)
from .visualization.animation import (
    toggle_mode_animation as _toggle_mode_animation,
)

# Import interactive spectrum functions
from .visualization.interactive import (
    add_scale_bar as _add_scale_bar,
)
from .visualization.interactive import (
    interactive_spectrum as _interactive_spectrum,
)
from .visualization.interactive import (
    update_mode_plots as _update_mode_plots,
)

# Import static plotting functions
from .visualization.static_plots import (
    plot_modes as _plot_modes,
)
from .visualization.static_plots import (
    update_single_mode_plot as _update_single_mode_plot,
)
from .vortex_optics import TopologicalAnimator, VortexOptics


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
    scalebar_length_nm: float | None = None  # Auto-computed when None
    scalebar_location: str = "lower right"
    scalebar_pad: float = 0.3
    scalebar_color: str = "white"
    scalebar_fontsize: int = 9
    scalebar_frame: bool = False
    scalebar_height_fraction: float = 0.01
    scale_units: str = "nm"

    # Colorbar settings
    colorbar_fraction: float = 0.046
    colorbar_pad: float = 0.04
    colorbar_ticklabel_size: int = 9
    colorbar_label_size: int = 10

    # Inset Colorbar configuration (Publication Ready)
    colorbar_inset: bool = True
    colorbar_inset_width: str = "80%"  # User requested 80% width
    colorbar_inset_height: str = "22%"  # Taller for better spacing
    colorbar_inset_position: str = "lower center"
    colorbar_inset_bg_alpha: float = 0.7
    colorbar_inset_fontsize: int = 11  # Larger fonts
    colorbar_inset_title_fontsize: int = 12

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
        extent: tuple[float, float, float, float] | None = None,
        metadata: dict[str, Any] | None = None,
        material_mask: np.ndarray | None = None,
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
        self.mode_array = np.asarray(mode_array)
        self.extent = extent or (0, mode_array.shape[1], 0, mode_array.shape[0])
        self.metadata = metadata or {}

        # Validate input
        if not isinstance(mode_array, np.ndarray):
            raise TypeError("mode_array must be numpy array")
        if mode_array.ndim != 3 or mode_array.shape[2] != 3:
            raise ValueError("mode_array must have shape (ny, nx, 3)")
        self.material_mask: np.ndarray | None = None
        if material_mask is not None:
            mask = np.asarray(material_mask, dtype=bool)
            if mask.shape != mode_array.shape[:2]:
                raise ValueError(
                    f"material_mask shape {mask.shape} must match mode spatial "
                    f"shape {mode_array.shape[:2]}"
                )
            self.material_mask = mask
            self.mode_array = np.where(mask[..., None], self.mode_array, 0)

    @property
    def masked_mode_array(self) -> np.ndarray:
        """Complex mode with non-material cells masked for visualization."""
        return masked_spatial(self.mode_array, self.material_mask)

    @property
    def magnitude(self) -> np.ndarray:
        """Get magnitude of mode for each component."""
        return np.abs(self.masked_mode_array)

    @property
    def phase(self) -> np.ndarray:
        """Get phase of mode for each component."""
        return np.angle(self.masked_mode_array)

    @property
    def total_magnitude(self) -> np.ndarray:
        """Get total magnitude across all components."""
        return np.sqrt(np.sum(self.magnitude**2, axis=2))

    def get_component(self, component: int | str) -> np.ndarray:
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

        return masked_spatial(self.mode_array[:, :, component], self.material_mask)


class FMRModeAnalyzer:
    """
    Professional FMR mode analyzer with interactive visualization.

    Provides both programmatic access to mode data and interactive
    spectrum visualization for frequency selection.
    """

    def __init__(
        self,
        zarr_path: str,
        dataset_name: str | None = None,
        config: ModeVisualizationConfig | None = None,
        mode_character_config: ModeCharacteristicConfig | None = None,
        debug: bool = False,
        log_level: str | int | None = None,
        view_slice: Any | None = None,
        preloaded_data: np.ndarray | None = None,
        component_index: int | None = None,
        time_step_scale: float = 1.0,
        view_geometry=None,
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
        self.view_slice = view_slice
        self.preloaded_data = preloaded_data
        self.component_index = component_index
        self.time_step_scale = float(time_step_scale)
        self.view_geometry = view_geometry
        view_identity = f"{view_slice!r};dt_scale={self.time_step_scale}"
        if preloaded_data is not None:
            materialized = np.ascontiguousarray(np.asarray(preloaded_data))
            digest = hashlib.blake2b(materialized.tobytes(), digest_size=12).hexdigest()
            view_identity = (
                f"{materialized.dtype}:{materialized.shape}:{digest};"
                f"dt_scale={self.time_step_scale}"
            )
        self.view_id = (
            hashlib.blake2b(view_identity.encode(), digest_size=8).hexdigest()
            if view_slice is not None or preloaded_data is not None
            else None
        )
        self.mode_group = f"modes/{dataset_name}"
        if self.view_id is not None:
            self.mode_group = f"{self.mode_group}/views/{self.view_id}"
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
        self._mode_animations: dict[
            Any, Any
        ] = {}  # Dict to track active animations per axis
        self._animated_axes: set[Any] = set()  # Set of axes currently being animated

        # Mode data cache using refactored ModeCache
        self._mode_cache = ModeCache(maxsize=128)

    @property
    def modes_available(self) -> bool:
        """Check if mode data is available.

        Modes are considered available if we have the complex mode array and frequencies.
        The spectrum can be derived from modes power data (power_sum or power_max).
        """
        # Core requirement: modes and frequencies
        if self.modes_path is None or self.freqs_path is None:
            return False

        # Spectrum can come from multiple sources, not required for modes_available
        # because we can compute it from modes/power_sum if needed
        return True

    @property
    def last_fwhm(self) -> PeakWidth | None:
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

    def _get_zarr_paths(self) -> tuple[str | None, str | None, str | None]:
        """
        Unified path resolution for zarr datasets.

        Returns:
        --------
        Tuple[str, str, str]
            (modes_path, freqs_path, spectrum_path) or None if not found
        """
        # Possible base paths for modes/frequencies - consistent order
        base_paths = [self.mode_group]
        if self.view_id is None:
            base_paths.append(f"tmodes/{self.dataset_name}")

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
        spectrum_candidates = []
        if self.view_id is None:
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
        self.frequencies: np.ndarray | None = None
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
        self.spectrum: np.ndarray | None = None

        # First try view-local modes data (most up-to-date).
        for power_path, power_kind in _mode_power_paths(
            self.mode_group,
            dataset_name=self.dataset_name,
            include_legacy=self.view_id is None,
        ):
            log.debug("Looking for fresh modes spectrum at: %s", power_path)
            if power_path in self.zarr_file:
                power_group_path = power_path.rsplit("/", 1)[0]
                if not _mode_power_cache_is_squared(self.zarr_file[power_group_path]):
                    log.warning(
                        "Ignoring legacy mode summary %s because its power "
                        "definition is unknown; recompute modes with force=True",
                        power_path,
                    )
                    continue
                self.spectrum = np.asarray(self.zarr_file[power_path])
                if np.iscomplexobj(self.spectrum):
                    self.spectrum = np.abs(self.spectrum)
                log.info(
                    "Using fresh modes %s as spectrum: shape %s",
                    power_kind,
                    self.spectrum.shape,
                )
                break

        if self.spectrum is None and self.spectrum_path:
            # Fallback to FFT spectrum (may be stale)
            log.warning(
                "No fresh modes spectrum found, falling back to FFT spectrum (may be outdated)"
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
            assert self.spectrum is not None
            assert self.spectrum is not None
            log.info(f"Loaded FFT spectrum data: shape {self.spectrum.shape}")
        elif self.spectrum is None:
            log.error("No spectrum data found - neither modes nor FFT data available")
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
        self,
        spectrum: np.ndarray,
        frequencies: np.ndarray,
        *,
        threshold: float | None = None,
        min_distance: int | None = None,
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
        return cast(
            list[Peak],
            detect_peaks(
                spectrum=spectrum,
                frequencies=frequencies,
                threshold=(
                    self.config.peak_threshold
                    if threshold is None
                    else float(threshold)
                ),
                min_distance=(
                    self.config.peak_min_distance
                    if min_distance is None
                    else int(min_distance)
                ),
                use_scipy=True,
            ),
        )

    def _runtime_material_mask(
        self, *, z_layer: int, ny: int, nx: int
    ) -> np.ndarray | None:
        """Resolve a 2D mask for new and legacy mode caches."""
        mask_path = f"{self.mode_group}/material_mask"
        if self.zarr_file is not None and mask_path in self.zarr_file:
            stored = np.asarray(self.zarr_file[mask_path], dtype=bool)
            if stored.ndim == 2:
                candidate = stored
            elif stored.ndim == 3 and 0 <= z_layer < stored.shape[0]:
                candidate = stored[z_layer]
            else:
                candidate = None
            if candidate is not None and candidate.shape == (ny, nx):
                return np.asarray(candidate, dtype=bool)

        try:
            if self.preloaded_data is not None:
                sample = np.asarray(self.preloaded_data)[:1]
            else:
                dset = self.zarr_file[self.dataset_name]
                key = list(
                    self.view_slice
                    if isinstance(self.view_slice, tuple)
                    else (slice(None),) * len(dset.shape)
                )
                if len(key) < len(dset.shape):
                    key.extend([slice(None)] * (len(dset.shape) - len(key)))
                time_token = key[0]
                start = 0
                if isinstance(time_token, slice):
                    start, _, step = time_token.indices(int(dset.shape[0]))
                    if step <= 0:
                        return None
                key[0] = slice(start, min(start + 1, int(dset.shape[0])))
                if len(dset.shape) == 5 or (
                    len(dset.shape) == 4 and int(dset.shape[-1]) in {1, 2, 3}
                ):
                    key[-1] = slice(None)
                sample = np.asarray(dset[tuple(key)])

            canonical = _normalize_mode_input_shape(sample, component_index=None)
            mask3d, _source = resolve_material_mask(canonical)
            mask_candidate: Any
            if mask3d.shape[0] == 1:
                mask_candidate = mask3d[0]
            elif 0 <= z_layer < mask3d.shape[0]:
                mask_candidate = mask3d[z_layer]
            else:
                return None
            if mask_candidate.shape == (ny, nx):
                return np.asarray(mask_candidate, dtype=bool)
        except Exception as exc:
            log.debug("Could not infer runtime material mask: %s", exc)
        return None

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

        # Validate and normalize z_layer bounds. New mode data is canonical 5D,
        # but retain read compatibility with legacy single-layer 4D caches.
        mode_shape = self.zarr_file[self.modes_path].shape
        if len(mode_shape) == 5:
            n_layers = mode_shape[1]
        elif len(mode_shape) == 4:
            n_layers = 1
        else:
            raise ValueError(f"Unsupported mode array shape: {mode_shape}")

        # Handle negative indexing (like Python lists)
        if z_layer < 0:
            z_layer = n_layers + z_layer
            log.debug(f"Converted negative z_layer to {z_layer}")

        if z_layer < 0 or z_layer >= n_layers:
            raise ValueError(
                f"z_layer {z_layer} out of range. Available layers: 0-{n_layers - 1} (or negative: -{n_layers} to -1)"
            )

        cache_frequency = float(actual_freq)
        cached_mode = self._mode_cache.get(cache_frequency, z_layer)
        if cached_mode is not None:
            return cached_mode

        # Load mode data for this frequency with bounds checking
        try:
            if len(mode_shape) == 5:
                mode_data = self.zarr_file[self.modes_path][freq_idx, z_layer, :, :, :]
            else:
                mode_data = self.zarr_file[self.modes_path][freq_idx, :, :, :]
        except IndexError as e:
            raise ValueError(
                f"Invalid indices: freq_idx={freq_idx}, z_layer={z_layer}. {e}"
            ) from e

        # A component-selected DatasetAwareWrapper keeps a singleton component
        # axis for analysis. Restore it to the original Cartesian slot so mx/my
        # are not silently mislabeled as mz by single-component renderers.
        component_index = self.component_index
        if component_index is None:
            try:
                group = self.zarr_file[self.mode_group]
                stored_index = int(group.attrs.get("component_index", -1))
                component_index = stored_index if stored_index in {0, 1, 2} else None
            except Exception:
                component_index = None
        if mode_data.shape[-1] == 1 and component_index in {0, 1, 2}:
            expanded = np.zeros(mode_data.shape[:-1] + (3,), dtype=mode_data.dtype)
            expanded[..., component_index] = mode_data[..., 0]
            mode_data = expanded

        # Create spatial extent
        ny, nx = mode_data.shape[:2]
        extent = _mode_extent_nm(
            self.view_geometry,
            nx=nx,
            ny=ny,
            dx_nm=self.dx,
            dy_nm=self.dy,
        )

        # Metadata
        metadata = {
            "frequency_index": freq_idx,
            "requested_frequency": frequency,
            "actual_frequency": actual_freq,
            "z_layer": z_layer,
            "spatial_resolution": (self.dx, self.dy),
            "mode_shape": mode_shape,
        }

        material_mask = self._runtime_material_mask(z_layer=z_layer, ny=ny, nx=nx)
        metadata["material_mask_available"] = material_mask is not None
        result = FMRModeData(
            actual_freq,
            mode_data,
            extent,
            metadata,
            material_mask=material_mask,
        )
        self._update_cache(cache_frequency, z_layer, result)
        return result

    def characterize_mode(
        self,
        frequency: float,
        z_layer: int = 0,
        *,
        core_position: tuple[float, float] | None = None,
        analysis_radius: float | None = None,
        config: ModeCharacteristicConfig | None = None,
        verbose: bool = False,
    ) -> ModeCharacterizationResult:
        """Classify the mode at ``frequency`` into gyration/breathing/azimuthal families - see analyzer.mode_analysis for details."""
        return _characterize_mode(
            self,
            frequency,
            z_layer,
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
        core_position: tuple[float, float] | None = None,
        R_dot: float | None = None,
        config: ModeCharacteristicConfig | None = None,
        verbose: bool = False,
    ) -> "VortexModeResult":
        """Advanced vortex/skyrmion mode classification - see analyzer.mode_analysis for details."""
        return _characterize_vortex_mode(
            self,
            frequency,
            z_layer,
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
        threshold: float | None = None,
        min_distance: int | None = None,
        component: int = 0,
        spectrum: np.ndarray | None = None,
        frequencies: np.ndarray | None = None,
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

        if spectrum_data is None or freq_data is None:
            log.warning("No spectrum data available for peak detection")
            return []

        threshold = (
            self.config.peak_threshold if threshold is None else float(threshold)
        )
        min_distance = (
            self.config.peak_min_distance if min_distance is None else int(min_distance)
        )
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and non-negative")
        if min_distance < 1:
            raise ValueError("min_distance must be at least 1")

        freq_data = np.asarray(freq_data, dtype=float)
        spectrum_data = _select_peak_spectrum_component(
            spectrum_data, freq_data, component
        )
        if not np.all(np.isfinite(freq_data)) or not np.all(np.isfinite(spectrum_data)):
            raise ValueError("Peak spectrum and frequencies must be finite")

        # Normalize spectrum for peak detection
        spectrum_work = np.asarray(spectrum_data, dtype=float).copy()
        if self.config.spectrum_normalize:
            maximum = float(np.max(spectrum_work)) if spectrum_work.size else 0.0
            if maximum <= 0:
                return []
            spectrum_work = spectrum_work / maximum

        # Filter frequency range
        freq_mask = (freq_data >= self.config.f_min) & (freq_data <= self.config.f_max)
        freqs_filtered = freq_data[freq_mask]
        spectrum_filtered = spectrum_work[freq_mask]

        # Detect peaks
        peaks = self._detect_peaks(
            spectrum_filtered,
            freqs_filtered,
            threshold=threshold,
            min_distance=min_distance,
        )

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
        components: list[int | str] | None = None,
        save_path: str | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """Plot mode visualization for a specific frequency - see visualization.static_plots.plot_modes for details."""
        return _plot_modes(self, frequency, z_layer, components, save_path)

    def interactive_spectrum(
        self,
        components: list[int | str] | None = None,
        z_layer: int = 0,
        method: int = 1,
        show: bool = True,
        force: bool = False,
        use_fft_spectrum: bool = True,
        saveanim: bool | str | None = None,
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

    def _update_mode_plots(self, components: list[int | str], z_layer: int) -> None:
        """Update mode plots for current frequency."""
        _update_mode_plots(self, components, z_layer)

    # Alias for backward compatibility
    interactive_spectrum_old = interactive_spectrum

    def _toggle_mode_animation(
        self,
        ax: Any,
        row_idx: int,
        col_idx: int,
        component: str | int,
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
        component: str | int,
        z_layer: int,
    ) -> None:
        """Start in-place animation for specific mode axis."""
        _start_mode_animation(self, ax, row_idx, col_idx, component, z_layer)

    def _update_single_mode_plot(
        self,
        ax: Any,
        row_idx: int,
        col_idx: int,
        component: str | int,
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
        t_slice: slice = slice(None),
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
        t_slice : slice
            Time slice to process (default: all timesteps)
        """
        if not force and f"{self.mode_group}/arr" in self.zarr_file:
            log.info("Mode data already exists, use force=True to recompute")
            return

        log.info(f"Computing FMR modes for dataset {self.dataset_name}")

        # Remove existing data if force=True
        if force:
            try:
                # Open in write mode for deletion
                zarr_write = zarr.open(self.zarr_path, mode="a")
                if self.mode_group in zarr_write:
                    del zarr_write[self.mode_group]
                    log.info(f"Removed existing modes data for {self.dataset_name}")
                if self.view_id is None and f"fft/{self.dataset_name}" in zarr_write:
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

        # Normalize time slice and determine number of selected samples.
        source = (
            np.asarray(self.preloaded_data)
            if self.preloaded_data is not None
            else dset[self.view_slice]
            if self.view_slice is not None
            else dset
        )
        total_samples = int(source.shape[0])
        if isinstance(t_slice, slice):
            t_slice_norm = t_slice
        else:
            raise TypeError(f"t_slice must be slice, got {type(t_slice).__name__}")

        start, stop, step = t_slice_norm.indices(total_samples)
        if step <= 0:
            raise ValueError("t_slice step must be positive")
        num_samples = len(range(start, stop, step))

        # Determine sampling interval dt
        dt: float | None = None
        t_array: np.ndarray | None = None
        try:
            raw_t = dset.attrs["t"][:]
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            log.debug(
                "No usable explicit time axis for dataset %s: %s",
                self.dataset_name,
                exc,
            )
        else:
            t_array = _select_mode_time_axis(
                raw_t,
                total_samples=total_samples,
                view_slice=self.view_slice,
                time_slice=t_slice_norm,
                expected_samples=num_samples,
            )
            if t_array is not None:
                dt = _uniform_mode_dt(t_array)
            else:
                log.debug(
                    "Explicit time-axis length does not match active mode view; "
                    "falling back to scalar dt metadata"
                )

        def _extract_dt(candidate: Any) -> float | None:
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
            raise ValueError(
                "Could not determine the mode FFT timestep from the active time "
                "axis, t_sampl, or dt metadata"
            )

        if t_array is None:
            dt *= step
            if not np.isfinite(self.time_step_scale) or self.time_step_scale <= 0:
                raise ValueError(
                    "time_step_scale must be finite and positive for mode FFT"
                )
            dt *= self.time_step_scale

        if t_array is None:
            t_array = np.arange(num_samples, dtype=float) * dt

        # Calculate frequencies using number of time samples
        if num_samples < 2:
            raise ValueError(
                f"Mode computation requires at least two time samples, got {num_samples} for t_slice={t_slice_norm}"
            )

        freqs = np.fft.rfftfreq(num_samples, dt) * 1e-9  # Convert to GHz

        # Load and process data
        log.info(
            "Loading magnetization data: full_shape=%s, t_slice=%s, z_slice=%s",
            dset.shape,
            t_slice_norm,
            z_slice,
        )
        arr = _normalize_mode_input_shape(
            np.asarray(source[t_slice_norm]),
            component_index=self.component_index,
        )
        if not isinstance(z_slice, slice):
            raise TypeError(f"z_slice must be slice, got {type(z_slice).__name__}")
        z_start, z_stop, z_step = z_slice.indices(arr.shape[1])
        if z_step <= 0:
            raise ValueError("z_slice step must be positive")
        if len(range(z_start, z_stop, z_step)) == 0:
            raise ValueError("z_slice selects no mode layers")
        arr = arr[:, z_slice, ...]
        log.info("Loading magnetization data finished")

        geometry_candidates = []
        for candidate_name in ("geom", "geometry", "Msat", "msat", "Ms"):
            try:
                if candidate_name in self.zarr_file:
                    geometry_candidates.append(
                        (candidate_name, np.asarray(self.zarr_file[candidate_name]))
                    )
            except Exception as exc:
                log.debug(
                    "Could not load material-mask candidate %s: %s",
                    candidate_name,
                    exc,
                )
        material_mask, material_mask_source = resolve_material_mask(
            arr,
            geometry_candidates=geometry_candidates,
        )
        arr = np.where(material_mask[None, ..., None], arr, 0)
        active_fraction = float(np.mean(material_mask)) if material_mask.size else 0.0
        log.info(
            "Material mask resolved from %s: %d/%d active cells (%.1f%%)",
            material_mask_source,
            int(np.count_nonzero(material_mask)),
            int(material_mask.size),
            100.0 * active_fraction,
        )

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
                if self.mode_group in zarr_write:
                    del zarr_write[self.mode_group]
                if self.view_id is None and f"fft/{self.dataset_name}" in zarr_write:
                    del zarr_write[f"fft/{self.dataset_name}"]

            # Create groups
            modes_group = zarr_write.require_group(self.mode_group)
            # Don't create fft_group here - let plot_spectrum/calculate_fft_data handle it
            # This avoids conflicts with standard FFT data format

            # Save frequencies in modes group only
            _write_zarr_array(
                modes_group,
                "freqs",
                data=freqs,
                shape=freqs.shape,
                dtype=freqs.dtype,
                chunks=freqs.shape,  # Use the data shape as chunk size
                overwrite=True,
            )

            # Save complex modes (chunked only on first dimension)
            chunks = (1,) + fft_result.shape[1:]
            _write_zarr_array(
                modes_group,
                "arr",
                data=fft_result.astype(np.complex64, copy=False),
                shape=fft_result.shape,
                dtype=np.complex64,
                chunks=chunks,
                overwrite=True,
            )
            _write_zarr_array(
                modes_group,
                "material_mask",
                data=material_mask.astype(np.uint8, copy=False),
                shape=material_mask.shape,
                dtype=np.uint8,
                chunks=material_mask.shape,
                overwrite=True,
            )

            # Save power spectrum summary in modes group
            power_max, power_sum = _mode_power_summaries(fft_result)
            _write_zarr_array(
                modes_group,
                "power_max",
                data=power_max.astype(np.float32, copy=False),
                shape=power_max.shape,
                dtype=np.float32,
                chunks=power_max.shape,  # Use data shape as chunk size
                overwrite=True,
            )
            _write_zarr_array(
                modes_group,
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
            modes_group.attrs["t_slice"] = str(t_slice_norm)
            modes_group.attrs["dt"] = dt
            modes_group.attrs["view_slice"] = repr(self.view_slice)
            modes_group.attrs["view_id"] = self.view_id or "full"
            modes_group.attrs["component_index"] = (
                int(self.component_index) if self.component_index is not None else -1
            )
            modes_group.attrs["time_step_scale"] = self.time_step_scale
            modes_group.attrs["power_definition"] = "abs_fft_squared"
            modes_group.attrs["material_mask_source"] = material_mask_source
            modes_group.attrs["material_mask_active_fraction"] = active_fraction

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
        component: str | int = "z",
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

    def install_ffmpeg(self) -> str | None:
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
        self._mode_analyzer: FMRModeAnalyzer | None = None

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
                log_level=log_level,
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
        component: str | int = "z",
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

    def install_ffmpeg(self) -> str | None:
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

        output = io.StringIO()
        console = Console(file=output, width=100, force_terminal=True)

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
        return output.getvalue()

    def _basic_frequency_display(self) -> str:
        """Basic fallback display without rich formatting."""
        return (
            f"FrequencyModeInterface(frequency_index={self.frequency_index}, "
            f"frequency={self.frequency:.2e} Hz)\n"
            f"Methods: plot_modes(), get_mode(), characterize(), frequency (property)\n"
            f"Parent analyzer has {len(self.parent.mode_analyzer.frequencies)} frequencies"
        )
