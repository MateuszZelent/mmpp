"""
Core SpinWaveAnalyzer class for computing spin-wave dispersion relations.

Provides high-level interface for dispersion analysis of micromagnetic simulation data,
similar to FMRModeAnalyzer but focused on wave propagation and k-space analysis.
"""

from __future__ import annotations
import numpy as np
import zarr
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import os

from ._fft_backend import fft as _fft, ifft as _ifft, fft2 as _fft2, rfft as _rfft, fftfreq as _fftfreq, rfftfreq as _rfftfreq, fftshift as _fftshift

from .models import DispersionResult1D, DispersionResult2D, DispersionBranch, DispersionConfig
from .utils import (
    normalize_magnetization_components,
    extract_magnetization_component,
    detrend_time_series,
    apply_window_1d,
    apply_filter_pipeline,
    normalize_filter_config,
    split_filter_stages,
    apply_dispersion_post_filters,
    compute_welch_power_spectrum,
    hann_window,
    k_axis_from_grid,
    fold_spectrum_1d,
    find_peaks_1d,
    group_velocity_1d,
    validate_grid_parameters
)

logger = logging.getLogger(__name__)


def _dispersion_window_stats(n: int, window: Optional[str]) -> tuple[float, float]:
    """Return coherent sum and power sum for a supported 1D window."""
    if window is None:
        return float(n), float(n)
    window_key = str(window).lower()
    if window_key == "hann":
        w = hann_window(n)
        return float(np.sum(w)), float(np.sum(w * w))
    raise ValueError(f"Unknown window '{window}'")


def _normalize_dispersion_scaling(
    scaling: Optional[str],
) -> str:
    scaling_key = str(scaling or "raw_power").lower()
    aliases = {
        "power": "raw_power",
        "raw": "raw_power",
        "amplitude": "amplitude_squared",
        "amp2": "amplitude_squared",
    }
    scaling_key = aliases.get(scaling_key, scaling_key)
    valid = {"raw_power", "amplitude_squared", "psd"}
    if scaling_key not in valid:
        raise ValueError(
            f"Unknown dispersion scaling '{scaling}'. Supported: {', '.join(sorted(valid))}"
        )
    return scaling_key


def _mirror_k_indices(k_axis: np.ndarray) -> np.ndarray:
    """Return indices that sample the spectrum at ``-k`` for each sorted k bin."""
    k_values = np.asarray(k_axis, dtype=float)
    return np.array(
        [int(np.argmin(np.abs(k_values + k_value))) for k_value in k_values],
        dtype=int,
    )


def _time_spacing_from_axis(time_values: Any, source: str) -> tuple[Optional[float], List[str]]:
    """Return effective dt from a monotonic time axis and any quality notes."""
    axis = np.asarray(time_values, dtype=float).reshape(-1)
    if axis.size < 2:
        return None, []
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"Time axis '{source}' contains non-finite values")

    deltas = np.diff(axis)
    if np.any(deltas <= 0):
        raise ValueError(f"Time axis '{source}' must be strictly increasing")

    spacing = float(np.mean(deltas))
    tolerance = max(abs(spacing) * 1e-6, np.finfo(float).eps * 10)
    max_delta = float(np.max(np.abs(deltas - spacing)))
    notes: List[str] = []
    if max_delta > tolerance:
        notes.append(
            f"Sampling warning: Non-uniform time axis '{source}' approximated by "
            f"mean dt={spacing:g} s for FFT dispersion (max dt deviation "
            f"{max_delta:g} s, tolerance {tolerance:g} s)"
        )
    return spacing, notes


def _uniform_spatial_spacing(axis_values: Any, source: str) -> Optional[float]:
    """Return positive spacing from a monotonic, uniformly sampled spatial axis."""
    axis = np.asarray(axis_values, dtype=float).reshape(-1)
    if axis.size < 2:
        return None
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"Spatial axis '{source}' contains non-finite values")

    deltas = np.diff(axis)
    increasing = np.all(deltas > 0)
    decreasing = np.all(deltas < 0)
    if not (increasing or decreasing):
        raise ValueError(f"Spatial axis '{source}' must be strictly monotonic")

    spacing = float(abs(deltas[0]))
    tolerance = max(spacing * 1e-6, np.finfo(float).eps * 10)
    max_delta = float(np.max(np.abs(np.abs(deltas) - spacing)))
    if max_delta > tolerance:
        raise ValueError(
            f"Non-uniform spatial axis '{source}' is not supported for FFT dispersion "
            f"(max spacing deviation {max_delta:g} m, tolerance {tolerance:g} m)"
        )
    return spacing


def _sampling_quality_notes(
    *,
    n_time: int,
    n_space: int,
    dt: float,
    dx: float,
    dk_max: Optional[float],
) -> list[str]:
    """Return non-fatal notes for sampling setups likely to limit FFT quality."""
    notes: list[str] = []
    if n_time <= 0 or n_space <= 0 or dt <= 0 or dx <= 0:
        return notes

    f_nyquist = 0.5 / dt
    k_nyquist = np.pi / dx
    df = 1.0 / (n_time * dt) if n_time > 0 else float("inf")
    dk = (2.0 * np.pi) / (n_space * dx) if n_space > 0 else float("inf")

    if n_time < 8:
        notes.append(
            f"Sampling warning: only {n_time} time samples; frequency axis is weakly resolved"
        )
    elif n_time < 32:
        notes.append(
            f"Sampling warning: coarse frequency resolution df={df:.3g} Hz with {n_time} time samples"
        )

    if n_space < 8:
        notes.append(
            f"Sampling warning: only {n_space} spatial samples; k-axis is weakly resolved"
        )
    elif n_space < 32:
        notes.append(
            f"Sampling warning: coarse k resolution dk={dk:.3g} rad/m with {n_space} spatial samples"
        )

    if np.isfinite(f_nyquist) and np.isfinite(k_nyquist):
        notes.append(
            f"Nyquist limits: |f|<={f_nyquist:.3g} Hz, |k|<={k_nyquist:.3g} rad/m"
        )

    if dk_max is not None and np.isfinite(k_nyquist) and abs(float(dk_max)) > k_nyquist:
        notes.append(
            "Sampling warning: config.dk_max exceeds spatial Nyquist limit; "
            "branch tracking may connect aliased k bins"
        )

    return notes


class SpinWaveAnalyzer:
    """
    Analyzer for spin-wave dispersion relations from micromagnetic simulations.
    
    Similar to FMRModeAnalyzer but focused on wave propagation analysis in k-space.
    Computes S(k,f) dispersion relations, tracks branches, and analyzes propagating modes.
    
    Parameters
    ----------
    zarr_path : str or Path
        Path to zarr file containing time-domain magnetization data
    config : Optional[DispersionConfig]
        Analysis configuration parameters
        
    Attributes
    ----------
    zarr_path : Path
        Path to data file
    zarr_file : zarr.Group
        Opened zarr file handle  
    config : DispersionConfig
        Analysis configuration
    M_data : Optional[np.ndarray]
        Cached magnetization data (T, Z, Y, X, 3)
    time_axis : Optional[np.ndarray]  
        Time axis [s]
    dt : float
        Time step [s]
    grid_spacings : Dict[str, float]
        Spatial grid spacings {dx, dy, dz} [m]
    """
    

    def __init__(
        self,
        zarr_path: str | Path,
        config: Optional[DispersionConfig] = None,
        tmax: Optional[int] = None,
        tmin: Optional[int] = None,
        slice_info: Optional[tuple] = None,
        dataset_name: Optional[str] = None,
    ):
        """
        Initialize spin-wave analyzer.

        Parameters
        ----------
        zarr_path : str
            Path to zarr simulation file
        config : DispersionConfig, optional
            Analysis configuration parameters
        tmin, tmax : int or None, optional
            Optional time-index window. ``tmin`` is the first loaded time index;
            ``tmax`` is the exclusive stop index. If both are None, loads all
            available timesteps.
        slice_info : tuple, optional
            Slicing information from DatasetAwareWrapper
        dataset_name : str, optional
            Name of magnetization dataset in zarr file
        """
        self.config: DispersionConfig = config or DispersionConfig()
        self.tmin: Optional[int] = tmin if tmin is None else int(tmin)
        self.tmax: Optional[int] = tmax if tmax is None else int(tmax)
        self.slice_info: Optional[tuple] = slice_info
        self.dataset_name: Optional[str] = dataset_name
        self.zarr_path = Path(zarr_path)

        # Data storage
        self.zarr_file: Optional[zarr.Group] = None
        self.M_data: Optional[np.ndarray] = None
        self._M_ref: Optional[Any] = None  # Underlying magnetization array reference
        self._M_path: Optional[str] = None
        self._base_indexer: Optional[tuple] = None
        self._time_axis_pos: Optional[int] = None
        self._time_axis_length: Optional[int] = None
        self._loaded_time: int = 0
        self.time_axis: Optional[np.ndarray] = None
        self._time_axis_notes: List[str] = []
        self.dt: float = 0.0
        self.grid_spacings: Dict[str, float] = {}

        # Load data
        self._load_data()

    def _load_data(self) -> None:
        """Load magnetization data from zarr file."""
        try:
            zarr_obj = zarr.open(str(self.zarr_path), mode="r")
            if isinstance(zarr_obj, zarr.Group):
                self.zarr_file = zarr_obj
            else:
                raise ValueError("Expected zarr Group, got Array")
            logger.info("Opened zarr file: %s", self.zarr_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to open zarr file %s: %s", self.zarr_path, exc)
            raise

        # Load time-domain magnetization data
        self._load_magnetization()
        self._extract_grid_parameters()

    def _load_magnetization(self) -> None:
        """Load time-domain magnetization data M(t,x,y,z) or single component."""
        possible_paths = []
        if self.dataset_name:
            possible_paths.append(self.dataset_name)
        possible_paths.extend([
            "m_layer",
            "m",
            "M",
            "magnetization",
            "m_full",
            "m_resonator",
            "table/m",
        ])

        if self.zarr_file is None:
            raise ValueError("Zarr file not loaded")

        logger.debug("Searching for magnetization data in paths: %s", possible_paths)
        logger.debug("Available datasets in zarr file: %s", list(self.zarr_file.keys()))
        logger.debug("Slice info: %s", self.slice_info)
        
        failed_attempts = []
        
        for path in possible_paths:
            if path not in self.zarr_file:
                logger.debug("Path '%s' not found in zarr file", path)
                continue
            try:
                M_ref = self.zarr_file[path]
                if not (hasattr(M_ref, "shape") and hasattr(M_ref, "dtype")):
                    logger.info("'%s' is not an array, skipping", path)
                    failed_attempts.append((path, "not an array"))
                    continue

                logger.info("Found magnetization at '%s': shape %s, dtype %s", path, M_ref.shape, M_ref.dtype)
                self._M_ref = M_ref
                self._M_path = path
                self._configure_indexing(M_ref)

                self.M_data = self._load_reference_data(self.tmin, self.tmax)
                logger.info("Loaded magnetization data: shape %s", self.M_data.shape)

                # Accept arrays with any last dimension (1 for single component, 3 for vector)
                # The slicing may have already selected a single component
                if self.M_data.ndim >= 1:
                    last_dim = self.M_data.shape[-1] if self.M_data.ndim > 1 else 1
                    if last_dim == 1:
                        logger.info(
                            "Single-component magnetization data detected (last axis = 1). "
                            "This is expected when component was pre-selected via slicing (e.g., [:,...,2])."
                        )
                    elif last_dim == 3:
                        logger.info("Vector magnetization data detected (mx, my, mz).")
                    else:
                        logger.warning(
                            "Magnetization array last axis is %d (expected 1 or 3). "
                            "Attempting to proceed anyway.",
                            last_dim,
                        )
                    # Successfully loaded data
                    logger.info("Successfully loaded magnetization from '%s'", path)
                    break
            except Exception as exc:  # noqa: BLE001
                import traceback
                logger.warning("Failed to access magnetization at '%s': %s", path, exc)
                logger.debug("Full traceback: %s", traceback.format_exc())
                failed_attempts.append((path, str(exc)))
                self._M_ref = None
                continue
        else:
            error_msg = f"No magnetization data found in {self.zarr_path}"
            if failed_attempts:
                error_msg += "\nFailed attempts:"
                for path, reason in failed_attempts:
                    error_msg += f"\n  - {path}: {reason}"
            raise ValueError(error_msg)

    def _configure_indexing(self, M_ref: Any) -> None:
        """Prepare base slice/indexer information for repeated loads."""
        shape = tuple(getattr(M_ref, "shape", ()))
        if not shape:
            self._base_indexer = None
            self._time_axis_pos = None
            self._time_axis_length = None
            logger.debug("Magnetization shape unavailable; skipping indexer configuration")
            return

        logger.debug("Configuring indexing for shape %s with slice_info %s", shape, self.slice_info)
        base_indexer = self._normalize_slice(shape)
        logger.debug("Normalized slice: %s", base_indexer)
        self._base_indexer = base_indexer
        self._time_axis_pos, self._time_axis_length = self._resolve_time_axis(base_indexer, shape)
        if self._time_axis_pos is None:
            logger.debug("Time axis collapsed by slice; full series not available for tmin/tmax control")
        else:
            logger.debug(
                "Time axis index=%s, length=%s after applying slice", self._time_axis_pos, self._time_axis_length
            )

    def _normalize_slice(self, shape: Tuple[int, ...]) -> tuple:
        """Expand slice_info/ellipsis into a tuple aligned with array dimensions."""
        ndim = len(shape)
        if self.slice_info is None:
            return tuple(slice(None) for _ in range(ndim))

        if isinstance(self.slice_info, tuple):
            entries = list(self.slice_info)
        else:
            entries = [self.slice_info]

        result: list[Any] = []
        dims_consumed = 0
        i = 0
        while i < len(entries):
            entry = entries[i]
            if entry is Ellipsis:
                remaining = entries[i + 1 :]
                remaining_dims = sum(1 for item in remaining if item is not None and item is not Ellipsis)
                fill = max(ndim - dims_consumed - remaining_dims, 0)
                result.extend(slice(None) for _ in range(fill))
                dims_consumed += fill
            else:
                result.append(entry)
                if entry is not None:
                    dims_consumed += 1
            i += 1

        while dims_consumed < ndim:
            result.append(slice(None))
            dims_consumed += 1

        return tuple(result)

    def _resolve_time_axis(
        self,
        indexer: tuple,
        shape: Tuple[int, ...],
    ) -> Tuple[Optional[int], Optional[int]]:
        """Locate time-axis entry and corresponding length after slicing."""
        dim_idx = 0
        for idx, entry in enumerate(indexer):
            if entry is None:
                continue
            if dim_idx == 0:
                if isinstance(entry, slice):
                    return idx, shape[0]
                return None, shape[0]
            dim_idx += 1
        return None, None

    def _limit_time_slice(
        self,
        base_slice: slice,
        tmin: Optional[int],
        tmax: Optional[int],
        axis_length: int,
    ) -> Tuple[slice, bool]:
        if axis_length <= 0:
            return base_slice, False

        start, stop, step = base_slice.indices(axis_length)
        if step <= 0:
            return base_slice, False

        requested_start = start
        if tmin is not None and tmin > 0:
            requested_start = min(start + int(tmin) * step, stop)

        requested_stop = stop
        if tmax is not None:
            if tmax <= 0:
                return base_slice, False
            requested_stop = min(start + int(tmax) * step, stop)

        if requested_start == start and requested_stop == stop:
            return base_slice, False

        if requested_stop < requested_start:
            requested_stop = requested_start

        return slice(requested_start, requested_stop, step), True

    def _indexer_for_time_window(
        self,
        tmin: Optional[int],
        tmax: Optional[int],
    ) -> Optional[tuple]:
        if self._base_indexer is None or self._time_axis_pos is None or self._time_axis_length is None:
            return self._base_indexer

        if tmin is None and tmax is None:
            return self._base_indexer

        base_slice = self._base_indexer[self._time_axis_pos]
        if not isinstance(base_slice, slice):
            return self._base_indexer

        limited_slice, changed = self._limit_time_slice(
            base_slice,
            None if tmin is None else int(tmin),
            None if tmax is None else int(tmax),
            self._time_axis_length,
        )
        if not changed:
            return self._base_indexer

        indexer = list(self._base_indexer)
        indexer[self._time_axis_pos] = limited_slice
        return tuple(indexer)

    def _slice_stride_for_source_axis(self, source_axis: int) -> int:
        """Return absolute slice stride for an original data axis."""
        if self._base_indexer is None or self._M_ref is None:
            return 1

        shape = tuple(getattr(self._M_ref, "shape", ()))
        dim_idx = 0
        for entry in self._base_indexer:
            if entry is None:
                continue
            if dim_idx == source_axis:
                if isinstance(entry, slice) and source_axis < len(shape):
                    _, _, step = entry.indices(shape[source_axis])
                    return max(abs(int(step)), 1)
                return 1
            dim_idx += 1
        return 1

    def _apply_effective_slice_spacings(self) -> None:
        """Scale dt/dx/dy/dz by slicing stride before FFT axes are built."""
        if self.slice_info is None:
            return

        time_stride = self._slice_stride_for_source_axis(0)
        if time_stride > 1 and self.dt > 0:
            self.dt *= time_stride
            logger.info("Effective dt after time slicing stride %d: %s", time_stride, self.dt)

        axis_to_source = {"dz": 1, "dy": 2, "dx": 3}
        for axis, source_axis in axis_to_source.items():
            if axis not in self.grid_spacings:
                continue
            stride = self._slice_stride_for_source_axis(source_axis)
            if stride > 1:
                self.grid_spacings[axis] *= stride
                logger.info(
                    "Effective %s after spatial slicing stride %d: %s",
                    axis,
                    stride,
                    self.grid_spacings[axis],
                )

    def _load_reference_data(
        self,
        tmin: Optional[int],
        tmax: Optional[int],
    ) -> np.ndarray:
        if self._M_ref is None:
            raise ValueError("No magnetization reference available")

        indexer = self._indexer_for_time_window(tmin, tmax)

        # ── Pre-flight RAM check ──────────────────────────────
        self._check_memory(self._M_ref, indexer)

        try:
            data = self._M_ref if indexer is None else self._M_ref[indexer]
        except TypeError:
            data = np.asarray(self._M_ref)
            if indexer is not None:
                data = data[indexer]

        data_array = np.array(data)
        normalized = normalize_magnetization_components(data_array)
        self._loaded_time = normalized.shape[0] if normalized.ndim > 0 else 0
        logger.debug(
            "Loaded %s time steps for dispersion analysis (requested tmin=%s, tmax=%s)",
            self._loaded_time,
            tmin,
            tmax,
        )
        return normalized

    # ── Memory estimation ─────────────────────────────────────

    @staticmethod
    def _get_available_ram_bytes() -> Optional[int]:
        """Return available RAM in bytes (Linux only)."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024  # kB → bytes
        except (OSError, ValueError):
            pass
        # Fallback: try psutil
        try:
            import psutil  # type: ignore[import-untyped]
            return psutil.virtual_memory().available
        except ImportError:
            return None

    def _check_memory(
        self,
        zarr_ref: Any,
        indexer: Any,
    ) -> None:
        """Estimate peak RAM usage and warn/abort if insufficient.

        Multiplier breakdown for dispersion pipeline:
          1× raw data (float32)
          1× complex64 cast
          1× spatial FFT result
          1× temporal FFT result
          0.5× power spectrum (float32)
        Total ≈ 4.5× raw data size.
        """
        PEAK_MULTIPLIER = 4.5
        SAFETY_MARGIN = 0.85  # leave 15% headroom

        # Get shape and dtype from zarr metadata (no data loaded)
        shape = zarr_ref.shape
        dtype = zarr_ref.dtype

        # Apply indexer to estimate sliced shape
        if indexer is not None:
            try:
                dummy = np.empty(0, dtype=dtype)
                # Create a fake array with just shape info to compute output shape
                sliced_shape = []
                for i, (s, idx) in enumerate(zip(shape, indexer if isinstance(indexer, tuple) else (indexer,))):
                    if isinstance(idx, slice):
                        sliced_shape.append(len(range(*idx.indices(s))))
                    elif isinstance(idx, int):
                        pass  # dimension dropped
                    else:
                        sliced_shape.append(s)
                if not sliced_shape:
                    sliced_shape = list(shape)
            except Exception:
                sliced_shape = list(shape)
        else:
            sliced_shape = list(shape)

        raw_bytes = int(np.prod(sliced_shape)) * np.dtype(dtype).itemsize
        peak_bytes = int(raw_bytes * PEAK_MULTIPLIER)
        avail = self._get_available_ram_bytes()

        raw_gb = raw_bytes / (1024**3)
        peak_gb = peak_bytes / (1024**3)

        logger.info(
            "Memory estimate: raw data %.1f GB (shape %s, %s), "
            "peak FFT pipeline ~%.1f GB (%.1f× multiplier)",
            raw_gb, sliced_shape, dtype, peak_gb, PEAK_MULTIPLIER,
        )

        if avail is not None:
            avail_gb = avail / (1024**3)
            logger.info(
                "Available system RAM: %.1f GB (need %.1f GB, headroom %.0f%%)",
                avail_gb, peak_gb, (avail_gb / peak_gb * 100) if peak_gb > 0 else 100,
            )

            if peak_bytes > avail * SAFETY_MARGIN:
                msg = (
                    f"\n{'='*65}\n"
                    f"  ⚠  INSUFFICIENT MEMORY for dispersion computation\n"
                    f"{'='*65}\n"
                    f"  Data shape:     {tuple(sliced_shape)} ({dtype})\n"
                    f"  Raw data size:  {raw_gb:.1f} GB\n"
                    f"  Peak estimate:  {peak_gb:.1f} GB\n"
                    f"  Available RAM:  {avail_gb:.1f} GB\n"
                    f"{'='*65}\n"
                    f"  Suggestions:\n"
                    f"    • Reduce time steps: m[:500,...] instead of m[:1500,...]\n"
                    f"    • Subsample space: m[:, :, ::2, ::2, :]\n"
                    f"    • Free other memory first\n"
                    f"{'='*65}"
                )
                logger.error(msg)
                raise MemoryError(msg)

    def _ensure_data_loaded(
        self,
        tmin: Optional[int] = None,
        tmax: Optional[int] = None,
    ) -> None:
        """Ensure magnetization data is loaded for the requested time window."""
        if self.M_data is not None and tmin == self.tmin and tmax == self.tmax:
            return

        if self._M_ref is None:
            raise ValueError("No magnetization reference available for deferred loading")

        logger.info("Reloading magnetization data for time window %s:%s", tmin, tmax)
        self.M_data = self._load_reference_data(tmin, tmax)
        self.tmin = tmin if tmin is None else int(tmin)
        self.tmax = tmax if tmax is None else int(tmax)

    def _extract_grid_parameters(self) -> None:
        """Extract time step and spatial grid parameters from zarr attributes."""
        if self.zarr_file is None:
            raise ValueError("Zarr file not loaded")
            
        attrs = dict(self.zarr_file.attrs)
        logger.info(f"Available zarr attributes: {list(attrs.keys())}")

        time_axis_dt: Optional[float] = None
        time_axis_source: Optional[str] = None
        declared_dt: Optional[float] = None
        declared_dt_source: Optional[str] = None
        dt_keys = ['t_sampl', 'dt', 'Dt', 'timestep', 'time_step']
        for key in dt_keys:
            if key in attrs:
                attr_val = attrs[key]
                if isinstance(attr_val, (int, float)):
                    declared_dt = float(attr_val)
                    declared_dt_source = key
                    break

        if hasattr(self, '_M_path') and self._M_path:
            try:
                dataset = self.zarr_file[self._M_path]
                if hasattr(dataset, 'attrs') and 't' in dataset.attrs:
                    t_attr = dataset.attrs['t']
                    time_axis_dt, notes = _time_spacing_from_axis(
                        t_attr,
                        f"{self._M_path}.attrs['t']",
                    )
                    self._time_axis_notes.extend(notes)
                    if time_axis_dt is not None:
                        self.time_axis = np.asarray(t_attr, dtype=float).reshape(-1)
                        time_axis_source = f"{self._M_path}.attrs['t']"
            except (KeyError, AttributeError, IndexError, TypeError) as e:
                logger.debug(f"Could not extract dt from dataset attrs: {e}")

        if time_axis_dt is None and 't' in self.zarr_file:
            t = np.array(self.zarr_file['t'])
            time_axis_dt, notes = _time_spacing_from_axis(t, "t")
            self._time_axis_notes.extend(notes)
            if time_axis_dt is not None:
                self.time_axis = np.asarray(t, dtype=float).reshape(-1)
                time_axis_source = "t"
        
        # Time axis metadata is more specific than scalar t_sampl/dt.  Accept
        # monotonic non-uniform output times by using their mean spacing and
        # surfacing a sampling note; FFT still uses one effective dt.
        if time_axis_dt is not None:
            self.dt = time_axis_dt
            logger.info(
                "Extracted effective dt = %s s from time axis '%s'",
                self.dt,
                time_axis_source,
            )
            if declared_dt is not None:
                dt_delta_tolerance = max(
                    abs(declared_dt),
                    abs(time_axis_dt),
                    np.finfo(float).tiny,
                ) * 1e-6
            if (
                declared_dt is not None
                and abs(declared_dt - time_axis_dt) > dt_delta_tolerance
            ):
                self._time_axis_notes.append(
                    f"Sampling warning: time axis dt={time_axis_dt:g} s differs "
                    f"from declared {declared_dt_source}={declared_dt:g} s; "
                    "using time axis dt for FFT dispersion"
                )
        elif declared_dt is not None:
            self.dt = declared_dt
            logger.info(
                "Extracted dt = %s s from global '%s'",
                self.dt,
                declared_dt_source,
            )
            
        if self.dt <= 0:
            logger.warning("Could not determine time step dt, using config value")
            self.dt = self.config.dt
            
        # Spatial grid spacings
        spacing_keys = {
            'dx': ['dx', 'Dx', 'gridsize_x', 'cell_size_x'],
            'dy': ['dy', 'Dy', 'gridsize_y', 'cell_size_y'], 
            'dz': ['dz', 'Dz', 'gridsize_z', 'cell_size_z']
        }
        
        for axis, keys in spacing_keys.items():
            for key in keys:
                if key in attrs:
                    attr_val = attrs[key]
                    if isinstance(attr_val, (int, float)):
                        self.grid_spacings[axis] = float(attr_val)
                        logger.info(f"Extracted {axis} = {self.grid_spacings[axis]} m from '{key}'")
                        break
            else:
                # Use config default if not found
                config_val = getattr(self.config, axis, None)
                if config_val is not None:
                    self.grid_spacings[axis] = config_val
                    logger.warning(f"Using config value for {axis} = {config_val} m")

        coordinate_keys = {
            "dx": ("x", "x_axis", "x_coords"),
            "dy": ("y", "y_axis", "y_coords"),
            "dz": ("z", "z_axis", "z_coords"),
        }
        for spacing_name, axis_keys in coordinate_keys.items():
            for axis_key in axis_keys:
                if axis_key not in self.zarr_file:
                    continue
                spacing = _uniform_spatial_spacing(
                    np.array(self.zarr_file[axis_key]),
                    axis_key,
                )
                if spacing is None:
                    continue
                if spacing_name not in self.grid_spacings:
                    self.grid_spacings[spacing_name] = spacing
                    logger.info(
                        "Inferred %s = %s m from spatial axis '%s'",
                        spacing_name,
                        spacing,
                        axis_key,
                    )
                break

        # Use config values as fallback
        if 'dx' not in self.grid_spacings and self.config.dx:
            self.grid_spacings['dx'] = self.config.dx
        if 'dy' not in self.grid_spacings and self.config.dy:
            self.grid_spacings['dy'] = self.config.dy
        if 'dz' not in self.grid_spacings and self.config.dz:
            self.grid_spacings['dz'] = self.config.dz

        self._apply_effective_slice_spacings()

        # Update config with effective extracted values
        if hasattr(self.config, 'dt'):
            self.config.dt = self.dt
        for axis, spacing in self.grid_spacings.items():
            if hasattr(self.config, axis):
                setattr(self.config, axis, spacing)
            
        logger.info(f"Grid parameters: dt={self.dt}, spacings={self.grid_spacings}")
        
        # Validate parameters
        validate_grid_parameters(
            self.dt,
            self.grid_spacings.get('dx'),
            self.grid_spacings.get('dy'),
            self.grid_spacings.get('dz')
        )
        
    @property
    def data_shape(self) -> Tuple[int, ...]:
        """Shape of magnetization data."""
        if self.M_data is not None:
            return self.M_data.shape
        else:
            return ()
        
    @property
    def time_length(self) -> int:
        """Number of time steps."""
        return self.data_shape[0] if self.M_data is not None else 0
        
    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        """Spatial grid shape (Z, Y, X).""" 
        shape = self.data_shape
        if len(shape) >= 4:
            return (shape[1], shape[2], shape[3])
        else:
            return (0, 0, 0)
    
    
    def compute_dispersion_1d(
        self,
        axis: str = "x",
        component: Optional[str] = None,
        avg_over_orthogonal: Optional[bool] = None,
        orthogonal_avg_mode: Optional[str] = None,
        time_window: Optional[str] = None,
        space_window: Optional[str] = None,
        detrend: Optional[str] = None,
        fold_period: Optional[float] = None,
        fold_agg: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        flipx: bool = True,
        store_complex: bool = True,
        scaling: Optional[str] = None,
    ) -> DispersionResult1D:
        """
        Compute 1D spin-wave dispersion S(k,f) along specified axis.
        
        Parameters
        ----------
        axis : {'x', 'y'}
            Propagation direction for dispersion analysis
        component : Optional[str]
            Magnetization component ('perp', 'mx', 'my', 'mz', 'sum')
            If None, uses config.component
        avg_over_orthogonal : Optional[bool] 
            Whether to average over orthogonal spatial dimensions
            If None, uses config.avg_over_orthogonal
        orthogonal_avg_mode : Optional[str]
            Strategy for collapsing the orthogonal axis. Supported values:
            - 'magnetization': average signal before spatial FFT (legacy default)
            - 'fft_power': mean spectral power after FFT (phase-robust)
            - 'fft_abs': mean FFT magnitude, squared back to power (preserves localized modes)
            - 'fft_power_max': keep max spectral power along orth axis
            - 'fft_power_median': median spectral power (outlier resistant)
            If None, uses config.orthogonal_avg_mode.
        time_window : Optional[str]
            Time-domain window function ('hann' or None)
            If None, uses config.time_window
        space_window : Optional[str]
            Spatial window function ('hann' or None) 
            If None, uses config.space_window
        detrend : Optional[str]
            Time detrending method ('mean', 'initial', None)
            If None, uses config.detrend
        fold_period : Optional[float]
            Real-space period [m] for Brillouin zone folding
            If None, uses config.fold_period
        fold_agg : Optional[str]
            Folding aggregation method ('sum', 'max')
            If None, uses config.fold_agg
        filters : Optional[dict]
            Optional filter configuration. Supports legacy preprocessing flags
            (remove_static/remove_average/hann_time/hann_space) and advanced
            ``pre``/``post``/``live`` technique dictionaries.
        flipx : bool, default=True
            Apply mirror flip to k-axis (k → -k) to correct NumPy FFT convention.
            When True (default), applies S[:,::-1] to swap positive/negative wave vectors.
            Also applies to COMSOL overlay data when active.
        store_complex : bool, default=True
            Store the phase-preserving complex spectrum in ``result.S_complex``.
            Disable to reduce peak and retained memory when only ``S(k,f)`` is needed.
        scaling : {'raw_power', 'amplitude_squared', 'psd'}, optional
            Spectral scaling for ``S(k,f)``. ``raw_power`` preserves legacy
            unnormalized ``|FFT|^2``; ``amplitude_squared`` corrects coherent FFT
            and window gain; ``psd`` applies a simple density normalization using
            time/spatial window energy.
            
        Returns
        -------
        DispersionResult1D
            Dispersion analysis results
        """
        # Use config defaults if not specified
        component = component or self.config.component
        avg_over_orthogonal = (
            avg_over_orthogonal if avg_over_orthogonal is not None else self.config.avg_over_orthogonal
        )
        orthogonal_avg_mode = (
            orthogonal_avg_mode
            if orthogonal_avg_mode is not None
            else getattr(self.config, "orthogonal_avg_mode", "magnetization")
        )
        orthogonal_avg_mode = str(orthogonal_avg_mode).lower()
        valid_modes = {
            "magnetization",
            "fft_power",
            "fft_abs",
            "fft_power_max",
            "fft_power_median",
        }
        if orthogonal_avg_mode not in valid_modes:
            raise ValueError(
                f"Unknown orthogonal_avg_mode='{orthogonal_avg_mode}'. "
                f"Supported: {', '.join(sorted(valid_modes))}",
            )
        if not avg_over_orthogonal and orthogonal_avg_mode == "magnetization":
            logger.warning(
                "orthogonal_avg_mode='magnetization' is incompatible with avg_over_orthogonal=False; "
                "falling back to 'fft_power' for orthogonal collapse",
            )
            orthogonal_avg_mode = "fft_power"
        time_window = time_window if time_window is not None else self.config.time_window
        space_window = space_window if space_window is not None else self.config.space_window
        detrend = detrend or self.config.detrend
        fold_period = fold_period if fold_period is not None else self.config.fold_period
        fold_agg = fold_agg or self.config.fold_agg
        store_complex = bool(store_complex)
        scaling = _normalize_dispersion_scaling(
            scaling if scaling is not None else getattr(self.config, "scaling", "raw_power")
        )
        
        if self.M_data is None:
            raise ValueError("No magnetization data loaded")
            
        # Get grid spacing for chosen axis
        if axis == "x":
            if 'dx' not in self.grid_spacings:
                raise ValueError("dx not available for x-axis analysis")
            dx = self.grid_spacings['dx']
            space_axis = 3  # X is axis 3 in (T,Z,Y,X,3)
            N_space = self.M_data.shape[3]
        elif axis == "y":
            if 'dy' not in self.grid_spacings:
                raise ValueError("dy not available for y-axis analysis")  
            dx = self.grid_spacings['dy']
            space_axis = 2  # Y is axis 2 in (T,Z,Y,X,3) 
            N_space = self.M_data.shape[2]
        else:
            raise ValueError("axis must be 'x' or 'y'")

        if N_space < 2:
            shape = tuple(int(v) for v in getattr(self.M_data, "shape", ()))
            raise ValueError(
                f"Cannot compute 1D dispersion along axis={axis!r}: selected "
                f"propagation axis has only {N_space} cell(s) after slicing "
                f"(normalized M_data shape={shape}). Choose a wider {axis}-range "
                "or switch the dispersion axis to the spatial direction that still "
                "has at least two cells. For a single z-plane prefer preserving the "
                "dimension, e.g. m[:, 0:1, y0:y1, x0:x1, :] or m.sel('z', ...)."
            )
            
        logger.info(f"Computing 1D dispersion along {axis}-axis, component='{component}'")
        # Extract magnetization component
        signal = extract_magnetization_component(self.M_data, component)

        filters_config = normalize_filter_config(filters)
        pre_filters, post_filters, live_filters = split_filter_stages(filters_config)
        if pre_filters.get("remove_average") and detrend == "mean":
            logger.debug(
                "remove_average filter skipped because detrend='mean' already subtracts the temporal mean",
            )
            pre_filters.pop("remove_average", None)

        # Stage summary for logging/debugging.
        active_pre = sorted(pre_filters.keys())
        active_post = sorted(post_filters.keys())
        active_live = sorted(live_filters.keys())

        if active_pre:
            logger.info(
                "Raw-data preprocessing filters active: %s",
                ", ".join(active_pre),
            )

        if active_post:
            logger.info(
                "Post-FFT filters active: %s",
                ", ".join(active_post),
            )

        if active_live:
            logger.info(
                "Live-capable post filters configured: %s",
                ", ".join(active_live),
            )

        if active_pre:
            preview_frame = signal[:1].copy()
            signal = apply_filter_pipeline(
                signal,
                pre_filters,
                time_axis=0,
                spatial_axes=(2, 3),
                dt=self.dt,
            )
            delta = float(np.linalg.norm(signal[:1] - preview_frame))
            logger.debug("Filter impact on first frame (L2 delta): %.3e", delta)

        # Preserve complex information for perpendicular analysis
        if np.iscomplexobj(signal):
            signal = signal.astype(np.complex64, copy=False)
            logger.info("Complex signal detected, preserving complex values (complex64)")
        else:
            signal = signal.astype(np.float32, copy=False)
            logger.info("Real-valued signal detected; continuing with float32 precision")

        logger.debug(f"Signal dtype after casting: {signal.dtype}, shape: {signal.shape}")

        # Detrend over time (axis 0)
        signal = detrend_time_series(signal, axis=0, method=detrend)

        # Apply time window
        signal = apply_window_1d(signal, axis=0, window=time_window)

        # Apply spatial window
        signal = apply_window_1d(signal, axis=space_axis, window=space_window)

        # Average over orthogonal axes if requested
        S_local = None
        orth_axis_values = None
        orth_axis_label = None
        store_local_spectra = not avg_over_orthogonal
        keep_orthogonal_dimension = store_local_spectra or orthogonal_avg_mode != "magnetization"

        if keep_orthogonal_dimension:
            # Always average over Z, keep orthogonal plane for spectrum-level aggregation
            spatial_signal = np.mean(signal, axis=1)  # -> (T, Y, X)
            if store_local_spectra:
                if axis == "x":
                    orth_axis_label = "y"
                    if "dy" in self.grid_spacings:
                        orth_axis_values = np.arange(spatial_signal.shape[1]) * self.grid_spacings["dy"]
                    else:
                        orth_axis_values = np.arange(spatial_signal.shape[1])
                else:
                    orth_axis_label = "x"
                    if "dx" in self.grid_spacings:
                        orth_axis_values = np.arange(spatial_signal.shape[2]) * self.grid_spacings["dx"]
                    else:
                        orth_axis_values = np.arange(spatial_signal.shape[2])
        else:
            if axis == "x":
                # Average over Z(1), Y(2) -> shape (T, X)
                spatial_signal = np.mean(signal, axis=(1, 2))
            else:  # axis == "y"
                # Average over Z(1), X(3) -> shape (T, Y)
                spatial_signal = np.mean(signal, axis=(1, 3))

        # Spatial FFT -> k-domain signal(t, k)
        if spatial_signal.ndim == 2:
            spatial_axis = 1
        else:
            spatial_axis = 2 if axis == "x" else 1

        sig_k = _fftshift(_fft(spatial_signal, axis=spatial_axis), axes=spatial_axis)
        k_axis = k_axis_from_grid(N_space, dx, shift=True)

        # Temporal full FFT at each k
        T_len = sig_k.shape[0]
        f_axis = _fftshift(_fftfreq(T_len, self.dt))
        Sk_full = _fft(sig_k, axis=0)
        Sk_shift = _fftshift(Sk_full, axes=0)
        del Sk_full

        # Store complex spectrum BEFORE taking abs (needed for mode reconstruction)
        # Shape: (Nk, Nf) or (Nk, N_orth, Nf) depending on keep_orthogonal_dimension
        S_complex_raw = np.moveaxis(Sk_shift, 0, -1) if store_complex else None

        # Compute power spectrum for visualization (optionally Welch-averaged).
        welch_cfg = pre_filters.get("welch_average")
        if welch_cfg is not None:
            welch_options = welch_cfg if isinstance(welch_cfg, dict) else {}
            logger.info(
                "Applying Welch temporal averaging: n_segments=%s overlap=%.2f",
                welch_options.get("n_segments", 4),
                float(welch_options.get("overlap", 0.5)),
            )
            power_shift = compute_welch_power_spectrum(
                sig_k,
                axis=0,
                n_segments=int(welch_options.get("n_segments", 4)),
                overlap=float(welch_options.get("overlap", 0.5)),
                n_fft=T_len,
                apply_hann=bool(welch_options.get("apply_hann", True)),
            )
            power = np.abs(power_shift).astype(np.float32, copy=False)
        else:
            power = np.abs(Sk_shift).astype(np.float32, copy=False)
            power *= power

        if not store_complex:
            del Sk_shift

        power = np.moveaxis(power, 0, -1)  # -> (..., Nf)

        time_coherent, time_energy = _dispersion_window_stats(T_len, time_window)
        space_coherent, space_energy = _dispersion_window_stats(N_space, space_window)
        coherent_gain = max(time_coherent * space_coherent, 1e-30)
        window_energy = max(time_energy * space_energy, 1e-30)
        scaling_factors: dict[str, float] = {
            "time_coherent_gain": float(time_coherent),
            "space_coherent_gain": float(space_coherent),
            "coherent_gain": float(coherent_gain),
            "time_window_energy": float(time_energy),
            "space_window_energy": float(space_energy),
            "window_energy": float(window_energy),
        }
        if scaling == "amplitude_squared":
            power = power / np.float32(coherent_gain * coherent_gain)
            scaling_factors["scale"] = float(1.0 / (coherent_gain * coherent_gain))
        elif scaling == "psd":
            scale = float(self.dt * dx / window_energy)
            power = power * np.float32(scale)
            scaling_factors["scale"] = scale
        else:
            scaling_factors["scale"] = 1.0

        if not keep_orthogonal_dimension:
            S = power.astype(np.float32, copy=False)
            S_complex = (
                S_complex_raw.astype(np.complex64, copy=False)
                if (store_complex and S_complex_raw is not None)
                else None
            )
        else:
            S_complex_orth = None
            if axis == "x":
                orthogonal_spectra = power.astype(np.float32, copy=False)
                if store_complex and S_complex_raw is not None:
                    S_complex_orth = S_complex_raw.astype(np.complex64, copy=False)
            else:
                orthogonal_spectra = np.moveaxis(power, 1, 0).astype(np.float32, copy=False)
                if store_complex and S_complex_raw is not None:
                    S_complex_orth = np.moveaxis(S_complex_raw, 1, 0).astype(np.complex64, copy=False)

            if store_local_spectra:
                S_local = orthogonal_spectra
                S_complex = S_complex_orth if store_complex else None
            else:
                S_complex = None
            S = self._collapse_orthogonal_spectra(orthogonal_spectra, orthogonal_avg_mode)
        
        # Apply flipx to correct NumPy FFT convention (swap +/- wave vectors).
        #
        # IMPORTANT: keep k_axis monotonic (it is already fftshifted ascending),
        # and mirror the k-dependent arrays by matching bins to -k. A simple
        # ``[::-1]`` is wrong for even fftshifted axes because the Nyquist bin
        # makes +m and -m land off by one.
        #
        # This must be done AFTER all FFT operations and averaging.
        if flipx:
            mirror_idx = _mirror_k_indices(k_axis)
            S = S[mirror_idx, :]
            if S_local is not None:
                S_local = S_local[:, mirror_idx, :]
            if S_complex is not None:
                if S_complex.ndim == 3:  # (N_orth, Nk, Nf)
                    S_complex = S_complex[:, mirror_idx, :]
                else:  # (Nk, Nf)
                    S_complex = S_complex[mirror_idx, :]

        S_raw = S.copy()
        S_local_raw = S_local

        # Apply post-FFT filters for visualization-friendly S(k,f).
        if post_filters or live_filters:
            post_config: dict[str, Any] = {
                "post": post_filters,
                "live": live_filters,
            }
            S = apply_dispersion_post_filters(
                S,
                k_axis=k_axis,
                f_axis=f_axis,
                filters=post_config,
                include_live=True,
            )

            apply_to_local = False
            for option in list(post_filters.values()) + list(live_filters.values()):
                if isinstance(option, dict) and bool(option.get("apply_to_local", False)):
                    apply_to_local = True
                    break
            if apply_to_local and S_local is not None:
                filtered_local = np.empty_like(S_local, dtype=float)
                for idx in range(S_local.shape[0]):
                    filtered_local[idx] = apply_dispersion_post_filters(
                        S_local[idx],
                        k_axis=k_axis,
                        f_axis=f_axis,
                        filters=post_config,
                        include_live=True,
                    )
                S_local = filtered_local

        logger.info(
            "Computed dispersion: S.shape=%s, k_range=[%.2e, %.2e], f_range=[%.1f, %.1f] Hz",
            S.shape,
            k_axis.min(),
            k_axis.max(),
            f_axis.min(),
            f_axis.max(),
        )

        notes = [f"1D dispersion along {axis}-axis"]
        if flipx:
            notes.append("k-axis flipped (flipx=True) to correct FFT convention")
        if not avg_over_orthogonal:
            notes.append("Orthogonal averaging disabled; local spectra stored in S_local")
        elif orthogonal_avg_mode != "magnetization":
            notes.append(f"Orthogonal collapse via {orthogonal_avg_mode}")
        if active_pre:
            notes.append(f"Pre-filters: {', '.join(active_pre)}")
        if "welch_average" in pre_filters:
            notes.append("Welch temporal averaging enabled for power spectrum")
        if active_post:
            notes.append(f"Post-filters: {', '.join(active_post)}")
        if active_live:
            notes.append(f"Live-capable filters configured: {', '.join(active_live)}")
        if not store_complex:
            notes.append("Complex spectrum disabled (store_complex=False)")
        notes.append(f"Spectral scaling: {scaling}")
        notes.extend(self._time_axis_notes)
        sampling_notes = _sampling_quality_notes(
            n_time=T_len,
            n_space=N_space,
            dt=self.dt,
            dx=dx,
            dk_max=getattr(self.config, "dk_max", None),
        )
        for note in sampling_notes:
            if note.startswith("Sampling warning:"):
                logger.warning(note)
            notes.append(note)

        # Create result object
        result = DispersionResult1D(
            S=S,
            k_axis=k_axis,
            f_axis=f_axis,
            axis=axis,
            component=component,
            config=self.config,
            dt=self.dt,
            dx=dx,
            flipx=flipx,
            notes=notes,
            S_local=S_local,
            S_local_raw=S_local_raw,
            S_local_display=S_local,
            S_complex=S_complex,
            S_raw=S_raw,
            S_display=S,
            scaling=scaling,
            scaling_factors=scaling_factors,
            orth_axis=orth_axis_values,
            orth_axis_label=orth_axis_label,
        )
        
        # Apply Brillouin zone folding if requested
        if fold_period is not None and fold_period > 0:
            logger.info(f"Applying BZ folding with period a={fold_period} m")
            k_folded, S_folded = fold_spectrum_1d(S, k_axis, fold_period, agg=fold_agg)
            result.S_folded = S_folded
            result.k_folded = k_folded
            result.fold_period = fold_period
            result.notes.append(f"BZ folded with period {fold_period} m")
            
        return result

    def _collapse_orthogonal_spectra(self, spectra: np.ndarray, mode: str) -> np.ndarray:
        """
        Reduce orthogonal spectra using the requested aggregation strategy.

        Parameters
        ----------
        spectra : np.ndarray
            Array shaped (N_orthogonal, Nk, Nf).
        mode : str
            Aggregation strategy name.

        Returns
        -------
        np.ndarray
            Collapsed spectrum with shape (Nk, Nf).
        """
        spectra = spectra.astype(np.float32, copy=False)
        if mode == "fft_power":
            return np.mean(spectra, axis=0)
        if mode == "fft_abs":
            amplitudes = np.sqrt(np.clip(spectra, 0.0, None))
            return np.square(np.mean(amplitudes, axis=0))
        if mode == "fft_power_max":
            return np.max(spectra, axis=0)
        if mode == "fft_power_median":
            return np.median(spectra, axis=0)

        raise ValueError(f"Unsupported orthogonal aggregation mode '{mode}'")
        
    def compute_dispersion_2d(
        self,
        component: Optional[str] = None,
        time_window: Optional[str] = None,
        detrend: Optional[str] = None
    ) -> DispersionResult2D:
        """
        Compute 2D spin-wave dispersion S(kx, ky, f).
        
        Parameters
        ----------
        component : Optional[str]
            Magnetization component to analyze
        time_window : Optional[str] 
            Time-domain window function
        detrend : Optional[str]
            Time detrending method
            
        Returns
        -------
        DispersionResult2D
            2D dispersion analysis results
        """
        # Use config defaults
        component = component or self.config.component
        time_window = time_window if time_window is not None else self.config.time_window
        detrend = detrend or self.config.detrend
        
        if self.M_data is None:
            raise ValueError("No magnetization data loaded")
            
        # Need both dx and dy for 2D analysis
        if 'dx' not in self.grid_spacings or 'dy' not in self.grid_spacings:
            raise ValueError("Both dx and dy required for 2D dispersion analysis")
            
        dx = self.grid_spacings['dx']
        dy = self.grid_spacings['dy']
        
        logger.info(f"Computing 2D dispersion, component='{component}'")
        
        # Average over Z if present, get (T, Y, X, 3)
        M_2d = self.M_data.mean(axis=1) if self.M_data.ndim == 5 else self.M_data
        
        # Extract component
        signal = extract_magnetization_component(M_2d, component)
        
        # Detrend and window in time
        signal = detrend_time_series(signal, axis=0, method=detrend)
        signal = apply_window_1d(signal, axis=0, window=time_window)
        
        # 2D spatial FFT  
        sig_k = _fftshift(_fft2(signal, axes=(1, 2)), axes=(1, 2))
        ky_axis = k_axis_from_grid(sig_k.shape[1], dy, shift=True)
        kx_axis = k_axis_from_grid(sig_k.shape[2], dx, shift=True)

        T_len = sig_k.shape[0]
        use_complex = np.iscomplexobj(sig_k)
        if use_complex:
            Sk_full = _fft(sig_k, axis=0)
            Sk_pos = Sk_full[: T_len // 2 + 1]
            f_axis = np.abs(_fftfreq(T_len, self.dt)[: Sk_pos.shape[0]])
        else:
            Sk_pos = _rfft(sig_k, axis=0)
            f_axis = _rfftfreq(T_len, self.dt)

        power = np.abs(Sk_pos) ** 2
        S = power.transpose(2, 1, 0).astype(np.float32, copy=False)

        logger.info(f"Computed 2D dispersion: S.shape={S.shape}")
        
        return DispersionResult2D(
            S=S,
            kx_axis=kx_axis,
            ky_axis=ky_axis,
            f_axis=f_axis,
            component=component,
            config=self.config,
            dt=self.dt,
            dx=dx,
            dy=dy,
            notes=[
                "2D dispersion S(kx,ky,f)",
                "Experimental API: compute_2d does not yet provide the full 1D "
                "raw/display/cache contract",
            ]
        )
        
    def track_branch(
        self,
        dispersion: DispersionResult1D,
        k_path: np.ndarray,
        f_seed: Optional[float] = None,
        dk_max: Optional[float] = None,
        df_max: Optional[float] = None
    ) -> DispersionBranch:
        """
        Track a dispersion branch along specified k-path.
        
        Parameters
        ----------
        dispersion : DispersionResult1D
            1D dispersion data
        k_path : np.ndarray
            Wave vector path to track [rad/m]
        f_seed : Optional[float]
            Starting frequency [Hz] for tracking
        dk_max : Optional[float]
            Maximum k deviation for sampling [rad/m]  
        df_max : Optional[float]
            Maximum f deviation between steps [Hz]
            
        Returns
        -------
        DispersionBranch
            Tracked branch data
        """
        dk_max = dk_max or self.config.dk_max
        df_max = df_max or self.config.df_max
        
        S, k_axis, f_axis = dispersion.get_active_data()
        
        f_tracked = np.zeros_like(k_path)
        amplitudes = np.zeros_like(k_path)
        idx_prev: Optional[int] = None
        
        # Initial step at k_path[0]
        k0 = k_path[0]
        mask0 = np.abs(k_axis - k0) <= dk_max
        if not np.any(mask0):
            raise ValueError(f"First k={k0} has no data within dk_max={dk_max}")
            
        spec0 = S[mask0, :].sum(axis=0)
        if f_seed is None:
            idx0 = int(np.argmax(spec0))
        else:
            idx0 = int(np.argmin(np.abs(f_axis - f_seed)))
            
        f_tracked[0] = f_axis[idx0]
        amplitudes[0] = spec0[idx0]
        idx_prev = idx0
        
        # Track along k_path
        for i in range(1, len(k_path)):
            ki = k_path[i]
            mask = np.abs(k_axis - ki) <= dk_max
            
            if not np.any(mask):
                logger.warning(f"No data at k={ki} within dk_max, using nearest")
                # Use closest available k
                closest_idx = np.argmin(np.abs(k_axis - ki))
                mask = np.zeros_like(k_axis, dtype=bool)
                mask[closest_idx] = True
                
            spec = S[mask, :].sum(axis=0)
            
            # Limit search around previous frequency
            if df_max is not None and idx_prev is not None:
                df_idx = int(df_max / (f_axis[1] - f_axis[0])) if len(f_axis) > 1 else len(f_axis)
                idx_min = max(0, idx_prev - df_idx)
                idx_max = min(len(f_axis), idx_prev + df_idx)
                search_slice = slice(idx_min, idx_max)
            else:
                search_slice = slice(None)
                
            idx_local = np.argmax(spec[search_slice])
            if isinstance(search_slice, slice):
                idx_global = search_slice.start + idx_local
            else:
                idx_global = idx_local
                
            f_tracked[i] = f_axis[idx_global]
            amplitudes[i] = spec[idx_global]
            idx_prev = idx_global
            
        logger.info(f"Tracked branch over {len(k_path)} k points")
        
        return DispersionBranch(
            k_path=k_path,
            f_values=f_tracked,
            amplitudes=amplitudes,
            tracking_config={
                'dk_max': dk_max,
                'df_max': df_max,
                'f_seed': f_seed
            },
            notes=[f"Branch tracked with dk_max={dk_max:.1e}"]
        )
        
    def find_all_peaks(
        self,
        dispersion: DispersionResult1D,
        min_prominence: Optional[float] = None
    ) -> List[Tuple[float, float, float]]:
        """
        Find all spectral peaks in dispersion data.
        
        Parameters
        ---------- 
        dispersion : DispersionResult1D
            Dispersion data to analyze
        min_prominence : Optional[float]
            Minimum peak prominence
            
        Returns
        -------
        List[Tuple[float, float, float]]
            List of (k, f, amplitude) tuples for detected peaks
        """
        min_prominence = min_prominence or self.config.min_prominence
        
        S, k_axis, f_axis = dispersion.get_active_data()
        
        peaks = []
        for ik, k_val in enumerate(k_axis):
            spectrum = S[ik, :]
            peak_indices = find_peaks_1d(spectrum, min_prominence=min_prominence)
            
            for peak_idx in peak_indices:
                f_val = f_axis[peak_idx]
                amplitude = spectrum[peak_idx]
                peaks.append((float(k_val), float(f_val), float(amplitude)))
                
        logger.info(f"Found {len(peaks)} peaks with prominence >= {min_prominence}")
        return peaks
        
    def __repr__(self) -> str:
        if self.M_data is not None:
            return (f"SpinWaveAnalyzer('{self.zarr_path}', "
                   f"shape={self.data_shape}, dt={self.dt}, "
                   f"spacings={self.grid_spacings})")
        else:
            return f"SpinWaveAnalyzer('{self.zarr_path}', no data loaded)"
