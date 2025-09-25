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

from .models import DispersionResult1D, DispersionResult2D, DispersionBranch, DispersionConfig
from .utils import (
    normalize_magnetization_components,
    extract_magnetization_component,
    detrend_time_series,
    apply_window_1d,
    apply_filter_pipeline,
    k_axis_from_grid,
    fold_spectrum_1d,
    find_peaks_1d,
    group_velocity_1d,
    validate_grid_parameters
)

logger = logging.getLogger(__name__)


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
        tmax: int = 100,
        slice_info: Optional[Any] = None,
    ):
        self.zarr_path = Path(zarr_path)
        self.config = config or DispersionConfig()
        self.tmax = int(tmax)
        self.slice_info = slice_info

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
        """Load time-domain magnetization data M(t,x,y,z)."""
        possible_paths = [
            "m_layer",
            "m",
            "M",
            "magnetization",
            "m_full",
            "m_resonator",
            "table/m",
        ]

        if self.zarr_file is None:
            raise ValueError("Zarr file not loaded")

        for path in possible_paths:
            if path not in self.zarr_file:
                continue
            try:
                M_ref = self.zarr_file[path]
                if not (hasattr(M_ref, "shape") and hasattr(M_ref, "dtype")):
                    logger.info("'%s' is not an array, skipping", path)
                    continue

                logger.info("Found magnetization at '%s': shape %s, dtype %s", path, M_ref.shape, M_ref.dtype)
                self._M_ref = M_ref
                self._M_path = path
                self._configure_indexing(M_ref)

                self.M_data = self._load_reference_data(self.tmax)
                logger.info("Loaded magnetization data: shape %s", self.M_data.shape)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to access magnetization at '%s': %s", path, exc)
                self._M_ref = None
                continue
        else:
            raise ValueError(f"No magnetization data found in {self.zarr_path}")

    def _configure_indexing(self, M_ref: Any) -> None:
        """Prepare base slice/indexer information for repeated loads."""
        shape = tuple(getattr(M_ref, "shape", ()))
        if not shape:
            self._base_indexer = None
            self._time_axis_pos = None
            self._time_axis_length = None
            logger.debug("Magnetization shape unavailable; skipping indexer configuration")
            return

        base_indexer = self._normalize_slice(shape)
        self._base_indexer = base_indexer
        self._time_axis_pos, self._time_axis_length = self._resolve_time_axis(base_indexer, shape)
        if self._time_axis_pos is None:
            logger.debug("Time axis collapsed by slice; full series not available for tmax control")
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
        tmax: int,
        axis_length: int,
    ) -> Tuple[slice, bool]:
        if tmax is None or tmax <= 0 or axis_length <= 0:
            return base_slice, False

        start, stop, step = base_slice.indices(axis_length)
        if step <= 0:
            return base_slice, False

        available = max(0, (stop - start + step - 1) // step)
        if available <= tmax:
            return base_slice, False

        new_stop = start + tmax * step
        return slice(start, min(new_stop, stop), step), True

    def _indexer_for_tmax(self, tmax: Optional[int]) -> Optional[tuple]:
        if self._base_indexer is None or self._time_axis_pos is None or self._time_axis_length is None:
            return self._base_indexer

        if tmax is None:
            return self._base_indexer

        base_slice = self._base_indexer[self._time_axis_pos]
        if not isinstance(base_slice, slice):
            return self._base_indexer

        limited_slice, changed = self._limit_time_slice(base_slice, int(tmax), self._time_axis_length)
        if not changed:
            return self._base_indexer

        indexer = list(self._base_indexer)
        indexer[self._time_axis_pos] = limited_slice
        return tuple(indexer)

    def _load_reference_data(self, tmax: Optional[int]) -> np.ndarray:
        if self._M_ref is None:
            raise ValueError("No magnetization reference available")

        indexer = self._indexer_for_tmax(tmax)
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
            "Loaded %s time steps for dispersion analysis (requested tmax=%s)",
            self._loaded_time,
            tmax,
        )
        return normalized

    def _ensure_data_loaded(self, tmax: int = 100) -> None:
        """Ensure magnetization data is loaded up to requested tmax."""
        if self.M_data is not None and tmax <= self._loaded_time:
            return

        if self._M_ref is None:
            raise ValueError("No magnetization reference available for deferred loading")

        logger.info("Reloading magnetization data up to %s time steps", tmax)
        self.M_data = self._load_reference_data(tmax)

    def _extract_grid_parameters(self) -> None:
        """Extract time step and spatial grid parameters from zarr attributes."""
        if self.zarr_file is None:
            raise ValueError("Zarr file not loaded")
            
        attrs = dict(self.zarr_file.attrs)
        logger.info(f"Available zarr attributes: {list(attrs.keys())}")
        
        # Time step
        dt_keys = ['t_sampl', 'dt', 'Dt', 'timestep', 'time_step']
        for key in dt_keys:
            if key in attrs:
                attr_val = attrs[key]
                if isinstance(attr_val, (int, float)):
                    self.dt = float(attr_val)
                    logger.info(f"Extracted dt = {self.dt} s from '{key}'")
                    break
        else:
            # Try to infer from time axis if available
            if 't' in self.zarr_file:
                t = np.array(self.zarr_file['t'])
                if len(t) > 1:
                    self.dt = float(t[1] - t[0])
                    logger.info(f"Inferred dt = {self.dt} s from time axis")
            
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
                    
        # Update config with extracted values
        if hasattr(self.config, 'dt'):
            self.config.dt = self.dt
        for axis, spacing in self.grid_spacings.items():
            if hasattr(self.config, axis):
                setattr(self.config, axis, spacing)
        
        for spatial_dim, keys in spacing_keys.items():
            for key in keys:
                if key in attrs:
                    self.grid_spacings[spatial_dim] = float(attrs[key])
                    break
            
        # Use config values as fallback
        if 'dx' not in self.grid_spacings and self.config.dx:
            self.grid_spacings['dx'] = self.config.dx
        if 'dy' not in self.grid_spacings and self.config.dy:
            self.grid_spacings['dy'] = self.config.dy
        if 'dz' not in self.grid_spacings and self.config.dz:
            self.grid_spacings['dz'] = self.config.dz
            
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
        time_window: Optional[str] = None,
        space_window: Optional[str] = None,
        detrend: Optional[str] = None,
        fold_period: Optional[float] = None,
        fold_agg: Optional[str] = None,
        filters: Optional[dict[str, bool]] = None,
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
            Optional preprocessing filters (remove_static, remove_average,
            hann_time, hann_space)
            
        Returns
        -------
        DispersionResult1D
            Dispersion analysis results
        """
        # Use config defaults if not specified
        component = component or self.config.component
        avg_over_orthogonal = avg_over_orthogonal if avg_over_orthogonal is not None else self.config.avg_over_orthogonal
        time_window = time_window if time_window is not None else self.config.time_window
        space_window = space_window if space_window is not None else self.config.space_window
        detrend = detrend or self.config.detrend
        fold_period = fold_period if fold_period is not None else self.config.fold_period
        fold_agg = fold_agg or self.config.fold_agg
        
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
            
        logger.info(f"Computing 1D dispersion along {axis}-axis, component='{component}'")
        # Extract magnetization component
        signal = extract_magnetization_component(self.M_data, component)

        filters_for_pipeline = None
        if filters:
            filters_for_pipeline = {key: bool(value) for key, value in filters.items() if bool(value)}
            if filters_for_pipeline.get("remove_average") and detrend == "mean":
                logger.debug(
                    "remove_average filter skipped because detrend='mean' already subtracts the temporal mean",
                )
                filters_for_pipeline.pop("remove_average", None)
            if not filters_for_pipeline:
                filters_for_pipeline = None

        active_filters = [name for name in (filters_for_pipeline or {}).keys()]
        if active_filters:
            preview_frame = signal[:1].copy()
            logger.info(
                "Applying raw-data filters before dispersion: %s",
                ", ".join(active_filters),
            )
            signal = apply_filter_pipeline(signal, filters_for_pipeline)
            delta = float(np.linalg.norm(signal[:1] - preview_frame))
            logger.debug("Filter impact on first frame (L2 delta): %.3e", delta)
        elif filters_for_pipeline is not None:
            logger.debug("Filter configuration provided but no active flags: %s", filters_for_pipeline)

        # Preserve complex information for perpendicular analysis
        if np.iscomplexobj(signal):
            signal = signal.astype(np.complex128, copy=False)
            logger.info("Complex signal detected, preserving full complex values")
        else:
            signal = signal.astype(np.float64, copy=False)
            logger.info("Real-valued signal detected; continuing with float64 precision")

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

        if avg_over_orthogonal:
            if axis == "x":
                # Average over Z(1), Y(2) -> shape (T, X)
                spatial_signal = np.mean(signal, axis=(1, 2))
            else:  # axis == "y"
                # Average over Z(1), X(3) -> shape (T, Y)
                spatial_signal = np.mean(signal, axis=(1, 3))
        else:
            # Average only over Z, keep full orthogonal slice for local analysis
            spatial_signal = np.mean(signal, axis=1)  # -> (T, Y, X)
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

        # Spatial FFT -> k-domain signal(t, k)
        if spatial_signal.ndim == 2:
            spatial_axis = 1
        else:
            spatial_axis = 2 if axis == "x" else 1

        sig_k = np.fft.fftshift(np.fft.fft(spatial_signal, axis=spatial_axis), axes=spatial_axis)
        k_axis = k_axis_from_grid(N_space, dx, shift=True)

        # Temporal full FFT at each k
        T_len = sig_k.shape[0]
        f_axis = np.fft.fftshift(np.fft.fftfreq(T_len, self.dt))
        Sk_full = np.fft.fft(sig_k, axis=0)
        Sk_shift = np.fft.fftshift(Sk_full, axes=0)
        power = np.abs(Sk_shift) ** 2
        power = np.moveaxis(power, 0, -1)  # -> (..., Nf)

        if avg_over_orthogonal:
            S = power.astype(np.float64, copy=False)
        else:
            if axis == "x":
                # power shape: (Ny, Nx, Nf)
                S_local = power.astype(np.float64, copy=False)
                S = np.mean(S_local, axis=0)
            else:
                # power shape: (Ny, Nx, Nf) -> move orth axis to front for storage
                S_local = np.moveaxis(power, 1, 0).astype(np.float64, copy=False)
                S = np.mean(power, axis=1)

        logger.info(
            "Computed dispersion: S.shape=%s, k_range=[%.2e, %.2e], f_range=[%.1f, %.1f] Hz",
            S.shape,
            k_axis.min(),
            k_axis.max(),
            f_axis.min(),
            f_axis.max(),
        )

        notes = [f"1D dispersion along {axis}-axis"]
        if not avg_over_orthogonal:
            notes.append("Orthogonal averaging disabled; local spectra stored in S_local")

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
            notes=notes,
            S_local=S_local,
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
        sig_k = np.fft.fftshift(np.fft.fft2(signal, axes=(1, 2)), axes=(1, 2))
        ky_axis = k_axis_from_grid(sig_k.shape[1], dy, shift=True)
        kx_axis = k_axis_from_grid(sig_k.shape[2], dx, shift=True)

        T_len = sig_k.shape[0]
        use_complex = np.iscomplexobj(sig_k)
        if use_complex:
            Sk_full = np.fft.fft(sig_k, axis=0)
            Sk_pos = Sk_full[: T_len // 2 + 1]
            f_axis = np.abs(np.fft.fftfreq(T_len, self.dt)[: Sk_pos.shape[0]])
        else:
            Sk_pos = np.fft.rfft(sig_k, axis=0)
            f_axis = np.fft.rfftfreq(T_len, self.dt)

        power = np.abs(Sk_pos) ** 2
        S = power.transpose(2, 1, 0).astype(np.float64, copy=False)

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
            notes=["2D dispersion S(kx,ky,f)"]
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