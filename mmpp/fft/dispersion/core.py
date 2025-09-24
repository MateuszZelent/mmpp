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
        tmax: int = 100
    ):
        self.zarr_path = Path(zarr_path)
        self.config = config or DispersionConfig()
        self.tmax = tmax  # Store tmax for data loading
        
        # Data storage
        self.zarr_file: Optional[zarr.Group] = None
        self.M_data: Optional[np.ndarray] = None
        self._M_ref: Optional[zarr.Array | zarr.Group] = None  # Reference to zarr data
        self._M_path: Optional[str] = None
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
            logger.info(f"Opened zarr file: {self.zarr_path}")
        except Exception as e:
            logger.error(f"Failed to open zarr file {self.zarr_path}: {e}")
            raise
            
        # Load time-domain magnetization data
        self._load_magnetization()
        self._extract_grid_parameters()
        
    def _load_magnetization(self) -> None:
        """Load time-domain magnetization data M(t,x,y,z)."""
        # Look for time-domain magnetization data
        # Common paths in MuMax3 output
        possible_paths = [
            "m_layer",  # Common in layered structures
            "m",  # Standard magnetization
            "M", 
            "magnetization",
            "m_full",  # Full time series
            "m_resonator",  # Resonator data
            "table/m",  # Sometimes in table format
        ]
        
        if self.zarr_file is None:
            raise ValueError("Zarr file not loaded")
            
        for path in possible_paths:
            if path in self.zarr_file:
                try:
                    # Check if it's an array (data) vs group (nested structure)
                    M_ref = self.zarr_file[path]
                    if hasattr(M_ref, 'shape') and hasattr(M_ref, 'dtype'):
                        # It's an array - load limited data immediately
                        logger.info(f"Found magnetization at '{path}': shape {M_ref.shape}, dtype {M_ref.dtype}")
                        
                        # Load only first tmax time steps to speed up analysis
                        import zarr
                        if isinstance(M_ref, zarr.Array):
                            actual_tmax = min(self.tmax, M_ref.shape[0])
                            logger.info(f"Loading first {actual_tmax} time steps...")
                            M_raw = np.array(M_ref[:actual_tmax])
                        else:
                            logger.warning("Not a zarr Array, loading all data")
                            M_raw = np.array(M_ref)
                        
                        # Normalize shape to (T, Z, Y, X, 3)
                        self.M_data = normalize_magnetization_components(M_raw)
                        logger.info(f"Loaded magnetization data: shape {self.M_data.shape}")
                        
                        break
                    else:
                        logger.info(f"'{path}' is not an array, skipping")
                except Exception as e:
                    logger.warning(f"Failed to access magnetization at '{path}': {e}")
                    continue
        else:
            raise ValueError(f"No magnetization data found in {self.zarr_path}")
    
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
    
    def _ensure_data_loaded(self, tmax: int = 100) -> None:
        """Load magnetization data if not already loaded.
        
        Parameters
        ----------
        tmax : int
            Maximum number of time steps to load (default 100 for faster loading)
        """
        if self.M_data is None and self._M_ref is not None:
            logger.info(f"Loading magnetization data from {self._M_path} (first {tmax} time steps)...")
            
            # Check if it's an Array with slicing support
            try:
                import zarr
                if isinstance(self._M_ref, zarr.Array):
                    actual_tmax = min(tmax, self._M_ref.shape[0])
                    M_raw = self._M_ref[:actual_tmax]  # zarr Array supports slicing
                    M_raw = np.array(M_raw)
                    logger.info(f"Loaded {actual_tmax} time steps from zarr Array")
                else:
                    # Fallback: load everything
                    M_raw = np.array(self._M_ref)
                    logger.info("Loaded all data (not a zarr Array)")
            except Exception as e:
                logger.warning(f"Error during selective loading: {e}, loading all data")
                M_raw = np.array(self._M_ref)
                
            self.M_data = normalize_magnetization_components(M_raw)
            logger.info(f"Final magnetization data shape: {self.M_data.shape}")
    
    def compute_dispersion_1d(
        self,
        axis: str = "x",
        component: Optional[str] = None,
        avg_over_orthogonal: Optional[bool] = None,
        time_window: Optional[str] = None,
        space_window: Optional[str] = None,
        detrend: Optional[str] = None,
        fold_period: Optional[float] = None,
        fold_agg: Optional[str] = None
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
        
        # Ensure real dtype for FFT
        if np.iscomplexobj(signal):  # complex
            logger.warning("Complex data detected, taking real part")
            signal = np.real(signal)
        signal = signal.astype(np.float64)  # Ensure float64 for FFT
        logger.info(f"Signal dtype: {signal.dtype}, shape: {signal.shape}")
        
        # Detrend over time (axis 0)
        signal = detrend_time_series(signal, axis=0, method=detrend)
        
        # Apply time window
        signal = apply_window_1d(signal, axis=0, window=time_window)
        
        # Apply spatial window 
        signal = apply_window_1d(signal, axis=space_axis, window=space_window)
        
        # Average over orthogonal axes if requested
        if avg_over_orthogonal:
            if axis == "x":
                # Average over Z(1), Y(2) -> shape (T, X) 
                signal = np.mean(signal, axis=(1, 2))
            else:  # axis == "y"
                # Average over Z(1), X(3) -> shape (T, Y)
                signal = np.mean(signal, axis=(1, 3))
        else:
            # Just average over Z, keep other spatial dimension
            signal = np.mean(signal, axis=1)  # -> (T, Y, X)
            if axis == "x":
                signal = np.mean(signal, axis=1)  # -> (T, X)
            else:
                signal = np.mean(signal, axis=2)  # -> (T, Y)
                
        # Spatial FFT -> k-domain signal(t, k)
        sig_k = np.fft.fftshift(np.fft.fft(signal, axis=1), axes=1)
        k_axis = k_axis_from_grid(N_space, dx, shift=True)
        
        # Temporal rFFT at each k
        T_len = sig_k.shape[0]
        Nf = T_len // 2 + 1
        f_axis = np.fft.rfftfreq(T_len, self.dt)
        
        S = np.zeros((N_space, Nf), dtype=np.float64)
        for i in range(N_space):
            sk_t = sig_k[:, i]  # This is complex from spatial FFT
            # Ensure we handle complex data properly for temporal FFT
            if np.iscomplexobj(sk_t):
                # For complex signal, use full FFT then take positive frequencies
                Sk_full = np.fft.fft(sk_t, axis=0)
                Sk = Sk_full[:Nf]  # Take first half (positive frequencies)
            else:
                Sk = np.fft.rfft(sk_t, axis=0)
            S[i, :] = np.abs(Sk) ** 2
            
        logger.info(f"Computed dispersion: S.shape={S.shape}, k_range=[{k_axis.min():.2e}, {k_axis.max():.2e}], f_range=[{f_axis.min():.1f}, {f_axis.max():.1f}] Hz")
        
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
            notes=[f"1D dispersion along {axis}-axis"]
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
        
        # Temporal rFFT for each (kx, ky)
        T_len = sig_k.shape[0]
        Nf = T_len // 2 + 1
        f_axis = np.fft.rfftfreq(T_len, self.dt)
        
        S = np.zeros((sig_k.shape[2], sig_k.shape[1], Nf), dtype=np.float64)  # (Nkx, Nky, Nf)
        
        for iy in range(sig_k.shape[1]):
            for ix in range(sig_k.shape[2]):
                sk_t = sig_k[:, iy, ix]  # This is complex from spatial FFT
                # Handle complex data properly for temporal FFT
                if np.iscomplexobj(sk_t):
                    # For complex signal, use full FFT then take positive frequencies
                    Sk_full = np.fft.fft(sk_t, axis=0)
                    Sk = Sk_full[:Nf]  # Take first half (positive frequencies)
                else:
                    Sk = np.fft.rfft(sk_t, axis=0)
                S[ix, iy, :] = np.abs(Sk) ** 2
                
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