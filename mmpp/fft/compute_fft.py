"""
FFT Computation Module

Core FFT computation functionality moved from old_fft_module.py and main.py.
Provides low-level FFT calculations without user interface elements.
"""

from typing import Optional, Dict, List, Union, Any, Tuple, Literal
import numpy as np
import time
from dataclasses import dataclass

# Import dependencies with error handling
try:
    import scipy.signal
    import scipy.fft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pyfftw
    PYFFTW_AVAILABLE = True
    # Configure pyFFTW if available
    pyfftw.config.NUM_THREADS = 4
    pyfftw.config.PLANNER_EFFORT = 'FFTW_PATIENT'
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)
except ImportError:
    PYFFTW_AVAILABLE = False

from pyzfn import Pyzfn


# Type hints
WINDOW_TYPES = Literal['none', 'hann', 'hamming', 'blackman', 'bartlett', 'kaiser', 'tukey', 'gaussian']
FILTER_TYPES = Literal['none', 'remove_mean', 'remove_static', 'detrend_linear', 'remove_mean_and_static']
FFT_ENGINES = Literal['numpy', 'pyfftw', 'scipy', 'auto']


@dataclass
class FFTComputeConfig:
    """Configuration for FFT computations."""
    
    window_function: WINDOW_TYPES = "hann"
    filter_type: FILTER_TYPES = "remove_mean"
    fft_engine: FFT_ENGINES = "auto"
    zero_padding: bool = True
    nfft: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if not SCIPY_AVAILABLE and self.fft_engine == 'scipy':
            self.fft_engine = 'numpy'
        if not PYFFTW_AVAILABLE and self.fft_engine == 'pyfftw':
            self.fft_engine = 'numpy'


@dataclass
class FFTComputeResult:
    """Result of FFT computation."""
    
    frequencies: np.ndarray
    spectrum: np.ndarray
    metadata: Dict[str, Any]
    config: FFTComputeConfig
    
    @property
    def peak_frequency(self) -> float:
        """Get frequency with maximum power."""
        peak_idx = np.argmax(self.spectrum)
        return self.frequencies[peak_idx]
    
    def save_to_zarr(self, zarr_path: str, dataset_name: str = "fft", force: bool = False) -> None:
        """
        Save FFT result to zarr file.
        
        Parameters:
        -----------
        zarr_path : str
            Path to zarr file
        dataset_name : str, optional
            Base dataset name (default: "fft")
        force : bool, optional
            Overwrite existing data (default: False)
        """
        import zarr
        import shutil
        
        # Open zarr file
        z = zarr.open(zarr_path, mode='a')
        
        # Create dataset path
        fft_path = f"fft/{dataset_name}"
        
        # Remove existing if force=True
        if force and fft_path in z:
            del z[fft_path]
        
        # Create fft group if it doesn't exist
        if "fft" not in z:
            fft_main_group = z.create_group("fft")
        else:
            fft_main_group = z["fft"]
        
        # Create dataset group within fft/
        if dataset_name not in fft_main_group:
            fft_group = fft_main_group.create_group(dataset_name)
            print(f"Created new FFT dataset group: fft/{dataset_name}")
        else:
            fft_group = fft_main_group[dataset_name]
            if not force:
                print(f"FFT dataset fft/{dataset_name} already exists. Use force=True to overwrite.")
                return
            print(f"Overwriting existing FFT dataset: fft/{dataset_name}")
        
        # Disable chunking for FFT data to avoid unnecessary fragmentation
        spectrum_chunks = None
        freq_chunks = None
        
        # Save spectrum data without chunking
        fft_group.create_dataset("spectrum", data=self.spectrum, 
                                chunks=spectrum_chunks, overwrite=force)
        fft_group.create_dataset("frequencies", data=self.frequencies, 
                                chunks=freq_chunks, overwrite=force)
        
        # Save metadata as attributes
        for key, value in self.metadata.items():
            fft_group.attrs[key] = value
        
        # Save config as attributes
        fft_group.attrs["window_function"] = self.config.window_function
        fft_group.attrs["filter_type"] = self.config.filter_type
        fft_group.attrs["fft_engine"] = self.config.fft_engine
        fft_group.attrs["zero_padding"] = self.config.zero_padding
        if self.config.nfft is not None:
            fft_group.attrs["nfft"] = self.config.nfft


class FFTCompute:
    """
    Core FFT computation engine.
    
    Handles low-level FFT calculations without user interface elements.
    """
    
    def __init__(self):
        """Initialize FFT compute engine."""
        self.config = FFTComputeConfig()
        
        # Available window functions
        self.AVAILABLE_WINDOWS = {
            'none': None,
            'hann': scipy.signal.windows.hann if SCIPY_AVAILABLE else np.hanning,
            'hamming': scipy.signal.windows.hamming if SCIPY_AVAILABLE else np.hamming,
            'blackman': scipy.signal.windows.blackman if SCIPY_AVAILABLE else np.blackman,
            'bartlett': scipy.signal.windows.bartlett if SCIPY_AVAILABLE else np.bartlett,
            'kaiser': lambda N: scipy.signal.windows.kaiser(N, beta=8.6) if SCIPY_AVAILABLE else np.kaiser(N, 8.6),
            'tukey': lambda N: scipy.signal.windows.tukey(N, alpha=0.25) if SCIPY_AVAILABLE else np.ones(N),
            'gaussian': lambda N: scipy.signal.windows.gaussian(N, std=N/6) if SCIPY_AVAILABLE else np.ones(N)
        }
        
        # Available engines
        self.AVAILABLE_ENGINES = {
            'numpy': 'NumPy FFT (basic)',
            'scipy': 'SciPy FFT (recommended)' if SCIPY_AVAILABLE else None,
            'pyfftw': 'pyFFTW (optimized)' if PYFFTW_AVAILABLE else None,
            'auto': 'Automatic selection'
        }
        
        # Remove unavailable engines
        self.AVAILABLE_ENGINES = {k: v for k, v in self.AVAILABLE_ENGINES.items() if v is not None}
    
    def determine_engine(self, data_size: int) -> str:
        """
        Determine best FFT engine based on data size.
        
        Parameters:
        -----------
        data_size : int
            Total number of elements to transform
            
        Returns:
        --------
        str
            Selected engine name
        """
        if self.config.fft_engine != 'auto':
            return self.config.fft_engine
        
        # Heuristic selection
        if data_size < 100000:
            return 'numpy'  # Small data - NumPy is fine
        elif data_size > 1000000 and PYFFTW_AVAILABLE:
            return 'pyfftw'  # Large data - pyFFTW if available
        elif SCIPY_AVAILABLE:
            return 'scipy'  # Default to scipy if available
        else:
            return 'numpy'  # Fallback
    
    def apply_window(self, data: np.ndarray, window_type: WINDOW_TYPES) -> np.ndarray:
        """
        Apply window function to data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data (time axis first)
        window_type : str
            Window function type
            
        Returns:
        --------
        np.ndarray
            Windowed data
        """
        if window_type == 'none' or self.AVAILABLE_WINDOWS[window_type] is None:
            return data
        
        n_time = data.shape[0]
        window_func = self.AVAILABLE_WINDOWS[window_type]
        
        if callable(window_func):
            window = window_func(n_time)
        else:
            window = np.ones(n_time)
        
        # Apply window along time axis
        if data.ndim == 1:
            return data * window
        else:
            # Broadcast window to match data shape
            window_shape = [1] * data.ndim
            window_shape[0] = n_time
            window = window.reshape(window_shape)
            return data * window
    
    def apply_filter(self, data: np.ndarray, filter_type: FILTER_TYPES) -> np.ndarray:
        """
        Apply filtering to data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data (time axis first)
        filter_type : str
            Filter type
            
        Returns:
        --------
        np.ndarray
            Filtered data
        """
        if filter_type == 'none':
            return data
        elif filter_type == 'remove_mean':
            return data - np.mean(data, axis=0, keepdims=True)
        elif filter_type == 'remove_static':
            return data - data[0:1, ...]
        elif filter_type == 'detrend_linear':
            if SCIPY_AVAILABLE:
                if data.ndim == 1:
                    return scipy.signal.detrend(data)
                else:
                    # Apply detrending along time axis
                    detrended = np.zeros_like(data)
                    for idx in np.ndindex(data.shape[1:]):
                        detrended[(slice(None),) + idx] = scipy.signal.detrend(data[(slice(None),) + idx])
                    return detrended
            else:
                # Simple linear detrend without scipy
                return data - np.mean(data, axis=0, keepdims=True)
        elif filter_type == 'remove_mean_and_static':
            data_filtered = data - np.mean(data, axis=0, keepdims=True)
            return data_filtered - data_filtered[0:1, ...]
        else:
            return data
    
    def compute_fft(self, data: np.ndarray, dt: float, engine: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT using specified engine.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data (time axis first)
        dt : float
            Time step
        engine : str
            FFT engine to use
            
        Returns:
        --------
        tuple
            (frequencies, fft_data)
        """
        n = data.shape[0]
        
        if engine == 'numpy':
            fft_data = np.fft.rfft(data, axis=0)
            frequencies = np.fft.rfftfreq(n, dt)
        elif engine == 'scipy' and SCIPY_AVAILABLE:
            fft_data = scipy.fft.rfft(data, axis=0)
            frequencies = scipy.fft.rfftfreq(n, dt)
        elif engine == 'pyfftw' and PYFFTW_AVAILABLE:
            fft_data = pyfftw.interfaces.numpy_fft.rfft(data, axis=0, threads=pyfftw.config.NUM_THREADS)
            frequencies = pyfftw.interfaces.numpy_fft.rfftfreq(n, dt)
        else:
            # Fallback to numpy
            fft_data = np.fft.rfft(data, axis=0)
            frequencies = np.fft.rfftfreq(n, dt)
        
        return frequencies, fft_data
    
    def calculate_fft_method1(self, data: np.ndarray, dt: float, 
                             window: WINDOW_TYPES = 'hann',
                             filter_type: FILTER_TYPES = 'remove_mean',
                             engine: Optional[str] = None) -> FFTComputeResult:
        """
        FFT Method 1: Apply filtering and windowing, then FFT, then average spatially.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data (time, ..., components)
        dt : float
            Time step
        window : str
            Window type
        filter_type : str
            Filter type
        engine : str, optional
            FFT engine
            
        Returns:
        --------
        FFTComputeResult
            FFT computation result
        """
        start_time = time.time()
        
        # Determine engine
        selected_engine = engine or self.determine_engine(data.size)
        
        # Apply filtering
        data_filtered = self.apply_filter(data, filter_type)
        
        # Apply windowing
        data_windowed = self.apply_window(data_filtered, window)
        
        # Compute FFT
        frequencies, fft_data = self.compute_fft(data_windowed, dt, selected_engine)
        
        # Calculate magnitude spectrum
        magnitude = np.abs(fft_data)
        
        # Average over spatial dimensions (keep time and component axes)
        if magnitude.ndim > 2:  # (freq, spatial..., components)
            # Average over spatial dimensions (all except first and last)
            spatial_axes = tuple(range(1, magnitude.ndim - 1))
            if spatial_axes:
                spectrum = np.mean(magnitude, axis=spatial_axes)
            else:
                spectrum = magnitude
        else:
            spectrum = magnitude
        
        calculation_time = time.time() - start_time
        
        metadata = {
            'method': 1,
            'window': window,
            'filter_type': filter_type,
            'engine': selected_engine,
            'calculation_time': calculation_time,
            'data_shape': data.shape,
            'dt': dt,
            'frequency_resolution': frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0
        }
        
        config = FFTComputeConfig(
            window_function=window,
            filter_type=filter_type,
            fft_engine=selected_engine
        )
        
        return FFTComputeResult(
            frequencies=frequencies,
            spectrum=spectrum,
            metadata=metadata,
            config=config
        )
    
    def calculate_fft_method2(self, data: np.ndarray, dt: float,
                             window: WINDOW_TYPES = 'hann',
                             filter_type: FILTER_TYPES = 'remove_mean',
                             engine: Optional[str] = None) -> FFTComputeResult:
        """
        FFT Method 2: Apply filtering, average spatially, then windowing and FFT.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data (time, ..., components)
        dt : float
            Time step
        window : str
            Window type
        filter_type : str
            Filter type
        engine : str, optional
            FFT engine
            
        Returns:
        --------
        FFTComputeResult
            FFT computation result
        """
        start_time = time.time()
        
        # Determine engine
        selected_engine = engine or self.determine_engine(data.size)
        
        # Apply filtering
        data_filtered = self.apply_filter(data, filter_type)
        
        # Average over spatial dimensions first
        if data_filtered.ndim > 2:  # (time, spatial..., components)
            spatial_axes = tuple(range(1, data_filtered.ndim - 1))
            if spatial_axes:
                data_averaged = np.mean(data_filtered, axis=spatial_axes)
            else:
                data_averaged = data_filtered
        else:
            data_averaged = data_filtered
        
        # Apply windowing
        data_windowed = self.apply_window(data_averaged, window)
        
        # Compute FFT
        frequencies, fft_data = self.compute_fft(data_windowed, dt, selected_engine)
        
        # Calculate magnitude spectrum
        spectrum = np.abs(fft_data)
        
        calculation_time = time.time() - start_time
        
        metadata = {
            'method': 2,
            'window': window,
            'filter_type': filter_type,
            'engine': selected_engine,
            'calculation_time': calculation_time,
            'data_shape': data.shape,
            'dt': dt,
            'frequency_resolution': frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0
        }
        
        config = FFTComputeConfig(
            window_function=window,
            filter_type=filter_type,
            fft_engine=selected_engine
        )
        
        return FFTComputeResult(
            frequencies=frequencies,
            spectrum=spectrum,
            metadata=metadata,
            config=config
        )
    
    def load_data_from_zarr(self, zarr_path: str, dataset: str, z_layer: int = -1) -> Tuple[np.ndarray, float]:
        """
        Load data from zarr file.
        
        Parameters:
        -----------
        zarr_path : str
            Path to zarr file
        dataset : str
            Dataset name
        z_layer : int
            Z-layer index (-1 for last layer)
            
        Returns:
        --------
        tuple
            (data, dt) where data is the loaded array and dt is time step
        """
        job = Pyzfn(zarr_path)
        
        # Get dataset
        if hasattr(job, dataset):
            data_set = getattr(job, dataset)
        else:
            raise ValueError(f"Dataset '{dataset}' not found")
        
        # Load data
        data = data_set[...]
        
        # Handle z-layer selection
        if len(data.shape) == 5:  # (t, z, y, x, comp)
            if z_layer == -1:
                data = data[:, -1, :, :, :]  # Take last layer
            else:
                data = data[:, z_layer, :, :, :]
        elif len(data.shape) == 4:  # (t, y, x, comp)
            pass  # No z dimension
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        # Get time step
        dt = getattr(job, 't_sampl', 1e-12)
        
        return data, dt
    
    def get_available_options(self) -> Dict[str, Any]:
        """Get available configuration options."""
        return {
            'windows': list(self.AVAILABLE_WINDOWS.keys()),
            'filters': ['none', 'remove_mean', 'remove_static', 'detrend_linear', 'remove_mean_and_static'],
            'engines': list(self.AVAILABLE_ENGINES.keys()),
            'dependencies': {
                'scipy': SCIPY_AVAILABLE,
                'pyfftw': PYFFTW_AVAILABLE
            }
        }
    
    def load_existing_fft_data(self, zarr_path: str, dataset_name: str = "fft") -> Optional[FFTComputeResult]:
        """
        Load existing FFT data from zarr file.
        
        Parameters:
        -----------
        zarr_path : str
            Path to zarr file
        dataset_name : str, optional
            Dataset name (default: "fft")
            
        Returns:
        --------
        Optional[FFTComputeResult]
            Loaded FFT result or None if not found
        """
        try:
            import zarr
            z = zarr.open(zarr_path, mode='r')
            
            fft_path = f"fft/{dataset_name}"
            if fft_path not in z:
                return None
            
            fft_group = z[fft_path]
            
            # Load data
            spectrum = np.array(fft_group["spectrum"])
            frequencies = np.array(fft_group["frequencies"])
            
            # Load metadata
            metadata = dict(fft_group.attrs)
            
            # Create config from attributes
            config = FFTComputeConfig(
                window_function=metadata.pop("window_function", "hann"),
                filter_type=metadata.pop("filter_type", "remove_mean"),
                fft_engine=metadata.pop("fft_engine", "auto"),
                zero_padding=metadata.pop("zero_padding", True),
                nfft=metadata.pop("nfft", None)
            )
            
            return FFTComputeResult(
                frequencies=frequencies,
                spectrum=spectrum,
                metadata=metadata,
                config=config
            )
            
        except Exception as e:
            print(f"Warning: Could not load existing FFT data: {e}")
            return None
    
    def _verify_fft_parameters(self, existing_result: FFTComputeResult, 
                               **kwargs) -> bool:
        """
        Verify if FFT parameters match existing result.
        
        Parameters:
        -----------
        existing_result : FFTComputeResult
            Existing FFT result to compare against
        **kwargs : Any
            FFT parameters to verify
            
        Returns:
        --------
        bool
            True if parameters match, False otherwise
        """
        # Extract parameters from kwargs with defaults
        window = kwargs.get('window', self.config.window_function)
        filter_type = kwargs.get('filter_type', self.config.filter_type)
        engine = kwargs.get('engine', self.config.fft_engine)
        zero_padding = kwargs.get('zero_padding', self.config.zero_padding)
        nfft = kwargs.get('nfft', self.config.nfft)
        
        # Compare with existing config
        config_match = (
            existing_result.config.window_function == window and
            existing_result.config.filter_type == filter_type and
            existing_result.config.fft_engine == engine and
            existing_result.config.zero_padding == zero_padding and
            existing_result.config.nfft == nfft
        )
        
        # Compare metadata that affects FFT calculation
        # (add other relevant metadata fields as needed)
        metadata_keys_to_check = ['z_layer', 'source_dataset']
        metadata_match = True
        for key in metadata_keys_to_check:
            if key in kwargs and key in existing_result.metadata:
                if kwargs[key] != existing_result.metadata[key]:
                    metadata_match = False
                    break
        
        return config_match and metadata_match
    
    def calculate_fft_data(self, zarr_path: str, dataset: str, z_layer: int = -1, 
                          method: int = 1, save: bool = False, force: bool = False, 
                          save_dataset_name: Optional[str] = None, **kwargs) -> FFTComputeResult:
        """
        Calculate FFT for data from zarr file.
        
        Parameters:
        -----------
        zarr_path : str
            Path to zarr file
        dataset : str
            Dataset name
        z_layer : int
            Z-layer index (-1 for last layer)
        method : int
            FFT method (1 or 2)
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
        FFTComputeResult
            FFT computation result
        """
        print(f"[DEBUG] calculate_fft_data called with: {dataset}, z_layer={z_layer}, method={method}, save={save}, force={force}")
        
        # Generate save dataset name if not provided
        if save_dataset_name is None:
            save_dataset_name = f"{dataset}_z{z_layer}_m{method}"
        
        # Try to load existing data if not forcing recalculation
        if not force:
            print(f"Checking for existing FFT data: fft/{save_dataset_name}")
            existing_result = self.load_existing_fft_data(zarr_path, save_dataset_name)
            if existing_result is not None:
                # Verify that parameters match
                if self._verify_fft_parameters(existing_result, z_layer=z_layer, 
                                               source_dataset=dataset, **kwargs):
                    print(f"✓ Loaded existing FFT data for {save_dataset_name} (parameters verified)")
                    return existing_result
                else:
                    print(f"⚠ Existing FFT data found but parameters don't match, recalculating...")
                    force = True  # Force recalculation if parameters don't match
            else:
                print(f"No existing FFT data found, calculating new FFT...")
        else:
            print(f"Force recalculation enabled, computing new FFT...")
        
        # Load data
        print(f"Loading data from {dataset} (z_layer={z_layer})...")
        data, dt = self.load_data_from_zarr(zarr_path, dataset, z_layer)
        print(f"Data shape: {data.shape}, dt: {dt}")
        
        # Extract configuration from kwargs
        window = kwargs.get('window', self.config.window_function)
        filter_type = kwargs.get('filter_type', self.config.filter_type)
        engine = kwargs.get('engine', self.config.fft_engine)
        
        print(f"Computing FFT with method {method} (window: {window}, filter: {filter_type}, engine: {engine})...")
        
        # Calculate FFT using specified method
        if method == 1:
            result = self.calculate_fft_method1(data, dt, window, filter_type, engine)
        elif method == 2:
            result = self.calculate_fft_method2(data, dt, window, filter_type, engine)
        else:
            raise ValueError(f"Unsupported FFT method: {method}")
        
        print(f"✓ FFT calculation completed in {result.metadata.get('calculation_time', 0):.3f}s")
        
        # Add additional metadata
        result.metadata.update({
            'zarr_path': zarr_path,
            'source_dataset': dataset,
            'z_layer': z_layer,
            'save_dataset_name': save_dataset_name
        })
        
        # Save to zarr if requested
        if save:
            try:
                print(f"Saving FFT data to fft/{save_dataset_name}...")
                result.save_to_zarr(zarr_path, save_dataset_name, force=force)
                print(f"✓ Successfully saved FFT data to fft/{save_dataset_name}")
            except Exception as e:
                print(f"❌ Warning: Could not save FFT data: {e}")
        else:
            print("FFT calculation completed (not saved)")
        
        return result
