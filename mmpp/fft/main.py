"""
Main FFT Analysis Module

Core FFT analysis functionality for time series data from micromagnetic simulations.
"""

from typing import Optional, Dict, List, Union, Any, Tuple
import numpy as np
from dataclasses import dataclass

# Import dependencies with error handling
try:
    import scipy.fft
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import numpy.fft
    NUMPY_FFT_AVAILABLE = True
except ImportError:
    NUMPY_FFT_AVAILABLE = False

from pyzfn import Pyzfn


@dataclass
class FFTConfig:
    """Configuration for FFT analysis operations."""
    
    window_function: str = "hann"  # Window function for spectral analysis
    overlap: float = 0.5  # Overlap for windowed analysis
    nfft: Optional[int] = None  # FFT length (None for auto)
    detrend: str = "linear"  # Detrending method
    scaling: str = "density"  # Scaling for PSD ('density' or 'spectrum')
    engine: str = "scipy"  # FFT engine ('scipy', 'numpy')
    cache_results: bool = True  # Whether to cache FFT results
    frequency_range: Optional[Tuple[float, float]] = None  # Frequency range to analyze
    zero_padding: bool = True  # Whether to apply zero padding
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        valid_windows = ['hann', 'hamming', 'blackman', 'bartlett', 'flattop']
        if self.window_function not in valid_windows:
            raise ValueError(f"Invalid window function. Must be one of: {valid_windows}")
        
        valid_engines = ['scipy', 'numpy']
        if self.engine not in valid_engines:
            raise ValueError(f"Invalid engine. Must be one of: {valid_engines}")


@dataclass  
class FFTResult:
    """Container for FFT analysis results."""
    
    frequencies: np.ndarray
    power_spectrum: np.ndarray
    phase_spectrum: np.ndarray
    complex_spectrum: np.ndarray
    metadata: Dict[str, Any]
    config: FFTConfig
    
    @property
    def peak_frequency(self) -> float:
        """Get the frequency with maximum power."""
        peak_idx = np.argmax(self.power_spectrum)
        return self.frequencies[peak_idx]
    
    @property
    def peak_power(self) -> float:
        """Get the maximum power."""
        return np.max(self.power_spectrum)
    
    def get_frequency_range(self, f_min: float, f_max: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extract power spectrum within frequency range."""
        mask = (self.frequencies >= f_min) & (self.frequencies <= f_max)
        return self.frequencies[mask], self.power_spectrum[mask]


class FFTAnalyzer:
    """
    Advanced FFT analyzer for micromagnetic simulation data.
    
    Inherits functionality from MMPP plotting system and extends it with
    comprehensive frequency domain analysis capabilities.
    """
    
    def __init__(self, results: Union[List[Any], Any], mmpp_instance: Optional[Any] = None):
        """
        Initialize FFT analyzer.
        
        Parameters:
        -----------
        results : List or single result
            ZarrJobResult objects to analyze
        mmpp_instance : MMPP, optional
            Reference to parent MMPP instance
        """
        # Handle both single results and lists
        if not isinstance(results, list):
            self.results = [results]
        else:
            self.results = results
            
        self.mmpp = mmpp_instance
        self.config = FFTConfig()
        
        # Check dependencies
        if not (SCIPY_AVAILABLE or NUMPY_FFT_AVAILABLE):
            raise ImportError("FFT analysis requires scipy or numpy. Install with: pip install scipy")
        
        # Cache for FFT results
        self._fft_cache: Dict[str, FFTResult] = {}
    
    def configure(self, **kwargs) -> 'FFTAnalyzer':
        """
        Configure FFT analysis settings.
        
        Parameters:
        -----------
        **kwargs : Any
            Configuration options matching FFTConfig fields
            
        Returns:
        --------
        FFTAnalyzer
            Self for method chaining
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown configuration option '{key}'")
        
        return self
    
    def _load_pyzfn_job(self, result) -> Pyzfn:
        """Load a Pyzfn job from a result."""
        return Pyzfn(result.path)
    
    def _extract_time_series(self, job: Pyzfn, dataset_name: str, 
                           comp: Optional[Union[str, int]] = None,
                           average: Optional[Tuple[Any, ...]] = None,
                           time_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Extract time series data for FFT analysis.
        
        Parameters:
        -----------
        job : Pyzfn
            Pyzfn job instance
        dataset_name : str
            Dataset name (e.g., 'm_z11')
        comp : Union[str, int], optional
            Component to extract (x/0, y/1, z/2)
        average : tuple, optional
            Axes to average over
        time_range : tuple, optional
            Time range to extract (t_min, t_max)
            
        Returns:
        --------
        tuple
            (time_data, signal_data, metadata)
        """
        try:
            # Get dataset
            dataset = getattr(job, dataset_name)
            
            # Extract time data
            if hasattr(job, 't'):
                time_data = job.t[...]
            elif 't' in dataset.attrs:
                time_data = dataset.attrs['t']
            else:
                # Generate time array from dt if available
                dt = job.attrs.get('dt', 1e-12)  # Default to 1 ps
                time_data = np.arange(len(dataset)) * dt
            
            # Extract signal data
            signal_data = dataset[...]
            
            # Select component if specified
            if comp is not None:
                if isinstance(comp, str):
                    comp_map = {'x': 0, 'y': 1, 'z': 2}
                    comp_idx = comp_map.get(comp.lower(), 2)
                else:
                    comp_idx = int(comp)
                
                if signal_data.ndim > comp_idx:
                    signal_data = signal_data[..., comp_idx]
            
            # Apply averaging if specified
            if average is not None:
                if isinstance(average, (list, tuple)):
                    avg_axes = tuple(ax for ax in average if ax < signal_data.ndim)
                    if avg_axes:
                        signal_data = np.mean(signal_data, axis=avg_axes)
            
            # Apply time range if specified
            if time_range is not None:
                t_min, t_max = time_range
                mask = (time_data >= t_min) & (time_data <= t_max)
                time_data = time_data[mask]
                if signal_data.ndim == 1:
                    signal_data = signal_data[mask]
                else:
                    signal_data = signal_data[mask, ...]
            
            # Metadata
            metadata = {
                'path': job.path,
                'dataset': dataset_name,
                'component': comp,
                'averaged_axes': average,
                'time_range': time_range,
                'dt': np.mean(np.diff(time_data)) if len(time_data) > 1 else 1e-12,
                'duration': time_data[-1] - time_data[0] if len(time_data) > 1 else 0,
                'n_samples': len(time_data),
                'sampling_rate': 1.0 / np.mean(np.diff(time_data)) if len(time_data) > 1 else 1e12
            }
            
            return time_data, signal_data, metadata
            
        except Exception as e:
            print(f"Error extracting time series from {job.path}: {e}")
            return None, None, None
    
    def _apply_window(self, signal: np.ndarray) -> np.ndarray:
        """Apply window function to signal."""
        if self.config.window_function == 'hann':
            window = np.hanning(len(signal))
        elif self.config.window_function == 'hamming':
            window = np.hamming(len(signal))
        elif self.config.window_function == 'blackman':
            window = np.blackman(len(signal))
        elif self.config.window_function == 'bartlett':
            window = np.bartlett(len(signal))
        elif self.config.window_function == 'flattop':
            window = np.ones(len(signal))  # Placeholder, implement flattop if needed
        else:
            window = np.ones(len(signal))  # No windowing
        
        return signal * window
    
    def _compute_fft(self, time_data: np.ndarray, signal_data: np.ndarray, 
                    metadata: Dict[str, Any]) -> FFTResult:
        """
        Compute FFT of the signal.
        
        Parameters:
        -----------
        time_data : np.ndarray
            Time array
        signal_data : np.ndarray  
            Signal array
        metadata : Dict[str, Any]
            Metadata dictionary
            
        Returns:
        --------
        FFTResult
            FFT analysis results
        """
        # Ensure 1D signal
        if signal_data.ndim > 1:
            signal_data = signal_data.flatten()
        
        # Apply detrending
        if self.config.detrend == 'linear':
            signal_data = scipy.signal.detrend(signal_data, type='linear') if SCIPY_AVAILABLE else signal_data
        elif self.config.detrend == 'constant':
            signal_data = signal_data - np.mean(signal_data)
        
        # Apply window function
        windowed_signal = self._apply_window(signal_data)
        
        # Determine FFT length
        nfft = self.config.nfft or len(windowed_signal)
        if self.config.zero_padding and nfft > len(windowed_signal):
            # Zero pad the signal
            padded_signal = np.zeros(nfft)
            padded_signal[:len(windowed_signal)] = windowed_signal
            windowed_signal = padded_signal
        
        # Compute FFT using specified engine
        if self.config.engine == 'scipy' and SCIPY_AVAILABLE:
            complex_spectrum = scipy.fft.fft(windowed_signal, n=nfft)
            frequencies = scipy.fft.fftfreq(nfft, d=metadata['dt'])
        else:
            complex_spectrum = np.fft.fft(windowed_signal, n=nfft)
            frequencies = np.fft.fftfreq(nfft, d=metadata['dt'])
        
        # Take only positive frequencies
        positive_freq_mask = frequencies >= 0
        frequencies = frequencies[positive_freq_mask]
        complex_spectrum = complex_spectrum[positive_freq_mask]
        
        # Compute power and phase spectra
        power_spectrum = np.abs(complex_spectrum)**2
        phase_spectrum = np.angle(complex_spectrum)
        
        # Apply frequency range filter if specified
        if self.config.frequency_range is not None:
            f_min, f_max = self.config.frequency_range
            freq_mask = (frequencies >= f_min) & (frequencies <= f_max)
            frequencies = frequencies[freq_mask]
            power_spectrum = power_spectrum[freq_mask]
            phase_spectrum = phase_spectrum[freq_mask]
            complex_spectrum = complex_spectrum[freq_mask]
        
        # Create result
        fft_metadata = metadata.copy()
        fft_metadata.update({
            'fft_length': nfft,
            'window_function': self.config.window_function,
            'frequency_resolution': frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0,
            'max_frequency': np.max(frequencies),
            'engine': self.config.engine
        })
        
        return FFTResult(
            frequencies=frequencies,
            power_spectrum=power_spectrum,
            phase_spectrum=phase_spectrum,
            complex_spectrum=complex_spectrum,
            metadata=fft_metadata,
            config=self.config
        )
    
    def analyze_single(self, result_index: int = 0, dataset_name: str = "m_z11",
                      comp: Optional[Union[str, int]] = None,
                      average: Optional[Tuple[Any, ...]] = None,
                      time_range: Optional[Tuple[float, float]] = None,
                      **kwargs) -> FFTResult:
        """
        Analyze a single result with FFT.
        
        Parameters:
        -----------
        result_index : int, optional
            Index of result to analyze (default: 0)
        dataset_name : str, optional
            Dataset name (default: "m_z11")
        comp : Union[str, int], optional
            Component to analyze
        average : tuple, optional
            Axes to average over
        time_range : tuple, optional
            Time range to analyze
        **kwargs : Any
            Additional configuration options
            
        Returns:
        --------
        FFTResult
            FFT analysis results
        """
        # Update config with kwargs
        self.configure(**kwargs)
        
        if result_index >= len(self.results):
            raise IndexError(f"Result index {result_index} out of range")
        
        result = self.results[result_index]
        
        # Create cache key
        cache_key = f"{result.path}_{dataset_name}_{comp}_{average}_{time_range}_{hash(str(self.config))}"
        
        # Check cache
        if self.config.cache_results and cache_key in self._fft_cache:
            return self._fft_cache[cache_key]
        
        # Load data
        job = self._load_pyzfn_job(result)
        time_data, signal_data, metadata = self._extract_time_series(
            job, dataset_name, comp, average, time_range
        )
        
        if time_data is None or signal_data is None:
            raise ValueError(f"Could not extract time series data from {result.path}")
        
        # Compute FFT
        fft_result = self._compute_fft(time_data, signal_data, metadata)
        
        # Cache result
        if self.config.cache_results:
            self._fft_cache[cache_key] = fft_result
        
        return fft_result
    
    def analyze_all(self, dataset_name: str = "m_z11", **kwargs) -> List[FFTResult]:
        """
        Analyze all results with FFT.
        
        Parameters:
        -----------
        dataset_name : str, optional
            Dataset name (default: "m_z11")
        **kwargs : Any
            Additional parameters passed to analyze_single
            
        Returns:
        --------
        List[FFTResult]
            List of FFT results for all datasets
        """
        results = []
        for i in range(len(self.results)):
            try:
                fft_result = self.analyze_single(i, dataset_name, **kwargs)
                results.append(fft_result)
            except Exception as e:
                print(f"Error analyzing result {i}: {e}")
                continue
        
        return results
    
    def clear_cache(self) -> None:
        """Clear FFT result cache."""
        self._fft_cache.clear()
        print("FFT cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about FFT cache."""
        return {
            'cached_results': len(self._fft_cache),
            'cache_enabled': self.config.cache_results,
            'cache_keys': list(self._fft_cache.keys())
        }
    
    def __repr__(self) -> str:
        return f"FFTAnalyzer({len(self.results)} results, engine={self.config.engine})"
