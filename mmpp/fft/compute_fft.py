"""
FFT Computation Module

Core FFT computation functionality moved from old_fft_module.py and main.py.
Provides low-level FFT calculations without user interface elements.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np

# Import psutil for memory monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import dependencies with error handling
try:
    import scipy.fft
    import scipy.signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pyfftw

    PYFFTW_AVAILABLE = True
    # Configure pyFFTW if available
    pyfftw.config.NUM_THREADS = 4
    pyfftw.config.PLANNER_EFFORT = "FFTW_PATIENT"
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)
except ImportError:
    PYFFTW_AVAILABLE = False

try:
    from ..pyzfn import Pyzfn

    PYZFN_AVAILABLE = True
except ImportError:
    Pyzfn = None  # type: ignore[assignment]
    PYZFN_AVAILABLE = False

# Import shared logging configuration
from ..cli.logging_config import get_mmpp_logger, setup_mmpp_logging

# Get logger for FFT module
log = get_mmpp_logger("mmpp.fft")

if not PYZFN_AVAILABLE:
    log.warning(
        "Pyzfn dependency not available - install pyzfn to enable FFT data loading"
    )


# Type hints
WINDOW_TYPES = Literal[
    "none", "hann", "hamming", "blackman", "bartlett", "kaiser", "tukey", "gaussian",
    "flattop", "nuttall"
]
FILTER_TYPE_OPTIONS = Literal[
    "none", "remove_mean", "remove_static", "detrend_linear", "remove_mean_and_static",
    # New filters from FMR literature:
    "savgol_smooth", "baseline_correction", "high_pass", "band_pass", "spectral_derivative"
]
FILTER_TYPES = Union[FILTER_TYPE_OPTIONS, list[FILTER_TYPE_OPTIONS]]
FFT_ENGINES = Literal["numpy", "pyfftw", "scipy", "auto"]


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
        if not SCIPY_AVAILABLE and self.fft_engine == "scipy":
            self.fft_engine = "numpy"
        if not PYFFTW_AVAILABLE and self.fft_engine == "pyfftw":
            self.fft_engine = "numpy"


@dataclass
class FFTComputeResult:
    """Result of FFT computation."""

    frequencies: np.ndarray
    spectrum: np.ndarray
    metadata: dict[str, Any]
    config: FFTComputeConfig

    @property
    def peak_frequency(self) -> float:
        """Get frequency with maximum power."""
        if self.spectrum.size == 0 or self.frequencies.size == 0:
            raise ValueError("FFT spectrum is empty; cannot determine peak frequency")

        power = np.abs(self.spectrum) ** 2
        if power.ndim > 1:
            reduction_axes = tuple(range(1, power.ndim))
            power = power.sum(axis=reduction_axes)

        peak_idx = int(np.argmax(power))
        if peak_idx >= self.frequencies.shape[0]:
            peak_idx = self.frequencies.shape[0] - 1

        return float(self.frequencies[peak_idx])

    def save_to_zarr(
        self, zarr_path: str, dataset_name: str = "fft", force: bool = False
    ) -> None:
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

        # Open zarr file
        z = zarr.open(zarr_path, mode="a")

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
            log.debug(f"Created new FFT dataset group: fft/{dataset_name}")
        else:
            fft_group = fft_main_group[dataset_name]
            if not force:
                log.warning(
                    f"FFT dataset fft/{dataset_name} already exists. Use force=True to overwrite."
                )
                return
            log.info(f"Overwriting existing FFT dataset: fft/{dataset_name}")

        # Disable chunking for FFT data to avoid unnecessary fragmentation
        spectrum_chunks = None
        freq_chunks = None

        # Save spectrum data - handle chunking properly based on array dimensions
        spectrum_chunks = None  # Let zarr decide chunking
        freq_chunks = None      # Let zarr decide chunking
        
        log.debug(f"Saving spectrum shape: {self.spectrum.shape}, frequencies shape: {self.frequencies.shape}")

        # Save spectrum data
        fft_group.create_dataset(
            "spectrum", 
            data=self.spectrum,
            shape=self.spectrum.shape,
            dtype=self.spectrum.dtype,
            overwrite=force
        )
        fft_group.create_dataset(
            "frequencies", 
            data=self.frequencies,
            shape=self.frequencies.shape,
            dtype=self.frequencies.dtype,
            overwrite=force
        )

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

    def __init__(self, debug: bool = False):
        """Initialize FFT compute engine.

        Parameters:
        -----------
        debug : bool, optional
            Enable debug logging (default: False)
        """
        self.config = FFTComputeConfig()

        # Set logging level based on debug flag
        setup_mmpp_logging(debug=debug, logger_name="mmpp.fft")
        if debug:
            log.debug("FFT debug logging enabled")

        # Available window functions
        self.AVAILABLE_WINDOWS = {
            "none": None,
            "hann": scipy.signal.windows.hann if SCIPY_AVAILABLE else np.hanning,
            "hamming": scipy.signal.windows.hamming if SCIPY_AVAILABLE else np.hamming,
            "blackman": (
                scipy.signal.windows.blackman if SCIPY_AVAILABLE else np.blackman
            ),
            "bartlett": (
                scipy.signal.windows.bartlett if SCIPY_AVAILABLE else np.bartlett
            ),
            "kaiser": lambda N: (
                scipy.signal.windows.kaiser(N, beta=8.6)
                if SCIPY_AVAILABLE
                else np.kaiser(N, 8.6)
            ),
            "tukey": lambda N: (
                scipy.signal.windows.tukey(N, alpha=0.25)
                if SCIPY_AVAILABLE
                else np.ones(N)
            ),
            "gaussian": lambda N: (
                scipy.signal.windows.gaussian(N, std=N / 6)
                if SCIPY_AVAILABLE
                else np.ones(N)
            ),
            # New windows for FMR analysis
            "flattop": (
                scipy.signal.windows.flattop if SCIPY_AVAILABLE else lambda N: np.ones(N)
            ),
            "nuttall": (
                scipy.signal.windows.nuttall if SCIPY_AVAILABLE else np.blackman
            ),
        }

        # Available engines
        self.AVAILABLE_ENGINES = {
            "numpy": "NumPy FFT (basic)",
            "scipy": "SciPy FFT (recommended)" if SCIPY_AVAILABLE else None,
            "pyfftw": "pyFFTW (optimized)" if PYFFTW_AVAILABLE else None,
            "auto": "Automatic selection",
        }

        # Remove unavailable engines
        self.AVAILABLE_ENGINES = {
            k: v for k, v in self.AVAILABLE_ENGINES.items() if v is not None
        }

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
        if self.config.fft_engine != "auto":
            return self.config.fft_engine

        # Heuristic selection
        if data_size < 100000:
            return "numpy"  # Small data - NumPy is fine
        elif data_size > 1000000 and PYFFTW_AVAILABLE:
            return "pyfftw"  # Large data - pyFFTW if available
        elif SCIPY_AVAILABLE:
            return "scipy"  # Default to scipy if available
        else:
            return "numpy"  # Fallback

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
        if window_type == "none" or self.AVAILABLE_WINDOWS[window_type] is None:
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
        filter_type : str or list of str
            Filter type or list of filter types to apply sequentially

        Returns:
        --------
        np.ndarray
            Filtered data
        """
        # Handle list of filters - apply sequentially
        if isinstance(filter_type, list):
            result = data
            for single_filter in filter_type:
                result = self._apply_single_filter(result, single_filter)
            return result
        else:
            return self._apply_single_filter(data, filter_type)

    def _apply_single_filter(self, data: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply a single filter to data."""
        if filter_type == "none":
            return data
        elif filter_type == "remove_mean":
            return data - np.mean(data, axis=0, keepdims=True)
        elif filter_type == "remove_static":
            return data - data[0:1, ...]
        elif filter_type == "detrend_linear":
            if SCIPY_AVAILABLE:
                if data.ndim == 1:
                    return scipy.signal.detrend(data)
                else:
                    # Apply detrending along time axis
                    detrended = np.zeros_like(data)
                    for idx in np.ndindex(data.shape[1:]):
                        detrended[(slice(None),) + idx] = scipy.signal.detrend(
                            data[(slice(None),) + idx]
                        )
                    return detrended
            else:
                # Simple linear detrend without scipy
                return data - np.mean(data, axis=0, keepdims=True)
        elif filter_type == "remove_mean_and_static":
            data_filtered = data - np.mean(data, axis=0, keepdims=True)
            return data_filtered - data_filtered[0:1, ...]
        # New filters from FMR literature
        elif filter_type == "savgol_smooth":
            return self._apply_savgol_smooth(data)
        elif filter_type == "baseline_correction":
            return self._apply_baseline_correction(data)
        elif filter_type == "high_pass":
            return self._apply_high_pass(data)
        elif filter_type == "band_pass":
            return self._apply_band_pass(data)
        elif filter_type == "spectral_derivative":
            return self._apply_spectral_derivative(data)
        else:
            log.warning(f"Unknown filter type: {filter_type}, returning data unchanged")
            return data

    def _apply_savgol_smooth(
        self, data: np.ndarray, window_length: int = 11, polyorder: int = 3
    ) -> np.ndarray:
        """Apply Savitzky-Golay smoothing filter.
        
        Reduces noise while preserving signal shape and peak positions.
        Common in spectroscopic data processing.
        """
        if not SCIPY_AVAILABLE:
            log.warning("Savitzky-Golay requires scipy, returning data unchanged")
            return data
        
        # Ensure window length is odd and not larger than data
        n_time = data.shape[0]
        window_length = min(window_length, n_time // 2 * 2 - 1)
        if window_length < 5:
            log.warning("Data too short for Savitzky-Golay filter")
            return data
        if window_length % 2 == 0:
            window_length -= 1
        polyorder = min(polyorder, window_length - 1)
        
        if data.ndim == 1:
            return scipy.signal.savgol_filter(data, window_length, polyorder)
        else:
            result = np.zeros_like(data)
            for idx in np.ndindex(data.shape[1:]):
                result[(slice(None),) + idx] = scipy.signal.savgol_filter(
                    data[(slice(None),) + idx], window_length, polyorder
                )
            return result

    def _apply_baseline_correction(
        self, data: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10
    ) -> np.ndarray:
        """Apply asymmetric least squares baseline correction.
        
        Removes slowly varying baseline drift common in VNA-FMR spectra.
        Uses iterative weighted least squares fitting.
        
        Parameters:
        -----------
        lam : float
            Smoothness penalty (larger = smoother baseline)
        p : float
            Asymmetry factor (0 < p < 1, smaller = penalize positive residuals more)
        niter : int
            Number of iterations
        """
        if not SCIPY_AVAILABLE:
            # Fallback to simple polynomial baseline
            return data - np.mean(data, axis=0, keepdims=True)
        
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        
        def baseline_als_1d(y, lam, p, niter):
            """ALS baseline for 1D array."""
            L = len(y)
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
            w = np.ones(L)
            for _ in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lam * D.dot(D.T)
                z = spsolve(Z, w * y)
                w = p * (y > z) + (1 - p) * (y <= z)
            return z
        
        if data.ndim == 1:
            baseline = baseline_als_1d(data, lam, p, niter)
            return data - baseline
        else:
            result = np.zeros_like(data)
            for idx in np.ndindex(data.shape[1:]):
                y = data[(slice(None),) + idx]
                baseline = baseline_als_1d(y, lam, p, niter)
                result[(slice(None),) + idx] = y - baseline
            return result

    def _apply_high_pass(
        self, data: np.ndarray, cutoff_fraction: float = 0.01
    ) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency components.
        
        Useful for removing DC offset and slow drifts.
        Uses FFT-based filtering for efficiency.
        
        Parameters:
        -----------
        cutoff_fraction : float
            Cutoff as fraction of Nyquist frequency (0-1)
        """
        if data.ndim == 1:
            return self._high_pass_1d(data, cutoff_fraction)
        else:
            result = np.zeros_like(data)
            for idx in np.ndindex(data.shape[1:]):
                result[(slice(None),) + idx] = self._high_pass_1d(
                    data[(slice(None),) + idx], cutoff_fraction
                )
            return result

    def _high_pass_1d(self, y: np.ndarray, cutoff_fraction: float) -> np.ndarray:
        """High-pass filter for 1D array using FFT."""
        n = len(y)
        fft = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(n)
        
        # Create smooth transition (Butterworth-like)
        cutoff = cutoff_fraction
        filter_shape = 1 - 1 / (1 + (freqs / max(cutoff, 1e-10)) ** 4)
        
        fft_filtered = fft * filter_shape
        return np.fft.irfft(fft_filtered, n=n)

    def _apply_band_pass(
        self, data: np.ndarray, low_fraction: float = 0.01, high_fraction: float = 0.9
    ) -> np.ndarray:
        """Apply band-pass filter to keep frequencies in specified range.
        
        Useful for isolating FMR resonance frequency range.
        
        Parameters:
        -----------
        low_fraction : float
            Low cutoff as fraction of Nyquist frequency
        high_fraction : float
            High cutoff as fraction of Nyquist frequency
        """
        if data.ndim == 1:
            return self._band_pass_1d(data, low_fraction, high_fraction)
        else:
            result = np.zeros_like(data)
            for idx in np.ndindex(data.shape[1:]):
                result[(slice(None),) + idx] = self._band_pass_1d(
                    data[(slice(None),) + idx], low_fraction, high_fraction
                )
            return result

    def _band_pass_1d(
        self, y: np.ndarray, low_fraction: float, high_fraction: float
    ) -> np.ndarray:
        """Band-pass filter for 1D array using FFT."""
        n = len(y)
        fft = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(n)
        
        # High-pass component
        hp = 1 - 1 / (1 + (freqs / max(low_fraction, 1e-10)) ** 4)
        # Low-pass component  
        lp = 1 / (1 + (freqs / max(high_fraction, 1e-10)) ** 4)
        
        filter_shape = hp * lp
        fft_filtered = fft * filter_shape
        return np.fft.irfft(fft_filtered, n=n)

    def _apply_spectral_derivative(
        self, data: np.ndarray, order: int = 1
    ) -> np.ndarray:
        """Apply spectral derivative to enhance peaks and resolve overlaps.
        
        First derivative helps identify peak positions.
        Second derivative enhances narrow peaks over broad backgrounds.
        
        Parameters:
        -----------
        order : int
            Derivative order (1 or 2)
        """
        if data.ndim == 1:
            return np.gradient(data) if order == 1 else np.gradient(np.gradient(data))
        else:
            result = np.zeros_like(data)
            for idx in np.ndindex(data.shape[1:]):
                if order == 1:
                    result[(slice(None),) + idx] = np.gradient(data[(slice(None),) + idx])
                else:
                    result[(slice(None),) + idx] = np.gradient(
                        np.gradient(data[(slice(None),) + idx])
                    )
            return result

    def compute_fft(
        self,
        data: np.ndarray,
        dt: float,
        engine: str,
        *,
        zero_padding: bool,
        nfft: Optional[int],
    ) -> tuple[np.ndarray, np.ndarray, int]:
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
        zero_padding : bool
            Whether to apply zero padding when determining FFT length
        nfft : int, optional
            Explicit FFT length to use

        Returns:
        --------
        tuple
            (frequencies, fft_data, fft_length)
        """
        n = data.shape[0]
        fft_length = n

        if nfft is not None:
            if nfft < n:
                raise ValueError(
                    f"Requested nfft ({nfft}) must be greater than or equal to data length ({n})"
                )
            fft_length = nfft
        elif zero_padding:
            next_power_two = 1 << (n - 1).bit_length()
            if next_power_two > n:
                fft_length = next_power_two

        if engine == "numpy":
            fft_data = np.fft.rfft(data, n=fft_length, axis=0)
            frequencies = np.fft.rfftfreq(fft_length, dt)
        elif engine == "scipy" and SCIPY_AVAILABLE:
            fft_data = scipy.fft.rfft(data, n=fft_length, axis=0)
            frequencies = scipy.fft.rfftfreq(fft_length, dt)
        elif engine == "pyfftw" and PYFFTW_AVAILABLE:
            fft_data = pyfftw.interfaces.numpy_fft.rfft(
                data, n=fft_length, axis=0, threads=pyfftw.config.NUM_THREADS
            )
            frequencies = pyfftw.interfaces.numpy_fft.rfftfreq(fft_length, dt)
        else:
            # Fallback to numpy
            fft_data = np.fft.rfft(data, n=fft_length, axis=0)
            frequencies = np.fft.rfftfreq(fft_length, dt)

        return frequencies, fft_data, fft_length

    def calculate_fft_method1(
        self,
        data: np.ndarray,
        dt: float,
        window: WINDOW_TYPES = "hann",
        filter_type: FILTER_TYPES = "remove_mean",
        engine: Optional[str] = None,
        zero_padding: bool = True,
        nfft: Optional[int] = None,
    ) -> FFTComputeResult:
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
        frequencies, fft_data, fft_length = self.compute_fft(
            data_windowed,
            dt,
            selected_engine,
            zero_padding=zero_padding,
            nfft=nfft,
        )

        spectrum = fft_data

        # Average over spatial dimensions (keep frequency/component axes)
        if spectrum.ndim > 2:
            spatial_axes = tuple(range(1, spectrum.ndim - 1))
            if spatial_axes:
                spectrum = np.mean(spectrum, axis=spatial_axes)

        calculation_time = time.time() - start_time

        metadata = {
            "method": 1,
            "window": window,
            "filter_type": filter_type,
            "engine": selected_engine,
            "zero_padding": zero_padding,
            "nfft_requested": nfft,
            "calculation_time": calculation_time,
            "data_shape": data.shape,
            "dt": dt,
            "frequency_resolution": (
                frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0
            ),
            "fft_length": fft_length,
        }

        config = FFTComputeConfig(
            window_function=window,
            filter_type=filter_type,
            fft_engine=selected_engine,
            zero_padding=zero_padding,
            nfft=nfft,
        )

        return FFTComputeResult(
            frequencies=frequencies, spectrum=spectrum, metadata=metadata, config=config
        )

    def calculate_fft_method2(
        self,
        data: np.ndarray,
        dt: float,
        window: WINDOW_TYPES = "hann",
        filter_type: FILTER_TYPES = "remove_mean",
        engine: Optional[str] = None,
        zero_padding: bool = True,
        nfft: Optional[int] = None,
    ) -> FFTComputeResult:
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
        frequencies, fft_data, fft_length = self.compute_fft(
            data_windowed,
            dt,
            selected_engine,
            zero_padding=zero_padding,
            nfft=nfft,
        )

        spectrum = fft_data

        calculation_time = time.time() - start_time

        metadata = {
            "method": 2,
            "window": window,
            "filter_type": filter_type,
            "engine": selected_engine,
            "zero_padding": zero_padding,
            "nfft_requested": nfft,
            "calculation_time": calculation_time,
            "data_shape": data.shape,
            "dt": dt,
            "frequency_resolution": (
                frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0
            ),
            "fft_length": fft_length,
        }

        config = FFTComputeConfig(
            window_function=window,
            filter_type=filter_type,
            fft_engine=selected_engine,
            zero_padding=zero_padding,
            nfft=nfft,
        )

        return FFTComputeResult(
            frequencies=frequencies, spectrum=spectrum, metadata=metadata, config=config
        )

    def load_data_from_zarr(
        self,
        zarr_path: str,
        dataset: str,
        z_layer: int = -1,
        tmax: Optional[int] = None,
        slice_info: Optional[Any] = None,
    ) -> tuple[np.ndarray, float]:
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
        tmax : int, optional
            Maximum number of time steps to load
        slice_info : Any, optional
            Slicing information (e.g., from job[0].m_layer[:100,...])

        Returns:
        --------
        tuple
            (data, dt) where data is the loaded array and dt is time step
        """
        # Start timing and memory monitoring
        start_time = time.time()
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        if not PYZFN_AVAILABLE:
            raise ImportError(
                "pyzfn is required to load FFT input data. Install pyzfn before running FFT analysis."
            )

        log.info(f"Loading data from zarr: {zarr_path}")
        log.debug(f"Dataset: {dataset}, z_layer: {z_layer}, tmax: {tmax}")

        try:
            job = Pyzfn(zarr_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to open zarr job at {zarr_path}: {exc}") from exc

        # Get dataset
        data_set = None
        if hasattr(job, dataset):
            data_set = getattr(job, dataset)
        else:
            # Try direct zarr access instead of job.z which doesn't exist
            try:
                import zarr
                z_root = zarr.open(zarr_path, mode="r")
                if dataset in z_root:
                    data_set = z_root[dataset]
                else:
                    log.debug(f"Dataset {dataset} not found in zarr root, checking if it's an attribute of Pyzfn job")
            except Exception as e:
                log.debug(f"Could not access zarr directly: {e}")

        if data_set is None:
            available = []
            try:
                import zarr

                z_root = zarr.open(zarr_path, mode="r")
                available.extend(list(z_root.group_keys()))
                available.extend(list(z_root.array_keys()))
                available = sorted({key.split("/")[0] for key in available})
            except Exception as exc:
                log.debug(
                    "Unable to enumerate datasets in %s: %s", zarr_path, exc
                )

            suggestion = (
                f" Available datasets: {', '.join(available)}"
                if available
                else ""
            )
            raise ValueError(
                f"Dataset '{dataset}' not found in zarr file '{zarr_path}'.{suggestion}"
            )

        # Load data with timing (apply slicing if provided)
        data_load_start = time.time()
        
        # Determine if we should apply tmax limit
        # Priority: explicit user slice > tmax parameter
        apply_tmax = tmax is not None and tmax > 0
        user_provided_time_slice = False
        
        if slice_info is not None:
            # Apply user-provided slicing (e.g., from job[0].m_layer[:100,...])
            log.info(f"Applying slice_info: {slice_info}")
            data = data_set[slice_info]
            
            # Fix dimension drop if component (or spatial dim) was selected via integer index
            # This ensures that calculate_fft_methodX can correctly distinguish spatial vs component dims
            if isinstance(slice_info, tuple) and len(slice_info) > 0:
                if isinstance(slice_info[-1], int):
                    # Last index was int -> dimension dropped. Restore it.
                    # This is crucial for single-component selection: (t, x, y) -> (t, x, y, 1)
                    # so that spatial averaging doesn't average over y.
                    data = data[..., np.newaxis]
                    log.debug(f"Restored dropped dimension: new shape {data.shape}")
            
            # Check if user explicitly sliced time dimension
            # If so, DON'T apply tmax (user's slice takes priority)
            if isinstance(slice_info, tuple) and len(slice_info) > 0:
                first_slice = slice_info[0]
                if first_slice is not Ellipsis:
                    if isinstance(first_slice, slice):
                        # User provided time slice (e.g., [:1000] or [100:200])
                        user_provided_time_slice = True
                        if first_slice.stop is not None:
                            # Explicit stop means user wants specific number of timesteps
                            log.debug(
                                "User provided explicit time slice %s - tmax parameter will be ignored",
                                first_slice
                            )
                            apply_tmax = False
                        else:
                            # [:] means "all timesteps" - also ignore tmax
                            log.debug("User provided [:] slice - using ALL timesteps (ignoring tmax)")
                            apply_tmax = False
                    elif isinstance(first_slice, int):
                        # User selected single timestep
                        user_provided_time_slice = True
                        apply_tmax = False
        else:
            # Load all data
            data = data_set[...]
            
        data_load_time = time.time() - data_load_start

        log.debug(f"Data loading time: {data_load_time:.3f}s")

        # Apply tmax limit ONLY if no explicit user time slice
        if apply_tmax:
            original_time_steps = data.shape[0] if len(data.shape) > 0 else 0
            if tmax < original_time_steps:
                data = data[:tmax]
                log.info(f"Applied tmax={tmax}: reduced from {original_time_steps} to {tmax} time steps")
            else:
                log.info(f"tmax={tmax} >= data length ({original_time_steps}), no truncation applied")

        # Calculate data size and loading speed
        data_size_mb = data.nbytes / 1024 / 1024
        loading_speed = data_size_mb / data_load_time if data_load_time > 0 else 0
        log.debug(f"Data size: {data_size_mb:.1f} MB")
        log.debug(f"Loading speed: {loading_speed:.1f} MB/s")

        # Handle z-layer selection BEFORE determining final shape
        # CRITICAL: z-layer selection must happen BEFORE component selection is interpreted
        # because slice [:,...,2] removes component axis, making 5D→4D, which could be
        # misinterpreted as (t,y,x,comp) instead of (t,z,y,x)
        layer_select_start = time.time()
        original_ndim = len(data.shape)
        
        if original_ndim == 5:  # (t, z, y, x, comp)
            if z_layer == -1:
                data = data[:, -1, :, :, :]  # Take last layer
                log.debug("Selected last z-layer from 5D data")
            else:
                data = data[:, z_layer, :, :, :]
                log.debug(f"Selected z-layer {z_layer} from 5D data")
        elif original_ndim == 4:
            # Ambiguity: could be (t,z,y,x) with component pre-selected, OR (t,y,x,comp)
            # Heuristic: if slice_info selected component (e.g., [...,2]), assume (t,z,y,x)
            # Check if last element of slice_info is an integer (component selection)
            component_was_selected = False
            if slice_info is not None and isinstance(slice_info, tuple):
                # Find the last non-Ellipsis element
                non_ellipsis_slices = [s for s in slice_info if s is not Ellipsis]
                if non_ellipsis_slices and isinstance(non_ellipsis_slices[-1], (int, np.integer)):
                    component_was_selected = True
                    log.debug("Detected component selection in slice - treating 4D as (t,z,y,x)")
            
            if component_was_selected:
                # User selected component via slicing, so this is (t, z, y, x)
                if z_layer == -1:
                    data = data[:, -1, :, :]  # Take last z-layer
                    log.debug("Selected last z-layer from 4D data (component pre-selected)")
                else:
                    data = data[:, z_layer, :, :]
                    log.debug(f"Selected z-layer {z_layer} from 4D data (component pre-selected)")
            else:
                # No component selection detected - assume (t, y, x, comp)
                log.debug("No z-dimension in 4D data (assuming t,y,x,comp)")
        elif original_ndim == 3:  # (t, y, x) or (t, y, comp)
            log.debug("3D dataset detected - using provided dimensions without z-layer selection")
        elif original_ndim == 2:  # (t, comp) or (t, y)
            log.debug("2D dataset detected - interpreting first axis as time")
        elif original_ndim == 1:  # (t,)
            log.debug("1D time series detected")
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        layer_select_time = time.time() - layer_select_start
        log.debug(f"Layer selection time: {layer_select_time:.3f}s")

        # Get time step - prioritize dataset-specific attrs over global
        # Order: dataset.attrs['t'] -> dataset.attrs['t_sampl'] -> job.attrs['t_sampl'] -> default
        dt = None
        try:
            # Method 1: Check dataset.attrs['t'] array (most specific, per-dataset time)
            if hasattr(data_set, 'attrs') and 't' in data_set.attrs:
                t_attr = data_set.attrs['t']
                if hasattr(t_attr, '__len__') and len(t_attr) >= 2:
                    dt = float(t_attr[1] - t_attr[0])
                    log.debug(f"Using dt from data_set.attrs['t']: {dt}")
            
            # Method 2: Check if data_set is wrapped by DatasetAwareWrapper with .dt property
            if dt is None and hasattr(data_set, 'dt'):
                dt = data_set.dt
                log.debug(f"Using dt from data_set.dt property: {dt}")
            
            # Method 3: Check dataset.attrs['t_sampl']
            if dt is None and hasattr(data_set, 'attrs') and 't_sampl' in data_set.attrs:
                dt = data_set.attrs['t_sampl']
                log.debug(f"Using dt from data_set.attrs['t_sampl']: {dt}")
            
            # Method 4: Fallback to global job.attrs['t_sampl']
            if dt is None and hasattr(job, 'attrs') and 't_sampl' in job.attrs:
                dt = job.attrs['t_sampl']
                log.warning(f"Using dt from job.attrs['t_sampl']: {dt} (dataset-specific dt not found)")
            
            # Method 5: Last resort default
            if dt is None:
                dt = 1e-12
                log.warning(f"t_sampl not found in attrs, using default: {dt}")
        except (AttributeError, TypeError, IndexError) as e:
            log.warning(f"Could not determine dt: {e}, using default")
            dt = 1e-12

        # Final timing and memory measurement
        total_time = time.time() - start_time
        if PSUTIL_AVAILABLE:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            log.debug(f"Memory increase: {memory_increase:.1f} MB")

        log.info(f"Data loaded successfully in {total_time:.3f}s, shape: {data.shape}")

        return data, dt

    def get_available_options(self) -> dict[str, Any]:
        """Get available configuration options."""
        return {
            "windows": list(self.AVAILABLE_WINDOWS.keys()),
            "filters": [
                "none",
                "remove_mean",
                "remove_static",
                "detrend_linear",
                "remove_mean_and_static",
                # New filters from FMR literature
                "savgol_smooth",
                "baseline_correction",
                "high_pass",
                "band_pass",
                "spectral_derivative",
            ],
            "engines": list(self.AVAILABLE_ENGINES.keys()),
            "dependencies": {"scipy": SCIPY_AVAILABLE, "pyfftw": PYFFTW_AVAILABLE},
        }

    def load_existing_fft_data(
        self, zarr_path: str, dataset_name: str = "fft"
    ) -> Optional[FFTComputeResult]:
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
            # Start timing and memory monitoring
            start_time = time.time()
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            log.debug(f"Loading existing FFT data from: {zarr_path}")
            log.debug(f"FFT dataset: fft/{dataset_name}")

            import zarr

            z = zarr.open(zarr_path, mode="r")

            fft_path = f"fft/{dataset_name}"
            if fft_path not in z:
                log.debug(f"FFT dataset {fft_path} not found")
                return None

            fft_group = z[fft_path]

            # Load data with timing
            data_load_start = time.time()
            spectrum = np.array(fft_group["spectrum"])
            frequencies = np.array(fft_group["frequencies"])
            data_load_time = time.time() - data_load_start

            log.debug(f"FFT data loading time: {data_load_time:.3f}s")

            # Calculate data sizes
            spectrum_size_mb = spectrum.nbytes / 1024 / 1024
            freq_size_mb = frequencies.nbytes / 1024 / 1024
            total_size_mb = spectrum_size_mb + freq_size_mb

            log.debug(f"Spectrum size: {spectrum_size_mb:.1f} MB")
            log.debug(f"Frequencies size: {freq_size_mb:.1f} MB")
            log.debug(f"Total FFT data size: {total_size_mb:.1f} MB")

            # Load metadata
            metadata = dict(fft_group.attrs)

            # Create config from attributes
            config = FFTComputeConfig(
                window_function=metadata.pop("window_function", "hann"),
                filter_type=metadata.pop("filter_type", "remove_mean"),
                fft_engine=metadata.pop("fft_engine", "auto"),
                zero_padding=metadata.pop("zero_padding", True),
                nfft=metadata.pop("nfft", None),
            )

            # Final timing and memory measurement
            total_time = time.time() - start_time
            if PSUTIL_AVAILABLE:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                log.debug(f"Memory increase: {memory_increase:.1f} MB")

            log.info(
                f"Loaded existing FFT data in {total_time:.3f}s, spectrum shape: {spectrum.shape}"
            )

            return FFTComputeResult(
                frequencies=frequencies,
                spectrum=spectrum,
                metadata=metadata,
                config=config,
            )

        except Exception as e:
            log.warning(f"Could not load existing FFT data: {e}")
            return None

    def _verify_fft_parameters(
        self, existing_result: FFTComputeResult, **kwargs
    ) -> bool:
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
        window = kwargs.get("window", self.config.window_function)
        filter_type = kwargs.get("filter_type", self.config.filter_type)
        engine = kwargs.get("engine", self.config.fft_engine)
        zero_padding = kwargs.get("zero_padding", self.config.zero_padding)
        nfft = kwargs.get("nfft", self.config.nfft)

        # Compare with existing config
        config_match = (
            existing_result.config.window_function == window
            and existing_result.config.filter_type == filter_type
            and existing_result.config.fft_engine == engine
            and existing_result.config.zero_padding == zero_padding
            and existing_result.config.nfft == nfft
        )

        # Compare metadata that affects FFT calculation
        # (add other relevant metadata fields as needed)
        metadata_keys_to_check = ["z_layer", "source_dataset", "slice_identifier"]
        metadata_match = True
        for key in metadata_keys_to_check:
            if key in kwargs and key in existing_result.metadata:
                if kwargs[key] != existing_result.metadata[key]:
                    metadata_match = False
                    break

        return config_match and metadata_match

    def calculate_fft_data(
        self,
        zarr_path: str,
        dataset: str,
        z_layer: int = -1,
        method: int = 1,
        save: bool = False,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
        slice_identifier: Optional[str] = None,
        tmax: Optional[int] = None,
        **kwargs,
    ) -> FFTComputeResult:
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
        slice_info : Any, optional
            Slicing arguments applied before loading (e.g., [:1000, ..., 0])
        slice_identifier : str, optional
            Deterministic identifier for cache/save naming (derived from slice_info)
        tmax : int, optional
            Maximum number of time steps to use for FFT calculation (default: None, use all)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        FFTComputeResult
            FFT computation result
        """
        log.debug(
            f"calculate_fft_data called with: {dataset}, z_layer={z_layer}, method={method}, save={save}, force={force}"
        )

        # Validate z_layer parameter
        if z_layer is None:
            raise ValueError("z_layer cannot be None. Use -1 for last layer or specify a valid layer index.")

        # Normalize z_layer to actual index for consistent naming
        # We need to load shape info to normalize z_layer=-1 to actual index
        try:
            if not PYZFN_AVAILABLE:
                raise ImportError("pyzfn required for data shape inspection")
            
            temp_job = Pyzfn(zarr_path)
            temp_data_set = None
            if hasattr(temp_job, dataset):
                temp_data_set = getattr(temp_job, dataset)
            else:
                z_group = getattr(temp_job, "z", None)
                if z_group is not None and dataset in z_group:
                    temp_data_set = z_group[dataset]
            
            # Fallback: Try direct zarr access
            if temp_data_set is None:
                try:
                    import zarr
                    z_root = zarr.open(zarr_path, mode="r")
                    if dataset in z_root:
                        temp_data_set = z_root[dataset]
                        log.debug(f"Found dataset '{dataset}' via direct zarr access")
                except Exception:
                    pass
            
            if temp_data_set is not None:
                data_shape = temp_data_set.shape
                if len(data_shape) == 5 and z_layer == -1:  # (t, z, y, x, comp)
                    normalized_z_layer = data_shape[1] - 1  # Last z layer
                    log.debug(f"Normalized z_layer={z_layer} to {normalized_z_layer} (shape: {data_shape})")
                elif len(data_shape) == 5 and z_layer < -1:  # Other negative indices
                    normalized_z_layer = data_shape[1] + z_layer
                    log.debug(f"Normalized negative z_layer={z_layer} to {normalized_z_layer} (shape: {data_shape})")
                else:
                    normalized_z_layer = z_layer
            else:
                log.debug(f"Dataset '{dataset}' not found for shape inspection, using z_layer as-is")
                normalized_z_layer = z_layer
        except Exception as e:
            log.warning(f"Failed to normalize z_layer: {e}, using z_layer as-is")
            normalized_z_layer = z_layer

        # Generate save dataset name if not provided - use normalized z_layer for consistency
        if save_dataset_name is None:
            save_dataset_name = f"{dataset}_z{normalized_z_layer}_m{method}"
            if slice_identifier:
                slice_hash = hashlib.md5(slice_identifier.encode("utf-8")).hexdigest()[:8]
                save_dataset_name = f"{save_dataset_name}_s{slice_hash}"

        # Try to load existing data if not forcing recalculation
        if not force:
            log.debug(f"Checking for existing FFT data: fft/{save_dataset_name}")
            existing_result = self.load_existing_fft_data(zarr_path, save_dataset_name)
            if existing_result is not None:
                # Verify that parameters match
                if self._verify_fft_parameters(
                    existing_result,
                    z_layer=z_layer,
                    source_dataset=dataset,
                    slice_identifier=slice_identifier,
                    **kwargs,
                ):
                    log.info(
                        f"✓ Loaded existing FFT data for {save_dataset_name} (parameters verified)"
                    )
                    log.debug(f"Parameters: z_layer={z_layer}→{normalized_z_layer}, dataset={dataset}, method={method}")
                    return existing_result
                else:
                    log.warning(
                        f"Existing FFT data found but parameters don't match, recalculating..."
                    )
                    log.debug(f"Mismatched parameters: z_layer={z_layer}→{normalized_z_layer}, dataset={dataset}")
                    force = True  # Force recalculation if parameters don't match
            else:
                log.info(f"No existing FFT data found for {save_dataset_name}, calculating new FFT...")
        else:
            log.info(f"Force recalculation enabled for {save_dataset_name}, computing new FFT...")
            log.debug(f"Parameters: z_layer={z_layer}→{normalized_z_layer}, dataset={dataset}, method={method}")

        # Load data
        log.info(f"Loading data from {dataset} (z_layer={z_layer})...")

        # Measure loading time and memory usage
        import os
        import time

        # Try to use psutil for memory monitoring, fallback if not available
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            psutil_available = True
        except ImportError:
            memory_before = 0
            psutil_available = False

        # Time the data loading
        load_start_time = time.time()
        data, dt = self.load_data_from_zarr(
            zarr_path, dataset, z_layer, tmax=tmax, slice_info=slice_info
        )
        load_end_time = time.time()

        # Memory after loading (if psutil available)
        if psutil_available:
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
        else:
            memory_after = 0
            memory_used = 0

        # Calculate data size in memory
        data_size_bytes = data.nbytes
        data_size_mb = data_size_bytes / 1024 / 1024
        data_size_gb = data_size_mb / 1024

        # Display results
        load_time = load_end_time - load_start_time
        log.info(f"Data shape: {data.shape}, dt: {dt}")
        log.debug(f"⏱️  Data loading time: {load_time:.3f}s")
        log.debug(f"💾 Data size: {data_size_mb:.1f} MB ({data_size_gb:.2f} GB)")

        if psutil_available:
            log.debug(
                f"🧠 Memory usage change: {memory_used:+.1f} MB (before: {memory_before:.1f} MB, after: {memory_after:.1f} MB)"
            )
        else:
            log.debug(
                "🧠 Memory monitoring unavailable (install psutil for memory stats)"
            )

        # Calculate loading speed
        if load_time > 0:
            loading_speed_mbps = data_size_mb / load_time
            log.debug(f"🚀 Loading speed: {loading_speed_mbps:.1f} MB/s")

        # Extract configuration from kwargs
        window = kwargs.get("window", self.config.window_function)
        filter_type = kwargs.get("filter_type", self.config.filter_type)
        engine = kwargs.get("engine", self.config.fft_engine)
        zero_padding = kwargs.get("zero_padding", self.config.zero_padding)
        nfft = kwargs.get("nfft", self.config.nfft)

        log.info(
            "Computing FFT with method %s (window: %s, filter: %s, engine: %s, zero_padding: %s, nfft: %s)...",
            method,
            window,
            filter_type,
            engine,
            zero_padding,
            nfft,
        )

        # Calculate FFT using specified method
        if method == 1:
            result = self.calculate_fft_method1(
                data,
                dt,
                window,
                filter_type,
                engine,
                zero_padding=zero_padding,
                nfft=nfft,
            )
        elif method == 2:
            result = self.calculate_fft_method2(
                data,
                dt,
                window,
                filter_type,
                engine,
                zero_padding=zero_padding,
                nfft=nfft,
            )
        else:
            raise ValueError(f"Unsupported FFT method: {method}")

        log.info(
            f"✓ FFT calculation completed in {result.metadata.get('calculation_time', 0):.3f}s"
        )

        # Add additional metadata
        result.metadata.update(
            {
                "zarr_path": zarr_path,
                "source_dataset": dataset,
                "z_layer": z_layer,
                "save_dataset_name": save_dataset_name,
                "slice_identifier": slice_identifier,
            }
        )

        # Save to zarr if requested
        if save:
            try:
                log.info(f"Saving FFT data to fft/{save_dataset_name}...")
                result.save_to_zarr(zarr_path, save_dataset_name, force=force)
                log.info(f"✓ Successfully saved FFT data to fft/{save_dataset_name}")
            except Exception as e:
                log.warning(f"Could not save FFT data: {e}")
        else:
            log.debug("FFT calculation completed (not saved)")

        return result
