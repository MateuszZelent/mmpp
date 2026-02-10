"""
FFT Computation Module

Core FFT computation functionality moved from old_fft_module.py and main.py.
Provides low-level FFT calculations without user interface elements.
"""

import hashlib
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
from ._compute_cache import load_existing_fft_result, verify_fft_parameters
from ._compute_engines import compute_fft_data, determine_engine_name
from ._compute_loading import (
    load_fft_input_data,
    load_fft_input_data_profiled,
    log_input_load_metrics,
    normalize_z_layer_index,
)
from ._compute_methods import build_fft_metadata, run_fft_method1, run_fft_method2
from .filters.preprocess import (
    apply_filter as apply_preprocess_filter,
    apply_single_filter as apply_single_preprocess_filter,
    baseline_correction as preprocess_baseline_correction,
    band_pass as preprocess_band_pass,
    high_pass as preprocess_high_pass,
    savgol_smooth as preprocess_savgol_smooth,
    spectral_derivative as preprocess_spectral_derivative,
)
from .filters.windows import apply_window as apply_fft_window

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
        return determine_engine_name(
            configured_engine=self.config.fft_engine,
            data_size=data_size,
            scipy_available=SCIPY_AVAILABLE,
            pyfftw_available=PYFFTW_AVAILABLE,
        )

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
        return apply_fft_window(data, window_type)

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
        return apply_preprocess_filter(data, filter_type)

    def _apply_single_filter(self, data: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply a single filter to data."""
        return apply_single_preprocess_filter(data, filter_type)

    def _apply_savgol_smooth(
        self, data: np.ndarray, window_length: int = 11, polyorder: int = 3
    ) -> np.ndarray:
        """Apply Savitzky-Golay smoothing filter.
        
        Reduces noise while preserving signal shape and peak positions.
        Common in spectroscopic data processing.
        """
        return preprocess_savgol_smooth(
            data,
            window_length=window_length,
            polyorder=polyorder,
        )

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
        return preprocess_baseline_correction(data, lam=lam, p=p, niter=niter)

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
        return preprocess_high_pass(data, cutoff_fraction=cutoff_fraction)

    def _high_pass_1d(self, y: np.ndarray, cutoff_fraction: float) -> np.ndarray:
        """High-pass filter for 1D array using FFT."""
        return preprocess_high_pass(y, cutoff_fraction=cutoff_fraction)

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
        return preprocess_band_pass(
            data,
            low_fraction=low_fraction,
            high_fraction=high_fraction,
        )

    def _band_pass_1d(
        self, y: np.ndarray, low_fraction: float, high_fraction: float
    ) -> np.ndarray:
        """Band-pass filter for 1D array using FFT."""
        return preprocess_band_pass(
            y,
            low_fraction=low_fraction,
            high_fraction=high_fraction,
        )

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
        return preprocess_spectral_derivative(data, order=order)

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
        return compute_fft_data(
            data=data,
            dt=dt,
            engine=engine,
            zero_padding=zero_padding,
            nfft=nfft,
            scipy_available=SCIPY_AVAILABLE,
            pyfftw_available=PYFFTW_AVAILABLE,
            scipy_module=(scipy if SCIPY_AVAILABLE else None),
            pyfftw_module=(pyfftw if PYFFTW_AVAILABLE else None),
        )

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
        """FFT Method 1: apply filter+window, FFT, then spatial averaging."""
        execution = run_fft_method1(
            data=data,
            dt=dt,
            window=window,
            filter_type=filter_type,
            engine=engine,
            zero_padding=zero_padding,
            nfft=nfft,
            determine_engine=self.determine_engine,
            apply_filter=self.apply_filter,
            apply_window=self.apply_window,
            compute_fft=self.compute_fft,
        )
        metadata = build_fft_metadata(
            method=1,
            window=window,
            filter_type=filter_type,
            selected_engine=execution.selected_engine,
            zero_padding=zero_padding,
            nfft=nfft,
            calculation_time=execution.calculation_time,
            data_shape=data.shape,
            dt=dt,
            frequencies=execution.frequencies,
            fft_length=execution.fft_length,
        )
        config = FFTComputeConfig(
            window_function=window,
            filter_type=filter_type,
            fft_engine=execution.selected_engine,
            zero_padding=zero_padding,
            nfft=nfft,
        )
        return FFTComputeResult(
            frequencies=execution.frequencies,
            spectrum=execution.spectrum,
            metadata=metadata,
            config=config,
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
        """FFT Method 2: apply filter, spatial averaging, window, then FFT."""
        execution = run_fft_method2(
            data=data,
            dt=dt,
            window=window,
            filter_type=filter_type,
            engine=engine,
            zero_padding=zero_padding,
            nfft=nfft,
            determine_engine=self.determine_engine,
            apply_filter=self.apply_filter,
            apply_window=self.apply_window,
            compute_fft=self.compute_fft,
        )
        metadata = build_fft_metadata(
            method=2,
            window=window,
            filter_type=filter_type,
            selected_engine=execution.selected_engine,
            zero_padding=zero_padding,
            nfft=nfft,
            calculation_time=execution.calculation_time,
            data_shape=data.shape,
            dt=dt,
            frequencies=execution.frequencies,
            fft_length=execution.fft_length,
        )
        config = FFTComputeConfig(
            window_function=window,
            filter_type=filter_type,
            fft_engine=execution.selected_engine,
            zero_padding=zero_padding,
            nfft=nfft,
        )
        return FFTComputeResult(
            frequencies=execution.frequencies,
            spectrum=execution.spectrum,
            metadata=metadata,
            config=config,
        )

    def load_data_from_zarr(
        self,
        zarr_path: str,
        dataset: str,
        z_layer: int = -1,
        tmax: Optional[int] = None,
        slice_info: Optional[Any] = None,
    ) -> tuple[np.ndarray, float]:
        """Load data from zarr file."""
        return load_fft_input_data(
            zarr_path=zarr_path,
            dataset=dataset,
            z_layer=z_layer,
            tmax=tmax,
            slice_info=slice_info,
            pyzfn_available=PYZFN_AVAILABLE,
            pyzfn_cls=Pyzfn,
            psutil_module=(psutil if PSUTIL_AVAILABLE else None),
            logger=log,
        )

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
        """Load existing FFT data from zarr file."""
        loaded = load_existing_fft_result(
            zarr_path=zarr_path,
            dataset_name=dataset_name,
            result_cls=FFTComputeResult,
            config_cls=FFTComputeConfig,
            psutil_module=(psutil if PSUTIL_AVAILABLE else None),
            logger=log,
        )
        return loaded

    def _verify_fft_parameters(
        self, existing_result: FFTComputeResult, **kwargs
    ) -> bool:
        """Verify if FFT parameters match existing result."""
        window = kwargs.get(
            "window",
            kwargs.get("window_function", self.config.window_function),
        )
        filter_type = kwargs.get("filter_type", self.config.filter_type)
        engine = kwargs.get("engine", self.config.fft_engine)
        zero_padding = kwargs.get("zero_padding", self.config.zero_padding)
        nfft = kwargs.get("nfft", self.config.nfft)

        return verify_fft_parameters(
            existing_result=existing_result,
            window=window,
            filter_type=filter_type,
            engine=engine,
            zero_padding=zero_padding,
            nfft=nfft,
            metadata_overrides=kwargs,
        )

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
        normalized_z_layer = normalize_z_layer_index(
            zarr_path=zarr_path,
            dataset=dataset,
            z_layer=z_layer,
            pyzfn_available=PYZFN_AVAILABLE,
            pyzfn_cls=Pyzfn,
            logger=log,
        )

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
        data, dt, load_metrics = load_fft_input_data_profiled(
            zarr_path=zarr_path,
            dataset=dataset,
            z_layer=z_layer,
            tmax=tmax,
            slice_info=slice_info,
            pyzfn_available=PYZFN_AVAILABLE,
            pyzfn_cls=Pyzfn,
            psutil_module=(psutil if PSUTIL_AVAILABLE else None),
            logger=log,
        )
        log_input_load_metrics(
            logger=log,
            data=data,
            dt=dt,
            metrics=load_metrics,
        )

        # Extract configuration from kwargs
        window = kwargs.get(
            "window",
            kwargs.get("window_function", self.config.window_function),
        )
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
