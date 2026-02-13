"""
Mode Data Loader Module

Provides slice-aware data loading for FMR mode analysis.
Supports component extraction and time slicing from DatasetAwareWrapper.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union
import numpy as np
import logging

# Get logger
log = logging.getLogger("mmpp.fft.modes")

# Component labels for magnetization
COMPONENT_LABELS = [r"$m_x$", r"$m_y$", r"$m_z$"]
COMPONENT_NAMES = ["mx", "my", "mz"]


@dataclass
class ModeDataContext:
    """Context for mode data loading with slice support.
    
    Attributes
    ----------
    zarr_path : str
        Path to zarr file
    dataset_name : str
        Dataset name (e.g., 'm', 'm_x11')
    slice_info : tuple, optional
        Slice info from DatasetAwareWrapper, e.g., (slice(0,200), ..., 1)
    z_layer : int
        Z-layer index (default: -1 for top layer)
    component_index : int, optional
        Magnetization component: 0=mx, 1=my, 2=mz
        If None, all components are loaded
    time_slice : slice, optional
        Extracted time slice from slice_info
    """
    zarr_path: str
    dataset_name: str
    slice_info: Optional[tuple] = None
    z_layer: int = -1
    component_index: Optional[int] = None
    time_slice: Optional[slice] = None
    
    # Derived fields (set in __post_init__)
    component_label: Optional[str] = field(default=None, init=False)
    
    def __post_init__(self):
        """Parse slice_info to extract component and time slice."""
        if self.slice_info is not None:
            self._parse_slice_info()
        
        # Set component label
        if self.component_index is not None and 0 <= self.component_index <= 2:
            self.component_label = COMPONENT_LABELS[self.component_index]
    
    def _parse_slice_info(self):
        """Parse slice_info tuple to extract component and time slice."""
        if not isinstance(self.slice_info, tuple):
            return
        
        # Check last element for component index
        if len(self.slice_info) > 0:
            last = self.slice_info[-1]
            if isinstance(last, (int, np.integer)):
                idx = int(last)
                if idx in (0, 1, 2):
                    self.component_index = idx
                    log.debug(f"Extracted component index {idx} from slice_info")
                elif idx == -1:
                    self.component_index = 2
                    log.debug("Extracted component index 2 (from -1) from slice_info")
            elif isinstance(last, slice):
                step = 1 if last.step is None else int(last.step)
                if step == 1 and isinstance(last.start, (int, np.integer)):
                    start = int(last.start)
                    if start in (0, 1, 2) and isinstance(last.stop, (int, np.integer)):
                        if int(last.stop) == start + 1:
                            self.component_index = start
                            log.debug(f"Extracted component index {start} from singleton slice_info")
                    elif start == -1 and last.stop is None:
                        self.component_index = 2
                        log.debug("Extracted component index 2 from slice_info[-1:None]")
        
        # Check first element for time slice
        if len(self.slice_info) > 0:
            first = self.slice_info[0]
            if isinstance(first, slice):
                self.time_slice = first
                log.debug(f"Extracted time slice {first} from slice_info")


class ModeDataLoader:
    """Loads mode data respecting slice_info from user selection.
    
    This class handles loading spectrum and mode data from zarr files,
    applying any user-specified slicing (time range, component selection).
    
    Parameters
    ----------
    context : ModeDataContext
        Configuration context for data loading
    
    Examples
    --------
    >>> context = ModeDataContext(
    ...     zarr_path="/path/to/data.zarr",
    ...     dataset_name="m",
    ...     slice_info=(slice(0, 200), ..., 1)  # First 200 timesteps, my
    ... )
    >>> loader = ModeDataLoader(context)
    >>> freqs, spectrum = loader.load_spectrum()
    """
    
    def __init__(self, context: ModeDataContext):
        self.context = context
        self._zarr_file = None
        self._frequencies = None
        self._spectrum = None
        self._modes_path = None
        self._freqs_path = None
        self._spectrum_path = None
        
    @property
    def zarr_file(self):
        """Lazy load zarr file."""
        if self._zarr_file is None:
            try:
                import zarr
                self._zarr_file = zarr.open(self.context.zarr_path, mode="r")
                log.debug(f"Opened zarr file: {self.context.zarr_path}")
            except Exception as e:
                log.error(f"Failed to open zarr file: {e}")
                raise
        return self._zarr_file
    
    @property
    def component_label(self) -> Optional[str]:
        """Get label for selected component."""
        return self.context.component_label
    
    @property
    def has_single_component(self) -> bool:
        """Check if single component is selected."""
        return self.context.component_index is not None
    
    def _resolve_paths(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Resolve zarr paths for modes, frequencies, and spectrum."""
        dset = self.context.dataset_name
        
        # Modes and frequencies paths
        base_paths = [f"modes/{dset}", f"tmodes/{dset}"]
        modes_path = None
        freqs_path = None
        
        for base in base_paths:
            if f"{base}/arr" in self.zarr_file and f"{base}/freqs" in self.zarr_file:
                modes_path = f"{base}/arr"
                freqs_path = f"{base}/freqs"
                break
        
        # Spectrum path candidates - start with exact matches
        spectrum_candidates = [
            f"fft/{dset}_z-1_m1/spectrum",
            f"fft/{dset}_z0_m1/spectrum",
            f"fft/{dset}/spectrum",
            f"fft/{dset}/spec",
            f"fft/{dset}/sum",
            f"modes/{dset}/power_sum",
            f"modes/{dset}/power_max",
        ]
        
        spectrum_path = None
        
        # First try exact matches
        for path in spectrum_candidates:
            if path in self.zarr_file:
                spectrum_path = path
                log.debug(f"Found spectrum at: {path}")
                break
        
        # If not found and we have slice_info, search for paths with slice hash
        if spectrum_path is None and self.context.slice_info is not None:
            try:
                # Scan fft/ directory for any paths matching dataset with slice hash
                # Pattern: fft/{dset}_z{z}_m{m}_s{hash}/spectrum
                fft_keys = [k for k in self.zarr_file.keys() if k.startswith('fft/')]
                
                for key in fft_keys:
                    # Check if key starts with our dataset and ends with /spectrum
                    if key.startswith(f'fft/{dset}_z') and key.endswith('/spectrum'):
                        # Check if it contains slice hash marker '_s'
                        if '_s' in key:
                            spectrum_path = key
                            log.info(f"Found sliced spectrum at: {spectrum_path}")
                            break
            except Exception as e:
                log.debug(f"Error scanning for sliced spectrum paths: {e}")
        
        return modes_path, freqs_path, spectrum_path
    
    def load_spectrum(self) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """Load spectrum data with component extraction.
        
        Returns
        -------
        frequencies : np.ndarray
            Frequency array in GHz
        spectrum : np.ndarray
            Power spectrum (1D if single component, 2D if all components)
        component_label : str or None
            Label for component if single component selected
        """
        modes_path, freqs_path, spectrum_path = self._resolve_paths()
        dset = self.context.dataset_name
        
        if spectrum_path is None:
            raise RuntimeError(
                f"Spectrum data not found for dataset '{self.context.dataset_name}'. "
                "Run compute_modes() first."
            )
        
        # Load spectrum first to determine size
        spectrum = np.array(self.zarr_file[spectrum_path])
        
        # Handle complex spectra
        if np.iscomplexobj(spectrum):
            spectrum = np.abs(spectrum)
        
        # Determine spectrum length (first dim is usually frequency)
        spec_len = spectrum.shape[0]
        
        # Try to find matching frequencies
        frequencies = None
        
        # Option 1: Use modes frequencies if available and match
        if freqs_path is not None:
            freqs_candidate = np.array(self.zarr_file[freqs_path])
            if len(freqs_candidate) == spec_len:
                frequencies = freqs_candidate
                log.debug(f"Using modes frequencies: {len(frequencies)} points")
        
        # Option 2: Try FFT frequencies path
        if frequencies is None:
            fft_freqs_candidates = [
                f"fft/{dset}_z-1_m1/frequencies",
                f"fft/{dset}_z0_m1/frequencies", 
                f"fft/{dset}/frequencies",
                f"fft/{dset}/freqs",
            ]
            for path in fft_freqs_candidates:
                if path in self.zarr_file:
                    freqs_candidate = np.array(self.zarr_file[path])
                    if len(freqs_candidate) == spec_len:
                        frequencies = freqs_candidate
                        log.debug(f"Using FFT frequencies from {path}")
                        break
            
            # If still not found and spectrum_path contains slice hash, 
            # try frequencies from same directory
            if frequencies is None and spectrum_path and '_s' in spectrum_path:
                freq_path_from_spectrum = spectrum_path.replace('/spectrum', '/frequencies')
                if freq_path_from_spectrum in self.zarr_file:
                    freqs_candidate = np.array(self.zarr_file[freq_path_from_spectrum])
                    if len(freqs_candidate) == spec_len:
                        frequencies = freqs_candidate
                        log.debug(f"Using frequencies from sliced FFT: {freq_path_from_spectrum}")
        
        # Option 3: Generate synthetic frequencies if nothing matches
        if frequencies is None:
            log.warning(
                f"No matching frequencies found for spectrum of length {spec_len}. "
                "Generating synthetic frequency array."
            )
            # Assume typical FFT output: 0 to Nyquist
            # Use 50 GHz as default Nyquist (typical for FMR simulations)
            frequencies = np.linspace(0, 50, spec_len)
        
        # Apply component extraction if specified
        if self.context.component_index is not None:
            if spectrum.ndim > 1:
                # Extract specific component
                comp_idx = self.context.component_index
                if spectrum.shape[-1] == 3:
                    spectrum = spectrum[..., comp_idx]
                    log.debug(f"Extracted component {comp_idx} from spectrum")
                else:
                    # Try to handle other dimensions
                    log.warning(
                        f"Cannot extract component {comp_idx} from spectrum "
                        f"with shape {spectrum.shape}"
                    )
        
        # Ensure spectrum is 1D or 2D (freqs, [components])
        if spectrum.ndim > 2:
            # Average over spatial dimensions, keeping frequency and components
            if spectrum.shape[-1] == 3:
                # (freqs, ..., 3) -> average spatial -> (freqs, 3)
                spatial_axes = tuple(range(1, spectrum.ndim - 1))
                spectrum = np.mean(spectrum, axis=spatial_axes)
            else:
                # Average all non-frequency dimensions
                spatial_axes = tuple(range(1, spectrum.ndim))
                spectrum = np.mean(spectrum, axis=spatial_axes)
            log.debug(f"Averaged spectrum to shape {spectrum.shape}")
        
        # Final check: ensure dimensions match
        if len(frequencies) != spectrum.shape[0]:
            log.warning(
                f"Dimension mismatch: freqs={len(frequencies)}, spectrum={spectrum.shape[0]}. "
                "Truncating to match."
            )
            min_len = min(len(frequencies), spectrum.shape[0])
            frequencies = frequencies[:min_len]
            spectrum = spectrum[:min_len]
        
        return frequencies, spectrum, self.context.component_label
    
    def load_mode_at_frequency(
        self, 
        frequency: float, 
        z_layer: Optional[int] = None
    ) -> Tuple[np.ndarray, float, dict]:
        """Load mode data at specified frequency.
        
        Parameters
        ----------
        frequency : float
            Target frequency in GHz
        z_layer : int, optional
            Z-layer override (uses context z_layer if not specified)
        
        Returns
        -------
        mode_data : np.ndarray
            Complex mode array with shape (ny, nx, 3) or (ny, nx) if single component
        actual_frequency : float
            Actual frequency loaded (nearest to requested)
        metadata : dict
            Additional metadata about the loaded mode
        """
        modes_path, freqs_path, _ = self._resolve_paths()
        
        if modes_path is None or freqs_path is None:
            raise RuntimeError(
                f"Mode data not found for dataset '{self.context.dataset_name}'. "
                "Run compute_modes() first."
            )
        
        z = z_layer if z_layer is not None else self.context.z_layer
        
        # Load frequencies and find nearest index
        frequencies = np.array(self.zarr_file[freqs_path])
        freq_idx = np.argmin(np.abs(frequencies - frequency))
        actual_freq = frequencies[freq_idx]
        
        if abs(actual_freq - frequency) > 0.1:
            log.warning(
                f"Requested frequency {frequency:.3f} GHz not exact, "
                f"using {actual_freq:.3f} GHz"
            )
        
        # Get mode shape for z_layer normalization
        mode_shape = self.zarr_file[modes_path].shape
        n_z = mode_shape[1]
        
        # Handle negative z indexing
        if z < 0:
            z = n_z + z
        
        # Load mode data
        mode_data = self.zarr_file[modes_path][freq_idx, z, :, :, :]
        
        # Apply component extraction if specified
        if self.context.component_index is not None:
            mode_data = mode_data[:, :, self.context.component_index]
            log.debug(f"Extracted component {self.context.component_index}")
        
        metadata = {
            "frequency_index": freq_idx,
            "actual_frequency": actual_freq,
            "requested_frequency": frequency,
            "z_layer": z,
            "component": self.context.component_index,
            "component_label": self.context.component_label,
        }
        
        return mode_data, actual_freq, metadata
    
    def get_available_frequencies(self) -> Optional[np.ndarray]:
        """Get array of available frequencies."""
        _, freqs_path, _ = self._resolve_paths()
        if freqs_path is not None:
            return np.array(self.zarr_file[freqs_path])
        return None
