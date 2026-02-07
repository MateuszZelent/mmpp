"""
Data access mixin for FMR mode analyzer.

This module handles loading, caching, and accessing data from zarr files
for FMR mode analysis.
"""

import numpy as np
import zarr
from typing import Optional, Any, Union, List, Tuple
from pathlib import Path

from ..models import Peak, FMRModeData
from ..compatibility import require_dependency, SCIPY_AVAILABLE
from ....cli.logging_config import get_mmpp_logger

if SCIPY_AVAILABLE:
    from scipy.signal import find_peaks

log = get_mmpp_logger(__name__)


class DataAccessMixin:
    """Mixin providing data access functionality for FMR mode analyzer."""
    
    def __init__(self):
        # Data storage
        self.zarr_file: Optional[zarr.Group] = None
        self.zarr_path: Optional[str] = None
        self.dataset_name: Optional[str] = None
        
        # Paths in zarr file
        self.modes_path: Optional[str] = None
        self.freqs_path: Optional[str] = None
        self.spectrum_path: Optional[str] = None
        
        # Loaded data
        self.frequencies: Optional[np.ndarray] = None
        self.spectrum: Optional[np.ndarray] = None
        
        # Spatial information
        self.dx: float = 1.0  # nm
        self.dy: float = 1.0  # nm
        
        # Cache
        self._mode_cache: dict[tuple[float, int], FMRModeData] = {}
        self._peak_cache: Optional[List[Peak]] = None
        
    def _list_available_datasets(self) -> List[str]:
        """Enumerate top-level datasets available in the zarr archive."""
        require_dependency('zarr', 'dataset enumeration')
        
        if not self.zarr_file:
            return []
            
        try:
            keys = set(self.zarr_file.group_keys()) | set(self.zarr_file.array_keys())
            return sorted({key.split("/")[0] for key in keys})
        except Exception as exc:
            log.debug("Unable to list datasets in %s: %s", self.zarr_path, exc)
            return []

    def _get_zarr_paths(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Unified path resolution for zarr datasets.

        Returns:
        --------
        Tuple[Optional[str], Optional[str], Optional[str]]
            (modes_path, freqs_path, spectrum_path) or None if not found
        """
        if not self.zarr_file or not self.dataset_name:
            return None, None, None
            
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
        
        # If not found, try scanning for paths with slice hash (from sliced FFT calls)
        # Pattern: fft/{dataset}_z{z}_m{m}_s{hash}/spectrum
        if spectrum_path is None:
            try:
                fft_keys = [k for k in self.zarr_file.keys() if k.startswith('fft/')]
                for key in fft_keys:
                    if key.startswith(f'fft/{self.dataset_name}_z') and key.endswith('/spectrum'):
                        if '_s' in key:  # Check for slice hash marker
                            spectrum_path = key
                            log.info(f"Found sliced spectrum at: {spectrum_path}")
                            break
            except Exception as e:
                log.debug(f"Error scanning for sliced spectrum paths: {e}")

        return modes_path, freqs_path, spectrum_path

    def _load_data(self) -> None:
        """Load mode and spectrum data from zarr file."""
        require_dependency('zarr', 'data loading')
        
        if not self.zarr_path:
            raise ValueError("zarr_path must be set before loading data")
            
        try:
            self.zarr_file = zarr.open(self.zarr_path, mode="r")  # type: ignore
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
        if self.freqs_path and self.zarr_file:
            self.frequencies = np.array(self.zarr_file[self.freqs_path])
            log.info(
                f"Loaded frequencies: {len(self.frequencies)} points, "
                f"range {self.frequencies[0]:.3f} - {self.frequencies[-1]:.3f} GHz"
            )
        else:
            self.frequencies = None
            log.debug("No frequency data loaded - will be computed with modes")

        # Load spectrum if available
        if self.spectrum_path and self.zarr_file:
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
            log.info(f"Loaded spectrum data: shape {self.spectrum.shape}")
        else:
            # Try to load power_sum from computed modes as fallback spectrum
            modes_power_path = f"modes/{self.dataset_name}/power_sum"
            if self.zarr_file and modes_power_path in self.zarr_file:
                self.spectrum = np.array(self.zarr_file[modes_power_path])
                if np.iscomplexobj(self.spectrum):
                    self.spectrum = np.abs(self.spectrum)
                log.info(f"Using computed modes power_sum as spectrum: shape {self.spectrum.shape}")
            else:
                self.spectrum = None

        # Get spatial information
        self._get_spatial_info()
        
        # Clear caches
        self._clear_cache()

    def _get_spatial_info(self) -> None:
        """Extract spatial information from zarr metadata."""
        if not self.zarr_file:
            return
            
        # Try to get spatial resolution from attributes
        self.dx = 1.0  # Default spatial resolution in nm
        self.dy = 1.0

        # Look for spatial attributes in various locations
        attrs_to_check = [
            self.zarr_file.attrs,
            (
                self.zarr_file[self.dataset_name].attrs
                if self.dataset_name and self.dataset_name in self.zarr_file
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
    ) -> List[Peak]:
        """
        Detect peaks in spectrum.

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
        if not hasattr(self, 'config'):
            # Set default values if config not available
            peak_threshold = 0.1
            peak_min_distance = 5
        else:
            peak_threshold = self.config.peak_threshold
            peak_min_distance = self.config.peak_min_distance
            
        if not SCIPY_AVAILABLE:
            log.warning("SciPy not available, using simple peak detection")
            # Simple peak detection without scipy
            peaks = []
            max_amplitude = np.max(spectrum)
            for i in range(1, len(spectrum) - 1):
                if (
                    spectrum[i] > spectrum[i - 1]
                    and spectrum[i] > spectrum[i + 1]
                    and spectrum[i] > peak_threshold * max_amplitude
                ):
                    peaks.append(
                        Peak(idx=i, freq=frequencies[i], amplitude=spectrum[i])
                    )
            return peaks

        try:
            # Normalize spectrum for peak detection
            norm_spectrum = spectrum / np.max(spectrum)

            # Find peaks using scipy
            peak_indices, properties = find_peaks(
                norm_spectrum,
                height=peak_threshold,
                distance=peak_min_distance,
            )

            # Create peak objects
            peaks = []
            for idx in peak_indices:
                peaks.append(
                    Peak(
                        idx=int(idx), 
                        freq=float(frequencies[idx]), 
                        amplitude=float(spectrum[idx])
                    )
                )

            # Sort by amplitude (descending)
            peaks.sort(key=lambda p: p.amplitude, reverse=True)

            log.debug(f"Detected {len(peaks)} peaks")
            return peaks

        except Exception as e:
            log.error(f"Peak detection failed: {e}")
            return []

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

        # Check cache first
        cache_key = (frequency, z_layer)
        if cache_key in self._mode_cache:
            return self._mode_cache[cache_key]

        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))
        actual_freq = self.frequencies[freq_idx]
        
        if abs(actual_freq - frequency) > 0.001:  # 1 MHz tolerance
            log.warning(f"Requested frequency {frequency:.3f} GHz not found, "
                       f"using closest: {actual_freq:.3f} GHz")

        # Load mode data
        if not self.modes_path or not self.zarr_file:
            raise RuntimeError("Mode data not available")

        try:
            modes_array_zarr = self.zarr_file[self.modes_path]
            # Convert to numpy array for easier handling
            modes_array = np.array(modes_array_zarr)
            
            # Handle different array shapes
            if modes_array.ndim == 5:  # (freq, z, y, x, components)
                if z_layer >= modes_array.shape[1] or z_layer < -modes_array.shape[1]:
                    raise ValueError(f"Z-layer {z_layer} out of range [0, {modes_array.shape[1]})")
                mode_data = modes_array[freq_idx, z_layer, :, :, :]
            elif modes_array.ndim == 4:  # (freq, y, x, components) - single z-layer
                if z_layer != 0:
                    raise ValueError("Z-layer selection not supported for this dataset")
                mode_data = modes_array[freq_idx, :, :, :]
            else:
                raise ValueError(f"Unsupported mode array shape: {modes_array.shape}")

            # Calculate spatial extent
            ny, nx = mode_data.shape[:2]
            extent = (0, nx * self.dx, 0, ny * self.dy)

            # Create FMRModeData object
            fmr_mode = FMRModeData(
                frequency=actual_freq,
                mode_array=mode_data,
                extent=extent,
                metadata={'z_layer': z_layer, 'freq_index': freq_idx}
            )

            # Cache the result
            self._mode_cache[cache_key] = fmr_mode
            
            return fmr_mode

        except Exception as e:
            log.error(f"Failed to load mode data for frequency {frequency:.3f} GHz: {e}")
            raise RuntimeError(f"Failed to load mode data: {e}")

    def get_peaks(self, use_cache: bool = True) -> List[Peak]:
        """
        Get detected peaks from spectrum.
        
        Parameters:
        -----------
        use_cache : bool, default True
            Whether to use cached results
            
        Returns:
        --------
        List[Peak]
            List of detected peaks
        """
        if use_cache and self._peak_cache is not None:
            return self._peak_cache
            
        if self.spectrum is None or self.frequencies is None:
            log.warning("No spectrum data available for peak detection")
            return []
            
        peaks = self._detect_peaks(self.spectrum, self.frequencies)
        
        if use_cache:
            self._peak_cache = peaks
            
        return peaks

    def get_available_frequencies(self) -> Optional[np.ndarray]:
        """Get array of available frequencies."""
        return self.frequencies.copy() if self.frequencies is not None else None

    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets in zarr file."""
        return self._list_available_datasets()

    def is_data_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self.zarr_file is not None

    def get_data_info(self) -> dict[str, Any]:
        """Get information about loaded data."""
        info = {
            'zarr_path': self.zarr_path,
            'dataset_name': self.dataset_name,
            'data_loaded': self.is_data_loaded(),
            'frequencies_available': self.frequencies is not None,
            'spectrum_available': self.spectrum is not None,
            'modes_available': self.modes_path is not None,
            'spatial_resolution': {'dx': self.dx, 'dy': self.dy}
        }
        
        if self.frequencies is not None:
            info['frequency_range'] = {
                'min': float(np.min(self.frequencies)),
                'max': float(np.max(self.frequencies)),
                'count': len(self.frequencies)
            }
            
        if self.spectrum is not None:
            info['spectrum_shape'] = self.spectrum.shape
            
        return info

    def _clear_cache(self) -> None:
        """Clear internal caches."""
        self._mode_cache.clear()
        self._peak_cache = None
        
    def reload_data(self, force: bool = False) -> None:
        """
        Reload data from zarr file.
        
        Parameters:
        -----------
        force : bool, default False
            Force reload even if data appears unchanged
        """
        if force:
            self._clear_cache()
            
        if self.zarr_path:
            self._load_data()
        else:
            log.warning("No zarr_path set, cannot reload data")
