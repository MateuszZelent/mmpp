"""
Peak Detection Utilities

Provides peak detection algorithms for FMR spectrum analysis.
Includes both scipy-based and simple fallback implementations.
"""

import numpy as np
from typing import List
import logging

from ..models import Peak

log = logging.getLogger("mmpp.fft.modes")

# Try to import scipy
try:
    from scipy.signal import find_peaks as scipy_find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def detect_peaks_scipy(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    threshold: float = 0.1,
    min_distance: int = 5,
) -> List[Peak]:
    """Detect peaks using scipy.signal.find_peaks.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Power spectrum data (1D array)
    frequencies : np.ndarray
        Frequency array in GHz (same length as spectrum)
    threshold : float
        Minimum peak height as fraction of max (default: 0.1)
    min_distance : int
        Minimum distance between peaks in samples (default: 5)
        
    Returns
    -------
    List[Peak]
        List of detected peaks, sorted by amplitude (descending)
        
    Raises
    ------
    ImportError
        If scipy is not available
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required for scipy peak detection")
    
    # Handle multi-dimensional spectrum
    if spectrum.ndim > 1:
        # Average over components for peak detection
        if spectrum.shape[-1] <= 3:  # Likely component dimension
            spectrum_1d = np.mean(spectrum, axis=-1)
        else:
            spectrum_1d = spectrum.flatten()
        log.debug(f"Converted {spectrum.ndim}D spectrum to 1D for peak detection")
    else:
        spectrum_1d = spectrum
    
    # Normalize spectrum for consistent threshold
    norm_spectrum = spectrum_1d / np.max(spectrum_1d) if np.max(spectrum_1d) > 0 else spectrum_1d
    
    try:
        # Find peaks using scipy
        peak_indices, properties = scipy_find_peaks(
            norm_spectrum,
            height=threshold,
            distance=min_distance,
        )
        
        # Create Peak objects
        peaks = []
        for idx in peak_indices:
            peaks.append(
                Peak(
                    idx=int(idx),
                    freq=float(frequencies[idx]),
                    amplitude=float(spectrum_1d[idx])
                )
            )
        
        # Sort by amplitude (descending)
        peaks.sort(key=lambda p: p.amplitude, reverse=True)
        
        log.debug(f"SciPy detected {len(peaks)} peaks")
        return peaks
        
    except Exception as e:
        log.error(f"SciPy peak detection failed: {e}")
        return []


def detect_peaks_simple(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    threshold: float = 0.1,
) -> List[Peak]:
    """Simple threshold-based peak detection (fallback when scipy unavailable).
    
    Detects local maxima that exceed the threshold.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Power spectrum data (1D array)
    frequencies : np.ndarray
        Frequency array in GHz
    threshold : float
        Minimum peak height as fraction of max (default: 0.1)
        
    Returns
    -------
    List[Peak]
        List of detected peaks, sorted by amplitude (descending)
    """
    max_val = np.max(spectrum)
    if max_val == 0:
        return []
    
    threshold_abs = threshold * max_val
    peaks = []
    
    # Find local maxima
    for i in range(1, len(spectrum) - 1):
        if (spectrum[i] > spectrum[i - 1] and 
            spectrum[i] > spectrum[i + 1] and 
            spectrum[i] > threshold_abs):
            peaks.append(
                Peak(
                    idx=i,
                    freq=float(frequencies[i]),
                    amplitude=float(spectrum[i])
                )
            )
    
    # Sort by amplitude (descending)
    peaks.sort(key=lambda p: p.amplitude, reverse=True)
    
    log.debug(f"Simple detection found {len(peaks)} peaks")
    return peaks


def detect_peaks(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    threshold: float = 0.1,
    min_distance: int = 5,
    use_scipy: bool = True,
) -> List[Peak]:
    """Auto-detect peaks using best available method.
    
    Uses scipy if available, falls back to simple detection.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Power spectrum data
    frequencies : np.ndarray
        Frequency array in GHz
    threshold : float
        Minimum peak height as fraction of max
    min_distance : int
        Minimum distance between peaks (only for scipy)
    use_scipy : bool
        Prefer scipy if available (default: True)
        
    Returns
    -------
    List[Peak]
        List of detected peaks
    """
    if use_scipy and SCIPY_AVAILABLE:
        return detect_peaks_scipy(spectrum, frequencies, threshold, min_distance)
    else:
        if use_scipy and not SCIPY_AVAILABLE:
            log.warning("SciPy not available, using simple peak detection")
        return detect_peaks_simple(spectrum, frequencies, threshold)


__all__ = [
    "detect_peaks",
    "detect_peaks_scipy",
    "detect_peaks_simple",
    "SCIPY_AVAILABLE",
]
