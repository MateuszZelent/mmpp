"""
Peak Detection Utilities

Provides peak detection algorithms for FMR spectrum analysis.
Includes both scipy-based and simple fallback implementations.
"""

import logging

import numpy as np

from ..models import Peak

log = logging.getLogger("mmpp.fft.modes")

# Try to import scipy
try:
    from scipy.signal import find_peaks as scipy_find_peaks

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _prepare_peak_inputs(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    *,
    threshold: float,
    min_distance: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Validate inputs and reduce component axes without losing frequency."""
    values = np.asarray(spectrum, dtype=float)
    freq = np.asarray(frequencies, dtype=float)
    if freq.ndim != 1 or freq.size < 3:
        raise ValueError("frequencies must be a 1D array with at least 3 samples")
    if not np.all(np.isfinite(freq)) or np.any(freq < 0):
        raise ValueError("frequencies must be finite and non-negative")
    if values.ndim == 0:
        raise ValueError("spectrum must have a frequency dimension")
    if values.ndim == 1:
        if values.size != freq.size:
            raise ValueError("spectrum and frequencies must have matching lengths")
        reduced = values
    elif values.shape[0] == freq.size:
        reduced = np.mean(values, axis=tuple(range(1, values.ndim)))
    elif values.shape[-1] == freq.size:
        reduced = np.mean(values, axis=tuple(range(values.ndim - 1)))
    else:
        raise ValueError("Cannot identify the frequency axis in spectrum")
    if not np.all(np.isfinite(reduced)) or np.any(reduced < 0):
        raise ValueError("spectrum must contain finite non-negative power")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or threshold_value < 0:
        raise ValueError("threshold must be finite and non-negative")
    if (
        isinstance(min_distance, (bool, np.bool_))
        or int(min_distance) != min_distance
        or int(min_distance) < 1
    ):
        raise ValueError("min_distance must be a positive integer")
    return reduced, freq, threshold_value, int(min_distance)


def detect_peaks_scipy(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    threshold: float = 0.1,
    min_distance: int = 5,
) -> list[Peak]:
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

    spectrum_1d, frequencies, threshold, min_distance = _prepare_peak_inputs(
        spectrum,
        frequencies,
        threshold=threshold,
        min_distance=min_distance,
    )

    # Normalize spectrum for consistent threshold
    norm_spectrum = (
        spectrum_1d / np.max(spectrum_1d) if np.max(spectrum_1d) > 0 else spectrum_1d
    )

    peak_indices, _ = scipy_find_peaks(
        norm_spectrum,
        height=threshold,
        distance=min_distance,
    )

    peaks = []
    for idx in peak_indices:
        peaks.append(
            Peak(
                idx=int(idx),
                freq=float(frequencies[idx]),
                amplitude=float(spectrum_1d[idx]),
            )
        )

    peaks.sort(key=lambda p: p.amplitude, reverse=True)
    log.debug(f"SciPy detected {len(peaks)} peaks")
    return peaks


def detect_peaks_simple(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    threshold: float = 0.1,
    min_distance: int = 5,
) -> list[Peak]:
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
    spectrum, frequencies, threshold, min_distance = _prepare_peak_inputs(
        spectrum,
        frequencies,
        threshold=threshold,
        min_distance=min_distance,
    )
    max_val = np.max(spectrum)
    if max_val == 0:
        return []

    threshold_abs = threshold * max_val
    candidates: list[int] = []

    # Find local maxima
    for i in range(1, len(spectrum) - 1):
        if (
            spectrum[i] > spectrum[i - 1]
            and spectrum[i] > spectrum[i + 1]
            and spectrum[i] > threshold_abs
        ):
            candidates.append(i)

    # Match scipy's distance semantics: retain stronger candidates first.
    selected: list[int] = []
    for index in sorted(candidates, key=lambda i: spectrum[i], reverse=True):
        if all(abs(index - other) >= min_distance for other in selected):
            selected.append(index)
    peaks = [
        Peak(
            idx=int(index),
            freq=float(frequencies[index]),
            amplitude=float(spectrum[index]),
        )
        for index in selected
    ]

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
) -> list[Peak]:
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
        return detect_peaks_simple(
            spectrum, frequencies, threshold, min_distance=min_distance
        )


__all__ = [
    "detect_peaks",
    "detect_peaks_scipy",
    "detect_peaks_simple",
    "SCIPY_AVAILABLE",
]
