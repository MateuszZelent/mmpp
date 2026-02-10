"""Pre-FFT filtering functions."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Union

import numpy as np

try:
    import scipy.signal

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    scipy = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

FilterType = Union[str, Sequence[str]]


def remove_mean(data: np.ndarray) -> np.ndarray:
    """Remove per-cell temporal mean."""
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr
    return arr - np.mean(arr, axis=0, keepdims=True)


def remove_static(data: np.ndarray) -> np.ndarray:
    """Subtract first time sample."""
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr
    return arr - arr[0:1, ...]


def detrend_linear(data: np.ndarray) -> np.ndarray:
    """Linear detrend along time axis."""
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr

    if _SCIPY_AVAILABLE:
        try:
            return scipy.signal.detrend(arr, axis=0)
        except Exception:
            logger.debug("scipy detrend failed, falling back to mean removal", exc_info=True)
    return remove_mean(arr)


def remove_mean_and_static(data: np.ndarray) -> np.ndarray:
    """Compose remove_mean + remove_static."""
    centered = remove_mean(data)
    return remove_static(centered)


def _apply_tracewise(data: np.ndarray, fn) -> np.ndarray:
    """Apply a 1D function over all traces along axis 0."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        return fn(arr)

    out = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape[1:]):
        out[(slice(None),) + idx] = fn(arr[(slice(None),) + idx])
    return out


def savgol_smooth(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing along time axis."""
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr
    if not _SCIPY_AVAILABLE:
        logger.warning("Savitzky-Golay requires scipy; returning input unchanged")
        return arr

    n_time = int(arr.shape[0])
    wl = min(int(window_length), n_time // 2 * 2 - 1)
    if wl < 5:
        return arr
    if wl % 2 == 0:
        wl -= 1
    po = min(int(polyorder), wl - 1)
    if po < 1:
        po = 1

    if arr.ndim == 1:
        return scipy.signal.savgol_filter(arr, window_length=wl, polyorder=po)
    return _apply_tracewise(
        arr, lambda trace: scipy.signal.savgol_filter(trace, window_length=wl, polyorder=po)
    )


def baseline_correction(
    data: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
) -> np.ndarray:
    """Asymmetric least squares baseline correction."""
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr

    if not _SCIPY_AVAILABLE:
        return remove_mean(arr)

    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
    except Exception:
        return remove_mean(arr)

    def _baseline_als_1d(y: np.ndarray) -> np.ndarray:
        length = len(y)
        if length < 3:
            return y
        d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2))
        w = np.ones(length)
        for _ in range(max(1, int(niter))):
            w_diag = sparse.spdiags(w, 0, length, length)
            z_matrix = w_diag + float(lam) * d.dot(d.T)
            z = spsolve(z_matrix, w * y)
            w = float(p) * (y > z) + (1.0 - float(p)) * (y <= z)
        return y - z

    if arr.ndim == 1:
        return _baseline_als_1d(arr)
    return _apply_tracewise(arr, _baseline_als_1d)


def _high_pass_1d(y: np.ndarray, cutoff_fraction: float) -> np.ndarray:
    n = len(y)
    if n == 0:
        return y
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n)
    cutoff = max(float(cutoff_fraction), 1e-10)
    shape = 1.0 - 1.0 / (1.0 + (freqs / cutoff) ** 4)
    return np.fft.irfft(fft * shape, n=n)


def high_pass(data: np.ndarray, cutoff_fraction: float = 0.01) -> np.ndarray:
    """FFT-domain high-pass filter."""
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr
    if arr.ndim == 1:
        return _high_pass_1d(arr, cutoff_fraction)
    return _apply_tracewise(arr, lambda y: _high_pass_1d(y, cutoff_fraction))


def _band_pass_1d(y: np.ndarray, low_fraction: float, high_fraction: float) -> np.ndarray:
    n = len(y)
    if n == 0:
        return y
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n)
    low = max(float(low_fraction), 1e-10)
    high = max(float(high_fraction), low + 1e-9)
    hp = 1.0 - 1.0 / (1.0 + (freqs / low) ** 4)
    lp = 1.0 / (1.0 + (freqs / high) ** 4)
    return np.fft.irfft(fft * (hp * lp), n=n)


def band_pass(
    data: np.ndarray,
    low_fraction: float = 0.01,
    high_fraction: float = 0.9,
) -> np.ndarray:
    """FFT-domain band-pass filter."""
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr
    if arr.ndim == 1:
        return _band_pass_1d(arr, low_fraction, high_fraction)
    return _apply_tracewise(arr, lambda y: _band_pass_1d(y, low_fraction, high_fraction))


def spectral_derivative(data: np.ndarray, order: int = 1) -> np.ndarray:
    """Derivative along time axis."""
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr
    ord_int = max(1, int(order))
    out = arr
    for _ in range(ord_int):
        out = np.gradient(out, axis=0)
    return out


def apply_single_filter(data: np.ndarray, filter_type: str) -> np.ndarray:
    """Apply one preprocessing filter by name."""
    name = str(filter_type or "none").lower()
    if name == "none":
        return data
    if name in {"remove_mean", "remove_average"}:
        return remove_mean(data)
    if name == "remove_static":
        return remove_static(data)
    if name in {"detrend", "detrend_linear"}:
        return detrend_linear(data)
    if name == "remove_mean_and_static":
        return remove_mean_and_static(data)
    if name == "savgol_smooth":
        return savgol_smooth(data)
    if name == "baseline_correction":
        return baseline_correction(data)
    if name == "high_pass":
        return high_pass(data)
    if name == "band_pass":
        return band_pass(data)
    if name == "spectral_derivative":
        return spectral_derivative(data)

    logger.warning("Unknown preprocessing filter '%s', returning input unchanged", filter_type)
    return data


def apply_filter(data: np.ndarray, filter_type: FilterType) -> np.ndarray:
    """Apply one filter or a filter chain."""
    if isinstance(filter_type, (list, tuple)):
        result = data
        for name in filter_type:
            result = apply_single_filter(result, str(name))
        return result
    return apply_single_filter(data, str(filter_type))
