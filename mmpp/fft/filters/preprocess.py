"""Pre-FFT filtering functions."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    import scipy.signal

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    scipy = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

FilterType = str | Sequence[str] | Mapping[str, Any]


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
            logger.debug("scipy detrend failed, using NumPy linear fit", exc_info=True)

    n_time = int(arr.shape[0])
    if n_time < 2:
        return remove_mean(arr)
    x = np.arange(n_time, dtype=float)
    x -= np.mean(x)
    denominator = float(np.sum(x * x))
    reshape = (n_time,) + (1,) * (arr.ndim - 1)
    mean = np.mean(arr, axis=0, keepdims=True)
    slope = np.sum(x.reshape(reshape) * (arr - mean), axis=0, keepdims=True)
    slope = slope / denominator
    return arr - (mean + slope * x.reshape(reshape))


def remove_mean_and_static(data: np.ndarray) -> np.ndarray:
    """Compose remove_mean + remove_static."""
    # Static subtraction followed by centering leaves no artificial DC term.
    # The opposite order reintroduces ``-centered[0]`` as a constant offset.
    return remove_mean(remove_static(data))


def _apply_tracewise(data: np.ndarray, fn) -> np.ndarray:
    """Apply a 1D function over all traces along axis 0."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        return fn(arr)

    out = np.zeros(arr.shape, dtype=np.result_type(arr.dtype, np.float64))
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
        raise ImportError(
            "Savitzky-Golay preprocessing requires scipy; install scipy or "
            "remove the savgol_smooth filter"
        )

    if isinstance(window_length, (bool, np.bool_)) or not isinstance(
        window_length, (int, np.integer)
    ):
        raise TypeError("Savitzky-Golay window_length must be an integer")
    if isinstance(polyorder, (bool, np.bool_)) or not isinstance(
        polyorder, (int, np.integer)
    ):
        raise TypeError("Savitzky-Golay polyorder must be an integer")
    if int(window_length) < 3:
        raise ValueError("Savitzky-Golay window_length must be >= 3")
    if int(polyorder) < 0:
        raise ValueError("Savitzky-Golay polyorder must be >= 0")

    n_time = int(arr.shape[0])
    wl = min(int(window_length), n_time // 2 * 2 - 1)
    if wl < 3:
        raise ValueError(
            f"Savitzky-Golay requires at least 3 time samples, got {n_time}"
        )
    if wl % 2 == 0:
        wl -= 1
    po = int(polyorder)
    if po >= wl:
        raise ValueError(
            f"Savitzky-Golay polyorder ({po}) must be smaller than the "
            f"effective window length ({wl})"
        )

    if arr.ndim == 1:
        return scipy.signal.savgol_filter(arr, window_length=wl, polyorder=po)
    return _apply_tracewise(
        arr,
        lambda trace: scipy.signal.savgol_filter(trace, window_length=wl, polyorder=po),
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

    lam_value = float(lam)
    p_value = float(p)
    if not np.isfinite(lam_value) or lam_value <= 0:
        raise ValueError("baseline lam must be finite and positive")
    if not np.isfinite(p_value) or not 0.0 < p_value < 1.0:
        raise ValueError("baseline p must be finite and in the interval (0, 1)")
    if isinstance(niter, (bool, np.bool_)) or not isinstance(niter, (int, np.integer)):
        raise TypeError("baseline niter must be an integer")
    if int(niter) < 1:
        raise ValueError("baseline niter must be >= 1")

    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "Asymmetric least-squares baseline correction requires scipy; "
            "install scipy or remove the baseline_correction pre-filter"
        )

    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
    except Exception:
        raise ImportError(
            "Asymmetric least-squares baseline correction requires scipy.sparse"
        ) from None

    def _baseline_als_1d(y: np.ndarray) -> np.ndarray:
        length = len(y)
        if length < 3:
            return y
        d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2))
        w = np.ones(length)
        for _ in range(int(niter)):
            w_diag = sparse.spdiags(w, 0, length, length)
            z_matrix = w_diag + lam_value * d.dot(d.T)
            z = spsolve(z_matrix, w * y)
            w = p_value * (y > z) + (1.0 - p_value) * (y <= z)
        return y - z

    if arr.ndim == 1:
        return _baseline_als_1d(arr)
    return _apply_tracewise(arr, _baseline_als_1d)


def _high_pass_1d(y: np.ndarray, cutoff_fraction: float) -> np.ndarray:
    n = len(y)
    if n == 0:
        return y
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n) / 0.5  # fraction of Nyquist, in [0, 1]
    cutoff = float(cutoff_fraction)
    shape = 1.0 - 1.0 / (1.0 + (freqs / cutoff) ** 4)
    return np.fft.irfft(fft * shape, n=n)


def high_pass(data: np.ndarray, cutoff_fraction: float = 0.01) -> np.ndarray:
    """FFT-domain high-pass filter."""
    if not 0.0 < float(cutoff_fraction) <= 1.0:
        raise ValueError("cutoff_fraction must be in the interval (0, 1]")
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr
    if arr.ndim == 1:
        return _high_pass_1d(arr, cutoff_fraction)
    return _apply_tracewise(arr, lambda y: _high_pass_1d(y, cutoff_fraction))


def _band_pass_1d(
    y: np.ndarray, low_fraction: float, high_fraction: float
) -> np.ndarray:
    n = len(y)
    if n == 0:
        return y
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n) / 0.5  # fraction of Nyquist, in [0, 1]
    low = float(low_fraction)
    high = float(high_fraction)
    hp = 1.0 - 1.0 / (1.0 + (freqs / low) ** 4)
    lp = 1.0 / (1.0 + (freqs / high) ** 4)
    return np.fft.irfft(fft * (hp * lp), n=n)


def band_pass(
    data: np.ndarray,
    low_fraction: float = 0.01,
    high_fraction: float = 0.9,
) -> np.ndarray:
    """FFT-domain band-pass filter."""
    low = float(low_fraction)
    high = float(high_fraction)
    if not 0.0 < low < high <= 1.0:
        raise ValueError(
            "band-pass fractions must satisfy 0 < low_fraction < high_fraction <= 1"
        )
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr
    if arr.ndim == 1:
        return _band_pass_1d(arr, low, high)
    return _apply_tracewise(arr, lambda y: _band_pass_1d(y, low, high))


def spectral_derivative(
    data: np.ndarray,
    order: int = 1,
    spacing: float = 1.0,
) -> np.ndarray:
    """Derivative along time axis using the physical sample spacing."""
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr
    if isinstance(order, (bool, np.bool_)) or int(order) != order or int(order) < 1:
        raise ValueError("spectral derivative order must be a positive integer")
    ord_int = int(order)
    spacing_value = float(spacing)
    if not np.isfinite(spacing_value) or spacing_value <= 0.0:
        raise ValueError("spectral derivative spacing must be finite and positive")
    out = arr
    for _ in range(ord_int):
        out = np.gradient(out, spacing_value, axis=0)
    return out


def apply_single_filter(
    data: np.ndarray, filter_type: str, **parameters: Any
) -> np.ndarray:
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
        return savgol_smooth(
            data,
            window_length=int(parameters.get("window_length", 11)),
            polyorder=int(parameters.get("polyorder", 3)),
        )
    if name == "baseline_correction":
        return baseline_correction(
            data,
            lam=float(parameters.get("lam", 1e5)),
            p=float(parameters.get("p", 0.01)),
            niter=int(parameters.get("niter", 10)),
        )
    if name == "high_pass":
        return high_pass(
            data, cutoff_fraction=float(parameters.get("cutoff_fraction", 0.01))
        )
    if name == "band_pass":
        return band_pass(
            data,
            low_fraction=float(parameters.get("low_fraction", 0.01)),
            high_fraction=float(parameters.get("high_fraction", 0.9)),
        )
    if name == "spectral_derivative":
        return spectral_derivative(
            data,
            order=int(parameters.get("order", 1)),
            spacing=float(parameters.get("spacing", parameters.get("dt", 1.0))),
        )

    raise ValueError(f"Unknown preprocessing filter: {filter_type!r}")


def apply_filter(
    data: np.ndarray, filter_type: FilterType, **parameters: Any
) -> np.ndarray:
    """Apply one filter or a filter chain."""
    if isinstance(filter_type, Mapping):
        result = data
        for name, option in filter_type.items():
            if option is False or option is None:
                continue
            configured = option if isinstance(option, Mapping) else {}
            result = apply_single_filter(result, str(name), **dict(configured))
        return result
    if isinstance(filter_type, (list, tuple)):
        result = data
        for name in filter_type:
            result = apply_single_filter(result, str(name))
        return result
    return apply_single_filter(data, str(filter_type), **parameters)
