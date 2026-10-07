"""Windowing helpers shared by FFT modules."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

try:
    import scipy.signal

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    scipy = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

WINDOW_TYPES = Literal[
    "none",
    "hann",
    "hamming",
    "blackman",
    "bartlett",
    "kaiser",
    "tukey",
    "gaussian",
    "flattop",
    "nuttall",
]


def get_window(window_type: str, n_time: int) -> np.ndarray:
    """Return a 1D window array for the requested type."""
    if n_time <= 0:
        return np.array([], dtype=float)

    name = str(window_type or "none").lower()
    if n_time == 1:
        return np.ones(1, dtype=float)
    if name == "none":
        return np.ones(n_time, dtype=float)
    if name == "hann":
        return np.hanning(n_time)
    if name == "hamming":
        return np.hamming(n_time)
    if name == "blackman":
        return np.blackman(n_time)
    if name == "bartlett":
        return np.bartlett(n_time)
    if name == "kaiser":
        if _SCIPY_AVAILABLE:
            return scipy.signal.windows.kaiser(n_time, beta=8.6)
        return np.kaiser(n_time, 8.6)
    if name == "tukey":
        if _SCIPY_AVAILABLE:
            return scipy.signal.windows.tukey(n_time, alpha=0.25)
        x = np.linspace(0.0, 1.0, n_time)
        alpha = 0.25
        window = np.ones(n_time, dtype=float)
        left = x < alpha / 2.0
        right = x >= 1.0 - alpha / 2.0
        window[left] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[left] / alpha - 1.0)))
        window[right] = 0.5 * (
            1.0 + np.cos(np.pi * (2.0 * x[right] / alpha - 2.0 / alpha + 1.0))
        )
        return window
    if name == "gaussian":
        if _SCIPY_AVAILABLE:
            return scipy.signal.windows.gaussian(n_time, std=n_time / 6)
        x = np.arange(n_time, dtype=float) - (n_time - 1.0) / 2.0
        return np.exp(-0.5 * (x / (n_time / 6.0)) ** 2)
    if name == "flattop":
        if _SCIPY_AVAILABLE:
            return scipy.signal.windows.flattop(n_time)
        x = 2.0 * np.pi * np.arange(n_time, dtype=float) / max(n_time - 1, 1)
        return (
            0.21557895
            - 0.41663158 * np.cos(x)
            + 0.277263158 * np.cos(2.0 * x)
            - 0.083578947 * np.cos(3.0 * x)
            + 0.006947368 * np.cos(4.0 * x)
        )
    if name == "nuttall":
        if _SCIPY_AVAILABLE:
            return scipy.signal.windows.nuttall(n_time)
        x = 2.0 * np.pi * np.arange(n_time, dtype=float) / max(n_time - 1, 1)
        return (
            0.355768
            - 0.487396 * np.cos(x)
            + 0.144232 * np.cos(2.0 * x)
            - 0.012604 * np.cos(3.0 * x)
        )

    logger.warning("Unknown window type '%s', using no window", window_type)
    return np.ones(n_time, dtype=float)


def apply_window(data: np.ndarray, window_type: str) -> np.ndarray:
    """Apply a window function along the first (time) axis."""
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0:
        return arr

    window = get_window(window_type, arr.shape[0])
    if arr.ndim == 1:
        return arr * window

    window_shape = [1] * arr.ndim
    window_shape[0] = arr.shape[0]
    return arr * window.reshape(window_shape)
