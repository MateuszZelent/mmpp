"""Trajectory filtering helpers."""

from __future__ import annotations

import warnings

import numpy as np

from ..core.models import TrajectoryResult

try:
    from scipy.signal import medfilt, savgol_filter

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - tested through fallback path
    medfilt = None  # type: ignore[assignment]
    savgol_filter = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average used as SciPy-free fallback."""
    if window <= 1:
        return np.asarray(signal, dtype=float)

    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def filter_trajectory(
    trajectory: TrajectoryResult,
    *,
    method: str = "savgol",
    window: int = 11,
    polyorder: int = 3,
) -> TrajectoryResult:
    """Return filtered trajectory using the selected method."""
    method_norm = method.lower()

    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)

    window = max(int(window), 3)
    if window % 2 == 0:
        window += 1

    if method_norm == "savgol":
        if SCIPY_AVAILABLE and savgol_filter is not None and x.size >= window:
            po = min(int(polyorder), window - 1)
            x_filtered = savgol_filter(x, window_length=window, polyorder=po)
            y_filtered = savgol_filter(y, window_length=window, polyorder=po)
        else:
            if not SCIPY_AVAILABLE:
                warnings.warn(
                    "SciPy is unavailable; Savitzky-Golay fallback to moving average.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            x_filtered = _moving_average(x, window)
            y_filtered = _moving_average(y, window)
    elif method_norm == "median":
        if SCIPY_AVAILABLE and medfilt is not None:
            x_filtered = medfilt(x, kernel_size=window)
            y_filtered = medfilt(y, kernel_size=window)
        else:
            warnings.warn(
                "SciPy is unavailable; median fallback to moving average.",
                RuntimeWarning,
                stacklevel=2,
            )
            x_filtered = _moving_average(x, window)
            y_filtered = _moving_average(y, window)
    else:
        raise ValueError("method must be 'savgol' or 'median'")

    metadata = dict(trajectory.metadata)
    metadata.update(
        {
            "filtered": True,
            "filter_method": method_norm,
            "filter_window": int(window),
            "filter_polyorder": int(polyorder),
        }
    )

    return TrajectoryResult(
        time=np.asarray(trajectory.time, dtype=float),
        x=np.asarray(x_filtered, dtype=float),
        y=np.asarray(y_filtered, dtype=float),
        polarity=np.asarray(trajectory.polarity, dtype=int),
        method=f"{trajectory.method}+{method_norm}",
        confidence=np.asarray(trajectory.confidence, dtype=float),
        metadata=metadata,
    )
