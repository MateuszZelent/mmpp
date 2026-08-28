"""Noise estimation and filtering utilities for hysteresis loops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..compute import numerical_derivative

try:
    from scipy.signal import butter, filtfilt, medfilt, savgol_filter

    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    butter = filtfilt = medfilt = savgol_filter = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False


@dataclass
class NoiseStats:
    """Noise summary computed from saturation-like regions."""

    snr: float
    rms_noise: float
    signal_level: float
    saturation_fraction: float
    filter_fallback: bool = False


@dataclass
class AnomalyReport:
    """Detected anomalies for hysteresis quality checks."""

    outlier_indices: list[int]
    discontinuity_indices: list[int]
    non_closing_loop: bool
    closing_error: float


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    half = window // 2
    out = np.empty_like(values)
    for i in range(values.size):
        start = max(0, i - half)
        stop = min(values.size, i + half + 1)
        out[i] = np.median(values[start:stop])
    return out


def estimate_noise_level(
    magnetization: np.ndarray,
    field: np.ndarray,
    saturation_mask: np.ndarray | None = None,
) -> NoiseStats:
    """Estimate SNR and RMS noise from approximately saturated regions."""
    mag = np.asarray(magnetization, dtype=float).reshape(-1)
    fld = np.asarray(field, dtype=float).reshape(-1)

    if mag.size == 0:
        return NoiseStats(float("nan"), float("nan"), float("nan"), 0.0)

    if saturation_mask is None:
        derivative = np.abs(numerical_derivative(fld, mag))
        cutoff = float(np.nanpercentile(derivative, 25)) if derivative.size else 0.0
        saturation_mask = derivative <= cutoff

    mask = np.asarray(saturation_mask, dtype=bool)
    if mask.size != mag.size:
        mask = np.resize(mask, mag.shape)

    sat_values = mag[mask]
    if sat_values.size < 3:
        sat_values = mag

    rms_noise = float(np.nanstd(sat_values))
    signal_level = float(np.nanmean(np.abs(sat_values)))
    if not np.isfinite(signal_level) or signal_level == 0.0:
        signal_level = float(np.nanmax(np.abs(mag)))

    if rms_noise <= 0.0:
        snr = float("inf")
    else:
        snr = float(signal_level / rms_noise)

    sat_fraction = float(np.count_nonzero(mask) / max(mask.size, 1))
    return NoiseStats(
        snr=snr,
        rms_noise=rms_noise,
        signal_level=signal_level,
        saturation_fraction=sat_fraction,
        filter_fallback=False,
    )


def _apply_savgol(mag: np.ndarray, window: int, order: int) -> tuple[np.ndarray, bool]:
    n = len(mag)
    if n < 3:
        return mag.copy(), True
    window = int(max(3, min(window, n)))
    if window % 2 == 0:
        window = max(3, window - 1)
    order = int(max(1, min(order, window - 1)))

    if _SCIPY_AVAILABLE and savgol_filter is not None:
        return np.asarray(
            savgol_filter(mag, window_length=window, polyorder=order), dtype=float
        ), False

    return _moving_average(mag, window), True


def _apply_median(mag: np.ndarray, window: int) -> tuple[np.ndarray, bool]:
    n = len(mag)
    if n < 3:
        return mag.copy(), True
    window = int(max(3, min(window, n)))
    if window % 2 == 0:
        window = max(3, window - 1)

    if _SCIPY_AVAILABLE and medfilt is not None:
        return np.asarray(medfilt(mag, kernel_size=window), dtype=float), False

    return _rolling_median(mag, window), True


def _apply_butterworth(mag: np.ndarray) -> tuple[np.ndarray, bool]:
    if (
        _SCIPY_AVAILABLE
        and butter is not None
        and filtfilt is not None
        and mag.size > 8
    ):
        b, a = butter(N=3, Wn=0.2, btype="low")
        return np.asarray(filtfilt(b, a, mag), dtype=float), False
    return _moving_average(mag, 7), True


def auto_filter(
    magnetization: np.ndarray, noise_stats: NoiseStats, config
) -> np.ndarray:
    """Apply adaptive filtering policy driven by SNR or explicit method."""
    mag = np.asarray(magnetization, dtype=float).reshape(-1)
    method = config.filter_method

    if method is None:
        if not bool(getattr(config, "auto_filter", True)):
            return mag

        snr = float(noise_stats.snr)
        if np.isfinite(snr) and snr > 50:
            return mag
        if np.isfinite(snr) and snr > 20:
            method = "savgol"
        else:
            method = "median_savgol"

    method_norm = str(method).lower()
    fallback_used = False

    if method_norm == "savgol":
        filtered, fallback_used = _apply_savgol(
            mag,
            window=getattr(config, "savgol_window", 11),
            order=getattr(config, "savgol_order", 3),
        )
    elif method_norm == "median":
        filtered, fallback_used = _apply_median(
            mag,
            window=getattr(config, "savgol_window", 11),
        )
    elif method_norm == "butter":
        filtered, fallback_used = _apply_butterworth(mag)
    elif method_norm == "median_savgol":
        stage1, fb1 = _apply_median(mag, window=getattr(config, "savgol_window", 11))
        filtered, fb2 = _apply_savgol(
            stage1,
            window=getattr(config, "savgol_window", 11),
            order=getattr(config, "savgol_order", 3),
        )
        fallback_used = fb1 or fb2
    elif method_norm in {"none", "null", ""}:
        filtered = mag
    else:
        filtered = mag
        fallback_used = False

    noise_stats.filter_fallback = bool(fallback_used)
    return np.asarray(filtered, dtype=float)


def detect_anomalies(
    field: np.ndarray,
    magnetization: np.ndarray,
    n_sigma: float = 3.0,
) -> AnomalyReport:
    """Detect outliers, discontinuities, and loop-closing mismatch."""
    fld = np.asarray(field, dtype=float).reshape(-1)
    mag = np.asarray(magnetization, dtype=float).reshape(-1)
    if mag.size == 0:
        return AnomalyReport([], [], False, 0.0)

    window = max(5, int(mag.size // 40) | 1)
    smooth = _moving_average(mag, window)
    residual = mag - smooth

    sigma = float(np.nanstd(residual))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.nanstd(mag))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1e-12

    outliers = np.where(np.abs(residual) > float(n_sigma) * sigma)[0]

    derivative = np.abs(numerical_derivative(fld, mag))
    if derivative.size:
        thresh = float(np.nanpercentile(derivative, 99))
        if not np.isfinite(thresh) or thresh <= 0:
            thresh = float(np.nanmax(derivative))
        thresh *= 1.5
        discontinuities = np.where(derivative > thresh)[0]
    else:
        discontinuities = np.array([], dtype=int)

    closing_error = float(np.abs(mag[0] - mag[-1])) if mag.size > 1 else 0.0
    non_closing = bool(closing_error > float(n_sigma) * sigma)

    return AnomalyReport(
        outlier_indices=[int(i) for i in outliers.tolist()],
        discontinuity_indices=[int(i) for i in discontinuities.tolist()],
        non_closing_loop=non_closing,
        closing_error=closing_error,
    )
