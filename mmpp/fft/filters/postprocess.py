"""Post-FFT spectrum filter functions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:  # pragma: no cover - optional dependency
    gaussian_filter1d = None

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - optional dependency
    savgol_filter = None

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _per_trace(values: np.ndarray, fn) -> np.ndarray:
    """Apply 1D function trace-wise over axis 0."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return fn(arr)
    flat = arr.reshape(arr.shape[0], -1)
    out = np.zeros_like(flat)
    for idx in range(flat.shape[1]):
        out[:, idx] = fn(flat[:, idx])
    return out.reshape(arr.shape)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    win = max(1, int(window))
    if win <= 1:
        return arr

    kernel = np.ones(win, dtype=float) / float(win)

    def _smooth_1d(trace: np.ndarray) -> np.ndarray:
        pad = win // 2
        padded = np.pad(trace, pad_width=pad, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[: trace.size]

    return _per_trace(arr, _smooth_1d)


def apply_smoothing(
    values: np.ndarray,
    smooth_filter: str = "none",
    smooth_window: int = 7,
    smooth_sigma: float = 1.0,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply smoothing mode to traces."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    mode = str(smooth_filter or "none").lower()
    if mode in {"none", ""}:
        return arr

    if mode in {"moving_average", "moving"}:
        win = _safe_int(smooth_window, 7)
        if win % 2 == 0:
            win += 1
        return moving_average(arr, win)

    if mode in {"gaussian", "gaussian_smooth"}:
        sigma = max(0.0, _safe_float(smooth_sigma, 1.0))
        if sigma == 0.0:
            return arr
        if gaussian_filter1d is None:
            win = max(3, int(round(4 * sigma)) | 1)
            return moving_average(arr, win)
        return gaussian_filter1d(arr, sigma=sigma, axis=0, mode="nearest")

    if mode in {"savgol", "savgol_smooth"}:
        win = max(3, _safe_int(smooth_window, 7))
        if win % 2 == 0:
            win += 1
        if win >= arr.shape[0]:
            win = max(3, arr.shape[0] - (1 - arr.shape[0] % 2))
        if win < 3:
            return arr
        if savgol_filter is None:
            return moving_average(arr, win)
        po = min(max(1, _safe_int(polyorder, 2)), win - 1)
        return savgol_filter(arr, window_length=win, polyorder=po, axis=0, mode="interp")

    return arr


def apply_baseline(
    values: np.ndarray,
    mode: str = "linear",
) -> np.ndarray:
    """Apply baseline correction."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    baseline_mode = str(mode or "none").lower()
    if baseline_mode == "none":
        return arr

    def _baseline_1d(trace: np.ndarray) -> np.ndarray:
        finite = np.isfinite(trace)
        if not np.any(finite):
            return np.zeros_like(trace)

        out = trace.copy()
        base = float(np.nanmedian(out[finite]))

        if baseline_mode == "mean":
            base = float(np.nanmean(out[finite]))
            out = out - base
        elif baseline_mode == "median":
            base = float(np.nanmedian(out[finite]))
            out = out - base
        elif baseline_mode == "linear":
            x = np.arange(out.size, dtype=float)[finite]
            y = out[finite]
            if x.size >= 2:
                try:
                    coeff = np.polyfit(x, y, deg=1)
                    trend = np.polyval(coeff, np.arange(out.size, dtype=float))
                    out = out - trend
                except Exception:
                    out = out - base
            else:
                out = out - base
        else:
            out = out - base

        min_val = float(np.nanmin(out))
        if np.isfinite(min_val) and min_val < 0:
            out = out - min_val
        return out

    return _per_trace(arr, _baseline_1d)


def apply_percentile_clip(
    values: np.ndarray,
    low: float = 0.0,
    high: float = 100.0,
) -> np.ndarray:
    """Clip to percentile interval and re-zero lower bound."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    lo = float(np.clip(min(low, high), 0.0, 100.0))
    hi = float(np.clip(max(low, high), 0.0, 100.0))
    if lo <= 0.0 and hi >= 100.0:
        return arr

    def _clip_1d(trace: np.ndarray) -> np.ndarray:
        finite = np.isfinite(trace)
        if not np.any(finite):
            return np.zeros_like(trace)
        lo_val = float(np.nanpercentile(trace[finite], lo))
        hi_val = float(np.nanpercentile(trace[finite], hi))
        if hi_val < lo_val:
            hi_val = lo_val
        clipped = np.clip(trace, lo_val, hi_val)
        if lo_val != 0.0:
            clipped = clipped - lo_val
        return clipped

    return _per_trace(arr, _clip_1d)


def apply_soft_threshold(values: np.ndarray, percentile: float = 0.0) -> np.ndarray:
    """Soft-threshold low-amplitude components."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    pct = float(np.clip(percentile, 0.0, 100.0))
    if pct <= 0.0:
        return arr

    def _threshold_1d(trace: np.ndarray) -> np.ndarray:
        finite = np.isfinite(trace)
        if not np.any(finite):
            return np.zeros_like(trace)
        threshold = float(np.nanpercentile(trace[finite], pct))
        if not np.isfinite(threshold) or threshold <= 0:
            return trace
        return np.sign(trace) * np.maximum(np.abs(trace) - threshold, 0.0)

    return _per_trace(arr, _threshold_1d)


def apply_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize traces to maximum absolute value."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    def _normalize_1d(trace: np.ndarray) -> np.ndarray:
        max_val = float(np.nanmax(np.abs(trace)))
        if max_val > 0:
            return trace / max_val
        return trace

    return _per_trace(arr, _normalize_1d)


def apply_log_transform(values: np.ndarray) -> np.ndarray:
    """log10 transform with epsilon protection."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    eps = np.finfo(float).eps
    return np.log10(np.clip(np.abs(arr), eps, None))


def apply_gamma(values: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gamma correction."""
    arr = np.asarray(values, dtype=float)
    gamma_val = _safe_float(gamma, 1.0)
    if arr.size == 0 or gamma_val == 1.0:
        return arr
    return np.power(np.abs(arr), gamma_val) * np.sign(arr)


def _coerce_options(option: Any) -> dict[str, Any]:
    if isinstance(option, dict):
        return {str(k): v for k, v in option.items() if k != "enabled"}
    return {}


def apply_postprocess_filters(
    spectrum: np.ndarray,
    frequencies: np.ndarray | None,
    stage_filters: dict[str, Any],
) -> np.ndarray:
    """Apply configured postprocess filters in deterministic order."""
    if not stage_filters:
        return np.asarray(spectrum)

    result = np.array(spectrum, copy=True)
    sequence = [
        ("baseline_correction", apply_baseline),
        ("percentile_clip", apply_percentile_clip),
        ("soft_threshold", apply_soft_threshold),
        ("gaussian_smooth", apply_smoothing),
        ("savgol_smooth", apply_smoothing),
        ("moving_average", apply_smoothing),
        ("normalize", apply_normalize),
        ("log_transform", apply_log_transform),
        ("gamma", apply_gamma),
    ]

    for name, func in sequence:
        if name not in stage_filters:
            continue
        option = stage_filters.get(name)
        opts = _coerce_options(option)
        try:
            if name == "baseline_correction":
                mode = opts.get("mode", option if isinstance(option, str) else "linear")
                result = func(result, mode=mode)
            elif name == "percentile_clip":
                low = _safe_float(opts.get("low", opts.get("clip_percentile_low", 0.0)), 0.0)
                high = _safe_float(opts.get("high", opts.get("clip_percentile_high", 100.0)), 100.0)
                result = func(result, low=low, high=high)
            elif name == "soft_threshold":
                pct = _safe_float(opts.get("percentile", option if isinstance(option, (int, float)) else 0.0), 0.0)
                result = func(result, percentile=pct)
            elif name in {"gaussian_smooth", "savgol_smooth", "moving_average"}:
                mode = name.replace("_smooth", "")
                smooth_mode = opts.get("smooth_filter", mode)
                if not isinstance(smooth_mode, str):
                    smooth_mode = mode
                win = _safe_int(opts.get("window", opts.get("smooth_window", 7)), 7)
                sigma = _safe_float(opts.get("sigma", opts.get("smooth_sigma", 1.0)), 1.0)
                po = _safe_int(opts.get("polyorder", 2), 2)
                result = func(
                    result,
                    smooth_filter=str(smooth_mode),
                    smooth_window=win,
                    smooth_sigma=sigma,
                    polyorder=po,
                )
            elif name == "gamma":
                gamma = _safe_float(opts.get("gamma", option), 1.0)
                result = func(result, gamma=gamma)
            else:
                result = func(result)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Postprocess filter %s failed: %s", name, exc)
    return result

