"""
Spectrum Filter Utilities Module

Provides filter configuration, normalization, and application for FMR spectrum analysis.
Similar to dispersion/utils.py but focused on 1D spectrum processing.
"""

from __future__ import annotations
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Cache schema version - bump when cached results are no longer compatible
SPECTRUM_CACHE_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Filter registry
# ---------------------------------------------------------------------------

# Pre-processing filters applied before FFT or on raw data
PREPROCESS_FILTER_KEYS = {
    "remove_static",
    "remove_average", 
    "hann_time",
    "detrend",
}

# Post-processing filters applied after FFT spectrum computation
POSTPROCESS_FILTER_KEYS = {
    "normalize",
    "log_transform",
    "gamma",
    "percentile_clip",
    "soft_threshold",
    "gaussian_smooth",
    "savgol_smooth",
    "baseline_correction",
}

# Live filters that can be recomputed from cached spectrum without recompute
LIVE_FILTER_KEYS = {
    "normalize",
    "log_transform", 
    "gamma",
    "percentile_clip",
    "soft_threshold",
    "gaussian_smooth",
    "savgol_smooth",
    "baseline_correction",
}


# ---------------------------------------------------------------------------
# Safe value helpers
# ---------------------------------------------------------------------------

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


def _is_enabled(option: Any) -> bool:
    if isinstance(option, dict):
        return bool(option.get("enabled", True))
    return bool(option)


def _coerce_options(option: Any) -> dict[str, Any]:
    if isinstance(option, dict):
        return {str(k): v for k, v in option.items() if k != "enabled"}
    return {}


def _merge_nested_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Filter configuration normalization
# ---------------------------------------------------------------------------

def normalize_filter_config(
    filters: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """
    Normalize user-provided filter config into a stable dict.

    Backward-compatible with legacy boolean flags and supports advanced
    stage-based configuration via ``pre``, ``post`` and ``live`` blocks.
    
    Examples
    --------
    >>> normalize_filter_config({"smooth_filter": "gaussian", "normalize": True})
    {'post': {'gaussian_smooth': True, 'normalize': True}}
    
    >>> normalize_filter_config({"pre": {"remove_static": True}})
    {'pre': {'remove_static': True}}
    """
    if not filters:
        return None

    normalized: dict[str, Any] = {}

    # Legacy aliases/flags from interactive_spectrum
    legacy_mappings = {
        "smooth_filter": ("post", "gaussian_smooth"),
        "baseline_mode": ("post", "baseline_correction"),
        "log_scale": ("post", "log_transform"),
    }
    
    for legacy_key, (stage, target_key) in legacy_mappings.items():
        if legacy_key in filters and filters[legacy_key]:
            value = filters[legacy_key]
            if value not in ("none", "None", False, None):
                normalized.setdefault(stage, {})[target_key] = value

    # Stage blocks
    for stage_name in ("pre", "post", "live"):
        stage_cfg = filters.get(stage_name)
        if isinstance(stage_cfg, dict):
            normalized.setdefault(stage_name, {}).update(
                {str(k): v for k, v in stage_cfg.items()}
            )

    # Root-level technique keys for convenience
    skip_keys = {"pre", "post", "live", "advanced"} | set(legacy_mappings.keys())
    for key, value in filters.items():
        if key in skip_keys:
            continue
        if key in PREPROCESS_FILTER_KEYS:
            if _is_enabled(value):
                normalized.setdefault("pre", {})[key] = value
        elif key in POSTPROCESS_FILTER_KEYS:
            if _is_enabled(value):
                normalized.setdefault("post", {})[key] = value
        elif key in LIVE_FILTER_KEYS:
            if _is_enabled(value):
                normalized.setdefault("live", {})[key] = value
        else:
            # Preserve unknown keys for cache signature
            normalized[key] = value

    return normalized or None


def split_filter_stages(
    filters: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Split filter config into preprocess, postprocess and live-stage blocks.
    
    Returns
    -------
    tuple[dict, dict, dict]
        (pre_filters, post_filters, live_filters)
    """
    cfg = normalize_filter_config(filters)
    if not cfg:
        return {}, {}, {}

    pre: dict[str, Any] = {}
    post: dict[str, Any] = {}
    live: dict[str, Any] = {}

    for stage_name, target in (("pre", pre), ("post", post), ("live", live)):
        stage_cfg = cfg.get(stage_name)
        if isinstance(stage_cfg, dict):
            for name, option in stage_cfg.items():
                if _is_enabled(option):
                    target[name] = option

    return pre, post, live


def classify_filter_execution(
    filters: Optional[dict[str, Any]],
) -> dict[str, list[str]]:
    """
    Classify active filters by execution stage.

    Returns
    -------
    dict[str, list[str]]
        - ``compute_stage``: preprocessing filters requiring recomputation
        - ``post_stage``: post-FFT filters
        - ``live_capable``: subset that can run from cached spectrum
    """
    pre, post, live = split_filter_stages(filters)

    compute_stage = sorted(pre.keys())
    post_stage = sorted(post.keys())
    live_capable = sorted(
        name for name in set(list(post.keys()) + list(live.keys()))
        if name in LIVE_FILTER_KEYS
    )

    return {
        "compute_stage": compute_stage,
        "post_stage": post_stage,
        "live_capable": live_capable,
    }


# ---------------------------------------------------------------------------
# Spectrum filter application
# ---------------------------------------------------------------------------

def apply_spectrum_filters(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    filters: Optional[dict[str, Any]],
    stage: str = "post",
) -> np.ndarray:
    """
    Apply filter pipeline to spectrum data.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Power spectrum array (1D or 2D with components)
    frequencies : np.ndarray
        Frequency axis
    filters : dict
        Filter configuration
    stage : str
        Which stage filters to apply: "pre", "post", or "live"
        
    Returns
    -------
    np.ndarray
        Filtered spectrum
    """
    if filters is None:
        return spectrum
        
    cfg = normalize_filter_config(filters)
    if cfg is None:
        return spectrum
    
    stage_filters = cfg.get(stage, {})
    if not stage_filters:
        return spectrum
    
    result = np.array(spectrum, copy=True)
    
    # Apply filters in order
    filter_sequence = [
        ("baseline_correction", _apply_baseline_correction),
        ("percentile_clip", _apply_percentile_clip),
        ("soft_threshold", _apply_soft_threshold),
        ("gaussian_smooth", _apply_gaussian_smooth),
        ("savgol_smooth", _apply_savgol_smooth),
        ("normalize", _apply_normalize),
        ("log_transform", _apply_log_transform),
        ("gamma", _apply_gamma),
    ]
    
    for filter_name, filter_func in filter_sequence:
        if filter_name in stage_filters:
            options = stage_filters[filter_name]
            opts = _coerce_options(options) if isinstance(options, dict) else {}
            try:
                result = filter_func(result, frequencies, **opts)
            except Exception as exc:
                logger.warning("Filter %s failed: %s", filter_name, exc)
    
    return result


# ---------------------------------------------------------------------------
# Individual filter implementations
# ---------------------------------------------------------------------------

def _apply_baseline_correction(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    mode: str = "linear",
    **kwargs,
) -> np.ndarray:
    """Apply baseline correction."""
    if mode == "none":
        return spectrum
    
    if mode == "mean":
        return spectrum - np.mean(spectrum)
    elif mode == "median":
        return spectrum - np.median(spectrum)
    elif mode == "linear":
        # Linear detrend
        x = np.arange(len(spectrum))
        coeffs = np.polyfit(x, spectrum, 1)
        baseline = np.polyval(coeffs, x)
        return spectrum - baseline
    
    return spectrum


def _apply_percentile_clip(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    low: float = 0.0,
    high: float = 100.0,
    **kwargs,
) -> np.ndarray:
    """Clip spectrum to percentile range."""
    low_val = np.percentile(spectrum, low)
    high_val = np.percentile(spectrum, high)
    return np.clip(spectrum, low_val, high_val)


def _apply_soft_threshold(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    percentile: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """Apply soft thresholding."""
    if percentile <= 0:
        return spectrum
    
    threshold = np.percentile(spectrum, percentile)
    result = spectrum - threshold
    result[result < 0] = 0
    return result


def _apply_gaussian_smooth(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    sigma: float = 1.0,
    window: int = 5,
    **kwargs,
) -> np.ndarray:
    """Apply Gaussian smoothing."""
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(spectrum, sigma=sigma)
    except ImportError:
        # Fallback: moving average
        kernel = np.ones(window) / window
        return np.convolve(spectrum, kernel, mode='same')


def _apply_savgol_smooth(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    **kwargs,
) -> np.ndarray:
    """Apply Savitzky-Golay filter."""
    try:
        from scipy.signal import savgol_filter
        # Ensure window_length is odd and larger than polyorder
        wl = max(polyorder + 2, window_length)
        if wl % 2 == 0:
            wl += 1
        wl = min(wl, len(spectrum) - 1)
        if wl <= polyorder:
            return spectrum
        return savgol_filter(spectrum, wl, polyorder)
    except ImportError:
        return spectrum


def _apply_normalize(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Normalize to maximum value."""
    max_val = np.max(np.abs(spectrum))
    if max_val > 0:
        return spectrum / max_val
    return spectrum


def _apply_log_transform(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Apply log transform."""
    eps = np.finfo(float).eps
    return np.log10(np.abs(spectrum) + eps)


def _apply_gamma(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    gamma: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Apply gamma correction."""
    if gamma == 1.0:
        return spectrum
    return np.power(np.abs(spectrum), gamma) * np.sign(spectrum)
