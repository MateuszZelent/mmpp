"""
Utility functions for spin-wave dispersion analysis.

Contains low-level functions for FFT, k-space operations, windowing, 
and mathematical operations used in dispersion calculations.
"""

from __future__ import annotations
from typing import Tuple, List, Optional, Sequence, Any
import logging
import numpy as np

from ._fft_backend import fft as _fft, fftfreq as _fftfreq, fftshift as _fftshift

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Advanced filter registry and configuration helpers
# ---------------------------------------------------------------------------

LEGACY_PREPROCESS_FILTER_KEYS = {
    "remove_static",
    "remove_average",
    "hann_time",
    "hann_space",
}

PREPROCESS_FILTER_KEYS = {
    "envelope_extraction",
    "wavelet_denoise",
    "wiener_time",
    "median_morph",
    "amplitude_equalization",
    "dynamic_compression",
    "psd_adaptive",
    "ica_denoise",
    "sparse_denoise",
    # Special: handled at temporal FFT stage inside SpinWaveAnalyzer
    "welch_average",
}

POSTPROCESS_FILTER_KEYS = {
    "fk_bandpass",
    "snr_filter",
    "gaussian_morph",
    "wiener2d",
    "wavelet2d",
}

# Filters that can be recomputed "live" from already computed S(k,f)
# without reloading raw M(t, x, y, z).
LIVE_POSTPROCESS_FILTER_KEYS = set(POSTPROCESS_FILTER_KEYS)

SPECIAL_DEFERRED_FILTER_KEYS = {"welch_average"}


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


def normalize_filter_config(
    filters: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """
    Normalize user-provided filter config into a stable dict.

    Backward-compatible with legacy boolean flags and supports advanced
    stage-based configuration via ``pre``, ``post`` and ``live`` blocks.
    """
    if not filters:
        return None

    normalized: dict[str, Any] = {}

    # Legacy aliases/flags
    if bool(filters.get("remove_static")):
        normalized["remove_static"] = True
    if bool(filters.get("remove_average")) or bool(filters.get("average")):
        normalized["remove_average"] = True
    if bool(filters.get("hann_time")):
        normalized["hann_time"] = True
    if bool(filters.get("hann_space")):
        normalized["hann_space"] = True

    # Stage blocks
    for stage_name in ("pre", "post", "live"):
        stage_cfg = filters.get(stage_name)
        if isinstance(stage_cfg, dict):
            normalized[stage_name] = {str(k): v for k, v in stage_cfg.items()}

    # Optional combined block
    advanced_cfg = filters.get("advanced")
    if isinstance(advanced_cfg, dict):
        normalized = _merge_nested_dict(normalized, {str(k): v for k, v in advanced_cfg.items()})

    # Root-level advanced keys for convenience
    skip_keys = {
        "remove_static",
        "remove_average",
        "average",
        "hann_time",
        "hann_space",
        "pre",
        "post",
        "live",
        "advanced",
    }
    for key, value in filters.items():
        if key in skip_keys:
            continue
        if key in PREPROCESS_FILTER_KEYS:
            normalized.setdefault("pre", {})[key] = value
        elif key in POSTPROCESS_FILTER_KEYS:
            normalized.setdefault("post", {})[key] = value
        elif key in LIVE_POSTPROCESS_FILTER_KEYS:
            normalized.setdefault("live", {})[key] = value
        else:
            # Preserve unknown keys for cache signature reproducibility
            normalized[key] = value

    return normalized or None


def split_filter_stages(
    filters: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Split filter config into preprocess, postprocess and live-stage blocks.
    """
    cfg = normalize_filter_config(filters)
    if not cfg:
        return {}, {}, {}

    pre: dict[str, Any] = {}
    post: dict[str, Any] = {}
    live: dict[str, Any] = {}

    # Legacy preprocess flags live at root.
    for key in LEGACY_PREPROCESS_FILTER_KEYS:
        if bool(cfg.get(key)):
            pre[key] = True

    for stage_name, target in (("pre", pre), ("post", post), ("live", live)):
        stage_cfg = cfg.get(stage_name)
        if isinstance(stage_cfg, dict):
            for name, option in stage_cfg.items():
                if _is_enabled(option):
                    target[name] = option

    # Root-level techniques (if present after merges)
    for name, option in cfg.items():
        if name in LEGACY_PREPROCESS_FILTER_KEYS or name in {"pre", "post", "live"}:
            continue
        if name in PREPROCESS_FILTER_KEYS and _is_enabled(option):
            pre[name] = option
        elif name in POSTPROCESS_FILTER_KEYS and _is_enabled(option):
            post[name] = option
        elif name in LIVE_POSTPROCESS_FILTER_KEYS and _is_enabled(option):
            live[name] = option

    return pre, post, live


def classify_filter_execution(
    filters: Optional[dict[str, Any]],
) -> dict[str, list[str]]:
    """
    Classify active filters by execution stage.

    Returns keys:
    - ``compute_stage``: raw-data preprocessing and deferred compute-time filters.
    - ``post_stage``: post-FFT filters applied when computing/plotting.
    - ``live_capable``: subset of post filters that can run from cached S(k,f).
    """
    pre, post, live = split_filter_stages(filters)

    compute_stage = sorted(pre.keys())
    post_stage = sorted(post.keys())
    live_capable = sorted(
        name for name in set(list(post.keys()) + list(live.keys()))
        if name in LIVE_POSTPROCESS_FILTER_KEYS
    )

    return {
        "compute_stage": compute_stage,
        "post_stage": post_stage,
        "live_capable": live_capable,
    }


def _flatten_time_axis(x: np.ndarray, time_axis: int) -> tuple[np.ndarray, tuple[int, ...]]:
    moved = np.moveaxis(np.asarray(x), time_axis, 0)
    return moved.reshape(moved.shape[0], -1), moved.shape


def _restore_time_axis(flat: np.ndarray, moved_shape: tuple[int, ...], time_axis: int) -> np.ndarray:
    moved = flat.reshape(moved_shape)
    return np.moveaxis(moved, 0, time_axis)


def _analytic_envelope_1d(series: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(series):
        return np.abs(series)

    try:
        from scipy.signal import hilbert  # type: ignore

        return np.abs(hilbert(series))
    except Exception:
        # NumPy-only analytic signal fallback
        n = series.size
        if n == 0:
            return np.zeros_like(series, dtype=float)
        spectrum = np.fft.fft(series)
        h = np.zeros(n, dtype=float)
        if n % 2 == 0:
            h[0] = 1.0
            h[n // 2] = 1.0
            h[1:n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(n + 1) // 2] = 2.0
        analytic = np.fft.ifft(spectrum * h)
        return np.abs(analytic)


def adaptive_envelope_extraction(
    x: np.ndarray,
    *,
    time_axis: int = 0,
    threshold_std: float = 2.0,
    margin_samples: int = 10,
    transition_samples: int = 5,
) -> np.ndarray:
    """
    A1: Adaptive packet extraction using Hilbert-envelope thresholding.
    """
    flat, moved_shape = _flatten_time_axis(x, time_axis=time_axis)
    t_len, n_series = flat.shape
    if t_len == 0 or n_series == 0:
        return x

    result = np.array(flat, copy=True)
    margin = max(0, _safe_int(margin_samples, 10))
    transition = max(0, _safe_int(transition_samples, 5))
    threshold_scale = _safe_float(threshold_std, 2.0)
    eps = 1e-20

    for idx in range(n_series):
        series = flat[:, idx]
        envelope = _analytic_envelope_1d(series)
        env_max = float(np.max(envelope))
        if env_max <= eps:
            continue

        norm_env = envelope / env_max
        threshold = float(np.mean(norm_env) + threshold_scale * np.std(norm_env))
        active = np.flatnonzero(norm_env > threshold)
        if active.size == 0:
            continue

        start = max(0, int(active[0]) - margin)
        end = min(t_len, int(active[-1]) + margin + 1)

        weights = np.zeros(t_len, dtype=float)
        weights[start:end] = 1.0

        if transition > 0:
            left_len = min(transition, end - start)
            if left_len > 1:
                weights[start:start + left_len] = np.linspace(0.0, 1.0, left_len)
            right_len = min(transition, end - start)
            if right_len > 1:
                weights[end - right_len:end] = np.linspace(1.0, 0.0, right_len)

        result[:, idx] = series * weights

    return _restore_time_axis(result, moved_shape, time_axis=time_axis)


def wavelet_denoise_time(
    x: np.ndarray,
    *,
    time_axis: int = 0,
    wavelet: str = "db4",
    level: int = 3,
    method: str = "visu",
    scale_decay: float = 0.75,
) -> np.ndarray:
    """
    A2: Wavelet denoising across time axis (optional PyWavelets dependency).
    """
    try:
        import pywt  # type: ignore
    except Exception:
        logger.warning("wavelet_denoise requested but PyWavelets is unavailable; skipping")
        return x

    flat, moved_shape = _flatten_time_axis(x, time_axis=time_axis)
    t_len, n_series = flat.shape
    if t_len < 4 or n_series == 0:
        return x

    level_val = max(1, _safe_int(level, 3))
    decay = float(np.clip(_safe_float(scale_decay, 0.75), 0.1, 1.0))
    method_val = str(method).lower()
    eps = 1e-20

    def denoise_real(signal_1d: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec(signal_1d, wavelet, level=level_val)
        if len(coeffs) <= 1:
            return signal_1d
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        sigma = float(max(sigma, eps))

        if method_val == "visu":
            base_threshold = sigma * np.sqrt(2.0 * np.log(max(signal_1d.size, 2)))
        else:
            base_threshold = sigma * np.sqrt(2.0 * np.log(max(signal_1d.size, 2)))

        coeffs_filtered: list[np.ndarray] = [coeffs[0]]
        n_detail = len(coeffs) - 1
        for i, detail in enumerate(coeffs[1:], start=0):
            scale = decay ** (n_detail - i - 1)
            th = float(base_threshold * scale)
            coeffs_filtered.append(pywt.threshold(detail, th, mode="soft"))

        reconstructed = pywt.waverec(coeffs_filtered, wavelet)
        return reconstructed[: signal_1d.size]

    result = np.empty_like(flat, dtype=flat.dtype)
    for idx in range(n_series):
        series = flat[:, idx]
        if np.iscomplexobj(series):
            real_part = denoise_real(np.real(series))
            imag_part = denoise_real(np.imag(series))
            result[:, idx] = real_part + 1j * imag_part
        else:
            result[:, idx] = denoise_real(series)

    return _restore_time_axis(result, moved_shape, time_axis=time_axis)


def wiener_time_filter(
    x: np.ndarray,
    *,
    time_axis: int = 0,
    window_size: int = 11,
    noise_variance: Optional[float] = None,
) -> np.ndarray:
    """
    A3: Adaptive Wiener filtering along the time axis.
    """
    try:
        from scipy.signal import wiener  # type: ignore
    except Exception:
        logger.warning("wiener_time requested but scipy.signal.wiener is unavailable; skipping")
        return x

    flat, moved_shape = _flatten_time_axis(x, time_axis=time_axis)
    t_len, n_series = flat.shape
    if t_len < 3 or n_series == 0:
        return x

    size = max(3, _safe_int(window_size, 11))
    if size % 2 == 0:
        size += 1

    result = np.empty_like(flat, dtype=flat.dtype)
    for idx in range(n_series):
        series = flat[:, idx]
        if noise_variance is None:
            estimate = float(np.var(np.diff(np.real(series))))
            if np.iscomplexobj(series):
                estimate += float(np.var(np.diff(np.imag(series))))
            local_noise = max(estimate, 1e-20)
        else:
            local_noise = max(float(noise_variance), 1e-20)

        if np.iscomplexobj(series):
            real_f = wiener(np.real(series), mysize=size, noise=local_noise)
            imag_f = wiener(np.imag(series), mysize=size, noise=local_noise)
            result[:, idx] = real_f + 1j * imag_f
        else:
            result[:, idx] = wiener(series, mysize=size, noise=local_noise)

    return _restore_time_axis(result, moved_shape, time_axis=time_axis)


def median_morphological_filter(
    x: np.ndarray,
    *,
    time_axis: int = 0,
    median_size: int = 3,
    morph_size: int = 5,
    threshold_std: float = 1.0,
    apply_closing: bool = True,
) -> np.ndarray:
    """
    A4: Median filtering plus morphological mask cleanup.
    """
    try:
        from scipy.ndimage import median_filter, binary_opening, binary_closing  # type: ignore
    except Exception:
        logger.warning("median_morph requested but scipy.ndimage is unavailable; skipping")
        return x

    size = [1] * x.ndim
    size[time_axis] = max(1, _safe_int(median_size, 3))

    if np.iscomplexobj(x):
        filtered_real = median_filter(np.real(x), size=tuple(size))
        filtered_imag = median_filter(np.imag(x), size=tuple(size))
        filtered = filtered_real + 1j * filtered_imag
    else:
        filtered = median_filter(x, size=tuple(size))

    amp = np.abs(filtered)
    threshold = float(np.mean(amp) + _safe_float(threshold_std, 1.0) * np.std(amp))
    mask = amp > threshold

    structure_shape = [1] * x.ndim
    structure_shape[time_axis] = max(1, _safe_int(morph_size, 5))
    structure = np.ones(structure_shape, dtype=bool)

    cleaned = binary_opening(mask, structure=structure)
    if bool(apply_closing):
        cleaned = binary_closing(cleaned, structure=structure)

    return filtered * cleaned.astype(filtered.dtype)


def amplitude_equalization(
    x: np.ndarray,
    *,
    time_axis: int = 0,
    smoothing_fraction: float = 0.05,
    smoothing_samples: Optional[int] = None,
    epsilon_rel: float = 1e-6,
    max_gain: float = 10.0,
    target: str = "mean",
) -> np.ndarray:
    """
    C1: Adaptive amplitude equalization using a smoothed envelope.
    """
    flat, moved_shape = _flatten_time_axis(x, time_axis=time_axis)
    t_len, n_series = flat.shape
    if t_len < 4 or n_series == 0:
        return x

    if smoothing_samples is not None:
        smooth_n = max(3, _safe_int(smoothing_samples, 7))
    else:
        smooth_n = max(3, int(round(t_len * _safe_float(smoothing_fraction, 0.05))))
    if smooth_n % 2 == 0:
        smooth_n += 1

    try:
        from scipy.ndimage import uniform_filter1d  # type: ignore

        def smooth_env(arr: np.ndarray) -> np.ndarray:
            return uniform_filter1d(arr, size=smooth_n, mode="nearest")
    except Exception:
        kernel = np.ones(smooth_n, dtype=float) / float(smooth_n)

        def smooth_env(arr: np.ndarray) -> np.ndarray:
            return np.convolve(arr, kernel, mode="same")

    max_gain_val = max(1.0, _safe_float(max_gain, 10.0))
    epsilon_scale = max(1e-12, _safe_float(epsilon_rel, 1e-6))
    target_mode = str(target).lower()

    result = np.array(flat, copy=True)
    for idx in range(n_series):
        series = flat[:, idx]
        envelope = _analytic_envelope_1d(series)
        smooth = smooth_env(envelope)
        eps = epsilon_scale * max(float(np.max(smooth)), 1e-12)

        if target_mode == "median":
            target_amp = float(np.median(envelope))
        elif target_mode in {"p95", "percentile95"}:
            target_amp = float(np.percentile(envelope, 95))
        elif target_mode in {"unity", "one"}:
            target_amp = 1.0
        else:
            target_amp = float(np.mean(envelope))

        gain = target_amp / (smooth + eps)
        gain = np.clip(gain, 1.0 / max_gain_val, max_gain_val)
        result[:, idx] = series * gain

    return _restore_time_axis(result, moved_shape, time_axis=time_axis)


def dynamic_range_compression(
    x: np.ndarray,
    *,
    method: str = "log",
    alpha: float = 10.0,
    beta: float = 0.5,
    mu: float = 255.0,
    preserve_scale: bool = True,
) -> np.ndarray:
    """
    C2: Dynamic range compression for real or complex-valued signals.
    """
    data = np.asarray(x)
    mag = np.abs(data)
    eps = 1e-20
    method_val = str(method).lower()

    if method_val == "log":
        alpha_val = max(_safe_float(alpha, 10.0), eps)
        comp_mag = np.log1p(alpha_val * mag) / np.log1p(alpha_val)
    elif method_val == "power":
        beta_val = float(np.clip(_safe_float(beta, 0.5), 0.05, 1.0))
        comp_mag = np.power(mag, beta_val)
    elif method_val == "compander":
        mu_val = max(_safe_float(mu, 255.0), eps)
        comp_mag = np.log1p(mu_val * mag) / np.log1p(mu_val)
    else:
        raise ValueError(f"Unknown dynamic compression method '{method}'")

    if bool(preserve_scale):
        source = float(np.median(mag))
        target_scale = float(np.median(comp_mag))
        if target_scale > eps:
            comp_mag *= source / target_scale

    if np.iscomplexobj(data):
        phase = data / np.maximum(mag, eps)
        return comp_mag * phase

    return np.sign(data) * comp_mag


def psd_adaptive_filter(
    x: np.ndarray,
    *,
    time_axis: int = 0,
    noise_percentile: float = 15.0,
    threshold_factor: float = 3.0,
) -> np.ndarray:
    """
    B1: PSD-based adaptive threshold filter along time axis.
    """
    flat, moved_shape = _flatten_time_axis(x, time_axis=time_axis)
    t_len, n_series = flat.shape
    if t_len < 4 or n_series == 0:
        return x

    pctl = float(np.clip(_safe_float(noise_percentile, 15.0), 0.0, 100.0))
    factor = max(_safe_float(threshold_factor, 3.0), 1.0)

    result = np.empty_like(flat, dtype=flat.dtype)
    for idx in range(n_series):
        series = flat[:, idx]
        spectrum = np.fft.fft(series)
        psd = np.abs(spectrum) ** 2
        noise_floor = float(np.percentile(psd, pctl))
        threshold = factor * max(noise_floor, 1e-20)
        mask = (psd >= threshold).astype(float)
        filtered = np.fft.ifft(spectrum * mask)
        if np.iscomplexobj(series):
            result[:, idx] = filtered
        else:
            result[:, idx] = np.real(filtered)

    return _restore_time_axis(result, moved_shape, time_axis=time_axis)


def ica_denoise_time(
    x: np.ndarray,
    *,
    time_axis: int = 0,
    n_components: Optional[int] = None,
    keep_components: int = 1,
    random_state: int = 42,
) -> np.ndarray:
    """
    F1: ICA-based denoising for multi-channel real signals.
    """
    if np.iscomplexobj(x):
        logger.warning("ica_denoise requires real-valued input; skipping complex signal")
        return x

    try:
        from sklearn.decomposition import FastICA  # type: ignore
    except Exception:
        logger.warning("ica_denoise requested but scikit-learn is unavailable; skipping")
        return x

    flat, moved_shape = _flatten_time_axis(x, time_axis=time_axis)
    t_len, n_series = flat.shape
    if n_series < 2 or t_len < 4:
        return x

    if n_components is None:
        n_comp = min(3, n_series, t_len - 1)
    else:
        n_comp = max(1, min(_safe_int(n_components, 2), n_series, t_len - 1))

    keep_n = max(1, min(_safe_int(keep_components, 1), n_comp))

    try:
        ica = FastICA(n_components=n_comp, random_state=random_state, max_iter=1000, tol=1e-4)
        transformed = ica.fit_transform(flat)
        power = np.var(transformed, axis=0)
        keep_idx = np.argsort(power)[-keep_n:]
        masked = np.zeros_like(transformed)
        masked[:, keep_idx] = transformed[:, keep_idx]
        reconstructed = masked @ ica.mixing_.T + ica.mean_
        return _restore_time_axis(reconstructed.astype(x.dtype, copy=False), moved_shape, time_axis=time_axis)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ica_denoise failed (%s); skipping", exc)
        return x


def sparse_denoise_time(
    x: np.ndarray,
    *,
    time_axis: int = 0,
    threshold_ratio: float = 0.1,
    method: str = "fft",
) -> np.ndarray:
    """
    F2: Sparse hard-threshold denoising in a transform domain.
    """
    flat, moved_shape = _flatten_time_axis(x, time_axis=time_axis)
    t_len, n_series = flat.shape
    if t_len < 4 or n_series == 0:
        return x

    ratio = float(np.clip(_safe_float(threshold_ratio, 0.1), 0.0, 1.0))
    method_val = str(method).lower()
    result = np.empty_like(flat, dtype=flat.dtype)

    for idx in range(n_series):
        series = flat[:, idx]

        if method_val == "fft":
            coeff = np.fft.fft(series)
            th = ratio * float(np.max(np.abs(coeff)))
            coeff_filtered = np.where(np.abs(coeff) >= th, coeff, 0.0)
            reconstructed = np.fft.ifft(coeff_filtered)
            if np.iscomplexobj(series):
                result[:, idx] = reconstructed
            else:
                result[:, idx] = np.real(reconstructed)
        else:
            raise ValueError(f"Unknown sparse denoise method '{method}'")

    return _restore_time_axis(result, moved_shape, time_axis=time_axis)


def compute_welch_power_spectrum(
    signal_k: np.ndarray,
    *,
    axis: int = 0,
    n_segments: int = 4,
    overlap: float = 0.5,
    n_fft: Optional[int] = None,
    apply_hann: bool = True,
) -> np.ndarray:
    """
    E2: Welch/Bartlett-style temporal averaging of FFT power.

    Returns the averaged power spectrum with frequency axis retained on ``axis``.
    """
    if signal_k.shape[axis] < 4:
        spec = _fft(signal_k, axis=axis)
        spec = _fftshift(spec, axes=axis)
        return np.abs(spec) ** 2

    moved = np.moveaxis(signal_k, axis, 0)
    t_len = moved.shape[0]
    n_fft_eff = t_len if n_fft is None else max(4, _safe_int(n_fft, t_len))

    n_seg = max(1, _safe_int(n_segments, 4))
    overlap_val = float(np.clip(_safe_float(overlap, 0.5), 0.0, 0.95))
    seg_len_default = max(4, t_len // n_seg)
    seg_len = min(t_len, seg_len_default)
    step = max(1, int(round(seg_len * (1.0 - overlap_val))))

    starts = list(range(0, max(t_len - seg_len + 1, 1), step))
    if not starts:
        starts = [0]
    if starts[-1] != max(t_len - seg_len, 0):
        starts.append(max(t_len - seg_len, 0))

    acc = None
    count = 0
    for start in starts:
        end = start + seg_len
        seg = moved[start:end]
        if seg.shape[0] == 0:
            continue
        seg_work = np.array(seg, copy=True)
        if apply_hann and seg_work.shape[0] > 1:
            window = hann_window(seg_work.shape[0]).reshape((-1,) + (1,) * (seg_work.ndim - 1))
            seg_work = seg_work * window

        if seg_work.shape[0] < n_fft_eff:
            pad_shape = (n_fft_eff - seg_work.shape[0],) + seg_work.shape[1:]
            seg_work = np.concatenate([seg_work, np.zeros(pad_shape, dtype=seg_work.dtype)], axis=0)
        elif seg_work.shape[0] > n_fft_eff:
            seg_work = seg_work[:n_fft_eff]

        spectrum = _fft(seg_work, axis=0)
        spectrum = _fftshift(spectrum, axes=0)
        power = np.abs(spectrum) ** 2
        if acc is None:
            acc = power
        else:
            acc = acc + power
        count += 1

    if acc is None or count <= 0:
        spec = _fft(signal_k, axis=axis)
        spec = _fftshift(spec, axes=axis)
        return np.abs(spec) ** 2

    averaged = acc / float(count)
    return np.moveaxis(averaged, 0, axis)


def fftfreq_axis(n: int, d: float, shift: bool = True) -> np.ndarray:
    """
    Frequency axis (Hz) for FFT length n and sample spacing d.
    
    Parameters
    ----------
    n : int
        FFT length
    d : float  
        Sample spacing (time step) [s]
    shift : bool
        If True, returns fftshifted (centered) axis
        
    Returns
    -------
    np.ndarray
        Frequency axis [Hz]
    """
    f = _fftfreq(n, d)
    return _fftshift(f) if shift else f


def k_axis_from_grid(n: int, d: float, shift: bool = True) -> np.ndarray:
    """
    Wavevector axis k (rad/m) for FFT length n and grid spacing d [m].
    
    Parameters
    ----------
    n : int
        FFT length (number of grid points)
    d : float
        Grid spacing [m]  
    shift : bool
        If True, returns fftshifted (centered) axis
        
    Returns
    -------
    np.ndarray
        Wavevector axis k [rad/m], range approximately [-π/d, π/d)
    """
    k = 2.0 * np.pi * _fftfreq(n, d)
    return _fftshift(k) if shift else k


def fold_k_to_bz(k: np.ndarray, a: float) -> np.ndarray:
    """
    Fold wavevector(s) k [rad/m] to first Brillouin zone (-π/a, π/a].
    
    Parameters
    ----------
    k : np.ndarray
        Wavevector(s) [rad/m]
    a : float
        Real-space period [m] defining BZ size
        
    Returns
    -------
    np.ndarray
        Folded wavevectors in first BZ
    """
    width = 2.0 * np.pi / a
    # Map to (-π/a, π/a]
    k_fold = (k + np.pi / a) % width - np.pi / a
    return k_fold


def fold_spectrum_1d(
    Skf: np.ndarray, 
    k: np.ndarray, 
    a: float, 
    agg: str = "sum"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fold a 1D dispersion S(k,f) into first BZ defined by period a.
    
    Parameters
    ----------
    Skf : np.ndarray
        Dispersion spectrum with shape (Nk, Nf)
    k : np.ndarray  
        Wavevector axis (Nk,) [rad/m]
    a : float
        Real-space period [m]
    agg : {'sum', 'max'}
        Aggregation method for aliased k bins
        
    Returns
    -------
    k_fold_sorted : np.ndarray
        Unique folded k values, sorted
    Skf_folded : np.ndarray  
        Folded spectrum (Nk_fold, Nf)
    """
    k_fold = fold_k_to_bz(k, a)
    
    # Group by unique folded k values (with tolerance)
    dk = np.median(np.diff(np.sort(k))) if len(k) > 1 else 1.0
    tol = dk * 0.25
    
    # Sort by folded k
    order = np.argsort(k_fold)
    kf_sorted = k_fold[order]
    Skf_sorted = Skf[order]
    
    # Build index groups for identical k values
    groups: List[np.ndarray] = []
    current = [0]
    for i in range(1, len(kf_sorted)):
        if abs(kf_sorted[i] - kf_sorted[current[-1]]) <= tol:
            current.append(i)
        else:
            groups.append(np.array(current, dtype=int))
            current = [i]
    groups.append(np.array(current, dtype=int))

    # Aggregate each group
    k_fold_unique = np.array([np.mean(kf_sorted[g]) for g in groups])
    
    if agg == "max":
        Skf_folded = np.stack([np.nanmax(Skf_sorted[g], axis=0) for g in groups], axis=0)
    else:  # sum
        Skf_folded = np.stack([np.nansum(Skf_sorted[g], axis=0) for g in groups], axis=0)

    # Sort by k
    srt = np.argsort(k_fold_unique)
    return k_fold_unique[srt], Skf_folded[srt, :]


def hann_window(n: int) -> np.ndarray:
    """
    Hann window (periodic) of length n.

    Parameters
    ----------
    n : int
        Window length

    Returns
    -------
    np.ndarray
        Hann window values
    """
    if n <= 1:
        return np.ones(n, dtype=float)
    idx = np.arange(n, dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * idx / (n - 1))


def apply_window_1d(
    x: np.ndarray, 
    axis: int, 
    window: Optional[str]
) -> np.ndarray:
    """
    Apply window function along specified axis.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    axis : int
        Axis along which to apply window
    window : Optional[str]
        Window type: 'hann' or None
        
    Returns
    -------
    np.ndarray
        Windowed array
    """
    if window is None:
        return x
        
    n = x.shape[axis]
    if window == "hann":
        w = hann_window(n)
    else:
        raise ValueError(f"Unknown window '{window}'")
        
    # Reshape for broadcasting
    shape = [1] * x.ndim
    shape[axis] = n
    return x * w.reshape(shape)



def apply_filter_pipeline(
    x: np.ndarray,
    filters: Optional[dict[str, Any]],
    *,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (2, 3),
    dt: Optional[float] = None,
) -> np.ndarray:
    """
    Apply raw-data preprocessing filters (compute stage).

    This function supports both legacy boolean flags and advanced technique
    dictionaries. Deferred filters such as ``welch_average`` are recognized
    for bookkeeping but are applied later in the FFT stage.
    """
    pre_filters, _, _ = split_filter_stages(filters)
    if not pre_filters:
        return x

    result = x
    copied = False

    def ensure_copy() -> None:
        nonlocal result, copied
        if not copied:
            result = np.array(result, copy=True)
            copied = True

    applied: list[str] = []

    if bool(pre_filters.get("remove_static")):
        ensure_copy()
        first = np.take(result, indices=0, axis=time_axis)
        expanded = np.expand_dims(first, axis=time_axis)
        result -= expanded
        applied.append("remove_static")

    if bool(pre_filters.get("remove_average")):
        ensure_copy()
        mean = np.mean(result, axis=time_axis, keepdims=True)
        result -= mean
        applied.append("remove_average")

    if bool(pre_filters.get("hann_time")):
        ensure_copy()
        result = apply_window_1d(result, axis=time_axis, window="hann")
        applied.append("hann_time")

    if bool(pre_filters.get("hann_space")) and spatial_axes:
        ensure_copy()
        ndims = result.ndim
        for axis in spatial_axes:
            if 0 <= axis < ndims and result.shape[axis] > 1:
                result = apply_window_1d(result, axis=axis, window="hann")
        applied.append("hann_space")

    if "envelope_extraction" in pre_filters:
        ensure_copy()
        options = _coerce_options(pre_filters["envelope_extraction"])
        result = adaptive_envelope_extraction(
            result,
            time_axis=time_axis,
            threshold_std=_safe_float(options.get("threshold_std"), 2.0),
            margin_samples=_safe_int(options.get("margin_samples"), 10),
            transition_samples=_safe_int(options.get("transition_samples"), 5),
        )
        applied.append("envelope_extraction")

    if "wavelet_denoise" in pre_filters:
        ensure_copy()
        options = _coerce_options(pre_filters["wavelet_denoise"])
        result = wavelet_denoise_time(
            result,
            time_axis=time_axis,
            wavelet=str(options.get("wavelet", "db4")),
            level=_safe_int(options.get("level"), 3),
            method=str(options.get("method", "visu")),
            scale_decay=_safe_float(options.get("scale_decay"), 0.75),
        )
        applied.append("wavelet_denoise")

    if "wiener_time" in pre_filters:
        ensure_copy()
        options = _coerce_options(pre_filters["wiener_time"])
        result = wiener_time_filter(
            result,
            time_axis=time_axis,
            window_size=_safe_int(options.get("window_size"), 11),
            noise_variance=options.get("noise_variance"),
        )
        applied.append("wiener_time")

    if "median_morph" in pre_filters:
        ensure_copy()
        options = _coerce_options(pre_filters["median_morph"])
        result = median_morphological_filter(
            result,
            time_axis=time_axis,
            median_size=_safe_int(options.get("median_size"), 3),
            morph_size=_safe_int(options.get("morph_size"), 5),
            threshold_std=_safe_float(options.get("threshold_std"), 1.0),
            apply_closing=bool(options.get("apply_closing", True)),
        )
        applied.append("median_morph")

    if "amplitude_equalization" in pre_filters:
        ensure_copy()
        options = _coerce_options(pre_filters["amplitude_equalization"])
        result = amplitude_equalization(
            result,
            time_axis=time_axis,
            smoothing_fraction=_safe_float(options.get("smoothing_fraction"), 0.05),
            smoothing_samples=options.get("smoothing_samples"),
            epsilon_rel=_safe_float(options.get("epsilon_rel"), 1e-6),
            max_gain=_safe_float(options.get("max_gain"), 10.0),
            target=str(options.get("target", "mean")),
        )
        applied.append("amplitude_equalization")

    if "dynamic_compression" in pre_filters:
        ensure_copy()
        options = _coerce_options(pre_filters["dynamic_compression"])
        result = dynamic_range_compression(
            result,
            method=str(options.get("method", "log")),
            alpha=_safe_float(options.get("alpha"), 10.0),
            beta=_safe_float(options.get("beta"), 0.5),
            mu=_safe_float(options.get("mu"), 255.0),
            preserve_scale=bool(options.get("preserve_scale", True)),
        )
        applied.append("dynamic_compression")

    if "psd_adaptive" in pre_filters:
        ensure_copy()
        options = _coerce_options(pre_filters["psd_adaptive"])
        result = psd_adaptive_filter(
            result,
            time_axis=time_axis,
            noise_percentile=_safe_float(options.get("noise_percentile"), 15.0),
            threshold_factor=_safe_float(options.get("threshold_factor"), 3.0),
        )
        applied.append("psd_adaptive")

    if "ica_denoise" in pre_filters:
        ensure_copy()
        options = _coerce_options(pre_filters["ica_denoise"])
        result = ica_denoise_time(
            result,
            time_axis=time_axis,
            n_components=options.get("n_components"),
            keep_components=_safe_int(options.get("keep_components"), 1),
            random_state=_safe_int(options.get("random_state"), 42),
        )
        applied.append("ica_denoise")

    if "sparse_denoise" in pre_filters:
        ensure_copy()
        options = _coerce_options(pre_filters["sparse_denoise"])
        result = sparse_denoise_time(
            result,
            time_axis=time_axis,
            threshold_ratio=_safe_float(options.get("threshold_ratio"), 0.1),
            method=str(options.get("method", "fft")),
        )
        applied.append("sparse_denoise")

    if "welch_average" in pre_filters:
        # Deferred by design: applied to temporal FFT power in compute stage.
        options = _coerce_options(pre_filters["welch_average"])
        n_seg = _safe_int(options.get("n_segments"), 4)
        ov = _safe_float(options.get("overlap"), 0.5)
        applied.append(f"welch_average(deferred n_segments={n_seg}, overlap={ov:.2f})")

    if applied:
        logger.info("Dispersion filters applied: %s", ", ".join(applied))

    return result


# ===========================================================================
# NON-DESTRUCTIVE IMAGE-LIKE ENHANCEMENT FILTERS (Live Filters)
# ===========================================================================
# These filters enhance visibility without destroying data - they transform
# how data is displayed rather than zeroing out regions.


def log_transform_dispersion(
    S_fk: np.ndarray,
    *,
    method: str = "log1p",
    scale: float = 1.0,
    floor_percentile: float = 1.0,
) -> np.ndarray:
    """
    Dynamic range compression via logarithmic transforms.
    
    Parameters
    ----------
    S_fk : np.ndarray
        Power spectrum (Nk, Nf).
    method : str
        Transform type: 'log1p', 'log10', 'arcsinh', 'sqrt'.
    scale : float
        Scaling factor applied before transform.
    floor_percentile : float
        Percentile for floor value (avoids log(0)).
        
    Returns
    -------
    np.ndarray
        Transformed spectrum with compressed dynamic range.
    """
    data = np.abs(S_fk).astype(np.float64)
    
    if method == "log1p":
        return np.log1p(data * scale)
    elif method == "log10":
        # Use percentile floor to avoid log(0)
        positive = data[data > 0]
        if positive.size > 0:
            floor = np.percentile(positive, max(0.0, min(100.0, floor_percentile)))
        else:
            floor = 1e-20
        return np.log10(np.maximum(data, floor) * scale)
    elif method == "arcsinh":
        return np.arcsinh(data * scale)
    elif method == "sqrt":
        return np.sqrt(data * scale)
    else:
        return data


def gamma_correction_dispersion(
    S_fk: np.ndarray,
    *,
    gamma: float = 0.5,
) -> np.ndarray:
    """
    Power-law (gamma) correction for dispersion enhancement.
    
    Parameters
    ----------
    S_fk : np.ndarray
        Power spectrum (Nk, Nf).
    gamma : float
        Gamma exponent:
        - gamma < 1: Reveals weak signals (expands low values)
        - gamma = 1: No change
        - gamma > 1: Emphasizes peaks (compresses low values)
        Typical values: 0.3-0.5 for revealing weak branches.
        
    Returns
    -------
    np.ndarray
        Gamma-corrected spectrum normalized to [0, 1].
    """
    data = np.abs(S_fk).astype(np.float64)
    dmin, dmax = data.min(), data.max()
    
    if dmax - dmin < 1e-20:
        return np.zeros_like(data)
    
    # Normalize to [0, 1]
    normalized = (data - dmin) / (dmax - dmin)
    
    # Apply gamma
    return np.power(normalized, max(0.01, gamma))


def clahe_dispersion(
    S_fk: np.ndarray,
    *,
    clip_limit: float = 0.03,
    tile_size: int = 16,
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Enhances local contrast without over-amplifying noise.
    
    Parameters
    ----------
    S_fk : np.ndarray
        Power spectrum (Nk, Nf).
    clip_limit : float
        Clipping limit (0.01-0.1). Lower = more natural appearance.
    tile_size : int
        Size of local regions (8-32). Smaller = more local enhancement.
        
    Returns
    -------
    np.ndarray
        CLAHE-enhanced spectrum in [0, 1].
    """
    try:
        from skimage.exposure import equalize_adapthist
    except ImportError:
        logger.warning(
            "clahe requires scikit-image; install with: pip install scikit-image"
        )
        # Fallback: simple histogram stretching
        data = np.abs(S_fk).astype(np.float64)
        dmin, dmax = data.min(), data.max()
        if dmax - dmin < 1e-20:
            return np.zeros_like(data)
        return (data - dmin) / (dmax - dmin)
    
    data = np.abs(S_fk).astype(np.float64)
    dmin, dmax = data.min(), data.max()
    
    if dmax - dmin < 1e-20:
        return np.zeros_like(data)
    
    # Normalize to [0, 1] for CLAHE
    normalized = (data - dmin) / (dmax - dmin)
    
    # Apply CLAHE
    kernel_size = max(4, min(tile_size, min(data.shape) // 2))
    return equalize_adapthist(
        normalized,
        kernel_size=kernel_size,
        clip_limit=max(0.001, min(1.0, clip_limit)),
        nbins=256,
    )


def local_contrast_normalization(
    S_fk: np.ndarray,
    *,
    sigma: float = 10.0,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    Local Contrast Normalization (LCN).
    
    Normalizes intensity relative to local neighborhood statistics.
    Smoother than CLAHE, preserves gradients better.
    
    Parameters
    ----------
    S_fk : np.ndarray
        Power spectrum (Nk, Nf).
    sigma : float
        Gaussian kernel sigma (5-15 for detailed, 20-50 for smooth).
    epsilon : float
        Small constant to avoid division by zero.
        
    Returns
    -------
    np.ndarray
        Locally normalized spectrum.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        logger.warning("local_contrast requires scipy.ndimage")
        return np.abs(S_fk)
    
    data = np.abs(S_fk).astype(np.float64)
    sigma_val = max(1.0, sigma)
    
    local_mean = gaussian_filter(data, sigma=sigma_val)
    local_var = gaussian_filter((data - local_mean) ** 2, sigma=sigma_val)
    local_std = np.sqrt(local_var + epsilon)
    
    normalized = (data - local_mean) / local_std
    
    # Rescale to [0, 1] for display consistency
    nmin, nmax = normalized.min(), normalized.max()
    if nmax - nmin < 1e-20:
        return np.zeros_like(data)
    return (normalized - nmin) / (nmax - nmin)


def unsharp_mask_dispersion(
    S_fk: np.ndarray,
    *,
    sigma: float = 2.0,
    alpha: float = 1.5,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Unsharp masking for edge/branch enhancement.
    
    Formula: enhanced = original + alpha * (original - blurred)
    
    Parameters
    ----------
    S_fk : np.ndarray
        Power spectrum (Nk, Nf).
    sigma : float
        Gaussian blur sigma (1-3 for fine detail, 5-10 for larger features).
    alpha : float
        Sharpening strength (0.5-1.0 subtle, 1.5-3.0 strong).
    threshold : float
        Ignore edges below this (reduces noise amplification).
        
    Returns
    -------
    np.ndarray
        Sharpened spectrum.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        logger.warning("unsharp_mask requires scipy.ndimage")
        return np.abs(S_fk)
    
    data = np.abs(S_fk).astype(np.float64)
    sigma_val = max(0.5, sigma)
    alpha_val = max(0.0, alpha)
    
    blurred = gaussian_filter(data, sigma=sigma_val)
    detail = data - blurred
    
    if threshold > 0:
        # Only sharpen significant edges
        detail = np.where(np.abs(detail) > threshold, detail, 0.0)
    
    enhanced = data + alpha_val * detail
    
    # Clip to valid range
    return np.clip(enhanced, 0.0, None)


def percentile_autoscale(
    S_fk: np.ndarray,
    *,
    low_percentile: float = 2.0,
    high_percentile: float = 99.0,
) -> tuple[np.ndarray, float, float]:
    """
    Percentile-based autoscaling to handle outliers.
    
    Parameters
    ----------
    S_fk : np.ndarray
        Power spectrum (Nk, Nf).
    low_percentile : float
        Lower percentile (0-50). Default 2.0.
    high_percentile : float
        Upper percentile (50-100). Default 99.0.
        
    Returns
    -------
    tuple[np.ndarray, float, float]
        (clipped_data, vmin, vmax) - data clipped to percentile range.
    """
    data = np.abs(S_fk).astype(np.float64)
    valid = data[~np.isnan(data)]
    
    if valid.size == 0:
        return data, 0.0, 1.0
    
    low_pct = max(0.0, min(50.0, low_percentile))
    high_pct = max(50.0, min(100.0, high_percentile))
    
    vmin = float(np.percentile(valid, low_pct))
    vmax = float(np.percentile(valid, high_pct))
    
    if vmax - vmin < 1e-20:
        vmax = vmin + 1.0
    
    clipped = np.clip(data, vmin, vmax)
    return clipped, vmin, vmax


def soft_threshold_dispersion(
    S_fk: np.ndarray,
    *,
    threshold_percentile: float = 50.0,
    smoothness: float = 5.0,
) -> np.ndarray:
    """
    Soft thresholding using sigmoid - non-destructive noise suppression.
    
    Unlike hard thresholding (which zeros out values), this uses a smooth
    sigmoid transition that preserves all data while suppressing noise.
    
    Parameters
    ----------
    S_fk : np.ndarray
        Power spectrum (Nk, Nf).
    threshold_percentile : float
        Percentile for threshold center (0-100).
    smoothness : float
        Sigmoid steepness (1-10). Higher = sharper transition.
        
    Returns
    -------
    np.ndarray
        Soft-thresholded spectrum.
    """
    data = np.abs(S_fk).astype(np.float64)
    
    threshold = float(np.percentile(data, max(0.0, min(100.0, threshold_percentile))))
    smoothness_val = max(0.1, smoothness)
    
    # Normalize data for sigmoid
    dmax = data.max()
    if dmax < 1e-20:
        return data
    
    data_norm = data / dmax
    threshold_norm = threshold / dmax
    
    # Sigmoid soft mask: 1 / (1 + exp(-smoothness * (x - threshold)))
    sigmoid = 1.0 / (1.0 + np.exp(-smoothness_val * (data_norm - threshold_norm) * 10))
    
    return data * sigmoid


# ===========================================================================
# ORIGINAL DESTRUCTIVE FILTERS (kept for backward compatibility)
# ===========================================================================


def fk_bandpass_filter(
    S_fk: np.ndarray,
    k_axis: np.ndarray,
    f_axis: np.ndarray,
    *,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
) -> np.ndarray:
    """B2: Band-pass filtering in (k, f) space."""
    mask = np.ones_like(S_fk, dtype=bool)

    if f_min is not None:
        mask &= (f_axis[np.newaxis, :] >= float(f_min))
    if f_max is not None:
        mask &= (f_axis[np.newaxis, :] <= float(f_max))
    if k_min is not None:
        mask &= (k_axis[:, np.newaxis] >= float(k_min))
    if k_max is not None:
        mask &= (k_axis[:, np.newaxis] <= float(k_max))

    return np.where(mask, S_fk, 0.0)


def snr_based_filter_dispersion(
    S_fk: np.ndarray,
    *,
    threshold_snr: float = 3.0,
    method: str = "percentile",
    noise_percentile: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """D3: SNR-based dispersion filtering."""
    abs_s = np.abs(S_fk)
    method_val = str(method).lower()
    eps = 1e-20

    if method_val == "mad":
        median_val = float(np.median(abs_s))
        noise_level = 1.4826 * float(np.median(np.abs(abs_s - median_val)))
    else:
        pctl = float(np.clip(_safe_float(noise_percentile, 5.0), 0.0, 100.0))
        noise_level = float(np.percentile(abs_s, pctl))

    noise_level = max(noise_level, eps)
    snr = abs_s / noise_level
    threshold = max(_safe_float(threshold_snr, 3.0), 0.0)
    soft = np.maximum(snr - threshold, 0.0) / np.maximum(snr, eps)
    filtered = abs_s * soft

    return filtered, snr


def enhance_dispersion_2d(
    S_fk: np.ndarray,
    *,
    sigma_f: float = 1.0,
    sigma_k: float = 1.0,
    threshold_std: float = 1.5,
    opening_size: int = 3,
    apply_closing: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D1: 2D Gaussian smoothing + adaptive threshold + morphology."""
    try:
        from scipy.ndimage import gaussian_filter, binary_opening, binary_closing  # type: ignore
    except Exception:
        logger.warning("gaussian_morph requested but scipy.ndimage is unavailable; returning unmodified spectrum")
        abs_s = np.abs(S_fk)
        mask = np.ones_like(abs_s, dtype=bool)
        return abs_s, abs_s, mask

    abs_s = np.abs(S_fk)
    log_s = np.log10(abs_s + 1e-20)
    smooth = gaussian_filter(log_s, sigma=(max(sigma_k, 0.0), max(sigma_f, 0.0)))
    threshold = float(np.mean(smooth) + _safe_float(threshold_std, 1.5) * np.std(smooth))
    mask = smooth > threshold

    open_n = max(1, _safe_int(opening_size, 3))
    structure = np.ones((open_n, open_n), dtype=bool)
    cleaned = binary_opening(mask, structure=structure)
    if bool(apply_closing):
        cleaned = binary_closing(cleaned, structure=structure)

    enhanced = np.where(cleaned, abs_s, 0.0)
    return enhanced, smooth, cleaned


def wiener_filter_2d(
    S_fk: np.ndarray,
    *,
    window_size: int = 5,
    noise_var: Optional[float] = None,
) -> np.ndarray:
    """D2: 2D Wiener-like adaptive filtering."""
    try:
        from scipy.ndimage import uniform_filter  # type: ignore
    except Exception:
        logger.warning("wiener2d requested but scipy.ndimage is unavailable; skipping")
        return np.abs(S_fk)

    abs_s = np.abs(S_fk)
    size = max(1, _safe_int(window_size, 5))
    local_mean = uniform_filter(abs_s, size=size)
    local_sq_mean = uniform_filter(abs_s ** 2, size=size)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    if noise_var is None:
        edge_k = max(2, min(10, abs_s.shape[0] // 8))
        edge_f = max(2, min(10, abs_s.shape[1] // 8))
        corner = abs_s[:edge_k, :edge_f]
        if corner.size == 0:
            noise = float(np.var(abs_s))
        else:
            noise = float(np.var(corner))
    else:
        noise = max(float(noise_var), 1e-20)

    gain = np.maximum(local_var - noise, 0.0) / np.maximum(local_var, 1e-20)
    return local_mean + gain * (abs_s - local_mean)


def wavelet_denoise_2d_dispersion(
    S_fk: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int = 2,
    threshold_scale: float = 1.0,
) -> np.ndarray:
    """E1: 2D wavelet denoising for dispersion maps."""
    try:
        import pywt  # type: ignore
    except Exception:
        logger.warning("wavelet2d requested but PyWavelets is unavailable; skipping")
        return np.abs(S_fk)

    data = np.abs(S_fk)
    level_val = max(1, _safe_int(level, 2))
    coeffs = pywt.wavedec2(data, wavelet, level=level_val)
    if len(coeffs) <= 1:
        return data

    sigma = np.median(np.abs(coeffs[-1][2])) / 0.6745
    sigma = float(max(sigma, 1e-20))
    threshold = (
        _safe_float(threshold_scale, 1.0)
        * sigma
        * np.sqrt(2.0 * np.log(max(data.size, 2)))
    )

    filtered_coeffs: list[Any] = [coeffs[0]]
    for detail in coeffs[1:]:
        c_h, c_v, c_d = detail
        filtered_coeffs.append(
            (
                pywt.threshold(c_h, threshold, mode="soft"),
                pywt.threshold(c_v, threshold, mode="soft"),
                pywt.threshold(c_d, threshold, mode="soft"),
            )
        )

    rec = pywt.waverec2(filtered_coeffs, wavelet)
    return rec[: data.shape[0], : data.shape[1]]


def apply_dispersion_post_filters(
    S_fk: np.ndarray,
    *,
    k_axis: np.ndarray,
    f_axis: np.ndarray,
    filters: Optional[dict[str, Any]],
    include_live: bool = True,
) -> np.ndarray:
    """
    Apply post-FFT filters to a dispersion map S(k,f).

    Parameters
    ----------
    S_fk : np.ndarray
        Spectrum shaped (Nk, Nf).
    k_axis, f_axis : np.ndarray
        Axis arrays corresponding to ``S_fk``.
    filters : dict | None
        Full filter configuration.
    include_live : bool
        When True, merges ``live`` filters into the post-processing pass.
    """
    _, post_filters, live_filters = split_filter_stages(filters)
    chain: dict[str, Any] = dict(post_filters)
    if include_live:
        chain.update(live_filters)

    if not chain:
        return S_fk

    result = np.array(S_fk, dtype=float, copy=True)
    applied: list[str] = []

    if "fk_bandpass" in chain:
        options = _coerce_options(chain["fk_bandpass"])
        result = fk_bandpass_filter(
            result,
            k_axis=k_axis,
            f_axis=f_axis,
            f_min=options.get("f_min"),
            f_max=options.get("f_max"),
            k_min=options.get("k_min"),
            k_max=options.get("k_max"),
        )
        applied.append("fk_bandpass")

    if "snr_filter" in chain:
        options = _coerce_options(chain["snr_filter"])
        result, _ = snr_based_filter_dispersion(
            result,
            threshold_snr=_safe_float(options.get("threshold_snr"), 3.0),
            method=str(options.get("method", "percentile")),
            noise_percentile=_safe_float(options.get("noise_percentile"), 5.0),
        )
        applied.append("snr_filter")

    if "wiener2d" in chain:
        options = _coerce_options(chain["wiener2d"])
        result = wiener_filter_2d(
            result,
            window_size=_safe_int(options.get("window_size"), 5),
            noise_var=options.get("noise_var"),
        )
        applied.append("wiener2d")

    if "wavelet2d" in chain:
        options = _coerce_options(chain["wavelet2d"])
        result = wavelet_denoise_2d_dispersion(
            result,
            wavelet=str(options.get("wavelet", "db4")),
            level=_safe_int(options.get("level"), 2),
            threshold_scale=_safe_float(options.get("threshold_scale"), 1.0),
        )
        applied.append("wavelet2d")

    if "gaussian_morph" in chain:
        options = _coerce_options(chain["gaussian_morph"])
        enhanced, _, _ = enhance_dispersion_2d(
            result,
            sigma_f=_safe_float(options.get("sigma_f"), 1.0),
            sigma_k=_safe_float(options.get("sigma_k"), 1.0),
            threshold_std=_safe_float(options.get("threshold_std"), 1.5),
            opening_size=_safe_int(options.get("opening_size"), 3),
            apply_closing=bool(options.get("apply_closing", False)),
        )
        result = enhanced
        applied.append("gaussian_morph")

    # =========================================================================
    # NON-DESTRUCTIVE ENHANCEMENT FILTERS (applied after traditional filters)
    # =========================================================================

    # Soft threshold - non-destructive noise suppression
    if "soft_threshold" in chain:
        options = _coerce_options(chain["soft_threshold"])
        result = soft_threshold_dispersion(
            result,
            threshold_percentile=_safe_float(options.get("threshold_percentile"), 50.0),
            smoothness=_safe_float(options.get("smoothness"), 5.0),
        )
        applied.append("soft_threshold")

    # Percentile autoscale - clip to robust range
    if "percentile_autoscale" in chain:
        options = _coerce_options(chain["percentile_autoscale"])
        result, _, _ = percentile_autoscale(
            result,
            low_percentile=_safe_float(options.get("low_percentile"), 2.0),
            high_percentile=_safe_float(options.get("high_percentile"), 99.0),
        )
        applied.append("percentile_autoscale")

    # Log transform - dynamic range compression
    if "log_transform" in chain:
        options = _coerce_options(chain["log_transform"])
        result = log_transform_dispersion(
            result,
            method=str(options.get("method", "log1p")),
            scale=_safe_float(options.get("scale"), 1.0),
            floor_percentile=_safe_float(options.get("floor_percentile"), 1.0),
        )
        applied.append("log_transform")

    # Gamma correction - reveal weak signals
    if "gamma" in chain:
        options = _coerce_options(chain["gamma"])
        result = gamma_correction_dispersion(
            result,
            gamma=_safe_float(options.get("gamma"), 0.5),
        )
        applied.append("gamma")

    # Local contrast normalization
    if "local_contrast" in chain:
        options = _coerce_options(chain["local_contrast"])
        result = local_contrast_normalization(
            result,
            sigma=_safe_float(options.get("sigma"), 10.0),
            epsilon=_safe_float(options.get("epsilon"), 1e-5),
        )
        applied.append("local_contrast")

    # CLAHE - adaptive histogram equalization
    if "clahe" in chain:
        options = _coerce_options(chain["clahe"])
        result = clahe_dispersion(
            result,
            clip_limit=_safe_float(options.get("clip_limit"), 0.03),
            tile_size=_safe_int(options.get("tile_size"), 16),
        )
        applied.append("clahe")

    # Unsharp mask - edge enhancement
    if "unsharp_mask" in chain:
        options = _coerce_options(chain["unsharp_mask"])
        result = unsharp_mask_dispersion(
            result,
            sigma=_safe_float(options.get("sigma"), 2.0),
            alpha=_safe_float(options.get("alpha"), 1.5),
            threshold=_safe_float(options.get("threshold"), 0.0),
        )
        applied.append("unsharp_mask")

    if applied:
        logger.info("Dispersion post-filters applied: %s", ", ".join(applied))

    return result


def detrend_time_series(
    Mt: np.ndarray, 
    axis: int = 0, 
    method: str = "mean"
) -> np.ndarray:
    """
    Detrend time series along specified axis.
    
    Parameters
    ----------
    Mt : np.ndarray
        Time series data
    axis : int
        Time axis
    method : {'mean', 'initial'}
        Detrending method:
        - 'mean': Remove time average
        - 'initial': Remove initial value
        
    Returns
    -------
    np.ndarray
        Detrended data
    """
    if method == "mean":
        mean = np.mean(Mt, axis=axis, keepdims=True)
        return Mt - mean
    elif method == "initial":
        init = np.take(Mt, indices=0, axis=axis)
        # Reshape for broadcasting
        slicer = [slice(None)] * Mt.ndim
        slicer[axis] = slice(0, 1)
        init = init.reshape([Mt.shape[i] if i != axis else 1 for i in range(Mt.ndim)])
        return Mt - init
    else:
        return Mt


def find_peaks_1d(
    y: np.ndarray, 
    min_prominence: float = 0.0
) -> np.ndarray:
    """
    Simple peak finder for 1D arrays.
    
    Parameters
    ----------
    y : np.ndarray
        1D signal
    min_prominence : float
        Minimum peak prominence to keep
        
    Returns
    -------
    np.ndarray
        Indices of detected peaks
    """
    y = np.asarray(y)
    if y.size < 3:
        return np.array([], dtype=int)

    prom = float(min_prominence or 0.0)
    if prom < 0:
        prom = 0.0

    # Prefer SciPy's reference implementation when available.
    if prom > 0:
        try:
            from scipy.signal import find_peaks as _scipy_find_peaks  # type: ignore
        except Exception:
            _scipy_find_peaks = None

        if _scipy_find_peaks is not None:
            peaks, _props = _scipy_find_peaks(y, prominence=prom)
            return np.asarray(peaks, dtype=int)

    dy = np.diff(y)
    # Candidate maxima where derivative changes from + to -.
    cand = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0] + 1

    if prom <= 0 or cand.size == 0:
        return np.asarray(cand, dtype=int)

    # Fallback prominence approximation (SciPy-like):
    # For each peak, extend a horizontal line at peak height until a higher point
    # is encountered (or boundary). Prominence is peak - max(min_left, min_right).
    keep: list[int] = []
    n = int(y.size)
    for idx in cand:
        peak = float(y[idx])

        left_min = peak
        j = int(idx)
        while j > 0:
            j -= 1
            left_min = min(left_min, float(y[j]))
            if float(y[j]) > peak:
                break

        right_min = peak
        j = int(idx)
        while j < n - 1:
            j += 1
            right_min = min(right_min, float(y[j]))
            if float(y[j]) > peak:
                break

        base = max(left_min, right_min)
        prominence = peak - base
        if prominence >= prom:
            keep.append(int(idx))

    return np.asarray(keep, dtype=int)


def group_velocity_1d(
    k_axis: np.ndarray,
    f_branch: np.ndarray, 
    angular: bool = True
) -> np.ndarray:
    """
    Estimate group velocity from dispersion branch.
    
    Parameters
    ----------
    k_axis : np.ndarray
        Wave vector values [rad/m]
    f_branch : np.ndarray
        Branch frequencies [Hz]
    angular : bool
        If True, return v_g = dω/dk [m/s] 
        If False, return df/dk [Hz⋅m]
        
    Returns
    -------
    np.ndarray
        Group velocity values
    """
    dk = np.gradient(k_axis)
    df = np.gradient(f_branch)
    
    vg = df / dk  # Hz⋅m
    
    if angular:
        vg *= 2 * np.pi  # Convert to rad/s per (rad/m) = m/s
        
    return vg


def normalize_magnetization_components(M: np.ndarray) -> np.ndarray:
    """
    Ensure magnetization array has proper shape and component ordering.
    
    Parameters
    ----------
    M : np.ndarray
        Magnetization array, expected shapes:
        - (T, Z, Y, X, 3)  - full 3-component vector
        - (T, Y, X, 3)     - 2D with 3 components
        - (T, X, 3)        - 1D with 3 components
        - (T, Z, Y, X)     - single component pre-selected
        - (T, Y, X)        - 2D single component
        - (T, X)           - 1D single component
        
    Returns
    -------
    np.ndarray
        Normalized array with shape (T, Z, Y, X, C) where C is 1 or 3
    """
    # Case 1: Full 5D data - can be (T, Z, Y, X, 3) or (T, Z, Y, X, 1)
    if M.ndim == 5:
        # (T, Z, Y, X, C) where C is 1 or 3
        if M.shape[-1] not in (1, 3):
            raise ValueError(f"5D array must have last axis=1 or 3, got {M.shape[-1]}")
        return M
    elif M.ndim == 4:
        # Could be (T, Y, X, 3) or (T, Z, Y, X) single component
        if M.shape[-1] == 3:
            # (T, Y, X, 3) -> (T, 1, Y, X, 3)
            T, Y, X, C = M.shape
            return M.reshape(T, 1, Y, X, C)
        else:
            # (T, Z, Y, X) single component -> (T, Z, Y, X, 1)
            return M[..., np.newaxis]
    elif M.ndim == 3:
        # Could be (T, X, 3) or (T, Y, X) single component
        if M.shape[-1] == 3:
            # (T, X, 3) -> (T, 1, 1, X, 3)
            T, X, C = M.shape
            return M.reshape(T, 1, 1, X, C)
        else:
            # (T, Y, X) single component -> (T, 1, Y, X, 1)
            T, Y, X = M.shape
            return M.reshape(T, 1, Y, X, 1)
    elif M.ndim == 2:
        # (T, X) single component -> (T, 1, 1, X, 1)
        T, X = M.shape
        return M.reshape(T, 1, 1, X, 1)
    else:
        raise ValueError(
            f"M must have 2-5 dimensions, got {M.ndim}. "
            f"Expected shapes: (T,Z,Y,X,3), (T,Y,X,3), (T,X,3), "
            f"or single-component (T,Z,Y,X), (T,Y,X), (T,X)"
        )
        
    return M


def extract_magnetization_component(
    M: np.ndarray, 
    component: str
) -> np.ndarray:
    """
    Extract specified magnetization component(s).
    
    Parameters
    ----------
    M : np.ndarray
        Magnetization array (..., C) where C is 1 (single component) or 3 (mx, my, mz)
    component : str
        Component to extract:
        - 'perp': mx + i*my (complex transverse)
        - 'mx', 'my', 'mz': individual components
        - 'sum': rough sum of all components
        - None or 'auto': use data as-is if already single component
        
    Returns
    -------
    np.ndarray
        Selected component(s), complex dtype
    """
    # If M only has 1 component (already selected via slicing), return it as-is
    if M.shape[-1] == 1:
        if component is None or component == "auto":
            # Already single component, just return it
            return M[..., 0].astype(np.complex128)
        else:
            # User specified component but data already has only 1 component
            # This is fine - just use what we have
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Magnetization data already has single component (shape[-1]=1). "
                f"Ignoring component='{component}' parameter and using existing data."
            )
            return M[..., 0].astype(np.complex128)
    
    # Standard case: M has 3 components
    if M.shape[-1] != 3:
        raise ValueError(
            f"Magnetization array must have last axis = 1 (single component) or 3 (mx,my,mz). "
            f"Got shape[-1] = {M.shape[-1]}"
        )
    
    mx = M[..., 0]
    my = M[..., 1] 
    mz = M[..., 2]

    if component == "perp" or component is None:
        return (mx + 1j * my).astype(np.complex128)
    elif component == "mx":
        return mx.astype(np.complex128)
    elif component == "my":
        return my.astype(np.complex128)
    elif component == "mz":
        return mz.astype(np.complex128)
    elif component == "sum":
        return ((mx + 1j * my) + mz).astype(np.complex128)
    else:
        raise ValueError(f"Unknown component '{component}'. Use 'perp', 'mx', 'my', 'mz', or 'sum'.")


def validate_grid_parameters(
    dt: float,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    dz: Optional[float] = None
) -> None:
    """
    Validate grid spacing parameters.
    
    Parameters
    ----------
    dt : float
        Time step [s]
    dx, dy, dz : Optional[float]
        Spatial grid spacings [m]
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    if dt <= 0:
        raise ValueError("Time step dt must be positive")
        
    for name, val in [("dx", dx), ("dy", dy), ("dz", dz)]:
        if val is not None and val <= 0:
            raise ValueError(f"Grid spacing {name} must be positive, got {val}")


def get_frequency_band_mask(
    f_axis: np.ndarray,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None
) -> np.ndarray:
    """
    Create boolean mask for frequency band selection.
    
    Parameters
    ----------
    f_axis : np.ndarray
        Frequency axis [Hz]
    f_min, f_max : Optional[float]
        Frequency band limits [Hz]
        
    Returns
    -------
    np.ndarray
        Boolean mask for frequency selection
    """
    mask = np.ones(len(f_axis), dtype=bool)
    
    if f_min is not None:
        mask &= (f_axis >= f_min)
    if f_max is not None:
        mask &= (f_axis <= f_max)
        
    return mask


def create_amplitude_phase_colormap(
    complex_data: np.ndarray,
    saturation: float = 1.0,
    amp_min: Optional[float] = None,
    amp_max: Optional[float] = None,
) -> np.ndarray:
    """
    Create RGB image where phase determines hue and amplitude determines brightness.
    
    This creates a visualization that combines both amplitude and phase information:
    - Hue (color): Determined by phase angle (-π to π)
    - Value (brightness): Determined by amplitude (0 to max)
    - Saturation: Fixed (can be adjusted)
    
    Parameters
    ----------
    complex_data : np.ndarray
        Complex array of shape (M, N) or any 2D shape
    saturation : float
        HSV saturation value (0 to 1). Default 1.0 for vivid colors.
    amp_min : float, optional
        Minimum amplitude for scaling. If None, uses data minimum.
    amp_max : float, optional  
        Maximum amplitude for scaling. If None, uses data maximum.
    
    Returns
    -------
    np.ndarray
        RGB image of shape (M, N, 3) with values in [0, 1]
    
    Examples
    --------
    >>> # Create test complex data
    >>> x = np.linspace(-np.pi, np.pi, 100)
    >>> y = np.linspace(-np.pi, np.pi, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> complex_data = (1 + 0.5*np.sin(X)) * np.exp(1j * Y)
    >>> 
    >>> # Generate RGB colormap
    >>> rgb = create_amplitude_phase_colormap(complex_data)
    >>> plt.imshow(rgb, origin='lower')
    >>> plt.title('Amplitude × Phase')
    >>> plt.show()
    """
    # Extract amplitude and phase
    amplitude = np.abs(complex_data)
    phase = np.angle(complex_data)  # Range: -π to π
    
    # Normalize amplitude to [0, 1]
    if amp_min is None:
        amp_min = amplitude.min()
    if amp_max is None:
        amp_max = amplitude.max()
    
    # Avoid division by zero
    if amp_max - amp_min < 1e-12:
        value = np.ones_like(amplitude)
    else:
        value = (amplitude - amp_min) / (amp_max - amp_min)
        value = np.clip(value, 0, 1)
    
    # Convert phase from [-π, π] to [0, 1] for hue
    # Phase = 0 → red, π/2 → green, -π/2 → purple, ±π → cyan
    hue = (phase + np.pi) / (2 * np.pi)  # Range: 0 to 1
    
    # Create HSV array
    hsv = np.stack([hue, np.full_like(hue, saturation), value], axis=-1)
    
    # Convert HSV to RGB using colorsys-based vectorized approach
    rgb = _hsv_to_rgb_array(hsv)
    
    return rgb


def _hsv_to_rgb_array(hsv: np.ndarray) -> np.ndarray:
    """
    Vectorized HSV to RGB conversion.
    
    Parameters
    ----------
    hsv : np.ndarray
        Array of shape (..., 3) with H, S, V in range [0, 1]
    
    Returns
    -------
    np.ndarray
        RGB array of same shape with values in [0, 1]
    """
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    
    # Convert using standard HSV→RGB algorithm
    c = v * s  # Chroma
    h_prime = h * 6.0  # Scale to [0, 6]
    x = c * (1 - np.abs(h_prime % 2 - 1))
    m = v - c
    
    # Initialize RGB
    rgb = np.zeros(hsv.shape)
    
    # Conditional assignment based on h_prime sector
    mask0 = (h_prime >= 0) & (h_prime < 1)
    mask1 = (h_prime >= 1) & (h_prime < 2)
    mask2 = (h_prime >= 2) & (h_prime < 3)
    mask3 = (h_prime >= 3) & (h_prime < 4)
    mask4 = (h_prime >= 4) & (h_prime < 5)
    mask5 = (h_prime >= 5) & (h_prime < 6)
    
    rgb[mask0, 0] = c[mask0]
    rgb[mask0, 1] = x[mask0]
    
    rgb[mask1, 0] = x[mask1]
    rgb[mask1, 1] = c[mask1]
    
    rgb[mask2, 1] = c[mask2]
    rgb[mask2, 2] = x[mask2]
    
    rgb[mask3, 1] = x[mask3]
    rgb[mask3, 2] = c[mask3]
    
    rgb[mask4, 0] = x[mask4]
    rgb[mask4, 2] = c[mask4]
    
    rgb[mask5, 0] = c[mask5]
    rgb[mask5, 2] = x[mask5]
    
    # Add minimum value
    rgb += m[..., np.newaxis]
    
    return np.clip(rgb, 0, 1)
