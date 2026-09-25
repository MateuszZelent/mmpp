"""Shared spectral helpers for one-dimensional post-processing signals."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:  # pragma: no cover - backend availability is environment-dependent
    from mmpp.fft.dispersion import _fft_backend as _central_fft_backend

    FFT_BACKEND_AVAILABLE = True
except Exception:  # pragma: no cover
    _central_fft_backend: Any = None  # type: ignore[no-redef]
    FFT_BACKEND_AVAILABLE = False

try:  # pragma: no cover - optional dependency fallback is tested indirectly
    from scipy.signal import spectrogram as _scipy_spectrogram
    from scipy.signal import welch as _scipy_welch

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _scipy_spectrogram: Any = None  # type: ignore[no-redef]
    _scipy_welch: Any = None  # type: ignore[no-redef]
    SCIPY_AVAILABLE = False


def _fft_backend_info() -> dict[str, Any]:
    if _central_fft_backend is None:
        return {"backend": "numpy", "central_backend": False}
    try:
        info = dict(_central_fft_backend.get_info())
    except Exception:
        info = {"backend": "central_fft_unavailable"}
    info["central_backend"] = True
    return info


def _fft(signal: np.ndarray) -> np.ndarray:
    if _central_fft_backend is None:
        return np.fft.fft(signal)
    return _central_fft_backend.fft(signal)


def _rfft(signal: np.ndarray) -> np.ndarray:
    if _central_fft_backend is None:
        return np.fft.rfft(signal)
    return _central_fft_backend.rfft(signal)


def _fftfreq(n: int, dt: float) -> np.ndarray:
    if _central_fft_backend is None:
        return np.fft.fftfreq(n, d=dt)
    return _central_fft_backend.fftfreq(n, d=dt)


def _rfftfreq(n: int, dt: float) -> np.ndarray:
    if _central_fft_backend is None:
        return np.fft.rfftfreq(n, d=dt)
    return _central_fft_backend.rfftfreq(n, d=dt)


def infer_dt(time: np.ndarray | None = None, *, dt: float | None = None) -> float:
    """Infer a positive sample spacing from a time axis or explicit ``dt``."""
    if dt is not None:
        value = float(dt)
        if np.isfinite(value) and value > 0.0:
            return value
        raise ValueError("dt must be finite and positive")

    if time is None:
        raise ValueError("Either time or dt must be provided")

    t = np.asarray(time, dtype=float).reshape(-1)
    if t.size < 2:
        return float("nan")
    value = float(np.median(np.diff(t)))
    return value if np.isfinite(value) and value > 0.0 else float("nan")


def _prepare_time_signal(
    signal: np.ndarray,
    time: np.ndarray | None,
    *,
    resample_nonuniform: bool,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Validate or resample a time-first signal before a spectral transform."""
    x = np.asarray(signal)
    if x.ndim != 1:
        x = x.reshape(-1)
    if time is None:
        return x, None, False

    t = np.asarray(time, dtype=float).reshape(-1)
    if t.size != x.size:
        raise ValueError(
            "The time axis length must match the one-dimensional signal before FFT"
        )
    if t.size < 2:
        return x, t, False

    deltas = np.diff(t)
    if not np.all(np.isfinite(t)) or np.any(deltas <= 0):
        raise ValueError("The time axis must be finite and strictly increasing")
    mean_dt = float(np.mean(deltas))
    tolerance = max(abs(mean_dt) * 1e-6, np.finfo(float).eps * 10)
    max_deviation = float(np.max(np.abs(deltas - mean_dt)))
    if max_deviation <= tolerance:
        return x, t, False

    relative_deviation = max_deviation / abs(mean_dt)
    if not resample_nonuniform:
        raise ValueError(
            "FFT requires a uniformly sampled time axis; the largest step "
            f"deviation is {relative_deviation:.3g} of mean dt "
            f"(tolerance {tolerance / abs(mean_dt):.3g}). To linearly resample "
            "the data before FFT, pass resample_nonuniform=True."
        )

    uniform_time = np.linspace(t[0], t[-1], t.size)
    flat = x.reshape(t.size, -1)
    output = np.empty(flat.shape, dtype=np.result_type(x.dtype, np.float32))
    for column in range(flat.shape[1]):
        values = flat[:, column]
        if np.iscomplexobj(values):
            output[:, column] = np.interp(
                uniform_time, t, values.real
            ) + 1j * np.interp(uniform_time, t, values.imag)
        else:
            output[:, column] = np.interp(uniform_time, t, values)
    warnings.warn(
        "Shared spectral FFT detected a non-uniform time axis and linearly "
        "resampled it onto an endpoint-preserving uniform grid "
        f"(largest step deviation={relative_deviation:.3%} of mean dt). "
        "Interpolation may slightly attenuate or broaden high-frequency peaks "
        "and alter quantitative amplitudes or phases; pass "
        "resample_nonuniform=False for strict validation.",
        UserWarning,
        stacklevel=3,
    )
    return output.reshape(x.shape), uniform_time, True


def _windowed_periodogram(
    signal: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Hann-windowed one-sided periodogram."""
    x = np.asarray(signal)
    n = int(x.size)
    if n < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    centered = x - np.mean(x)
    window = np.hanning(n)
    if np.iscomplexobj(centered):
        spectrum = _fft(centered * window)
        frequencies = _fftfreq(n, float(dt))
        mask = frequencies >= 0.0
        frequencies = frequencies[mask]
        power = (np.abs(spectrum) ** 2)[mask]
    else:
        spectrum = _rfft(np.asarray(centered, dtype=float) * window)
        frequencies = _rfftfreq(n, float(dt))
        power = np.abs(spectrum) ** 2

    denom = max(float(np.sum(window**2)), 1e-30)
    return np.asarray(frequencies, dtype=float), np.asarray(power / denom, dtype=float)


def compute_psd(
    signal: np.ndarray,
    time: np.ndarray | None = None,
    *,
    dt: float | None = None,
    method: str = "welch",
    nperseg: int | None = None,
    noverlap: int | None = None,
    scaling: str = "density",
    detrend: str | bool = "constant",
    resample_nonuniform: bool = True,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Compute a one-dimensional power spectral density.

    ``method='welch'`` uses SciPy when available. If SciPy is unavailable, the
    fallback is an explicitly windowed periodogram so the fallback has the same
    broad leakage assumptions as Welch instead of a raw rectangular FFT.
    """
    if not isinstance(resample_nonuniform, (bool, np.bool_)):
        raise TypeError("resample_nonuniform must be boolean")
    x, prepared_time, did_resample = _prepare_time_signal(
        signal,
        time,
        resample_nonuniform=bool(resample_nonuniform),
    )
    sample_dt = infer_dt(prepared_time, dt=dt)
    if x.size < 2 or not np.isfinite(sample_dt):
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            str(method).lower(),
            {"status": "insufficient_samples"},
        )

    method_norm = str(method).lower()
    if method_norm == "fft":
        method_norm = "periodogram"
    if method_norm not in {"welch", "periodogram"}:
        raise ValueError("method must be 'welch', 'periodogram', or 'fft'")

    fs = 1.0 / sample_dt
    metadata: dict[str, Any] = {
        "requested_method": str(method).lower(),
        "dt": sample_dt,
        "fs": fs,
        "n_samples": int(x.size),
        "sidedness": "one-sided" if not np.iscomplexobj(x) else "positive frequencies",
        "resample_nonuniform": bool(resample_nonuniform),
        "resampled_nonuniform": did_resample,
    }

    if method_norm == "welch":
        if SCIPY_AVAILABLE and _scipy_welch is not None:
            seg = int(nperseg) if nperseg is not None else min(256, x.size)
            seg = max(8, min(seg, x.size))
            overlap = seg // 2 if noverlap is None else int(noverlap)
            overlap = min(max(overlap, 0), seg - 1)
            frequencies, power = _scipy_welch(
                x,
                fs=fs,
                nperseg=seg,
                noverlap=overlap,
                detrend=detrend,
                scaling=scaling,
            )
            metadata.update(
                {
                    "backend": "scipy.signal.welch",
                    "window": "hann (periodic)",
                    "nperseg": seg,
                    "nfft": seg,
                    "noverlap": overlap,
                    "detrend": detrend,
                    "scaling": scaling,
                    "average": "mean",
                }
            )
            return (
                np.asarray(frequencies, dtype=float),
                np.asarray(np.real(power), dtype=float),
                "welch",
                metadata,
            )

        warnings.warn(
            "SciPy is unavailable; falling back from Welch to a Hann-windowed periodogram.",
            RuntimeWarning,
            stacklevel=2,
        )

    frequencies, power = _windowed_periodogram(x, sample_dt)
    metadata.update(
        {
            **_fft_backend_info(),
            "window": "hann (symmetric)",
            "nperseg": int(x.size),
            "nfft": int(x.size),
            "noverlap": 0,
            "detrend": "constant",
            "normalization": "sum(window**2)",
        }
    )
    return frequencies, power, "periodogram", metadata


def _numpy_stft_psd(
    signal: np.ndarray,
    dt: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a Hann-windowed STFT PSD fallback."""
    step = max(1, nperseg - noverlap)
    starts = np.arange(0, max(signal.size - nperseg + 1, 1), step)
    window = np.hanning(nperseg)
    norm = max(float(np.sum(window**2)), 1e-30)

    spectra = []
    times = []
    for start in starts:
        segment = np.asarray(signal[start : start + nperseg], dtype=float)
        if segment.size < nperseg:
            padded = np.zeros(nperseg, dtype=float)
            padded[: segment.size] = segment
            segment = padded

        segment = (segment - float(np.mean(segment))) * window
        spectrum = _rfft(segment)
        spectra.append((np.abs(spectrum) ** 2) / norm)
        times.append((start + nperseg / 2.0) * dt)

    frequencies = _rfftfreq(nperseg, dt)
    matrix = (
        np.asarray(spectra, dtype=float).T if spectra else np.empty((0, 0), dtype=float)
    )
    return np.asarray(times, dtype=float), np.asarray(frequencies, dtype=float), matrix


def compute_spectrogram_psd(
    signal: np.ndarray,
    time: np.ndarray | None = None,
    *,
    dt: float | None = None,
    nperseg: int | None = None,
    noverlap: int | None = None,
    resample_nonuniform: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Compute a PSD spectrogram with SciPy and a NumPy STFT fallback."""
    if not isinstance(resample_nonuniform, (bool, np.bool_)):
        raise TypeError("resample_nonuniform must be boolean")
    prepared_signal, prepared_time, did_resample = _prepare_time_signal(
        signal,
        time,
        resample_nonuniform=bool(resample_nonuniform),
    )
    x = np.asarray(prepared_signal, dtype=float).reshape(-1)
    sample_dt = infer_dt(prepared_time, dt=dt)
    if x.size < 2 or not np.isfinite(sample_dt):
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.empty((0, 0), dtype=float),
            "stft",
            {"status": "insufficient_samples"},
        )

    seg = int(nperseg) if nperseg is not None else min(128, x.size)
    seg = max(8, min(seg, x.size))
    overlap = seg // 2 if noverlap is None else int(noverlap)
    overlap = min(max(overlap, 0), seg - 1)
    fs = 1.0 / sample_dt
    metadata: dict[str, Any] = {
        "dt": sample_dt,
        "fs": fs,
        "nperseg": int(seg),
        "noverlap": int(overlap),
        "resample_nonuniform": bool(resample_nonuniform),
        "resampled_nonuniform": did_resample,
    }

    if SCIPY_AVAILABLE and _scipy_spectrogram is not None:
        frequencies, times, power = _scipy_spectrogram(
            x,
            fs=fs,
            nperseg=seg,
            noverlap=overlap,
            detrend="constant",
            mode="psd",
        )
        return (
            np.asarray(times, dtype=float),
            np.asarray(frequencies, dtype=float),
            np.asarray(power, dtype=float),
            "scipy_stft",
            metadata,
        )

    warnings.warn(
        "SciPy is unavailable; using NumPy STFT fallback.",
        RuntimeWarning,
        stacklevel=2,
    )
    times, frequencies, power = _numpy_stft_psd(x, sample_dt, seg, overlap)
    metadata.update(_fft_backend_info())
    return times, frequencies, power, "numpy_stft", metadata


__all__ = [
    "FFT_BACKEND_AVAILABLE",
    "SCIPY_AVAILABLE",
    "compute_psd",
    "compute_spectrogram_psd",
    "infer_dt",
]
