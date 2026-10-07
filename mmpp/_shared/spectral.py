"""Shared spectral helpers for one-dimensional post-processing signals."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:  # pragma: no cover - backend availability is environment-dependent
    from mmpp.fft import _backend as _central_fft_backend

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


def _detrend_segment(signal: np.ndarray, detrend: Any) -> np.ndarray:
    """Apply SciPy-compatible constant/linear detrending to one segment."""
    values = np.asarray(signal)
    if detrend is False or (isinstance(detrend, np.bool_) and not bool(detrend)):
        return values
    if callable(detrend):
        detrended = np.asarray(detrend(values))
        if detrended.shape != values.shape:
            raise ValueError("A detrend callable must preserve the segment shape")
        return detrended

    mode = str(detrend).lower()
    if mode == "constant":
        return values - np.mean(values)
    if mode == "linear":
        coordinate = np.arange(values.size, dtype=float)
        centered_coordinate = coordinate - np.mean(coordinate)
        centered_values = values - np.mean(values)
        slope = np.sum(centered_values * centered_coordinate) / np.sum(
            centered_coordinate**2
        )
        return values - (np.mean(values) + slope * centered_coordinate)
    raise ValueError("detrend must be 'constant', 'linear', False, or callable")


def _hann_window(size: int) -> np.ndarray:
    """Return a periodic Hann window, with a useful singleton definition."""
    if int(size) == 1:
        return np.ones(1, dtype=float)
    return np.hanning(int(size) + 1)[:-1]


def _windowed_periodogram(
    signal: np.ndarray,
    dt: float,
    *,
    scaling: str = "density",
    detrend: Any = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Hann-windowed periodogram with SciPy-compatible units."""
    x = np.asarray(signal).reshape(-1)
    n = int(x.size)
    if n < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    if scaling not in {"density", "spectrum"}:
        raise ValueError("scaling must be 'density' or 'spectrum'")

    window = _hann_window(n)
    transformed = _detrend_segment(x, detrend) * window
    if np.iscomplexobj(transformed):
        spectrum = _fft(transformed)
        frequencies = _fftfreq(n, float(dt))
        power = np.abs(spectrum) ** 2
        denominator = (
            (1.0 / float(dt)) * float(np.sum(window**2))
            if scaling == "density"
            else float(np.sum(window)) ** 2
        )
    else:
        spectrum = _rfft(np.asarray(transformed, dtype=float))
        frequencies = _rfftfreq(n, float(dt))
        power = np.abs(spectrum) ** 2
        if n % 2 == 0:
            power[1:-1] *= 2.0
        else:
            power[1:] *= 2.0
        denominator = (
            (1.0 / float(dt)) * float(np.sum(window**2))
            if scaling == "density"
            else float(np.sum(window)) ** 2
        )
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("The selected window has zero spectral normalization")
    return np.asarray(frequencies, dtype=float), np.asarray(
        power / denominator, dtype=float
    )


def _numpy_welch(
    signal: np.ndarray,
    *,
    fs: float,
    nperseg: int,
    noverlap: int,
    scaling: str,
    detrend: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy Welch fallback using the same periodic Hann and PSD units as SciPy."""
    x = np.asarray(signal).reshape(-1)
    step = nperseg - noverlap
    window = _hann_window(nperseg)
    spectra: list[np.ndarray] = []
    for start in range(0, x.size - nperseg + 1, step):
        segment = _detrend_segment(x[start : start + nperseg], detrend) * window
        if np.iscomplexobj(segment):
            transform = _fft(segment)
            scale = (
                fs * float(np.sum(window**2))
                if scaling == "density"
                else float(np.sum(window)) ** 2
            )
            spectra.append(np.abs(transform) ** 2 / scale)
        else:
            transform = _rfft(np.asarray(segment, dtype=float))
            power = np.abs(transform) ** 2
            if nperseg % 2 == 0:
                power[1:-1] *= 2.0
            else:
                power[1:] *= 2.0
            scale = (
                fs * float(np.sum(window**2))
                if scaling == "density"
                else float(np.sum(window)) ** 2
            )
            spectra.append(power / scale)

    if not spectra:
        return np.array([], dtype=float), np.array([], dtype=float)
    mean_power = np.mean(np.stack(spectra), axis=0)
    frequencies = (
        _fftfreq(nperseg, 1.0 / fs)
        if np.iscomplexobj(x)
        else _rfftfreq(nperseg, 1.0 / fs)
    )
    return np.asarray(frequencies, dtype=float), np.asarray(mean_power, dtype=float)


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
    """Compute a one-dimensional density or spectrum.

    ``method='welch'`` uses SciPy when available and an equivalent segmented
    NumPy implementation otherwise. Complex signals retain both frequency
    directions; real signals use the correctly folded one-sided spectrum.
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
    if scaling not in {"density", "spectrum"}:
        raise ValueError("scaling must be 'density' or 'spectrum'")

    fs = 1.0 / sample_dt
    metadata: dict[str, Any] = {
        "requested_method": str(method).lower(),
        "dt": sample_dt,
        "fs": fs,
        "n_samples": int(x.size),
        "sidedness": "one-sided" if not np.iscomplexobj(x) else "two-sided",
        "resample_nonuniform": bool(resample_nonuniform),
        "resampled_nonuniform": did_resample,
    }

    if method_norm == "welch":
        if SCIPY_AVAILABLE and _scipy_welch is not None:
            seg = int(nperseg) if nperseg is not None else min(256, x.size)
            if seg <= 0:
                raise ValueError("nperseg must be a positive integer")
            seg = min(seg, x.size)
            if seg < 2:
                raise ValueError("nperseg must be at least 2")
            overlap = seg // 2 if noverlap is None else int(noverlap)
            if overlap < 0:
                raise ValueError("noverlap must be non-negative")
            if overlap >= seg:
                raise ValueError("noverlap must be smaller than nperseg")
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
            "SciPy is unavailable; using a NumPy Hann-windowed spectral estimate.",
            RuntimeWarning,
            stacklevel=2,
        )

    if method_norm == "welch":
        seg = int(nperseg) if nperseg is not None else min(256, x.size)
        if seg <= 0:
            raise ValueError("nperseg must be a positive integer")
        seg = min(seg, x.size)
        if seg < 2:
            raise ValueError("nperseg must be at least 2")
        overlap = seg // 2 if noverlap is None else int(noverlap)
        if overlap < 0:
            raise ValueError("noverlap must be non-negative")
        if overlap >= seg:
            raise ValueError("noverlap must be smaller than nperseg")
        frequencies, power = _numpy_welch(
            x,
            fs=fs,
            nperseg=seg,
            noverlap=overlap,
            scaling=scaling,
            detrend=detrend,
        )
        metadata.update(
            {
                **_fft_backend_info(),
                "window": "hann (periodic)",
                "nperseg": seg,
                "nfft": seg,
                "noverlap": overlap,
                "detrend": detrend,
                "scaling": scaling,
                "average": "mean",
                "normalization": scaling,
            }
        )
        return frequencies, power, "welch", metadata

    frequencies, power = _windowed_periodogram(
        x, sample_dt, scaling=scaling, detrend=detrend
    )
    metadata.update(
        {
            **_fft_backend_info(),
            "window": "hann (periodic)",
            "nperseg": int(x.size),
            "nfft": int(x.size),
            "noverlap": 0,
            "detrend": detrend,
            "scaling": scaling,
            "normalization": scaling,
        }
    )
    return frequencies, power, "periodogram", metadata


def _numpy_stft_psd(
    signal: np.ndarray,
    dt: float,
    nperseg: int,
    noverlap: int,
    time_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a density-scaled, Hann-windowed STFT PSD fallback."""
    step = max(1, nperseg - noverlap)
    starts = np.arange(0, max(signal.size - nperseg + 1, 1), step)
    window = _hann_window(nperseg)
    norm = max(float((1.0 / dt) * np.sum(window**2)), 1e-30)

    spectra = []
    times = []
    for start in starts:
        segment = np.asarray(signal[start : start + nperseg])
        if segment.size < nperseg:
            padded = np.zeros(nperseg, dtype=segment.dtype)
            padded[: segment.size] = segment
            segment = padded

        segment = _detrend_segment(segment, "constant") * window
        if np.iscomplexobj(segment):
            spectrum = _fft(segment)
            power = np.abs(spectrum) ** 2 / norm
            frequencies = _fftfreq(nperseg, dt)
        else:
            spectrum = _rfft(np.asarray(segment, dtype=float))
            power = np.abs(spectrum) ** 2 / norm
            if nperseg % 2 == 0:
                power[1:-1] *= 2.0
            else:
                power[1:] *= 2.0
            frequencies = _rfftfreq(nperseg, dt)
        spectra.append(power)
        times.append(time_offset + (start + nperseg / 2.0) * dt)

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
    x = np.asarray(prepared_signal).reshape(-1)
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
    if seg <= 0:
        raise ValueError("nperseg must be a positive integer")
    seg = min(seg, x.size)
    overlap = seg // 2 if noverlap is None else int(noverlap)
    if overlap < 0:
        raise ValueError("noverlap must be non-negative")
    if overlap >= seg:
        raise ValueError("noverlap must be smaller than nperseg")
    fs = 1.0 / sample_dt
    time_offset = (
        float(prepared_time[0])
        if prepared_time is not None and prepared_time.size
        else 0.0
    )
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
            window="hann",
            nperseg=seg,
            noverlap=overlap,
            detrend="constant",
            scaling="density",
            mode="psd",
        )
        return (
            np.asarray(times, dtype=float) + time_offset,
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
    times, frequencies, power = _numpy_stft_psd(
        x, sample_dt, seg, overlap, time_offset=time_offset
    )
    metadata.update(_fft_backend_info())
    return times, frequencies, power, "numpy_stft", metadata


__all__ = [
    "FFT_BACKEND_AVAILABLE",
    "SCIPY_AVAILABLE",
    "compute_psd",
    "compute_spectrogram_psd",
    "infer_dt",
]
