"""Time-frequency spectrogram for vortex trajectory signals."""

from __future__ import annotations

import warnings

import numpy as np

from ..core.models import TrajectoryResult
from .models import VortexSpectrogramResult

try:
    from scipy.signal import spectrogram

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    spectrogram = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


def _select_signal(trajectory: TrajectoryResult, component: str) -> np.ndarray:
    component_norm = component.lower()
    if component_norm == "x":
        return np.asarray(trajectory.x, dtype=float)
    if component_norm == "y":
        return np.asarray(trajectory.y, dtype=float)
    if component_norm == "radius":
        return np.asarray(trajectory.r, dtype=float)
    raise ValueError("component must be 'x', 'y', or 'radius'")


def _numpy_stft(signal: np.ndarray, dt: float, nperseg: int, noverlap: int):
    step = max(1, nperseg - noverlap)
    starts = np.arange(0, max(signal.size - nperseg + 1, 1), step)
    window = np.hanning(nperseg)
    norm = np.sum(window**2)

    spectra = []
    times = []
    for start in starts:
        segment = signal[start : start + nperseg]
        if segment.size < nperseg:
            pad = np.zeros(nperseg, dtype=float)
            pad[: segment.size] = segment
            segment = pad

        segment = (segment - np.mean(segment)) * window
        fft = np.fft.rfft(segment)
        power = (np.abs(fft) ** 2) / max(norm, 1e-18)
        spectra.append(power)
        times.append((start + nperseg / 2.0) * dt)

    frequencies = np.fft.rfftfreq(nperseg, d=dt)
    matrix = np.asarray(spectra, dtype=float).T if spectra else np.empty((0, 0), dtype=float)
    return np.asarray(times, dtype=float), np.asarray(frequencies, dtype=float), matrix


def compute_spectrogram(
    trajectory: TrajectoryResult,
    *,
    component: str = "radius",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> VortexSpectrogramResult:
    """Compute spectrogram for selected trajectory component."""
    signal = _select_signal(trajectory, component)
    t = np.asarray(trajectory.time, dtype=float)

    if t.size < 2 or signal.size < 2:
        return VortexSpectrogramResult(
            times=np.array([], dtype=float),
            frequencies=np.array([], dtype=float),
            power=np.empty((0, 0), dtype=float),
            method="stft",
            component=component,
            metadata={"status": "insufficient_samples"},
        )

    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt

    seg = int(nperseg) if nperseg is not None else min(128, signal.size)
    seg = max(8, min(seg, signal.size))
    if noverlap is None:
        nover = seg // 2
    else:
        nover = int(noverlap)
    nover = min(max(nover, 0), seg - 1)

    if SCIPY_AVAILABLE and spectrogram is not None:
        frequencies, times, power = spectrogram(
            signal,
            fs=fs,
            nperseg=seg,
            noverlap=nover,
            detrend="constant",
            mode="psd",
        )
        used_method = "scipy_stft"
    else:
        warnings.warn(
            "SciPy is unavailable; using NumPy STFT fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        times, frequencies, power = _numpy_stft(signal, dt, seg, nover)
        used_method = "numpy_stft"

    return VortexSpectrogramResult(
        times=np.asarray(times, dtype=float),
        frequencies=np.asarray(frequencies, dtype=float),
        power=np.asarray(power, dtype=float),
        method=used_method,
        component=component,
        metadata={
            "dt": dt,
            "fs": fs,
            "nperseg": int(seg),
            "noverlap": int(nover),
        },
    )
