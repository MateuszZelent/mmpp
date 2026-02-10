"""Gyration spectrum computation from vortex core trajectory."""

from __future__ import annotations

import warnings

import numpy as np

from ..core.models import TrajectoryResult
from .models import VortexSpectrumResult

try:
    from scipy.signal import welch

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    welch = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


def _numpy_periodogram(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    signal = signal - float(np.mean(signal))
    n = signal.size

    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    fft = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=dt)
    power = (np.abs(fft) ** 2) / max(n, 1)
    return np.asarray(frequencies, dtype=float), np.asarray(power, dtype=float)


def _compute_scalar_spectrum(
    signal: np.ndarray,
    time: np.ndarray,
    *,
    method: str = "welch",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, float | str]]:
    """Compute scalar power spectrum using Welch or periodogram."""
    t = np.asarray(time, dtype=float)
    s = np.asarray(signal, dtype=float)
    if t.size < 2:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            method,
            {"status": "insufficient_samples"},
        )

    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    method_norm = method.lower()

    if method_norm == "welch":
        if SCIPY_AVAILABLE and welch is not None:
            seg = int(nperseg) if nperseg is not None else min(256, t.size)
            seg = max(8, min(seg, t.size))
            if noverlap is None:
                nover = seg // 2
            else:
                nover = int(noverlap)
            frequencies, power = welch(s, fs=fs, nperseg=seg, noverlap=nover)
            frequencies = np.asarray(frequencies, dtype=float)
            power = np.asarray(power, dtype=float)
            used_method = "welch"
        else:
            warnings.warn(
                "SciPy is unavailable; falling back from Welch to periodogram.",
                RuntimeWarning,
                stacklevel=2,
            )
            frequencies, power = _numpy_periodogram(s, dt)
            used_method = "periodogram"
    elif method_norm == "periodogram":
        frequencies, power = _numpy_periodogram(s, dt)
        used_method = "periodogram"
    else:
        raise ValueError("method must be 'welch' or 'periodogram'")

    meta: dict[str, float | str] = {
        "requested_method": method_norm,
        "dt": dt,
        "fs": fs,
    }
    return frequencies, power, used_method, meta


def compute_gyration_spectrum(
    trajectory: TrajectoryResult,
    *,
    method: str = "welch",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> VortexSpectrumResult:
    """Compute vortex gyration spectrum from tracked core coordinates."""
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)

    fx, pxx, used_x, meta = _compute_scalar_spectrum(
        x,
        trajectory.time,
        method=method,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    fy, pyy, used_y, _ = _compute_scalar_spectrum(
        y,
        trajectory.time,
        method=method,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    if fx.size == 0 or fy.size == 0:
        return VortexSpectrumResult(
            frequencies=np.array([], dtype=float),
            power=np.array([], dtype=float),
            method=method,
            metadata={"status": "insufficient_samples"},
        )

    size = min(fx.size, fy.size)
    frequencies = np.asarray(fx[:size], dtype=float)
    power = np.asarray(pxx[:size] + pyy[:size], dtype=float)
    used_method = used_x if used_x == used_y else "mixed"

    return VortexSpectrumResult(
        frequencies=np.asarray(frequencies, dtype=float),
        power=np.asarray(power, dtype=float),
        method=used_method,
        component="gyration",
        metadata=meta,
    )


def compute_breathing_spectrum(
    trajectory: TrajectoryResult,
    *,
    method: str = "welch",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> VortexSpectrumResult:
    """Compute breathing-mode spectrum from orbit radius signal ``r(t)``."""
    frequencies, power, used_method, meta = _compute_scalar_spectrum(
        trajectory.r,
        trajectory.time,
        method=method,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return VortexSpectrumResult(
        frequencies=np.asarray(frequencies, dtype=float),
        power=np.asarray(power, dtype=float),
        method=used_method,
        component="breathing",
        metadata=meta,
    )
