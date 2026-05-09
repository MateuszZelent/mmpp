"""Gyration spectrum computation from vortex core trajectory."""

from __future__ import annotations

import numpy as np

from mmpp._shared.spectral import compute_psd

from ..core.models import TrajectoryResult
from .models import VortexSpectrumResult


def _compute_scalar_spectrum(
    signal: np.ndarray,
    time: np.ndarray,
    *,
    method: str = "welch",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, float | str]]:
    """Compute scalar power spectrum using Welch or periodogram."""
    return compute_psd(
        np.asarray(signal, dtype=float),
        time=np.asarray(time, dtype=float),
        method=method,
        nperseg=nperseg,
        noverlap=noverlap,
    )


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
