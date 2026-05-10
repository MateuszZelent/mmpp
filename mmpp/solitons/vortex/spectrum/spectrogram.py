"""Time-frequency spectrogram for vortex trajectory signals."""

from __future__ import annotations

import numpy as np

from mmpp._shared.spectral import compute_spectrogram_psd

from ..core.models import TrajectoryResult
from .models import VortexSpectrogramResult


def _select_signal(trajectory: TrajectoryResult, component: str) -> np.ndarray:
    component_norm = component.lower()
    if component_norm == "x":
        return np.asarray(trajectory.x, dtype=float)
    if component_norm == "y":
        return np.asarray(trajectory.y, dtype=float)
    if component_norm == "radius":
        return np.asarray(trajectory.r, dtype=float)
    raise ValueError("component must be 'x', 'y', or 'radius'")


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

    times, frequencies, power, used_method, metadata = compute_spectrogram_psd(
        signal,
        time=t,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    return VortexSpectrogramResult(
        times=np.asarray(times, dtype=float),
        frequencies=np.asarray(frequencies, dtype=float),
        power=np.asarray(power, dtype=float),
        method=used_method,
        component=component,
        metadata=metadata,
    )
