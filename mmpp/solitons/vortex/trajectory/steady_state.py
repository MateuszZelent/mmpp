"""Steady-state extraction for vortex trajectories."""

from __future__ import annotations

import numpy as np

from ..core.models import TrajectoryResult


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling std with same-length output."""
    window = max(int(window), 3)
    if window % 2 == 0:
        window += 1

    n = values.size
    out = np.zeros(n, dtype=float)

    half = window // 2
    for idx in range(n):
        lo = max(0, idx - half)
        hi = min(n, idx + half + 1)
        out[idx] = float(np.std(values[lo:hi]))

    return out


def extract_steady_state(
    trajectory: TrajectoryResult,
    *,
    threshold: float = 0.05,
    window: int = 31,
    min_samples: int | None = None,
    frequency_threshold: float | None = None,
) -> TrajectoryResult:
    """Extract steady-state portion of trajectory using variance stabilization."""
    n = trajectory.time.size
    if n == 0:
        return trajectory

    if min_samples is None:
        min_samples = max(8, n // 5)
    min_samples = min(max(1, int(min_samples)), n)

    radius_std = _rolling_std(np.asarray(trajectory.r, dtype=float), window)
    tail = radius_std[-min_samples:]
    ref = float(np.median(tail)) if tail.size else float(np.median(radius_std))
    denom = max(abs(ref), 1e-18)
    cond = np.abs(radius_std - ref) / denom <= float(threshold)

    if frequency_threshold is not None:
        omega = np.asarray(trajectory.instantaneous_frequency, dtype=float)
        omega_tail = omega[-min_samples:]
        omega_ref = float(np.median(omega_tail)) if omega_tail.size else float(np.median(omega))
        omega_denom = max(abs(omega_ref), 1e-18)
        cond_omega = np.abs(omega - omega_ref) / omega_denom <= float(frequency_threshold)
        cond = cond & cond_omega

    start_idx = n - min_samples
    for idx in range(0, n - min_samples + 1):
        segment = cond[idx:]
        if segment.size >= min_samples and float(np.mean(segment)) >= 0.9:
            start_idx = idx
            break

    metadata = dict(trajectory.metadata)
    metadata.update(
        {
            "steady_state": True,
            "steady_state_start_index": int(start_idx),
            "steady_state_threshold": float(threshold),
            "steady_state_window": int(window),
        }
    )

    return TrajectoryResult(
        time=np.asarray(trajectory.time[start_idx:], dtype=float),
        x=np.asarray(trajectory.x[start_idx:], dtype=float),
        y=np.asarray(trajectory.y[start_idx:], dtype=float),
        polarity=np.asarray(trajectory.polarity[start_idx:], dtype=int),
        method=trajectory.method,
        confidence=np.asarray(trajectory.confidence[start_idx:], dtype=float),
        metadata=metadata,
    )
