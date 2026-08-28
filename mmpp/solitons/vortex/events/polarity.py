"""Polarity-switch detection for vortex trajectories."""

from __future__ import annotations

import numpy as np

from ..core.models import TrajectoryResult
from .models import PolaritySwitchEvent


def _binary_polarity_series(values: np.ndarray, threshold: float) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return np.array([], dtype=int)

    out = np.zeros(series.size, dtype=int)
    current = 1 if series[0] >= 0.0 else -1
    for idx, value in enumerate(series):
        abs_value = abs(float(value))
        if abs_value >= float(threshold):
            current = 1 if value >= 0.0 else -1
        out[idx] = current
    return out


def detect_polarity_switches(
    trajectory: TrajectoryResult,
    *,
    threshold: float = 0.5,
    refractory: float = 0.5e-9,
) -> list[PolaritySwitchEvent]:
    """Detect polarity switches from tracked polarity time series."""
    time = np.asarray(trajectory.time, dtype=float)
    polarity = _binary_polarity_series(
        np.asarray(trajectory.polarity, dtype=float), threshold
    )
    confidence = np.asarray(trajectory.confidence, dtype=float)

    if time.size != polarity.size:
        raise ValueError(
            "trajectory.time and trajectory.polarity must have the same length"
        )

    events: list[PolaritySwitchEvent] = []
    if polarity.size < 2:
        return events

    last_time = -np.inf
    for idx in range(1, polarity.size):
        prev_state = int(polarity[idx - 1])
        next_state = int(polarity[idx])
        if prev_state == next_state:
            continue

        t = float(time[idx])
        if t - last_time < float(refractory):
            continue
        last_time = t

        conf = float(
            np.mean(confidence[max(0, idx - 1) : min(idx + 1, confidence.size)])
        )
        events.append(
            PolaritySwitchEvent(
                time=t,
                index=int(idx),
                from_p=prev_state,
                to_p=next_state,
                confidence=conf,
            )
        )

    return events


__all__ = ["detect_polarity_switches"]
