"""Core-expulsion detection for vortex trajectories."""

from __future__ import annotations

import numpy as np

from ..core.models import TrajectoryResult
from .models import CoreExpulsionEvent


def detect_core_expulsions(
    trajectory: TrajectoryResult,
    *,
    disk_radius: float,
    center: tuple[float, float] = (0.0, 0.0),
    expulsion_ratio: float = 0.95,
    refractory: float = 0.5e-9,
    min_duration: float = 0.0,
) -> list[CoreExpulsionEvent]:
    """Detect intervals where tracked core radius reaches sample edge."""
    if disk_radius <= 0.0:
        raise ValueError("disk_radius must be positive")

    threshold = float(expulsion_ratio) * float(disk_radius)
    time = np.asarray(trajectory.time, dtype=float)
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)
    cx = float(center[0])
    cy = float(center[1])
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    confidence = np.asarray(trajectory.confidence, dtype=float)

    if time.size != radius.size:
        raise ValueError("trajectory.time and trajectory radius must have the same length")

    events: list[CoreExpulsionEvent] = []
    if radius.size < 2:
        return events

    above = radius >= threshold
    last_event_time = -np.inf
    idx = 1
    while idx < above.size:
        if not (above[idx] and not above[idx - 1]):
            idx += 1
            continue

        start = idx
        end = idx
        while end + 1 < above.size and above[end + 1]:
            end += 1

        t_event = float(time[start])
        duration = float(time[end] - time[start]) if end > start else 0.0
        if t_event - last_event_time >= float(refractory) and duration >= float(min_duration):
            conf = float(np.mean(confidence[start : min(end + 1, confidence.size)]))
            events.append(
                CoreExpulsionEvent(
                    time=t_event,
                    index=int(start),
                    radius=float(radius[start]),
                    threshold=threshold,
                    confidence=conf,
                    duration=duration,
                    metadata={"center": (cx, cy)},
                )
            )
            last_event_time = t_event

        idx = end + 1

    return events


__all__ = ["detect_core_expulsions"]
