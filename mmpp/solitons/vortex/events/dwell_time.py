"""Dwell-time statistics for classified vortex states."""

from __future__ import annotations

import numpy as np

from .models import DwellTimeResult


def dwell_time_statistics(
    time: np.ndarray,
    state_labels: np.ndarray,
    *,
    state: str = "G-state",
) -> DwellTimeResult:
    """Compute dwell-time statistics for selected state label."""
    t = np.asarray(time, dtype=float)
    labels = np.asarray(state_labels, dtype=str)

    if t.size != labels.size:
        raise ValueError("time and state_labels must have the same length")
    if t.size == 0:
        return DwellTimeResult(state=state, dwell_times=np.array([], dtype=float))

    mask = labels == str(state)
    dwell: list[float] = []

    idx = 0
    while idx < mask.size:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        end = idx
        while end + 1 < mask.size and mask[end + 1]:
            end += 1
        if end > start:
            dwell.append(float(t[end] - t[start]))
        idx = end + 1

    dwell_arr = np.asarray(dwell, dtype=float)
    return DwellTimeResult(
        state=str(state),
        dwell_times=dwell_arr,
        metadata={
            "n_segments": int(dwell_arr.size),
            "total_time": float(np.sum(dwell_arr)) if dwell_arr.size else 0.0,
        },
    )


__all__ = ["dwell_time_statistics"]
