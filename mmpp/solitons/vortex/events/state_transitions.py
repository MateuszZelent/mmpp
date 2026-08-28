"""G/C state transition detection for vortex trajectories."""

from __future__ import annotations

import numpy as np

from ..core.models import TrajectoryResult
from .models import StateSwitchEvent


def classify_gc_states(
    trajectory: TrajectoryResult,
    *,
    radius_threshold: float = 0.6,
    smoothing_window: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each sample as ``G-state`` or ``C-state`` using normalized orbit radius."""
    radius = np.asarray(trajectory.r, dtype=float)
    if radius.size == 0:
        return np.array([], dtype="<U8"), np.array([], dtype=float)

    window = max(int(smoothing_window), 1)
    if window > 1 and radius.size >= window:
        kernel = np.ones(window, dtype=float) / float(window)
        smooth_radius = np.convolve(radius, kernel, mode="same")
    else:
        smooth_radius = radius

    ref = float(np.percentile(smooth_radius, 95))
    ref = max(ref, 1e-30)
    normalized = smooth_radius / ref
    labels = np.where(normalized <= float(radius_threshold), "G-state", "C-state")
    return labels.astype("<U8"), normalized


def _estimate_min_dwell_time(
    trajectory: TrajectoryResult, min_dwell_periods: int
) -> float:
    if trajectory.time.size < 2:
        return 0.0
    dt = float(np.median(np.diff(np.asarray(trajectory.time, dtype=float))))
    omega = np.asarray(trajectory.instantaneous_frequency, dtype=float)
    finite = np.isfinite(omega) & (np.abs(omega) > 0.0)
    if not np.any(finite):
        return float(max(min_dwell_periods, 0)) * dt
    freq_hz = np.median(np.abs(omega[finite]) / (2.0 * np.pi))
    if freq_hz <= 0.0:
        return float(max(min_dwell_periods, 0)) * dt
    return float(max(min_dwell_periods, 0)) / float(freq_hz)


def _segment_labels(labels: np.ndarray) -> list[tuple[int, int, str]]:
    if labels.size == 0:
        return []
    segments: list[tuple[int, int, str]] = []
    start = 0
    current = str(labels[0])
    for idx in range(1, labels.size):
        label = str(labels[idx])
        if label != current:
            segments.append((start, idx - 1, current))
            start = idx
            current = label
    segments.append((start, labels.size - 1, current))
    return segments


def _merge_short_segments(
    segments: list[tuple[int, int, str]],
    time: np.ndarray,
    min_dwell_time: float,
) -> list[tuple[int, int, str]]:
    if not segments:
        return []

    if len(segments) > 1:
        first_start, first_end, first_label = segments[0]
        first_duration = float(time[first_end] - time[first_start])
        if first_duration < min_dwell_time:
            next_start, next_end, next_label = segments[1]
            segments = [(first_start, next_end, next_label)] + segments[2:]

    merged: list[tuple[int, int, str]] = []
    for start, end, label in segments:
        duration = float(time[end] - time[start]) if end > start else 0.0
        if merged and duration < min_dwell_time:
            prev_start, _, prev_label = merged[-1]
            merged[-1] = (prev_start, end, prev_label)
        else:
            merged.append((start, end, label))
    return merged


def detect_state_switches(
    trajectory: TrajectoryResult,
    *,
    radius_threshold: float = 0.6,
    min_dwell_periods: int = 3,
    refractory: float = 0.5e-9,
    smoothing_window: int = 9,
) -> tuple[list[StateSwitchEvent], np.ndarray]:
    """Detect transitions between ``G-state`` and ``C-state``."""
    time = np.asarray(trajectory.time, dtype=float)
    confidence = np.asarray(trajectory.confidence, dtype=float)
    labels, normalized_radius = classify_gc_states(
        trajectory,
        radius_threshold=radius_threshold,
        smoothing_window=smoothing_window,
    )

    if labels.size != time.size:
        raise ValueError("trajectory.time and state labels must have the same length")

    min_dwell_time = _estimate_min_dwell_time(trajectory, min_dwell_periods)
    segments = _segment_labels(labels)
    merged = _merge_short_segments(segments, time, min_dwell_time)

    filtered_labels = np.asarray(labels, dtype="<U8").copy()
    for start, end, label in merged:
        filtered_labels[start : end + 1] = label

    events: list[StateSwitchEvent] = []
    last_time = -np.inf
    for idx in range(1, len(merged)):
        prev_start, prev_end, prev_label = merged[idx - 1]
        curr_start, _, curr_label = merged[idx]
        if prev_label == curr_label:
            continue

        t = float(time[curr_start])
        if t - last_time < float(refractory):
            continue
        last_time = t

        conf = float(
            np.mean(
                confidence[
                    max(0, curr_start - 1) : min(curr_start + 1, confidence.size)
                ]
            )
        )
        events.append(
            StateSwitchEvent(
                time=t,
                index=int(curr_start),
                from_state=prev_label,
                to_state=curr_label,
                confidence=conf,
                metadata={
                    "radius_before_norm": float(
                        np.mean(normalized_radius[prev_start : prev_end + 1])
                    ),
                    "radius_after_norm": float(
                        np.mean(
                            normalized_radius[
                                curr_start : min(curr_start + 3, normalized_radius.size)
                            ]
                        )
                    ),
                    "min_dwell_time": float(min_dwell_time),
                },
            )
        )

    return events, filtered_labels


__all__ = ["classify_gc_states", "detect_state_switches"]
