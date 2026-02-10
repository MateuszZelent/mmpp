"""Azimuthal mode-index estimation utilities."""

from __future__ import annotations


def estimate_azimuthal_index(*, mode_type: str, rotation_sense: str) -> int:
    """Estimate azimuthal index ``m`` using simple phase-rotation heuristics."""
    mode = mode_type.lower()
    sense = rotation_sense.upper()

    if mode == "gyration":
        return 0
    if mode == "breathing":
        return 0

    if mode == "azimuthal":
        return 1 if sense == "CCW" else -1

    return 0
