"""Radial mode-index estimation utilities."""

from __future__ import annotations


def estimate_radial_index(*, mode_type: str, harmonic: float) -> int:
    """Estimate radial index ``n`` from harmonic relation to base gyration mode."""
    mode = mode_type.lower()

    if mode == "gyration":
        return 0
    if mode == "breathing":
        return 1

    if mode == "azimuthal":
        return max(0, int(round(abs(harmonic) - 1.0)))

    return 0
