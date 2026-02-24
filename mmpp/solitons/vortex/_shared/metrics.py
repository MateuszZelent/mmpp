"""Shared trajectory metric helpers (phase 1 scaffold)."""

from __future__ import annotations

import numpy as np


def trajectory_mean_radius(trajectory) -> float:
    """Return mean orbit radius for a trajectory."""
    return float(np.mean(np.asarray(trajectory.r, dtype=float))) if trajectory.r.size else 0.0


__all__ = ["trajectory_mean_radius"]
