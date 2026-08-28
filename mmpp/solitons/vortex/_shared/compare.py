"""Comparison accessors for trajectory-to-trajectory overlays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrajectoryComparison:
    """Pairwise comparison between two trajectory results."""

    lhs: Any
    rhs: Any
    label: tuple[str, str] = ("lhs", "rhs")
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def delta_f_mean(self) -> float:
        f_lhs = float(
            np.mean(np.asarray(self.lhs.instantaneous_frequency, dtype=float))
        )
        f_rhs = float(
            np.mean(np.asarray(self.rhs.instantaneous_frequency, dtype=float))
        )
        return abs(f_lhs - f_rhs)

    @property
    def metrics(self):
        return _TrajectoryComparisonMetrics(self)

    @property
    def plot(self):
        return _TrajectoryComparisonPlot(self)


class _TrajectoryComparisonMetrics:
    def __init__(self, comparison: TrajectoryComparison):
        self._comparison = comparison

    @property
    def delta_f_mean(self) -> float:
        return self._comparison.delta_f_mean


class _TrajectoryComparisonPlot:
    def __init__(self, comparison: TrajectoryComparison):
        self._comparison = comparison

    def overlay_orbit(self, *, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5.2, 4.6), dpi=110)

        lhs_style = dict(kwargs)
        rhs_style = dict(kwargs)
        lhs_style.setdefault("color", "#1d4ed8")
        rhs_style.setdefault("color", "#dc2626")
        rhs_style.setdefault("linestyle", "--")

        lhs_label, rhs_label = self._comparison.label
        ax.plot(
            np.asarray(self._comparison.lhs.x, dtype=float),
            np.asarray(self._comparison.lhs.y, dtype=float),
            label=str(lhs_label),
            **lhs_style,
        )
        ax.plot(
            np.asarray(self._comparison.rhs.x, dtype=float),
            np.asarray(self._comparison.rhs.y, dtype=float),
            label=str(rhs_label),
            **rhs_style,
        )
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Orbit overlay")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.legend()
        return ax


class TrajectoryComparisonAccessor:
    """Entry-point accessor from ``TrajectoryResult.compare``."""

    def __init__(self, reference):
        self._reference = reference

    def with_(
        self,
        other,
        *,
        label: tuple[str, str] = ("reference", "candidate"),
        metadata: dict[str, Any] | None = None,
    ) -> TrajectoryComparison:
        return TrajectoryComparison(
            lhs=self._reference,
            rhs=other,
            label=label,
            metadata=dict(metadata or {}),
        )


__all__ = ["TrajectoryComparison", "TrajectoryComparisonAccessor"]
