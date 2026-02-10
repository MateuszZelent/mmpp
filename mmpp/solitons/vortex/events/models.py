"""Result models for vortex event detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PolaritySwitchEvent:
    """Detected polarity switch event."""

    time: float
    index: int
    from_p: int
    to_p: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSwitchEvent:
    """Detected G/C state transition event."""

    time: float
    index: int
    from_state: str
    to_state: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreExpulsionEvent:
    """Detected core-expulsion event when orbit reaches disk edge."""

    time: float
    index: int
    radius: float
    threshold: float
    confidence: float
    duration: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DwellTimeResult:
    """State dwell-time statistics."""

    state: str
    dwell_times: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of dwell intervals."""
        return int(np.asarray(self.dwell_times).size)

    @property
    def mean_dwell_time(self) -> float:
        """Mean dwell time in seconds."""
        values = np.asarray(self.dwell_times, dtype=float)
        return float(np.mean(values)) if values.size else float("nan")

    @property
    def std_dwell_time(self) -> float:
        """Standard deviation of dwell times in seconds."""
        values = np.asarray(self.dwell_times, dtype=float)
        return float(np.std(values)) if values.size else float("nan")

    @property
    def total_time(self) -> float:
        """Total accumulated dwell time in seconds."""
        values = np.asarray(self.dwell_times, dtype=float)
        return float(np.sum(values)) if values.size else 0.0

    @property
    def fitted_tau(self) -> float:
        """Characteristic exponential time estimated as sample mean."""
        return self.mean_dwell_time

    @property
    def plt(self) -> DwellTimePlotAccessor:
        """Plotting accessor."""
        return DwellTimePlotAccessor(self)


class DwellTimePlotAccessor:
    """Plot helpers for :class:`DwellTimeResult`."""

    def __init__(self, result: DwellTimeResult):
        self._result = result

    def dwell_histogram(self, *, ax=None, bins: int = 20, as_ns: bool = True, **kwargs):
        """Plot dwell-time histogram."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        values = np.asarray(self._result.dwell_times, dtype=float)
        if as_ns:
            values = values * 1e9
            xlabel = "Dwell time [ns]"
        else:
            xlabel = "Dwell time [s]"

        if values.size:
            ax.hist(values, bins=min(max(int(bins), 1), max(values.size, 1)), **kwargs)
        else:
            ax.hist([], bins=1, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"Dwell-time distribution: {self._result.state}")
        return ax
