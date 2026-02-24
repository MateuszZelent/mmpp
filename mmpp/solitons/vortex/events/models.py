"""Result models for vortex event detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmpp._shared.repr_html import make_simple_card

from .._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)


@dataclass
class PolaritySwitchEvent:
    """Detected polarity switch event."""

    time: float
    index: int
    from_p: int
    to_p: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def _repr_html_(self) -> str:
        rows = [
            ("time_s", f"{float(self.time):.6g}"),
            ("index", str(int(self.index))),
            ("transition", f"{int(self.from_p)} -> {int(self.to_p)}"),
            ("confidence", f"{float(self.confidence):.6g}"),
        ]
        return make_simple_card(
            title="PolaritySwitchEvent",
            subtitle="Detected polarity transition event",
            rows=rows,
        )


@dataclass
class StateSwitchEvent:
    """Detected G/C state transition event."""

    time: float
    index: int
    from_state: str
    to_state: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def _repr_html_(self) -> str:
        rows = [
            ("time_s", f"{float(self.time):.6g}"),
            ("index", str(int(self.index))),
            ("transition", f"{self.from_state} -> {self.to_state}"),
            ("confidence", f"{float(self.confidence):.6g}"),
        ]
        return make_simple_card(
            title="StateSwitchEvent",
            subtitle="Detected G/C state transition",
            rows=rows,
        )


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

    def _repr_html_(self) -> str:
        rows = [
            ("time_s", f"{float(self.time):.6g}"),
            ("index", str(int(self.index))),
            ("radius_nm", f"{float(self.radius) * 1e9:.6g}"),
            ("threshold_nm", f"{float(self.threshold) * 1e9:.6g}"),
            ("duration_ns", f"{float(self.duration) * 1e9:.6g}"),
            ("confidence", f"{float(self.confidence):.6g}"),
        ]
        return make_simple_card(
            title="CoreExpulsionEvent",
            subtitle="Detected core expulsion near disk boundary",
            rows=rows,
        )


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

    def _repr_html_(self) -> str:
        rows = [
            ("state", str(self.state)),
            ("count", str(self.count)),
            ("mean_ns", f"{self.mean_dwell_time * 1e9:.6g}"),
            ("std_ns", f"{self.std_dwell_time * 1e9:.6g}"),
            ("total_ns", f"{self.total_time * 1e9:.6g}"),
            (".plt.dwell_histogram()", "Plot dwell-time distribution"),
        ]
        return make_simple_card(
            title="DwellTimeResult",
            subtitle="State dwell-time statistics",
            rows=rows,
        )


class DwellTimePlotAccessor:
    """Plot helpers for :class:`DwellTimeResult`."""

    def __init__(self, result: DwellTimeResult):
        self._result = result

    def dwell_histogram(self, *, ax=None, bins: int = 20, as_ns: bool = True, **kwargs):
        """Plot dwell-time histogram."""
        hist_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(hist_kwargs)
        figure_kwargs = pop_figure_kwargs(hist_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        values = np.asarray(self._result.dwell_times, dtype=float)
        if as_ns:
            values = values * 1e9
            xlabel = "Dwell time [ns]"
        else:
            xlabel = "Dwell time [s]"

        if values.size:
            ax.hist(values, bins=min(max(int(bins), 1), max(values.size, 1)), **hist_kwargs)
        else:
            ax.hist([], bins=1, **hist_kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"Dwell-time distribution: {self._result.state}")
        apply_axes_style(ax, style_kwargs)
        return ax
