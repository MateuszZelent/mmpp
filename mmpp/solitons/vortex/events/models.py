"""Result models for vortex event detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "Polarity Switch Event",
            icon="↕️",
            subtitle="Detected transition of the vortex core polarity.",
            sections=[
                metrics_section_html(
                    [
                        ("time_s", f"{float(self.time):.6g}", NODE_COLOR_COMPUTE),
                        ("index", int(self.index), None),
                        (
                            "transition",
                            f"{int(self.from_p)} -> {int(self.to_p)}",
                            NODE_COLOR_ANALYSIS,
                        ),
                        (
                            "confidence",
                            f"{float(self.confidence):.6g}",
                            NODE_COLOR_ANALYSIS,
                        ),
                    ]
                ),
                examples_section_html(
                    "event = jobs[-1].solitons.vortex.events.polarity_switches()[0]\n"
                    "event.time, event.from_p, event.to_p",
                    title="Event Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Polarity switch event API help",
                prefix="jobs[-1].solitons.vortex.events.polarity_switches()[0]",
                properties=[
                    ("time", "Event time in seconds"),
                    ("index", "Sample index"),
                    ("from_p", "Initial polarity"),
                    ("to_p", "Final polarity"),
                    ("confidence", "Detection confidence"),
                ],
                subtitle="Live attributes of a detected polarity switch.",
                chrome=False,
            ),
            uid=f"polarity-switch-event-{str(_uuid.uuid4())[:8]}",
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
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "State Switch Event",
            icon="🔄",
            subtitle="Detected transition between vortex dynamical states.",
            sections=[
                metrics_section_html(
                    [
                        ("time_s", f"{float(self.time):.6g}", NODE_COLOR_COMPUTE),
                        ("index", int(self.index), None),
                        (
                            "transition",
                            f"{self.from_state} -> {self.to_state}",
                            NODE_COLOR_ANALYSIS,
                        ),
                        (
                            "confidence",
                            f"{float(self.confidence):.6g}",
                            NODE_COLOR_ANALYSIS,
                        ),
                    ]
                ),
                examples_section_html(
                    "event = jobs[-1].solitons.vortex.events.state_switches()[0]\n"
                    "event.from_state, event.to_state",
                    title="Event Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="State switch event API help",
                prefix="jobs[-1].solitons.vortex.events.state_switches()[0]",
                properties=[
                    ("time", "Event time in seconds"),
                    ("index", "Sample index"),
                    ("from_state", "Initial state label"),
                    ("to_state", "Final state label"),
                    ("confidence", "Detection confidence"),
                ],
                subtitle="Live attributes of a detected state transition.",
                chrome=False,
            ),
            uid=f"state-switch-event-{str(_uuid.uuid4())[:8]}",
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
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "Core Expulsion Event",
            icon="🧨",
            subtitle="Detected approach to or expulsion across the disk boundary.",
            sections=[
                metrics_section_html(
                    [
                        ("time_s", f"{float(self.time):.6g}", NODE_COLOR_COMPUTE),
                        ("index", int(self.index), None),
                        (
                            "radius_nm",
                            f"{float(self.radius) * 1e9:.6g}",
                            NODE_COLOR_ANALYSIS,
                        ),
                        (
                            "threshold_nm",
                            f"{float(self.threshold) * 1e9:.6g}",
                            NODE_COLOR_ANALYSIS,
                        ),
                        (
                            "duration_ns",
                            f"{float(self.duration) * 1e9:.6g}",
                            NODE_COLOR_COMPUTE,
                        ),
                        (
                            "confidence",
                            f"{float(self.confidence):.6g}",
                            NODE_COLOR_ANALYSIS,
                        ),
                    ]
                ),
                examples_section_html(
                    "event = jobs[-1].solitons.vortex.events.core_expulsions()[0]\n"
                    "event.radius, event.duration",
                    title="Event Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Core expulsion event API help",
                prefix="jobs[-1].solitons.vortex.events.core_expulsions()[0]",
                properties=[
                    ("time", "Event time in seconds"),
                    ("index", "Sample index"),
                    ("radius", "Detected orbit radius"),
                    ("threshold", "Expulsion threshold radius"),
                    ("duration", "Event duration in seconds"),
                    ("confidence", "Detection confidence"),
                ],
                subtitle="Live attributes of a detected core expulsion event.",
                chrome=False,
            ),
            uid=f"core-expulsion-event-{str(_uuid.uuid4())[:8]}",
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
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            NODE_COLOR_PLOT,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "Dwell Time Result",
            icon="⏱️",
            subtitle="State dwell-time statistics and histogram plotting support.",
            sections=[
                metrics_section_html(
                    [
                        ("state", self.state, NODE_COLOR_ANALYSIS),
                        ("count", self.count, None),
                        (
                            "mean_ns",
                            f"{self.mean_dwell_time * 1e9:.6g}",
                            NODE_COLOR_COMPUTE,
                        ),
                        (
                            "std_ns",
                            f"{self.std_dwell_time * 1e9:.6g}",
                            NODE_COLOR_COMPUTE,
                        ),
                        (
                            "total_ns",
                            f"{self.total_time * 1e9:.6g}",
                            NODE_COLOR_ANALYSIS,
                        ),
                        (
                            "fitted_tau_ns",
                            f"{self.fitted_tau * 1e9:.6g}",
                            NODE_COLOR_PLOT,
                        ),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Plotting:",
                            [
                                (".plt.dwell_histogram(...)", NODE_COLOR_PLOT),
                            ],
                        ),
                    ]
                ),
                examples_section_html(
                    "dwell = jobs[-1].solitons.vortex.events.dwell_times(state='G-state')\n"
                    "dwell.mean_dwell_time\n"
                    "dwell.plt.dwell_histogram()",
                    title="Result Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Dwell-time result API help",
                prefix="jobs[-1].solitons.vortex.events.dwell_times()",
                properties=[
                    ("state", "Selected state label"),
                    ("dwell_times", "Array of dwell intervals in seconds"),
                    ("count", "Number of dwell intervals"),
                    ("mean_dwell_time", "Mean dwell time in seconds"),
                    ("std_dwell_time", "Standard deviation in seconds"),
                    ("total_time", "Accumulated dwell time"),
                    ("fitted_tau", "Characteristic exponential time"),
                    ("plt", "Plotting accessor"),
                ],
                subtitle="Live attributes for dwell-time statistics.",
                chrome=False,
            ),
            uid=f"dwell-time-result-{str(_uuid.uuid4())[:8]}",
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
            ax.hist(
                values, bins=min(max(int(bins), 1), max(values.size, 1)), **hist_kwargs
            )
        else:
            ax.hist([], bins=1, **hist_kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"Dwell-time distribution: {self._result.state}")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "DwellTimePlotAccessor",
            [
                (
                    ".dwell_histogram(bins=20, as_ns=True)",
                    "Dwell-time distribution histogram",
                    "bins: number of histogram bins. as_ns: convert to nanoseconds. Accepts matplotlib kwargs.",
                ),
            ],
        )
