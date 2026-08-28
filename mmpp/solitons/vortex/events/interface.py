"""High-level event-detection interface for vortex dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..._method_helpers import InteractiveNodeMixin
from .._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)
from ..config import VortexConfig
from ..core.models import TrajectoryResult
from .core_expulsion import detect_core_expulsions
from .dwell_time import dwell_time_statistics
from .models import (
    CoreExpulsionEvent,
    DwellTimeResult,
    PolaritySwitchEvent,
    StateSwitchEvent,
)
from .polarity import detect_polarity_switches
from .state_transitions import detect_state_switches


class EventsInterface(InteractiveNodeMixin):
    """Event detection namespace (polarity, state transitions, expulsion, dwell time)."""

    _interactive_owner = "job[0].vortex.events"
    _interactive_nodes = frozenset(
        {"polarity_switches", "state_switches", "core_expulsions", "dwell_times"}
    )
    _interactive_examples = {
        "polarity_switches": ["events = job[0].vortex.events.polarity_switches()"],
        "state_switches": ["events = job[0].vortex.events.state_switches()"],
        "core_expulsions": ["events = job[0].vortex.events.core_expulsions()"],
        "dwell_times": ["dwell = job[0].vortex.events.dwell_times()"],
    }

    def __init__(
        self,
        job_result,
        dataset_name: str | None,
        slice_info: Any | None,
        config: VortexConfig,
        core_interface,
        trajectory_interface,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config
        self._core = core_interface
        self._trajectory = trajectory_interface

        self._last_polarity: list[PolaritySwitchEvent] | None = None
        self._last_states: tuple[list[StateSwitchEvent], np.ndarray] | None = None
        self._last_expulsions: list[CoreExpulsionEvent] | None = None

    def _resolve_trajectory(
        self, trajectory: TrajectoryResult | None = None
    ) -> TrajectoryResult:
        if trajectory is not None:
            return trajectory
        return self._core.track()

    def _infer_disk_radius(self) -> float:
        attrs = getattr(self._job, "attrs", {})
        dx = float(attrs.get("dx", 1.0e-9))
        dy = float(attrs.get("dy", dx))

        for key in ("D", "diameter", "disk_diameter", "pillar_diameter"):
            if key in attrs:
                try:
                    diameter = float(attrs[key])
                except Exception:
                    continue
                if diameter > 0.0:
                    return 0.5 * diameter

        if self._dataset_name is not None:
            try:
                shape = tuple(getattr(self._job[self._dataset_name], "shape", ()))
                if len(shape) >= 3 and shape[-1] == 3:
                    nx = float(shape[-2])
                    ny = float(shape[-3])
                    return 0.45 * min(nx * dx, ny * dy)
            except Exception:
                pass
        return 50.0e-9

    def _infer_disk_center(self) -> tuple[float, float]:
        attrs = getattr(self._job, "attrs", {})
        dx = float(attrs.get("dx", 1.0e-9))
        dy = float(attrs.get("dy", dx))

        if self._dataset_name is not None:
            try:
                shape = tuple(getattr(self._job[self._dataset_name], "shape", ()))
                if len(shape) >= 3 and shape[-1] == 3:
                    nx = float(shape[-2])
                    ny = float(shape[-3])
                    return ((nx - 1.0) * 0.5 * dx, (ny - 1.0) * 0.5 * dy)
            except Exception:
                pass
        return (0.0, 0.0)

    def polarity_switches(
        self,
        *,
        trajectory: TrajectoryResult | None = None,
        threshold: float = 0.5,
        refractory: float = 0.5e-9,
        force: bool = False,
    ) -> list[PolaritySwitchEvent]:
        """Detect polarity switches ``p=+1 <-> -1``."""
        if not force and self._last_polarity is not None and trajectory is None:
            return self._last_polarity

        result = detect_polarity_switches(
            self._resolve_trajectory(trajectory),
            threshold=threshold,
            refractory=refractory,
        )
        if trajectory is None:
            self._last_polarity = result
        return result

    def state_switches(
        self,
        *,
        trajectory: TrajectoryResult | None = None,
        radius_threshold: float = 0.6,
        min_dwell_periods: int = 3,
        refractory: float = 0.5e-9,
        smoothing_window: int = 9,
        force: bool = False,
    ) -> list[StateSwitchEvent]:
        """Detect G/C state transitions."""
        if not force and self._last_states is not None and trajectory is None:
            return self._last_states[0]

        events, labels = detect_state_switches(
            self._resolve_trajectory(trajectory),
            radius_threshold=radius_threshold,
            min_dwell_periods=min_dwell_periods,
            refractory=refractory,
            smoothing_window=smoothing_window,
        )
        if trajectory is None:
            self._last_states = (events, labels)
        return events

    def core_expulsions(
        self,
        *,
        trajectory: TrajectoryResult | None = None,
        disk_radius: float | None = None,
        center: tuple[float, float] | None = None,
        expulsion_ratio: float = 0.95,
        refractory: float = 0.5e-9,
        min_duration: float = 0.0,
        force: bool = False,
    ) -> list[CoreExpulsionEvent]:
        """Detect core-expulsion events near disk boundary."""
        if (
            not force
            and self._last_expulsions is not None
            and trajectory is None
            and disk_radius is None
            and center is None
        ):
            return self._last_expulsions

        radius = (
            self._infer_disk_radius() if disk_radius is None else float(disk_radius)
        )
        center_xy = (
            self._infer_disk_center()
            if center is None
            else (float(center[0]), float(center[1]))
        )
        result = detect_core_expulsions(
            self._resolve_trajectory(trajectory),
            disk_radius=radius,
            center=center_xy,
            expulsion_ratio=expulsion_ratio,
            refractory=refractory,
            min_duration=min_duration,
        )
        if trajectory is None and disk_radius is None and center is None:
            self._last_expulsions = result
        return result

    def dwell_times(
        self,
        *,
        state: str = "G-state",
        trajectory: TrajectoryResult | None = None,
        radius_threshold: float = 0.6,
        min_dwell_periods: int = 3,
        refractory: float = 0.5e-9,
        smoothing_window: int = 9,
    ) -> DwellTimeResult:
        """Compute dwell-time statistics for selected state."""
        traj = self._resolve_trajectory(trajectory)
        if trajectory is None and self._last_states is not None:
            labels = self._last_states[1]
        else:
            _, labels = detect_state_switches(
                traj,
                radius_threshold=radius_threshold,
                min_dwell_periods=min_dwell_periods,
                refractory=refractory,
                smoothing_window=smoothing_window,
            )
        return dwell_time_statistics(traj.time, labels, state=state)

    @property
    def plt(self):
        """Plot accessor."""
        return EventsPlotAccessor(self)

    def _repr_html_(self) -> str:
        import uuid as _uuid
        from html import escape as _esc

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

        context_rows = [
            ("dataset", self._dataset_name or "auto-detect", NODE_COLOR_COMPUTE),
            (
                "slice",
                "custom" if self._slice_info is not None else "full geometry",
                None,
            ),
            ("polarity threshold", 0.5, NODE_COLOR_ANALYSIS),
            ("expulsion ratio", 0.95, NODE_COLOR_ANALYSIS),
        ]
        accessors = [
            (
                "Detect:",
                [
                    (".polarity_switches(threshold=0.5, ...)", NODE_COLOR_COMPUTE),
                    (".state_switches(radius_threshold=0.6, ...)", NODE_COLOR_COMPUTE),
                    (".core_expulsions(expulsion_ratio=0.95, ...)", NODE_COLOR_COMPUTE),
                    (".dwell_times(state='G-state', ...)", NODE_COLOR_COMPUTE),
                ],
            ),
            (
                "Plotting:",
                [
                    (".plt.event_timeline(...)", NODE_COLOR_PLOT),
                    (".plt.dwell_histogram(state='G-state', ...)", NODE_COLOR_PLOT),
                ],
            ),
        ]
        method_rows = [
            (
                "polarity_switches(...)",
                "Returns a list of PolaritySwitchEvent objects with transition sign, time, sample index, and confidence.",
            ),
            (
                "state_switches(...)",
                "Detects G-state/C-state transitions using orbit-radius criteria and dwell/refractory filters.",
            ),
            (
                "core_expulsions(...)",
                "Detects intervals where the vortex approaches or crosses the disk boundary using inferred or explicit geometry.",
            ),
            (
                "dwell_times(...)",
                "Aggregates dwell intervals into DwellTimeResult for histogramming and characteristic-time estimation.",
            ),
            (
                "plt",
                "Plot shortcuts for timeline overlays and dwell-time histograms.",
            ),
        ]
        method_body = "".join(
            "<tr>"
            f"<td style='padding:6px 8px;font-family:monospace;color:{NODE_COLOR_COMPUTE};vertical-align:top;'>{_esc(name)}</td>"
            f"<td style='padding:6px 8px;color:#f8f8f2;'>{_esc(desc)}</td>"
            "</tr>"
            for name, desc in method_rows
        )
        example = (
            "# Detect polarity switches\n"
            "switches = jobs[-1].solitons.vortex.events.polarity_switches()\n"
            "print(f'{len(switches)} polarity switches detected')\n"
            "\n"
            "# Detect state transitions\n"
            "states = jobs[-1].solitons.vortex.events.state_switches()\n"
            "\n"
            "# Core expulsion events\n"
            "expulsions = jobs[-1].solitons.vortex.events.core_expulsions()\n"
            "\n"
            "# Dwell-time statistics\n"
            "dwell = jobs[-1].solitons.vortex.events.dwell_times(state='G-state')\n"
            "\n"
            "# Plot event timeline\n"
            "jobs[-1].solitons.vortex.events.plt.event_timeline()\n"
            "jobs[-1].solitons.vortex.events.plt.dwell_histogram()"
        )
        method_html = (
            "<div style='background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%);"
            "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(98,114,164,0.35);'>"
            "<b style='color:#bd93f9;'>Event Detection Methods</b>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;margin-top:8px;'>"
            f"{method_body}</table></div>"
        )
        api_card = api_help_html(
            self,
            title="Vortex events API help",
            prefix="jobs[-1].solitons.vortex.events",
            properties=[
                ("plt", "Plot accessor for event timeline and dwell histogram")
            ],
            methods=[
                "polarity_switches",
                "state_switches",
                "core_expulsions",
                "dwell_times",
            ],
            subtitle="Live signatures for vortex event detection methods.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Events Interface",
            icon="🚨",
            subtitle="Detection of polarity flips, state transitions, core expulsions, and dwell-time statistics.",
            sections=[
                metrics_section_html(context_rows),
                accessors_section_html(accessors),
                method_html,
                examples_section_html(example, title="Event Workflows"),
            ],
            api=api_card,
            uid=f"vortex-events-{str(_uuid.uuid4())[:8]}",
        )


class EventsPlotAccessor(InteractiveNodeMixin):
    """Plotting facade for :class:`EventsInterface`."""

    _interactive_owner = "job[0].vortex.events.plt"
    _interactive_nodes = frozenset({"event_timeline", "dwell_histogram"})
    _interactive_examples = {
        "event_timeline": ["job[0].vortex.events.plt.event_timeline()"],
        "dwell_histogram": ["job[0].vortex.events.plt.dwell_histogram()"],
    }

    def __init__(self, interface: EventsInterface):
        self._interface = interface

    def event_timeline(
        self, *, trajectory: TrajectoryResult | None = None, ax=None, **kwargs
    ):
        """Plot trajectory with event markers."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        traj = self._interface._resolve_trajectory(trajectory)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        x_kwargs = dict(plot_kwargs)
        y_kwargs = dict(plot_kwargs)
        x_kwargs.pop("label", None)
        y_kwargs.pop("label", None)
        x_kwargs.setdefault("color", "#1f77b4")
        y_kwargs.setdefault("color", "#ff7f0e")
        y_kwargs.setdefault("linestyle", "--")

        ax.plot(traj.time, traj.x, label="x(t)", **x_kwargs)
        ax.plot(traj.time, traj.y, label="y(t)", **y_kwargs)

        for polarity_event in self._interface.polarity_switches(trajectory=traj):
            ax.axvline(polarity_event.time, color="red", linestyle=":", alpha=0.6)
        for state_event in self._interface.state_switches(trajectory=traj):
            ax.axvline(state_event.time, color="green", linestyle="-.", alpha=0.5)
        for expulsion_event in self._interface.core_expulsions(trajectory=traj):
            ax.axvline(expulsion_event.time, color="purple", linestyle="--", alpha=0.5)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax.set_title("Event timeline")
        ax.legend()
        apply_axes_style(ax, style_kwargs)
        return ax

    def dwell_histogram(
        self,
        *,
        state: str = "G-state",
        trajectory: TrajectoryResult | None = None,
        ax=None,
        **kwargs,
    ):
        """Plot dwell-time histogram for selected state."""
        result = self._interface.dwell_times(state=state, trajectory=trajectory)
        return result.plt.dwell_histogram(ax=ax, **kwargs)

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, node_card_html, plot_accessor_html

        html = plot_accessor_html(
            "EventsPlotAccessor",
            [
                (
                    ".event_timeline()",
                    "Trajectory + event markers (polarity, state, expulsion)",
                    "trajectory: optional pre-computed TrajectoryResult.",
                ),
                (
                    ".dwell_histogram(state='G-state')",
                    "Dwell-time distribution histogram",
                    "state: 'G-state' or 'C-state'. bins, as_ns kwargs.",
                ),
            ],
        )
        api_card = api_help_html(
            self,
            title="Vortex events plot API help",
            prefix="jobs[-1].solitons.vortex.events.plt",
            methods=["event_timeline", "dwell_histogram"],
            subtitle="Live signatures for event plotting shortcuts.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Events Plot Accessor",
            icon="🎨",
            subtitle="Plot shortcuts for event timelines and dwell-time distributions.",
            sections=[html],
            api=api_card,
            uid=f"vortex-events-plot-{str(_uuid.uuid4())[:8]}",
        )
