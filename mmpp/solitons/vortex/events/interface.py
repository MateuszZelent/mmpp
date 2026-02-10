"""High-level event-detection interface for vortex dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

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


class EventsInterface:
    """Event detection namespace (polarity, state transitions, expulsion, dwell time)."""

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

    def _resolve_trajectory(self, trajectory: TrajectoryResult | None = None) -> TrajectoryResult:
        if trajectory is not None:
            return trajectory
        return self._core.track()

    def _infer_disk_radius(self) -> float:
        attrs = getattr(self._job, "attrs", {})
        dx = float(attrs.get("dx", 1.0e-9))
        dy = float(attrs.get("dy", dx))

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

        radius = self._infer_disk_radius() if disk_radius is None else float(disk_radius)
        center_xy = self._infer_disk_center() if center is None else (float(center[0]), float(center[1]))
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


class EventsPlotAccessor:
    """Plotting facade for :class:`EventsInterface`."""

    def __init__(self, interface: EventsInterface):
        self._interface = interface

    def event_timeline(self, *, trajectory: TrajectoryResult | None = None, ax=None):
        """Plot trajectory with event markers."""
        import matplotlib.pyplot as plt

        traj = self._interface._resolve_trajectory(trajectory)
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(traj.time, traj.x, label="x(t)", color="#1f77b4")
        ax.plot(traj.time, traj.y, label="y(t)", color="#ff7f0e", linestyle="--")

        for event in self._interface.polarity_switches(trajectory=traj):
            ax.axvline(event.time, color="red", linestyle=":", alpha=0.6)
        for event in self._interface.state_switches(trajectory=traj):
            ax.axvline(event.time, color="green", linestyle="-.", alpha=0.5)
        for event in self._interface.core_expulsions(trajectory=traj):
            ax.axvline(event.time, color="purple", linestyle="--", alpha=0.5)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax.set_title("Event timeline")
        ax.legend()
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
