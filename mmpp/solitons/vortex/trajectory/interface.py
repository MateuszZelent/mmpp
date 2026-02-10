"""High-level trajectory analysis interface."""

from __future__ import annotations

from typing import Any

from ..config import VortexConfig
from .filtering import filter_trajectory
from .orbit import OrbitInterface
from .phase import PhaseAnalyzer
from .steady_state import extract_steady_state


class TrajectoryInterface:
    """Orbit and phase analysis namespace."""

    def __init__(
        self,
        job_result,
        dataset_name: str | None,
        slice_info: Any | None,
        config: VortexConfig,
        core_interface,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config
        self._core = core_interface
        self._orbit = None

    @property
    def raw(self):
        """Raw tracked trajectory (computed lazily)."""
        return self._core.track()

    def filtered(self, method: str | None = None, **kwargs):
        """Return filtered trajectory."""
        selected_method = method or self._config.trajectory.filter_method
        if "window" not in kwargs:
            kwargs["window"] = self._config.trajectory.filter_window
        return filter_trajectory(self.raw, method=selected_method, **kwargs)

    def steady_state(self, threshold: float | None = None, **kwargs):
        """Return steady-state portion of the trajectory."""
        selected_threshold = (
            self._config.trajectory.steady_state_threshold
            if threshold is None
            else float(threshold)
        )
        if "window" not in kwargs:
            kwargs["window"] = self._config.trajectory.steady_state_window
        return extract_steady_state(self.raw, threshold=selected_threshold, **kwargs)

    @property
    def orbit(self) -> OrbitInterface:
        """Orbit fitting namespace."""
        if self._orbit is None:
            self._orbit = OrbitInterface(self)
        return self._orbit

    @property
    def phase(self) -> PhaseAnalyzer:
        """Phase analysis namespace."""
        return PhaseAnalyzer(self.raw)
