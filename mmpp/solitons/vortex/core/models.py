"""Data models for vortex core tracking."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrajectoryResult:
    """Result of vortex core tracking over time."""

    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    polarity: np.ndarray
    method: str
    confidence: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def z(self) -> np.ndarray:
        """Complex trajectory signal ``z(t) = (x-x0) + i(y-y0)``."""
        x_center = float(np.mean(self.x)) if self.x.size else 0.0
        y_center = float(np.mean(self.y)) if self.y.size else 0.0
        return (self.x - x_center) + 1j * (self.y - y_center)

    @property
    def r(self) -> np.ndarray:
        """Orbit radius versus time."""
        return np.abs(self.z)

    @property
    def phi(self) -> np.ndarray:
        """Instantaneous orbital angle."""
        return np.angle(self.z)

    @property
    def phi_unwrapped(self) -> np.ndarray:
        """Unwrapped orbital angle."""
        return np.unwrap(self.phi)

    @property
    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Numerical velocity components ``(vx, vy)``."""
        if self.time.size < 2:
            zeros = np.zeros_like(self.x, dtype=float)
            return zeros, zeros

        vx = np.gradient(self.x, self.time)
        vy = np.gradient(self.y, self.time)
        return np.asarray(vx, dtype=float), np.asarray(vy, dtype=float)

    @property
    def instantaneous_frequency(self) -> np.ndarray:
        """Angular frequency estimated as ``d(phi_unwrapped)/dt``."""
        if self.time.size < 2:
            return np.zeros_like(self.time, dtype=float)

        phi_unwrapped = self.phi_unwrapped
        return np.asarray(np.gradient(phi_unwrapped, self.time), dtype=float)

    @property
    def rotation_sense(self) -> str:
        """Rotation direction inferred from mean angular frequency."""
        omega = self.instantaneous_frequency
        return "CCW" if float(np.mean(omega)) >= 0.0 else "CW"

    @property
    def plt(self) -> TrajectoryPlotAccessor:
        """Plotting accessor."""
        return TrajectoryPlotAccessor(self)


class TrajectoryPlotAccessor:
    """Plotting namespace for :class:`TrajectoryResult`."""

    def __init__(self, result: TrajectoryResult):
        self._result = result

    def xy(self, *, ax=None, component: str = "both", **kwargs):
        """Plot X(t), Y(t) or both components."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        component_norm = component.lower()
        if component_norm not in {"both", "x", "y"}:
            raise ValueError("component must be one of {'both', 'x', 'y'}")

        if component_norm in {"both", "x"}:
            ax.plot(self._result.time, self._result.x, label="x", **kwargs)
        if component_norm in {"both", "y"}:
            y_kwargs = dict(kwargs)
            if component_norm == "both" and "linestyle" not in y_kwargs:
                y_kwargs["linestyle"] = "--"
            ax.plot(self._result.time, self._result.y, label="y", **y_kwargs)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Core position [m]")
        if component_norm == "both":
            ax.legend()
        return ax

    def orbit_2d(self, *, ax=None, show_center: bool = True, **kwargs):
        """Plot orbit trajectory in XY plane."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self._result.x, self._result.y, **kwargs)
        if show_center:
            ax.scatter(
                [float(np.mean(self._result.x))],
                [float(np.mean(self._result.y))],
                color="red",
                s=20,
                label="center",
            )
            ax.legend()

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Core orbit (2D)")
        ax.set_aspect("equal")
        return ax

    def overview(self, *, fig=None):
        """Create compact overview panel for trajectory diagnostics."""
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure(figsize=(10, 8))
        axes = fig.subplots(2, 2)

        self.xy(ax=axes[0, 0])
        axes[0, 0].set_title("X/Y vs time")

        self.orbit_2d(ax=axes[0, 1])

        axes[1, 0].plot(self._result.time, self._result.r)
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].set_ylabel("r [m]")
        axes[1, 0].set_title("Orbit radius")

        omega_hz = self._result.instantaneous_frequency / (2.0 * np.pi)
        axes[1, 1].plot(self._result.time, omega_hz)
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("Frequency [Hz]")
        axes[1, 1].set_title("Instantaneous frequency")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
            fig.tight_layout()
        return fig
