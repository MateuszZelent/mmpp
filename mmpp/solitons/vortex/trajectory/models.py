"""Result models for vortex trajectory analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OrbitFitResult:
    """Fitted geometric description of the core orbit."""

    center: tuple[float, float]
    semi_major: float
    semi_minor: float
    eccentricity: float
    tilt_angle: float
    residual: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def radius(self) -> float:
        """Geometric-mean orbit radius."""
        return float(np.sqrt(max(self.semi_major * self.semi_minor, 0.0)))

    @property
    def is_circular(self) -> bool:
        """Heuristic circularity flag based on eccentricity."""
        return self.eccentricity < 0.1


@dataclass
class PhaseResult:
    """Phase analysis output for vortex trajectory."""

    time: np.ndarray
    phase: np.ndarray
    phase_unwrapped: np.ndarray
    omega: np.ndarray
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frequency_hz(self) -> np.ndarray:
        """Instantaneous frequency in Hz."""
        return np.asarray(self.omega, dtype=float) / (2.0 * np.pi)

    @property
    def plt(self) -> PhasePlotAccessor:
        """Plotting accessor."""
        return PhasePlotAccessor(self)


class PhasePlotAccessor:
    """Plotting namespace for :class:`PhaseResult`."""

    def __init__(self, result: PhaseResult):
        self._result = result

    def phase_portrait(self, *, ax=None, **kwargs):
        """Plot phase portrait X vs dX/dt reconstructed from phase signal."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        x_signal = np.cos(self._result.phase_unwrapped)
        dx_dt = np.gradient(x_signal, self._result.time)
        ax.plot(x_signal, dx_dt, **kwargs)
        ax.set_xlabel("cos(phi)")
        ax.set_ylabel("d(cos(phi))/dt")
        ax.set_title("Phase portrait")
        return ax

    def frequency_vs_time(self, *, ax=None, unit: str = "hz", **kwargs):
        """Plot instantaneous frequency versus time."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        unit_norm = unit.lower()
        if unit_norm in {"hz", "f"}:
            values = self._result.frequency_hz
            ylabel = "Frequency [Hz]"
        elif unit_norm in {"ghz"}:
            values = self._result.frequency_hz * 1e-9
            ylabel = "Frequency [GHz]"
        elif unit_norm in {"rad/s", "omega", "w"}:
            values = self._result.omega
            ylabel = "Angular frequency [rad/s]"
        else:
            raise ValueError("unit must be 'hz', 'ghz', or 'rad/s'")

        ax.plot(self._result.time, values, **kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title("Instantaneous frequency")
        return ax
