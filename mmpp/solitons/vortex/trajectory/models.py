"""Result models for vortex trajectory analysis."""

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
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        x_signal = np.cos(self._result.phase_unwrapped)
        dx_dt = np.gradient(x_signal, self._result.time)
        ax.plot(x_signal, dx_dt, **plot_kwargs)
        ax.set_xlabel("cos(phi)")
        ax.set_ylabel("d(cos(phi))/dt")
        ax.set_title("Phase portrait")
        apply_axes_style(ax, style_kwargs)
        return ax

    def frequency_vs_time(self, *, ax=None, unit: str = "hz", **kwargs):
        """Plot instantaneous frequency versus time."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

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

        ax.plot(self._result.time, values, **plot_kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title("Instantaneous frequency")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("PhasePlotAccessor", [
            (".phase_portrait()",
             "Phase portrait cos(φ) vs d(cos(φ))/dt",
             "Reconstructs phase-space trajectory."),
            (".frequency_vs_time(unit='hz')",
             "Instantaneous frequency vs time",
             "unit: 'hz', 'ghz', or 'rad/s'."),
        ])
