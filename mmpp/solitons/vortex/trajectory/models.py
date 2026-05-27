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
            "Orbit Fit Result",
            icon="⭕",
            subtitle="Geometric ellipse fit of the tracked vortex orbit.",
            sections=[
                metrics_section_html(
                    [
                        ("radius", f"{self.radius:.6g}", NODE_COLOR_PLOT),
                        (
                            "semi_major",
                            f"{float(self.semi_major):.6g}",
                            NODE_COLOR_COMPUTE,
                        ),
                        (
                            "semi_minor",
                            f"{float(self.semi_minor):.6g}",
                            NODE_COLOR_COMPUTE,
                        ),
                        (
                            "eccentricity",
                            f"{float(self.eccentricity):.6g}",
                            NODE_COLOR_ANALYSIS,
                        ),
                        (
                            "residual",
                            f"{float(self.residual):.6g}",
                            NODE_COLOR_ANALYSIS,
                        ),
                        ("is_circular", self.is_circular, None),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Geometry:",
                            [
                                (".center", NODE_COLOR_COMPUTE),
                                (".tilt_angle", NODE_COLOR_COMPUTE),
                                (".radius", NODE_COLOR_PLOT),
                            ],
                        ),
                    ]
                ),
                examples_section_html(
                    "fit = jobs[-1].solitons.vortex.trajectory.orbit.fit()\n"
                    "fit.radius\n"
                    "fit.eccentricity",
                    title="Fit Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Orbit fit result API help",
                prefix="jobs[-1].solitons.vortex.trajectory.orbit.fit()",
                properties=[
                    ("center", "Ellipse center (x, y)"),
                    ("semi_major", "Semi-major axis"),
                    ("semi_minor", "Semi-minor axis"),
                    ("eccentricity", "Orbit eccentricity"),
                    ("tilt_angle", "Ellipse tilt angle"),
                    ("residual", "Fit residual"),
                    ("radius", "Geometric-mean orbit radius"),
                    ("is_circular", "Heuristic circularity flag"),
                ],
                subtitle="Live attributes of the fitted vortex orbit.",
                chrome=False,
            ),
            uid=f"orbit-fit-result-{str(_uuid.uuid4())[:8]}",
        )


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

        return plot_accessor_html(
            "PhasePlotAccessor",
            [
                (
                    ".phase_portrait()",
                    "Phase portrait cos(φ) vs d(cos(φ))/dt",
                    "Reconstructs phase-space trajectory.",
                ),
                (
                    ".frequency_vs_time(unit='hz')",
                    "Instantaneous frequency vs time",
                    "unit: 'hz', 'ghz', or 'rad/s'.",
                ),
            ],
        )
