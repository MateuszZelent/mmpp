"""Static plotting helpers for trajectory-centric analysis results."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from ..._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)

if TYPE_CHECKING:
    from ..models import TrajectoryResult


class TrajectoryPlotAccessor:
    """Plotting namespace for :class:`TrajectoryResult`."""

    def __init__(self, result: "TrajectoryResult"):
        self._result = result

    def xy(self, *, ax=None, component: str = "both", **kwargs):
        """Plot X(t), Y(t), or a single selected component."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        component_norm = str(component).lower()
        if component_norm not in {"both", "x", "y"}:
            raise ValueError("component must be one of {'both', 'x', 'y'}")

        if component_norm in {"both", "x"}:
            ax.plot(self._result.time, self._result.x, label="x", **plot_kwargs)
        if component_norm in {"both", "y"}:
            y_kwargs = dict(plot_kwargs)
            if component_norm == "both" and "linestyle" not in y_kwargs:
                y_kwargs["linestyle"] = "--"
            ax.plot(self._result.time, self._result.y, label="y", **y_kwargs)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Core position [m]")
        if component_norm == "both":
            ax.legend()
        apply_axes_style(ax, style_kwargs)
        return ax

    def orbit_2d(self, *, ax=None, show_center: bool = True, **kwargs):
        """Plot orbit trajectory in XY plane."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        ax.plot(self._result.x, self._result.y, **plot_kwargs)
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
        apply_axes_style(ax, style_kwargs)
        return ax

    def overview(self, *, fig=None):
        """Create compact 2x2 trajectory diagnostics panel."""
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

    def interactive(self, **kwargs):
        """Open interactive orbit/snapshot viewer with matplotlib controls."""
        from .interactive import trajectory_interactive

        return trajectory_interactive(self._result, **kwargs)

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("TrajectoryPlotAccessor", [
            (".xy(component='both')",
             "Plot X(t) and/or Y(t) core position",
             "component: 'both', 'x', or 'y'. Accepts matplotlib kwargs."),
            (".orbit_2d(show_center=True)",
             "2-D orbit trajectory in XY plane",
             "show_center: mark mean position. Aspect ratio forced equal."),
            (".overview()",
             "Compact 2×2 diagnostics panel",
             "X/Y vs time, orbit, radius, instantaneous frequency."),
            (".interactive()",
             "Interactive orbit/snapshot viewer",
             "matplotlib-based interactive controls."),
        ])
