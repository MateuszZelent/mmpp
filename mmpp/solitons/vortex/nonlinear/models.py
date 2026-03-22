"""Result models for vortex nonlinear analysis."""

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
class AmplitudeEquationResult:
    """Complex-amplitude dynamics derived from tracked vortex orbit."""

    time: np.ndarray
    complex_amplitude: np.ndarray
    power: np.ndarray
    phase: np.ndarray
    omega: np.ndarray
    method: str
    reference_radius: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frequency_hz(self) -> np.ndarray:
        """Instantaneous frequency in Hz."""
        return np.asarray(self.omega, dtype=float) / (2.0 * np.pi)

    @property
    def plt(self) -> AmplitudePlotAccessor:
        """Plotting accessor."""
        return AmplitudePlotAccessor(self)


@dataclass
class STParametersResult:
    """Slavin-Tiberkevich parameters extracted from a single trajectory."""

    omega_0: float
    f_0_ghz: float
    N: float
    Gamma_G: float
    Q: float
    sigma: float
    I_threshold: float
    generation_power: float
    linewidth_hz: float
    quality_factor: float
    linewidth_resolution_limited: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def plt(self) -> STPlotAccessor:
        """Plotting accessor for single-point ST parameters."""
        return STPlotAccessor(self)


@dataclass
class STBatchResult:
    """Batch Slavin-Tiberkevich summary across current sweep."""

    currents: np.ndarray
    powers: np.ndarray
    linewidths: np.ndarray
    frequencies_hz: np.ndarray
    N: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frequencies_ghz(self) -> np.ndarray:
        """Dominant frequencies in GHz."""
        return np.asarray(self.frequencies_hz, dtype=float) * 1e-9

    @property
    def plt(self) -> STBatchPlotAccessor:
        """Plotting accessor for batch ST results."""
        return STBatchPlotAccessor(self)


class AmplitudePlotAccessor:
    """Plot helpers for :class:`AmplitudeEquationResult`."""

    def __init__(self, result: AmplitudeEquationResult):
        self._result = result

    def power_vs_time(self, *, ax=None, **kwargs):
        """Plot ``p(t)=|c(t)|^2``."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        ax.plot(self._result.time, self._result.power, **plot_kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Generation power p(t) [a.u.]")
        ax.set_title("Amplitude equation: power")
        apply_axes_style(ax, style_kwargs)
        return ax

    def phase_vs_time(self, *, ax=None, as_unwrapped: bool = True, **kwargs):
        """Plot trajectory phase versus time."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        if as_unwrapped:
            values = np.asarray(self._result.phase, dtype=float)
            ylabel = "Phase [rad]"
        else:
            values = np.angle(self._result.complex_amplitude)
            ylabel = "Wrapped phase [rad]"

        ax.plot(self._result.time, values, **plot_kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title("Amplitude equation: phase")
        apply_axes_style(ax, style_kwargs)
        return ax

    def complex_plane(self, *, ax=None, **kwargs):
        """Plot complex amplitude trajectory in Re-Im plane."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        c = np.asarray(self._result.complex_amplitude, dtype=np.complex128)
        ax.plot(c.real, c.imag, **plot_kwargs)
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
        ax.set_title("Complex amplitude c(t)")
        ax.set_aspect("equal")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("AmplitudePlotAccessor", [
            (".power_vs_time()", "p(t)=|c(t)|² generation power vs time",
             "Accepts matplotlib kwargs."),
            (".phase_vs_time(as_unwrapped=True)", "Phase vs time",
             "as_unwrapped: True for cumulative, False for wrapped [-π,π]."),
            (".complex_plane()", "Complex amplitude c(t) in Re-Im plane",
             "Equal aspect ratio. Accepts matplotlib kwargs."),
        ])


class STPlotAccessor:
    """Plot helpers for :class:`STParametersResult`."""

    def __init__(self, result: STParametersResult):
        self._result = result

    def power_vs_current(self, *, ax=None, current_a: float | None = None, **kwargs):
        """Plot single-point generation power against current."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        if current_a is None:
            current_a = self._result.metadata.get("current_a")
        x_val = float(current_a) if current_a is not None else 0.0

        ax.plot([x_val], [self._result.generation_power], marker="o", **plot_kwargs)
        ax.set_xlabel("Current [A]" if current_a is not None else "Index")
        ax.set_ylabel("Generation power p_gen [a.u.]")
        ax.set_title("Slavin-Tiberkevich: power vs current")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("STPlotAccessor", [
            (".power_vs_current(current_a=...)",
             "Single-point generation power at given current",
             "current_a: current value in Amperes. Falls back to metadata if None."),
        ])


class STBatchPlotAccessor:
    """Plot helpers for :class:`STBatchResult`."""

    def __init__(self, result: STBatchResult):
        self._result = result

    def power_vs_current(self, *, ax=None, **kwargs):
        """Plot generation power as function of current."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        ax.plot(self._result.currents, self._result.powers, marker="o", **plot_kwargs)
        ax.set_xlabel("Current [A]")
        ax.set_ylabel("Generation power p_gen [a.u.]")
        ax.set_title("Power vs current")
        apply_axes_style(ax, style_kwargs)
        return ax

    def linewidth_vs_current(self, *, ax=None, as_mhz: bool = True, **kwargs):
        """Plot linewidth versus current."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        values = np.asarray(self._result.linewidths, dtype=float)
        ylabel = "Linewidth [Hz]"
        if as_mhz:
            values = values * 1e-6
            ylabel = "Linewidth [MHz]"

        ax.plot(self._result.currents, values, marker="o", **plot_kwargs)
        ax.set_xlabel("Current [A]")
        ax.set_ylabel(ylabel)
        ax.set_title("Linewidth vs current")
        apply_axes_style(ax, style_kwargs)
        return ax

    def frequency_vs_current(self, *, ax=None, as_ghz: bool = True, **kwargs):
        """Plot dominant gyration frequency versus current."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        values = np.asarray(self._result.frequencies_hz, dtype=float)
        ylabel = "Frequency [Hz]"
        if as_ghz:
            values = values * 1e-9
            ylabel = "Frequency [GHz]"

        ax.plot(self._result.currents, values, marker="o", **plot_kwargs)
        ax.set_xlabel("Current [A]")
        ax.set_ylabel(ylabel)
        ax.set_title("Frequency vs current")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("STBatchPlotAccessor", [
            (".power_vs_current()", "Generation power p_gen vs current I",
             "Plots full sweep. Accepts matplotlib kwargs."),
            (".linewidth_vs_current(as_mhz=True)", "Linewidth Δf vs current I",
             "as_mhz: convert to MHz."),
            (".frequency_vs_current(as_ghz=True)", "Dominant frequency f₀ vs current I",
             "as_ghz: convert to GHz."),
        ])

@dataclass
class ThieleForceBalanceResult:
    """Force decomposition from the Thiele equation on a tracked trajectory."""

    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    gyro_force: np.ndarray
    conservative_force: np.ndarray
    dissipative_force: np.ndarray
    stt_force: np.ndarray
    oersted_force: np.ndarray
    residual_force: np.ndarray
    G: float
    D: float
    kappa: float
    polarity: int
    vorticity: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def residual_norm(self) -> np.ndarray:
        """Residual-force norm over time."""
        return np.linalg.norm(np.asarray(self.residual_force, dtype=float), axis=1)

    @property
    def gyro_norm(self) -> np.ndarray:
        """Gyro-force norm over time."""
        return np.linalg.norm(np.asarray(self.gyro_force, dtype=float), axis=1)

    @property
    def residual_ratio(self) -> np.ndarray:
        """Point-wise residual ratio ``|F_res|/|F_gyro|``."""
        gyro = self.gyro_norm
        return self.residual_norm / np.clip(gyro, 1e-30, None)

    @property
    def plt(self) -> ThieleForcePlotAccessor:
        """Plotting accessor."""
        return ThieleForcePlotAccessor(self)


class ThieleForcePlotAccessor:
    """Plot helpers for :class:`ThieleForceBalanceResult`."""

    def __init__(self, result: ThieleForceBalanceResult):
        self._result = result

    def force_balance(self, *, ax=None, as_norm: bool = True, **kwargs):
        """Plot force decomposition over time."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        t = np.asarray(self._result.time, dtype=float)
        shared_kwargs = {k: v for k, v in plot_kwargs.items() if k != "label"}
        if as_norm:
            ax.plot(
                t,
                np.linalg.norm(self._result.gyro_force, axis=1),
                label="|F_gyro|",
                **shared_kwargs,
            )
            ax.plot(
                t,
                np.linalg.norm(self._result.conservative_force, axis=1),
                label="|F_cons|",
                linestyle="--",
                **{k: v for k, v in shared_kwargs.items() if k != "linestyle"},
            )
            ax.plot(
                t,
                np.linalg.norm(self._result.dissipative_force, axis=1),
                label="|F_diss|",
                linestyle=":",
                **{k: v for k, v in shared_kwargs.items() if k != "linestyle"},
            )
            ax.plot(
                t,
                np.linalg.norm(self._result.residual_force, axis=1),
                label="|F_res|",
                linewidth=1.2,
                **{k: v for k, v in shared_kwargs.items() if k != "linewidth"},
            )
            ax.set_ylabel("Force norm [a.u.]")
        else:
            ax.plot(t, self._result.gyro_force[:, 0], label="F_gyro,x", **shared_kwargs)
            ax.plot(
                t,
                self._result.gyro_force[:, 1],
                label="F_gyro,y",
                linestyle="--",
                **{k: v for k, v in shared_kwargs.items() if k != "linestyle"},
            )
            ax.plot(
                t,
                self._result.residual_force[:, 0],
                label="F_res,x",
                linestyle=":",
                **{k: v for k, v in shared_kwargs.items() if k != "linestyle"},
            )
            ax.plot(
                t,
                self._result.residual_force[:, 1],
                label="F_res,y",
                linestyle="-.",
                **{k: v for k, v in shared_kwargs.items() if k != "linestyle"},
            )
            ax.set_ylabel("Force component [a.u.]")

        ax.set_xlabel("Time [s]")
        ax.set_title("Thiele force balance")
        ax.legend()
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("ThieleForcePlotAccessor", [
            (".force_balance(as_norm=True)",
             "Force decomposition (gyro, conservative, dissipative, residual) vs time",
             "as_norm: True for |F| norms, False for x/y components."),
        ])
