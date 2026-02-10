"""Result models for vortex nonlinear analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self._result.time, self._result.power, **kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Generation power p(t) [a.u.]")
        ax.set_title("Amplitude equation: power")
        return ax

    def phase_vs_time(self, *, ax=None, as_unwrapped: bool = True, **kwargs):
        """Plot trajectory phase versus time."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        if as_unwrapped:
            values = np.asarray(self._result.phase, dtype=float)
            ylabel = "Phase [rad]"
        else:
            values = np.angle(self._result.complex_amplitude)
            ylabel = "Wrapped phase [rad]"

        ax.plot(self._result.time, values, **kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title("Amplitude equation: phase")
        return ax

    def complex_plane(self, *, ax=None, **kwargs):
        """Plot complex amplitude trajectory in Re-Im plane."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        c = np.asarray(self._result.complex_amplitude, dtype=np.complex128)
        ax.plot(c.real, c.imag, **kwargs)
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
        ax.set_title("Complex amplitude c(t)")
        ax.set_aspect("equal")
        return ax


class STPlotAccessor:
    """Plot helpers for :class:`STParametersResult`."""

    def __init__(self, result: STParametersResult):
        self._result = result

    def power_vs_current(self, *, ax=None, current_a: float | None = None, **kwargs):
        """Plot single-point generation power against current."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        if current_a is None:
            current_a = self._result.metadata.get("current_a")
        x_val = float(current_a) if current_a is not None else 0.0

        ax.plot([x_val], [self._result.generation_power], marker="o", **kwargs)
        ax.set_xlabel("Current [A]" if current_a is not None else "Index")
        ax.set_ylabel("Generation power p_gen [a.u.]")
        ax.set_title("Slavin-Tiberkevich: power vs current")
        return ax


class STBatchPlotAccessor:
    """Plot helpers for :class:`STBatchResult`."""

    def __init__(self, result: STBatchResult):
        self._result = result

    def power_vs_current(self, *, ax=None, **kwargs):
        """Plot generation power as function of current."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self._result.currents, self._result.powers, marker="o", **kwargs)
        ax.set_xlabel("Current [A]")
        ax.set_ylabel("Generation power p_gen [a.u.]")
        ax.set_title("Power vs current")
        return ax

    def linewidth_vs_current(self, *, ax=None, as_mhz: bool = True, **kwargs):
        """Plot linewidth versus current."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        values = np.asarray(self._result.linewidths, dtype=float)
        ylabel = "Linewidth [Hz]"
        if as_mhz:
            values = values * 1e-6
            ylabel = "Linewidth [MHz]"

        ax.plot(self._result.currents, values, marker="o", **kwargs)
        ax.set_xlabel("Current [A]")
        ax.set_ylabel(ylabel)
        ax.set_title("Linewidth vs current")
        return ax

    def frequency_vs_current(self, *, ax=None, as_ghz: bool = True, **kwargs):
        """Plot dominant gyration frequency versus current."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        values = np.asarray(self._result.frequencies_hz, dtype=float)
        ylabel = "Frequency [Hz]"
        if as_ghz:
            values = values * 1e-9
            ylabel = "Frequency [GHz]"

        ax.plot(self._result.currents, values, marker="o", **kwargs)
        ax.set_xlabel("Current [A]")
        ax.set_ylabel(ylabel)
        ax.set_title("Frequency vs current")
        return ax


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
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        t = np.asarray(self._result.time, dtype=float)
        if as_norm:
            ax.plot(t, np.linalg.norm(self._result.gyro_force, axis=1), label="|F_gyro|", **kwargs)
            ax.plot(
                t,
                np.linalg.norm(self._result.conservative_force, axis=1),
                label="|F_cons|",
                linestyle="--",
            )
            ax.plot(
                t,
                np.linalg.norm(self._result.dissipative_force, axis=1),
                label="|F_diss|",
                linestyle=":",
            )
            ax.plot(
                t,
                np.linalg.norm(self._result.residual_force, axis=1),
                label="|F_res|",
                linewidth=1.2,
            )
            ax.set_ylabel("Force norm [a.u.]")
        else:
            ax.plot(t, self._result.gyro_force[:, 0], label="F_gyro,x", **kwargs)
            ax.plot(t, self._result.gyro_force[:, 1], label="F_gyro,y", linestyle="--")
            ax.plot(t, self._result.residual_force[:, 0], label="F_res,x", linestyle=":")
            ax.plot(t, self._result.residual_force[:, 1], label="F_res,y", linestyle="-.")
            ax.set_ylabel("Force component [a.u.]")

        ax.set_xlabel("Time [s]")
        ax.set_title("Thiele force balance")
        ax.legend()
        return ax
