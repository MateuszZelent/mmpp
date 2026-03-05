"""Result models for vortex electrical-signal post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)
from mmpp._shared.repr_html import make_simple_card


@dataclass
class MagnetoresistanceResult:
    """Magnetoresistance (or TMR proxy) trace."""

    time: np.ndarray
    resistance_ohm: np.ndarray
    projection: np.ndarray
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_resistance_ohm(self) -> float:
        """Mean resistance over the full trace."""
        values = np.asarray(self.resistance_ohm, dtype=float)
        return float(np.mean(values)) if values.size else float("nan")

    @property
    def plt(self) -> MagnetoresistancePlotAccessor:
        """Plotting accessor."""
        return MagnetoresistancePlotAccessor(self)

    def _repr_html_(self) -> str:
        n = int(np.asarray(self.time).size)
        rows = [
            ("samples", str(n)),
            ("method", str(self.method)),
            ("mean_resistance_ohm", f"{self.mean_resistance_ohm:.6g}"),
            ("peak_to_peak_ohm", f"{(np.ptp(self.resistance_ohm) if n else float('nan')):.6g}"),
            (".plt.time_trace()", "Plot R(t)"),
        ]
        return make_simple_card(
            title="MagnetoresistanceResult",
            subtitle="Reconstructed resistance trace",
            rows=rows,
        )


@dataclass
class VoltageResult:
    """Voltage trace reconstructed as ``V(t)=I(t)R(t)``."""

    time: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    resistance_ohm: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rms_voltage_v(self) -> float:
        """Root-mean-square voltage."""
        values = np.asarray(self.voltage_v, dtype=float)
        return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")

    @property
    def plt(self) -> VoltagePlotAccessor:
        """Plotting accessor."""
        return VoltagePlotAccessor(self)

    def _repr_html_(self) -> str:
        n = int(np.asarray(self.time).size)
        rows = [
            ("samples", str(n)),
            ("rms_voltage_v", f"{self.rms_voltage_v:.6g}"),
            (
                "current_mean_a",
                f"{(np.mean(self.current_a) if n else float('nan')):.6g}",
            ),
            (".plt.time_trace()", "Plot V(t)"),
        ]
        return make_simple_card(
            title="VoltageResult",
            subtitle="Voltage reconstructed from current and resistance",
            rows=rows,
        )


@dataclass
class SignalSpectrumResult:
    """One-sided power spectrum of a scalar electrical trace."""

    frequencies_hz: np.ndarray
    power: np.ndarray
    quantity: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def peak_frequency_hz(self) -> float:
        """Frequency at the dominant spectral peak."""
        f = np.asarray(self.frequencies_hz, dtype=float)
        p = np.asarray(self.power, dtype=float)
        if f.size == 0 or p.size == 0:
            return float("nan")
        return float(f[int(np.argmax(p))])

    @property
    def peak_frequency_ghz(self) -> float:
        """Peak frequency in GHz."""
        value = self.peak_frequency_hz
        return float(value * 1e-9) if np.isfinite(value) else float("nan")

    @property
    def plt(self) -> SignalSpectrumPlotAccessor:
        """Plotting accessor."""
        return SignalSpectrumPlotAccessor(self)

    def _repr_html_(self) -> str:
        n = int(np.asarray(self.frequencies_hz).size)
        rows = [
            ("samples", str(n)),
            ("quantity", str(self.quantity)),
            ("peak_frequency_ghz", f"{self.peak_frequency_ghz:.6g}"),
            ("method", str(self.metadata.get("method", "unknown"))),
            (".plt.power_spectrum()", "Plot one-sided PSD"),
        ]
        return make_simple_card(
            title="SignalSpectrumResult",
            subtitle="Electrical signal power spectrum",
            rows=rows,
        )


class MagnetoresistancePlotAccessor:
    """Plot helpers for :class:`MagnetoresistanceResult`."""

    def __init__(self, result: MagnetoresistanceResult):
        self._result = result

    def time_trace(self, *, ax=None, as_mohm: bool = False, **kwargs):
        """Plot resistance versus time."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        values = np.asarray(self._result.resistance_ohm, dtype=float)
        ylabel = "Resistance [Ohm]"
        if as_mohm:
            values = values * 1e3
            ylabel = "Resistance [mOhm]"

        ax.plot(self._result.time, values, **plot_kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title("Magnetoresistance")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("MagnetoresistancePlotAccessor", [
            (".time_trace(as_mohm=False)",
             "Resistance R(t) vs time",
             "as_mohm: convert to mOhm. Accepts matplotlib kwargs."),
        ])


class VoltagePlotAccessor:
    """Plot helpers for :class:`VoltageResult`."""

    def __init__(self, result: VoltageResult):
        self._result = result

    def time_trace(self, *, ax=None, as_mv: bool = False, **kwargs):
        """Plot voltage versus time."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        values = np.asarray(self._result.voltage_v, dtype=float)
        ylabel = "Voltage [V]"
        if as_mv:
            values = values * 1e3
            ylabel = "Voltage [mV]"

        ax.plot(self._result.time, values, **plot_kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title("Voltage trace")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("VoltagePlotAccessor", [
            (".time_trace(as_mv=False)",
             "Voltage V(t) vs time",
             "as_mv: convert to mV. Accepts matplotlib kwargs."),
        ])


class SignalSpectrumPlotAccessor:
    """Plot helpers for :class:`SignalSpectrumResult`."""

    def __init__(self, result: SignalSpectrumResult):
        self._result = result

    def power_spectrum(self, *, ax=None, as_ghz: bool = True, **kwargs):
        """Plot one-sided power spectrum."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        freq = np.asarray(self._result.frequencies_hz, dtype=float)
        if as_ghz:
            freq = freq * 1e-9
            xlabel = "Frequency [GHz]"
        else:
            xlabel = "Frequency [Hz]"

        ax.plot(freq, self._result.power, **plot_kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Power [a.u.]")
        ax.set_title(f"Signal spectrum ({self._result.quantity})")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("SignalSpectrumPlotAccessor", [
            (".power_spectrum(as_ghz=True)",
             "One-sided power spectrum",
             "as_ghz: frequency in GHz. Accepts matplotlib kwargs."),
        ])


__all__ = [
    "MagnetoresistanceResult",
    "VoltageResult",
    "SignalSpectrumResult",
]
