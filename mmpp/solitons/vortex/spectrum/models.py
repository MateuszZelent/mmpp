"""Result models for vortex spectral analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmpp._shared.repr_html import make_simple_card

from .._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)


@dataclass
class VortexSpectrumResult:
    """Power spectrum of vortex trajectory dynamics."""

    frequencies: np.ndarray
    power: np.ndarray
    method: str
    component: str = "gyration"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def amplitude(self) -> np.ndarray:
        """Amplitude spectrum."""
        return np.sqrt(np.clip(np.asarray(self.power, dtype=float), 0.0, None))

    @property
    def peak_frequency_hz(self) -> float:
        """Dominant spectral peak frequency in Hz."""
        if self.frequencies.size == 0:
            return float("nan")
        idx = int(np.argmax(self.power))
        return float(self.frequencies[idx])

    @property
    def peak_frequency_ghz(self) -> float:
        """Dominant spectral peak frequency in GHz."""
        return self.peak_frequency_hz * 1e-9

    @property
    def plt(self) -> VortexSpectrumPlotAccessor:
        """Plotting accessor."""
        return VortexSpectrumPlotAccessor(self)

    def _repr_html_(self) -> str:
        return make_simple_card(
            title=self.__class__.__name__,
            subtitle=f"{self.component} spectrum",
            rows=[
                ("method", str(self.method)),
                ("n_freq", str(int(np.asarray(self.frequencies).size))),
                ("peak_ghz", f"{self.peak_frequency_ghz:.6g}"),
            ],
        )


@dataclass
class VortexSpectrogramResult:
    """Time-frequency representation of vortex dynamics."""

    times: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray
    method: str
    component: str = "radius"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def plt(self) -> VortexSpectrogramPlotAccessor:
        """Plotting accessor."""
        return VortexSpectrogramPlotAccessor(self)

    def _repr_html_(self) -> str:
        return make_simple_card(
            title=self.__class__.__name__,
            subtitle=f"{self.component} spectrogram",
            rows=[
                ("method", str(self.method)),
                ("n_times", str(int(np.asarray(self.times).size))),
                ("n_freq", str(int(np.asarray(self.frequencies).size))),
            ],
        )


class VortexSpectrumPlotAccessor:
    """Plot helpers for :class:`VortexSpectrumResult`."""

    def __init__(self, result: VortexSpectrumResult):
        self._result = result

    def power_spectrum(
        self, *, ax=None, as_ghz: bool = True, log_scale: bool = False, **kwargs
    ):
        """Plot power spectrum."""
        plot_kwargs = dict(kwargs)
        save = plot_kwargs.pop("save", None)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        x = self._result.frequencies * (1e-9 if as_ghz else 1.0)
        y = np.asarray(self._result.power, dtype=float)

        if log_scale:
            y = np.log10(np.clip(y, 1e-30, None))
            ylabel = "log10(Power)"
        else:
            ylabel = "Power [a.u.]"

        ax.plot(x, y, **plot_kwargs)
        ax.set_xlabel("Frequency [GHz]" if as_ghz else "Frequency [Hz]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Vortex {self._result.component} spectrum")
        apply_axes_style(ax, style_kwargs)
        if save is not None:
            ax.figure.savefig(save)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "VortexSpectrumPlotAccessor",
            [
                (
                    ".power_spectrum(as_ghz=True, log_scale=False)",
                    "Power spectrum of vortex gyration",
                    "as_ghz: frequency in GHz. log_scale: log10 power axis. Accepts matplotlib kwargs.",
                ),
            ],
        )


class VortexSpectrogramPlotAccessor:
    """Plot helpers for :class:`VortexSpectrogramResult`."""

    def __init__(self, result: VortexSpectrogramResult):
        self._result = result

    def spectrogram(
        self, *, ax=None, as_ghz: bool = True, db_scale: bool = True, **kwargs
    ):
        """Plot time-frequency spectrogram."""
        mesh_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(mesh_kwargs)
        figure_kwargs = pop_figure_kwargs(mesh_kwargs)
        colorbar = bool(mesh_kwargs.pop("colorbar", True))
        colorbar_options = mesh_kwargs.pop("colorbar_kwargs", {})
        colorbar_kwargs = {} if colorbar_options is None else dict(colorbar_options)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        freqs = self._result.frequencies * (1e-9 if as_ghz else 1.0)
        power = np.asarray(self._result.power, dtype=float)
        if db_scale:
            power = 10.0 * np.log10(np.clip(power, 1e-30, None))

        mesh = ax.pcolormesh(
            self._result.times, freqs, power, shading="auto", **mesh_kwargs
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [GHz]" if as_ghz else "Frequency [Hz]")
        ax.set_title("Vortex spectrogram")
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, **colorbar_kwargs)
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "VortexSpectrogramPlotAccessor",
            [
                (
                    ".spectrogram(as_ghz=True, db_scale=True)",
                    "Time-frequency spectrogram of vortex dynamics",
                    "as_ghz: frequency in GHz. db_scale: 10*log10 power. colorbar, colorbar_kwargs.",
                ),
            ],
        )
