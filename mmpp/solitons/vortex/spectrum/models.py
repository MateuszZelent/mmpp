"""Result models for vortex spectral analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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


class VortexSpectrumPlotAccessor:
    """Plot helpers for :class:`VortexSpectrumResult`."""

    def __init__(self, result: VortexSpectrumResult):
        self._result = result

    def power_spectrum(self, *, ax=None, as_ghz: bool = True, log_scale: bool = False, **kwargs):
        """Plot power spectrum."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        x = self._result.frequencies * (1e-9 if as_ghz else 1.0)
        y = np.asarray(self._result.power, dtype=float)

        if log_scale:
            y = np.log10(np.clip(y, 1e-30, None))
            ylabel = "log10(Power)"
        else:
            ylabel = "Power [a.u.]"

        ax.plot(x, y, **kwargs)
        ax.set_xlabel("Frequency [GHz]" if as_ghz else "Frequency [Hz]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Vortex {self._result.component} spectrum")
        return ax


class VortexSpectrogramPlotAccessor:
    """Plot helpers for :class:`VortexSpectrogramResult`."""

    def __init__(self, result: VortexSpectrogramResult):
        self._result = result

    def spectrogram(self, *, ax=None, as_ghz: bool = True, db_scale: bool = True, **kwargs):
        """Plot time-frequency spectrogram."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        freqs = self._result.frequencies * (1e-9 if as_ghz else 1.0)
        power = np.asarray(self._result.power, dtype=float)
        if db_scale:
            power = 10.0 * np.log10(np.clip(power, 1e-30, None))

        mesh = ax.pcolormesh(self._result.times, freqs, power, shading="auto", **kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [GHz]" if as_ghz else "Frequency [Hz]")
        ax.set_title("Vortex spectrogram")
        ax.figure.colorbar(mesh, ax=ax)
        return ax
