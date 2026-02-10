"""High-level spectral analysis interface for vortex dynamics."""

from __future__ import annotations

from typing import Any

from ..config import VortexConfig
from .gyration import compute_breathing_spectrum, compute_gyration_spectrum
from .models import VortexSpectrogramResult, VortexSpectrumResult
from .spectrogram import compute_spectrogram


class VortexSpectrumInterface:
    """Spectrum analysis namespace."""

    def __init__(
        self,
        job_result,
        dataset_name: str | None,
        slice_info: Any | None,
        config: VortexConfig,
        core_interface,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config
        self._core = core_interface
        self._last_gyration: VortexSpectrumResult | None = None
        self._last_spectrogram: VortexSpectrogramResult | None = None

    def gyration(self, method: str | None = None, **kwargs) -> VortexSpectrumResult:
        """Compute gyration power spectrum from core trajectory."""
        trajectory = kwargs.pop("trajectory", None)
        if trajectory is None:
            trajectory = self._core.track()

        selected_method = method or self._config.spectrum.method
        if "nperseg" not in kwargs and self._config.spectrum.nperseg is not None:
            kwargs["nperseg"] = self._config.spectrum.nperseg
        if "noverlap" not in kwargs and self._config.spectrum.noverlap is not None:
            kwargs["noverlap"] = self._config.spectrum.noverlap

        result = compute_gyration_spectrum(trajectory, method=selected_method, **kwargs)
        self._last_gyration = result
        return result

    def breathing(self, method: str | None = None, **kwargs) -> VortexSpectrumResult:
        """Compute breathing-mode spectrum from orbit radius signal."""
        trajectory = kwargs.pop("trajectory", None)
        if trajectory is None:
            trajectory = self._core.track()

        selected_method = method or self._config.spectrum.method
        if "nperseg" not in kwargs and self._config.spectrum.nperseg is not None:
            kwargs["nperseg"] = self._config.spectrum.nperseg
        if "noverlap" not in kwargs and self._config.spectrum.noverlap is not None:
            kwargs["noverlap"] = self._config.spectrum.noverlap

        return compute_breathing_spectrum(trajectory, method=selected_method, **kwargs)

    def spectrogram(self, component: str = "radius", **kwargs) -> VortexSpectrogramResult:
        """Compute trajectory spectrogram."""
        trajectory = kwargs.pop("trajectory", None)
        if trajectory is None:
            trajectory = self._core.track()

        if "nperseg" not in kwargs and self._config.spectrum.nperseg is not None:
            kwargs["nperseg"] = self._config.spectrum.nperseg
        if "noverlap" not in kwargs and self._config.spectrum.noverlap is not None:
            kwargs["noverlap"] = self._config.spectrum.noverlap

        result = compute_spectrogram(trajectory, component=component, **kwargs)
        self._last_spectrogram = result
        return result

    @property
    def plt(self):
        """Convenience plotting namespace."""
        return SpectrumInterfacePlotAccessor(self)


class SpectrumInterfacePlotAccessor:
    """Plotting facade for :class:`VortexSpectrumInterface`."""

    def __init__(self, interface: VortexSpectrumInterface):
        self._interface = interface

    def power_spectrum(self, **kwargs):
        """Compute and plot gyration power spectrum."""
        result = self._interface.gyration()
        return result.plt.power_spectrum(**kwargs)

    def spectrogram(self, **kwargs):
        """Compute and plot spectrogram."""
        result = self._interface.spectrogram()
        return result.plt.spectrogram(**kwargs)
