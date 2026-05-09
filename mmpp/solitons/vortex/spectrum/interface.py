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
        self._last_breathing: VortexSpectrumResult | None = None
        self._last_spectrogram: VortexSpectrogramResult | None = None

    def gyration(self, method: str | None = None, **kwargs) -> VortexSpectrumResult:
        """Compute gyration power spectrum from core trajectory."""
        force = bool(kwargs.pop("force", False))
        trajectory = kwargs.pop("trajectory", None)
        if not force and trajectory is None and self._last_gyration is not None:
            return self._last_gyration
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
        force = bool(kwargs.pop("force", False))
        trajectory = kwargs.pop("trajectory", None)
        if not force and trajectory is None and self._last_breathing is not None:
            return self._last_breathing
        if trajectory is None:
            trajectory = self._core.track()

        selected_method = method or self._config.spectrum.method
        if "nperseg" not in kwargs and self._config.spectrum.nperseg is not None:
            kwargs["nperseg"] = self._config.spectrum.nperseg
        if "noverlap" not in kwargs and self._config.spectrum.noverlap is not None:
            kwargs["noverlap"] = self._config.spectrum.noverlap

        result = compute_breathing_spectrum(
            trajectory, method=selected_method, **kwargs
        )
        self._last_breathing = result
        return result

    def spectrogram(
        self, component: str = "radius", **kwargs
    ) -> VortexSpectrogramResult:
        """Compute trajectory spectrogram."""
        force = bool(kwargs.pop("force", False))
        trajectory = kwargs.pop("trajectory", None)
        if not force and trajectory is None and self._last_spectrogram is not None:
            return self._last_spectrogram
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

    def _repr_html_(self) -> str:
        from html import escape as _esc

        methods = [
            (".gyration(method=...)", "Gyration power spectrum from core trajectory"),
            (".breathing(method=...)", "Breathing-mode spectrum from orbit radius"),
            (".spectrogram(component='radius')", "Time-frequency spectrogram"),
            (".plt.power_spectrum()", "Compute & plot gyration spectrum"),
            (".plt.spectrogram()", "Compute & plot spectrogram"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        example = (
            "# Gyration spectrum\n"
            "spec = vortex.spectrum.gyration()\n"
            "spec.plt.power_spectrum()\n"
            "\n"
            "# Breathing mode spectrum\n"
            "breath = vortex.spectrum.breathing()\n"
            "\n"
            "# Time-frequency spectrogram\n"
            "sgram = vortex.spectrum.spectrogram()\n"
            "sgram.plt.spectrogram()\n"
            "\n"
            "# Or use plot shortcuts\n"
            "vortex.spectrum.plt.power_spectrum()\n"
            "vortex.spectrum.plt.spectrogram()"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);">'
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Vortex Spectrum Interface</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            "Gyration frequency spectrum from core trajectory</div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )


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

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "SpectrumInterfacePlotAccessor",
            [
                (
                    ".power_spectrum()",
                    "Compute + plot gyration power spectrum",
                    "Delegates to VortexSpectrumResult.plt.power_spectrum().",
                ),
                (
                    ".spectrogram()",
                    "Compute + plot time-frequency spectrogram",
                    "Delegates to VortexSpectrogramResult.plt.spectrogram().",
                ),
            ],
        )
