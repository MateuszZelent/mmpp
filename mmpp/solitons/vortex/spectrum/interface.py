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
        """Compute gyration power spectrum from core trajectory.

        Parameters
        ----------
        exclude_annihilated : bool
            When ``True`` and the simulation health check detects core
            annihilation or polarity reversal, raise a ``ValueError`` instead
            of returning a (likely spurious) spectrum.  Default ``False``
            (show a warning instead).
        """
        force = bool(kwargs.pop("force", False))
        trajectory = kwargs.pop("trajectory", None)
        exclude_annihilated = bool(kwargs.pop("exclude_annihilated", False))
        if not force and trajectory is None and self._last_gyration is not None:
            return self._last_gyration
        if trajectory is None:
            trajectory = self._core.track()

        # Health check
        try:
            from ..health import check_core_health

            status = check_core_health(
                self._job,
                dataset_name=self._dataset_name,
                trajectory=trajectory,
                slice_info=self._slice_info,
            )
            if not status.is_healthy:
                if exclude_annihilated:
                    raise ValueError(
                        "Gyration spectrum excluded: " + "; ".join(status.warnings)
                    )
                status.issue_python_warnings()
        except ValueError:
            raise
        except Exception:
            pass

        selected_method = method or self._config.spectrum.method
        if "nperseg" not in kwargs and self._config.spectrum.nperseg is not None:
            kwargs["nperseg"] = self._config.spectrum.nperseg
        if "noverlap" not in kwargs and self._config.spectrum.noverlap is not None:
            kwargs["noverlap"] = self._config.spectrum.noverlap

        result = compute_gyration_spectrum(trajectory, method=selected_method, **kwargs)
        self._last_gyration = result
        return result

    def breathing(self, method: str | None = None, **kwargs) -> VortexSpectrumResult:
        """Compute breathing-mode spectrum from orbit radius signal.

        Parameters
        ----------
        exclude_annihilated : bool
            When ``True`` and the simulation health check detects core
            annihilation or polarity reversal, raise a ``ValueError`` instead
            of returning a spurious spectrum.  Default ``False``.
        """
        force = bool(kwargs.pop("force", False))
        trajectory = kwargs.pop("trajectory", None)
        exclude_annihilated = bool(kwargs.pop("exclude_annihilated", False))
        if not force and trajectory is None and self._last_breathing is not None:
            return self._last_breathing
        if trajectory is None:
            trajectory = self._core.track()

        # Health check
        try:
            from ..health import check_core_health

            status = check_core_health(
                self._job,
                dataset_name=self._dataset_name,
                trajectory=trajectory,
                slice_info=self._slice_info,
            )
            if not status.is_healthy:
                if exclude_annihilated:
                    raise ValueError(
                        "Breathing spectrum excluded: " + "; ".join(status.warnings)
                    )
                status.issue_python_warnings()
        except ValueError:
            raise
        except Exception:
            pass

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
        import uuid as _uuid

        from html import escape as _esc

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

        context_rows = [
            ("dataset", self._dataset_name or "auto-detect", None),
            (
                "slice",
                "custom" if self._slice_info is not None else "full geometry",
                None,
            ),
            ("default method", self._config.spectrum.method, NODE_COLOR_COMPUTE),
            ("nperseg", self._config.spectrum.nperseg or "backend default", None),
            ("noverlap", self._config.spectrum.noverlap or "backend default", None),
        ]
        accessors = [
            (
                "Compute:",
                [
                    (".gyration(method='welch', ...)", NODE_COLOR_COMPUTE),
                    (".breathing(method='welch', ...)", NODE_COLOR_COMPUTE),
                    (".spectrogram(component='radius', ...)", NODE_COLOR_COMPUTE),
                ],
            ),
            (
                "Results:",
                [
                    (".gyration().plt.power_spectrum(...)", NODE_COLOR_ANALYSIS),
                    (".breathing().peak_frequency_ghz", NODE_COLOR_ANALYSIS),
                    (".spectrogram().plt.spectrogram(...)", NODE_COLOR_ANALYSIS),
                ],
            ),
            (
                "Plotting:",
                [
                    (".plt.power_spectrum(...)", NODE_COLOR_PLOT),
                    (".plt.spectrogram(...)", NODE_COLOR_PLOT),
                ],
            ),
        ]
        namespace_rows = [
            (
                "gyration(...)",
                "Power spectrum of core orbit. Accepts override of FFT method plus PSD kwargs such as nperseg, noverlap, detrend and force.",
            ),
            (
                "breathing(...)",
                "Radius-oscillation spectrum. Useful for breathing mode and nonlinear expansion/contraction diagnostics.",
            ),
            (
                "spectrogram(component=...)",
                "Sliding time-frequency view. Use radius for breathing-like content, or pass other supported trajectory-derived components.",
            ),
            (
                "plt",
                "Shortcut plotting facade when you want one-liners without keeping the intermediate result object.",
            ),
        ]
        namespace_body = "".join(
            "<tr>"
            f"<td style='padding:6px 8px;font-family:monospace;color:{NODE_COLOR_COMPUTE};vertical-align:top;'>{_esc(name)}</td>"
            f"<td style='padding:6px 8px;color:#f8f8f2;'>{_esc(desc)}</td>"
            "</tr>"
            for name, desc in namespace_rows
        )
        arguments_rows = [
            (
                "method",
                "FFT / PSD backend selection. Falls back to config when omitted.",
            ),
            (
                "trajectory",
                "Precomputed trajectory to reuse instead of calling core tracking again.",
            ),
            ("force", "Recompute even when cached last result exists."),
            (
                "exclude_annihilated",
                "Raise instead of returning a likely spurious spectrum after core annihilation or polarity reversal.",
            ),
            ("component", "For spectrogram only. Default 'radius'."),
        ]
        arguments_body = "".join(
            "<tr>"
            f"<td style='padding:6px 8px;font-family:monospace;color:{NODE_COLOR_ANALYSIS};vertical-align:top;'>{_esc(name)}</td>"
            f"<td style='padding:6px 8px;color:#f8f8f2;'>{_esc(desc)}</td>"
            "</tr>"
            for name, desc in arguments_rows
        )
        example = (
            "# Dominant gyration frequency\n"
            "spec = jobs[-1].solitons.vortex.spectrum.gyration(method='welch')\n"
            "spec.peak_frequency_ghz\n"
            "spec.plt.power_spectrum()\n"
            "\n"
            "# Breathing mode spectrum\n"
            "breath = jobs[-1].solitons.vortex.spectrum.breathing()\n"
            "breath.plt.power_spectrum(log_scale=True)\n"
            "\n"
            "# Time-frequency spectrogram\n"
            "sgram = jobs[-1].solitons.vortex.spectrum.spectrogram(component='radius')\n"
            "sgram.plt.spectrogram()\n"
            "\n"
            "# One-line plot shortcuts\n"
            "jobs[-1].solitons.vortex.spectrum.plt.power_spectrum()\n"
            "jobs[-1].solitons.vortex.spectrum.plt.spectrogram()"
        )
        namespace_html = (
            "<div style='background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%);"
            "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(98,114,164,0.35);'>"
            "<b style='color:#bd93f9;'>Namespace Catalog</b>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;margin-top:8px;'>"
            f"{namespace_body}</table></div>"
        )
        arguments_html = (
            "<div style='background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%);"
            "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(98,114,164,0.35);'>"
            "<b style='color:#bd93f9;'>Important Arguments</b>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;margin-top:8px;'>"
            f"{arguments_body}</table></div>"
        )
        api_card = api_help_html(
            self,
            title="Vortex spectrum API help",
            prefix="jobs[-1].solitons.vortex.spectrum",
            properties=[("plt", "Plotting shortcuts for spectrum and spectrogram")],
            methods=["gyration", "breathing", "spectrogram"],
            subtitle="Live signatures for vortex spectral analysis methods and cached result entrypoints.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Spectrum Interface",
            icon="📈",
            subtitle="Frequency-domain diagnostics for gyration, breathing, and time-frequency evolution of the vortex core.",
            sections=[
                metrics_section_html(context_rows),
                accessors_section_html(accessors),
                namespace_html,
                arguments_html,
                examples_section_html(example, title="Spectrum Workflows"),
            ],
            api=api_card,
            uid=f"vortex-spectrum-{str(_uuid.uuid4())[:8]}",
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
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, node_card_html, plot_accessor_html

        html = plot_accessor_html(
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
        api_card = api_help_html(
            self,
            title="Vortex spectrum plot API help",
            prefix="vortex.spectrum.plt",
            methods=["power_spectrum", "spectrogram"],
            subtitle="Live signatures for spectrum plotting shortcuts.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Spectrum Plot Accessor",
            icon="🎨",
            subtitle="One-line plotting shortcuts for immediate spectrum and spectrogram previews.",
            sections=[html],
            api=api_card,
            uid=f"vortex-spectrum-plot-{str(_uuid.uuid4())[:8]}",
        )
