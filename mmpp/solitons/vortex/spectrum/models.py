"""Result models for vortex spectral analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..._method_helpers import InteractiveNodeMixin
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
            "Vortex Spectrum Result",
            icon="📊",
            subtitle=f"{self.component} spectrum with dominant peak and plotting helpers.",
            sections=[
                metrics_section_html(
                    [
                        ("component", self.component, NODE_COLOR_ANALYSIS),
                        ("method", self.method, NODE_COLOR_COMPUTE),
                        ("n_freq", int(np.asarray(self.frequencies).size), None),
                        ("peak_ghz", f"{self.peak_frequency_ghz:.6g}", NODE_COLOR_PLOT),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Data:",
                            [
                                (".frequencies", NODE_COLOR_COMPUTE),
                                (".power", NODE_COLOR_COMPUTE),
                                (".amplitude", NODE_COLOR_ANALYSIS),
                            ],
                        ),
                        (
                            "Plotting:",
                            [
                                (".plt.power_spectrum(...)", NODE_COLOR_PLOT),
                                (".peak_frequency_ghz", NODE_COLOR_PLOT),
                            ],
                        ),
                    ]
                ),
                examples_section_html(
                    "spec = jobs[-1].solitons.vortex.spectrum.gyration()\n"
                    "spec.peak_frequency_ghz\n"
                    "spec.plt.power_spectrum(log_scale=True)",
                    title="Result Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Vortex spectrum result API help",
                prefix="jobs[-1].solitons.vortex.spectrum.gyration()",
                properties=[
                    ("frequencies", "Frequency axis in Hz"),
                    ("power", "Power spectral density samples"),
                    ("amplitude", "Square-root amplitude spectrum"),
                    ("peak_frequency_hz", "Dominant peak frequency in Hz"),
                    ("peak_frequency_ghz", "Dominant peak frequency in GHz"),
                    ("plt", "Plotting accessor"),
                ],
                subtitle="Live attributes for a computed vortex spectrum result.",
                chrome=False,
            ),
            uid=f"vortex-spectrum-result-{str(_uuid.uuid4())[:8]}",
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
            "Vortex Spectrogram Result",
            icon="🌊",
            subtitle=f"{self.component} time-frequency map with immediate plotting support.",
            sections=[
                metrics_section_html(
                    [
                        ("component", self.component, NODE_COLOR_ANALYSIS),
                        ("method", self.method, NODE_COLOR_COMPUTE),
                        ("n_times", int(np.asarray(self.times).size), None),
                        (
                            "n_freq",
                            int(np.asarray(self.frequencies).size),
                            NODE_COLOR_PLOT,
                        ),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Data:",
                            [
                                (".times", NODE_COLOR_COMPUTE),
                                (".frequencies", NODE_COLOR_COMPUTE),
                                (".power", NODE_COLOR_ANALYSIS),
                            ],
                        ),
                        (
                            "Plotting:",
                            [
                                (".plt.spectrogram(...)", NODE_COLOR_PLOT),
                            ],
                        ),
                    ]
                ),
                examples_section_html(
                    "sgram = jobs[-1].solitons.vortex.spectrum.spectrogram()\n"
                    "sgram.plt.spectrogram(as_ghz=True, db_scale=True)",
                    title="Result Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Vortex spectrogram result API help",
                prefix="jobs[-1].solitons.vortex.spectrum.spectrogram()",
                properties=[
                    ("times", "Time axis in seconds"),
                    ("frequencies", "Frequency axis in Hz"),
                    ("power", "Time-frequency power map"),
                    ("plt", "Plotting accessor"),
                ],
                subtitle="Live attributes for a computed vortex spectrogram result.",
                chrome=False,
            ),
            uid=f"vortex-spectrogram-result-{str(_uuid.uuid4())[:8]}",
        )


class VortexSpectrumPlotAccessor(InteractiveNodeMixin):
    """Plot helpers for :class:`VortexSpectrumResult`."""

    _interactive_owner = "spectrum.plt"
    _interactive_nodes = frozenset({"power_spectrum"})

    def __init__(self, result: VortexSpectrumResult):
        self._result = result

    def power_spectrum(
        self, *, ax=None, as_ghz: bool = True, log_scale: bool = False, **kwargs
    ):
        """Plot power spectrum.

        Parameters
        ----------
        health : CoreHealthStatus or None
            When provided, an annotation warning is drawn on the axes if the
            simulation showed core annihilation or boundary collision.
        """
        plot_kwargs = dict(kwargs)
        save = plot_kwargs.pop("save", None)
        health = plot_kwargs.pop("health", None)
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

        # Attach health annotation when annihilation/collision was detected
        if health is not None:
            try:
                health.warn_on_plot(ax)
            except Exception:
                pass

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


class VortexSpectrogramPlotAccessor(InteractiveNodeMixin):
    """Plot helpers for :class:`VortexSpectrogramResult`."""

    _interactive_owner = "spectrogram.plt"
    _interactive_nodes = frozenset({"spectrogram"})

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
        health = mesh_kwargs.pop("health", None)
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

        if health is not None:
            try:
                health.warn_on_plot(ax)
            except Exception:
                pass

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
