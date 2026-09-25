"""Result models for vortex spectral analysis."""

from __future__ import annotations

import os
import textwrap
from collections import Counter
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
    _interactive_descriptions = {
        "power_spectrum": "Plot the vortex gyration PSD; info='full' adds its input and computation methodology below the axes."
    }
    _interactive_examples = {"power_spectrum": ["spec.plt.power_spectrum(info='full')"]}

    def __init__(self, result: VortexSpectrumResult):
        self._result = result

    def _methodology_text(self) -> str:
        metadata = self._result.metadata
        lines: list[str] = []

        source_file = metadata.get("source_file")
        input_file_count = metadata.get("input_file_count")
        if input_file_count is None:
            lines.append("Input files: source not recorded (precomputed trajectory)")
        elif source_file:
            lines.append(f"Input files ({input_file_count}):")
            path_line = "  "
            path_parts = str(source_file).split(os.sep)
            for index, part in enumerate(path_parts):
                segment = part if index == 0 else os.sep + part
                if len(path_line) + len(segment) > 94 and path_line.strip():
                    lines.append(path_line)
                    path_line = "  " + segment
                else:
                    path_line += segment
            if path_line.strip():
                lines.append(path_line)
        else:
            lines.append(f"Input files: {input_file_count}; source path not recorded")

        source = metadata.get("source", "unknown")
        details = [f"source={source}"]
        dataset = metadata.get("dataset")
        if dataset:
            details.append(f"dataset={dataset}")
        if metadata.get("slice_info"):
            details.append(f"slice={metadata['slice_info']}")
        if metadata.get("z_layer") is not None:
            details.append(f"z-layer={metadata['z_layer']}")
        magnetization_component = metadata.get("magnetization_component")
        if magnetization_component:
            details.append(f"core located from {magnetization_component}")
        elif source == "table":
            details.append(
                "trajectory from table columns "
                f"{metadata.get('x_column', 'x')} and {metadata.get('y_column', 'y')}"
            )
        lines.append("Tracking input: " + "; ".join(details))

        requested = metadata.get("trajectory_requested_method")
        used = metadata.get("trajectory_method", "unknown")
        tracker_line = f"Core tracker: method={used}"
        if requested and requested != used:
            tracker_line += f" (requested={requested})"
        fallback_from = metadata.get("fallback_from")
        if fallback_from:
            tracker_line += f"; fallback from {fallback_from}"
        frame_methods = metadata.get("tracking_frame_methods", [])
        if frame_methods:
            if isinstance(frame_methods, dict):
                counts = frame_methods
            else:
                counts = Counter(str(value) for value in frame_methods)
            tracker_line += "; frames=" + ", ".join(
                f"{name}:{count}" for name, count in sorted(counts.items())
            )
        fallbacks = int(metadata.get("tracking_frame_fallbacks", 0))
        if fallbacks and not frame_methods:
            tracker_line += f"; frame fallbacks={fallbacks}"
        lines.append(tracker_line)

        lines.append(
            "Spectrum signal: PSD(x_core) + PSD(y_core); "
            "FFT is applied to tracked positions, not directly to mx/my/mz."
        )
        if metadata.get("sidedness"):
            lines.append(f"Spectrum sides: {metadata['sidedness']}")
        backend = metadata.get("backend", "unspecified backend")
        requested_estimator = metadata.get("requested_method")
        estimator = self._result.method
        if requested_estimator and requested_estimator != estimator:
            estimator += f" (requested={requested_estimator})"
        lines.append(f"PSD estimator: {estimator}; backend={backend}")

        n_samples = metadata.get("n_samples")
        dt = metadata.get("dt")
        fs = metadata.get("fs")
        sampling = []
        if n_samples is not None:
            sampling.append(f"N={int(n_samples)}")
        if dt is not None:
            sampling.append(f"dt={float(dt):.6g} s")
        if fs is not None:
            sampling.append(f"fs={float(fs):.6g} Hz")
        if sampling:
            lines.append("Sampling: " + "; ".join(sampling))

        settings = []
        for key in (
            "nperseg",
            "nfft",
            "noverlap",
            "window",
            "detrend",
            "scaling",
            "average",
        ):
            if key in metadata:
                settings.append(f"{key}={metadata[key]}")
        if "normalization" in metadata:
            settings.append(f"normalization={metadata['normalization']}")
        if settings:
            lines.append("Estimator settings: " + "; ".join(settings))

        wrap_width = 94
        return "\n".join(
            textwrap.fill(
                line,
                width=wrap_width,
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            for line in lines
        )

    @staticmethod
    def _add_info_block(ax, text: str) -> None:
        figure = ax.figure
        fontsize = 7.5
        lines = text.splitlines()
        figure_height = float(figure.get_size_inches()[1])
        block_height = len(lines) * fontsize * 1.35 / 72.0 + 0.06
        reserved_bottom = min(0.52, max(0.22, block_height / figure_height))

        layout_engine = figure.get_layout_engine()
        if layout_engine is None and figure.subplotpars.bottom < reserved_bottom:
            figure.subplots_adjust(bottom=reserved_bottom)
        elif layout_engine is not None:
            figure.canvas.draw()
            axes_positions = [(axis, axis.get_position()) for axis in figure.axes]
            figure.set_layout_engine(None)
            available_height = 1.0 - reserved_bottom
            for axis, position in axes_positions:
                axis.set_position(
                    [
                        position.x0,
                        reserved_bottom + position.y0 * available_height,
                        position.width,
                        position.height * available_height,
                    ]
                )

        figure.text(
            0.01,
            0.012,
            text,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            family="monospace",
            linespacing=1.2,
        )

    def power_spectrum(
        self,
        *,
        ax=None,
        as_ghz: bool = True,
        log_scale: bool = False,
        info: str | None = None,
        **kwargs,
    ):
        """Plot power spectrum.

        Parameters
        ----------
        health : CoreHealthStatus or None
            When provided, an annotation warning is drawn on the axes if the
            simulation showed core annihilation or boundary collision.
        info : {None, "full"}
            Draw the recorded input and computation methodology below the plot.
        """
        if info not in {None, "full"}:
            raise ValueError("info must be None or 'full'")

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

        if info == "full":
            self._add_info_block(ax, self._methodology_text())

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
                    ".power_spectrum(as_ghz=True, log_scale=False, info=None)",
                    "Power spectrum of vortex gyration",
                    "info='full' adds input files, core tracking source and method, PSD estimator, and FFT settings below the plot.",
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
