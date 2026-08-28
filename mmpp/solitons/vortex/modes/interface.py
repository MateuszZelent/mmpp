"""High-level mode classification interface for vortex dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..._method_helpers import InteractiveNodeMixin
from .._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)
from ..config import VortexConfig
from .classifier import VortexModesClassifier
from .models import VortexModeResult


class VortexModesInterface(InteractiveNodeMixin):
    """Mode-classification namespace."""

    _interactive_owner = "job[0].vortex.modes"
    _interactive_nodes = frozenset({"classify", "classify_all"})
    _interactive_examples = {
        "classify": ["mode = job[0].vortex.modes.classify(f=0.5, unit='ghz')"],
        "classify_all": ["modes = job[0].vortex.modes.classify_all(max_modes=6)"],
    }

    def __init__(
        self,
        job_result,
        dataset_name: str | None,
        slice_info: Any | None,
        config: VortexConfig,
        core_interface,
        spectrum_interface,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config
        self._core = core_interface
        self._spectrum = spectrum_interface
        self._classifier: VortexModesClassifier | None = None

    def _get_classifier(self) -> VortexModesClassifier:
        if self._classifier is None:
            # Mode classification is spectrum-driven and does not require the
            # most expensive sub-pixel tracker by default.
            trajectory = self._core.track(method=self._config.modes.tracking_method)
            self._classifier = VortexModesClassifier(trajectory)
        return self._classifier

    def classify(
        self, f: float | None = None, *, unit: str = "ghz"
    ) -> VortexModeResult:
        """Classify mode nearest to target frequency or dominant mode when ``f`` is None."""
        classifier = self._get_classifier()

        if f is None:
            modes = classifier.classify_all(
                spectrum_method=self._config.spectrum.method
            )
            if not modes:
                return VortexModeResult(
                    m_index=0, n_index=0, mode_type="unknown", source="none"
                )
            return modes[0]

        unit_norm = unit.lower()
        if unit_norm == "ghz":
            frequency_hz = float(f) * 1e9
        elif unit_norm == "hz":
            frequency_hz = float(f)
        else:
            raise ValueError("unit must be 'ghz' or 'hz'")

        return classifier.classify(
            frequency_hz=frequency_hz,
            spectrum_method=self._config.spectrum.method,
        )

    def classify_all(
        self, *, max_modes: int = 6, min_prominence: float = 0.05
    ) -> list[VortexModeResult]:
        """Classify all dominant modes."""
        if max_modes == 6:
            max_modes = int(self._config.modes.max_modes)
        if abs(min_prominence - 0.05) < 1e-12:
            min_prominence = float(self._config.modes.min_prominence)

        return self._get_classifier().classify_all(
            spectrum_method=self._config.spectrum.method,
            max_modes=max_modes,
            min_prominence=min_prominence,
        )

    @property
    def gyration(self) -> VortexModeResult | None:
        """Best gyration-like mode if available."""
        modes = self.classify_all()
        for item in modes:
            if item.mode_type == "gyration":
                return item
        return None

    @property
    def breathing(self) -> VortexModeResult | None:
        """Best breathing-like mode if available."""
        modes = self.classify_all()
        for item in modes:
            if item.mode_type == "breathing":
                return item
        return None

    @property
    def plt(self):
        """Plot accessor."""
        return VortexModesPlotAccessor(self)

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
            ("dataset", self._dataset_name or "auto-detect", NODE_COLOR_COMPUTE),
            (
                "slice",
                "custom" if self._slice_info is not None else "full geometry",
                None,
            ),
            (
                "tracking method",
                self._config.modes.tracking_method,
                NODE_COLOR_ANALYSIS,
            ),
            ("max modes", self._config.modes.max_modes, None),
            ("min prominence", self._config.modes.min_prominence, NODE_COLOR_ANALYSIS),
        ]
        accessors = [
            (
                "Classify:",
                [
                    (".classify(f=None, unit='ghz')", NODE_COLOR_COMPUTE),
                    (".classify_all(max_modes=6, ...)", NODE_COLOR_COMPUTE),
                ],
            ),
            (
                "Quick Picks:",
                [
                    (".gyration", NODE_COLOR_ANALYSIS),
                    (".breathing", NODE_COLOR_ANALYSIS),
                ],
            ),
            (
                "Plotting:",
                [
                    (".plt.mode_map(...)", NODE_COLOR_PLOT),
                    (".plt.mode_table()", NODE_COLOR_PLOT),
                ],
            ),
        ]
        methods = [
            (
                "classify(f=None, unit='ghz')",
                "Returns the dominant mode when f is omitted, or the closest classified mode around the requested target frequency.",
            ),
            (
                "classify_all(max_modes=6)",
                "Extracts several dominant peaks and labels them as gyration, breathing, or higher-order dynamical modes.",
            ),
            (
                "gyration",
                "Convenience property returning the best gyration-like classified mode, or None.",
            ),
            (
                "breathing",
                "Convenience property returning the best breathing-like classified mode, or None.",
            ),
            (
                "plt",
                "Plot/table helper facade for visual inspection and tabular summaries.",
            ),
        ]
        method_rows = "".join(
            "<tr>"
            f"<td style='padding:6px 8px;font-family:monospace;color:{NODE_COLOR_COMPUTE};vertical-align:top;'>{_esc(m)}</td>"
            f"<td style='padding:6px 8px;color:#f8f8f2;'>{_esc(d)}</td>"
            "</tr>"
            for m, d in methods
        )
        example = (
            "# Classify dominant mode\n"
            "mode = jobs[-1].solitons.vortex.modes.classify()\n"
            "print(f'{mode.mode_type} at {mode.frequency_ghz:.2f} GHz')\n"
            "\n"
            "# Classify specific frequency\n"
            "mode = jobs[-1].solitons.vortex.modes.classify(f=0.5, unit='ghz')\n"
            "\n"
            "# All modes\n"
            "modes = jobs[-1].solitons.vortex.modes.classify_all()\n"
            "jobs[-1].solitons.vortex.modes.plt.mode_map()\n"
            "\n"
            "# Quick access\n"
            "gyro = jobs[-1].solitons.vortex.modes.gyration\n"
            "breath = jobs[-1].solitons.vortex.modes.breathing"
        )
        methods_html = (
            "<div style='background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%);"
            "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(98,114,164,0.35);'>"
            "<b style='color:#bd93f9;'>Mode Classification Methods</b>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;margin-top:8px;'>"
            f"{method_rows}</table></div>"
        )
        api = api_help_html(
            self,
            title="Vortex modes API help",
            prefix="jobs[-1].solitons.vortex.modes",
            properties=[
                ("gyration", "Best gyration-like mode, or None"),
                ("breathing", "Best breathing-like mode, or None"),
                ("plt", "Plotting facade for classified modes"),
            ],
            methods=["classify", "classify_all"],
            subtitle="Live public API for vortex mode classification.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Modes Interface",
            icon="🎼",
            subtitle="Mode classification for gyration, breathing, and higher-order vortex dynamics.",
            sections=[
                metrics_section_html(context_rows),
                accessors_section_html(accessors),
                methods_html,
                examples_section_html(example, title="Modes Workflows"),
            ],
            api=api,
            uid=f"mmpp-vortex-modes-{str(_uuid.uuid4())[:8]}",
        )


class VortexModesPlotAccessor(InteractiveNodeMixin):
    """Plotting facade for :class:`VortexModesInterface`."""

    _interactive_owner = "job[0].vortex.modes.plt"
    _interactive_nodes = frozenset({"mode_map", "mode_table"})

    def __init__(self, interface: VortexModesInterface):
        self._interface = interface

    def mode_map(self, f: float | None = None, *, unit: str = "ghz", ax=None, **kwargs):
        """Plot detected modes as frequency-power bars."""
        bar_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(bar_kwargs)
        figure_kwargs = pop_figure_kwargs(bar_kwargs)

        modes = self._interface.classify_all()
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        if not modes:
            ax.set_title("No modes detected")
            ax.set_xlabel("Frequency [GHz]")
            ax.set_ylabel("Power [a.u.]")
            apply_axes_style(ax, style_kwargs)
            return ax

        freqs = np.array([item.frequency_ghz for item in modes], dtype=float)
        power = np.array([item.power for item in modes], dtype=float)
        labels = [item.mode_type for item in modes]

        width = float(max((np.max(freqs) - np.min(freqs)) * 0.03, 0.01))
        bar_kwargs.setdefault("width", width)
        ax.bar(freqs, power, **bar_kwargs)
        for fx, py, label in zip(freqs, power, labels, strict=False):
            ax.text(fx, py, label, rotation=45, ha="left", va="bottom", fontsize=8)

        if f is not None:
            x = float(f) if unit.lower() == "ghz" else float(f) * 1e-9
            ax.axvline(x, color="red", linestyle="--", label="selected")
            ax.legend()

        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel("Power [a.u.]")
        ax.set_title("Detected vortex modes")
        apply_axes_style(ax, style_kwargs)
        return ax

    def mode_table(self):
        """Return mode table as list of dict rows."""
        modes = self._interface.classify_all()
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(modes):
            rows.append(
                {
                    "rank": idx,
                    "mode": item.mode_type,
                    "m": item.m_index,
                    "n": item.n_index,
                    "f_ghz": item.frequency_ghz,
                    "power": item.power,
                    "confidence": item.confidence,
                    "source": item.source,
                }
            )
        return rows

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, node_card_html, plot_accessor_html

        overview = plot_accessor_html(
            "VortexModesPlotAccessor",
            [
                (
                    ".mode_map(f=None, unit='ghz')",
                    "Frequency-power bar chart of detected modes",
                    "f: optional target frequency marker. unit: 'ghz' or 'hz'.",
                ),
                (
                    ".mode_table()",
                    "Returns list of dicts with mode details",
                    "rank, mode, m, n, f_ghz, power, confidence, source.",
                ),
            ],
        )
        api = api_help_html(
            self,
            title="Vortex modes plot API help",
            prefix="jobs[-1].solitons.vortex.modes.plt",
            methods=["mode_map", "mode_table"],
            subtitle="Plot and table helpers for classified vortex modes.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Modes Plot Accessor",
            icon="🎨",
            subtitle="Plot and table shortcuts for classified vortex modes.",
            sections=[overview],
            api=api,
            uid=f"mmpp-vortex-modes-plot-{str(_uuid.uuid4())[:8]}",
        )
