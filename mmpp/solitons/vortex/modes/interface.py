"""High-level mode classification interface for vortex dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)
from ..config import VortexConfig
from .classifier import VortexModesClassifier
from .models import VortexModeResult


class VortexModesInterface:
    """Mode-classification namespace."""

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
            trajectory = self._core.track()
            self._classifier = VortexModesClassifier(trajectory)
        return self._classifier

    def classify(self, f: float | None = None, *, unit: str = "ghz") -> VortexModeResult:
        """Classify mode nearest to target frequency or dominant mode when ``f`` is None."""
        classifier = self._get_classifier()

        if f is None:
            modes = classifier.classify_all(spectrum_method=self._config.spectrum.method)
            if not modes:
                return VortexModeResult(m_index=0, n_index=0, mode_type="unknown", source="none")
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

    def classify_all(self, *, max_modes: int = 6, min_prominence: float = 0.05) -> list[VortexModeResult]:
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
        from html import escape as _esc

        methods = [
            (".classify(f=None, unit='ghz')", "Classify mode at frequency f (or dominant)"),
            (".classify_all(max_modes=6)", "Classify all dominant modes"),
            (".gyration", "Best gyration-like mode (or None)"),
            (".breathing", "Best breathing-like mode (or None)"),
            (".plt.mode_map()", "Plot modes as frequency-power bars"),
            (".plt.mode_table()", "Return mode table as list of dicts"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        example = (
            "# Classify dominant mode\n"
            "mode = vortex.modes.classify()\n"
            "print(f'{mode.mode_type} at {mode.frequency_ghz:.2f} GHz')\n"
            "\n"
            "# Classify specific frequency\n"
            "mode = vortex.modes.classify(f=0.5, unit='ghz')\n"
            "\n"
            "# All modes\n"
            "modes = vortex.modes.classify_all()\n"
            "vortex.modes.plt.mode_map()\n"
            "\n"
            "# Quick access\n"
            "gyro = vortex.modes.gyration   # VortexModeResult or None\n"
            "breath = vortex.modes.breathing"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Vortex Modes Interface</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            "Mode classification (gyration, breathing, higher-order)</div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods &amp; Properties</div>"
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


class VortexModesPlotAccessor:
    """Plotting facade for :class:`VortexModesInterface`."""

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
        for fx, py, label in zip(freqs, power, labels):
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
