"""High-level mode classification interface for vortex dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

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


class VortexModesPlotAccessor:
    """Plotting facade for :class:`VortexModesInterface`."""

    def __init__(self, interface: VortexModesInterface):
        self._interface = interface

    def mode_map(self, f: float | None = None, *, unit: str = "ghz", ax=None):
        """Plot detected modes as frequency-power bars."""
        import matplotlib.pyplot as plt

        modes = self._interface.classify_all()
        if ax is None:
            _, ax = plt.subplots()

        if not modes:
            ax.set_title("No modes detected")
            ax.set_xlabel("Frequency [GHz]")
            ax.set_ylabel("Power [a.u.]")
            return ax

        freqs = np.array([item.frequency_ghz for item in modes], dtype=float)
        power = np.array([item.power for item in modes], dtype=float)
        labels = [item.mode_type for item in modes]

        ax.bar(freqs, power, width=max((np.max(freqs) - np.min(freqs)) * 0.03, 0.01))
        for fx, py, label in zip(freqs, power, labels):
            ax.text(fx, py, label, rotation=45, ha="left", va="bottom", fontsize=8)

        if f is not None:
            x = float(f) if unit.lower() == "ghz" else float(f) * 1e-9
            ax.axvline(x, color="red", linestyle="--", label="selected")
            ax.legend()

        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel("Power [a.u.]")
        ax.set_title("Detected vortex modes")
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
