"""Result models for vortex mode classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mmpp._shared.repr_html import make_simple_card


@dataclass
class VortexModeResult:
    """Classification result for a single vortex dynamical mode."""

    m_index: int
    n_index: int
    l_index: int | None = None

    mode_type: str = "unknown"
    rotation_sense: str = "unknown"
    confidence: float = 0.0

    frequency_hz: float = 0.0
    power: float = 0.0

    source: str = "trajectory"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frequency_ghz(self) -> float:
        """Frequency in GHz."""
        return float(self.frequency_hz) * 1e-9

    @property
    def label(self) -> str:
        """Human-readable mode label."""
        return f"{self.mode_type}(m={self.m_index}, n={self.n_index})"

    def _repr_html_(self) -> str:
        rows = [
            ("label", self.label),
            ("rotation_sense", str(self.rotation_sense)),
            ("frequency_ghz", f"{self.frequency_ghz:.6g}"),
            ("power", f"{float(self.power):.6g}"),
            ("confidence", f"{float(self.confidence):.6g}"),
            ("source", str(self.source)),
        ]
        return make_simple_card(
            title="VortexModeResult",
            subtitle="Single classified vortex dynamical mode",
            rows=rows,
        )
