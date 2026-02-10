"""Result models for vortex mode classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
