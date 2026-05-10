"""Data models for vortex topology detection results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmpp._shared.repr_html import make_simple_card


@dataclass
class TopologyResult:
    """Complete topological characterization of a vortex-like state."""

    polarity: int
    vorticity: int
    chirality: int
    Q: float
    core_position: tuple[float, float]
    topological_density: np.ndarray
    state: str
    method: str
    confidence: float
    chirality_confidence: float = 0.0
    convention: str = "down"

    @property
    def is_consistent(self) -> bool:
        """Return ``True`` when measured Q matches expected topological relation."""
        if self.state in {"vortex", "antivortex"}:
            expected = self.polarity * self.vorticity / 2.0
            return abs(self.Q - expected) < 0.1
        if self.state == "skyrmion":
            return abs(abs(self.Q) - 1.0) < 0.1
        return False

    def _repr_html_(self) -> str:
        rows = [
            ("state", str(self.state)),
            ("method", str(self.method)),
            ("polarity", str(int(self.polarity))),
            ("vorticity", str(int(self.vorticity))),
            ("chirality", str(int(self.chirality))),
            ("Q", f"{float(self.Q):.6g}"),
            ("confidence", f"{float(self.confidence):.6g}"),
            ("is_consistent", str(bool(self.is_consistent))),
        ]
        return make_simple_card(
            title="TopologyResult",
            subtitle="Detected topology invariants for a single snapshot",
            rows=rows,
        )


__all__ = ["TopologyResult"]
