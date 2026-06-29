"""Data models for vortex topology detection results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "Topology Result",
            icon="🧲",
            subtitle="Detected topology invariants for a single magnetization snapshot.",
            sections=[
                metrics_section_html(
                    [
                        ("state", self.state, NODE_COLOR_ANALYSIS),
                        ("method", self.method, NODE_COLOR_COMPUTE),
                        ("polarity", int(self.polarity), None),
                        ("vorticity", int(self.vorticity), None),
                        ("chirality", int(self.chirality), None),
                        ("Q", f"{float(self.Q):.6g}", NODE_COLOR_ANALYSIS),
                        (
                            "confidence",
                            f"{float(self.confidence):.6g}",
                            NODE_COLOR_COMPUTE,
                        ),
                        ("is_consistent", bool(self.is_consistent), None),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Fields:",
                            [
                                (".core_position", NODE_COLOR_COMPUTE),
                                (".topological_density", NODE_COLOR_ANALYSIS),
                                (".chirality_confidence", NODE_COLOR_ANALYSIS),
                            ],
                        ),
                    ]
                ),
                examples_section_html(
                    "top = jobs[-1].solitons.vortex.topology.detect()\n"
                    "top.state\n"
                    "top.core_position\n"
                    "top.is_consistent",
                    title="Result Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Topology result API help",
                prefix="jobs[-1].solitons.vortex.topology.detect()",
                properties=[
                    ("polarity", "Detected core polarity"),
                    ("vorticity", "Detected winding sign"),
                    ("chirality", "Detected in-plane chirality"),
                    ("Q", "Topological charge"),
                    ("core_position", "Estimated core position"),
                    ("topological_density", "Local density map"),
                    ("state", "Classified topology label"),
                    ("method", "Detection backend"),
                    ("confidence", "Detection confidence"),
                    ("chirality_confidence", "Confidence of chirality estimate"),
                    ("convention", "Applied topology convention"),
                    ("is_consistent", "Checks expected relation between invariants"),
                ],
                subtitle="Live attributes of the detected topology snapshot.",
                chrome=False,
            ),
            uid=f"topology-result-{str(_uuid.uuid4())[:8]}",
        )


__all__ = ["TopologyResult"]
