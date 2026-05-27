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
            "Vortex Mode Result",
            icon="🎵",
            subtitle="Single classified vortex dynamical mode with spectral label and confidence.",
            sections=[
                metrics_section_html(
                    [
                        ("label", self.label, NODE_COLOR_ANALYSIS),
                        ("rotation_sense", self.rotation_sense, None),
                        ("frequency_ghz", f"{self.frequency_ghz:.6g}", NODE_COLOR_PLOT),
                        ("power", f"{float(self.power):.6g}", NODE_COLOR_COMPUTE),
                        (
                            "confidence",
                            f"{float(self.confidence):.6g}",
                            NODE_COLOR_ANALYSIS,
                        ),
                        ("source", self.source, None),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Indices:",
                            [
                                (".m_index", NODE_COLOR_COMPUTE),
                                (".n_index", NODE_COLOR_COMPUTE),
                                (".l_index", NODE_COLOR_COMPUTE),
                            ],
                        ),
                    ]
                ),
                examples_section_html(
                    "mode = jobs[-1].solitons.vortex.modes.classify()\n"
                    "mode.label\n"
                    "mode.frequency_ghz",
                    title="Mode Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Vortex mode result API help",
                prefix="jobs[-1].solitons.vortex.modes.classify()",
                properties=[
                    ("m_index", "Azimuthal mode index"),
                    ("n_index", "Radial mode index"),
                    ("l_index", "Optional additional index"),
                    ("mode_type", "Mode classification label"),
                    ("rotation_sense", "Rotation-sense classification"),
                    ("confidence", "Classification confidence"),
                    ("frequency_hz", "Mode frequency in Hz"),
                    ("frequency_ghz", "Mode frequency in GHz"),
                    ("power", "Spectral power"),
                    ("source", "Classification source"),
                    ("label", "Human-readable mode label"),
                ],
                subtitle="Live attributes of a classified vortex mode.",
                chrome=False,
            ),
            uid=f"vortex-mode-result-{str(_uuid.uuid4())[:8]}",
        )
