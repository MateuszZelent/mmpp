"""State container for interactive dispersion explorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DispersionExplorerState:
    """Mutable interactive state for dispersion heatmap exploration."""

    fmin_ghz: float = 0.0
    fmax_ghz: float = 1.0
    source: str = "display"
    kscale: str = "rad_um"
    cmap: str = "viridis"
    positive_frequencies: bool = True
    lognorm: bool = False
    selected_k: float | None = None
    selected_f: float | None = None
    selected_power: float | None = None
    mode_type: str = "abs"
    show_flags: dict[str, bool] | None = None
    analytical: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.show_flags is None:
            self.show_flags = {
                "grid": True,
                "selection": True,
                "notes": True,
            }
        if self.analytical is None:
            self.analytical = {
                "enabled": False,
                "model": "kalinikos",
                "sw_config": "DE",
                "n_modes": 1,
                "k_points": 500,
            }
