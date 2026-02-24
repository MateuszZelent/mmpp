"""State container for interactive hysteresis explorer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HysteresisExplorerState:
    """Mutable interactive state."""

    current_idx: int = 0
    current_branch: str = "ascending"
    field_value: float = 0.0
    magnetization_value: float = 0.0
    is_animating: bool = False
    animation_speed: float = 1.0
    snapshot_component: str = "snapshot"
    z_layer: int | str = 0
    roi: tuple[int, int, int, int] | None = None
    loop_panel_weight: float = 1.15
    snapshot_panel_weight: float = 1.0
    show_flags: dict[str, bool] = field(
        default_factory=lambda: {
            "hc": True,
            "mr": True,
            "ms": False,
            "arrow": True,
            "branch_colors": True,
            "trail": True,
        }
    )
