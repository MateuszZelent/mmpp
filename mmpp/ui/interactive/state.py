"""State model for shared interactive controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InteractiveState:
    """Mutable state tracked by interactive viewers."""

    frame_index: int = 0
    playing: bool = False
    speed: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
