"""Base models shared by soliton analysis modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SolitonConfig:
    """Base configuration container for soliton modules."""

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolitonResult:
    """Base result container for soliton modules."""

    metadata: dict[str, Any] = field(default_factory=dict)
