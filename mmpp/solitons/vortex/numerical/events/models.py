"""Compatibility bridge for event result models."""

from ...events.models import (
    CoreExpulsionEvent,
    DwellTimeResult,
    PolaritySwitchEvent,
    StateSwitchEvent,
)

__all__ = [
    "PolaritySwitchEvent",
    "StateSwitchEvent",
    "CoreExpulsionEvent",
    "DwellTimeResult",
]
