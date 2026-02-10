"""Event detection namespace for vortex dynamics."""

from .core_expulsion import detect_core_expulsions
from .dwell_time import dwell_time_statistics
from .interface import EventsInterface
from .models import (
    CoreExpulsionEvent,
    DwellTimeResult,
    PolaritySwitchEvent,
    StateSwitchEvent,
)
from .polarity import detect_polarity_switches
from .state_transitions import classify_gc_states, detect_state_switches

__all__ = [
    "EventsInterface",
    "PolaritySwitchEvent",
    "StateSwitchEvent",
    "CoreExpulsionEvent",
    "DwellTimeResult",
    "detect_polarity_switches",
    "detect_state_switches",
    "classify_gc_states",
    "detect_core_expulsions",
    "dwell_time_statistics",
]
