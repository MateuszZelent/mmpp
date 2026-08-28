"""Numerical vortex analysis namespace (compatibility layer)."""

from .core import CoreInterface, TrackingMethod, TrajectoryResult, track_core
from .energy import EnergyInterface
from .events import EventsInterface
from .modes import VortexModeResult, VortexModesInterface
from .nonlinear import NonlinearInterface
from .signals import SignalsInterface
from .topology import TopologyInterface, TopologyResult, detect_topology

__all__ = [
    "CoreInterface",
    "TrajectoryResult",
    "TrackingMethod",
    "track_core",
    "TopologyInterface",
    "TopologyResult",
    "detect_topology",
    "VortexModesInterface",
    "VortexModeResult",
    "NonlinearInterface",
    "EventsInterface",
    "SignalsInterface",
    "EnergyInterface",
]
