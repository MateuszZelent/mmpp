"""Numerical signal-processing namespace for vortex dynamics."""

from .interface import SignalsInterface
from .magnetoresistance import compute_magnetoresistance
from .models import MagnetoresistanceResult, SignalSpectrumResult, VoltageResult
from .power_spectrum import compute_signal_power_spectrum
from .voltage import compute_voltage

__all__ = [
    "SignalsInterface",
    "MagnetoresistanceResult",
    "VoltageResult",
    "SignalSpectrumResult",
    "compute_magnetoresistance",
    "compute_voltage",
    "compute_signal_power_spectrum",
]
