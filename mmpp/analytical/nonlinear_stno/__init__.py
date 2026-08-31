"""
Experimental nonlinear Spin-Torque Nano-Oscillator (STNO) extension.

The additional 4D back-action terms are phenomenological and uncalibrated;
they are not the validated rigid-vortex Thiele baseline.
"""

from .analyzer import SpectrumAnalyzer
from .engine import run_all_sweeps_parallel
from .physics import STNOParameters

__all__ = [
    "STNOParameters",
    "run_all_sweeps_parallel",
    "SpectrumAnalyzer",
    "DashboardPlotter",
]


def __getattr__(name):
    """Lazily load plotting helpers so compute imports do not require Matplotlib."""
    if name == "DashboardPlotter":
        from .plotter import DashboardPlotter

        globals()[name] = DashboardPlotter
        return DashboardPlotter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
