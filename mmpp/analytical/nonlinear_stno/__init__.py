"""
Nonlinear Spin-Torque Nano-Oscillator (STNO) Module
Implements the Dual Back-Action Theory (DBAT 2.0) for broadband dynamics.
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
