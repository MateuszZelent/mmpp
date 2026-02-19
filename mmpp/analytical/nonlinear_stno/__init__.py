"""
Nonlinear Spin-Torque Nano-Oscillator (STNO) Module
Implements the Dual Back-Action Theory (DBAT 2.0) for broadband dynamics.
"""

from .physics import STNOParameters
from .engine import run_all_sweeps_parallel
from .analyzer import SpectrumAnalyzer
from .plotter import DashboardPlotter

__all__ = [
    'STNOParameters',
    'run_all_sweeps_parallel',
    'SpectrumAnalyzer',
    'DashboardPlotter',
]
