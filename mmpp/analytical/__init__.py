"""
mmpp.analytical - Analytical Models for Magnetism

This module provides analytical models for ferromagnetic resonance (FMR)
and spin wave dispersion relations with a fluent plotting API.

Usage:
    import mmpp
    import numpy as np
    
    # FMR frequency
    fmr = mmpp.analytical.kittel(Ms=8e5, B=0.1, Ku=1e4)
    fmr.plt.plot()
    
    # Dispersion relation
    k = np.linspace(-1e7, 1e7, 500)
    disp = mmpp.analytical.damon_eshbach(k=k, Ms=8e5, B=0.1, d=100e-9, Aex=13e-12)
    disp.plt.plot()
"""

from .constants import MU0, GAMMA_E, G_FACTOR_DEFAULT, gamma
from .base import AnalyticalResult, DispersionResult, FMRResult
from .fmr import (
    kittel,
    kittel_oop,
    kittel_exchange,
)
from .dispersion import (
    kalinikos,
    kalinikos_no_approx,
    damon_eshbach,
    backward_volume,
    forward_volume,
    bottcher,
    kim,
    cortes_ortuno,
)

__all__ = [
    # Constants
    "MU0",
    "GAMMA_E",
    "G_FACTOR_DEFAULT",
    "gamma",
    # Base classes
    "AnalyticalResult",
    "DispersionResult",
    "FMRResult",
    # FMR models
    "kittel",
    "kittel_oop",
    "kittel_exchange",
    # Dispersion models
    "kalinikos",
    "kalinikos_no_approx",
    "damon_eshbach",
    "backward_volume",
    "forward_volume",
    "bottcher",
    "kim",
    "cortes_ortuno",
]
