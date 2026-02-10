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

from .base import AnalyticalResult, DispersionResult, FMRResult
from .constants import G_FACTOR_DEFAULT, GAMMA_E, MU0, gamma
from .dispersion import (
    backward_volume,
    bottcher,
    cortes_ortuno,
    damon_eshbach,
    forward_volume,
    kalinikos,
    kalinikos_no_approx,
    kim,
)
from .fmr import (
    kittel,
    kittel_exchange,
    kittel_oop,
)
from .thiele import (
    CIPThieleModel,
    CPPThieleModel,
    DiskGeometry,
    MaterialParams,
    ThieleFJFitResult,
    ThieleOptimizationResult,
    ThieleTrajectoryResult,
    current_ac,
    current_dc,
    current_pulse,
    ellipse_area,
    f0_novosad_ghz,
    fit_omega0_N_to_fJ,
    omega0_novosad,
    slonczewski_mtj_efficiency,
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
    # Thiele vortex dynamics
    "MaterialParams",
    "DiskGeometry",
    "ThieleTrajectoryResult",
    "ThieleFJFitResult",
    "ThieleOptimizationResult",
    "CIPThieleModel",
    "CPPThieleModel",
    "ellipse_area",
    "slonczewski_mtj_efficiency",
    "fit_omega0_N_to_fJ",
    "current_dc",
    "current_ac",
    "current_pulse",
    "omega0_novosad",
    "f0_novosad_ghz",
]
