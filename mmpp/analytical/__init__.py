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

from . import nonlinear_stno
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
from .field_resolved_thiele import (
    CurrentDrive,
    FieldResolvedCalibration,
    FieldResolvedCPPThieleModel,
    FieldResolvedTrajectoryResult,
    FrequencyExtractionConfig,
    OerstedCalibration,
    SaturationCalibration,
    ThermalCalibration,
)
from .fmr import (
    kittel,
    kittel_exchange,
    kittel_oop,
)
from .nonlinear_stno import (
    SpectrumAnalyzer,
    STNOParameters,
    run_all_sweeps_parallel,
)
from .thiele import (
    CIPThieleModel,
    CPPThieleModel,
    DiskGeometry,
    ExternalField,
    ExternalFieldLike,
    FieldCalibration,
    FieldFunc,
    MaterialParams,
    SlonczewskiCPPReduction,
    ThieleFJFitResult,
    ThieleOptimizationResult,
    ThieleTrajectoryResult,
    current_ac,
    current_dc,
    current_pulse,
    ellipse_area,
    f0_novosad_ghz,
    field_ac,
    field_ac_vector,
    field_dc,
    field_rotating_inplane,
    fit_omega0_N_to_fJ,
    omega0_novosad,
    reduce_mumax_slonczewski_cpp,
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
    "SlonczewskiCPPReduction",
    "DiskGeometry",
    "ExternalField",
    "ExternalFieldLike",
    "FieldCalibration",
    "FieldFunc",
    "ThieleTrajectoryResult",
    "ThieleFJFitResult",
    "ThieleOptimizationResult",
    "CIPThieleModel",
    "CPPThieleModel",
    "ellipse_area",
    "slonczewski_mtj_efficiency",
    "reduce_mumax_slonczewski_cpp",
    "fit_omega0_N_to_fJ",
    "current_dc",
    "current_ac",
    "current_pulse",
    "field_dc",
    "field_ac",
    "field_ac_vector",
    "field_rotating_inplane",
    "omega0_novosad",
    "f0_novosad_ghz",
    "FieldResolvedCPPThieleModel",
    "FieldResolvedCalibration",
    "FieldResolvedTrajectoryResult",
    "CurrentDrive",
    "SaturationCalibration",
    "OerstedCalibration",
    "ThermalCalibration",
    "FrequencyExtractionConfig",
    # Nonlinear STNO dynamics (DBAT 2.0)
    "nonlinear_stno",
    "STNOParameters",
    "run_all_sweeps_parallel",
    "SpectrumAnalyzer",
    "DashboardPlotter",
]


def __getattr__(name):
    """Lazily expose plotting-only analytical helpers."""
    if name == "DashboardPlotter":
        from .nonlinear_stno import DashboardPlotter

        globals()[name] = DashboardPlotter
        return DashboardPlotter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
