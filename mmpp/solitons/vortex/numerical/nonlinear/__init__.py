"""Numerical-nonlinear namespace bridge for vortex dynamics."""

from .amplitude_equation import compute_amplitude_equation
from .interactive import ThieleInteractiveDashboard, build_thiele_dashboard
from .interface import NonlinearInterface
from .models import (
    AmplitudeEquationResult,
    STBatchResult,
    STParametersResult,
    ThieleForceBalanceResult,
)
from .nonliniearthiele import ThieleAnalyzer
from .slavin_tiberkevich import extract_st_parameters

__all__ = [
    "NonlinearInterface",
    "AmplitudeEquationResult",
    "STParametersResult",
    "STBatchResult",
    "ThieleForceBalanceResult",
    "ThieleAnalyzer",
    "ThieleInteractiveDashboard",
    "build_thiele_dashboard",
    "compute_amplitude_equation",
    "extract_st_parameters",
]
