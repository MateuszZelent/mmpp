"""Compatibility bridge for nonlinear analysis result models."""

from ...nonlinear.models import (
    AmplitudeEquationResult,
    STBatchResult,
    STParametersResult,
    ThieleForceBalanceResult,
)

__all__ = [
    "AmplitudeEquationResult",
    "STParametersResult",
    "STBatchResult",
    "ThieleForceBalanceResult",
]
