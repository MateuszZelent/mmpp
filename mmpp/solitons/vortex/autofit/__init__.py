"""Vortex autofit: physics-informed parameter fitting for analytical models."""

from .config import AutofitConfig, ParameterSpec
from .diagnostics import (
    assess_fit_success,
    collapse_guard_penalty,
    cpp_linear_threshold_metrics_from_params,
    cpp_threshold_guard_penalty,
    edge_collision_guard_penalty,
    frequency_guard_penalty,
)
from .interface import AutofitInterface
from .result import AutofitDiagnostics, VortexAutofitResult
from .seeds import (
    build_cpp_threshold_seed_candidates,
    select_threshold_aware_seed,
    unique_seed_candidates,
)
from .simulation import SimulationContext

__all__ = [
    "AutofitInterface",
    "AutofitConfig",
    "ParameterSpec",
    "VortexAutofitResult",
    "AutofitDiagnostics",
    "assess_fit_success",
    "collapse_guard_penalty",
    "cpp_linear_threshold_metrics_from_params",
    "cpp_threshold_guard_penalty",
    "edge_collision_guard_penalty",
    "frequency_guard_penalty",
    "build_cpp_threshold_seed_candidates",
    "select_threshold_aware_seed",
    "unique_seed_candidates",
    "SimulationContext",
]
