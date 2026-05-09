"""Analytical-vs-numerical vortex comparison public surface.

The implementation currently lives in :mod:`mmpp.solitons.vortex.plotting`
because plotting overlays and comparison metrics share legacy state. This
module provides the canonical import path for comparison-specific objects while
preserving backward compatibility.
"""

from __future__ import annotations

from .plotting import (
    VortexAnalyticalComparison,
    VortexAnalyticalComparisonPlotAccessor,
    VortexAnalyticalMetrics,
    VortexForceBalanceComparison,
    VortexSTComparison,
)

__all__ = [
    "VortexAnalyticalComparison",
    "VortexAnalyticalComparisonPlotAccessor",
    "VortexAnalyticalMetrics",
    "VortexForceBalanceComparison",
    "VortexSTComparison",
]
