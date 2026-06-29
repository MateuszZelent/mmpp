"""
Spin-Wave Dispersion Analysis Module

Provides comprehensive analysis of spin-wave dispersion relations S(k,f)
from micromagnetic simulation data, similar to FMR mode analysis but focused
on wave propagation and k-space dynamics.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Core
    "SpinWaveAnalyzer": (".core", "SpinWaveAnalyzer"),
    "FFTDispersionInterface": (".interface", "FFTDispersionInterface"),
    "DispersionConfig": (".models", "DispersionConfig"),
    "DispersionResult1D": (".models", "DispersionResult1D"),
    "DispersionResult2D": (".models", "DispersionResult2D"),
    "DispersionBranch": (".models", "DispersionBranch"),
    # Utils
    "fftfreq_axis": (".utils", "fftfreq_axis"),
    "fold_k_to_bz": (".utils", "fold_k_to_bz"),
    "fold_spectrum_1d": (".utils", "fold_spectrum_1d"),
    "k_axis_from_grid": (".utils", "k_axis_from_grid"),
    "find_peaks_1d": (".utils", "find_peaks_1d"),
    "group_velocity_1d": (".utils", "group_velocity_1d"),
    # COMSOL
    "ComsolDispersionData": (".comsol", "ComsolDispersionData"),
    "read_data_from_comsol": (".comsol", "read_data_from_comsol"),
    # Filtering, plotting, analysis
    "DispersionFilterChain": (".filter_chain", "DispersionFilterChain"),
    "DispersionPlotAccessor": ("._plotting.accessor", "DispersionPlotAccessor"),
    "DispersionAnalyzeAccessor": (".analyze", "DispersionAnalyzeAccessor"),
    "LowestFrequencyResult": (".analyze", "LowestFrequencyResult"),
    "LowestFrequencyPlotAccessor": (".analyze", "LowestFrequencyPlotAccessor"),
    # Bulk scanning
    "BulkMinimumFrequencyResult": (".bulk", "BulkMinimumFrequencyResult"),
    "BulkMinimumPlotAccessor": (".bulk", "BulkMinimumPlotAccessor"),
    "scan_minimum_frequency": (".bulk", "scan_minimum_frequency"),
    # BZ folding and modes
    "BrillouinZoneConfig": (".modes", "BrillouinZoneConfig"),
    "DispersionMode": (".modes", "DispersionMode"),
    "FoldedDispersionResult": (".modes", "FoldedDispersionResult"),
    "BrillouinZoneFolding": (".modes", "BrillouinZoneFolding"),
    "BrillouinZoneDetector": (".modes", "BrillouinZoneDetector"),
    "InteractiveDispersionModes": (".modes", "InteractiveDispersionModes"),
    "SpinWaveModeAnimator": (".modes", "SpinWaveModeAnimator"),
    "extract_amplitude_phase": (".modes", "extract_amplitude_phase"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load public dispersion exports only when they are accessed."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
