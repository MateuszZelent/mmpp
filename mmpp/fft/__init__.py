"""
FFT Module

Provides comprehensive FFT analysis capabilities similar to numpy.fft.
Main entry point through the FFT class.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "FFT": (".core", "FFT"),
    "FFTCompute": (".compute_fft", "FFTCompute"),
    "FFTComputeResult": (".compute_fft", "FFTComputeResult"),
    "FilterConfig": (".filters", "FilterConfig"),
    "FilterPipeline": (".filters", "FilterPipeline"),
    "PostprocessConfig": (".filters", "PostprocessConfig"),
    "PreprocessConfig": (".filters", "PreprocessConfig"),
    "MultiSpectrumResult": (".spectrum", "MultiSpectrumResult"),
    "SpectrumFilterChain": (".spectrum", "SpectrumFilterChain"),
    "SpectrumResult": (".spectrum", "SpectrumResult"),
    "TransmissionConfig": (".transmission", "TransmissionConfig"),
    "TransmissionPlotConfig": (".transmission", "TransmissionPlotConfig"),
    "TransmissionPlotter": (".transmission", "TransmissionPlotter"),
    "TransmissionResult": (".transmission", "TransmissionResult"),
    "FFTTransmissionInterface": (
        ".transmission.interface",
        "FFTTransmissionInterface",
    ),
    "FFTModeInterface": (".modes", "FFTModeInterface"),
    "FMRModeAnalyzer": (".modes", "FMRModeAnalyzer"),
    "ModeVisualizationConfig": (".modes", "ModeVisualizationConfig"),
    "ModeCharacterAnalyzer": (".mode_characterization", "ModeCharacterAnalyzer"),
    "ModeCharacteristicConfig": (
        ".mode_characterization",
        "ModeCharacteristicConfig",
    ),
    "ModeCharacterizationResult": (
        ".mode_characterization",
        "ModeCharacterizationResult",
    ),
    "SpinWaveAnalyzer": (".dispersion", "SpinWaveAnalyzer"),
    "DispersionResult1D": (".dispersion", "DispersionResult1D"),
    "DispersionResult2D": (".dispersion", "DispersionResult2D"),
    "DispersionBranch": (".dispersion", "DispersionBranch"),
    "DispersionConfig": (".dispersion", "DispersionConfig"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load public FFT exports only when they are accessed."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
