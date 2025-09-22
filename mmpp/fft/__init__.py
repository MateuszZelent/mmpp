"""
FFT Module

Provides comprehensive FFT analysis capabilities similar to numpy.fft.
Main entry point through the FFT class.
"""

from .compute_fft import FFTCompute, FFTComputeResult
from .core import FFT

# Import mode visualization with error handling
try:
    from .modes import FFTModeInterface, FMRModeAnalyzer, ModeVisualizationConfig
    from .mode_characterization import (
        ModeCharacterAnalyzer,
        ModeCharacteristicConfig,
        ModeCharacterizationResult,
    )

    __all__ = [
        "FFT",
        "FFTCompute",
        "FFTComputeResult",
        "FMRModeAnalyzer",
        "FFTModeInterface",
        "ModeVisualizationConfig",
        "ModeCharacterAnalyzer",
        "ModeCharacteristicConfig",
        "ModeCharacterizationResult",
    ]
except ImportError:
    __all__ = ["FFT", "FFTCompute", "FFTComputeResult"]
