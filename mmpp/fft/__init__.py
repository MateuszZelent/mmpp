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

    __all__ = [
        "FFT",
        "FFTCompute",
        "FFTComputeResult",
        "FMRModeAnalyzer",
        "FFTModeInterface",
        "ModeVisualizationConfig",
    ]
except ImportError:
    __all__ = ["FFT", "FFTCompute", "FFTComputeResult"]
