"""
FFT Module

Provides comprehensive FFT analysis capabilities similar to numpy.fft.
Main entry point through the FFT class.
"""

from .core import FFT
from .compute_fft import FFTCompute, FFTComputeResult

# Import mode visualization with error handling
try:
    from .modes import FMRModeAnalyzer, FFTModeInterface, ModeVisualizationConfig

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
