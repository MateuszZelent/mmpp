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
    # Import dispersion analysis
    from .dispersion import (
        SpinWaveAnalyzer,
        DispersionResult1D,
        DispersionResult2D,
        DispersionBranch,
        DispersionConfig
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
        "SpinWaveAnalyzer",
        "DispersionResult1D",
        "DispersionResult2D", 
        "DispersionBranch",
        "DispersionConfig",
    ]
except ImportError:
    __all__ = ["FFT", "FFTCompute", "FFTComputeResult"]
