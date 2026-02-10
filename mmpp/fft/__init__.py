"""
FFT Module

Provides comprehensive FFT analysis capabilities similar to numpy.fft.
Main entry point through the FFT class.
"""

from .compute_fft import FFTCompute, FFTComputeResult
from .core import FFT
from .filters import FilterConfig, FilterPipeline, PostprocessConfig, PreprocessConfig
from .spectrum import MultiSpectrumResult, SpectrumFilterChain, SpectrumResult
from .transmission import (
    TransmissionConfig,
    TransmissionResult,
    TransmissionPlotConfig,
    TransmissionPlotter,
)
from .transmission.interface import FFTTransmissionInterface

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
        "SpectrumResult",
        "MultiSpectrumResult",
        "SpectrumFilterChain",
        "FilterConfig",
        "PreprocessConfig",
        "PostprocessConfig",
        "FilterPipeline",
        "FFTTransmissionInterface",
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
        "TransmissionConfig",
        "TransmissionResult",
        "TransmissionPlotConfig",
        "TransmissionPlotter",
    ]
except ImportError:
    __all__ = [
        "FFT",
        "FFTCompute",
        "FFTComputeResult",
        "SpectrumFilterChain",
        "FilterConfig",
        "PreprocessConfig",
        "PostprocessConfig",
        "FilterPipeline",
        "FFTTransmissionInterface",
        "TransmissionConfig",
        "TransmissionResult",
        "TransmissionPlotConfig",
        "TransmissionPlotter",
    ]
