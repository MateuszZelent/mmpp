"""
MMPP2 FFT Analysis Module

Advanced FFT analysis capabilities for micromagnetic simulations including:
- Time series FFT analysis
- FMR (Ferromagnetic Resonance) analysis
- Mode visualization
- Frequency domain analysis
- Power spectral density calculations
"""

__version__ = "0.1.0"

# Import core computation module
try:
    from .compute_fft import FFTCompute, FFTComputeResult, FFTComputeConfig
    _COMPUTE_AVAILABLE = True
except ImportError:
    _COMPUTE_AVAILABLE = False
    
    class FFTCompute:
        def __init__(self, *args, **kwargs):
            raise ImportError("FFT computation dependencies not available. Install with: pip install scipy")

# Import console interface
try:
    from .console import FFTConsole
    _CONSOLE_AVAILABLE = True
except ImportError:
    _CONSOLE_AVAILABLE = False
    
    class FFTConsole:
        def __init__(self, *args, **kwargs):
            raise ImportError("FFT console dependencies not available. Install with: pip install rich")

# Import main FFT classes with error handling
try:
    from .main import FFTAnalyzer, FFTConfig
    _MAIN_AVAILABLE = True
except ImportError:
    _MAIN_AVAILABLE = False
    
    class FFTAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError("FFT dependencies not available. Install with: pip install scipy")
    
    class FFTConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("FFT dependencies not available. Install with: pip install scipy")

try:
    from .plot import FFTPlotter
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False
    
    class FFTPlotter:
        def __init__(self, *args, **kwargs):
            raise ImportError("FFT plotting dependencies not available. Install with: pip install matplotlib")

# Legacy compatibility - keep old_fft_module for backward compatibility
try:
    from .old_fft_module import FMRAnalyzer as LegacyFMRAnalyzer
    _LEGACY_AVAILABLE = True
except ImportError:
    _LEGACY_AVAILABLE = False

# Export main classes
__all__ = [
    "FFTCompute",
    "FFTComputeResult", 
    "FFTComputeConfig",
    "FFTConsole",
    "FFTAnalyzer",
    "FFTConfig", 
    "FFTPlotter",
]

# Legacy exports
if _LEGACY_AVAILABLE:
    __all__.append("LegacyFMRAnalyzer")

# Feature availability flags
__features__ = {
    "compute": _COMPUTE_AVAILABLE,
    "console": _CONSOLE_AVAILABLE,
    "main": _MAIN_AVAILABLE,
    "plot": _PLOT_AVAILABLE,
    "legacy": _LEGACY_AVAILABLE,
}
