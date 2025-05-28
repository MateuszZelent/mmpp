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

# Import main FFT classes with error handling
try:
    from .main import FFTAnalyzer, FFTConfig
    _MAIN_AVAILABLE = True
except ImportError:
    _MAIN_AVAILABLE = False
    
    class FFTAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError("FFT dependencies not available. Install with: pip install mmpp2[fft]")
    
    class FFTConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("FFT dependencies not available. Install with: pip install mmpp2[fft]")

try:
    from .plot import FFTPlotter
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False
    
    class FFTPlotter:
        def __init__(self, *args, **kwargs):
            raise ImportError("FFT plotting dependencies not available. Install with: pip install mmpp2[fft]")

try:
    from .cache import FFTCache
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    
    class FFTCache:
        def __init__(self, *args, **kwargs):
            raise ImportError("FFT cache dependencies not available.")

try:
    from .console import FFTConsole
    _CONSOLE_AVAILABLE = True
except ImportError:
    _CONSOLE_AVAILABLE = False
    
    class FFTConsole:
        def __init__(self, *args, **kwargs):
            raise ImportError("Rich console dependencies not available. Install with: pip install rich")

# FMR submodule
try:
    from .FMR import FMRAnalyzer, FMRPlotter, FMRModeVisualizer
    _FMR_AVAILABLE = True
except ImportError:
    _FMR_AVAILABLE = False
    
    class FMRAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError("FMR analysis dependencies not available.")
    
    class FMRPlotter:
        def __init__(self, *args, **kwargs):
            raise ImportError("FMR plotting dependencies not available.")
    
    class FMRModeVisualizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("FMR visualization dependencies not available.")

# Main FFT proxy class that integrates with MMPP
class FFTProxy:
    """
    Main FFT interface for MMPP results.
    
    Provides access to all FFT analysis capabilities through a unified interface.
    """
    
    def __init__(self, results, mmpp_instance=None):
        """
        Initialize FFT proxy.
        
        Parameters:
        -----------
        results : List or single result
            ZarrJobResult objects to analyze
        mmpp_instance : MMPP, optional
            Reference to parent MMPP instance
        """
        self._results = results if isinstance(results, list) else [results]
        self._mmpp = mmpp_instance
        self._analyzer = None
        self._plotter = None
        self._cache = None
        self._console = None
        self._fmr = None
    
    @property
    def analyzer(self) -> FFTAnalyzer:
        """Get FFT analyzer instance."""
        if self._analyzer is None:
            self._analyzer = FFTAnalyzer(self._results, self._mmpp)
        return self._analyzer
    
    @property
    def plotter(self) -> FFTPlotter:
        """Get FFT plotter instance."""
        if self._plotter is None:
            self._plotter = FFTPlotter(self._results, self._mmpp)
        return self._plotter
    
    @property
    def cache(self) -> FFTCache:
        """Get FFT cache instance."""
        if self._cache is None:
            self._cache = FFTCache(self._results, self._mmmp)
        return self._cache
    
    @property
    def console(self) -> FFTConsole:
        """Get FFT console for rich data presentation."""
        if self._console is None:
            self._console = FFTConsole(self._results, self._mmpp)
        return self._console
    
    @property
    def fmr(self) -> 'FMRProxy':
        """Get FMR analysis interface."""
        if self._fmr is None:
            self._fmr = FMRProxy(self._results, self._mmpp)
        return self._fmr
    
    def __len__(self) -> int:
        return len(self._results)
    
    def __getitem__(self, index):
        return self._results[index]
    
    def __iter__(self):
        return iter(self._results)
    
    def __repr__(self) -> str:
        return f"FFTProxy({len(self._results)} results)"

class FMRProxy:
    """
    FMR analysis interface.
    
    Provides access to FMR-specific analysis and visualization capabilities.
    """
    
    def __init__(self, results, mmpp_instance=None):
        self._results = results if isinstance(results, list) else [results]
        self._mmpp = mmpp_instance
        self._analyzer = None
        self._plotter = None
        self._visualizer = None
    
    @property
    def analyzer(self) -> FMRAnalyzer:
        """Get FMR analyzer instance."""
        if self._analyzer is None:
            self._analyzer = FMRAnalyzer(self._results, self._mmpp)
        return self._analyzer
    
    @property
    def plotter(self) -> FMRPlotter:
        """Get FMR plotter instance."""
        if self._plotter is None:
            self._plotter = FMRPlotter(self._results, self._mmpp)
        return self._plotter
    
    @property
    def mode_visualization(self) -> FMRModeVisualizer:
        """Get FMR mode visualizer instance."""
        if self._visualizer is None:
            self._visualizer = FMRModeVisualizer(self._results, self._mmpp)
        return self._visualizer
    
    def __call__(self, dset: str = "m_z11", method: str = "fft", engine: str = "numpy", **kwargs):
        """
        Perform FMR analysis.
        
        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        method : str, optional
            Analysis method (default: "fft")
        engine : str, optional
            Computation engine (default: "numpy")
        **kwargs : Any
            Additional analysis parameters
            
        Returns:
        --------
        FMRResult
            FMR analysis results
        """
        return self.analyzer.analyze(dset=dset, method=method, engine=engine, **kwargs)
    
    def __repr__(self) -> str:
        return f"FMRProxy({len(self._results)} results)"

# Export main classes
__all__ = [
    "FFTAnalyzer",
    "FFTConfig", 
    "FFTPlotter",
    "FFTCache",
    "FFTConsole",
    "FMRAnalyzer",
    "FMRPlotter", 
    "FMRModeVisualizer",
    "FFTProxy",
    "FMRProxy",
]

# Feature availability flags
__features__ = {
    "main": _MAIN_AVAILABLE,
    "plot": _PLOT_AVAILABLE,
    "cache": _CACHE_AVAILABLE,
    "console": _CONSOLE_AVAILABLE,
    "fmr": _FMR_AVAILABLE,
}
