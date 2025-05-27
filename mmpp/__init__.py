"""
MMPP2 - Micro Magnetic Post Processing Library

A Python library for simulation and analysis of micromagnetic simulations 
with advanced post-processing capabilities.
"""

__version__ = "0.1.0"
__author__ = "Mateusz Zelent"
__email__ = "mateusz.zelent@amu.edu.pl"

# Import main classes with error handling for missing dependencies
try:
    from .core import MMPPAnalyzer, SimulationResult, MMPPConfig
    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class MMPPAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Core dependencies not available. Install with: pip install mmpp2[dev]")
    
    class SimulationResult:
        def __init__(self, *args, **kwargs):
            raise ImportError("Core dependencies not available. Install with: pip install mmpp2[dev]")
    
    class MMPPConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("Core dependencies not available. Install with: pip install mmpp2[dev]")

try:
    from .plotting import MMPPlotter, PlotConfig, PlotterProxy
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class MMPPlotter:
        def __init__(self, *args, **kwargs):
            raise ImportError("Plotting dependencies not available. Install with: pip install mmpp2[plotting]")
    
    class PlotConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("Plotting dependencies not available. Install with: pip install mmpp2[plotting]")
    
    class PlotterProxy:
        def __init__(self, *args, **kwargs):
            raise ImportError("Plotting dependencies not available. Install with: pip install mmpp2[plotting]")

try:
    from .simulation import SimulationManager
    _SIMULATION_AVAILABLE = True
except ImportError:
    _SIMULATION_AVAILABLE = False
    # Create dummy class for graceful degradation
    class SimulationManager:
        def __init__(self, *args, **kwargs):
            raise ImportError("Simulation dependencies not available. Install with: pip install mmpp2[dev]")

# Make main classes available at package level
__all__ = [
    "MMPPAnalyzer",
    "SimulationResult", 
    "MMPPConfig",
    "MMPPlotter",
    "PlotConfig",
    "PlotterProxy",
    "SimulationManager",
]

# Feature availability flags
__features__ = {
    "core": _CORE_AVAILABLE,
    "plotting": _PLOTTING_AVAILABLE,
    "simulation": _SIMULATION_AVAILABLE,
}
