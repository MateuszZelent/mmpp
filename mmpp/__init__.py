"""
mmpp - Micro Magnetic Post Processing Library

A Python library for simulation and analysis of micromagnetic simulations
with advanced post-processing capabilities.
"""

__version__ = "0.5.3"
__author__ = "Mateusz Zelent"
__email__ = "mateusz.zelent@amu.edu.pl"

# Import main classes with error handling for missing dependencies
try:
    from .core import MMPP, ScanResult, ZarrJobResult

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

    # Create dummy classes for graceful degradation
    class MMPP:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Core dependencies not available. Install with: pip install mmpp[dev]"
            )

    class ScanResult:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Core dependencies not available. Install with: pip install mmpp[dev]"
            )

    class ZarrJobResult:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Core dependencies not available. Install with: pip install mmppp[dev]"
            )


# Try to import plotting classes
try:
    from .plotting import MMPPlotter, PlotConfig, PlotterProxy, fonts

    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

    # Create dummy classes for graceful degradation
    class MMPPlotter:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Plotting dependencies not available. Install with: pip install mmpp[plotting]"
            )

    class PlotConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Plotting dependencies not available. Install with: pip install mmpp[plotting]"
            )

    class PlotterProxy:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Plotting dependencies not available. Install with: pip install mmpp[plotting]"
            )

    # Create dummy font manager
    class DummyFontManager:
        def __init__(self):
            pass

        @property
        def paths(self):
            return []

        @property
        def available(self):
            return []

        def add_path(self, path):
            print("Font management not available - install matplotlib")
            return False

        def set_default_font(self, font):
            print("Font management not available - install matplotlib")
            return False

        def __repr__(self):
            return "FontManager: Not available (matplotlib not installed)"

    fonts = DummyFontManager()


try:
    from .simulation import SimulationManager

    _SIMULATION_AVAILABLE = True
except ImportError:
    _SIMULATION_AVAILABLE = False

    # Create dummy class for graceful degradation
    class SimulationManager:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Simulation dependencies not available. Install with: pip install mmpp[dev]"
            )


def open(base_path: str, **kwargs):
    """
    Open and initialize an MMPP instance for the given directory path.

    This is the main entry point for using the mmpp library. It creates
    an MMPP instance that scans the provided directory for zarr files
    and builds a database for analysis.

    Parameters:
    -----------
    base_path : str
        Path to the directory containing zarr simulation files
    **kwargs : dict
        Additional keyword arguments passed to MMPP constructor:
        - max_workers : int, optional (default: 8)
            Maximum number of worker threads for scanning
        - database_name : str, optional (default: "mmpy_database")
            Name of the database file (without extension)
        - force : bool, optional (default: False)
            If True, force rescan even if database exists

    Returns:
    --------
    MMPP
        An initialized MMPP instance ready for analysis

    Examples:
    ---------
    >>> import mmpp as mp
    >>> db = mp.open("/path/to/simulation/data")
    >>> results = db.find(f0=2.15e+09)
    >>> results.plot("time", "my")  # Current API
    >>> results.mpl.plot("time", "my")  # Short alias
    """
    if not _CORE_AVAILABLE:
        raise ImportError(
            "Core MMPP functionality not available. Install with: pip install mmpp[dev]"
        )

    # Extract force parameter for special handling
    force = kwargs.pop("force", False)

    # Create MMPP instance
    mmpp_instance = MMPP(base_path, **kwargs)

    # If force is True, trigger a rescan
    if force:
        mmpp_instance.force_rescan()
    elif mmpp_instance.dataframe is None:
        # If no database exists, perform initial scan
        mmpp_instance.scan()

    return mmpp_instance


# Make main classes available at package level
__all__ = [
    "MMPP",
    "ScanResult",
    "ZarrJobResult",
    "MMPPlotter",
    "PlotConfig",
    "PlotterProxy",
    "SimulationManager",
    "open",
    "fonts",  # Font management
]

# Feature availability flags
__features__ = {
    "core": _CORE_AVAILABLE,
    "plotting": _PLOTTING_AVAILABLE,
    "simulation": _SIMULATION_AVAILABLE,
    "mmpp": _CORE_AVAILABLE,
}

# Auto-load paper style if available
if _PLOTTING_AVAILABLE:
    try:
        from .plotting import load_paper_style

        load_paper_style(verbose=False)
    except Exception:
        # Silently fail if style loading fails
        pass
