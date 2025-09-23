"""
mmpp - Micro Magnetic Post Processing Library

A Python library for simulation and analysis with advanced post-processing capabilities.
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
    from .cli.swap.simulation import SimulationManager, SimulationSwapper

    _SIMULATION_AVAILABLE = True
except ImportError:
    _SIMULATION_AVAILABLE = False

    # Create dummy class for graceful degradation
    class SimulationManager:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Simulation dependencies not available. Install with: pip install mmpp[dev]"
            )

    class SimulationSwapper:
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


def install_ffmpeg(force: bool = False, verbose: bool = True) -> str:
    """
    Install FFmpeg for MMPP animation functionality and configure system PATH.
    
    This function installs FFmpeg binary and ensures it's available for matplotlib
    animations. It handles system PATH configuration and matplotlib writer setup.
    
    Parameters:
    -----------
    force : bool, default False
        Force reinstallation even if FFmpeg is already available
    verbose : bool, default True
        Print installation progress and diagnostics
        
    Returns:
    --------
    str
        Path to the installed FFmpeg binary
        
    Raises:
    -------
    ImportError
        If FFT/animation modules are not available
    RuntimeError
        If FFmpeg installation fails
        
    Examples:
    ---------
    >>> import mmpp
    >>> ffmpeg_path = mmpp.install_ffmpeg()
    >>> print(f"FFmpeg installed at: {ffmpeg_path}")
    
    >>> # Force reinstall with verbose output
    >>> mmpp.install_ffmpeg(force=True, verbose=True)
    """
    try:
        from .fft.modes import install_ffmpeg as _install_ffmpeg, check_ffmpeg_installation
        import os
        import sys
        
        if verbose:
            print("🎬 Installing FFmpeg for MMPP animation support...")
        
        # Check current status
        status = check_ffmpeg_installation()
        if status.get('available') and not force:
            ffmpeg_path = status['path']
            if verbose:
                print(f"✅ FFmpeg already available at: {ffmpeg_path}")
                print(f"Version: {status.get('version', 'unknown')}")
            
            # Ensure PATH is configured
            _configure_system_path(ffmpeg_path, verbose=verbose)
            return ffmpeg_path
        
        # Install FFmpeg
        if verbose:
            print("⚡ Installing FFmpeg binary...")
        
        ffmpeg_path = _install_ffmpeg(force=force)
        
        if not ffmpeg_path or not os.path.exists(ffmpeg_path):
            raise RuntimeError("FFmpeg installation failed - binary not found")
        
        # Configure system PATH
        _configure_system_path(ffmpeg_path, verbose=verbose)
        
        # Verify installation
        final_status = check_ffmpeg_installation()
        if not final_status.get('available'):
            raise RuntimeError("FFmpeg installation verification failed")
        
        if verbose:
            print(f"🎉 FFmpeg successfully installed at: {ffmpeg_path}")
            print(f"Version: {final_status.get('version', 'unknown')}")
            print("✅ MMPP animation functionality is now ready!")
        
        return ffmpeg_path
        
    except ImportError:
        raise ImportError(
            "FFT/animation modules not available. Install with: pip install mmpp[animation]"
        )
    except Exception as e:
        raise RuntimeError(f"FFmpeg installation failed: {e}")


def _configure_system_path(ffmpeg_path: str, verbose: bool = True) -> None:
    """Configure system PATH to include FFmpeg directory."""
    import os
    import sys
    
    # Get directory containing FFmpeg
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    
    # Add to Python's PATH for current session
    if ffmpeg_dir not in os.environ.get('PATH', '').split(os.pathsep):
        current_path = os.environ.get('PATH', '')
        os.environ['PATH'] = f"{ffmpeg_dir}{os.pathsep}{current_path}"
        
        if verbose:
            print(f"📁 Added to PATH: {ffmpeg_dir}")
    
    # Configure matplotlib animation writer
    try:
        import matplotlib
        matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        
        # Also set the writer directly for immediate use
        from matplotlib.animation import writers
        if 'ffmpeg' in writers.list():
            # Fix lambda to accept any arguments matplotlib might pass
            writers['ffmpeg'].bin_path = lambda *args, **kwargs: ffmpeg_path
            
        if verbose:
            print(f"🎯 Configured matplotlib animation writer")
            
    except ImportError:
        if verbose:
            print("⚠️  matplotlib not available for writer configuration")
    except Exception as e:
        if verbose:
            print(f"⚠️  matplotlib configuration warning: {e}")


# Make main classes available at package level
__all__ = [
    "MMPP",
    "ScanResult", 
    "ZarrJobResult",
    "MMPPlotter",
    "PlotConfig",
    "PlotterProxy",
    "SimulationManager",
    "SimulationSwapper",
    "open",
    "fonts",  # Font management
    "install_ffmpeg",  # FFmpeg installation
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
