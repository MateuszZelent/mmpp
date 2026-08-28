"""
mmpp - Micro Magnetic Post Processing Library

A Python library for simulation and analysis with advanced post-processing capabilities.
"""

import warnings
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Optional, cast

__version__ = "0.5.5"
__author__ = "Mateusz Zelent"
__email__ = "mateusz.zelent@amu.edu.pl"

# Matplotlib emits this warning when `tight_layout()` is called on figures
# with an already-active layout engine; keep logs/tests clean for this known case.
warnings.filterwarnings(
    "ignore",
    message="The figure layout has changed to tight",
    category=UserWarning,
)


def _patch_matplotlib_tight_layout_warning() -> None:
    """Silence Matplotlib tight-layout warning at source for all figures."""
    try:
        from matplotlib.figure import Figure
    except Exception:
        return

    if getattr(Figure.tight_layout, "_mmpp_tight_layout_patched", False):
        return

    original_tight_layout: Any = Figure.tight_layout

    def _tight_layout_without_warning(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The figure layout has changed to tight",
                category=UserWarning,
            )
            return original_tight_layout(self, *args, **kwargs)

    setattr(_tight_layout_without_warning, "_mmpp_tight_layout_patched", True)  # noqa: B010
    Figure.tight_layout = _tight_layout_without_warning  # type: ignore[method-assign]


from . import analyze

# Import main classes with error handling for missing dependencies
try:
    from .core import MMPP, ScanResult, ZarrJobResult

    _CORE_AVAILABLE = True
    _CORE_IMPORT_ERROR = None
except Exception as e:
    _CORE_AVAILABLE = False
    _CORE_IMPORT_ERROR = f"{type(e).__name__}: {e}"

    def _core_repair_hint() -> str:
        if _CORE_IMPORT_ERROR and "numpy.dtype size changed" in _CORE_IMPORT_ERROR:
            return (
                "Detected a binary incompatibility between NumPy and a compiled "
                "dependency such as pandas. Restart the notebook kernel after "
                "activating an environment with consistent numpy/pandas/zarr/"
                "numcodecs/h5py builds. If user-site packages are leaking into "
                "the kernel, start Jupyter with PYTHONNOUSERSITE=1."
            )
        return (
            "Install with: pip install mmpp[dev]\nOr repair the package reported above."
        )

    def _format_core_import_error() -> str:
        error_msg = "Core dependencies not available. "
        if _CORE_IMPORT_ERROR:
            error_msg += f"\nImport error: {_CORE_IMPORT_ERROR}\n"
        error_msg += _core_repair_hint()
        return error_msg

    # Create dummy classes for graceful degradation
    class MMPP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_format_core_import_error())

    class ScanResult:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_format_core_import_error())

    class ZarrJobResult:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_format_core_import_error())


# Backward-compatible / convenience aliases
MMPPAnalyzer = MMPP  # MMPP is the primary analysis entry-point
SimulationResult = ZarrJobResult  # ZarrJobResult represents a single simulation result

try:
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class MMPPConfig:
        """Global MMPP configuration placeholder (reserved for future use)."""

        verbose: bool = False
        cache_enabled: bool = True

    del _dataclass
except Exception:

    class MMPPConfig:  # type: ignore[no-redef]
        pass


_PLOTTING_AVAILABLE = False
_PLOTTING_IMPORT_ERROR = None
_PLOTTING_EXPORTS = {
    "MMPPlotter",
    "PlotConfig",
    "PlotterProxy",
    "fonts",
    "check_fonts",
}


try:
    from .cli.swap.simulation import SimulationManager, SimulationSwapper

    _SIMULATION_AVAILABLE = True
    _SIMULATION_IMPORT_ERROR = None
except ImportError as e:
    _SIMULATION_AVAILABLE = False
    _SIMULATION_IMPORT_ERROR = str(e)

    # Create dummy class for graceful degradation
    class SimulationManager:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            error_msg = "Simulation dependencies not available. "
            if _SIMULATION_IMPORT_ERROR:
                error_msg += f"\nMissing dependency: {_SIMULATION_IMPORT_ERROR}\n"
            error_msg += "Install with: pip install mmpp[dev]\n"
            error_msg += "Or install specific package that is missing above."
            raise ImportError(error_msg)

    class SimulationSwapper:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            error_msg = "Simulation dependencies not available. "
            if _SIMULATION_IMPORT_ERROR:
                error_msg += f"\nMissing dependency: {_SIMULATION_IMPORT_ERROR}\n"
            error_msg += "Install with: pip install mmpp[dev]\n"
            error_msg += "Or install specific package that is missing above."
            raise ImportError(error_msg)


def _module_available(module_name: str) -> tuple[bool, str | None]:
    """Return whether *module_name* can be found without importing it."""
    try:
        return find_spec(module_name) is not None, None
    except Exception as exc:
        return False, str(exc)


def _dependency_group_available(packages: list[str]) -> tuple[bool, str | None]:
    missing = []
    errors = []
    for package in packages:
        available, error = _module_available(package)
        if not available:
            missing.append(package)
            if error:
                errors.append(f"{package}: {error}")

    if not missing:
        return True, None

    detail = ", ".join(missing)
    if errors:
        detail += f" ({'; '.join(errors)})"
    return False, detail


def check_dependencies(verbose: bool = True):
    """
    Check which mmpp dependencies are available and which are missing.

    This function provides a detailed report of installed and missing dependencies,
    helping users diagnose installation issues.

    Returns:
    --------
    dict
        Dictionary with dependency status information

    Examples:
    ---------
    >>> import mmpp
    >>> status = mmpp.check_dependencies()
    >>> print(status)
    """
    fft_available, fft_error = _dependency_group_available(["numpy"])
    dispersion_available, dispersion_error = _dependency_group_available(
        ["numpy", "zarr"]
    )
    interactive_available, interactive_error = _dependency_group_available(
        ["IPython", "ipywidgets"]
    )
    plotting_available, plotting_error = _dependency_group_available(["matplotlib"])
    scipy_available, scipy_error = _dependency_group_available(["scipy"])
    pyfftw_available, pyfftw_error = _dependency_group_available(["pyfftw"])

    status = {
        "core": {
            "available": _CORE_AVAILABLE,
            "error": _CORE_IMPORT_ERROR,
            "required_packages": ["numpy", "pandas", "zarr", "rich"],
        },
        "plotting": {
            "available": plotting_available,
            "error": plotting_error,
            "required_packages": ["matplotlib"],
        },
        "simulation": {
            "available": _SIMULATION_AVAILABLE,
            "error": _SIMULATION_IMPORT_ERROR,
            "required_packages": ["PyYAML"],
        },
        "fft": {
            "available": fft_available,
            "error": fft_error,
            "required_packages": ["numpy"],
        },
        "dispersion": {
            "available": dispersion_available,
            "error": dispersion_error,
            "required_packages": ["numpy", "zarr"],
        },
        "interactive": {
            "available": interactive_available,
            "error": interactive_error,
            "required_packages": ["IPython", "ipywidgets"],
        },
        "scipy": {
            "available": scipy_available,
            "error": scipy_error,
            "required_packages": ["scipy"],
        },
        "pyfftw": {
            "available": pyfftw_available,
            "error": pyfftw_error,
            "required_packages": ["pyfftw"],
        },
    }

    if not verbose:
        return status

    # Print formatted report
    print("=" * 60)
    print("MMPP Dependency Status Report")
    print("=" * 60)

    for module, info in status.items():
        status_icon = "✅" if info["available"] else "❌"
        print(f"\n{status_icon} {module.upper()}")
        print(f"   Available: {info['available']}")

        if not info["available"] and info["error"]:
            print(f"   Error: {info['error']}")

        packages = cast(list[str], info["required_packages"])
        print(f"   Required packages: {', '.join(packages)}")

        if not info["available"]:
            print("   Install with: pip install mmpp[dev]")

    print("\n" + "=" * 60)
    print("Overall Status:")
    all_available = all(info["available"] for info in status.values())
    if all_available:
        print("✅ All dependencies are available!")
    else:
        missing = [name for name, info in status.items() if not info["available"]]
        print(f"❌ Missing: {', '.join(missing)}")
        print("   Run: pip install mmpp[dev]")
    print("=" * 60)

    return status


def __getattr__(name: str):
    """Lazily expose heavyweight optional namespaces."""
    if name == "analytical":
        module = import_module(".analytical", __name__)
        globals()[name] = module
        return module
    if name in _PLOTTING_EXPORTS:
        global _PLOTTING_AVAILABLE, _PLOTTING_IMPORT_ERROR
        try:
            _patch_matplotlib_tight_layout_warning()
            plotting = import_module(".plotting", __name__)
            value = getattr(plotting, name)
            _PLOTTING_AVAILABLE = bool(getattr(plotting, "MATPLOTLIB_AVAILABLE", True))
            _PLOTTING_IMPORT_ERROR = None if _PLOTTING_AVAILABLE else "matplotlib"
        except ImportError as exc:
            _PLOTTING_AVAILABLE = False
            _PLOTTING_IMPORT_ERROR = str(exc)
            raise ImportError(
                f"{name} requires plotting dependencies. "
                "Install with: pip install mmpp[plotting]"
            ) from exc
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        error_msg = "Core MMPP functionality not available."
        if _CORE_IMPORT_ERROR:
            error_msg += f"\nImport error: {_CORE_IMPORT_ERROR}"
        try:
            error_msg += f"\n{_core_repair_hint()}"
        except NameError:
            error_msg += "\nInstall with: pip install mmpp[dev]"
        raise ImportError(error_msg)

    # Extract force parameter for special handling
    force = kwargs.pop("force", False)

    # Create MMPP instance — constructor handles scanning unless force_rescan is set
    mmpp_instance = MMPP(base_path, force_rescan=force, **kwargs)

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
        import os
        import sys

        from .fft.modes import (
            check_ffmpeg_installation,
        )
        from .fft.modes import (
            install_ffmpeg as _install_ffmpeg,
        )

        if verbose:
            print("🎬 Installing FFmpeg for MMPP animation support...")

        # Check current status
        status = check_ffmpeg_installation()
        if status.get("available") and not force:
            ffmpeg_path = status["path"]
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
        if not final_status.get("available"):
            raise RuntimeError("FFmpeg installation verification failed")

        if verbose:
            print(f"🎉 FFmpeg successfully installed at: {ffmpeg_path}")
            print(f"Version: {final_status.get('version', 'unknown')}")
            print("✅ MMPP animation functionality is now ready!")

        return ffmpeg_path

    except ImportError:
        raise ImportError(
            "FFT/animation modules not available. Install with: pip install mmpp[animation]"
        ) from None
    except Exception as e:
        raise RuntimeError(f"FFmpeg installation failed: {e}") from e


def _configure_system_path(ffmpeg_path: str, verbose: bool = True) -> None:
    """Configure system PATH to include FFmpeg directory."""
    import os
    import sys

    # Get directory containing FFmpeg
    ffmpeg_dir = os.path.dirname(ffmpeg_path)

    # Add to Python's PATH for current session
    if ffmpeg_dir not in os.environ.get("PATH", "").split(os.pathsep):
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{current_path}"

        if verbose:
            print(f"📁 Added to PATH: {ffmpeg_dir}")

    # Configure matplotlib animation writer
    try:
        import matplotlib

        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path

        # Also set the writer directly for immediate use
        from matplotlib.animation import writers

        if "ffmpeg" in writers.list():
            # Fix lambda to accept any arguments matplotlib might pass
            writers["ffmpeg"].bin_path = lambda *args, **kwargs: ffmpeg_path  # type: ignore[attr-defined]

        if verbose:
            print("🎯 Configured matplotlib animation writer")

    except ImportError:
        if verbose:
            print("⚠️  matplotlib not available for writer configuration")
    except Exception as e:
        if verbose:
            print(f"⚠️  matplotlib configuration warning: {e}")


# Make main classes available at package level
# ─────────────────────────────────────────────────────────────────────
# Thiele Interactive Dashboard (re-export from solitons module)
# ─────────────────────────────────────────────────────────────────────
try:
    from .solitons.vortex.nonlinear.interactive import build_thiele_dashboard
except Exception as e:
    _SOLITONS_IMPORT_ERROR = f"{type(e).__name__}: {e}"

    # Fallback if solitons module not available
    def build_thiele_dashboard(analyzer: Any | None = None, **kwargs: Any) -> Any:
        """Thiele interactive dashboard (solitons module required)."""
        raise ImportError(
            "build_thiele_dashboard requires solitons module. "
            "Install with: pip install mmpp[solitons]\n"
            f"Import error: {_SOLITONS_IMPORT_ERROR}"
        )


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
    "check_fonts",  # Font diagnostic
    "install_ffmpeg",  # FFmpeg installation
    "check_dependencies",  # Dependency checker
    "analytical",  # Analytical models module
    "build_thiele_dashboard",  # Interactive Thiele dashboard
]

# Feature availability flags
__features__ = {
    "core": _CORE_AVAILABLE,
    "plotting": _PLOTTING_AVAILABLE,
    "simulation": _SIMULATION_AVAILABLE,
    "mmpp": _CORE_AVAILABLE,
}
