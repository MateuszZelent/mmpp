"""
Compatibility module for optional dependencies in MMPP FFT modes.

This module centralizes all dependency checks and imports with graceful
degradation when optional dependencies are not available.
"""

from typing import Optional
from ...cli.logging_config import get_mmpp_logger

log = get_mmpp_logger(__name__)

# Core dependencies flags
ZARR_AVAILABLE = False
PYZFN_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False
SCIPY_AVAILABLE = False
ANIMATION_AVAILABLE = False
FFMPEG_AVAILABLE = False

# Scientific colormaps flags  
CMCRAMERI_AVAILABLE = False
CMOCEAN_AVAILABLE = False
AXES_GRID_AVAILABLE = False

# Electromagnetic analysis flag
EM_ANALYSIS_AVAILABLE = False

# Import electromagnetic analysis module
try:
    # Electromagnetic analysis capabilities available
    EM_ANALYSIS_AVAILABLE = True
except ImportError:
    EM_ANALYSIS_AVAILABLE = False
    log.warning("Electromagnetic analysis module not available")

# Import core dependencies with error handling
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    log.error("Zarr not available - mode analysis disabled")

try:
    from ...pyzfn import Pyzfn
    PYZFN_AVAILABLE = True
except ImportError:
    PYZFN_AVAILABLE = False

try:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    log.warning("Matplotlib not available - mode visualization disabled")

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    log.warning("SciPy not available - peak detection features limited")

# Check for animation support
try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    ANIMATION_AVAILABLE = True

    # Check for FFmpeg support
    try:
        from matplotlib.animation import FFMpegWriter
        FFMPEG_AVAILABLE = True
        log.debug("FFmpeg available for MP4 animations")
    except ImportError:
        FFMPEG_AVAILABLE = False
        log.debug("FFmpeg not available - MP4 animations will fallback to GIF")

except ImportError:
    ANIMATION_AVAILABLE = False
    FFMPEG_AVAILABLE = False
    log.warning("Animation support not available")

# Check for scientific colormaps
try:
    import cmcrameri.cm as cmc
    CMCRAMERI_AVAILABLE = True
    log.debug("cmcrameri colormaps available")
except ImportError:
    CMCRAMERI_AVAILABLE = False
    log.debug("cmcrameri not available - using standard matplotlib colormaps")

try:
    import cmocean
    CMOCEAN_AVAILABLE = True
    log.debug("cmocean colormaps available")
except ImportError:
    CMOCEAN_AVAILABLE = False
    log.debug("cmocean not available - using standard matplotlib colormaps")

try:
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    AXES_GRID_AVAILABLE = True
except ImportError:
    AXES_GRID_AVAILABLE = False
    log.warning(
        "mpl_toolkits.axes_grid1 not available - colorbar and scalebar enhancements disabled"
    )


def check_required_dependencies():
    """
    Check if all required dependencies are available.
    
    Returns:
    --------
    bool
        True if all required dependencies are available
    """
    required = [ZARR_AVAILABLE, MATPLOTLIB_AVAILABLE]
    return all(required)


def check_animation_support():
    """
    Check if animation features are available.
    
    Returns:
    --------
    bool
        True if animation support is available
    """
    return ANIMATION_AVAILABLE


def check_peak_detection_support():
    """
    Check if peak detection features are available.
    
    Returns:
    --------
    bool
        True if SciPy is available for peak detection
    """
    return SCIPY_AVAILABLE


def get_available_features():
    """
    Get a dictionary of available feature flags.
    
    Returns:
    --------
    dict
        Dictionary mapping feature names to availability flags
    """
    return {
        'zarr': ZARR_AVAILABLE,
        'pyzfn': PYZFN_AVAILABLE,
        'matplotlib': MATPLOTLIB_AVAILABLE,
        'scipy': SCIPY_AVAILABLE,
        'animation': ANIMATION_AVAILABLE,
        'ffmpeg': FFMPEG_AVAILABLE,
        'cmcrameri': CMCRAMERI_AVAILABLE,
        'cmocean': CMOCEAN_AVAILABLE,
        'axes_grid': AXES_GRID_AVAILABLE,
        'em_analysis': EM_ANALYSIS_AVAILABLE
    }


def require_dependency(dependency_name: str, feature_description: Optional[str] = None):
    """
    Raise an error if a required dependency is not available.
    
    Parameters:
    -----------
    dependency_name : str
        Name of the dependency to check
    feature_description : str, optional
        Description of the feature that requires this dependency
    """
    features = get_available_features()
    
    if dependency_name not in features:
        raise ValueError(f"Unknown dependency: {dependency_name}")
    
    if not features[dependency_name]:
        if feature_description:
            raise ImportError(
                f"{dependency_name} is required for {feature_description}. "
                f"Please install it with: pip install {dependency_name}"
            )
        else:
            raise ImportError(
                f"{dependency_name} is not available. "
                f"Please install it with: pip install {dependency_name}"
            )


# Conditional imports based on availability
def get_matplotlib_imports():
    """Get matplotlib imports if available."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.figure import Figure
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    
    return {
        'Axes': Axes,
        'MouseEvent': MouseEvent,
        'Figure': Figure,
        'mcolors': mcolors,
        'plt': plt
    }


def get_animation_imports():
    """Get animation imports if available."""
    if not ANIMATION_AVAILABLE:
        return None
    
    from matplotlib.animation import FuncAnimation, PillowWriter
    imports = {
        'FuncAnimation': FuncAnimation,
        'PillowWriter': PillowWriter
    }
    
    if FFMPEG_AVAILABLE:
        from matplotlib.animation import FFMpegWriter
        imports['FFMpegWriter'] = FFMpegWriter
    
    return imports


def get_scipy_imports():
    """Get SciPy imports if available."""
    if not SCIPY_AVAILABLE:
        return None
    
    from scipy.signal import find_peaks
    return {'find_peaks': find_peaks}


def get_colormap_imports():
    """Get scientific colormap imports if available."""
    imports = {}
    
    if CMCRAMERI_AVAILABLE:
        import cmcrameri.cm as cmc
        imports['cmc'] = cmc
    
    if CMOCEAN_AVAILABLE:
        import cmocean
        imports['cmocean'] = cmocean
    
    return imports if imports else None