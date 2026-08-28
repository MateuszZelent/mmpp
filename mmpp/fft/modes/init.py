"""
MMPP FFT Modes package.

This package provides comprehensive FMR mode analysis and visualization
capabilities for micromagnetic simulations.
"""

# Import main components for backward compatibility
from .compat import (
    check_animation_support,
    check_required_dependencies,
    get_available_features,
)
from .config import ModeVisualizationConfig, create_default_config
from .ffmpeg_utils import (
    check_ffmpeg_available,
    check_ffmpeg_installation,
    install_ffmpeg,
    install_ffmpeg_simple,
)
from .models import FMRModeData, Peak
from .style import MidpointNormalize, setup_animation_styling

# Import analyzer components when available
try:
    from . import FMRModeAnalyzer

    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

# Import interfaces when available
try:
    from .interfaces import FFTModeInterface, FrequencyModeInterface

    INTERFACES_AVAILABLE = True
except ImportError:
    INTERFACES_AVAILABLE = False

# Version info
__version__ = "1.0.0"

# Public API
__all__ = [
    # Configuration
    "ModeVisualizationConfig",
    "create_default_config",
    # Data models
    "Peak",
    "FMRModeData",
    # FFmpeg utilities
    "install_ffmpeg",
    "check_ffmpeg_installation",
    "check_ffmpeg_available",
    "install_ffmpeg_simple",
    # Styling
    "MidpointNormalize",
    "setup_animation_styling",
    # Compatibility
    "check_required_dependencies",
    "check_animation_support",
    "get_available_features",
]

# Add analyzer to exports if available
if ANALYZER_AVAILABLE:
    __all__.append("FMRModeAnalyzer")

# Add interfaces to exports if available
if INTERFACES_AVAILABLE:
    __all__.extend(["FFTModeInterface", "FrequencyModeInterface"])


def get_package_info():
    """Get information about the modes package."""
    return {
        "version": __version__,
        "analyzer_available": ANALYZER_AVAILABLE,
        "interfaces_available": INTERFACES_AVAILABLE,
        "features": get_available_features(),
    }
