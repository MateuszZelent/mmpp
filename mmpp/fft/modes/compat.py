"""Optional dependency detection for mmpp.fft.modes."""
from __future__ import annotations

from typing import Optional

from ...cli.logging_config import get_mmpp_logger

log = get_mmpp_logger("mmpp.fft.modes")

# zarr
try:
    import zarr as _zarr

    ZARR_AVAILABLE = True
except ImportError:
    _zarr = None
    ZARR_AVAILABLE = False
    log.error("Zarr not available - mode analysis disabled")

zarr = _zarr

# pyzfn
try:
    from ...pyzfn import Pyzfn as _Pyzfn

    PYZFN_AVAILABLE = True
except ImportError:
    _Pyzfn = None
    PYZFN_AVAILABLE = False

Pyzfn = _Pyzfn

# Matplotlib figure primitives
try:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    Axes = MouseEvent = Figure = None  # type: ignore
    MATPLOTLIB_AVAILABLE = False
    log.warning("Matplotlib not available - mode visualization disabled")

# Animation support
try:
    from matplotlib.animation import FuncAnimation, PillowWriter

    ANIMATION_AVAILABLE = True
    try:
        from matplotlib.animation import FFMpegWriter

        FFMPEG_AVAILABLE = True
        log.debug("FFmpeg available for MP4 animations")
    except ImportError:
        FFMpegWriter = None  # type: ignore
        FFMPEG_AVAILABLE = False
        log.debug("FFmpeg not available - MP4 animations will fallback to GIF")
except ImportError:
    FuncAnimation = PillowWriter = FFMpegWriter = None  # type: ignore
    ANIMATION_AVAILABLE = False
    FFMPEG_AVAILABLE = False
    log.warning("Animation support not available")

# SciPy peak detection
try:
    from scipy.signal import find_peaks

    SCIPY_AVAILABLE = True
except ImportError:
    find_peaks = None  # type: ignore
    SCIPY_AVAILABLE = False
    log.warning("SciPy not available - peak detection features limited")

# Scientific colormaps
try:
    import cmcrameri.cm as cmc

    CMCRAMERI_AVAILABLE = True
    log.debug("cmcrameri colormaps available")
except ImportError:
    cmc = None  # type: ignore
    CMCRAMERI_AVAILABLE = False
    log.debug("cmcrameri not available - using standard matplotlib colormaps")

try:
    import cmocean

    CMOCEAN_AVAILABLE = True
    log.debug("cmocean colormaps available")
except ImportError:
    cmocean = None  # type: ignore
    CMOCEAN_AVAILABLE = False
    log.debug("cmocean not available - using standard matplotlib colormaps")

# mpl_toolkits enhancements
try:
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    AXES_GRID_AVAILABLE = True
except ImportError:
    AnchoredSizeBar = None  # type: ignore
    AXES_GRID_AVAILABLE = False
    log.warning("mpl_toolkits.axes_grid1 not available - colorbar and scalebar enhancements disabled")

__all__ = [
    "zarr",
    "ZARR_AVAILABLE",
    "Pyzfn",
    "PYZFN_AVAILABLE",
    "Axes",
    "MouseEvent",
    "Figure",
    "MATPLOTLIB_AVAILABLE",
    "FuncAnimation",
    "PillowWriter",
    "FFMpegWriter",
    "ANIMATION_AVAILABLE",
    "FFMPEG_AVAILABLE",
    "find_peaks",
    "SCIPY_AVAILABLE",
    "cmc",
    "CMCRAMERI_AVAILABLE",
    "cmocean",
    "CMOCEAN_AVAILABLE",
    "AnchoredSizeBar",
    "AXES_GRID_AVAILABLE",
]
