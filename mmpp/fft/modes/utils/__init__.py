"""
Utilities for FMR mode analysis.

Helper functions for peak detection, scale bars, and validation.
"""

from .peak_detection import (
    SCIPY_AVAILABLE as PEAK_SCIPY_AVAILABLE,
)
from .peak_detection import (
    detect_peaks,
    detect_peaks_scipy,
    detect_peaks_simple,
)
from .scalebar import (
    calculate_optimal_length,
    format_scalebar_label,
)

__all__ = [
    # Peak detection
    "detect_peaks",
    "detect_peaks_scipy",
    "detect_peaks_simple",
    "PEAK_SCIPY_AVAILABLE",
    # Scale bars
    "calculate_optimal_length",
    "format_scalebar_label",
]
