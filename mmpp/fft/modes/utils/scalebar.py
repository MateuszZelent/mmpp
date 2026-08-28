"""
Scale bar utilities for mode visualizations.

Provides functions for calculating optimal scale bar lengths
and formatting labels with appropriate units.
"""

import logging
import math

log = logging.getLogger("mmpp.fft.modes")


def calculate_optimal_length(width_nm: float) -> float | None:
    """Calculate nice round number for scale bar.

    Picks a sensible scale bar length (1, 2, or 5 × 10^n)
    that is approximately 1/4 of the sample width.

    Parameters
    ----------
    width_nm : float
        Sample width in nanometers

    Returns
    -------
    Optional[float]
        Optimal scale bar length in nm, or None if invalid width

    Examples
    --------
    >>> calculate_optimal_length(1000)  # 1 µm sample
    200.0  # 200 nm scale bar
    >>> calculate_optimal_length(5000)  # 5 µm sample
    1000.0  # 1 µm scale bar
    """
    if width_nm <= 0:
        return None

    # Target approximately 1/4 of width
    target = width_nm / 4
    if target <= 0:
        return None

    # Find order of magnitude
    exponent = math.floor(math.log10(target)) if target > 0 else 0

    # Try nice round numbers: 1, 2, 5 × 10^n
    best = None
    for multiplier in (1, 2, 5):
        candidate = multiplier * (10**exponent)
        # Use at most 90% of width
        if candidate <= width_nm * 0.9:
            best = candidate

    # Fallback: half width
    if best is None:
        best = width_nm / 2

    return round(best, 3)


def format_scalebar_label(length_nm: float, units: str = "nm") -> str:
    """Format scale bar label with appropriate units.

    Automatically converts to µm if length >= 1000 nm.

    Parameters
    ----------
    length_nm : float
        Scale bar length in nanometers
    units : str
        Preferred units (default: "nm")

    Returns
    -------
    str
        Formatted label with value and units

    Examples
    --------
    >>> format_scalebar_label(500)
    '500 nm'
    >>> format_scalebar_label(2000)
    '2 µm'
    >>> format_scalebar_label(1500)
    '1.5 µm'
    """
    units_lower = units.lower()

    # Auto-convert to µm for large values
    if units_lower == "nm" and length_nm >= 1000:
        value_um = length_nm / 1000.0
        return f"{value_um:g} µm"

    return f"{length_nm:g} {units}"


__all__ = [
    "calculate_optimal_length",
    "format_scalebar_label",
]
