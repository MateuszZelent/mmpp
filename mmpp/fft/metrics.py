"""Utility functions for FFT-derived metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PeakWidth:
    """Half-width at half-maximum result for a single spectral peak."""

    peak_index: int
    peak_frequency: float
    peak_value: float
    half_level: float
    left_frequency: float
    right_frequency: float
    width: float


def _as_float_array(data: Iterable[float]) -> np.ndarray:
    """Convert data to a 1D float NumPy array."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 1:
        array = np.ravel(array)
        if array.ndim != 1:
            raise ValueError("Input data must be one-dimensional")
    return array


def compute_half_width_at_half_max(
    frequencies: Iterable[float], values: Iterable[float]
) -> Optional[PeakWidth]:
    """
    Compute the half-width at half-maximum (FWHM) for the dominant peak.

    Parameters
    ----------
    frequencies:
        Iterable of monotonically increasing frequency samples.
    values:
        Iterable of non-negative spectral amplitudes corresponding to the
        ``frequencies`` array.

    Returns
    -------
    PeakWidth or None
        The FWHM description for the highest peak, or ``None`` if it cannot be
        determined (e.g. insufficient samples or non-positive data).
    """
    freqs = _as_float_array(frequencies)
    amps = _as_float_array(values)

    if freqs.size != amps.size:
        raise ValueError("Frequency and amplitude arrays must have the same length")
    if freqs.size < 3:
        return None

    # Handle NaNs or negative values gracefully by zeroing them for peak search
    if np.all(~np.isfinite(amps)) or np.nanmax(amps) <= 0:
        return None

    finite_mask = np.isfinite(amps)
    if not np.all(finite_mask):
        amps = amps.copy()
        amps[~finite_mask] = 0.0

    peak_idx = int(np.nanargmax(amps))
    peak_value = float(amps[peak_idx])
    if peak_value <= 0:
        return None

    half_level = peak_value / 2.0

    # Search left side for first point below half level
    left_idx = peak_idx
    while left_idx > 0 and amps[left_idx] >= half_level:
        left_idx -= 1

    if amps[left_idx] >= half_level:
        # Spectrum does not drop below half level on the left side
        return None

    # Linear interpolation between left_idx and left_idx + 1
    if left_idx == peak_idx:
        left_frequency = float(freqs[peak_idx])
    else:
        x0, y0 = freqs[left_idx], amps[left_idx]
        x1, y1 = freqs[left_idx + 1], amps[left_idx + 1]
        if y1 == y0:
            left_frequency = float((x0 + x1) / 2.0)
        else:
            left_frequency = float(x0 + (half_level - y0) * (x1 - x0) / (y1 - y0))

    # Search right side
    right_idx = peak_idx
    last_index = amps.size - 1
    while right_idx < last_index and amps[right_idx] >= half_level:
        right_idx += 1

    if amps[right_idx] >= half_level:
        return None

    if right_idx == peak_idx:
        right_frequency = float(freqs[peak_idx])
    else:
        x0, y0 = freqs[right_idx - 1], amps[right_idx - 1]
        x1, y1 = freqs[right_idx], amps[right_idx]
        if y1 == y0:
            right_frequency = float((x0 + x1) / 2.0)
        else:
            right_frequency = float(x0 + (half_level - y0) * (x1 - x0) / (y1 - y0))

    width = float(right_frequency - left_frequency)

    return PeakWidth(
        peak_index=peak_idx,
        peak_frequency=float(freqs[peak_idx]),
        peak_value=peak_value,
        half_level=float(half_level),
        left_frequency=left_frequency,
        right_frequency=right_frequency,
        width=width,
    )


def format_width_value(width_ghz: float) -> str:
    """Format a frequency width (in GHz) using adaptive units."""

    if width_ghz >= 0.1:
        return f"{width_ghz:.2f} GHz"
    if width_ghz >= 1e-3:
        return f"{width_ghz * 1e3:.1f} MHz"
    if width_ghz >= 1e-6:
        return f"{width_ghz * 1e6:.1f} kHz"
    return f"{width_ghz * 1e9:.1f} Hz"


def normalize_peak_width_option(value: Any) -> Tuple[bool, str]:
    """Resolve user input for peak-width annotation requests."""

    default_label = "FWHM"
    alias_labels = {"fwhm": "FWHM", "fwhh": "FWHM", "hwfh": "HWFH"}
    positive_tokens = {"1", "true", "yes", "on"}
    negative_tokens = {"0", "false", "no", "off", "none"}

    if value is None:
        return False, default_label

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return False, default_label

        token = normalized.lower()
        if token in negative_tokens:
            return False, default_label
        if token in alias_labels:
            return True, alias_labels[token]
        if token in positive_tokens:
            return True, default_label
        return True, normalized.upper()

    if isinstance(value, (int, float)):
        return (value != 0), default_label

    return (bool(value), default_label if bool(value) else default_label)


__all__ = [
    "PeakWidth",
    "compute_half_width_at_half_max",
    "format_width_value",
    "normalize_peak_width_option",
]
