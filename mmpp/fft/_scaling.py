"""Internal helpers for explicit FFT spectrum scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .filters.windows import get_window

SPECTRUM_SCALINGS = Literal["raw", "continuous_ft", "amplitude", "power", "psd"]
SPECTRUM_KINDS = Literal["complex", "magnitude"]


@dataclass(frozen=True)
class WindowScalingStats:
    """Window statistics needed for amplitude / power / PSD normalization."""

    n_samples: int
    sum_window: float
    sum_window_squared: float


def compute_window_scaling_stats(
    window_type: str, n_samples: int
) -> WindowScalingStats:
    """Compute window sums used for FFT scaling corrections."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive for FFT scaling")

    window = np.asarray(get_window(window_type, n_samples), dtype=float)
    if window.shape != (int(n_samples),):
        raise ValueError(
            f"Window {window_type!r} returned shape {window.shape}, expected "
            f"({int(n_samples)},)"
        )
    if not np.all(np.isfinite(window)):
        raise ValueError(f"Window {window_type!r} contains non-finite values")
    return WindowScalingStats(
        n_samples=int(n_samples),
        sum_window=float(np.sum(window)),
        sum_window_squared=float(np.sum(window**2)),
    )


def _onesided_interior_slice(fft_length: int, n_bins: int) -> slice:
    """Return the slice for one-sided bins that need doubling."""
    if n_bins <= 1:
        return slice(0, 0)
    has_nyquist = fft_length % 2 == 0
    return slice(1, -1 if has_nyquist else None)


def apply_spectrum_scaling(
    *,
    spectrum: np.ndarray,
    scaling: SPECTRUM_SCALINGS,
    dt: float,
    fft_length: int,
    window_stats: WindowScalingStats,
    spectrum_kind_hint: SPECTRUM_KINDS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply requested scaling to an FFT spectrum."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if isinstance(fft_length, (bool, np.bool_)) or not isinstance(
        fft_length, (int, np.integer)
    ):
        raise TypeError(f"fft_length must be an integer, got {fft_length!r}")
    fft_length = int(fft_length)
    if fft_length < int(window_stats.n_samples):
        raise ValueError(
            f"fft_length ({fft_length}) cannot be smaller than the number of "
            f"windowed samples ({window_stats.n_samples})"
        )

    arr = np.asarray(spectrum)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    sum_window = float(window_stats.sum_window)
    sum_window_squared = float(window_stats.sum_window_squared)
    if not np.isfinite(sum_window) or sum_window <= 0:
        raise ValueError("Window sum must be positive for amplitude/power scaling")
    if not np.isfinite(sum_window_squared) or sum_window_squared <= 0:
        raise ValueError("Window energy must be positive for PSD scaling")

    n_bins = int(arr.shape[0])
    expected_bins = fft_length // 2 + 1
    if n_bins != expected_bins:
        raise ValueError(
            f"One-sided spectrum has {n_bins} bins, expected {expected_bins} "
            f"for fft_length={fft_length}"
        )
    interior = _onesided_interior_slice(int(fft_length), n_bins)
    one_sided_bin_factor = np.ones(n_bins, dtype=float)
    one_sided_bin_factor[interior] = 2.0

    metadata: dict[str, Any] = {
        "scaling": scaling,
        "n_samples": int(window_stats.n_samples),
        "window_sum": sum_window,
        "window_sum_squared": sum_window_squared,
        "one_sided": True,
        "spectrum_kind": spectrum_kind_hint,
        "complex_available": spectrum_kind_hint == "complex",
        "phase_available": spectrum_kind_hint == "complex",
        "power_quantity": "raw_power",
    }

    if scaling == "raw":
        return arr, metadata

    if scaling == "continuous_ft":
        metadata["power_quantity"] = "continuous_ft_power"
        return arr * float(dt), metadata

    reshape = (n_bins,) + (1,) * max(arr.ndim - 1, 0)
    bin_factor = one_sided_bin_factor.reshape(reshape)

    if scaling == "amplitude":
        scaled = (arr / sum_window) * bin_factor
        metadata["power_quantity"] = "amplitude_squared"
        return scaled, metadata

    power = np.abs(arr) ** 2
    if scaling == "power":
        scaled_power = (power / (sum_window**2)) * bin_factor
        metadata.update(
            {
                "spectrum_kind": "magnitude",
                "complex_available": False,
                "phase_available": False,
                "power_quantity": "power",
            }
        )
        return np.sqrt(np.clip(scaled_power, 0.0, None)), metadata

    if scaling == "psd":
        fs = 1.0 / float(dt)
        scaled_psd = (power / (fs * sum_window_squared)) * bin_factor
        metadata.update(
            {
                "spectrum_kind": "magnitude",
                "complex_available": False,
                "phase_available": False,
                "power_quantity": "psd",
            }
        )
        return np.sqrt(np.clip(scaled_psd, 0.0, None)), metadata

    raise ValueError(f"Unsupported FFT scaling: {scaling}")
