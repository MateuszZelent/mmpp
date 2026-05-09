"""Internal helpers for FFT method execution.

These helpers keep :mod:`mmpp.fft.compute_fft` thinner while preserving the
public ``FFTCompute`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from ._scaling import SPECTRUM_SCALINGS, apply_spectrum_scaling, compute_window_scaling_stats


@dataclass(frozen=True)
class MethodExecutionResult:
    """Execution output shared by FFT method implementations."""

    frequencies: np.ndarray
    spectrum: np.ndarray
    fft_length: int
    requested_engine: str
    selected_engine: str
    calculation_time: float
    scaling_metadata: dict[str, Any]


def run_fft_method1(
    *,
    data: np.ndarray,
    dt: float,
    window: str,
    filter_type: Any,
    engine: str | None,
    scaling: SPECTRUM_SCALINGS,
    zero_padding: bool,
    nfft: int | None,
    determine_engine: Callable[[int], str],
    apply_filter: Callable[[np.ndarray, Any], np.ndarray],
    apply_window: Callable[[np.ndarray, str], np.ndarray],
    compute_fft: Callable[..., tuple[np.ndarray, np.ndarray, int]],
) -> MethodExecutionResult:
    """FFT method 1: filter -> spatial average -> window -> FFT.

    Averages the magnetization over space first, then computes FFT
    on the spatially-averaged time series.  Each component is kept
    separately (only y, x axes are averaged).
    """
    start_time = time.time()
    requested_engine = engine or "auto"
    selected_engine = determine_engine(data.size) if requested_engine == "auto" else requested_engine

    data_filtered = apply_filter(data, filter_type)
    if data_filtered.ndim > 2:
        # Average over spatial axes (y, x), keep time (0) and component (-1)
        spatial_axes = tuple(range(1, data_filtered.ndim - 1))
        data_averaged = np.mean(data_filtered, axis=spatial_axes) if spatial_axes else data_filtered
    else:
        data_averaged = data_filtered

    window_stats = compute_window_scaling_stats(window, int(data_averaged.shape[0]))
    data_windowed = apply_window(data_averaged, window)
    frequencies, fft_data, fft_length = compute_fft(
        data_windowed,
        dt,
        selected_engine,
        zero_padding=zero_padding,
        nfft=nfft,
    )
    spectrum, scaling_metadata = apply_spectrum_scaling(
        spectrum=fft_data,
        scaling=scaling,
        dt=dt,
        fft_length=fft_length,
        window_stats=window_stats,
        spectrum_kind_hint="complex",
    )

    return MethodExecutionResult(
        frequencies=frequencies,
        spectrum=spectrum,
        fft_length=fft_length,
        requested_engine=requested_engine,
        selected_engine=selected_engine,
        calculation_time=time.time() - start_time,
        scaling_metadata=scaling_metadata,
    )


def run_fft_method2(
    *,
    data: np.ndarray,
    dt: float,
    window: str,
    filter_type: Any,
    engine: str | None,
    scaling: SPECTRUM_SCALINGS,
    zero_padding: bool,
    nfft: int | None,
    determine_engine: Callable[[int], str],
    apply_filter: Callable[[np.ndarray, Any], np.ndarray],
    apply_window: Callable[[np.ndarray, str], np.ndarray],
    compute_fft: Callable[..., tuple[np.ndarray, np.ndarray, int]],
) -> MethodExecutionResult:
    """FFT method 2: filter+window -> FFT per pixel -> spatial average of |FFT|².

    Computes FFT for every pixel independently, then averages the
    power spectra (|FFT|²) over space.  Each component is kept
    separately (only y, x axes are averaged).  The result is real-valued
    sqrt(mean(|FFT|²)) so that downstream ``np.abs(spectrum)**2``
    still recovers the averaged power.
    """
    start_time = time.time()
    requested_engine = engine or "auto"
    selected_engine = determine_engine(data.size) if requested_engine == "auto" else requested_engine

    data_filtered = apply_filter(data, filter_type)
    window_stats = compute_window_scaling_stats(window, int(data_filtered.shape[0]))
    data_windowed = apply_window(data_filtered, window)

    frequencies, fft_data, fft_length = compute_fft(
        data_windowed,
        dt,
        selected_engine,
        zero_padding=zero_padding,
        nfft=nfft,
    )

    spectrum = fft_data
    if spectrum.ndim > 2:
        # Average over spatial axes (y, x), keep freq (0) and component (-1)
        spatial_axes = tuple(range(1, spectrum.ndim - 1))
        if spatial_axes:
            # Average POWER spectra (|FFT|²) per pixel, then take sqrt.
            # This is physically different from method 1 (average signal, then FFT)
            # because <|FFT(x_i)|²> ≠ |FFT(<x_i>)|² in general.
            power = np.abs(spectrum) ** 2
            spectrum = np.sqrt(np.mean(power, axis=spatial_axes))

    spectrum, scaling_metadata = apply_spectrum_scaling(
        spectrum=spectrum,
        scaling=scaling,
        dt=dt,
        fft_length=fft_length,
        window_stats=window_stats,
        spectrum_kind_hint="magnitude",
    )

    return MethodExecutionResult(
        frequencies=frequencies,
        spectrum=spectrum,
        fft_length=fft_length,
        requested_engine=requested_engine,
        selected_engine=selected_engine,
        calculation_time=time.time() - start_time,
        scaling_metadata=scaling_metadata,
    )


def build_fft_metadata(
    *,
    method: int,
    window: str,
    filter_type: Any,
    requested_engine: str,
    selected_engine: str,
    scaling: SPECTRUM_SCALINGS,
    zero_padding: bool,
    nfft: int | None,
    calculation_time: float,
    data_shape: tuple[int, ...],
    dt: float,
    frequencies: np.ndarray,
    fft_length: int,
    scaling_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build stable metadata for FFT method runs."""
    return {
        "method": method,
        "window": window,
        "filter_type": filter_type,
        "engine_requested": requested_engine,
        "engine_selected": selected_engine,
        "engine": selected_engine,
        "scaling": scaling,
        "zero_padding": zero_padding,
        "nfft_requested": nfft,
        "calculation_time": calculation_time,
        "data_shape": data_shape,
        "dt": dt,
        "frequency_resolution": (
            float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.0
        ),
        "fft_length": fft_length,
        **scaling_metadata,
    }
