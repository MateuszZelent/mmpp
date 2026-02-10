"""Internal helpers for FFT method execution.

These helpers keep :mod:`mmpp.fft.compute_fft` thinner while preserving the
public ``FFTCompute`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class MethodExecutionResult:
    """Execution output shared by FFT method implementations."""

    frequencies: np.ndarray
    spectrum: np.ndarray
    fft_length: int
    selected_engine: str
    calculation_time: float


def run_fft_method1(
    *,
    data: np.ndarray,
    dt: float,
    window: str,
    filter_type: Any,
    engine: str | None,
    zero_padding: bool,
    nfft: int | None,
    determine_engine: Callable[[int], str],
    apply_filter: Callable[[np.ndarray, Any], np.ndarray],
    apply_window: Callable[[np.ndarray, str], np.ndarray],
    compute_fft: Callable[..., tuple[np.ndarray, np.ndarray, int]],
) -> MethodExecutionResult:
    """FFT method 1: filter+window -> FFT -> spatial average."""
    start_time = time.time()
    selected_engine = engine or determine_engine(data.size)

    data_filtered = apply_filter(data, filter_type)
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
        spatial_axes = tuple(range(1, spectrum.ndim - 1))
        if spatial_axes:
            spectrum = np.mean(spectrum, axis=spatial_axes)

    return MethodExecutionResult(
        frequencies=frequencies,
        spectrum=spectrum,
        fft_length=fft_length,
        selected_engine=selected_engine,
        calculation_time=time.time() - start_time,
    )


def run_fft_method2(
    *,
    data: np.ndarray,
    dt: float,
    window: str,
    filter_type: Any,
    engine: str | None,
    zero_padding: bool,
    nfft: int | None,
    determine_engine: Callable[[int], str],
    apply_filter: Callable[[np.ndarray, Any], np.ndarray],
    apply_window: Callable[[np.ndarray, str], np.ndarray],
    compute_fft: Callable[..., tuple[np.ndarray, np.ndarray, int]],
) -> MethodExecutionResult:
    """FFT method 2: filter -> spatial average -> window -> FFT."""
    start_time = time.time()
    selected_engine = engine or determine_engine(data.size)

    data_filtered = apply_filter(data, filter_type)
    if data_filtered.ndim > 2:
        spatial_axes = tuple(range(1, data_filtered.ndim - 1))
        data_averaged = np.mean(data_filtered, axis=spatial_axes) if spatial_axes else data_filtered
    else:
        data_averaged = data_filtered

    data_windowed = apply_window(data_averaged, window)
    frequencies, fft_data, fft_length = compute_fft(
        data_windowed,
        dt,
        selected_engine,
        zero_padding=zero_padding,
        nfft=nfft,
    )

    return MethodExecutionResult(
        frequencies=frequencies,
        spectrum=fft_data,
        fft_length=fft_length,
        selected_engine=selected_engine,
        calculation_time=time.time() - start_time,
    )


def build_fft_metadata(
    *,
    method: int,
    window: str,
    filter_type: Any,
    selected_engine: str,
    zero_padding: bool,
    nfft: int | None,
    calculation_time: float,
    data_shape: tuple[int, ...],
    dt: float,
    frequencies: np.ndarray,
    fft_length: int,
) -> dict[str, Any]:
    """Build stable metadata for FFT method runs."""
    return {
        "method": method,
        "window": window,
        "filter_type": filter_type,
        "engine": selected_engine,
        "zero_padding": zero_padding,
        "nfft_requested": nfft,
        "calculation_time": calculation_time,
        "data_shape": data_shape,
        "dt": dt,
        "frequency_resolution": (
            float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.0
        ),
        "fft_length": fft_length,
    }

