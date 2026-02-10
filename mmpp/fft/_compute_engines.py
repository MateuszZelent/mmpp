"""Internal engine selection and FFT execution helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def determine_engine_name(
    *,
    configured_engine: str,
    data_size: int,
    scipy_available: bool,
    pyfftw_available: bool,
) -> str:
    """Determine best FFT engine from configuration and data size."""
    if configured_engine != "auto":
        return configured_engine

    if data_size < 100000:
        return "numpy"
    if data_size > 1000000 and pyfftw_available:
        return "pyfftw"
    if scipy_available:
        return "scipy"
    return "numpy"


def compute_fft_data(
    *,
    data: np.ndarray,
    dt: float,
    engine: str,
    zero_padding: bool,
    nfft: int | None,
    scipy_available: bool,
    pyfftw_available: bool,
    scipy_module: Any = None,
    pyfftw_module: Any = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Execute RFFT for a selected backend."""
    n = data.shape[0]
    fft_length = n

    if nfft is not None:
        if nfft < n:
            raise ValueError(
                f"Requested nfft ({nfft}) must be greater than or equal to data length ({n})"
            )
        fft_length = nfft
    elif zero_padding:
        next_power_two = 1 << (n - 1).bit_length()
        if next_power_two > n:
            fft_length = next_power_two

    if engine == "numpy":
        fft_data = np.fft.rfft(data, n=fft_length, axis=0)
        frequencies = np.fft.rfftfreq(fft_length, dt)
    elif engine == "scipy" and scipy_available and scipy_module is not None:
        fft_data = scipy_module.fft.rfft(data, n=fft_length, axis=0)
        frequencies = scipy_module.fft.rfftfreq(fft_length, dt)
    elif engine == "pyfftw" and pyfftw_available and pyfftw_module is not None:
        fft_data = pyfftw_module.interfaces.numpy_fft.rfft(
            data, n=fft_length, axis=0, threads=pyfftw_module.config.NUM_THREADS
        )
        frequencies = pyfftw_module.interfaces.numpy_fft.rfftfreq(fft_length, dt)
    else:
        fft_data = np.fft.rfft(data, n=fft_length, axis=0)
        frequencies = np.fft.rfftfreq(fft_length, dt)

    return frequencies, fft_data, fft_length

