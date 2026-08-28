"""Internal engine selection and FFT execution helpers."""

from __future__ import annotations

from numbers import Integral
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
    supported = {"auto", "numpy", "scipy", "pyfftw"}
    if configured_engine not in supported:
        raise ValueError(
            f"Unsupported FFT engine {configured_engine!r}; "
            f"expected one of {sorted(supported)}"
        )
    if configured_engine != "auto":
        if configured_engine == "scipy" and not scipy_available:
            raise ImportError("FFT engine 'scipy' requested but scipy is unavailable")
        if configured_engine == "pyfftw" and not pyfftw_available:
            raise ImportError("FFT engine 'pyfftw' requested but pyfftw is unavailable")
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
    supported = {"numpy", "scipy", "pyfftw"}
    if engine not in supported:
        raise ValueError(
            f"Unsupported FFT engine {engine!r}; expected one of {sorted(supported)}"
        )
    data = np.asarray(data)
    if data.ndim < 1 or data.shape[0] < 1:
        raise ValueError("FFT input must contain at least one time sample")
    if np.iscomplexobj(data):
        raise TypeError(
            "RFFT input must be real-valued; use a complex FFT explicitly for "
            "complex time-domain data"
        )
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be finite and positive, got {dt!r}")
    if not isinstance(zero_padding, (bool, np.bool_)):
        raise TypeError("zero_padding must be a boolean")

    n = int(data.shape[0])
    fft_length = n

    if nfft is not None:
        if isinstance(nfft, (bool, np.bool_)) or not isinstance(nfft, Integral):
            raise TypeError(f"nfft must be an integer or None, got {nfft!r}")
        nfft = int(nfft)
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
    elif engine == "scipy":
        if not scipy_available or scipy_module is None:
            raise ImportError("FFT engine 'scipy' requested but scipy is unavailable")
        fft_data = scipy_module.fft.rfft(data, n=fft_length, axis=0)
        frequencies = scipy_module.fft.rfftfreq(fft_length, dt)
    elif engine == "pyfftw":
        if not pyfftw_available or pyfftw_module is None:
            raise ImportError("FFT engine 'pyfftw' requested but pyfftw is unavailable")
        fft_data = pyfftw_module.interfaces.numpy_fft.rfft(
            data, n=fft_length, axis=0, threads=pyfftw_module.config.NUM_THREADS
        )
        frequencies = pyfftw_module.interfaces.numpy_fft.rfftfreq(fft_length, dt)
    return frequencies, fft_data, fft_length
