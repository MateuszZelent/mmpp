"""Selectable FFT backend: scipy (default), pyfftw, or numpy fallback.

All dispersion FFT calls are routed through this module so the backend and
thread count are controlled from a single place.

Configuration
-------------
Environment variables (checked at import time):

``MMPP_FFT_BACKEND``
    One of ``"scipy"`` (default), ``"pyfftw"``, or ``"numpy"``.
``MMPP_FFT_WORKERS``
    Integer number of threads.  ``-1`` (default) = **all CPU cores**.

Programmatic API (can be called at runtime)::

    from mmpp.fft.dispersion._fft_backend import set_backend, set_workers, get_info
    set_backend("pyfftw")   # switch to FFTW
    set_workers(8)           # limit to 8 threads
    print(get_info())        # {'backend': 'pyfftw', 'workers': 8}
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKEND: str = "numpy"  # effective backend name
_WORKERS: int = -1        # -1 = all cores
# FFTW planner effort: FFTW_ESTIMATE (fast, default) or FFTW_MEASURE (slow planning, faster execution)
# FFTW_MEASURE benchmarks hundreds of algorithm variants per array shape — can take MINUTES for large arrays.
# Only use FFTW_MEASURE if you're running the same transform shape many times and can amortize the planning cost.
_PLANNER_EFFORT: str = os.environ.get("MMPP_FFT_PLANNER", "FFTW_ESTIMATE").strip()

# scipy.fft
try:
    import scipy.fft as _sp_fft  # type: ignore[import-untyped]
    _HAS_SCIPY = True
except ImportError:
    _sp_fft = None  # type: ignore[assignment]
    _HAS_SCIPY = False

# pyfftw
try:
    import pyfftw  # type: ignore[import-untyped]
    import pyfftw.interfaces.numpy_fft as _pw_fft  # type: ignore[import-untyped]
    _HAS_PYFFTW = True
    # Enable pyfftw cache for repeated transforms of the same size
    pyfftw.interfaces.cache.enable()
except ImportError:
    _pw_fft = None  # type: ignore[assignment]
    pyfftw = None  # type: ignore[assignment]
    _HAS_PYFFTW = False


def _resolve_default_backend() -> str:
    """Pick the best available backend, respecting MMPP_FFT_BACKEND env var."""
    env = os.environ.get("MMPP_FFT_BACKEND", "").strip().lower()
    if env:
        if env == "pyfftw" and _HAS_PYFFTW:
            return "pyfftw"
        if env == "scipy" and _HAS_SCIPY:
            return "scipy"
        if env == "numpy":
            return "numpy"
        logger.warning(
            "MMPP_FFT_BACKEND='%s' requested but not available; falling back",
            env,
        )
    # auto: prefer scipy > pyfftw > numpy
    if _HAS_SCIPY:
        return "scipy"
    if _HAS_PYFFTW:
        return "pyfftw"
    return "numpy"


def _resolve_default_workers() -> int:
    env = os.environ.get("MMPP_FFT_WORKERS", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return -1  # all cores


_BACKEND = _resolve_default_backend()
_WORKERS = _resolve_default_workers()

logger.debug("FFT backend: %s (workers=%s)", _BACKEND, _WORKERS)


# ---------------------------------------------------------------------------
# Public API for runtime configuration
# ---------------------------------------------------------------------------

def set_backend(name: str) -> None:
    """Switch FFT backend at runtime.

    Parameters
    ----------
    name : ``"scipy"`` | ``"pyfftw"`` | ``"numpy"``
    """
    global _BACKEND
    name = name.strip().lower()
    if name == "scipy" and not _HAS_SCIPY:
        raise ImportError("scipy is not installed")
    if name in ("pyfftw", "fftw") and not _HAS_PYFFTW:
        raise ImportError("pyfftw is not installed (pip install pyfftw)")
    if name in ("pyfftw", "fftw"):
        name = "pyfftw"
    if name not in ("scipy", "pyfftw", "numpy"):
        raise ValueError(f"Unknown backend '{name}'. Use: scipy, pyfftw, numpy")
    _BACKEND = name
    logger.info("FFT backend changed to: %s", _BACKEND)


def set_workers(n: int) -> None:
    """Set number of FFT worker threads.  ``-1`` = all cores."""
    global _WORKERS
    _WORKERS = int(n)
    logger.info("FFT workers set to: %s", _WORKERS)


def get_info() -> dict[str, Any]:
    """Return current FFT backend configuration."""
    return {
        "backend": _BACKEND,
        "workers": _WORKERS,
        "scipy_available": _HAS_SCIPY,
        "pyfftw_available": _HAS_PYFFTW,
    }


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def fft(
    a: np.ndarray,
    n: Optional[int] = None,
    axis: int = -1,
    workers: Optional[int] = None,
) -> np.ndarray:
    """Forward 1-D FFT, multi-threaded when possible."""
    w = workers if workers is not None else _WORKERS
    _threads = os.cpu_count() or 1 if w <= 0 else w
    if _BACKEND == "scipy":
        return _sp_fft.fft(a, n=n, axis=axis, workers=w)  # type: ignore[union-attr]
    if _BACKEND == "pyfftw":
        return _pw_fft.fft(a, n=n, axis=axis, threads=_threads, planner_effort=_PLANNER_EFFORT)  # type: ignore[union-attr]
    return np.fft.fft(a, n=n, axis=axis)


def ifft(
    a: np.ndarray,
    n: Optional[int] = None,
    axis: int = -1,
    workers: Optional[int] = None,
) -> np.ndarray:
    """Inverse 1-D FFT."""
    w = workers if workers is not None else _WORKERS
    _threads = os.cpu_count() or 1 if w <= 0 else w
    if _BACKEND == "scipy":
        return _sp_fft.ifft(a, n=n, axis=axis, workers=w)  # type: ignore[union-attr]
    if _BACKEND == "pyfftw":
        return _pw_fft.ifft(a, n=n, axis=axis, threads=_threads, planner_effort=_PLANNER_EFFORT)  # type: ignore[union-attr]
    return np.fft.ifft(a, n=n, axis=axis)


def fft2(
    a: np.ndarray,
    s: Any = None,
    axes: Any = (-2, -1),
    workers: Optional[int] = None,
) -> np.ndarray:
    """Forward 2-D FFT."""
    w = workers if workers is not None else _WORKERS
    _threads = os.cpu_count() or 1 if w <= 0 else w
    if _BACKEND == "scipy":
        return _sp_fft.fft2(a, s=s, axes=axes, workers=w)  # type: ignore[union-attr]
    if _BACKEND == "pyfftw":
        return _pw_fft.fft2(a, s=s, axes=axes, threads=_threads, planner_effort=_PLANNER_EFFORT)  # type: ignore[union-attr]
    return np.fft.fft2(a, s=s, axes=axes)


def rfft(
    a: np.ndarray,
    n: Optional[int] = None,
    axis: int = -1,
    workers: Optional[int] = None,
) -> np.ndarray:
    """Forward real-input 1-D FFT."""
    w = workers if workers is not None else _WORKERS
    _threads = os.cpu_count() or 1 if w <= 0 else w
    if _BACKEND == "scipy":
        return _sp_fft.rfft(a, n=n, axis=axis, workers=w)  # type: ignore[union-attr]
    if _BACKEND == "pyfftw":
        return _pw_fft.rfft(a, n=n, axis=axis, threads=_threads, planner_effort=_PLANNER_EFFORT)  # type: ignore[union-attr]
    return np.fft.rfft(a, n=n, axis=axis)


# ---------------------------------------------------------------------------
# Frequency / shift helpers (pure arithmetic — no workers needed)
# ---------------------------------------------------------------------------

def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """Frequency axis for length *n* and sample spacing *d*."""
    return np.fft.fftfreq(n, d)


def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """Frequency axis for real-input FFT."""
    return np.fft.rfftfreq(n, d)


def fftshift(x: np.ndarray, axes: Any = None) -> np.ndarray:
    """Shift zero-frequency component to centre."""
    return np.fft.fftshift(x, axes=axes)


def ifftshift(x: np.ndarray, axes: Any = None) -> np.ndarray:
    """Inverse of :func:`fftshift`."""
    return np.fft.ifftshift(x, axes=axes)
