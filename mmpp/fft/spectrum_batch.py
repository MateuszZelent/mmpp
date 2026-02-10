"""Backward-compatible shim for batch spectrum API.

Public imports remain available from ``mmpp.fft.spectrum_batch`` while
implementation is split under ``mmpp.fft.spectrum.batch``.
"""

from __future__ import annotations

from .spectrum.batch.compute import BatchSpectrum
from .spectrum.batch.result import BatchSpectrumResult, SpectrumEntry

__all__ = ["BatchSpectrum", "BatchSpectrumResult", "SpectrumEntry"]
