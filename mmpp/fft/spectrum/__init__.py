"""Spectrum result abstractions and fluent plotting API."""

from .filter_chain import SpectrumFilterChain
from .helpers import SpectrumHelper
from .multi import MultiSpectrumResult
from .result import SpectrumResult

__all__ = [
    "SpectrumResult",
    "MultiSpectrumResult",
    "SpectrumHelper",
    "SpectrumFilterChain",
]
