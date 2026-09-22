"""Batch spectrum namespace (refactor shim)."""

from .result import BatchSpectrumAnalysis, BatchSpectrumResult, SpectrumEntry

__all__ = [
    "BatchSpectrum",
    "BatchSpectrumAnalysis",
    "BatchSpectrumResult",
    "SpectrumEntry",
]


def __getattr__(name: str):
    if name == "BatchSpectrum":
        from .compute import BatchSpectrum

        return BatchSpectrum
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
