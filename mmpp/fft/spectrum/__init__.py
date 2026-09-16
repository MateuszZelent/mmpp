"""Spectrum result abstractions and fluent plotting API.

The public classes are resolved lazily so importing the FFT computation path
does not import optional plotting dependencies.  Plotting remains available as
soon as the corresponding class is accessed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "SpectrumResult": (".result", "SpectrumResult"),
    "MultiSpectrumResult": (".multi", "MultiSpectrumResult"),
    "SpectrumHelper": (".helpers", "SpectrumHelper"),
    "SpectrumFilterChain": (".filter_chain", "SpectrumFilterChain"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load public spectrum exports only when they are accessed."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
