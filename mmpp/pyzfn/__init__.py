"""pyzfn package.

This package provides functionality for equations, ovf, utils, and the Pyzfn class.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .pyzfn import Pyzfn

_LAZY_MODULES = {"equations", "ovf", "utils"}

__all__ = ["Pyzfn", "equations", "ovf", "utils"]


def __getattr__(name: str) -> Any:
    """Load optional pyzfn helper modules only when requested."""
    if name in _LAZY_MODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
