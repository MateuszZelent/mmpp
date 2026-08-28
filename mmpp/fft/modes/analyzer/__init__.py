"""
Analyzer sub-package for FMR mode analysis.

This package contains the core analysis components split into
focused modules for better maintainability.
"""

from typing import Any

from .cache import ModeCache

DataAccessMixin: Any = None
try:  # Optional: legacy mixin has extra compatibility dependencies.
    from .data_access import DataAccessMixin as _DataAccessMixin

    DataAccessMixin = _DataAccessMixin
except Exception:  # pragma: no cover - optional import
    DataAccessMixin = None

characterize_mode: Any = None
characterize_vortex_mode: Any = None
print_characterization_details: Any = None
try:
    from .mode_analysis import (
        characterize_mode as _characterize_mode,
        characterize_vortex_mode as _characterize_vortex_mode,
        print_characterization_details as _print_characterization_details,
    )

    characterize_mode = _characterize_mode
    characterize_vortex_mode = _characterize_vortex_mode
    print_characterization_details = _print_characterization_details
except Exception:  # pragma: no cover - optional import
    pass

__all__ = [
    "ModeCache",
]

if DataAccessMixin is not None:
    __all__.append("DataAccessMixin")
if characterize_mode is not None:
    __all__.extend(
        [
            "characterize_mode",
            "characterize_vortex_mode",
            "print_characterization_details",
        ]
    )
