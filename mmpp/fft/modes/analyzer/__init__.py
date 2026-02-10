"""
Analyzer sub-package for FMR mode analysis.

This package contains the core analysis components split into
focused modules for better maintainability.
"""

from .cache import ModeCache

DataAccessMixin = None
try:  # Optional: legacy mixin has extra compatibility dependencies.
    from .data_access import DataAccessMixin
except Exception:  # pragma: no cover - optional import
    DataAccessMixin = None

characterize_mode = characterize_vortex_mode = print_characterization_details = None
try:
    from .mode_analysis import (
        characterize_mode,
        characterize_vortex_mode,
        print_characterization_details,
    )
except Exception:  # pragma: no cover - optional import
    characterize_mode = characterize_vortex_mode = print_characterization_details = None

__all__ = [
    'ModeCache',
]

if DataAccessMixin is not None:
    __all__.append('DataAccessMixin')
if characterize_mode is not None:
    __all__.extend(
        [
            'characterize_mode',
            'characterize_vortex_mode',
            'print_characterization_details',
        ]
    )
