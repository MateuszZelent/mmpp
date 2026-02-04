"""
Internal submodules for FFTDispersionInterface.

This package contains separated concerns for better code organization:
- cache: Cache & zarr operations
- colornorm: Color normalization utilities
- k0_filtering: k≈0 dynamic filtering
- plotting: Plot generation methods
- html_display: HTML representation & help
"""

from .cache import CacheManager
from .colornorm import ColorNormResolver
from .k0_filtering import K0Filter
from .plotting import DispersionPlotter
from .html_display import HTMLDisplayHelper

__all__ = [
    "CacheManager",
    "ColorNormResolver",
    "K0Filter",
    "DispersionPlotter",
    "HTMLDisplayHelper",
]
