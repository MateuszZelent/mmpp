"""
Internal submodules for FFTDispersionInterface.

This package contains separated concerns for better code organization:
- cache: Cache & zarr operations
- k0_filtering: k≈0 dynamic filtering

Some planned submodules (e.g. colornorm/plotting/html_display) are not yet
implemented in this repository. Imports are therefore best-effort to avoid
breaking consumers that import this internal package.
"""

from .cache import CacheManager

try:
    from .k0_filtering import K0Filter
except Exception:  # noqa: BLE001 - optional internal helper
    K0Filter = None  # type: ignore[assignment]

try:
    from .colornorm import ColorNormResolver  # type: ignore
except Exception:  # noqa: BLE001 - optional internal helper
    ColorNormResolver = None  # type: ignore[assignment]

try:
    from .plotting import DispersionPlotter  # type: ignore
except Exception:  # noqa: BLE001 - optional internal helper
    DispersionPlotter = None  # type: ignore[assignment]

try:
    from .html_display import HTMLDisplayHelper  # type: ignore
except Exception:  # noqa: BLE001 - optional internal helper
    HTMLDisplayHelper = None  # type: ignore[assignment]

__all__ = ["CacheManager"]
if K0Filter is not None:
    __all__.append("K0Filter")
if ColorNormResolver is not None:
    __all__.append("ColorNormResolver")
if DispersionPlotter is not None:
    __all__.append("DispersionPlotter")
if HTMLDisplayHelper is not None:
    __all__.append("HTMLDisplayHelper")
