"""
Internal submodules for FFTDispersionInterface.

This package contains separated concerns for better code organization:
- k0_filtering: k≈0 dynamic filtering
"""

try:
    from .k0_filtering import K0Filter
except Exception:  # noqa: BLE001 - optional internal helper
    K0Filter = None  # type: ignore[assignment]

__all__: list[str] = []
if K0Filter is not None:
    __all__.append("K0Filter")
