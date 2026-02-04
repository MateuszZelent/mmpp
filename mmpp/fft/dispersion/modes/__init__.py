"""
Interactive Dispersion Modes Analysis Module.

Provides tools for Brillouin zone folding, automatic parameter detection,
and interactive visualization of spin-wave dispersion modes in magnonic crystals.

Main Features:
- BZ folding with band structure visualization
- Automatic detection of lattice constants
- Interactive Jupyter widgets for exploration
- Mode tracking and characterization

Usage:
------
>>> job[0].fft.dispersion.dispersion_modes.plot_interactive()
>>> job[0].fft.dispersion.dispersion_modes.fold(lattice_constant=470e-9)
"""

from .models import (
    BrillouinZoneConfig,
    DispersionMode,
    FoldedDispersionResult,
)
from .folding import BrillouinZoneFolding
from .detection import BrillouinZoneDetector
from .interactive import InteractiveDispersionModes

__all__ = [
    "BrillouinZoneConfig",
    "DispersionMode",
    "FoldedDispersionResult",
    "BrillouinZoneFolding",
    "BrillouinZoneDetector",
    "InteractiveDispersionModes",
]
