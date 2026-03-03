"""
Interactive Dispersion Modes Analysis Module.

Provides tools for Brillouin zone folding, automatic parameter detection,
and interactive visualization of spin-wave dispersion modes in magnonic crystals.

Main Features:
- BZ folding with band structure visualization
- Automatic detection of lattice constants
- Interactive Jupyter widgets for exploration
- Mode tracking and characterization
- Spin wave animation with proper complex amplitude handling

Usage:
------
>>> job[0].fft.dispersion.dispersion_modes.plot_interactive()
>>> job[0].fft.dispersion.dispersion_modes.fold(lattice_constant=470e-9)
>>> job[0].fft.dispersion.dispersion_modes.animate_mode(k=1e6, f=5e9)
"""

from .models import (
    BrillouinZoneConfig,
    DispersionMode,
    FoldedDispersionResult,
)
from .folding import BrillouinZoneFolding
from .detection import BrillouinZoneDetector
from .interactive import InteractiveDispersionModes
from .mode_profile import ModeProfile
from .animation import (
    extract_amplitude_phase,
    compute_spinwave_field,
    generate_animation_frames,
    SpinWaveModeAnimator,
    animate_mode_from_folding,
)
from .bridge import (
    DispersionModesBridge,
    DispersionModeResult,
    DispersionModePlotAccessor,
    DispersionModesPlotAccessor,
)

__all__ = [
    # Models
    "BrillouinZoneConfig",
    "DispersionMode",
    "FoldedDispersionResult",
    "ModeProfile",
    # Core
    "BrillouinZoneFolding",
    "BrillouinZoneDetector",
    "InteractiveDispersionModes",
    # Animation
    "extract_amplitude_phase",
    "compute_spinwave_field",
    "generate_animation_frames",
    "SpinWaveModeAnimator",
    "animate_mode_from_folding",
    # Bridge (new fluent API)
    "DispersionModesBridge",
    "DispersionModeResult",
    "DispersionModePlotAccessor",
    "DispersionModesPlotAccessor",
]
