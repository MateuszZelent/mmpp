"""
MMPP Spin-Wave Dispersion Analysis Module

This module provides tools for computing and analyzing spin-wave dispersion relations
S(k,f) from time-domain micromagnetic simulation data (e.g., MuMax3).

Key Features:
- 1D dispersion S(k,f) along specified propagation directions
- 2D dispersion S(kx,ky,f) for full spatial analysis  
- Brillouin zone folding for periodic structures
- Dispersion branch tracking and peak detection
- Group velocity calculations
- Mode classification at specific (k,f) points

Integration with MMPP:
- Uses MMPP data structures and zarr format
- Consistent with FMR mode analysis workflow
- Provides interactive visualization tools
- Supports batch processing across frequencies

Usage:
    from mmpp.fft.dispersion import SpinWaveAnalyzer
    
    analyzer = SpinWaveAnalyzer('simulation.zarr')
    dispersion_1d = analyzer.compute_dispersion_1d(axis='x')
    branches = analyzer.track_branches(dispersion_1d)
"""

from .core import SpinWaveAnalyzer
from .models import (
    DispersionResult1D,
    DispersionResult2D, 
    DispersionBranch,
    DispersionConfig
)
from .utils import (
    fold_k_to_bz,
    fftfreq_axis,
    k_axis_from_grid,
    group_velocity_1d
)

__all__ = [
    'SpinWaveAnalyzer',
    'DispersionResult1D',
    'DispersionResult2D',
    'DispersionBranch', 
    'DispersionConfig',
    'fold_k_to_bz',
    'fftfreq_axis',
    'k_axis_from_grid',
    'group_velocity_1d'
]