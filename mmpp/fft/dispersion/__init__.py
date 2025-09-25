"""
Spin-Wave Dispersion Analysis Module

Provides comprehensive analysis of spin-wave dispersion relations S(k,f) 
from micromagnetic simulation data, similar to FMR mode analysis but focused
on wave propagation and k-space dynamics.

Main Features:
- 1D and 2D dispersion relation computation
- Branch tracking and characterization  
- Brillouin zone folding and analysis
- Peak detection and group velocity calculation
- Integration with MMPP job results and visualization

Usage Examples:
--------------
# Basic dispersion analysis
>>> job[0].fft.dispersion.plot_dispersion()
>>> job[0].m_layer.fft.dispersion.compute_1d(axis="x")

# Advanced analysis
>>> analyzer = job[0].fft.dispersion.analyzer
>>> result = analyzer.compute_dispersion_1d(axis="x")
>>> branch = analyzer.track_branch(result, k_path, f_seed=5e9)
"""


from .core import SpinWaveAnalyzer
from .interface import FFTDispersionInterface
from .models import (
    DispersionConfig,
    DispersionResult1D,
    DispersionResult2D,
    DispersionBranch,
)
from .utils import (
    fftfreq_axis,
    fold_k_to_bz,
    fold_spectrum_1d,
    k_axis_from_grid,
    find_peaks_1d,
    group_velocity_1d,
)

__all__ = [
    "SpinWaveAnalyzer",
    "FFTDispersionInterface",
    "DispersionConfig",
    "DispersionResult1D",
    "DispersionResult2D",
    "DispersionBranch",
    "fftfreq_axis",
    "fold_k_to_bz",
    "fold_spectrum_1d",
    "k_axis_from_grid",
    "find_peaks_1d",
    "group_velocity_1d",
]
