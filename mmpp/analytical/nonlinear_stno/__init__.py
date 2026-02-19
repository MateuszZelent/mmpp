"""
mmpp.analytical.nonlinear_stno
================================
Nonlinear STNO (Spin-Torque Nano-Oscillator) vortex dynamics framework
based on Dual Back-Action Theory (DBAT 2.0).

This sub-package implements a modular, separation-of-concerns architecture:

* :mod:`.physics`  – ``STNOParameters`` dataclass (material constants, DBAT 2.0
                      coupling vectors, analytical stationarity).
* :mod:`.engine`   – ``run_all_sweeps_parallel`` JIT-compiled 4D RK4 integrator
                      (*Numba* required for full speed).
* :mod:`.analyzer` – ``SpectrumAnalyzer`` for FFT / PSD extraction.
* :mod:`.plotter`  – ``DashboardPlotter`` for 2×2 publication-quality maps.

Quick-start
-----------
>>> import numpy as np
>>> from mmpp.analytical.nonlinear_stno import (
...     STNOParameters,
...     run_all_sweeps_parallel,
...     SpectrumAnalyzer,
...     DashboardPlotter,
... )
>>>
>>> device = STNOParameters()
>>> H_all = np.zeros(60)
>>> w0, N, d0, d1, wsw, chi = device.evaluate_field_arrays(H_all)
>>> p = device.get_numba_p_const()
>>>
>>> all_V = run_all_sweeps_parallel(
...     Jdc_all, Jac_all, fmod_all,
...     w0, N, d0, d1, wsw, chi,
...     t_max=400e-9, dt_out=0.5e-12, substeps=10, p=p,
... )
>>>
>>> analyzer = SpectrumAnalyzer(dt_out=0.5e-12)
>>> f_axis, psd_db = analyzer.compute_psd(all_V)
>>>
>>> DashboardPlotter().plot_2x2(
...     f_axis, map_Jac, map_fmod, map_Jdc, map_Field,
...     sweeps, theory_lines,
... )
"""

from .physics import STNOParameters
from .engine import run_all_sweeps_parallel
from .analyzer import SpectrumAnalyzer
from .plotter import DashboardPlotter

__version__ = "1.0.0"

__all__ = [
    "STNOParameters",
    "run_all_sweeps_parallel",
    "SpectrumAnalyzer",
    "DashboardPlotter",
]
