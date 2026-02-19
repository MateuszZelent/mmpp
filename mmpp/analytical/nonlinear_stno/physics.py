"""
mmpp.analytical.nonlinear_stno.physics
=======================================
Material physics and DBAT 2.0 parameter definitions for STNO vortex dynamics.

This module provides the STNOParameters dataclass that encapsulates all
micromagnetic properties, coupling constants and field-dependent arrays
required by the numerical engine.

Example
-------
>>> from mmpp.analytical.nonlinear_stno import STNOParameters
>>> device = STNOParameters()
>>> device.g_nl = 2000.0   # override nonlinear shock multiplier
>>> p = device.get_numba_constants()
>>> w0, N, d0, d1, wsw, chi = device.evaluate_field_arrays(H_array)
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class STNOParameters:
    """Klasa definiująca właściwości mikromagnetycznych i sprzężeń (DBAT 2.0)."""
    
    # 1. Stałe fizyczne i geometria (System SI)
    gamma: float = 1.76e11
    mu0: float = 4 * np.pi * 1e-7
    Ms: float = 8.0e5
    d_disk: float = 250e-9
    L_th: float = 10e-9
    Rc: float = 5e-9
    alpha: float = 0.01
    P_pol: float = 0.4
    hbar: float = 1.054e-34
    e_charge: float = 1.602e-19
    
    # 2. Dual Back-Action Theory (DBAT 2.0) - Zjawiska Nieliniowe
    eta_inc: float = 2.0       # [1/s] Tłumienie od magnonów (Incoherent Fading Memory)
    g_lin: float = 1.5         # [-] Liniowe sprzężenie inercyjne
    g_nl: float = 1500.0       # [-] Nieliniowy mnożnik szoku kinematycznego Macha
    mu_sw: float = 600.0       # [-] Siła odczytu TMR fali spinowej
    xi_xpm: float = 0.05       # [-] Cross-Phase Modulation (XPM)
    kappa_coh: float = 1.2e8   # [rad/s] Coherent Phase-Pulling (Locking)
    v_limit: float = 5e10      # [1/s] Relatywistyczny limit kinetyczny
    
    # 3. Zależności Polowe (Zeeman + Field-Like Torque)
    p_core: float = 1.0
    lam_w0_1: float = 3.0;  lam_w0_2: float = 0.0
    lam_N_1: float = 0.0;   lam_N_2: float = 0.0
    lam_d0_1: float = 0.0;  lam_d0_2: float = 0.0
    lam_d1_1: float = 0.0;  lam_d1_2: float = 0.0
    a_wsw_1: float = 2 * np.pi * 5e9
    a_wsw_2: float = 0.0
    lam_chi_1: float = 2.5     

    def __post_init__(self):
        self._recompute_derived()

    def _recompute_derived(self):
        """Przelicza stałe pochodne po inicjalizacji lub po ręcznej zmianie atrybutów."""
        self.R = self.d_disk / 2.0
        self.beta_param = self.L_th / self.R
        self.w_M = self.gamma * self.mu0 * self.Ms
        
        self.w0_base = (5.0 / (9.0 * np.pi)) * self.w_M * self.beta_param
        self.N_base = 0.85
        
        self.d0_base = self.alpha * (5.0 + 4.0 * np.log(self.R / self.Rc)) / 8.0
        self.d1_base = (11.0 / 6.0) * self.alpha
        
        self.sigma = (self.hbar * self.P_pol) / (2 * self.e_charge * self.L_th * self.Ms)
        self.chi_pref = self.gamma * self.sigma / 2.0
        
        self.w_sw_base = self.w_M * np.sqrt(self.beta_param / 2.0)
        self.G_sw = 0.015 * self.w_sw_base

    def get_numba_constants(self):
        """Zwraca skompilowaną tablicę parametrów na potrzeby silnika JIT."""
        return np.array([
            self.chi_pref, self.eta_inc, self.g_lin, self.g_nl, 
            self.G_sw, self.mu_sw, self.xi_xpm, self.kappa_coh, self.v_limit
        ], dtype=np.float64)

    def evaluate_field_arrays(self, H_array):
        """Transformuje pole magnetyczne [T] na wektory nieliniowych parametrów operacyjnych."""
        w0_min = 0.15 * self.w0_base
        wsw_min = 0.20 * self.w_sw_base

        w0_arr = np.maximum(self.w0_base * (1.0 + self.p_core * self.lam_w0_1 * H_array + self.lam_w0_2 * H_array**2), w0_min)
        N_arr = self.N_base * (1.0 + self.lam_N_1 * H_array + self.lam_N_2 * H_array**2)
        d0_arr = self.d0_base * (1.0 + self.lam_d0_1 * H_array + self.lam_d0_2 * H_array**2)
        d1_arr = self.d1_base * (1.0 + self.lam_d1_1 * H_array + self.lam_d1_2 * H_array**2)
        wsw_arr = np.maximum(self.w_sw_base + self.a_wsw_1 * H_array + self.a_wsw_2 * H_array**2, wsw_min)
        chi_arr = self.chi_pref * (1.0 + self.lam_chi_1 * H_array)

        return w0_arr, N_arr, d0_arr, d1_arr, wsw_arr, chi_arr

    def analytical_carrier_exact(self, J_dc, H_val):
        """Rozwiązanie stacjonarne modelu. Zwraca częstotliwość f_G [Hz]."""
        w0, N, d0, d1, _, chi = self.evaluate_field_arrays(np.array([H_val]))
        w0, N, d0, d1, chi = w0[0], N[0], d0[0], d1[0], chi[0]
        
        A = d1 * N
        B = d0 * N + d1
        C = d0 - (chi * J_dc) / w0
        
        if abs(A) < 1e-25:
            q = max(-C/B, 0.0) if abs(B) > 1e-25 else 0.0
        else:
            delta = B*B - 4*A*C
            if delta < 0: return w0 / (2*np.pi)
            q = max((-B + np.sqrt(delta)) / (2*A), 0.0)
            
        return (w0 * (1.0 + N * q)) / (2*np.pi)
