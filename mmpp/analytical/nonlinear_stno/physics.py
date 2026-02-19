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
>>> p = device.get_numba_p_const()
>>> w0, N, d0, d1, wsw, chi = device.evaluate_field_arrays(H_array)
"""

import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class STNOParameters:
    """Micromagnetic parameter set with Dual Back-Action Theory (DBAT 2.0).

    Attributes
    ----------
    gamma : float
        Gyromagnetic ratio [rad/(s·T)].
    mu0 : float
        Vacuum permeability [H/m].
    Ms : float
        Saturation magnetisation [A/m].
    d_disk : float
        Disk diameter [m].
    L_th : float
        Magnetic layer thickness [m].
    Rc : float
        Vortex core radius [m].
    alpha : float
        Gilbert damping constant (dimensionless).
    P_pol : float
        Spin polarisation efficiency (0–1).
    hbar : float
        Reduced Planck constant [J·s].
    e_charge : float
        Electron charge magnitude [C].

    DBAT 2.0 coupling constants
    ----------------------------
    eta_inc : float
        Incoherent magnon damping coefficient.
    g_lin : float
        Linear inertial coupling.
    g_nl : float
        Nonlinear kinematic-shock multiplier.
    mu_sw : float
        TMR spin-wave readout gain.
    xi_xpm : float
        Cross-Phase Modulation (XPM) coefficient.
    kappa_coh : float
        Coherent Phase-Pulling (locking) rate [1/s].
    v_limit : float
        Relativistic kinematic limit for numerical continuity [1/s].

    Field-dependence coefficients
    ------------------------------
    lam_w0_1, lam_w0_2 : linear/quadratic field shift of ω₀.
    lam_N_1, lam_N_2   : linear/quadratic field shift of N.
    lam_d0_1, lam_d0_2 : linear/quadratic field shift of d₀.
    lam_d1_1, lam_d1_2 : linear/quadratic field shift of d₁.
    a_wsw_1, a_wsw_2   : linear/quadratic field shift of ω_sw.
    lam_chi_1          : linear field correction of χ.
    """

    # ------------------------------------------------------------------ #
    # 1. Fundamental constants and geometry                               #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 2. DBAT 2.0 coupling constants                                      #
    # ------------------------------------------------------------------ #
    eta_inc: float = 2.0       # Incoherent magnon damping
    g_lin: float = 1.5         # Linear inertial coupling
    g_nl: float = 1500.0       # Nonlinear kinematic-shock multiplier
    mu_sw: float = 600.0       # TMR spin-wave readout gain
    xi_xpm: float = 0.05       # Cross-Phase Modulation (XPM)
    kappa_coh: float = 1.2e8   # Coherent Phase-Pulling rate [1/s]
    v_limit: float = 5e10      # Relativistic kinematic limit [1/s]

    # ------------------------------------------------------------------ #
    # 3. Field-dependence coefficients                                    #
    # ------------------------------------------------------------------ #
    p_core: float = 1.0
    lam_w0_1: float = 3.0
    lam_w0_2: float = 0.0
    lam_N_1: float = 0.0
    lam_N_2: float = 0.0
    lam_d0_1: float = 0.0
    lam_d0_2: float = 0.0
    lam_d1_1: float = 0.0
    lam_d1_2: float = 0.0
    a_wsw_1: float = 2 * np.pi * 5e9
    a_wsw_2: float = 0.0
    lam_chi_1: float = 2.5

    def __post_init__(self) -> None:
        """Compute derived constants once all primary parameters are set."""
        self._recompute_derived()

    def _recompute_derived(self) -> None:
        """Recalculate all derived quantities (call after manual attribute changes)."""
        self.R = self.d_disk / 2.0
        self.beta_param = self.L_th / self.R
        self.w_M = self.gamma * self.mu0 * self.Ms

        self.w0_base = (5.0 / (9.0 * np.pi)) * self.w_M * self.beta_param
        self.N_base = 0.85

        self.d0_base = self.alpha * (5.0 + 4.0 * np.log(self.R / self.Rc)) / 8.0
        self.d1_base = (11.0 / 6.0) * self.alpha

        self.sigma = (self.hbar * self.P_pol) / (
            2 * self.e_charge * self.L_th * self.Ms
        )
        self.chi_pref = self.gamma * self.sigma / 2.0

        self.w_sw_base = self.w_M * np.sqrt(self.beta_param / 2.0)
        self.G_sw = 0.015 * self.w_sw_base

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_numba_p_const(self) -> np.ndarray:
        """Return the flat C-contiguous parameter array consumed by the JIT engine.

        Returns
        -------
        np.ndarray, shape (9,), dtype float64
            ``[chi_pref, eta_inc, g_lin, g_nl, G_sw, mu_sw, xi_xpm, kappa_coh, v_limit]``
        """
        return np.array(
            [
                self.chi_pref,
                self.eta_inc,
                self.g_lin,
                self.g_nl,
                self.G_sw,
                self.mu_sw,
                self.xi_xpm,
                self.kappa_coh,
                self.v_limit,
            ],
            dtype=np.float64,
        )

    def evaluate_field_arrays(
        self, H_array: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate field-dependent coupling vectors for an array of H values.

        Parameters
        ----------
        H_array : np.ndarray
            External field values in Tesla (µ₀H).

        Returns
        -------
        w0_arr, N_arr, d0_arr, d1_arr, wsw_arr, chi_arr : np.ndarray
            Each array has the same length as *H_array*.
        """
        w0_min = 0.15 * self.w0_base
        wsw_min = 0.20 * self.w_sw_base

        w0_arr = np.maximum(
            self.w0_base
            * (
                1.0
                + self.p_core * self.lam_w0_1 * H_array
                + self.lam_w0_2 * H_array**2
            ),
            w0_min,
        )
        N_arr = self.N_base * (
            1.0 + self.lam_N_1 * H_array + self.lam_N_2 * H_array**2
        )
        d0_arr = self.d0_base * (
            1.0 + self.lam_d0_1 * H_array + self.lam_d0_2 * H_array**2
        )
        d1_arr = self.d1_base * (
            1.0 + self.lam_d1_1 * H_array + self.lam_d1_2 * H_array**2
        )
        wsw_arr = np.maximum(
            self.w_sw_base
            + self.a_wsw_1 * H_array
            + self.a_wsw_2 * H_array**2,
            wsw_min,
        )
        chi_arr = self.chi_pref * (1.0 + self.lam_chi_1 * H_array)

        return w0_arr, N_arr, d0_arr, d1_arr, wsw_arr, chi_arr

    def analytical_carrier_exact(self, J_dc: float, H_val: float) -> float:
        """Compute the analytical steady-state carrier frequency f_G [Hz].

        Uses the quadratic stationarity condition derived from the DBAT 2.0
        amplitude equation.

        Parameters
        ----------
        J_dc : float
            DC current density [A/m²].
        H_val : float
            External field [T] (µ₀H).

        Returns
        -------
        float
            Carrier frequency f_G in Hz.
        """
        w0, N, d0, d1, _, chi = self.evaluate_field_arrays(np.array([H_val]))
        w0, N, d0, d1, chi = w0[0], N[0], d0[0], d1[0], chi[0]

        A = d1 * N
        B = d0 * N + d1
        C = d0 - (chi * J_dc) / w0

        if abs(A) < 1e-25:
            q = max(-C / B, 0.0) if abs(B) > 1e-25 else 0.0
        else:
            delta = B * B - 4 * A * C
            if delta < 0:
                return w0 / (2 * np.pi)
            q = max((-B + np.sqrt(delta)) / (2 * A), 0.0)

        return (w0 * (1.0 + N * q)) / (2 * np.pi)
