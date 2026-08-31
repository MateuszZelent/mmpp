"""
mmpp.analytical.nonlinear_stno.physics
=======================================
Parameters for an experimental phenomenological STNO vortex extension.

This module provides the STNOParameters dataclass that encapsulates all
micromagnetic properties, coupling constants and field-dependent arrays
required by the numerical engine.  The additional spin-wave/back-action terms
are not a first-principles consequence of the Thiele equation and have no
quantitative status until calibrated and validated against independent data.

Example
-------
>>> from mmpp.analytical.nonlinear_stno import STNOParameters
>>> device = STNOParameters()
>>> device.g_nl = 2000.0   # override nonlinear shock multiplier
>>> p = device.get_numba_constants()
>>> w0, N, d0, d1, wsw, chi = device.evaluate_field_arrays(H_array)
"""

import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass
class STNOParameters:
    """Experimental phenomenological 4D STNO parameters.

    The gyrotropic base terms follow the reduced CPP model, whereas the
    spin-wave envelope, back-action, wall, and rate-limiter coefficients are
    empirical.  They must be fitted and independently validated before any
    quantitative interpretation.
    """

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

    # 2. Empirical back-action extension (not a validated Thiele theory)
    eta_inc: float = 2.0  # [1/s] Tłumienie od magnonów (Incoherent Fading Memory)
    g_lin: float = 1.5  # [-] Liniowe sprzężenie inercyjne
    g_nl: float = 1500.0  # [-] Nieliniowy mnożnik szoku kinematycznego Macha
    mu_sw: float = 600.0  # [-] Siła odczytu TMR fali spinowej
    xi_xpm: float = 0.05  # [-] Cross-Phase Modulation (XPM)
    kappa_coh: float = 1.2e8  # [rad/s] Coherent Phase-Pulling (Locking)
    v_limit: float = 5e10  # [1/s] smooth numerical radial-rate limiter

    # 3. Empirical edge regularisation
    edge_wall_start: float = 0.42
    edge_wall_strength: float = 3500.0
    edge_wall_width: float = 0.1
    orbit_floor: float = 1e-6
    orbit_ceiling: float = 0.95

    # 4. Empirical field response
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

    model_status: str = field(init=False, default="experimental_uncalibrated")

    def __post_init__(self):
        self._validate()
        self._recompute_derived()
        warnings.warn(
            "STNOParameters configures the experimental, uncalibrated 4D "
            "nonlinear_stno extension. Use CPPThieleModel for the validated "
            "rigid-vortex baseline.",
            UserWarning,
            stacklevel=2,
        )

    def _validate(self) -> None:
        numeric = {
            name: value for name, value in vars(self).items() if name != "model_status"
        }
        if not all(np.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("all STNO parameters must be finite")
        for name in ("gamma", "mu0", "Ms", "d_disk", "L_th", "Rc", "hbar", "e_charge"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if hasattr(self, "G_sw") and self.G_sw < 0.0:
            raise ValueError("G_sw must be non-negative")
        if self.eta_inc < 0.0 or self.g_lin < 0.0 or self.g_nl < 0.0:
            raise ValueError("back-action damping/coupling terms must be non-negative")
        if self.kappa_coh < 0.0 or self.v_limit <= 0.0:
            raise ValueError("kappa_coh must be non-negative and v_limit positive")
        if not 0.0 < self.edge_wall_start < 1.0:
            raise ValueError("edge_wall_start must lie in (0, 1)")
        if self.edge_wall_strength < 0.0 or self.edge_wall_width <= 0.0:
            raise ValueError("edge wall strength/width are invalid")
        if not 0.0 < self.orbit_floor < self.orbit_ceiling <= 1.0:
            raise ValueError("orbit bounds must satisfy 0 < floor < ceiling <= 1")

    def _recompute_derived(self):
        """Przelicza stałe pochodne po inicjalizacji lub po ręcznej zmianie atrybutów."""
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

    def get_numba_constants(self):
        """Return the validated 14-element empirical kernel parameter vector."""
        self._validate()
        self._recompute_derived()
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
                self.edge_wall_start,
                self.edge_wall_strength,
                self.edge_wall_width,
                self.orbit_floor,
                self.orbit_ceiling,
            ],
            dtype=np.float64,
        )

    def evaluate_field_arrays(self, H_array):
        """Evaluate empirical coefficient arrays for flux density ``B`` [T]."""
        self._validate()
        self._recompute_derived()
        H_array = np.asarray(H_array, dtype=float)
        if H_array.ndim != 1 or not np.all(np.isfinite(H_array)):
            raise ValueError("field array must be one-dimensional and finite [T]")
        w0_min = 0.15 * self.w0_base
        wsw_min = 0.20 * self.w_sw_base

        w0_raw = self.w0_base * (
            1.0 + self.p_core * self.lam_w0_1 * H_array + self.lam_w0_2 * H_array**2
        )
        w0_arr = np.maximum(w0_raw, w0_min)
        N_arr = self.N_base * (1.0 + self.lam_N_1 * H_array + self.lam_N_2 * H_array**2)
        d0_arr = self.d0_base * (
            1.0 + self.lam_d0_1 * H_array + self.lam_d0_2 * H_array**2
        )
        d1_arr = self.d1_base * (
            1.0 + self.lam_d1_1 * H_array + self.lam_d1_2 * H_array**2
        )
        wsw_raw = self.w_sw_base + self.a_wsw_1 * H_array + self.a_wsw_2 * H_array**2
        wsw_arr = np.maximum(wsw_raw, wsw_min)
        chi_arr = self.chi_pref * (1.0 + self.lam_chi_1 * H_array)

        if np.any(w0_raw < w0_min) or np.any(wsw_raw < wsw_min):
            warnings.warn(
                "Experimental field response reached a configured frequency "
                "floor; those points are outside the calibrated range.",
                UserWarning,
                stacklevel=2,
            )
        if np.any(d0_arr < 0.0) or np.any(d1_arr < 0.0):
            raise ValueError("field response produced negative damping")

        return w0_arr, N_arr, d0_arr, d1_arr, wsw_arr, chi_arr

    def analytical_carrier_exact(self, J_dc, H_val):
        """Rozwiązanie stacjonarne modelu. Zwraca częstotliwość f_G [Hz]."""
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
