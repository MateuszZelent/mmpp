"""
mmpp.analytical.nonlinear_stno.engine
======================================
Numba-JIT numerical engine for 4D STNO vortex ODE integration.

Design principles
-----------------
* No Python objects – only flat NumPy C-arrays enter the JIT boundary.
* ``_rhs_4D_dynamic_micromag`` is a pure scalar kernel (**fastmath** enabled).
* ``run_all_sweeps_parallel`` runs all parameter sweeps on all CPU cores in
  parallel via ``prange``.

State vector ``(u, Φ, cₓ, cᵧ)``
---------------------------------
u   – normalised vortex core displacement amplitude
Φ   – gyration phase [rad]
cₓ  – spin-wave envelope (real component)
cᵧ  – spin-wave envelope (imaginary component)

Parameter vector ``p[9]``
--------------------------
p[0] chi_pref   – current-to-torque conversion prefactor
p[1] eta_inc    – incoherent magnon damping coefficient
p[2] g_lin      – linear inertial coupling g_lin
p[3] g_nl       – nonlinear kinematic-shock multiplier
p[4] G_sw       – spin-wave decay rate
p[5] mu_sw      – TMR spin-wave readout gain
p[6] xi_xpm     – Cross-Phase Modulation coefficient
p[7] kappa_coh  – Coherent Phase-Pulling rate [1/s]
p[8] v_limit    – Relativistic kinematic velocity limit [1/s]

Example
-------
>>> from mmpp.analytical.nonlinear_stno import STNOParameters, run_all_sweeps_parallel
>>> device = STNOParameters()
>>> p = device.get_numba_p_const()
>>> w0, N, d0, d1, wsw, chi = device.evaluate_field_arrays(H_all)
>>> all_V = run_all_sweeps_parallel(
...     Jdc, Jac, fmod, w0, N, d0, d1, wsw, chi,
...     t_max=400e-9, dt_out=0.5e-12, substeps=10, p=p
... )
"""

import math
import numpy as np

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:

    @njit(fastmath=True)
    def _rhs_4D_dynamic_micromag(
        t,
        u,
        Phi,
        cx,
        cy,
        J_dc,
        J_ac,
        f_mod,
        w0_curr,
        N_curr,
        d0_curr,
        d1_curr,
        wsw_curr,
        chi_curr,
        p,
    ):
        """Scalar RHS of the 4D STNO vortex ODE (Numba JIT kernel).

        Parameters
        ----------
        t : float
            Current simulation time [s].
        u : float
            Vortex amplitude (normalised, 0–1).
        Phi : float
            Gyration phase [rad].
        cx, cy : float
            Spin-wave envelope components.
        J_dc, J_ac, f_mod : float
            DC/AC current densities [A/m²] and modulation frequency [Hz].
        w0_curr, N_curr, d0_curr, d1_curr, wsw_curr, chi_curr : float
            Field-dependent coupling scalars for this simulation point.
        p : 1-D float64 array, length 9
            Global DBAT 2.0 constant vector (see module docstring).

        Returns
        -------
        u_dot, Phi_dot, cx_dot, cy_dot : float
        """
        u_safe = max(u, 1e-6)

        # Hard-wall confinement near u ≈ 0.42 (disk boundary)
        wall = 0.0
        if u_safe > 0.42:
            wall = 3500.0 * ((u_safe - 0.42) / 0.1) ** 3

        J_t = J_dc + J_ac * math.cos(2.0 * math.pi * f_mod * t)
        chi_val = chi_curr * J_t
        n_mag = cx * cx + cy * cy

        omega_u = w0_curr * (1.0 + N_curr * u_safe * u_safe)
        Gamma_plus = (d0_curr + d1_curr * u_safe * u_safe + wall) * omega_u
        Gamma_eff = Gamma_plus + p[1] * n_mag  # + incoherent magnon damping

        phase_pulling = p[7] * (cx * math.cos(Phi) + cy * math.sin(Phi))
        u_dot_raw = u_safe * (chi_val - Gamma_eff) - phase_pulling
        Phi_dot = omega_u * max(1.0 - p[6] * n_mag, 0.1)

        # Relativistic velocity limiter (smooth tanh saturation)
        v_lim = p[8]
        u_dot = v_lim * math.tanh(u_dot_raw / v_lim)

        # Back-action pump driving the spin-wave mode
        M_v = u_dot / w0_curr
        v_eff = u_safe * u_dot * (p[2] + p[3] * M_v * M_v)

        pump_x = v_eff * math.cos(Phi)
        pump_y = -v_eff * math.sin(Phi)

        Gsw = p[4]
        dcx = -Gsw * cx + wsw_curr * cy + pump_x
        dcy = -wsw_curr * cx - Gsw * cy + pump_y

        return u_dot, Phi_dot, dcx, dcy

    @njit(parallel=True, fastmath=True)
    def run_all_sweeps_parallel(
        Jdc_arr,
        Jac_arr,
        fmod_arr,
        w0_arr,
        N_arr,
        d0_arr,
        d1_arr,
        wsw_arr,
        chi_arr,
        t_max,
        dt_out,
        substeps,
        p,
    ):
        """Integrate all STNO trajectories in parallel using 4th-order Runge-Kutta.

        Each trajectory is integrated independently, enabling full CPU
        parallelism via Numba ``prange``.  The inner loop uses a fixed-step
        RK4 scheme with *substeps* micro-steps per output sample.

        Parameters
        ----------
        Jdc_arr, Jac_arr, fmod_arr : 1-D float64 arrays, length n_sims
            Per-simulation DC/AC current densities and modulation frequencies.
        w0_arr, N_arr, d0_arr, d1_arr, wsw_arr, chi_arr : 1-D float64 arrays
            Field-dependent coupling vectors (one entry per simulation).
        t_max : float
            Total integration time [s].
        dt_out : float
            Output sampling interval [s].
        substeps : int
            Number of RK4 micro-steps per output sample.
        p : 1-D float64 array, length 9
            DBAT 2.0 constant vector (see :func:`STNOParameters.get_numba_p_const`).

        Returns
        -------
        all_V : np.ndarray, shape (n_sims, steps_out)
            Simulated voltage-proxy time series for each sweep point.
        """
        n_sims = len(Jdc_arr)
        steps_out = int(t_max / dt_out)
        dt_sim = dt_out / float(substeps)
        all_V = np.zeros((n_sims, steps_out), dtype=np.float64)

        for i in prange(n_sims):
            u, Phi, cx, cy = 0.2, 0.0, 0.0, 0.0
            t_curr = 0.0

            w0_c = w0_arr[i]
            N_c = N_arr[i]
            d0_c = d0_arr[i]
            d1_c = d1_arr[i]
            wsw_c = wsw_arr[i]
            chi_c = chi_arr[i]
            J_dc = Jdc_arr[i]
            J_ac = Jac_arr[i]
            f_mod = fmod_arr[i]

            for step in range(steps_out):
                # Record output sample (voltage proxy)
                J_t_out = J_dc + J_ac * math.cos(2.0 * math.pi * f_mod * t_curr)
                all_V[i, step] = J_t_out * (u * math.cos(Phi) + p[5] * cx)

                # RK4 micro-steps
                for s in range(substeps):
                    ku1, kP1, kcx1, kcy1 = _rhs_4D_dynamic_micromag(
                        t_curr,
                        u,
                        Phi,
                        cx,
                        cy,
                        J_dc,
                        J_ac,
                        f_mod,
                        w0_c,
                        N_c,
                        d0_c,
                        d1_c,
                        wsw_c,
                        chi_c,
                        p,
                    )
                    ku2, kP2, kcx2, kcy2 = _rhs_4D_dynamic_micromag(
                        t_curr + dt_sim / 2,
                        u + ku1 * dt_sim / 2,
                        Phi + kP1 * dt_sim / 2,
                        cx + kcx1 * dt_sim / 2,
                        cy + kcy1 * dt_sim / 2,
                        J_dc,
                        J_ac,
                        f_mod,
                        w0_c,
                        N_c,
                        d0_c,
                        d1_c,
                        wsw_c,
                        chi_c,
                        p,
                    )
                    ku3, kP3, kcx3, kcy3 = _rhs_4D_dynamic_micromag(
                        t_curr + dt_sim / 2,
                        u + ku2 * dt_sim / 2,
                        Phi + kP2 * dt_sim / 2,
                        cx + kcx2 * dt_sim / 2,
                        cy + kcy2 * dt_sim / 2,
                        J_dc,
                        J_ac,
                        f_mod,
                        w0_c,
                        N_c,
                        d0_c,
                        d1_c,
                        wsw_c,
                        chi_c,
                        p,
                    )
                    ku4, kP4, kcx4, kcy4 = _rhs_4D_dynamic_micromag(
                        t_curr + dt_sim,
                        u + ku3 * dt_sim,
                        Phi + kP3 * dt_sim,
                        cx + kcx3 * dt_sim,
                        cy + kcy3 * dt_sim,
                        J_dc,
                        J_ac,
                        f_mod,
                        w0_c,
                        N_c,
                        d0_c,
                        d1_c,
                        wsw_c,
                        chi_c,
                        p,
                    )

                    u += (dt_sim / 6.0) * (ku1 + 2 * ku2 + 2 * ku3 + ku4)
                    Phi += (dt_sim / 6.0) * (kP1 + 2 * kP2 + 2 * kP3 + kP4)
                    cx += (dt_sim / 6.0) * (kcx1 + 2 * kcx2 + 2 * kcx3 + kcx4)
                    cy += (dt_sim / 6.0) * (kcy1 + 2 * kcy2 + 2 * kcy3 + kcy4)

                    t_curr += dt_sim

                    # Amplitude guard-rails
                    if u < 1e-6:
                        u = 1e-6
                    elif u > 0.95:
                        u = 0.95

        return all_V

else:  # pragma: no cover – fallback when Numba is not installed
    import warnings

    warnings.warn(
        "numba is not installed; run_all_sweeps_parallel will be extremely slow "
        "(pure-Python fallback). Install numba for production use.",
        ImportWarning,
        stacklevel=2,
    )

    def _rhs_4D_dynamic_micromag(  # type: ignore[misc]
        t, u, Phi, cx, cy, J_dc, J_ac, f_mod,
        w0_curr, N_curr, d0_curr, d1_curr, wsw_curr, chi_curr, p,
    ):
        u_safe = max(u, 1e-6)
        wall = 0.0
        if u_safe > 0.42:
            wall = 3500.0 * ((u_safe - 0.42) / 0.1) ** 3
        J_t = J_dc + J_ac * math.cos(2.0 * math.pi * f_mod * t)
        chi_val = chi_curr * J_t
        n_mag = cx * cx + cy * cy
        omega_u = w0_curr * (1.0 + N_curr * u_safe * u_safe)
        Gamma_eff = (d0_curr + d1_curr * u_safe * u_safe + wall) * omega_u + p[1] * n_mag
        phase_pulling = p[7] * (cx * math.cos(Phi) + cy * math.sin(Phi))
        u_dot_raw = u_safe * (chi_val - Gamma_eff) - phase_pulling
        Phi_dot = omega_u * max(1.0 - p[6] * n_mag, 0.1)
        v_lim = p[8]
        u_dot = v_lim * math.tanh(u_dot_raw / v_lim)
        M_v = u_dot / w0_curr
        v_eff = u_safe * u_dot * (p[2] + p[3] * M_v * M_v)
        dcx = -p[4] * cx + wsw_curr * cy + v_eff * math.cos(Phi)
        dcy = -wsw_curr * cx - p[4] * cy - v_eff * math.sin(Phi)
        return u_dot, Phi_dot, dcx, dcy

    def run_all_sweeps_parallel(  # type: ignore[misc]
        Jdc_arr, Jac_arr, fmod_arr,
        w0_arr, N_arr, d0_arr, d1_arr, wsw_arr, chi_arr,
        t_max, dt_out, substeps, p,
    ):
        n_sims = len(Jdc_arr)
        steps_out = int(t_max / dt_out)
        dt_sim = dt_out / float(substeps)
        all_V = np.zeros((n_sims, steps_out), dtype=np.float64)
        for i in range(n_sims):
            u, Phi, cx, cy = 0.2, 0.0, 0.0, 0.0
            t_curr = 0.0
            for step in range(steps_out):
                J_t_out = Jdc_arr[i] + Jac_arr[i] * math.cos(
                    2.0 * math.pi * fmod_arr[i] * t_curr
                )
                all_V[i, step] = J_t_out * (u * math.cos(Phi) + p[5] * cx)
                for _ in range(substeps):
                    args = (
                        t_curr, u, Phi, cx, cy,
                        Jdc_arr[i], Jac_arr[i], fmod_arr[i],
                        w0_arr[i], N_arr[i], d0_arr[i], d1_arr[i],
                        wsw_arr[i], chi_arr[i], p,
                    )
                    ku1, kP1, kcx1, kcy1 = _rhs_4D_dynamic_micromag(*args)
                    ku2, kP2, kcx2, kcy2 = _rhs_4D_dynamic_micromag(
                        t_curr + dt_sim / 2,
                        u + ku1 * dt_sim / 2, Phi + kP1 * dt_sim / 2,
                        cx + kcx1 * dt_sim / 2, cy + kcy1 * dt_sim / 2,
                        *args[5:],
                    )
                    ku3, kP3, kcx3, kcy3 = _rhs_4D_dynamic_micromag(
                        t_curr + dt_sim / 2,
                        u + ku2 * dt_sim / 2, Phi + kP2 * dt_sim / 2,
                        cx + kcx2 * dt_sim / 2, cy + kcy2 * dt_sim / 2,
                        *args[5:],
                    )
                    ku4, kP4, kcx4, kcy4 = _rhs_4D_dynamic_micromag(
                        t_curr + dt_sim,
                        u + ku3 * dt_sim, Phi + kP3 * dt_sim,
                        cx + kcx3 * dt_sim, cy + kcy3 * dt_sim,
                        *args[5:],
                    )
                    u += (dt_sim / 6.0) * (ku1 + 2 * ku2 + 2 * ku3 + ku4)
                    Phi += (dt_sim / 6.0) * (kP1 + 2 * kP2 + 2 * kP3 + kP4)
                    cx += (dt_sim / 6.0) * (kcx1 + 2 * kcx2 + 2 * kcx3 + kcx4)
                    cy += (dt_sim / 6.0) * (kcy1 + 2 * kcy2 + 2 * kcy3 + kcy4)
                    t_curr += dt_sim
                    u = max(1e-6, min(0.95, u))
        return all_V
