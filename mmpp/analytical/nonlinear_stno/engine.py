"""
mmpp.analytical.nonlinear_stno.engine
======================================
Numba-JIT engine for an experimental phenomenological 4D STNO ODE.

This is not a micromagnetic solver and the extra spin-wave/back-action terms
are not part of the validated rigid-vortex Thiele baseline.

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

Parameter vector ``p[14]``
---------------------------
p[0] chi_pref   – current-to-torque conversion prefactor
p[1] eta_inc    – incoherent magnon damping coefficient
p[2] g_lin      – linear inertial coupling g_lin
p[3] g_nl       – nonlinear kinematic-shock multiplier
p[4] G_sw       – spin-wave decay rate
p[5] mu_sw      – TMR spin-wave readout gain
p[6] xi_xpm     – Cross-Phase Modulation coefficient
p[7] kappa_coh  – Coherent Phase-Pulling rate [1/s]
p[8] v_limit    – smooth radial-rate limit [1/s]
p[9:14]         – empirical wall start/strength/width and orbit bounds

Example
-------
>>> from mmpp.analytical.nonlinear_stno import STNOParameters, run_all_sweeps_parallel
>>> device = STNOParameters()
>>> p = device.get_numba_constants()
>>> w0, N, d0, d1, wsw, chi = device.evaluate_field_arrays(H_all)
>>> all_V = run_all_sweeps_parallel(
...     Jdc, Jac, fmod, w0, N, d0, d1, wsw, chi,
...     t_max=400e-9, dt_out=0.5e-12, substeps=10, p=p
... )
"""

import math
import warnings

import numpy as np

# =====================================================================
# NUMBA-FREE FALLBACK (Bezpieczeństwo Środowiskowe CI/CD)
# =====================================================================
try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "Pakiet 'numba' nie został znaleziony. Silnik użyje czystego Pythona. "
        "Całkowanie układów Stiff ODE będzie działać drastycznie wolniej.",
        ImportWarning,
        stacklevel=2,
    )

    # Dummy decorators by zachować ciągłość interfejsu
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range


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
    u_safe = max(u, p[12])

    # Empirical edge penalty; it is a calibration term, not a derived force.
    wall = 0.0
    if u_safe > p[9]:
        wall = p[10] * ((u_safe - p[9]) / p[11]) ** 3

    J_t = J_dc + J_ac * math.cos(2.0 * math.pi * f_mod * t)
    chi_val = chi_curr * J_t
    n_mag = cx * cx + cy * cy

    # Empirical incoherent back-action
    omega_u = w0_curr * (1.0 + N_curr * u_safe * u_safe)
    Gamma_plus = (d0_curr + d1_curr * u_safe * u_safe + wall) * omega_u
    Gamma_eff = Gamma_plus + p[1] * n_mag

    # Empirical coherent phase pulling
    phase_pulling = p[7] * (cx * math.cos(Phi) + cy * math.sin(Phi))
    u_dot_raw = u_safe * (chi_val - Gamma_eff) - phase_pulling

    # Empirical cross-phase modulation
    Phi_dot = omega_u * max(1.0 - p[6] * n_mag, 0.1)

    # Smooth numerical rate limiter (no relativistic interpretation).
    v_lim = p[8]
    u_dot = v_lim * math.tanh(u_dot_raw / v_lim)

    # Empirical nonlinear rate coupling (no Mach-number interpretation).
    M_v = u_dot / w0_curr
    v_eff = u_safe * u_dot * (p[2] + p[3] * M_v * M_v)

    pump_x = v_eff * math.cos(Phi)
    pump_y = -v_eff * math.sin(Phi)

    # Phenomenological spin-wave envelope
    Gsw = p[4]
    dcx = -Gsw * cx + wsw_curr * cy + pump_x
    dcy = -wsw_curr * cx - Gsw * cy + pump_y

    return u_dot, Phi_dot, dcx, dcy


@njit(parallel=HAS_NUMBA, fastmath=True)
def _run_all_sweeps_parallel_kernel(
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
    """Validated wrapper's parallel RK4 implementation."""
    n_sims = len(Jdc_arr)
    steps_out = int(math.floor(t_max / dt_out)) + 1
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
            J_t_out = J_dc + J_ac * math.cos(2.0 * math.pi * f_mod * t_curr)
            all_V[i, step] = J_t_out * (u * math.cos(Phi) + p[5] * cx)

            if step == steps_out - 1:
                break

            for _s in range(substeps):
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

                # Explicit empirical validity interval for the reduced orbit.
                if u < p[12]:
                    u = p[12]
                elif u > p[13]:
                    u = p[13]

    return all_V


def _validated_sweep_array(name, values, n_sims=None):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if n_sims is not None and array.size != n_sims:
        raise ValueError(f"{name} must have {n_sims} entries")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


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
    """Integrate the experimental 4D model after strict input validation.

    The returned array is a dimensionless/electrical ``signal proxy``.  It is
    not a calibrated voltage or power spectral density.
    """
    jdc = _validated_sweep_array("Jdc_arr", Jdc_arr)
    n_sims = int(jdc.size)
    if n_sims == 0:
        raise ValueError("at least one sweep is required")
    jac = _validated_sweep_array("Jac_arr", Jac_arr, n_sims)
    fmod = _validated_sweep_array("fmod_arr", fmod_arr, n_sims)
    w0 = _validated_sweep_array("w0_arr", w0_arr, n_sims)
    nonlinear = _validated_sweep_array("N_arr", N_arr, n_sims)
    d0 = _validated_sweep_array("d0_arr", d0_arr, n_sims)
    d1 = _validated_sweep_array("d1_arr", d1_arr, n_sims)
    wsw = _validated_sweep_array("wsw_arr", wsw_arr, n_sims)
    chi = _validated_sweep_array("chi_arr", chi_arr, n_sims)

    duration = float(t_max)
    output_step = float(dt_out)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("t_max must be finite and positive")
    if not math.isfinite(output_step) or output_step <= 0.0:
        raise ValueError("dt_out must be finite and positive")
    if isinstance(substeps, bool) or int(substeps) != substeps or int(substeps) < 1:
        raise ValueError("substeps must be a positive integer")
    if np.any(w0 <= 0.0) or np.any(wsw <= 0.0):
        raise ValueError("w0_arr and wsw_arr must be positive angular frequencies")
    if np.any(d0 < 0.0) or np.any(d1 < 0.0):
        raise ValueError("d0_arr and d1_arr must be non-negative")

    parameters = np.asarray(p, dtype=np.float64).reshape(-1)
    if parameters.size == 9:
        warnings.warn(
            "Legacy 9-element nonlinear_stno parameter vector: applying the "
            "historical empirical edge defaults.",
            DeprecationWarning,
            stacklevel=2,
        )
        parameters = np.concatenate(
            [parameters, np.array([0.42, 3500.0, 0.1, 1e-6, 0.95])]
        )
    if parameters.size != 14 or not np.all(np.isfinite(parameters)):
        raise ValueError("p must contain 14 finite model coefficients")
    if parameters[4] < 0.0 or parameters[8] <= 0.0:
        raise ValueError("spin-wave damping must be non-negative and v_limit positive")
    if not 0.0 < parameters[9] < 1.0 or parameters[10] < 0.0:
        raise ValueError("empirical edge-wall coefficients are invalid")
    if parameters[11] <= 0.0 or not (0.0 < parameters[12] < parameters[13] <= 1.0):
        raise ValueError("empirical wall width/orbit bounds are invalid")

    warnings.warn(
        "run_all_sweeps_parallel uses the experimental, uncalibrated 4D "
        "nonlinear_stno extension; do not interpret it quantitatively without "
        "independent calibration.",
        UserWarning,
        stacklevel=2,
    )
    return _run_all_sweeps_parallel_kernel(
        jdc,
        jac,
        fmod,
        w0,
        nonlinear,
        d0,
        d1,
        wsw,
        chi,
        duration,
        output_step,
        int(substeps),
        np.ascontiguousarray(parameters),
    )
