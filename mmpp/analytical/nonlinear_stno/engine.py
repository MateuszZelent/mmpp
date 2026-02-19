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
>>> p = device.get_numba_constants()
>>> w0, N, d0, d1, wsw, chi = device.evaluate_field_arrays(H_all)
>>> all_V = run_all_sweeps_parallel(
...     Jdc, Jac, fmod, w0, N, d0, d1, wsw, chi,
...     t_max=400e-9, dt_out=0.5e-12, substeps=10, p=p
... )
"""

import math
import numpy as np
import warnings

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
        ImportWarning
    )
    
    # Dummy decorators by zachować ciągłość interfejsu
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator
    prange = range

@njit(fastmath=True)
def _rhs_4D_dynamic_micromag(t, u, Phi, cx, cy, J_dc, J_ac, f_mod, w0_curr, N_curr, d0_curr, d1_curr, wsw_curr, chi_curr, p):
    u_safe = max(u, 1e-6)
    
    # Twarda Bariera Magnetostatyczna
    wall = 0.0
    if u_safe > 0.42:
        wall = 3500.0 * ((u_safe - 0.42) / 0.1)**3

    J_t = J_dc + J_ac * math.cos(2.0 * math.pi * f_mod * t)
    chi_val = chi_curr * J_t
    n_mag = cx*cx + cy*cy

    # DBAT: Incoherent Back-Action
    omega_u = w0_curr * (1.0 + N_curr * u_safe*u_safe)
    Gamma_plus = (d0_curr + d1_curr * u_safe*u_safe + wall) * omega_u
    Gamma_eff = Gamma_plus + p[1] * n_mag  

    # DBAT: Coherent Phase Pulling
    phase_pulling = p[7] * (cx * math.cos(Phi) + cy * math.sin(Phi))
    u_dot_raw = u_safe * (chi_val - Gamma_eff) - phase_pulling

    # DBAT: Cross-Phase Modulation
    Phi_dot = omega_u * max(1.0 - p[6] * n_mag, 0.1)

    # Relatywistyczny limit kinetyczny dla gładkości RK4
    v_lim = p[8]
    u_dot = v_lim * math.tanh(u_dot_raw / v_lim)
    
    # Kinematyczny Szok Macha
    M_v = u_dot / w0_curr
    v_eff = u_safe * u_dot * (p[2] + p[3] * M_v * M_v)
    
    pump_x =  v_eff * math.cos(Phi)
    pump_y = -v_eff * math.sin(Phi)

    # Rezonator Fali Spinowej
    Gsw = p[4]
    dcx = -Gsw * cx + wsw_curr * cy + pump_x
    dcy = -wsw_curr * cx - Gsw * cy + pump_y

    return u_dot, Phi_dot, dcx, dcy

@njit(parallel=HAS_NUMBA, fastmath=True)
def run_all_sweeps_parallel(Jdc_arr, Jac_arr, fmod_arr, w0_arr, N_arr, d0_arr, d1_arr, wsw_arr, chi_arr, t_max, dt_out, substeps, p):
    """Zrównoleglony rdzeń całkujący RK4 z wewnętrznym Sub-Steppingiem."""
    n_sims = len(Jdc_arr)
    steps_out = int(t_max / dt_out)
    dt_sim = dt_out / float(substeps) 
    
    all_V = np.zeros((n_sims, steps_out), dtype=np.float64)

    for i in prange(n_sims):
        u, Phi, cx, cy = 0.2, 0.0, 0.0, 0.0
        t_curr = 0.0

        w0_c = w0_arr[i]; N_c = N_arr[i]; d0_c = d0_arr[i]
        d1_c = d1_arr[i]; wsw_c = wsw_arr[i]; chi_c = chi_arr[i]
        J_dc = Jdc_arr[i]; J_ac = Jac_arr[i]; f_mod = fmod_arr[i]

        for step in range(steps_out):
            J_t_out = J_dc + J_ac * math.cos(2.0 * math.pi * f_mod * t_curr)
            all_V[i, step] = J_t_out * (u * math.cos(Phi) + p[5] * cx)
            
            for s in range(substeps):
                ku1, kP1, kcx1, kcy1 = _rhs_4D_dynamic_micromag(
                    t_curr, u, Phi, cx, cy, J_dc, J_ac, f_mod, w0_c, N_c, d0_c, d1_c, wsw_c, chi_c, p)
                ku2, kP2, kcx2, kcy2 = _rhs_4D_dynamic_micromag(
                    t_curr+dt_sim/2, u+ku1*dt_sim/2, Phi+kP1*dt_sim/2, cx+kcx1*dt_sim/2, cy+kcy1*dt_sim/2,
                    J_dc, J_ac, f_mod, w0_c, N_c, d0_c, d1_c, wsw_c, chi_c, p)
                ku3, kP3, kcx3, kcy3 = _rhs_4D_dynamic_micromag(
                    t_curr+dt_sim/2, u+ku2*dt_sim/2, Phi+kP2*dt_sim/2, cx+kcx2*dt_sim/2, cy+kcy2*dt_sim/2,
                    J_dc, J_ac, f_mod, w0_c, N_c, d0_c, d1_c, wsw_c, chi_c, p)
                ku4, kP4, kcx4, kcy4 = _rhs_4D_dynamic_micromag(
                    t_curr+dt_sim, u+ku3*dt_sim, Phi+kP3*dt_sim, cx+kcx3*dt_sim, cy+kcy3*dt_sim,
                    J_dc, J_ac, f_mod, w0_c, N_c, d0_c, d1_c, wsw_c, chi_c, p)

                u   += (dt_sim/6.0) * (ku1 + 2*ku2 + 2*ku3 + ku4)
                Phi += (dt_sim/6.0) * (kP1 + 2*kP2 + 2*kP3 + kP4)
                cx  += (dt_sim/6.0) * (kcx1 + 2*kcx2 + 2*kcx3 + kcx4)
                cy  += (dt_sim/6.0) * (kcy1 + 2*kcy2 + 2*kcy3 + kcy4)
                
                t_curr += dt_sim

                # Rygorystyczne obcinanie krawędzi poza obszarem ewaluacji pochodnych
                if u < 1e-6: u = 1e-6
                elif u > 0.95: u = 0.95

    return all_V
