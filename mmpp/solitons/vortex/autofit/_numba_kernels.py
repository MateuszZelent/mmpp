# ruff: noqa: N802, N803, N806
"""
Numba-accelerated RK4 integrators for Thiele vortex ODE.

Provides ~50-100x speedup over pure-Python ``solve_ivp`` by:
- JIT-compiling the RHS to native code (no Python overhead per call)
- Using fixed-step RK4 (no adaptive overhead)
- Working with scalars only (no NumPy array allocation per step)
- Pre-computing all constant parameters outside the loop

Falls back to pure Python if Numba is not installed.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

# ── Numba import with graceful fallback ──────────────────────────

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "numba not found — autofit will use pure-Python ODE integrator. "
        "Install numba for ~50x speedup: pip install numba",
        ImportWarning,
        stacklevel=2,
    )

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# =====================================================================
# CPP Thiele RHS kernel (Guslienko 2014)
# =====================================================================


@njit(fastmath=True)
def _cpp_rhs(sx, sy, chi_val, omega0_eff, N, d0, d1, polarity, seq_x, seq_y):
    """CPP Thiele RHS — pure scalar, no allocations.

    Parameters (all pre-computed scalars):
        sx, sy      : normalised core position
        chi_val     : STT pumping rate chi(J) = chi_scale * gamma * sigma * J / 2
        omega0_eff  : effective linear frequency (includes Oersted + field shifts)
        N           : nonlinear frequency coefficient
        d0, d1      : damping coefficients
        polarity    : +1 or -1
        seq_x, seq_y: equilibrium shift from in-plane field
    """
    # relative position
    srel_x = sx - seq_x
    srel_y = sy - seq_y
    u = math.sqrt(srel_x * srel_x + srel_y * srel_y)
    u = max(u, 1e-15)

    omega_val = omega0_eff * (1.0 + N * u * u)
    d_val = d0 + d1 * u * u
    radial = chi_val - d_val * omega_val

    dsx = radial * srel_x - polarity * omega_val * srel_y
    dsy = radial * srel_y + polarity * omega_val * srel_x
    return dsx, dsy


@njit(fastmath=True)
def integrate_cpp_rk4(
    t0,
    t1,
    dt_out,
    sx0,
    sy0,
    chi_val,
    omega0_eff,
    N,
    d0,
    d1,
    polarity,
    seq_x,
    seq_y,
    substeps,
):
    """Fixed-step RK4 integration of CPP Thiele equation.

    Parameters
    ----------
    t0, t1 : float
        Integration interval [s].
    dt_out : float
        Output sampling period [s].
    sx0, sy0 : float
        Initial normalised core position.
    chi_val : float
        Pre-computed chi(J).
    omega0_eff : float
        Pre-computed effective omega0.
    N, d0, d1, polarity, seq_x, seq_y : float
        Model constants.
    substeps : int
        Number of RK4 sub-steps per output step.

    Returns
    -------
    t_out : ndarray (n_steps,)
    sx_out, sy_out : ndarray (n_steps,)
    """
    n_steps = int((t1 - t0) / dt_out) + 1
    dt_sub = dt_out / float(substeps)

    t_out = np.empty(n_steps, dtype=np.float64)
    sx_out = np.empty(n_steps, dtype=np.float64)
    sy_out = np.empty(n_steps, dtype=np.float64)

    sx = sx0
    sy = sy0
    t = t0

    for i in range(n_steps):
        t_out[i] = t
        sx_out[i] = sx
        sy_out[i] = sy

        for _s in range(substeps):
            k1x, k1y = _cpp_rhs(
                sx, sy, chi_val, omega0_eff, N, d0, d1, polarity, seq_x, seq_y
            )

            k2x, k2y = _cpp_rhs(
                sx + 0.5 * dt_sub * k1x,
                sy + 0.5 * dt_sub * k1y,
                chi_val,
                omega0_eff,
                N,
                d0,
                d1,
                polarity,
                seq_x,
                seq_y,
            )

            k3x, k3y = _cpp_rhs(
                sx + 0.5 * dt_sub * k2x,
                sy + 0.5 * dt_sub * k2y,
                chi_val,
                omega0_eff,
                N,
                d0,
                d1,
                polarity,
                seq_x,
                seq_y,
            )

            k4x, k4y = _cpp_rhs(
                sx + dt_sub * k3x,
                sy + dt_sub * k3y,
                chi_val,
                omega0_eff,
                N,
                d0,
                d1,
                polarity,
                seq_x,
                seq_y,
            )

            sx += (dt_sub / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            sy += (dt_sub / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

            t += dt_sub

    return t_out, sx_out, sy_out


# =====================================================================
# CIP Thiele RHS kernel (Moon et al.)
# =====================================================================


@njit(fastmath=True)
def _cip_rhs(X, Y, w0, u0_cx, u0_cy, alpha, beta, dG, polarity, X_eq, Y_eq):
    """CIP Thiele RHS — pure scalar.

    Parameters:
        X, Y        : core position [m]
        w0          : effective omega0 (including field shift)
        u0_cx, u0_cy: pre-computed u0 * current_dir components = u0_prefactor * J * dir
        alpha       : Gilbert damping
        beta        : non-adiabatic STT parameter
        dG          : D/G0 ratio
        polarity    : +1 or -1
        X_eq, Y_eq  : equilibrium position from in-plane field [m]
    """
    Xr = X - X_eq
    Yr = Y - Y_eq
    p = polarity

    det = (alpha * dG) * (alpha * dG) + p * p

    rhs_I = -w0 * Xr - p * u0_cy + beta * dG * u0_cx
    rhs_II = -w0 * Yr + p * u0_cx + beta * dG * u0_cy

    dXdt = (alpha * dG * rhs_I + p * rhs_II) / det
    dYdt = (-p * rhs_I + alpha * dG * rhs_II) / det
    return dXdt, dYdt


@njit(fastmath=True)
def integrate_cip_rk4(
    t0,
    t1,
    dt_out,
    X0,
    Y0,
    w0,
    u0_cx,
    u0_cy,
    alpha,
    beta,
    dG,
    polarity,
    X_eq,
    Y_eq,
    substeps,
):
    """Fixed-step RK4 integration of CIP Thiele equation.

    Returns
    -------
    t_out, X_out, Y_out : ndarray (n_steps,)
    """
    n_steps = int((t1 - t0) / dt_out) + 1
    dt_sub = dt_out / float(substeps)

    t_out = np.empty(n_steps, dtype=np.float64)
    X_out = np.empty(n_steps, dtype=np.float64)
    Y_out = np.empty(n_steps, dtype=np.float64)

    X = X0
    Y = Y0
    t = t0

    for i in range(n_steps):
        t_out[i] = t
        X_out[i] = X
        Y_out[i] = Y

        for _s in range(substeps):
            k1x, k1y = _cip_rhs(
                X, Y, w0, u0_cx, u0_cy, alpha, beta, dG, polarity, X_eq, Y_eq
            )

            k2x, k2y = _cip_rhs(
                X + 0.5 * dt_sub * k1x,
                Y + 0.5 * dt_sub * k1y,
                w0,
                u0_cx,
                u0_cy,
                alpha,
                beta,
                dG,
                polarity,
                X_eq,
                Y_eq,
            )

            k3x, k3y = _cip_rhs(
                X + 0.5 * dt_sub * k2x,
                Y + 0.5 * dt_sub * k2y,
                w0,
                u0_cx,
                u0_cy,
                alpha,
                beta,
                dG,
                polarity,
                X_eq,
                Y_eq,
            )

            k4x, k4y = _cip_rhs(
                X + dt_sub * k3x,
                Y + dt_sub * k3y,
                w0,
                u0_cx,
                u0_cy,
                alpha,
                beta,
                dG,
                polarity,
                X_eq,
                Y_eq,
            )

            X += (dt_sub / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            Y += (dt_sub / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

            t += dt_sub

    return t_out, X_out, Y_out


# =====================================================================
# Warm-up helper (trigger JIT compilation before timing-critical code)
# =====================================================================


def warmup():
    """Trigger Numba compilation for both kernels (called once at import)."""
    if not HAS_NUMBA:
        return
    # CPP — tiny integration to trigger compilation
    integrate_cpp_rk4(
        0.0, 1e-10, 1e-11, 0.01, 0.0, 1e8, 5e9, 0.25, 0.01, 0.018, 1, 0.0, 0.0, 2
    )
    # CIP
    integrate_cip_rk4(
        0.0, 1e-10, 1e-11, 1e-9, 0.0, 5e9, 1.0, 0.0, 0.01, 0.005, 0.5, 1, 0.0, 0.0, 2
    )


__all__ = [
    "HAS_NUMBA",
    "integrate_cpp_rk4",
    "integrate_cip_rk4",
    "warmup",
]
