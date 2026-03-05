# ruff: noqa: N802, N803, N806, PLR0913
"""
Spin wave dispersion relation models.

Provides analytical models for magnon/spin wave dispersion in thin films.

References
----------
- B. A. Kalinikos & A. N. Slavin, J. Phys. C 19, 7013 (1986)
- R. W. Damon & J. R. Eshbach, J. Phys. Chem. Solids 19, 308 (1961)
- T. Böttcher et al., IEEE Trans. Magn. 57, 9427561 (2021)
- J.-V. Kim et al., Phys. Rev. Lett. 117, 197204 (2016)
- D. Cortés-Ortuño & P. Landeros, J. Phys.: Condens. Matter 25, 156001 (2013)
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

from .base import DispersionResult
from .constants import MU0, gamma


# ==============================================================================
# Helper functions
# ==============================================================================

def _dipolar_factor_P(k: np.ndarray, d: float) -> np.ndarray:
    """
    Compute the thin-film dipolar thickness factor P(kd).
    
    P(kd) = 1 - (1 - exp(-|k|d)) / (|k|d)
    
    Uses |k| for stability with both positive and negative k.
    P(0) = 0 in the limit.
    """
    kd = np.abs(k) * d
    
    # Use expm1 for numerical stability: 1 - exp(-x) = -expm1(-x)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = 1.0 - (-np.expm1(-kd)) / (kd + 1e-30)
    
    # Handle k=0 case
    P = np.where(kd > 1e-10, P, 0.0)
    return P


def _exchange_length_squared(Aex: float, Ms: float) -> float:
    """
    Exchange length squared: l_ex² = 2A / (μ₀ Ms²).
    """
    return 2.0 * Aex / (MU0 * Ms * Ms)


# ==============================================================================
# Cubic anisotropy helpers
# ==============================================================================

def _cubic_energy_deriv1(phi_M: float, Kc1: float, phi_ani: float) -> float:
    """First derivative of in-plane cubic energy w.r.t. φ (torque).

    For α₃ ≈ 0 (in-plane film):
        E_cub ≈ (Kc1/4) sin²(2(φ − φ_ani))
        dE/dφ = (Kc1/2) sin(4(φ − φ_ani))
    """
    return 0.5 * Kc1 * math.sin(4.0 * (phi_M - phi_ani))


def _cubic_energy_deriv2(phi_M: float, Kc1: float, phi_ani: float) -> float:
    """Second derivative of in-plane cubic energy w.r.t. φ (stiffness).

        d²E/dφ² = 2 Kc1 cos(4(φ − φ_ani))
    """
    return 2.0 * Kc1 * math.cos(4.0 * (phi_M - phi_ani))


def _cubic_equilibrium_angle(
    B: float,
    Ms: float,
    Kc1: float,
    phi_H: float,
    phi_ani: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Solve the in-plane equilibrium angle φ_M via Newton iteration.

    Equation:  H_ext sin(φ_M − φ_H) + (Kc1 / (2μ₀Ms)) sin(4(φ_M − φ_ani)) = 0

    If |Kc1| is negligible compared to H_ext, φ_M ≈ φ_H.
    """
    H_ext = B / MU0
    coeff = Kc1 / (2.0 * MU0 * Ms)

    # Skip iteration if anisotropy is negligible
    if abs(coeff) < 1e-6 * max(abs(H_ext), 1.0):
        return phi_H

    phi_M = phi_H  # initial guess
    for _ in range(max_iter):
        f_val = H_ext * math.sin(phi_M - phi_H) + coeff * math.sin(4.0 * (phi_M - phi_ani))
        df_val = H_ext * math.cos(phi_M - phi_H) + 4.0 * coeff * math.cos(4.0 * (phi_M - phi_ani))
        if abs(df_val) < 1e-30:
            break
        delta = f_val / df_val
        phi_M -= delta
        if abs(delta) < tol:
            break
    return phi_M


def _cubic_stiffness_field(
    phi_M: float,
    Kc1: float,
    Ms: float,
    phi_ani: float,
) -> float:
    """In-plane stiffness field from cubic anisotropy [A/m].

    H_cub^(φφ) = (1 / μ₀Ms) d²E_cub/dφ² = (2Kc1 / μ₀Ms) cos(4(φ_M − φ_ani))
    """
    return _cubic_energy_deriv2(phi_M, Kc1, phi_ani) / (MU0 * Ms)


# ==============================================================================
# Dispersion models
# ==============================================================================

def kalinikos(
    *,
    k: Union[float, np.ndarray],
    B: float,
    Ms: float,
    d: float,
    Aex: float,
    Ku: float = 0.0,
    Kc1: float = 0.0,
    Kc2: float = 0.0,
    phi: float = np.pi / 2,
    phi_ani: float = 0.0,
    g: float = 2.0,
) -> DispersionResult:
    """
    Kalinikos-Slavin dipole-exchange dispersion (1986)
    with optional cubic anisotropy (Kc1, Kc2).
    
    General in-plane magnetized thin film dispersion with arbitrary
    propagation angle phi between k and M.
    
    Parameters
    ----------
    k : float or array
        Wavevector in 1/m (can be negative for nonreciprocal effects)
    B : float
        Applied magnetic field in Tesla (in-plane)
    Ms : float
        Saturation magnetization in A/m
    d : float
        Film thickness in meters
    Aex : float
        Exchange stiffness in J/m
    Ku : float, optional
        Uniaxial anisotropy in J/m³ (default: 0)
    Kc1 : float, optional
        First-order cubic anisotropy constant in J/m³ (default: 0).
        Enters as in-plane four-fold stiffness field.
    Kc2 : float, optional
        Second-order cubic anisotropy constant in J/m³ (default: 0).
        Reserved — currently only Kc1 is used in the dispersion.
    phi : float, optional
        Angle of wavevector k relative to the **applied field** direction,
        in radians (default: π/2 for DE geometry).
        phi=0: BVMSW (k ∥ H), phi=π/2: MSSW/DE (k ⟂ H).
        When cubic anisotropy is present, the equilibrium magnetization
        direction φ_M may differ from φ_H; this is handled automatically.
    phi_ani : float, optional
        Orientation of the first cubic axis c1 in the film plane,
        in radians (default: 0). Matches mumax3 ``anisC1`` convention.
    g : float, optional
        Landé g-factor (default: 2.0)
        
    Returns
    -------
    DispersionResult
        Result with k and f arrays
        
    Notes
    -----
    When Kc1 ≠ 0 the model:
    
    1. Solves the in-plane equilibrium angle φ_M from the torque equation
       H_ext·sin(φ_M − φ_H) + (Kc1 / 2μ₀Ms)·sin(4(φ_M − φ_ani)) = 0.
    2. Adds the cubic stiffness field
       H_cub = (2Kc1 / μ₀Ms)·cos(4(φ_M − φ_ani)) to the internal field.
    3. Reinterprets the dipolar angle as the angle between k and the
       equilibrium M direction.
    
    For Kc1 = Kc2 = 0 the formula reduces to the standard Kalinikos-Slavin.
    
    Examples
    --------
    >>> k = np.linspace(-1e7, 1e7, 500)
    >>> disp = mmpp.analytical.kalinikos(k=k, B=0.1, Ms=8e5, d=100e-9, Aex=13e-12)
    >>> disp.plt.plot()
    
    With cubic anisotropy (Fe-like, easy axes along [100]/[010]):
    
    >>> disp = mmpp.analytical.kalinikos(
    ...     k=k, B=0.04, Ms=996e3, d=20e-9, Aex=25.5e-12,
    ...     Kc1=-8.1e3, phi_ani=np.pi/4, phi=0.0,
    ... )
    
    References
    ----------
    B. A. Kalinikos & A. N. Slavin, J. Phys. C 19, 7013 (1986).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    B = float(B)
    Ms = float(Ms)
    d = float(d)
    Aex = float(Aex)
    Ku = float(Ku)
    Kc1 = float(Kc1)
    Kc2 = float(Kc2)
    phi_ani = float(phi_ani)
    gamma_val = gamma(g)
    
    # Exchange length squared
    lex2 = _exchange_length_squared(Aex, Ms)
    
    # Dipolar factor P(kd)
    P = _dipolar_factor_P(k, d)
    
    # ── Cubic anisotropy: equilibrium angle & stiffness ──────
    # phi is the angle of k relative to H direction.
    # If Kc1 != 0, equilibrium M may rotate away from H.
    phi_H = 0.0  # H defines our reference direction
    phi_M = 0.0  # magnetization equilibrium angle (relative to H)
    H_cub = 0.0  # cubic stiffness field [A/m]

    if abs(Kc1) > 0:
        # The user-supplied phi is angle(k, H).  We need angle(k, M).
        # Solve equilibrium: phi_M is measured from the lab x-axis;
        # we set H along x (phi_H = 0) so that phi = angle(k, H) = angle(k, x).
        phi_M = _cubic_equilibrium_angle(B, Ms, Kc1, phi_H, phi_ani)
        H_cub = _cubic_stiffness_field(phi_M, Kc1, Ms, phi_ani)

    # Effective dipolar angle = angle between k and equilibrium M
    phi_eff = phi - phi_M

    # Internal field H₀ (in A/m).
    # For an in-plane magnetized thin film: H₀ = H_ext_parallel + H_uni + H_cub
    H_ext_par = B / MU0  # for small phi_M this ≈ B/μ₀
    if abs(phi_M) > 1e-10:
        H_ext_par = (B / MU0) * math.cos(phi_M - phi_H)
    H0 = H_ext_par + 2.0 * Ku / (MU0 * Ms) + H_cub
    
    # Exchange stiffness field contribution (varies with k)
    H_ex = Ms * lex2 * (k * k)   # = 2A k²/(μ₀ Ms)
    
    # Angular factors (using effective angle between k and M)
    c2 = np.cos(phi_eff) ** 2
    s2 = np.sin(phi_eff) ** 2
    
    # Dipolar angular factor F_{00}(kd, φ_eff) — Kalinikos eq. 9
    denom = H0 + H_ex
    with np.errstate(divide='ignore', invalid='ignore'):
        F = 1.0 - P * c2 + (Ms * P * (1.0 - P) / (denom + 1e-30)) * s2
    
    # Characteristic frequencies (rad/s)
    omega_H = gamma_val * MU0 * max(H0, 0.0)
    omega_M = gamma_val * MU0 * Ms
    omega_ex = omega_M * lex2 * (k * k)
    
    omega0 = omega_H + omega_ex
    
    # Final dispersion: ω² = ω₀(ω₀ + ω_M·F)
    under_sqrt = omega0 * (omega0 + omega_M * F)
    under_sqrt = np.maximum(under_sqrt, 0.0)
    omega = np.sqrt(under_sqrt)
    
    f_ghz = omega / (2.0 * math.pi * 1e9)
    
    params_dict = {
        "B": B, "Ms": Ms, "d": d, "Aex": Aex, "Ku": Ku,
        "phi": phi, "g": g,
    }
    meta = {
        "geometry": f"phi_k={phi:.3f} rad",
        "reference": "J. Phys. C 19, 7013 (1986)",
    }
    if abs(Kc1) > 0 or abs(Kc2) > 0:
        params_dict.update({"Kc1": Kc1, "Kc2": Kc2, "phi_ani": phi_ani})
        meta["phi_M"] = f"{phi_M:.4f} rad"
        meta["H_cub"] = f"{MU0 * H_cub * 1e3:.2f} mT"
        meta["phi_eff"] = f"{phi_eff:.4f} rad"
    
    return DispersionResult(
        model_name="Kalinikos-Slavin 1986",
        k=k,
        f=f_ghz,
        params=params_dict,
        metadata=meta,
    )


def kalinikos_no_approx(
    *,
    k: Union[float, np.ndarray],
    B: float,
    Ms: float,
    d: float,
    Aex: float,
    Ku: float = 0.0,
    n: int = 0,
    perpendicular: bool = False,
    g: float = 2.0,
) -> DispersionResult:
    """
    Kalinikos-Slavin dispersion with PSSW mode support.
    
    Extends the standard formula to include perpendicular standing
    spin wave (PSSW) modes via the mode index n.
    Supports both in-plane and out-of-plane geometries.
    
    Parameters
    ----------
    k : float or array
        In-plane wavevector in 1/m
    B : float
        Applied magnetic field in Tesla
    Ms : float
        Saturation magnetization in A/m
    d : float
        Film thickness in meters
    Aex : float
        Exchange stiffness in J/m
    Ku : float, optional
        Uniaxial anisotropy in J/m³ (default: 0)
    n : int, optional
        PSSW mode index (default: 0 for fundamental mode)
        n > 0 includes quantization along thickness.
    perpendicular : bool, optional
        If True, use out-of-plane geometry (default: False)
    g : float, optional
        Landé g-factor (default: 2.0)
        
    Returns
    -------
    DispersionResult
        Result with k and f arrays
        
    Notes
    -----
    For n ≠ 0, the out-of-plane wavevector is k_z = nπ/d, contributing
    to the exchange energy.
    
    References
    ----------
    B. A. Kalinikos & A. N. Slavin, J. Phys. C 19, 7013 (1986).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    B = float(B)
    Ms = float(Ms)
    d = float(d)
    Aex = float(Aex)
    Ku = float(Ku)
    gamma_val = gamma(g)
    
    if n == 0 and not perpendicular:
        # Use standard in-plane geometry formula (DE, phi=π/2)
        return kalinikos(k=k, B=B, Ms=Ms, d=d, Aex=Aex, Ku=Ku, phi=np.pi/2, g=g)
    
    if n == 0 and perpendicular:
        # Delegate to forward_volume for fundamental OOP mode
        return forward_volume(k=k, B=B, Ms=Ms, d=d, Aex=Aex, Ku=Ku, g=g)
    
    # PSSW mode: include k_z quantization
    kz = abs(n) * math.pi / d
    k_total_sq = k * k + kz * kz
    
    # Dipolar factor (for in-plane k only)
    kd = np.abs(k) * d
    with np.errstate(divide='ignore', invalid='ignore'):
        Fk = np.where(kd > 1e-10, (1.0 - np.exp(-kd)) / kd, 1.0)
    
    # Anisotropy field
    Han = 2.0 * Ku / (MU0 * Ms)
    
    # Exchange field (includes k_z)
    Hex = 2.0 * Aex / (MU0 * Ms) * k_total_sq
    
    # Internal fields
    H0 = B / MU0
    omega_M = gamma_val * MU0 * Ms
    
    if perpendicular:
        # OOP: static demagnetization -Ms
        H_static = np.maximum(0.0, H0 - Ms + Han + Hex)
        omega_0 = gamma_val * MU0 * H_static
        # Dynamic demagnetization for higher OOP modes:
        # P_nn = k²/k_total² (diagonal approximation)
        P_nn = (k * k) / (k_total_sq + 1e-30)
        under_sqrt = np.maximum(omega_0 * (omega_0 + omega_M * P_nn), 0.0)
    else:
        # In-plane: same convention as kalinikos() — no in-plane
        # demagnetization subtracted from H₀.
        omega_0 = gamma_val * MU0 * (H0 + Han + Hex)
        under_sqrt = np.maximum(omega_0 * (omega_0 + omega_M * (1.0 - Fk)), 0.0)
    
    omega = np.sqrt(under_sqrt)
    f_ghz = omega / (2.0 * math.pi * 1e9)
    
    geometry = "perpendicular" if perpendicular else "in-plane"
    
    return DispersionResult(
        model_name=f"Kalinikos PSSW n={n}",
        k=k,
        f=f_ghz,
        params={"B": B, "Ms": Ms, "d": d, "Aex": Aex, "Ku": Ku, "n": n, "perpendicular": perpendicular, "g": g},
        metadata={"geometry": geometry, "mode_index": n},
    )


def damon_eshbach(
    *,
    k: Union[float, np.ndarray],
    B: float,
    Ms: float,
    d: float,
    Aex: float = 0.0,
    Ku: float = 0.0,
    g: float = 2.0,
) -> DispersionResult:
    """
    Damon-Eshbach (MSSW) surface wave dispersion.
    
    Magnetostatic surface spin waves with k perpendicular to M.
    
    Parameters
    ----------
    k : float or array
        Wavevector in 1/m (k ⟂ M)
    B : float
        Applied magnetic field in Tesla
    Ms : float
        Saturation magnetization in A/m
    d : float
        Film thickness in meters
    Aex : float, optional
        Exchange stiffness in J/m (default: 0)
    Ku : float, optional
        Uniaxial anisotropy in J/m³ (default: 0)
    g : float, optional
        Landé g-factor (default: 2.0)
        
    Returns
    -------
    DispersionResult
        Result with k and f arrays
        
    Examples
    --------
    >>> k = np.linspace(0, 2e7, 200)
    >>> de = mmpp.analytical.damon_eshbach(k=k, B=0.1, Ms=8e5, d=100e-9, Aex=13e-12)
    >>> de.plt.plot(title="Damon-Eshbach Mode")
    
    References
    ----------
    R. W. Damon & J. R. Eshbach, J. Phys. Chem. Solids 19, 308 (1961).
    """
    # DE is Kalinikos with phi = π/2
    result = kalinikos(k=k, B=B, Ms=Ms, d=d, Aex=Aex, Ku=Ku, phi=np.pi/2, g=g)
    result.model_name = "Damon-Eshbach (MSSW)"
    result.metadata["geometry"] = "k ⟂ M (surface wave)"
    result.metadata["reference"] = "J. Phys. Chem. Solids 19, 308 (1961)"
    return result


def backward_volume(
    *,
    k: Union[float, np.ndarray],
    B: float,
    Ms: float,
    d: float,
    Aex: float = 0.0,
    Ku: float = 0.0,
    g: float = 2.0,
) -> DispersionResult:
    """
    Backward Volume Magnetostatic Spin Wave (BVMSW) dispersion.
    
    Volume waves with k parallel to M, characterized by negative
    group velocity.
    
    Parameters
    ----------
    k : float or array
        Wavevector in 1/m (k ∥ M)
    B : float
        Applied magnetic field in Tesla
    Ms : float
        Saturation magnetization in A/m
    d : float
        Film thickness in meters
    Aex : float, optional
        Exchange stiffness in J/m (default: 0)
    Ku : float, optional
        Uniaxial anisotropy in J/m³ (default: 0)
    g : float, optional
        Landé g-factor (default: 2.0)
        
    Returns
    -------
    DispersionResult
        Result with k and f arrays
        
    Notes
    -----
    BVMSW modes have negative group velocity dω/dk < 0 in the
    magnetostatic regime (small k).
        
    Examples
    --------
    >>> k = np.linspace(0, 2e7, 200)
    >>> bv = mmpp.analytical.backward_volume(k=k, B=0.1, Ms=8e5, d=100e-9)
    >>> bv.plt.plot(title="Backward Volume Mode")
    """
    # BV is Kalinikos with phi = 0
    result = kalinikos(k=k, B=B, Ms=Ms, d=d, Aex=Aex, Ku=Ku, phi=0.0, g=g)
    result.model_name = "Backward Volume (BVMSW)"
    result.metadata["geometry"] = "k ∥ M (volume wave)"
    return result


def forward_volume(
    *,
    k: Union[float, np.ndarray],
    B: float,
    Ms: float,
    d: float,
    Aex: float = 0.0,
    Ku: float = 0.0,
    g: float = 2.0,
) -> DispersionResult:
    """
    Forward Volume Magnetostatic Spin Wave (FVMSW) dispersion.
    
    Volume waves in a perpendicularly magnetized film (M ⟂ film plane).
    
    Parameters
    ----------
    k : float or array
        In-plane wavevector in 1/m
    B : float
        Applied magnetic field in Tesla (perpendicular to film)
    Ms : float
        Saturation magnetization in A/m
    d : float
        Film thickness in meters
    Aex : float, optional
        Exchange stiffness in J/m (default: 0)
    Ku : float, optional
        Uniaxial anisotropy in J/m³ (default: 0)
    g : float, optional
        Landé g-factor (default: 2.0)
        
    Returns
    -------
    DispersionResult
        Result with k and f arrays
        
    Notes
    -----
    Dispersion relation:
    
    .. math::
    
        \\omega^2 = \\omega_0 (\\omega_0 + \\omega_M P(kd))
    
    where ω₀ = γ·max(0, B_int + B_ex), B_int = B − μ₀Ms + 2Ku/Ms.
        
    Examples
    --------
    >>> k = np.linspace(0, 2e7, 200)
    >>> fv = mmpp.analytical.forward_volume(k=k, B=0.5, Ms=1.4e5, d=1e-6)
    >>> fv.plt.plot(title="Forward Volume Mode")
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    B = float(B)
    Ms = float(Ms)
    d = float(d)
    Aex = float(Aex)
    Ku = float(Ku)
    gamma_val = gamma(g)
    
    P = _dipolar_factor_P(k, d)
    
    # Exchange contribution
    B_ex = 2.0 * Aex * (k * k) / Ms if Aex > 0 else 0.0
    
    # Internal field for OOP geometry: B_ext − μ₀Ms + 2Ku/Ms
    h_anis = 2.0 * Ku / Ms
    B_internal = B - MU0 * Ms + h_anis
    
    omega0 = gamma_val * np.maximum(0.0, B_internal + B_ex)
    omega_M = gamma_val * MU0 * Ms
    
    under_sqrt = omega0 * (omega0 + omega_M * P)
    under_sqrt = np.maximum(under_sqrt, 0.0)
    omega = np.sqrt(under_sqrt)
    
    f_ghz = omega / (2.0 * math.pi * 1e9)
    
    return DispersionResult(
        model_name="Forward Volume (FVMSW)",
        k=k,
        f=f_ghz,
        params={"B": B, "Ms": Ms, "d": d, "Aex": Aex, "Ku": Ku, "g": g},
        metadata={"geometry": "M ⟂ film plane"},
    )


def bottcher(
    *,
    k: Union[float, np.ndarray],
    B: float,
    Ms: float,
    d: float,
    Aex: float,
    Ku: float = 0.0,
    perpendicular: bool = False,
    g: float = 2.0,
) -> DispersionResult:
    """
    Böttcher et al. 2021 dipole-exchange dispersion.
    
    Accurate formula for ultrathin films without DMI.
    Supports both in-plane and out-of-plane (perpendicular) geometries.
    
    Parameters
    ----------
    k : float or array
        Wavevector in 1/m
    B : float
        Applied magnetic field in Tesla
    Ms : float
        Saturation magnetization in A/m
    d : float
        Film thickness in meters
    Aex : float
        Exchange stiffness in J/m
    Ku : float, optional
        Uniaxial anisotropy in J/m³ (default: 0)
    perpendicular : bool, optional
        If True, use out-of-plane geometry (default: False)
    g : float, optional
        Landé g-factor (default: 2.0)
        
    Returns
    -------
    DispersionResult
        Result with k and f arrays
        
    References
    ----------
    T. Böttcher et al., IEEE Trans. Magn. 57, 9427561 (2021).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    B = float(B)
    Ms = float(Ms)
    d = float(d)
    Aex = float(Aex)
    Ku = float(Ku)
    gamma_val = gamma(g)
    
    # Form factor g(x) = 1 - (1 - exp(-|x|)) / |x|
    kd = np.abs(k) * d
    with np.errstate(divide='ignore', invalid='ignore'):
        gk = np.where(kd > 1e-10, 1.0 - (1.0 - np.exp(-kd)) / kd, 0.0)
    
    # Exchange length
    lam_ex = 2.0 * Aex / (MU0 * Ms)
    
    # Anisotropy field
    H_u = 2.0 * Ku / (MU0 * Ms)
    H_ext = B / MU0
    
    if perpendicular:
        # OOP: H_int = H_ext - Ms + H_u
        H_int = np.maximum(0.0, H_ext - Ms + H_u)
        term1 = H_int + lam_ex * k * k
        term2 = H_int + lam_ex * k * k + Ms * gk
    else:
        # In-plane Böttcher
        term1 = H_ext + lam_ex * k * k + Ms * gk
        term2 = H_ext - H_u + lam_ex * k * k + Ms - Ms * gk
    
    radicand = term1 * term2
    radicand = np.maximum(radicand, 0.0)
    
    omega = gamma_val * MU0 * np.sqrt(radicand)
    f_ghz = omega / (2.0 * math.pi * 1e9)
    
    geometry = "perpendicular" if perpendicular else "in-plane"
    
    return DispersionResult(
        model_name="Böttcher 2021",
        k=k,
        f=f_ghz,
        params={"B": B, "Ms": Ms, "d": d, "Aex": Aex, "Ku": Ku, "perpendicular": perpendicular, "g": g},
        metadata={"geometry": geometry, "reference": "IEEE Trans. Magn. 57, 9427561 (2021)"},
    )


def kim(
    *,
    k: Union[float, np.ndarray],
    B: float,
    Ms: float,
    d: float,
    Aex: float,
    D: float = 0.0,
    Ku: float = 0.0,
    phi: float = 0.0,
    g: float = 2.0,
) -> DispersionResult:
    """
    Kim et al. 2016 dispersion with interfacial DMI.
    
    Includes the asymmetric (nonreciprocal) contribution from
    Dzyaloshinskii-Moriya interaction.
    
    Parameters
    ----------
    k : float or array
        Wavevector in 1/m
    B : float
        Applied magnetic field in Tesla
    Ms : float
        Saturation magnetization in A/m
    d : float
        Film thickness in meters
    Aex : float
        Exchange stiffness in J/m
    D : float, optional
        Interfacial DMI constant in J/m² (default: 0)
    Ku : float, optional
        Uniaxial anisotropy in J/m³ (default: 0)
    phi : float, optional
        Propagation angle in radians (default: 0 for DE geometry)
    g : float, optional
        Landé g-factor (default: 2.0)
        
    Returns
    -------
    DispersionResult
        Result with k and f arrays
        
    Notes
    -----
    DMI induces a linear-in-k frequency shift:
    
    .. math::
    
        \\Delta f_{DMI} = \\frac{\\gamma D k \\cos\\phi}{\\pi M_s}
    
    This breaks the f(k) = f(-k) symmetry.
    
    Examples
    --------
    >>> k = np.linspace(-1e7, 1e7, 500)
    >>> # With 1 mJ/m² DMI
    >>> disp = mmpp.analytical.kim(k=k, B=0.1, Ms=8e5, d=20e-9, Aex=13e-12, D=1e-3)
    >>> disp.plt.plot()  # Asymmetric dispersion!
    
    References
    ----------
    J.-V. Kim et al., Phys. Rev. Lett. 117, 197204 (2016).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    B = float(B)
    Ms = float(Ms)
    d = float(d)
    Aex = float(Aex)
    D = float(D)
    Ku = float(Ku)
    gamma_val = gamma(g)
    
    theta = phi  # Angle for consistency with paper convention
    kx = k * np.cos(theta)
    
    # Exchange contribution
    exch = 2.0 * Aex * k * k / Ms
    
    # Demagnetizing contributions (ultrathin approximation)
    with np.errstate(divide='ignore', invalid='ignore'):
        dem_s = np.where(k != 0, MU0 * Ms * d * kx * kx / (2.0 * np.abs(k)), 0.0)
        dem_b = np.where(k != 0, MU0 * Ms * d * np.abs(k) / 2.0, 0.0)
    
    # Anisotropy contribution
    anis = 2.0 * (Ku / Ms - MU0 * Ms / 2.0)
    
    # Two branches
    branch1 = B + exch + dem_s
    branch2 = B + exch - anis - dem_b
    
    radicand = branch1 * branch2
    radicand = np.maximum(radicand, 0.0)
    
    # Symmetric part
    omega_sym = gamma_val * np.sqrt(radicand)
    
    # DMI contribution (linear in k)
    omega_dmi = 2.0 * gamma_val * D * kx / Ms
    
    omega = omega_sym + omega_dmi
    omega = np.maximum(omega, 0.0)
    
    f_ghz = omega / (2.0 * math.pi * 1e9)
    
    return DispersionResult(
        model_name="Kim 2016 (DMI)",
        k=k,
        f=f_ghz,
        params={"B": B, "Ms": Ms, "d": d, "Aex": Aex, "D": D, "Ku": Ku, "phi": phi, "g": g},
        metadata={
            "geometry": f"phi={phi:.3f} rad",
            "dmi": D,
            "reference": "Phys. Rev. Lett. 117, 197204 (2016)",
        },
    )


def cortes_ortuno(
    *,
    k: Union[float, np.ndarray],
    B: float,
    Ms: float,
    d: float,
    Aex: float,
    D: float = 0.0,
    Ku: float = 0.0,
    phi: float = 0.0,
    g: float = 2.0,
) -> DispersionResult:
    """
    Cortés-Ortuño & Landeros 2013 general dispersion with DMI.
    
    Comprehensive model for in-plane magnetized thin films with
    interfacial DMI, valid for arbitrary film thickness.
    
    Parameters
    ----------
    k : float or array
        Wavevector in 1/m
    B : float
        Applied magnetic field in Tesla
    Ms : float
        Saturation magnetization in A/m
    d : float
        Film thickness in meters
    Aex : float
        Exchange stiffness in J/m
    D : float, optional
        Interfacial DMI constant in J/m² (default: 0)
    Ku : float, optional
        Uniaxial anisotropy in J/m³ (default: 0)
    phi : float, optional
        Propagation angle in radians (default: 0 for DE)
    g : float, optional
        Landé g-factor (default: 2.0)
        
    Returns
    -------
    DispersionResult
        Result with k and f arrays
        
    Notes
    -----
    More accurate than Kim 2016 for thicker films. Reduces to Kim
    in the ultrathin (|kd| ≪ 1) limit.
    
    References
    ----------
    D. Cortés-Ortuño & P. Landeros, J. Phys.: Condens. Matter 25, 156001 (2013).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    B = float(B)
    Ms = float(Ms)
    d = float(d)
    Aex = float(Aex)
    D = float(D)
    Ku = float(Ku)
    gamma_val = gamma(g)
    
    theta = phi
    H = B / MU0
    
    # Demagnetizing term
    with np.errstate(divide='ignore', invalid='ignore'):
        demag = np.where(
            k != 0,
            Ms * d * (np.cos(theta) * k) ** 2 / (2.0 * np.abs(k)),
            0.0
        )
    
    # Exchange
    exch = 2.0 * Aex * k * k / Ms
    
    # Two main terms
    part1 = MU0 * (H + demag) + exch
    
    # Thickness factor for part2
    with np.errstate(divide='ignore', invalid='ignore'):
        thick_factor = np.where(k != 0, 1.0 - np.abs(k) * d / 2.0, 1.0)
    
    part2 = MU0 * (H + Ms * thick_factor) + 2.0 * (Aex * k * k - Ku) / Ms
    
    radicand = part1 * part2
    radicand = np.maximum(radicand, 0.0)
    
    # Symmetric frequency
    omega_sym = gamma_val * np.sqrt(radicand)
    
    # DMI contribution
    omega_dmi = 2.0 * gamma_val * D * np.cos(theta) * k / Ms
    
    omega = omega_sym + omega_dmi
    omega = np.maximum(omega, 0.0)
    
    f_ghz = omega / (2.0 * math.pi * 1e9)
    
    return DispersionResult(
        model_name="Cortés-Ortuño 2013",
        k=k,
        f=f_ghz,
        params={"B": B, "Ms": Ms, "d": d, "Aex": Aex, "D": D, "Ku": Ku, "phi": phi, "g": g},
        metadata={
            "geometry": f"phi={phi:.3f} rad",
            "dmi": D,
            "reference": "J. Phys.: Condens. Matter 25, 156001 (2013)",
        },
    )
