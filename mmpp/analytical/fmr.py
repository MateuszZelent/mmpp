# ruff: noqa: N802, N803, N806, PLR0913
"""
Ferromagnetic Resonance (FMR) models.

Provides analytical models for uniform FMR frequencies in thin films.

References
----------
- C. Kittel, Phys. Rev. 73, 155 (1948)
- B. A. Kalinikos & A. N. Slavin, J. Phys. C 19, 7013 (1986)
"""

from __future__ import annotations

import math

import numpy as np

from .base import FMRResult
from .constants import MU0, gamma


def kittel(
    *,
    B: float | np.ndarray | None = None,
    Ms: float,
    Ku: float = 0.0,
    g: float = 2.0,
) -> FMRResult:
    """
    Kittel formula for in-plane FMR (1948).

    Calculates the uniform (k=0) FMR frequency for a thin film
    with in-plane applied field and optional uniaxial anisotropy.

    Parameters
    ----------
    B : float or array or None
        Applied magnetic field in Tesla.
        If None, returns a "template" FMRResult with empty arrays
        but stored params — useful for passing to plotting functions
        that will supply B values automatically.
    Ms : float
        Saturation magnetization in A/m
    Ku : float, optional
        Uniaxial anisotropy constant in J/m³ (default: 0)
        Positive Ku favors perpendicular magnetization.
    g : float, optional
        Landé g-factor (default: 2.0)

    Returns
    -------
    FMRResult
        Result with B and f arrays, accessible via .plt.plot()

    Notes
    -----
    The frequency is given by:

    .. math::

        f = \\frac{\\gamma}{2\\pi} \\sqrt{B \\cdot (B + \\mu_0 M_s - 2K_u/M_s)}

    For Ku > 0 (perpendicular anisotropy), the effective demagnetizing
    field is reduced.

    Examples
    --------
    >>> import numpy as np
    >>> import mmpp
    >>> B = np.linspace(0, 0.5, 100)
    >>> fmr = mmpp.analytical.kittel(B=B, Ms=8e5, Ku=1e4)
    >>> fmr.plt.plot(title="Kittel FMR")

    >>> # Template without B (for heatmap overlay)
    >>> template = mmpp.analytical.kittel(Ms=8e5, Ku=1e4)

    References
    ----------
    C. Kittel, *Phys. Rev.* **73**, 155 (1948).
    """
    Ms = float(Ms)
    Ku = float(Ku)
    params = {"Ms": Ms, "Ku": Ku, "g": g}
    metadata = {"geometry": "in-plane", "reference": "Phys. Rev. 73, 155 (1948)"}

    if B is None:
        # Template mode: return FMRResult with empty arrays but stored params
        return FMRResult(
            model_name="Kittel 1948 (in-plane)",
            B=np.array([]),
            f=np.array([]),
            params=params,
            metadata=metadata,
        )

    B = np.atleast_1d(np.asarray(B, dtype=float))
    gamma_val = gamma(g)

    # Effective field including anisotropy
    # H_eff = B + μ0*Ms - 2*Ku/Ms (demagnetizing - anisotropy)
    H_eff = B + MU0 * Ms - 2.0 * Ku / Ms

    radicand = B * H_eff
    radicand = np.maximum(radicand, 0.0)  # Avoid negative sqrt

    omega = gamma_val * np.sqrt(radicand)
    f_ghz = omega / (2.0 * math.pi * 1e9)

    return FMRResult(
        model_name="Kittel 1948 (in-plane)",
        B=B,
        f=f_ghz,
        params=params,
        metadata=metadata,
    )


def kittel_oop(
    *,
    B: float | np.ndarray,
    Ms: float,
    Ku: float = 0.0,
    g: float = 2.0,
) -> FMRResult:
    """
    Kittel formula for out-of-plane (perpendicular) FMR.

    Calculates the uniform FMR frequency when both field and
    magnetization are perpendicular to the film plane.

    Parameters
    ----------
    B : float or array
        Applied magnetic field in Tesla (perpendicular to film)
    Ms : float
        Saturation magnetization in A/m
    Ku : float, optional
        Uniaxial anisotropy constant in J/m³ (default: 0)
    g : float, optional
        Landé g-factor (default: 2.0)

    Returns
    -------
    FMRResult
        Result with B and f arrays

    Notes
    -----
    For perpendicular geometry:

    .. math::

        f = \\frac{\\gamma}{2\\pi} (B - \\mu_0 M_s + 2K_u/M_s)

    This is a linear relation (no sqrt), valid for saturated
    perpendicular magnetization.

    Examples
    --------
    >>> B = np.linspace(0.3, 1.0, 100)
    >>> fmr = mmpp.analytical.kittel_oop(B=B, Ms=1.4e6, Ku=1e6)
    >>> fmr.plt.plot(title="OOP FMR")

    References
    ----------
    B. A. Kalinikos & A. N. Slavin, J. Phys. C 19, 7013 (1986).
    """
    B = np.atleast_1d(np.asarray(B, dtype=float))
    Ms = float(Ms)
    Ku = float(Ku)
    gamma_val = gamma(g)

    # Effective frequency (linear in B for OOP)
    omega = gamma_val * (B - MU0 * Ms + 2.0 * Ku / Ms)
    omega = np.maximum(omega, 0.0)  # Frequency must be positive

    f_ghz = omega / (2.0 * math.pi * 1e9)

    return FMRResult(
        model_name="Kittel OOP",
        B=B,
        f=f_ghz,
        params={"Ms": Ms, "Ku": Ku, "g": g},
        metadata={
            "geometry": "perpendicular",
            "reference": "J. Phys. C 19, 7013 (1986)",
        },
    )


def kittel_exchange(
    *,
    B: float | np.ndarray,
    Ms: float,
    Aex: float,
    k: float = 0.0,
    Ku: float = 0.0,
    g: float = 2.0,
) -> FMRResult:
    """
    Kittel formula with exchange contribution.

    Extends the in-plane Kittel formula to include exchange stiffness,
    allowing calculation of spin wave frequencies at finite k.

    Parameters
    ----------
    B : float or array
        Applied magnetic field in Tesla
    Ms : float
        Saturation magnetization in A/m
    Aex : float
        Exchange stiffness in J/m
    k : float, optional
        Wavevector magnitude in 1/m (default: 0 for FMR)
    Ku : float, optional
        Uniaxial anisotropy constant in J/m³ (default: 0)
    g : float, optional
        Landé g-factor (default: 2.0)

    Returns
    -------
    FMRResult
        Result with B and f arrays

    Notes
    -----
    The exchange field contribution is:

    .. math::

        B_{ex} = \\frac{2A_{ex}}{M_s} k^2

    Which adds to the internal field in the Kittel formula.

    Examples
    --------
    >>> B = np.linspace(0, 0.5, 100)
    >>> # Spin wave with k = 1e7 1/m
    >>> sw = mmpp.analytical.kittel_exchange(B=B, Ms=8e5, Aex=13e-12, k=1e7)
    >>> sw.plt.plot()
    """
    B = np.atleast_1d(np.asarray(B, dtype=float))
    Ms = float(Ms)
    Aex = float(Aex)
    Ku = float(Ku)
    gamma_val = gamma(g)

    # Exchange field contribution
    B_ex = 2.0 * Aex * k * k / Ms

    # Total internal field
    B_int = B + B_ex
    H_eff = B_int + MU0 * Ms - 2.0 * Ku / Ms

    radicand = B_int * H_eff
    radicand = np.maximum(radicand, 0.0)

    omega = gamma_val * np.sqrt(radicand)
    f_ghz = omega / (2.0 * math.pi * 1e9)

    return FMRResult(
        model_name="Kittel + Exchange",
        B=B,
        f=f_ghz,
        params={"Ms": Ms, "Aex": Aex, "k": k, "Ku": Ku, "g": g},
        metadata={"geometry": "in-plane", "includes_exchange": True},
    )
