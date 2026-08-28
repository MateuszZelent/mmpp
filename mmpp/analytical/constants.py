# ruff: noqa: N802, N803, N806
"""
Physical constants for magnetism calculations.

All values are in SI units.
"""

from __future__ import annotations

import math

# Vacuum permeability (H/m or N/A²)
MU0: float = 4.0 * math.pi * 1.0e-7

# Electron gyromagnetic ratio (rad/s/T) for g=2
# γ = g * μ_B / ℏ ≈ 1.76085963e11 rad/s/T for free electron
GAMMA_E: float = 1.760859630e11

# Default Landé g-factor for electrons
G_FACTOR_DEFAULT: float = 2.0


def gamma(g_factor: float = G_FACTOR_DEFAULT) -> float:
    """
    Calculate gyromagnetic ratio from Landé g-factor.

    Parameters
    ----------
    g_factor : float
        Landé g-factor (dimensionless), default 2.0

    Returns
    -------
    float
        Gyromagnetic ratio in rad/s/T

    Examples
    --------
    >>> gamma(2.0)  # Free electron
    1.760859630e+11
    >>> gamma(2.1)  # Typical for Fe, Co, Ni
    1.848902612e+11
    """
    # γ = g * (e / 2m_e) = g * γ_e / 2
    # Using GAMMA_E / 2 for consistency with backend/frontend.
    return GAMMA_E / 2.0 * g_factor


def gamma_to_ghz_per_t(g_factor: float = G_FACTOR_DEFAULT) -> float:
    """
    Calculate gyromagnetic ratio in GHz/T.

    Parameters
    ----------
    g_factor : float
        Landé g-factor (dimensionless)

    Returns
    -------
    float
        Gyromagnetic ratio in GHz/T (≈28 GHz/T for g=2)
    """
    return gamma(g_factor) / (2.0 * math.pi * 1e9)
