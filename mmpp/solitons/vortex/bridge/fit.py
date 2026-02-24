"""Fitting glue for numerical trajectory -> analytical model."""

from __future__ import annotations

from ..model.thiele.fit import fit_from_trajectory

def fit_thiele_from_trajectory(*args, **kwargs):
    """Fit Thiele proxy parameters directly from a numerical trajectory."""
    return fit_from_trajectory(*args, **kwargs)


__all__ = ["fit_thiele_from_trajectory"]
