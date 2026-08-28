"""Voltage reconstruction from current and resistance traces."""

from __future__ import annotations

import numpy as np

from .models import MagnetoresistanceResult, VoltageResult


def _coerce_current_array(
    current_a: float | np.ndarray,
    *,
    n_samples: int,
) -> np.ndarray:
    if np.isscalar(current_a):
        return np.full(int(n_samples), float(np.asarray(current_a).item()), dtype=float)

    values = np.asarray(current_a, dtype=float).reshape(-1)
    if values.size != int(n_samples):
        raise ValueError(
            f"Current array length mismatch: expected {n_samples}, got {values.size}"
        )
    return values


def compute_voltage(
    magnetoresistance: MagnetoresistanceResult,
    *,
    current_a: float | np.ndarray,
) -> VoltageResult:
    """Compute voltage trace ``V(t) = I(t) * R(t)``."""
    resistance = np.asarray(magnetoresistance.resistance_ohm, dtype=float)
    current = _coerce_current_array(current_a, n_samples=resistance.size)
    voltage = current * resistance

    return VoltageResult(
        time=np.asarray(magnetoresistance.time, dtype=float),
        voltage_v=np.asarray(voltage, dtype=float),
        current_a=np.asarray(current, dtype=float),
        resistance_ohm=np.asarray(resistance, dtype=float),
        metadata={
            "source": "mr",
            "mr_method": magnetoresistance.method,
        },
    )


__all__ = ["compute_voltage"]
