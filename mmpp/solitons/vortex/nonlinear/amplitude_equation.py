"""Amplitude-equation helpers for vortex auto-oscillator analysis."""

from __future__ import annotations

import numpy as np

from ..core.models import TrajectoryResult
from .models import AmplitudeEquationResult


def compute_amplitude_equation(
    trajectory: TrajectoryResult,
    *,
    reference_radius: float | None = None,
    center: tuple[float, float] | None = None,
    method: str = "complex",
) -> AmplitudeEquationResult:
    """Compute normalized complex amplitude ``c(t)`` and derived quantities."""
    method_norm = method.lower()
    if method_norm != "complex":
        raise ValueError("Only method='complex' is currently supported")

    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)
    time = np.asarray(trajectory.time, dtype=float)

    if center is None:
        x0 = float(np.mean(x)) if x.size else 0.0
        y0 = float(np.mean(y)) if y.size else 0.0
    else:
        x0 = float(center[0])
        y0 = float(center[1])

    z = (x - x0) + 1j * (y - y0)

    if reference_radius is None:
        reference_radius = float(np.sqrt(np.mean(np.abs(z) ** 2))) if z.size else 1.0
    reference_radius = max(float(reference_radius), 1e-18)

    c = z / reference_radius
    power = np.abs(c) ** 2
    phase = np.unwrap(np.angle(c))

    if time.size >= 2:
        omega = np.gradient(phase, time)
    else:
        omega = np.zeros_like(time, dtype=float)

    return AmplitudeEquationResult(
        time=time,
        complex_amplitude=np.asarray(c, dtype=np.complex128),
        power=np.asarray(power, dtype=float),
        phase=np.asarray(phase, dtype=float),
        omega=np.asarray(omega, dtype=float),
        method=method_norm,
        reference_radius=reference_radius,
        metadata={
            "center": (x0, y0),
            "n_points": int(time.size),
        },
    )


__all__ = ["compute_amplitude_equation"]
