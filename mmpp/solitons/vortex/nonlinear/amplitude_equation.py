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
        center_source = "trajectory_mean"
    else:
        x0 = float(center[0])
        y0 = float(center[1])
        center_source = "explicit"
    if not np.isfinite([x0, y0]).all():
        raise ValueError("center must contain two finite coordinates")

    z = (x - x0) + 1j * (y - y0)

    if reference_radius is None:
        reference_radius = float(np.sqrt(np.mean(np.abs(z) ** 2))) if z.size else 1.0
        radius_source = "trajectory_rms"
    else:
        radius_source = "explicit_reference"
    reference_radius = float(reference_radius)
    if not np.isfinite(reference_radius) or reference_radius <= 0.0:
        raise ValueError("reference_radius must be finite and positive")

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
            "center_source": center_source,
            "reference_radius_source": radius_source,
            "power_comparable_across_trajectories": radius_source
            == "explicit_reference",
            "n_points": int(time.size),
        },
    )


__all__ = ["compute_amplitude_equation"]
