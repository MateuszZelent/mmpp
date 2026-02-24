"""Trajectory-to-Thiele proxy fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmpp._shared.repr_html import make_simple_card
from ..._shared.models import TrajectoryResult


@dataclass
class ThieleTrajectoryFitResult:
    """Lightweight fit result for trajectory -> Thiele proxy model."""

    omega0_rad_s: float
    radius_m: float
    center: tuple[float, float]
    damping: float
    nonlinear_coeff_N: float
    simulated_trajectory: TrajectoryResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def _repr_html_(self) -> str:
        rows = [
            ("omega0_rad_s", f"{self.omega0_rad_s:.6g}"),
            ("radius_m", f"{self.radius_m:.6g}"),
            ("center", f"({self.center[0]:.6g}, {self.center[1]:.6g})"),
            ("damping", f"{self.damping:.6g}"),
            ("nonlinear_coeff_N", f"{self.nonlinear_coeff_N:.6g}"),
            (".simulated_trajectory", "TrajectoryResult usable in compare/plot"),
        ]
        return make_simple_card(
            title="ThieleTrajectoryFitResult",
            subtitle="Proxy fit from numerical trajectory to Thiele-like model",
            rows=rows,
        )


def _estimate_omega_from_trajectory(time: np.ndarray, z: np.ndarray) -> float:
    if time.size < 2:
        return float("nan")
    phi = np.unwrap(np.angle(z))
    omega = np.gradient(phi, time)
    finite = np.isfinite(omega)
    if not np.any(finite):
        return float("nan")
    return float(np.median(omega[finite]))


def fit_from_trajectory(*args, **kwargs):
    """Fit a minimal Thiele-like proxy and return simulated trajectory.

    This is a pragmatic stage-3 fit: it captures orbit center, average radius
    and angular frequency from the numerical trajectory, then synthesizes a
    damped circular proxy trajectory for side-by-side comparison.
    """
    if not args:
        raise TypeError("fit_from_trajectory requires a TrajectoryResult argument")
    trajectory = args[0]
    if not isinstance(trajectory, TrajectoryResult):
        raise TypeError("fit_from_trajectory expects TrajectoryResult as first argument")

    damping = float(kwargs.pop("damping", 0.0))
    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unknown keyword arguments: {unknown}")

    time = np.asarray(trajectory.time, dtype=float)
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)
    if time.size < 2 or x.size != time.size or y.size != time.size:
        raise ValueError("trajectory must contain at least 2 consistent samples")

    x0 = float(np.mean(x))
    y0 = float(np.mean(y))
    z = (x - x0) + 1j * (y - y0)
    radius = float(np.mean(np.abs(z)))
    omega0 = _estimate_omega_from_trajectory(time, z)
    if not np.isfinite(omega0):
        omega0 = 0.0

    t0 = float(time[0])
    phase0 = float(np.angle(z[0])) if z.size else 0.0
    duration = max(float(time[-1] - t0), 1e-30)
    decay = np.exp(-max(damping, 0.0) * (time - t0) / duration)
    phase = phase0 + omega0 * (time - t0)

    z_sim = radius * decay * np.exp(1j * phase)
    x_sim = x0 + np.real(z_sim)
    y_sim = y0 + np.imag(z_sim)

    sim_traj = TrajectoryResult(
        time=time,
        x=np.asarray(x_sim, dtype=float),
        y=np.asarray(y_sim, dtype=float),
        polarity=np.asarray(trajectory.polarity, dtype=int),
        method="thiele_fit_proxy",
        confidence=np.ones_like(time, dtype=float),
        metadata={
            "source_method": trajectory.method,
            "omega0_rad_s": float(omega0),
            "radius_m": float(radius),
            "center": (x0, y0),
            "damping": float(damping),
        },
    )

    return ThieleTrajectoryFitResult(
        omega0_rad_s=float(omega0),
        radius_m=float(radius),
        center=(x0, y0),
        damping=float(max(damping, 0.0)),
        nonlinear_coeff_N=0.0,
        simulated_trajectory=sim_traj,
        metadata={
            "fit_kind": "trajectory_proxy",
            "source_method": trajectory.method,
        },
    )


__all__ = ["ThieleTrajectoryFitResult", "fit_from_trajectory"]
