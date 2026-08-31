"""Trajectory-to-Thiele proxy fitting helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmpp._shared.repr_html import make_simple_card

from ..._shared.models import TrajectoryResult


@dataclass
class ThieleTrajectoryFitResult:
    """Kinematic trajectory summary; not a physical Thiele-parameter fit."""

    omega0_rad_s: float
    radius_m: float
    center: tuple[float, float]
    damping: float
    nonlinear_coeff_N: float
    simulated_trajectory: TrajectoryResult
    is_physical_parameter_fit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def _repr_html_(self) -> str:
        rows = [
            ("omega0_rad_s", f"{self.omega0_rad_s:.6g}"),
            ("radius_m", f"{self.radius_m:.6g}"),
            ("center", f"({self.center[0]:.6g}, {self.center[1]:.6g})"),
            ("damping", f"{self.damping:.6g}"),
            ("nonlinear_coeff_N", f"{self.nonlinear_coeff_N:.6g}"),
            ("physical parameter fit", "no - kinematic proxy only"),
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


def summarize_trajectory_kinematics(*args, **kwargs):
    """Summarize orbit kinematics and synthesize a circular proxy trajectory.

    This routine does not identify physical Thiele parameters.  A trajectory
    alone generally cannot separate spin torque, damping, stiffness, and
    nonlinear frequency shift.  It only measures orbit center, average radius,
    and angular rate before synthesizing a damped circular comparison curve.
    """
    if not args:
        raise TypeError(
            "summarize_trajectory_kinematics requires a TrajectoryResult argument"
        )
    trajectory = args[0]
    if not isinstance(trajectory, TrajectoryResult):
        raise TypeError(
            "summarize_trajectory_kinematics expects TrajectoryResult as first argument"
        )

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
    radius_mean = float(np.mean(np.abs(z)))
    radius_initial = float(np.abs(z[0])) if z.size else 0.0
    omega0 = _estimate_omega_from_trajectory(time, z)
    if not np.isfinite(omega0):
        omega0 = 0.0

    t0 = float(time[0])
    phase0 = float(np.angle(z[0])) if z.size else 0.0
    duration = max(float(time[-1] - t0), 1e-30)
    damping_eff = float(max(damping, 0.0))
    decay = np.exp(-damping_eff * (time - t0) / duration)
    phase = phase0 + omega0 * (time - t0)

    radius_base = radius_initial if damping_eff > 0.0 else radius_mean
    z_sim = radius_base * decay * np.exp(1j * phase)
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
            "radius_m": float(radius_base),
            "radius_mean_m": float(radius_mean),
            "radius_initial_m": float(radius_initial),
            "center": (x0, y0),
            "damping": float(damping_eff),
        },
    )

    return ThieleTrajectoryFitResult(
        omega0_rad_s=float(omega0),
        radius_m=float(radius_base),
        center=(x0, y0),
        damping=float(damping_eff),
        nonlinear_coeff_N=0.0,
        simulated_trajectory=sim_traj,
        is_physical_parameter_fit=False,
        metadata={
            "fit_kind": "kinematic_trajectory_proxy",
            "is_physical_parameter_fit": False,
            "identifiable_quantities": ("center", "mean_radius", "angular_rate"),
            "non_identifiable_quantities": (
                "Gilbert damping",
                "spin-torque efficiency",
                "stiffness",
                "nonlinear frequency coefficient",
            ),
            "source_method": trajectory.method,
            "radius_mean_m": float(radius_mean),
            "radius_initial_m": float(radius_initial),
        },
    )


def fit_from_trajectory(*args, **kwargs):
    """Deprecated alias for :func:`summarize_trajectory_kinematics`.

    The historical name implied that physical Thiele coefficients were
    identifiable from one trajectory.  Use the explicit kinematic name, or the
    multi-current ``fit_omega0_N_to_fJ``/autofit workflow for parameter fitting.
    """
    warnings.warn(
        "fit_from_trajectory returns a kinematic proxy, not fitted physical "
        "Thiele parameters; use summarize_trajectory_kinematics for clarity",
        DeprecationWarning,
        stacklevel=2,
    )
    return summarize_trajectory_kinematics(*args, **kwargs)


__all__ = [
    "ThieleTrajectoryFitResult",
    "fit_from_trajectory",
    "summarize_trajectory_kinematics",
]
