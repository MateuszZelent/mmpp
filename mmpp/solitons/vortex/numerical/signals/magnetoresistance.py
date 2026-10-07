"""Magnetoresistance/TMR proxy reconstruction from vortex trajectories."""

from __future__ import annotations

import numpy as np

from ..._shared.models import TrajectoryResult
from .models import MagnetoresistanceResult


def _normalize_polarizer(
    polarizer: tuple[float, float, float] | tuple[float, float],
) -> tuple[float, float, float]:
    vec = np.asarray(polarizer, dtype=float).reshape(-1)
    if vec.size not in {2, 3}:
        raise ValueError("polarizer must be a tuple of length 2 or 3")

    if vec.size == 2:
        x, y = float(vec[0]), float(vec[1])
        z = 0.0
    else:
        x, y, z = float(vec[0]), float(vec[1]), float(vec[2])

    norm = float(np.sqrt(x * x + y * y + z * z))
    if norm <= 1e-30:
        raise ValueError("polarizer cannot be a zero vector")
    return x / norm, y / norm, z / norm


def _projection_from_trajectory(
    trajectory: TrajectoryResult,
    *,
    polarizer: tuple[float, float, float] | tuple[float, float],
    disk_radius: float | None,
    disk_center: tuple[float, float] | None,
    chirality: int | None,
    xi_shape_factor: float = 2.0 / 3.0,
) -> np.ndarray:
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)
    metadata = dict(getattr(trajectory, "metadata", {}) or {})

    px, py, pz = _normalize_polarizer(polarizer)
    polarity = np.asarray(trajectory.polarity, dtype=float)
    if pz != 0.0 and np.any(polarity == 0.0):
        raise ValueError(
            "The out-of-plane trajectory proxy requires known core polarity"
        )

    if disk_radius is None:
        disk_radius = metadata.get("disk_radius")
    if disk_center is None:
        disk_center = metadata.get("disk_center")
    if px != 0.0 or py != 0.0:
        if disk_radius is None:
            raise ValueError(
                "In-plane trajectory projection requires an explicit physical "
                "disk_radius or trajectory.metadata['disk_radius']"
            )
        try:
            radius_ref = float(disk_radius)
        except (TypeError, ValueError):
            radius_ref = float("nan")
        if not np.isfinite(radius_ref) or radius_ref <= 0.0:
            raise ValueError(
                "In-plane trajectory projection requires an explicit physical "
                "disk_radius or trajectory.metadata['disk_radius']"
            )
        if disk_center is None:
            raise ValueError(
                "In-plane trajectory projection requires disk_center or "
                "trajectory.metadata['disk_center']"
            )
        center = np.asarray(disk_center, dtype=float).reshape(-1)
        if center.size != 2 or not np.isfinite(center).all():
            raise ValueError("disk_center must contain two finite coordinates")
        x_norm = (x - center[0]) / radius_ref
        y_norm = (y - center[1]) / radius_ref
    else:
        radius_ref = None
        x_norm = np.zeros_like(x)
        y_norm = np.zeros_like(y)

    c = int(
        np.sign(chirality if chirality is not None else metadata.get("chirality", 1))
        or 1
    )
    xi = float(max(xi_shape_factor, 0.0))

    # Average in-plane magnetization induced by vortex-core displacement.
    mx_avg = -float(c) * xi * y_norm
    my_avg = float(c) * xi * x_norm
    mz_avg = polarity

    projection = px * mx_avg + py * my_avg + pz * mz_avg
    return np.clip(np.asarray(projection, dtype=float), -1.0, 1.0)


def compute_magnetoresistance(
    trajectory: TrajectoryResult,
    *,
    polarizer: tuple[float, float, float] | tuple[float, float] = (1.0, 0.0, 0.0),
    resistance_parallel_ohm: float = 100.0,
    delta_resistance_ohm: float = 40.0,
    disk_radius: float | None = None,
    disk_center: tuple[float, float] | None = None,
    chirality: int | None = None,
) -> MagnetoresistanceResult:
    """Compute MR/TMR proxy trace from tracked vortex trajectory."""
    trajectory_metadata = dict(getattr(trajectory, "metadata", {}) or {})
    projection = _projection_from_trajectory(
        trajectory,
        polarizer=polarizer,
        disk_radius=disk_radius,
        disk_center=disk_center,
        chirality=chirality,
    )

    r_p = float(resistance_parallel_ohm)
    d_r = float(delta_resistance_ohm)
    resistance = r_p + 0.5 * d_r * (1.0 - projection)

    return MagnetoresistanceResult(
        time=np.asarray(trajectory.time, dtype=float),
        resistance_ohm=np.asarray(resistance, dtype=float),
        projection=np.asarray(projection, dtype=float),
        method="trajectory_proxy",
        metadata={
            "resistance_parallel_ohm": r_p,
            "delta_resistance_ohm": d_r,
            "polarizer": tuple(float(v) for v in _normalize_polarizer(polarizer)),
            "disk_radius": (
                float(disk_radius)
                if disk_radius is not None and np.isfinite(float(disk_radius))
                else trajectory_metadata.get("disk_radius")
            ),
            "disk_center": (
                tuple(float(value) for value in disk_center)
                if disk_center is not None
                else trajectory_metadata.get("disk_center")
            ),
            "chirality": int(np.sign(chirality) or 1)
            if chirality is not None
            else None,
            "source_method": trajectory.method,
            "interpretation": (
                "uncalibrated trajectory proxy; not a field/contact simulation"
            ),
        },
    )


__all__ = ["compute_magnetoresistance"]
