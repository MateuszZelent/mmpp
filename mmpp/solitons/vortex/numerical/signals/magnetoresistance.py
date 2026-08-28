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


def _estimate_disk_radius(
    x: np.ndarray,
    y: np.ndarray,
    *,
    disk_radius: float | None,
    metadata: dict,
) -> float:
    if (
        disk_radius is not None
        and np.isfinite(float(disk_radius))
        and float(disk_radius) > 0.0
    ):
        return float(disk_radius)

    meta_r = metadata.get("disk_radius", None)
    if meta_r is not None:
        try:
            value = float(meta_r)
            if np.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass

    x0 = float(np.mean(x)) if x.size else 0.0
    y0 = float(np.mean(y)) if y.size else 0.0
    radius = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    guess = float(np.percentile(radius, 95)) * 1.1 if radius.size else 1.0e-9
    return max(guess, 1e-12)


def _projection_from_trajectory(
    trajectory: TrajectoryResult,
    *,
    polarizer: tuple[float, float, float] | tuple[float, float],
    disk_radius: float | None,
    chirality: int | None,
    xi_shape_factor: float = 2.0 / 3.0,
) -> np.ndarray:
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)
    metadata = dict(getattr(trajectory, "metadata", {}) or {})

    x0 = float(np.mean(x)) if x.size else 0.0
    y0 = float(np.mean(y)) if y.size else 0.0
    radius_ref = _estimate_disk_radius(
        x,
        y,
        disk_radius=disk_radius,
        metadata=metadata,
    )

    x_norm = (x - x0) / radius_ref
    y_norm = (y - y0) / radius_ref

    c = int(
        np.sign(chirality if chirality is not None else metadata.get("chirality", 1))
        or 1
    )
    xi = float(max(xi_shape_factor, 0.0))

    # Average in-plane magnetization induced by vortex-core displacement.
    mx_avg = -float(c) * xi * y_norm
    my_avg = float(c) * xi * x_norm
    mz_avg = np.asarray(trajectory.polarity, dtype=float)

    px, py, pz = _normalize_polarizer(polarizer)
    projection = px * mx_avg + py * my_avg + pz * mz_avg
    return np.clip(np.asarray(projection, dtype=float), -1.0, 1.0)


def compute_magnetoresistance(
    trajectory: TrajectoryResult,
    *,
    polarizer: tuple[float, float, float] | tuple[float, float] = (1.0, 0.0, 0.0),
    resistance_parallel_ohm: float = 100.0,
    delta_resistance_ohm: float = 40.0,
    disk_radius: float | None = None,
    chirality: int | None = None,
) -> MagnetoresistanceResult:
    """Compute MR/TMR proxy trace from tracked vortex trajectory."""
    projection = _projection_from_trajectory(
        trajectory,
        polarizer=polarizer,
        disk_radius=disk_radius,
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
                else None
            ),
            "chirality": int(np.sign(chirality) or 1)
            if chirality is not None
            else None,
            "source_method": trajectory.method,
        },
    )


__all__ = ["compute_magnetoresistance"]
