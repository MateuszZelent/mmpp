"""Orbit fitting routines for vortex core trajectories."""

from __future__ import annotations

import numpy as np

from ..._method_helpers import InteractiveNodeMixin
from ..core.models import TrajectoryResult
from .models import OrbitFitResult


def fit_orbit_ellipse(trajectory: TrajectoryResult) -> OrbitFitResult:
    """Fit an ellipse using second-moment analysis of the tracked orbit."""
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)

    if x.size < 3:
        center = (
            float(np.mean(x)) if x.size else 0.0,
            float(np.mean(y)) if y.size else 0.0,
        )
        return OrbitFitResult(
            center=center,
            semi_major=0.0,
            semi_minor=0.0,
            eccentricity=0.0,
            tilt_angle=0.0,
            residual=0.0,
            metadata={"n_points": int(x.size), "status": "insufficient_points"},
        )

    cx = float(np.mean(x))
    cy = float(np.mean(y))

    coords = np.column_stack((x - cx, y - cy))
    covariance = np.cov(coords, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    lambda_major = max(float(eigenvalues[0]), 0.0)
    lambda_minor = max(float(eigenvalues[1]), 0.0)

    semi_major = float(np.sqrt(max(2.0 * lambda_major, 0.0)))
    semi_minor = float(np.sqrt(max(2.0 * lambda_minor, 0.0)))

    if semi_major < 1e-18:
        eccentricity = 0.0
    else:
        ratio = np.clip((semi_minor / semi_major) ** 2, 0.0, 1.0)
        eccentricity = float(np.sqrt(1.0 - ratio))

    major_vector = eigenvectors[:, 0]
    tilt_angle = float(np.arctan2(major_vector[1], major_vector[0]))

    rotated = coords @ eigenvectors
    u = rotated[:, 0]
    v = rotated[:, 1]

    denom_major = max(semi_major, 1e-18)
    denom_minor = max(semi_minor, 1e-18)
    normalized = (u / denom_major) ** 2 + (v / denom_minor) ** 2
    residual = float(np.mean((normalized - 1.0) ** 2))

    return OrbitFitResult(
        center=(cx, cy),
        semi_major=semi_major,
        semi_minor=semi_minor,
        eccentricity=eccentricity,
        tilt_angle=tilt_angle,
        residual=residual,
        metadata={
            "n_points": int(x.size),
            "covariance": covariance,
        },
    )


class OrbitInterface(InteractiveNodeMixin):
    """Fluent orbit API hanging off :class:`TrajectoryInterface`."""

    _interactive_owner = "job[0].vortex.trajectory.orbit"
    _interactive_nodes = frozenset({"fit"})

    def __init__(self, trajectory_interface):
        self._trajectory_interface = trajectory_interface
        self._fit_cache: OrbitFitResult | None = None

    def fit(self, model: str = "ellipse") -> OrbitFitResult:
        """Fit orbit model to tracked core trajectory."""
        model_norm = model.lower()
        if model_norm != "ellipse":
            raise ValueError("Only model='ellipse' is supported in phase 2")

        result = fit_orbit_ellipse(self._trajectory_interface.raw)
        self._fit_cache = result
        return result

    @property
    def _fit(self) -> OrbitFitResult:
        if self._fit_cache is None:
            self._fit_cache = self.fit()
        return self._fit_cache

    @property
    def radius(self) -> float:
        """Mean orbit radius derived from fitted ellipse."""
        return self._fit.radius

    @property
    def eccentricity(self) -> float:
        """Orbit eccentricity from fitted ellipse."""
        return self._fit.eccentricity

    @property
    def center(self) -> tuple[float, float]:
        """Orbit center coordinates."""
        return self._fit.center
