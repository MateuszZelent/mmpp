# ruff: noqa: UP007
"""Deterministic synthetic skyrmion textures for regression tests."""

from __future__ import annotations

from typing import Optional

import numpy as np


def generate_synthetic_skyrmion(
    Nx: int = 128,
    Ny: int = 128,
    *,
    dx: float = 1e-9,
    dy: float = 1e-9,
    radius: float = 24e-9,
    wall_scale: float = 4e-9,
    center: Optional[tuple[float, float]] = None,
    background_polarity: int = 1,
    helicity: float = 0.0,
    model: str = "domain_wall",
    sigma: Optional[float] = None,
    noise: float = 0.0,
    seed: int = 7,
) -> np.ndarray:
    """Return a normalized isolated-skyrmion field with a known radial profile.

    ``center`` is expressed as pixel coordinates ``(x, y)``.  The domain-wall
    model uses the circular 360-degree wall ansatz; the Gaussian model uses a
    contrast Gaussian whose contrast-50 radius is ``sigma * sqrt(2 log(2))``.
    """
    if background_polarity not in (-1, 1):
        raise ValueError("background_polarity must be -1 or +1")
    if center is None:
        center = ((Nx - 1) / 2.0, (Ny - 1) / 2.0)

    x = (np.arange(Nx, dtype=float) - float(center[0])) * float(dx)
    y = (np.arange(Ny, dtype=float) - float(center[1])) * float(dy)
    x_grid, y_grid = np.meshgrid(x, y)
    radial_distance = np.hypot(x_grid, y_grid)
    azimuth = np.arctan2(y_grid, x_grid) + float(helicity)

    selected_model = str(model).lower()
    if selected_model in {"domain_wall", "ansatz"}:
        scale = max(float(wall_scale), np.finfo(float).eps)
        inner = np.clip((radial_distance - float(radius)) / scale, -60.0, 60.0)
        outer = np.clip((radial_distance + float(radius)) / scale, -60.0, 60.0)
        theta = 2.0 * np.arctan(np.exp(inner))
        theta += 2.0 * np.arctan(np.exp(outer))
        mz = float(background_polarity) * np.cos(theta)
    elif selected_model == "gaussian":
        selected_sigma = float(sigma if sigma is not None else radius)
        contrast = np.exp(
            -(radial_distance**2) / (2.0 * max(selected_sigma, 1e-30) ** 2)
        )
        mz = float(background_polarity) * (1.0 - 2.0 * contrast)
    else:
        raise ValueError("model must be 'domain_wall', 'ansatz', or 'gaussian'")

    in_plane = np.sqrt(np.clip(1.0 - mz**2, 0.0, 1.0))
    mx = in_plane * np.cos(azimuth)
    my = in_plane * np.sin(azimuth)
    field = np.stack([mx, my, mz], axis=-1)

    if noise > 0.0:
        rng = np.random.default_rng(int(seed))
        field = field + rng.normal(scale=float(noise), size=field.shape)

    norm = np.linalg.norm(field, axis=-1, keepdims=True)
    return field / np.where(norm > 1e-30, norm, 1.0)


__all__ = ["generate_synthetic_skyrmion"]
