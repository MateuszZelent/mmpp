"""Shared topology primitives reused by soliton modules."""

from __future__ import annotations

from typing import Any

import numpy as np


def _y_axis_value(convention: Any | None) -> str:
    if convention is None:
        return "down"
    y_axis = getattr(convention, "y_axis", "down")
    return "up" if str(y_axis).lower() == "up" else "down"


def _orient_field(
    m_hat: np.ndarray,
    convention: Any | None,
) -> tuple[np.ndarray, float, bool]:
    """Return field oriented for right-handed XY derivatives."""
    if _y_axis_value(convention) == "up":
        return np.flip(m_hat, axis=0), -1.0, True
    return m_hat, 1.0, False


def normalize_magnetization(m: np.ndarray) -> np.ndarray:
    """Normalize magnetization vectors with safe divide guard."""
    arr = np.asarray(m, dtype=float)
    if arr.ndim < 3 or arr.shape[-1] < 3:
        raise ValueError("Expected magnetization with trailing vector axis of size >= 3")

    arr = arr[..., :3]
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norm, 1e-30, None)


def topological_density_fd(
    m: np.ndarray,
    dx: float,
    dy: float,
    *,
    convention: Any | None = None,
) -> tuple[np.ndarray, float]:
    """Finite-difference topological density and integrated charge."""
    m_hat = normalize_magnetization(m)
    oriented, sign, flipped = _orient_field(m_hat, convention)
    edge_order = 2 if (oriented.shape[0] > 2 and oriented.shape[1] > 2) else 1

    dm_dx = np.gradient(oriented, float(dx), axis=1, edge_order=edge_order)
    dm_dy = np.gradient(oriented, float(dy), axis=0, edge_order=edge_order)

    cross = np.cross(dm_dx, dm_dy)
    q = np.einsum("...i,...i", oriented, cross) / (4.0 * np.pi)
    q = np.asarray(q, dtype=float) * sign
    if flipped:
        q = np.flip(q, axis=0)
    q_total = float(np.sum(q) * float(dx) * float(dy))
    return q, q_total


def _triangle_charge(m1: np.ndarray, m2: np.ndarray, m3: np.ndarray) -> float:
    numerator = float(np.dot(m1, np.cross(m2, m3)))
    denominator = float(1.0 + np.dot(m1, m2) + np.dot(m2, m3) + np.dot(m3, m1))
    omega = 2.0 * np.arctan2(numerator, denominator)
    return float(omega / (4.0 * np.pi))


def berg_luscher_Q(
    m: np.ndarray,
    *,
    convention: Any | None = None,
    return_density: bool = False,
    dx: float = 1.0,
    dy: float = 1.0,
) -> float | tuple[np.ndarray, float]:
    """Berg-Luscher topological charge, optionally with density map."""
    m_hat = normalize_magnetization(m)
    oriented, sign, flipped = _orient_field(m_hat, convention)
    ny, nx, _ = oriented.shape

    integral_map = np.zeros((ny, nx), dtype=float)
    q_total = 0.0

    for iy in range(ny - 1):
        for ix in range(nx - 1):
            m_a = oriented[iy, ix]
            m_b = oriented[iy, ix + 1]
            m_c = oriented[iy + 1, ix + 1]
            m_d = oriented[iy + 1, ix]

            q1 = _triangle_charge(m_a, m_b, m_c)
            q2 = _triangle_charge(m_a, m_c, m_d)
            q_total += q1 + q2

            integral_map[iy, ix] += q1 / 3.0
            integral_map[iy, ix + 1] += q1 / 3.0
            integral_map[iy + 1, ix + 1] += q1 / 3.0

            integral_map[iy, ix] += q2 / 3.0
            integral_map[iy + 1, ix + 1] += q2 / 3.0
            integral_map[iy + 1, ix] += q2 / 3.0

    q_total *= sign
    if not return_density:
        return float(q_total)

    density = integral_map / (float(dx) * float(dy))
    density = np.asarray(density, dtype=float) * sign
    if flipped:
        density = np.flip(density, axis=0)
    return np.asarray(density, dtype=float), float(q_total)


def guiding_center(
    q_density: np.ndarray,
    dx: float,
    dy: float,
    *,
    convention: Any | None = None,
    threshold_fraction: float = 0.0,
) -> tuple[float, float, float]:
    """Compute q-weighted guiding center and ROI confidence."""
    q = np.asarray(q_density, dtype=float)
    if q.ndim != 2 or q.size == 0:
        return 0.0, 0.0, 0.0

    abs_q = np.abs(q)
    if float(np.max(abs_q)) <= 0.0:
        return 0.0, 0.0, 0.0

    if threshold_fraction > 0.0:
        mask = abs_q >= float(threshold_fraction) * float(np.max(abs_q))
        weights = np.where(mask, q, 0.0)
    else:
        weights = q

    total = float(np.sum(weights))
    if abs(total) <= 1e-30:
        return 0.0, 0.0, 0.0

    ny, nx = q.shape
    x_grid = np.arange(nx, dtype=float) * float(dx)
    y_index = np.arange(ny, dtype=float)
    if _y_axis_value(convention) == "up":
        y_grid = (ny - 1 - y_index) * float(dy)
    else:
        y_grid = y_index * float(dy)

    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    x_center = float(np.sum(weights * x_mesh) / total)
    y_center = float(np.sum(weights * y_mesh) / total)

    confidence = float(np.sum(np.abs(weights)) / max(np.sum(abs_q), 1e-30))
    return x_center, y_center, confidence


__all__ = [
    "normalize_magnetization",
    "topological_density_fd",
    "berg_luscher_Q",
    "guiding_center",
]
