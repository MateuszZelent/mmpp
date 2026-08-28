"""Low-level topological invariant estimators for vortex states."""

from __future__ import annotations

import numpy as np


def polarity(mz_core: float, threshold: float = 0.5) -> int:
    """Estimate polarity from ``m_z`` at the core center."""
    if mz_core >= threshold:
        return 1
    if mz_core <= -threshold:
        return -1
    return 1 if mz_core >= 0.0 else -1


def winding_number(phi_ring: np.ndarray) -> float:
    """Compute winding number from ordered in-plane angles on a closed contour."""
    phi_ring = np.asarray(phi_ring, dtype=float)
    if phi_ring.size < 2:
        return 0.0

    closed = np.concatenate([phi_ring, phi_ring[:1]])
    unwrapped = np.unwrap(closed)
    total_rotation: float = float(np.sum(np.diff(unwrapped)))
    return float(total_rotation / (2.0 * np.pi))


def topological_charge(topological_density: np.ndarray, dx: float, dy: float) -> float:
    """Integrate topological density map to total charge ``Q``."""
    q = np.asarray(topological_density, dtype=float)
    return float(np.sum(q) * float(dx) * float(dy))


def chirality_ring(
    m_xy: np.ndarray,
    core_pos: tuple[float, float],
    ring: tuple[float, float],
    *,
    dx: float = 1.0,
    dy: float = 1.0,
) -> int:
    """Estimate chirality from the ring-averaged azimuthal magnetization component."""
    if m_xy.ndim != 3 or m_xy.shape[-1] != 2:
        raise ValueError("m_xy must have shape (Ny, Nx, 2)")

    ny, nx, _ = m_xy.shape
    x_coords = np.arange(nx, dtype=float) * dx
    y_coords = np.arange(ny, dtype=float) * dy
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    cx, cy = core_pos
    rx = x_grid - cx
    ry = y_grid - cy

    radius = np.hypot(rx, ry)
    r_min, r_max = ring
    mask = (radius >= r_min) & (radius <= r_max)
    if not np.any(mask):
        return 0

    phi = np.arctan2(ry, rx)
    phi_hat_x = -np.sin(phi)
    phi_hat_y = np.cos(phi)

    m_phi = m_xy[..., 0] * phi_hat_x + m_xy[..., 1] * phi_hat_y
    value = float(np.mean(m_phi[mask]))
    if value == 0.0:
        return 0
    return 1 if value > 0.0 else -1


def chirality_ring_with_confidence(
    m_xy: np.ndarray,
    core_pos: tuple[float, float],
    ring: tuple[float, float],
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    y_axis: str = "down",
) -> tuple[int, float]:
    """Estimate chirality and confidence on annulus around the core."""
    if m_xy.ndim != 3 or m_xy.shape[-1] != 2:
        raise ValueError("m_xy must have shape (Ny, Nx, 2)")

    ny, nx, _ = m_xy.shape
    x_coords = np.arange(nx, dtype=float) * dx
    y_idx = np.arange(ny, dtype=float)
    if str(y_axis).lower() == "up":
        y_coords = (ny - 1 - y_idx) * dy
    else:
        y_coords = y_idx * dy
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    cx, cy = core_pos
    rx = x_grid - cx
    ry = y_grid - cy

    radius = np.hypot(rx, ry)
    r_min, r_max = ring
    mask = (radius >= r_min) & (radius <= r_max)
    if not np.any(mask):
        return 0, 0.0

    phi = np.arctan2(ry, rx)
    phi_hat_x = -np.sin(phi)
    phi_hat_y = np.cos(phi)
    m_phi = m_xy[..., 0] * phi_hat_x + m_xy[..., 1] * phi_hat_y

    values = m_phi[mask]
    mean_m_phi = float(np.mean(values))
    mean_abs = float(np.mean(np.abs(values)))
    if mean_abs <= 1e-30:
        return 0, 0.0

    confidence = float(np.clip(abs(mean_m_phi) / mean_abs, 0.0, 1.0))
    if mean_m_phi == 0.0:
        return 0, confidence
    return (1 if mean_m_phi > 0.0 else -1), confidence


def polarity_from_core(mz_core: float, threshold: float = 0.5) -> int:
    """Alias retained for planned API naming."""
    return polarity(mz_core=mz_core, threshold=threshold)


def chirality_from_ring(*args, with_confidence: bool = False, **kwargs):
    """Alias retained for planned API naming."""
    if with_confidence:
        return chirality_ring_with_confidence(*args, **kwargs)
    return chirality_ring(*args, **kwargs)


def winding_number_from_ring(phi_ring):
    """Alias retained for planned API naming."""
    return winding_number(phi_ring)


def classify_state(polarity_value: int, vorticity_value: int, q_value: float) -> str:
    """Minimal state classifier used by scaffolding tests/docs."""
    p = int(polarity_value)
    w = int(vorticity_value)
    q = float(q_value)

    if abs(abs(q) - 1.0) < 0.25:
        return "skyrmion"
    if abs(q - p * w * 0.5) < 0.25:
        return "vortex" if w > 0 else "antivortex"
    return "unknown"


__all__ = [
    "polarity",
    "topological_charge",
    "winding_number",
    "chirality_ring",
    "chirality_ring_with_confidence",
    "polarity_from_core",
    "chirality_from_ring",
    "winding_number_from_ring",
    "classify_state",
]
