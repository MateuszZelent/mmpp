from __future__ import annotations

import numpy as np


def generate_synthetic_vortex(
    Nx: int = 64,
    Ny: int = 64,
    *,
    p: int = 1,
    w: int = 1,
    core_radius_px: float = 4.0,
    center_pix: tuple[float, float] | None = None,
) -> np.ndarray:
    """Generate normalized ``(Ny, Nx, 3)`` vortex-like magnetization."""
    if p not in (-1, 1):
        raise ValueError("p must be -1 or +1")
    if w not in (-1, 1):
        raise ValueError("w must be -1 or +1")

    if center_pix is None:
        cx = (Nx - 1) / 2.0
        cy = (Ny - 1) / 2.0
    else:
        cx = float(center_pix[0])
        cy = float(center_pix[1])

    x = np.arange(Nx, dtype=float) - cx
    y = np.arange(Ny, dtype=float) - cy
    x_grid, y_grid = np.meshgrid(x, y)

    radius = np.hypot(x_grid, y_grid)
    phi = np.arctan2(y_grid, x_grid)

    mz = float(p) * np.exp(-(radius / float(core_radius_px)) ** 2)
    m_perp = np.sqrt(np.clip(1.0 - mz**2, 0.0, 1.0))

    # ``w=+1`` yields vortex-like in-plane circulation; ``w=-1`` flips it.
    mx = -float(w) * m_perp * np.sin(phi)
    my = float(w) * m_perp * np.cos(phi)

    m = np.stack([mx, my, mz], axis=-1)
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norm > 1e-12, norm, 1.0)


def generate_vortex_mz_near_edge(
    Nx: int = 64,
    Ny: int = 64,
    *,
    core_pix: tuple[float, float] = (60.0, 32.0),
    core_radius_px: float = 4.0,
) -> np.ndarray:
    """Generate vortex magnetization with core intentionally near sample edge."""
    return generate_synthetic_vortex(
        Nx=Nx,
        Ny=Ny,
        p=1,
        w=1,
        core_radius_px=core_radius_px,
        center_pix=core_pix,
    )


def generate_vortex_mz_centered(
    Nx: int = 64,
    Ny: int = 64,
    *,
    core_pix: tuple[float, float] = (32.0, 32.0),
    core_radius_px: float = 4.0,
) -> np.ndarray:
    """Generate centered vortex magnetization for reference confidence checks."""
    return generate_synthetic_vortex(
        Nx=Nx,
        Ny=Ny,
        p=1,
        w=1,
        core_radius_px=core_radius_px,
        center_pix=core_pix,
    )

