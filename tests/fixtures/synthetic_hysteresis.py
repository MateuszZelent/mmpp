"""Reusable synthetic hysteresis generators for stage-2/3 tests."""

from __future__ import annotations

import numpy as np


def make_synthetic_loop(
    *,
    hc: float = 0.10,
    mr: float = 0.80,
    ms: float = 1.0,
    n_points_half: int = 120,
    noise_std: float = 0.0,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic major loop with optional additive Gaussian noise."""
    n_half = max(int(n_points_half), 8)
    field_half = np.linspace(-0.5, 0.5, n_half, dtype=float)

    m_asc = float(ms) * np.tanh((field_half - float(hc)) / 0.03)
    m_desc = float(ms) * np.tanh((field_half + float(hc)) / 0.03)

    field = np.concatenate([field_half, field_half[::-1]])
    magnetization = np.concatenate([m_asc, m_desc[::-1]])

    ref = np.mean(np.abs([m_asc[n_half // 2], m_desc[n_half // 2]]))
    scale = float(mr) / max(float(ref), 1e-12)
    magnetization = magnetization * scale

    if float(noise_std) > 0.0:
        rng = np.random.default_rng(int(seed))
        magnetization = magnetization + rng.normal(0.0, float(noise_std), size=magnetization.size)

    return np.asarray(field, dtype=float), np.asarray(magnetization, dtype=float)


__all__ = ["make_synthetic_loop"]
