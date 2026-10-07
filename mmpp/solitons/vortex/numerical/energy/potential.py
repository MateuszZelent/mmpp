"""Effective-potential reconstruction for vortex radial dynamics."""

from __future__ import annotations

import numpy as np

from ..._shared.models import TrajectoryResult
from .models import EffectivePotentialResult

_K_B = 1.380649e-23


def _radius_series(trajectory: TrajectoryResult) -> np.ndarray:
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)
    x0 = float(np.mean(x)) if x.size else 0.0
    y0 = float(np.mean(y)) if y.size else 0.0
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)


def _hist_probability(radius: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(radius, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return np.array([], dtype=float), np.array([], dtype=float)

    hist, edges = np.histogram(values, bins=max(int(bins), 8), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prob = hist.astype(float)
    total = float(np.sum(prob))
    if total > 0.0:
        prob /= total
    return np.asarray(centers, dtype=float), np.asarray(prob, dtype=float)


def potential_from_boltzmann(
    trajectory: TrajectoryResult,
    *,
    temperature_k: float,
    bins: int = 64,
    assume_equilibrium: bool = False,
) -> EffectivePotentialResult:
    """Estimate radial PMF from equilibrium occupancy, correcting the 2D measure.

    For an axisymmetric potential ``U(r)``, the radial histogram includes the
    annular measure ``2*pi*r``. This returns ``-kBT*ln(P(r)/(2*pi*r))`` up to an
    additive constant; it is not a spatially resolved pinning potential.
    """
    if not isinstance(assume_equilibrium, (bool, np.bool_)) or not assume_equilibrium:
        raise ValueError(
            "Radial Boltzmann inversion requires explicit assume_equilibrium=True"
        )
    temperature_k = float(temperature_k)
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature_k must be finite and positive")
    radius = _radius_series(trajectory)
    centers, prob = _hist_probability(radius, bins=bins)
    if centers.size == 0:
        return EffectivePotentialResult(
            radius_m=np.array([], dtype=float),
            potential_j=np.array([], dtype=float),
            probability=np.array([], dtype=float),
            method="radial_pmf",
            metadata={"status": "insufficient_samples"},
        )

    shell_measure = 2.0 * np.pi * np.maximum(centers, np.finfo(float).tiny)
    density_per_area = prob / shell_measure
    p_safe = np.clip(density_per_area, 1e-300, None)
    potential = -_K_B * temperature_k * np.log(p_safe)
    potential = potential - float(np.min(potential))

    return EffectivePotentialResult(
        radius_m=np.asarray(centers, dtype=float),
        potential_j=np.asarray(potential, dtype=float),
        probability=np.asarray(prob, dtype=float),
        method="radial_pmf",
        metadata={
            "temperature_k": temperature_k,
            "bins": int(max(bins, 8)),
            "n_samples": int(radius.size),
            "potential_kind": "radial_free_energy",
            "equilibrium_assumed": True,
            "radial_symmetry_assumed": True,
            "measure_correction": "divide radial occupancy by 2*pi*r",
            "spatial_pinning_localization": False,
        },
    )


def potential_from_energy_channel(
    trajectory: TrajectoryResult,
    energy_total: np.ndarray,
    *,
    bins: int = 64,
) -> EffectivePotentialResult:
    """Estimate effective potential as binned conditional mean of ``E_total(r)``."""
    radius = _radius_series(trajectory)
    energy = np.asarray(energy_total, dtype=float).reshape(-1)
    if radius.size != energy.size:
        raise ValueError(
            f"trajectory length mismatch: radius={radius.size}, energy={energy.size}"
        )

    mask = np.isfinite(radius) & np.isfinite(energy)
    radius = radius[mask]
    energy = energy[mask]
    if radius.size < 4:
        return EffectivePotentialResult(
            radius_m=np.array([], dtype=float),
            potential_j=np.array([], dtype=float),
            probability=np.array([], dtype=float),
            method="energy_bin",
            metadata={"status": "insufficient_samples"},
        )

    edges = np.linspace(
        float(np.min(radius)), float(np.max(radius)), max(int(bins), 8) + 1
    )
    idx = np.clip(np.digitize(radius, edges) - 1, 0, edges.size - 2)
    centers = 0.5 * (edges[:-1] + edges[1:])

    sums = np.bincount(idx, weights=energy, minlength=centers.size)
    counts = np.bincount(idx, minlength=centers.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_energy = sums / np.clip(counts, 1, None)
    mean_energy[counts == 0] = np.nan

    prob_counts = counts.astype(float)
    total = float(np.sum(prob_counts))
    prob = prob_counts / total if total > 0 else np.zeros_like(prob_counts)

    valid = np.isfinite(mean_energy)
    centers = centers[valid]
    mean_energy = mean_energy[valid]
    prob = prob[valid]

    if mean_energy.size:
        mean_energy = mean_energy - float(np.min(mean_energy))

    return EffectivePotentialResult(
        radius_m=np.asarray(centers, dtype=float),
        potential_j=np.asarray(mean_energy, dtype=float),
        probability=np.asarray(prob, dtype=float),
        method="energy_bin",
        metadata={
            "bins": int(max(bins, 8)),
            "n_samples": int(radius.size),
            "potential_kind": "conditional_radial_energy_profile",
            "spatial_pinning_localization": False,
        },
    )


__all__ = ["potential_from_boltzmann", "potential_from_energy_channel"]
