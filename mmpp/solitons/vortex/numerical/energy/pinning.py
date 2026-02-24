"""Pinning-site detection from effective radial potentials."""

from __future__ import annotations

import numpy as np

from .models import EffectivePotentialResult, PinningResult, PinningSite


def detect_pinning_sites(
    potential: EffectivePotentialResult,
    *,
    min_depth_fraction: float = 0.05,
) -> PinningResult:
    """Detect local minima interpreted as pinning sites."""
    radius = np.asarray(potential.radius_m, dtype=float)
    values = np.asarray(potential.potential_j, dtype=float)
    if radius.size < 3 or values.size < 3:
        return PinningResult(
            potential=potential,
            sites=[],
            metadata={"status": "insufficient_samples"},
        )

    finite = np.isfinite(radius) & np.isfinite(values)
    radius = radius[finite]
    values = values[finite]
    if values.size < 3:
        return PinningResult(
            potential=potential,
            sites=[],
            metadata={"status": "no_finite_values"},
        )

    v_range = float(np.nanmax(values) - np.nanmin(values))
    depth_threshold = float(max(min_depth_fraction, 0.0)) * max(v_range, 1e-30)

    sites: list[PinningSite] = []
    for i in range(1, values.size - 1):
        if not (values[i] < values[i - 1] and values[i] <= values[i + 1]):
            continue

        local_barrier = min(float(values[i - 1]), float(values[i + 1]))
        depth = float(local_barrier - values[i])
        if depth < depth_threshold:
            continue

        confidence = float(np.clip(depth / max(v_range, 1e-30), 0.0, 1.0))
        sites.append(
            PinningSite(
                radius_m=float(radius[i]),
                potential_j=float(values[i]),
                depth_j=depth,
                confidence=confidence,
                metadata={"index": int(i)},
            )
        )

    return PinningResult(
        potential=potential,
        sites=sites,
        metadata={
            "status": "ok",
            "min_depth_fraction": float(min_depth_fraction),
            "n_sites": len(sites),
        },
    )


__all__ = ["detect_pinning_sites"]
