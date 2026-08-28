"""Adapters between analytical model outputs and canonical trajectory contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._shared.models import TrajectoryResult


def thiele_to_trajectory_result(
    analytical_result,
    *,
    method: str,
    polarity: int,
    metadata: dict[str, Any] | None = None,
) -> TrajectoryResult:
    """Convert ``ThieleTrajectoryResult`` to canonical ``TrajectoryResult``."""
    time = np.asarray(analytical_result.t, dtype=float)
    x = np.asarray(analytical_result.x, dtype=float)
    y = np.asarray(analytical_result.y, dtype=float)

    n = int(time.size)
    polarity_series = np.full(n, 1 if int(polarity) >= 0 else -1, dtype=int)
    confidence = np.ones(n, dtype=float)

    meta = dict(metadata or {})
    meta.setdefault("source", "analytical")
    meta.setdefault("model_name", getattr(analytical_result, "model_name", "Thiele"))

    result_metadata = getattr(analytical_result, "metadata", None)
    if isinstance(result_metadata, dict):
        for key, value in result_metadata.items():
            meta.setdefault(key, value)

    params = getattr(analytical_result, "params", None)
    if isinstance(params, dict):
        meta.setdefault("model_params", params)

    return TrajectoryResult(
        time=time,
        x=x,
        y=y,
        polarity=polarity_series,
        method=str(method),
        confidence=confidence,
        metadata=meta,
    )


__all__ = ["thiele_to_trajectory_result"]
