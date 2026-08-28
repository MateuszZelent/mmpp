"""Bootstrap uncertainty estimates for hysteresis metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..compute import segment_branches
from .core import (
    compute_coercive_field,
    compute_loop_area,
    compute_max_susceptibility,
    compute_remanence,
    compute_saturation_points,
    compute_squareness,
)


@dataclass
class ConfidenceIntervalResult:
    """Confidence interval container."""

    metric: str
    value: float
    low: float
    high: float
    half_width: float
    level: float
    unit: str = "input"


def _estimate_metric(
    result, metric_name: str, field: np.ndarray, mag: np.ndarray
) -> tuple[float, str]:
    name = str(metric_name).lower()
    branches = segment_branches(field)
    field_unit = str(result.metadata.get("field_unit", "input"))
    sat = compute_saturation_points(
        field,
        mag,
        threshold=result.config.saturation_threshold,
        window=result.config.saturation_window,
    )

    metric: Any
    if name in {"coercive_field", "hc"}:
        metric = compute_coercive_field(field, mag, branches, unit=field_unit)
        return float(metric.mean), field_unit
    if name in {"remanence", "mr"}:
        metric = compute_remanence(field, mag, branches)
        return float(metric.mean), "a.u."
    if name in {"saturation_points", "ms"}:
        return float(sat.ms_mean), "a.u."
    if name in {"loop_area", "area"}:
        return float(compute_loop_area(field, mag)), f"a.u.*{field_unit}"
    if name in {"squareness", "s"}:
        rem = compute_remanence(field, mag, branches)
        return float(compute_squareness(rem, sat)), "ratio"
    if name in {"max_susceptibility", "chi_max", "susceptibility"}:
        chi = compute_max_susceptibility(field, mag)
        return float(chi.chi_max), "a.u."
    raise ValueError(f"Unsupported metric for CI: {metric_name}")


def _block_bootstrap_indices(
    n_points: int,
    n_samples: int,
    rng: np.random.Generator,
    block_size: int,
) -> np.ndarray:
    out = np.empty((n_samples, n_points), dtype=int)
    for sample_idx in range(n_samples):
        idx: list[int] = []
        while len(idx) < n_points:
            start = int(rng.integers(0, n_points))
            stop = min(start + block_size, n_points)
            idx.extend(range(start, stop))
        out[sample_idx, :] = np.asarray(idx[:n_points], dtype=int)
    return out


def bootstrap_confidence_interval(
    result,
    *,
    metric_name: str,
    n_samples: int | None = None,
    ci: float | None = None,
    seed: int = 123,
    block_size: int | None = None,
) -> ConfidenceIntervalResult:
    """Estimate confidence interval via block bootstrap."""
    field = np.asarray(result.field, dtype=float)
    mag = np.asarray(result.metrics._processed_magnetization(), dtype=float)
    n_points = int(field.size)
    if n_points < 20:
        raise ValueError("Need at least 20 points for bootstrap confidence intervals")

    n_boot = int(
        n_samples if n_samples is not None else result.config.bootstrap_n_samples
    )
    level = float(ci if ci is not None else result.config.bootstrap_ci)
    if not (0.0 < level < 1.0):
        raise ValueError("ci must be in (0, 1)")

    rng = np.random.default_rng(int(seed))
    blk = int(block_size if block_size is not None else max(10, n_points // 10))
    bootstrap_idx = _block_bootstrap_indices(n_points, n_boot, rng, blk)

    estimates: list[float] = []
    for idx in bootstrap_idx:
        try:
            v, _unit = _estimate_metric(result, metric_name, field[idx], mag[idx])
        except Exception:
            continue
        if np.isfinite(v):
            estimates.append(float(v))

    if len(estimates) < max(30, n_boot // 5):
        raise ValueError(
            "Bootstrap failed: too few valid samples. "
            "Try lowering n_samples or increasing block_size."
        )

    est = np.asarray(estimates, dtype=float)
    alpha = (1.0 - level) / 2.0
    low = float(np.quantile(est, alpha))
    high = float(np.quantile(est, 1.0 - alpha))
    value, unit = _estimate_metric(result, metric_name, field, mag)
    half_width = float((high - low) / 2.0)

    return ConfidenceIntervalResult(
        metric=str(metric_name),
        value=float(value),
        low=low,
        high=high,
        half_width=half_width,
        level=level,
        unit=unit,
    )


__all__ = ["ConfidenceIntervalResult", "bootstrap_confidence_interval"]
