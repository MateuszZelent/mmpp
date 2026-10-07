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
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    out = np.empty((n_samples, n_points), dtype=int)
    for sample_idx in range(n_samples):
        idx: list[int] = []
        while len(idx) < n_points:
            start = int(rng.integers(0, n_points))
            stop = min(start + block_size, n_points)
            idx.extend(range(start, stop))
        out[sample_idx, :] = np.asarray(idx[:n_points], dtype=int)
    return out


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth one branch while keeping its original sampling protocol."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 3 or window <= 1:
        return arr.copy()
    width = min(int(window), arr.size)
    if width % 2 == 0:
        width -= 1
    pad = width // 2
    padded = np.pad(arr, pad_width=pad, mode="reflect")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _resample_branch_residuals(
    field: np.ndarray,
    magnetization: np.ndarray,
    *,
    rng: np.random.Generator,
    block_size: int,
) -> np.ndarray:
    """Resample within-branch residuals without reordering field values."""
    sampled = np.asarray(magnetization, dtype=float).copy()
    for branch in segment_branches(field):
        section = branch.slice
        values = sampled[section]
        if values.size < 3:
            continue
        local_block = min(max(int(block_size), 1), values.size)
        baseline = _moving_average(values, window=max(local_block, 3))
        residual = values - baseline
        residual -= float(np.mean(residual))

        indices: list[int] = []
        while len(indices) < values.size:
            start = int(rng.integers(0, values.size))
            indices.extend(
                ((start + offset) % values.size) for offset in range(local_block)
            )
        sampled[section] = baseline + residual[np.asarray(indices[: values.size])]
    return sampled


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
    if n_boot <= 0:
        raise ValueError("n_samples must be positive")
    level = float(ci if ci is not None else result.config.bootstrap_ci)
    if not (0.0 < level < 1.0):
        raise ValueError("ci must be in (0, 1)")

    rng = np.random.default_rng(int(seed))
    blk = int(block_size if block_size is not None else max(10, n_points // 10))
    if blk <= 0:
        raise ValueError("block_size must be positive")
    branches = segment_branches(field)
    if not branches:
        raise ValueError("The field protocol does not contain a resampleable branch")

    estimates: list[float] = []
    for _ in range(n_boot):
        mag_sample = _resample_branch_residuals(
            field,
            mag,
            rng=rng,
            block_size=blk,
        )
        try:
            v, _unit = _estimate_metric(result, metric_name, field, mag_sample)
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
