"""Seed-candidate helpers for vortex autofit warm starts."""

from __future__ import annotations

from typing import Any

import numpy as np

from .diagnostics import cpp_linear_threshold_metrics_from_params
from .features import TrajectoryFeatures


def _clip_candidate_to_specs(
    params: dict[str, float],
    active_names: list[str],
    param_specs: dict[str, Any],
) -> None:
    """Clamp seed candidate values to declared optimisation bounds."""
    for name in active_names:
        spec = param_specs.get(name)
        if spec is None or name not in params:
            continue
        params[name] = float(np.clip(float(params[name]), spec.lower, spec.upper))


def build_cpp_threshold_seed_candidates(
    features_num: TrajectoryFeatures,
    base_params: dict[str, Any],
    initial_params: dict[str, float],
    active_names: list[str],
    param_specs: dict[str, Any],
) -> list[dict[str, float]]:
    """Build a small pool of threshold-aware warm starts for CPP fits."""
    if not active_names:
        return [dict(initial_params)]

    candidates: list[dict[str, float]] = [dict(initial_params)]
    stable_target = (
        float(features_num.mean_radius) > 0.0
        and float(features_num.radius_drift_ratio) >= 0.85
    )
    if not stable_target:
        return candidates

    full_initial = dict(base_params)
    full_initial.update(initial_params)
    metrics = cpp_linear_threshold_metrics_from_params(full_initial)
    if metrics is None:
        return candidates

    ratio = float(metrics["chi_ratio"])
    R = max(float(full_initial.get("R", 0.0) or 0.0), 1e-30)
    target_u = np.clip(float(features_num.tail_mean_radius) / R, 1e-6, 0.98)
    margin = max(1.05, 1.0 + 0.35 * target_u)
    threshold_params = [
        name for name in ("chi_scale", "P_model", "d0_scale") if name in active_names
    ]
    if not threshold_params:
        return candidates

    candidate = dict(initial_params)
    n_scalers = max(len(threshold_params), 1)

    if ratio <= 0.0:
        if "P_model" in active_names:
            desired_sign = np.sign(
                -float(full_initial.get("polarity", 1))
                * float(full_initial.get("current_density", 0.0))
            )
            if desired_sign == 0.0:
                desired_sign = 1.0
            current = float(
                candidate.get(
                    "P_model", full_initial.get("P_model", full_initial.get("P", 0.0))
                )
            )
            candidate["P_model"] = abs(current) * float(desired_sign)
            candidates.append(candidate)
        return unique_seed_candidates(candidates)

    required_scale = max(margin / max(ratio, 1e-12), 1.0)
    per_param_scale = required_scale ** (1.0 / n_scalers)

    if "chi_scale" in active_names:
        current = float(candidate.get("chi_scale", full_initial.get("chi_scale", 1.0)))
        candidate["chi_scale"] = current * per_param_scale
    if "P_model" in active_names:
        current = float(
            candidate.get(
                "P_model", full_initial.get("P_model", full_initial.get("P", 0.0))
            )
        )
        candidate["P_model"] = current * per_param_scale
    if "d0_scale" in active_names:
        current = float(candidate.get("d0_scale", full_initial.get("d0_scale", 1.0)))
        candidate["d0_scale"] = current / per_param_scale

    for factor in (0.9, 1.0, 1.1, 1.25):
        variant = dict(candidate)
        if "chi_scale" in active_names and "chi_scale" in candidate:
            variant["chi_scale"] = candidate["chi_scale"] * factor
        if "P_model" in active_names and "P_model" in candidate:
            variant["P_model"] = candidate["P_model"] * factor
        if "d0_scale" in active_names and "d0_scale" in candidate:
            variant["d0_scale"] = candidate["d0_scale"] / factor
        _clip_candidate_to_specs(variant, active_names, param_specs)
        candidates.append(variant)

    return unique_seed_candidates(candidates)


def unique_seed_candidates(
    candidates: list[dict[str, float]],
) -> list[dict[str, float]]:
    unique: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for candidate in candidates:
        key = tuple(
            sorted((name, round(float(value), 12)) for name, value in candidate.items())
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(candidate))
    return unique


def select_threshold_aware_seed(
    *,
    features_num: TrajectoryFeatures,
    base_params: dict[str, Any],
    initial_params: dict[str, float],
    active_names: list[str],
    param_specs: dict[str, Any],
    evaluator,
) -> tuple[dict[str, float], float]:
    """Pick the best warm start from a small threshold-aware candidate pool."""
    best_params = dict(initial_params)
    best_loss = float("inf")
    for candidate in build_cpp_threshold_seed_candidates(
        features_num,
        base_params,
        initial_params,
        active_names,
        param_specs,
    ):
        loss, _ = evaluator(candidate)
        if loss < best_loss:
            best_loss = float(loss)
            best_params = dict(candidate)
    return best_params, best_loss


__all__ = [
    "build_cpp_threshold_seed_candidates",
    "select_threshold_aware_seed",
    "unique_seed_candidates",
]
