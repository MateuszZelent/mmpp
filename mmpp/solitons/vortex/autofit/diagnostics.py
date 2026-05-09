"""Autofit guard penalties and success diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .features import TrajectoryFeatures


def cpp_linear_threshold_metrics_from_params(
    params: dict[str, Any],
) -> dict[str, float] | None:
    """Compute linear CPP threshold metrics for the current parameter set."""
    required = {"Ms", "alpha", "A", "R", "current_density", "omega0"}
    if not required.issubset(params):
        return None

    J = float(params.get("current_density", 0.0) or 0.0)
    omega0 = float(params.get("omega0", 0.0) or 0.0)
    if not np.isfinite(J) or not np.isfinite(omega0) or omega0 <= 0.0:
        return None

    _HBAR = 1.054571817e-34
    _E_CHARGE = 1.602176634e-19
    _GAMMA_E = 1.76085963023e11
    MU0 = 4e-7 * math.pi

    Ms = float(params["Ms"])
    alpha = float(params["alpha"])
    P_model = float(params.get("P_model", params.get("P", 0.0)))
    A = float(params.get("A", 1.3e-11))
    R = float(params["R"])
    L = float(params.get("L_stt", params.get("L", 0.0)))
    chi_scale = float(params.get("chi_scale", 1.0))
    domega0_dJ = float(params.get("domega0_dJ", 0.0))
    d0_scale = float(params.get("d0_scale", 1.0))
    polarity = int(np.sign(float(params.get("polarity", 1))) or 1)

    if Ms <= 0.0 or R <= 0.0 or L <= 0.0:
        return None

    sigma_per_p = _HBAR / (2.0 * _E_CHARGE * L * Ms)
    chi_prefactor_per_p = _GAMMA_E * sigma_per_p / 2.0
    chi = chi_scale * (-float(polarity)) * chi_prefactor_per_p * P_model * J

    lex = math.sqrt(2.0 * A / (MU0 * Ms * Ms))
    Rc = max(lex, 1e-10)
    ratio = R / max(Rc, 1e-10)
    d0 = alpha * (5.0 + 4.0 * math.log(max(ratio, 1.1))) / 8.0
    omega0_eff = omega0 + domega0_dJ * J
    threshold = d0 * d0_scale * omega0_eff
    chi_ratio = chi / max(threshold, 1e-30)

    return {
        "chi": float(chi),
        "threshold": float(threshold),
        "chi_ratio": float(chi_ratio),
        "omega0_eff": float(omega0_eff),
        "P_model": float(P_model),
    }


def cpp_threshold_guard_penalty(
    features_num: TrajectoryFeatures,
    features_ana: TrajectoryFeatures,
    params: dict[str, Any],
) -> float:
    """Penalise sub-threshold analytical candidates for stable numerical orbits."""
    if (
        float(features_num.mean_radius) <= 0.0
        or float(features_num.radius_drift_ratio) < 0.85
    ):
        return 0.0

    metrics = cpp_linear_threshold_metrics_from_params(params)
    if metrics is None:
        return 0.0

    ratio = float(metrics["chi_ratio"])
    penalty = 0.0
    if ratio <= 0.0:
        penalty += 6.0 + min(abs(ratio), 4.0)
    elif ratio < 1.0:
        penalty += 4.0 * ((1.0 - ratio) / 0.25) ** 2

    tail_target = max(float(features_num.tail_mean_radius), 1e-30)
    tail_ratio = float(features_ana.tail_mean_radius) / tail_target
    if ratio < 1.05 and tail_ratio < 0.8:
        penalty += ((0.8 - tail_ratio) / 0.8) ** 2 * 2.5

    return float(penalty)


def collapse_guard_penalty(
    features_num: TrajectoryFeatures,
    features_ana: TrajectoryFeatures,
) -> float:
    target_radius = float(features_num.mean_radius)
    if target_radius <= 0.0:
        return 0.0
    if float(features_num.radius_drift_ratio) < 0.85:
        return 0.0

    ana_radius_ratio = float(features_ana.mean_radius) / target_radius
    ana_drift = float(features_ana.radius_drift_ratio)
    penalty = 0.0
    if ana_radius_ratio < 0.7:
        penalty += ((0.7 - ana_radius_ratio) / 0.7) ** 2 * 3.0
    if ana_drift < 0.8:
        penalty += ((0.8 - ana_drift) / 0.8) ** 2 * 2.0
    return float(penalty)


def frequency_guard_penalty(
    features_num: TrajectoryFeatures,
    features_ana: TrajectoryFeatures,
) -> float:
    """Strongly penalise near-static analytical candidates for stable targets."""
    f_num = float(features_num.dominant_freq_hz)
    if f_num <= 0.0 or float(features_num.mean_radius) <= 0.0:
        return 0.0
    if float(features_num.radius_drift_ratio) < 0.85:
        return 0.0

    f_ana = max(float(features_ana.dominant_freq_hz), 0.0)
    freq_ratio = f_ana / max(f_num, 1e-30)
    radius_ratio = float(features_ana.mean_radius) / max(
        float(features_num.mean_radius), 1e-30
    )
    drift_ratio = float(features_ana.radius_drift_ratio)

    penalty = 0.0
    if freq_ratio < 0.5:
        penalty += ((0.5 - freq_ratio) / 0.5) ** 2 * 6.0
    if freq_ratio < 0.25 and radius_ratio > 0.25:
        penalty += ((0.25 - freq_ratio) / 0.25) ** 2 * 10.0
    if freq_ratio < 0.15 and drift_ratio > 0.2:
        penalty += 12.0
    return float(penalty)


def edge_collision_guard_penalty(
    features_num: TrajectoryFeatures,
    features_ana: TrajectoryFeatures,
    *,
    ana_trajectory,
    reference_radius: float,
) -> float:
    """Penalise analytical edge hits when the numerical orbit stays well inside the disk."""
    R = float(reference_radius)
    if not np.isfinite(R) or R <= 0.0:
        return 0.0

    ana_edge = bool(getattr(ana_trajectory, "metadata", {}).get("edge_limited", False))
    num_max = float(features_num.max_core_distance)
    ana_max = float(features_ana.max_core_distance)

    if num_max >= 0.95 * R:
        return 0.0

    penalty = 0.0
    if ana_edge:
        penalty += 8.0 * ((0.95 * R - num_max) / max(R, 1e-30)) ** 2

    if ana_max > 0.98 * R and num_max < 0.9 * R:
        excess = (ana_max - 0.98 * R) / max(R, 1e-30)
        margin = (0.9 * R - num_max) / max(R, 1e-30)
        penalty += 6.0 * max(excess, 0.0) ** 2 + 3.0 * max(margin, 0.0) ** 2

    return float(penalty)


def assess_fit_success(
    *,
    baseline_loss: float,
    final_loss: float,
    comparison,
    diagnostics,
    features_num: TrajectoryFeatures,
    cpp_metrics: dict[str, float] | None,
) -> tuple[bool, list[str]]:
    """Decide whether the fit is physically acceptable, not just lower-loss."""
    failures: list[str] = []

    if diagnostics.n_evaluations <= 0:
        failures.append("Fit did not run any evaluations.")
    if final_loss >= baseline_loss:
        failures.append(
            f"Fit did not improve loss (baseline={baseline_loss:.4g}, fitted={final_loss:.4g})."
        )

    m = comparison.metrics
    stable_target = (
        float(features_num.mean_radius) > 0.0
        and float(features_num.radius_drift_ratio) >= 0.85
    )
    if stable_target:
        num_freq = max(float(m.numerical_freq_ghz), 1e-12)
        num_orbit = max(float(m.numerical_radius_nm), 1e-9)
        freq_rel = abs(float(m.delta_freq_mean) * 1e-9) / num_freq
        orbit_rel = abs(float(m.delta_radius_mean) * 1e9) / num_orbit
        ana_freq_ratio = float(m.analytical_freq_ghz) / num_freq

        if freq_rel > 0.25:
            failures.append(
                f"Frequency mismatch too large for a stable target (df/f={freq_rel:.2f})."
            )
        if orbit_rel > 0.35:
            failures.append(
                f"Orbit-radius mismatch too large for a stable target (dr/r={orbit_rel:.2f})."
            )
        if ana_freq_ratio < 0.2 and float(m.analytical_radius_nm) > 0.2 * num_orbit:
            failures.append(
                "Analytical solution has substantial radius but nearly zero gyrotropic frequency."
            )
        ana_edge = bool(
            getattr(comparison.analytical, "metadata", {}).get("edge_limited", False)
        )
        numerical_inside = (
            float(m.numerical_core_distance_max_nm)
            < 0.95 * float(comparison.resolved_params.get("R", 0.0)) * 1e9
        )
        if ana_edge and numerical_inside:
            failures.append(
                "Analytical solution collides with the nanodot edge while numerical orbit remains inside the disk."
            )
        if cpp_metrics is not None:
            chi_ratio = float(cpp_metrics.get("chi_ratio", np.nan))
            if np.isfinite(chi_ratio) and chi_ratio < 0.9:
                failures.append(
                    f"Best-fit CPP state remains sub-threshold or near-threshold (J/Jth={chi_ratio:.2f})."
                )

    return len(failures) == 0, failures


__all__ = [
    "assess_fit_success",
    "collapse_guard_penalty",
    "cpp_linear_threshold_metrics_from_params",
    "cpp_threshold_guard_penalty",
    "edge_collision_guard_penalty",
    "frequency_guard_penalty",
]
