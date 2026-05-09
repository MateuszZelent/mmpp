"""Loss functions for vortex trajectory autofit."""

from __future__ import annotations

import numpy as np

from .config import ParameterSpec
from .features import TrajectoryFeatures


def compute_loss(
    features_num: TrajectoryFeatures,
    features_ana: TrajectoryFeatures,
    *,
    weights: dict[str, float],
    param_values: dict[str, float],
    param_specs: dict[str, ParameterSpec],
) -> tuple[float, dict[str, float]]:
    """Compute total weighted loss and per-component breakdown.

    Parameters
    ----------
    features_num : TrajectoryFeatures
        Features from the numerical (target) trajectory.
    features_ana : TrajectoryFeatures
        Features from the analytical (model) trajectory.
    weights : dict[str, float]
        Per-component weights (``w_xy``, ``w_r``, ``w_phi``, etc.).
    param_values : dict[str, float]
        Current parameter values (for regularisation).
    param_specs : dict[str, ParameterSpec]
        Parameter specifications (for prior penalties).

    Returns
    -------
    total_loss : float
        Weighted sum of all loss components.
    breakdown : dict[str, float]
        Individual (unweighted) loss terms.
    """
    breakdown: dict[str, float] = {}

    breakdown["L_xy"] = _loss_xy(features_num, features_ana)
    breakdown["L_r"] = _loss_radius(features_num, features_ana)
    breakdown["L_core"] = _loss_core_distance(features_num, features_ana)
    breakdown["L_phi"] = _loss_phase(features_num, features_ana)
    breakdown["L_freq"] = _loss_frequency(features_num, features_ana)
    breakdown["L_psd"] = _loss_psd(features_num, features_ana)
    breakdown["L_ellip"] = _loss_ellipticity(features_num, features_ana)
    breakdown["L_stability"] = _loss_stability(features_num, features_ana)
    breakdown["L_reg"] = _loss_regularization(param_values, param_specs)

    total = 0.0
    weight_map = {
        "L_xy": weights.get("w_xy", 0.0),
        "L_r": weights.get("w_r", 0.0),
        "L_core": weights.get("w_core", 0.0),
        "L_phi": weights.get("w_phi", 0.0),
        "L_freq": weights.get("w_freq", 0.0),
        "L_psd": weights.get("w_psd", 0.0),
        "L_ellip": weights.get("w_ellip", 0.0),
        "L_stability": weights.get("w_stability", 0.0),
        "L_reg": weights.get("w_reg", 0.0),
    }
    for key, value in breakdown.items():
        total += weight_map.get(key, 0.0) * value

    return total, breakdown


def _loss_xy(num: TrajectoryFeatures, ana: TrajectoryFeatures) -> float:
    """Normalised MSE of x(t) and y(t)."""
    n = min(num.x.size, ana.x.size)
    if n == 0:
        return 0.0

    x_num, y_num = num.x[:n], num.y[:n]
    x_ana, y_ana = ana.x[:n], ana.y[:n]

    # Normalise by variance of numerical trajectory
    var_x = float(np.var(x_num))
    var_y = float(np.var(y_num))
    scale = max(var_x + var_y, 1e-30)

    mse = float(np.mean((x_num - x_ana) ** 2 + (y_num - y_ana) ** 2))
    return mse / scale


def _loss_radius(num: TrajectoryFeatures, ana: TrajectoryFeatures) -> float:
    """Normalised MSE of r(t)."""
    n = min(num.r.size, ana.r.size)
    if n == 0:
        return 0.0

    r_num, r_ana = num.r[:n], ana.r[:n]
    scale = max(float(np.mean(r_num ** 2)), 1e-30)
    return float(np.mean((r_num - r_ana) ** 2)) / scale


def _loss_core_distance(num: TrajectoryFeatures, ana: TrajectoryFeatures) -> float:
    """Relative mismatch of absolute core distance from disk center."""
    num_mean = float(num.mean_core_distance)
    ana_mean = float(ana.mean_core_distance)
    num_max = float(num.max_core_distance)
    ana_max = float(ana.max_core_distance)

    mean_scale = max(abs(num_mean), 1e-30)
    max_scale = max(abs(num_max), 1e-30)
    mean_term = ((num_mean - ana_mean) / mean_scale) ** 2
    max_term = ((num_max - ana_max) / max_scale) ** 2
    return 0.5 * (mean_term + max_term)


def _loss_phase(num: TrajectoryFeatures, ana: TrajectoryFeatures) -> float:
    """Normalised MSE of unwrapped phase after alignment."""
    n = min(num.phi_unwrapped.size, ana.phi_unwrapped.size)
    if n < 2:
        return 0.0

    phi_num = num.phi_unwrapped[:n]
    phi_ana = ana.phi_unwrapped[:n]

    # Align phase offset
    offset = float(np.mean(phi_num - phi_ana))
    phi_ana_aligned = phi_ana + offset

    # Normalise by total phase accumulated
    total_phase = max(abs(float(phi_num[-1] - phi_num[0])), 1e-6)
    return float(np.mean((phi_num - phi_ana_aligned) ** 2)) / (total_phase ** 2)


def _loss_frequency(num: TrajectoryFeatures, ana: TrajectoryFeatures) -> float:
    """Relative squared error of dominant frequency."""
    f_num = num.dominant_freq_hz
    f_ana = ana.dominant_freq_hz

    if abs(f_num) < 1e-3:
        return 0.0

    return ((f_num - f_ana) / f_num) ** 2


def _loss_psd(num: TrajectoryFeatures, ana: TrajectoryFeatures) -> float:
    """Normalised PSD difference in the spectral peak window."""
    if num.psd_freqs.size == 0 or ana.psd_freqs.size == 0:
        return 0.0

    # Find peak in numerical PSD
    peak_idx = int(np.argmax(num.psd_power))
    peak_freq = num.psd_freqs[peak_idx]

    # Window around peak: ±30% of peak frequency
    f_low = peak_freq * 0.7
    f_high = peak_freq * 1.3

    # Mask for numerical
    mask_num = (num.psd_freqs >= f_low) & (num.psd_freqs <= f_high)
    if not np.any(mask_num):
        return 0.0

    psd_num_window = num.psd_power[mask_num]
    freqs_window = num.psd_freqs[mask_num]

    # Interpolate analytical PSD onto same frequency grid
    psd_ana_window = np.interp(freqs_window, ana.psd_freqs, ana.psd_power)

    # Normalise by numerical peak power
    scale = max(float(np.max(psd_num_window)), 1e-30)
    return float(np.mean((psd_num_window - psd_ana_window) ** 2)) / (scale ** 2)


def _loss_ellipticity(num: TrajectoryFeatures, ana: TrajectoryFeatures) -> float:
    """Squared difference of orbit eccentricity."""
    return (num.eccentricity - ana.eccentricity) ** 2


def _loss_stability(num: TrajectoryFeatures, ana: TrajectoryFeatures) -> float:
    """Penalise mismatch of radius-envelope drift and collapsed analytical orbits."""
    drift_mismatch = (num.radius_drift_ratio - ana.radius_drift_ratio) ** 2
    tail_scale = max(float(num.tail_mean_radius), 1e-30)
    tail_mismatch = ((float(num.tail_mean_radius) - float(ana.tail_mean_radius)) / tail_scale) ** 2

    target_radius = max(num.mean_radius, 1e-30)
    ana_radius_ratio = ana.mean_radius / target_radius
    collapse_penalty = 0.0
    if num.mean_radius > 0 and num.radius_drift_ratio >= 0.85 and ana_radius_ratio < 0.7:
        collapse_penalty = ((0.7 - ana_radius_ratio) / 0.7) ** 2
        tail_ratio = float(ana.tail_mean_radius) / tail_scale
        if tail_ratio < 0.75:
            collapse_penalty += ((0.75 - tail_ratio) / 0.75) ** 2

    return drift_mismatch + 0.5 * tail_mismatch + collapse_penalty


def _loss_regularization(
    param_values: dict[str, float],
    param_specs: dict[str, ParameterSpec],
) -> float:
    """Prior-based regularisation penalty."""
    penalty = 0.0
    for name, value in param_values.items():
        spec = param_specs.get(name)
        if spec is None or spec.prior_mean is None or spec.prior_std is None:
            continue
        if spec.prior_std <= 0:
            continue

        if spec.prior_type == "log_normal":
            if value <= 0 or spec.prior_mean <= 0:
                continue
            penalty += ((np.log(value) - np.log(spec.prior_mean)) / spec.prior_std) ** 2
        else:
            # Gaussian
            penalty += ((value - spec.prior_mean) / spec.prior_std) ** 2

    return penalty


__all__ = ["compute_loss"]
