"""Slavin-Tiberkevich parameter extraction from vortex trajectory data."""

from __future__ import annotations

import numpy as np

from ..core.models import TrajectoryResult
from ..spectrum.gyration import compute_gyration_spectrum
from .amplitude_equation import compute_amplitude_equation
from .models import STParametersResult


def _interpolate_half_height(
    x1: float, y1: float, x2: float, y2: float, y_half: float
) -> float:
    if abs(y2 - y1) < 1e-30:
        return float(0.5 * (x1 + x2))
    alpha = (y_half - y1) / (y2 - y1)
    return float(x1 + alpha * (x2 - x1))


def _estimate_linewidth_fwhm(
    frequencies: np.ndarray,
    power: np.ndarray,
) -> tuple[float, bool, dict[str, float | str]]:
    """Estimate spectral linewidth via FWHM around the dominant positive-frequency peak."""
    f = np.asarray(frequencies, dtype=float)
    p = np.asarray(power, dtype=float)

    mask = np.isfinite(f) & np.isfinite(p) & (f >= 0.0) & (p >= 0.0)
    f = f[mask]
    p = p[mask]

    if f.size < 3:
        return float("nan"), True, {"status": "insufficient_samples"}

    df = float(np.median(np.diff(f))) if f.size > 1 else float("nan")

    nonzero = np.where(f > 0.0)[0]
    if nonzero.size == 0:
        return float("nan"), True, {"status": "no_positive_frequency_peak", "df": df}

    peak_idx_local = int(np.argmax(p[nonzero]))
    peak_idx = int(nonzero[peak_idx_local])
    peak_power = float(p[peak_idx])
    if peak_power <= 0.0:
        return float("nan"), True, {"status": "non_positive_peak", "df": df}

    half = 0.5 * peak_power

    left = peak_idx
    while left > 0 and p[left] > half:
        left -= 1

    right = peak_idx
    while right < f.size - 1 and p[right] > half:
        right += 1

    if left == peak_idx or right == peak_idx:
        linewidth = max(df, 0.0) if np.isfinite(df) else float("nan")
        resolution_limited = bool(np.isfinite(df))
        return (
            linewidth,
            resolution_limited,
            {
                "status": "width_below_resolution",
                "df": df,
                "peak_frequency_hz": float(f[peak_idx]),
            },
        )

    f_left = _interpolate_half_height(f[left], p[left], f[left + 1], p[left + 1], half)
    f_right = _interpolate_half_height(
        f[right - 1], p[right - 1], f[right], p[right], half
    )

    linewidth = max(float(f_right - f_left), 0.0)
    if np.isfinite(df):
        linewidth = max(linewidth, df)

    resolution_limited = bool(np.isfinite(df) and linewidth <= 2.0 * df)

    return (
        linewidth,
        resolution_limited,
        {
            "status": "ok",
            "df": df,
            "peak_frequency_hz": float(f[peak_idx]),
            "f_left_hz": f_left,
            "f_right_hz": f_right,
        },
    )


def _fit_omega_vs_power(
    power: np.ndarray, omega: np.ndarray
) -> tuple[float, float, dict[str, float | str]]:
    """Fit linear relation ``omega(p)=omega_0+N*p``."""
    p = np.asarray(power, dtype=float)
    w = np.asarray(omega, dtype=float)

    mask = np.isfinite(p) & np.isfinite(w)
    p = p[mask]
    w = w[mask]

    if p.size < 3:
        omega_0 = float(np.mean(w)) if w.size else float("nan")
        return omega_0, 0.0, {"status": "insufficient_points", "n_points": int(p.size)}

    if float(np.std(p)) < 1e-15:
        omega_0 = float(np.mean(w))
        return omega_0, 0.0, {"status": "constant_power", "n_points": int(p.size)}

    slope, intercept = np.polyfit(p, w, 1)
    return (
        float(intercept),
        float(slope),
        {
            "status": "ok",
            "n_points": int(p.size),
        },
    )


def extract_st_parameters(
    trajectory: TrajectoryResult,
    *,
    spectrum_method: str = "welch",
    phase_method: str = "complex",
    steady_state_fraction: float = 0.4,
    reference_radius: float | None = None,
    current_a: float | None = None,
) -> STParametersResult:
    """Extract Slavin-Tiberkevich parameters from tracked trajectory."""
    amp = compute_amplitude_equation(
        trajectory,
        reference_radius=reference_radius,
        method=phase_method,
    )

    omega_0, n_coeff, fit_meta = _fit_omega_vs_power(amp.power, amp.omega)

    spectrum = compute_gyration_spectrum(trajectory, method=spectrum_method)
    if spectrum.frequencies.size == 0:
        f0_hz = float("nan")
        linewidth_hz = float("nan")
        linewidth_resolution_limited = True
        linewidth_meta: dict[str, float | str] = {"status": "spectrum_empty"}
    else:
        linewidth_hz, linewidth_resolution_limited, linewidth_meta = (
            _estimate_linewidth_fwhm(
                spectrum.frequencies,
                spectrum.power,
            )
        )
        f0_hz = float(
            linewidth_meta.get("peak_frequency_hz", spectrum.peak_frequency_hz)
        )

    gamma_g = (
        float(2.0 * np.pi * linewidth_hz) if np.isfinite(linewidth_hz) else float("nan")
    )

    n_points = amp.power.size
    fraction = float(np.clip(steady_state_fraction, 0.05, 1.0))
    start = int(max(0, np.floor((1.0 - fraction) * max(n_points - 1, 0))))
    tail = np.asarray(amp.power[start:], dtype=float)
    generation_power = float(np.mean(tail)) if tail.size else float("nan")

    if np.isfinite(generation_power) and generation_power > 1e-18 and tail.size > 1:
        q_coeff = float(np.std(tail) / generation_power)
    else:
        q_coeff = 0.0

    if np.isfinite(linewidth_hz) and linewidth_hz > 0.0 and np.isfinite(f0_hz):
        quality_factor = float(f0_hz / linewidth_hz)
    else:
        quality_factor = float("nan")

    metadata: dict[str, float | str | bool] = {
        "phase_method": phase_method,
        "spectrum_method": spectrum.method,
        "steady_state_fraction": fraction,
        "fit_status": str(fit_meta.get("status", "unknown")),
        "linewidth_status": str(linewidth_meta.get("status", "unknown")),
        "linewidth_resolution_limited": bool(linewidth_resolution_limited),
    }
    for key in ("df", "f_left_hz", "f_right_hz"):
        if key in linewidth_meta:
            metadata[key] = float(linewidth_meta[key])

    if current_a is not None:
        metadata["current_a"] = float(current_a)

    return STParametersResult(
        omega_0=float(omega_0),
        f_0_ghz=float(f0_hz * 1e-9) if np.isfinite(f0_hz) else float("nan"),
        N=float(n_coeff),
        Gamma_G=gamma_g,
        Q=q_coeff,
        sigma=float("nan"),
        I_threshold=float("nan"),
        generation_power=generation_power,
        linewidth_hz=float(linewidth_hz),
        quality_factor=quality_factor,
        linewidth_resolution_limited=bool(linewidth_resolution_limited),
        metadata=metadata,
    )


__all__ = ["extract_st_parameters"]
