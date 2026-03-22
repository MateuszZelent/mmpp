"""Feature extraction from vortex trajectories for autofit loss computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._shared.models import TrajectoryResult


@dataclass
class TrajectoryFeatures:
    """Extracted features from a vortex trajectory."""

    # Time-domain
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    r: np.ndarray
    phi_unwrapped: np.ndarray
    f_inst: np.ndarray  # instantaneous frequency [Hz]

    # Scalar summaries
    center_x: float
    center_y: float
    mean_radius: float
    max_radius: float
    mean_core_distance: float
    max_core_distance: float
    head_mean_radius: float
    tail_mean_radius: float
    radius_drift_ratio: float
    dominant_freq_hz: float

    # Spectral
    psd_freqs: np.ndarray
    psd_power: np.ndarray

    # Geometric
    eccentricity: float

    metadata: dict[str, Any] = field(default_factory=dict)


def _physical_core_coordinates(trajectory: TrajectoryResult) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates in the physical disk frame.

    Analytical trajectories may be shifted for overlay/alignment against the
    numerical orbit. For core-distance metrics we need the raw model-frame
    coordinates, so undo the display shift when metadata provides it.
    """
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)
    meta = dict(getattr(trajectory, "metadata", {}) or {})
    shift = meta.get("alignment_shift")
    if shift is None:
        return x, y
    try:
        sx, sy = shift
        return x - float(sx), y - float(sy)
    except Exception:
        return x, y


def extract_features(
    trajectory: TrajectoryResult,
    *,
    reference_radius: float | None = None,
) -> TrajectoryFeatures:
    """Extract feature set from a vortex trajectory.

    Parameters
    ----------
    trajectory : TrajectoryResult
        Numerical or analytical vortex trajectory.
    reference_radius : float, optional
        Disk radius for normalisation. If None, features are in absolute units.
    """
    time = np.asarray(trajectory.time, dtype=float)
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)

    center_x = float(np.mean(x)) if x.size else 0.0
    center_y = float(np.mean(y)) if y.size else 0.0

    dx = x - center_x
    dy = y - center_y
    r = np.hypot(dx, dy)
    phi = np.unwrap(np.arctan2(dy, dx))

    mean_radius = float(np.mean(r)) if r.size else 0.0
    max_radius = float(np.max(r)) if r.size else 0.0
    x_phys, y_phys = _physical_core_coordinates(trajectory)
    abs_r = np.hypot(x_phys, y_phys)
    mean_core_distance = float(np.mean(abs_r)) if abs_r.size else 0.0
    max_core_distance = float(np.max(abs_r)) if abs_r.size else 0.0
    if r.size:
        window = max(int(np.ceil(r.size * 0.1)), 1)
        head_mean_radius = float(np.mean(r[:window]))
        tail_mean_radius = float(np.mean(r[-window:]))
        radius_drift_ratio = tail_mean_radius / max(head_mean_radius, 1e-30)
    else:
        head_mean_radius = 0.0
        tail_mean_radius = 0.0
        radius_drift_ratio = 1.0

    # Instantaneous frequency from gradient of unwrapped phase
    if time.size >= 2:
        dphi_dt = np.gradient(phi, time)
        f_inst = np.abs(dphi_dt) / (2.0 * np.pi)
    else:
        f_inst = np.zeros_like(time)

    # PSD via Welch-like simple periodogram
    psd_freqs, psd_power = _compute_psd(time, dx, dy)

    # Dominant frequency from PSD
    if psd_power.size > 0:
        dominant_freq_hz = float(psd_freqs[np.argmax(psd_power)])
    else:
        dominant_freq_hz = float(np.mean(f_inst)) if f_inst.size else 0.0

    # Eccentricity from orbit ellipse fit
    eccentricity = _estimate_eccentricity(dx, dy)

    return TrajectoryFeatures(
        time=time,
        x=x,
        y=y,
        r=r,
        phi_unwrapped=phi,
        f_inst=f_inst,
        center_x=center_x,
        center_y=center_y,
        mean_radius=mean_radius,
        max_radius=max_radius,
        mean_core_distance=mean_core_distance,
        max_core_distance=max_core_distance,
        head_mean_radius=head_mean_radius,
        tail_mean_radius=tail_mean_radius,
        radius_drift_ratio=radius_drift_ratio,
        dominant_freq_hz=dominant_freq_hz,
        psd_freqs=psd_freqs,
        psd_power=psd_power,
        eccentricity=eccentricity,
        metadata={
            "reference_radius": reference_radius,
            "n_samples": int(time.size),
        },
    )


def _compute_psd(
    time: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute combined PSD of x and y fluctuations."""
    if time.size < 4:
        return np.array([]), np.array([])

    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        return np.array([]), np.array([])

    n = dx.size
    # Zero-pad to next power of 2 for efficiency
    nfft = 1 << (n - 1).bit_length()

    window = np.hanning(n)
    windowed_x = (dx - np.mean(dx)) * window
    windowed_y = (dy - np.mean(dy)) * window

    fft_x = np.fft.rfft(windowed_x, n=nfft)
    fft_y = np.fft.rfft(windowed_y, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=dt)

    # Combined PSD (sum of x and y power)
    power = (np.abs(fft_x) ** 2 + np.abs(fft_y) ** 2) / (n * np.sum(window ** 2))

    # Skip DC component
    return freqs[1:], power[1:]


def _estimate_eccentricity(dx: np.ndarray, dy: np.ndarray) -> float:
    """Estimate orbit eccentricity from covariance of centered position."""
    if dx.size < 3:
        return 0.0

    cov = np.cov(dx, dy)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

    if eigenvalues[0] <= 0:
        return 0.0

    ratio = eigenvalues[1] / eigenvalues[0]
    # Eccentricity: 0 = perfect circle, 1 = line
    return float(np.sqrt(1.0 - np.clip(ratio, 0.0, 1.0)))


__all__ = ["TrajectoryFeatures", "extract_features", "_physical_core_coordinates"]
