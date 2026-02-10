"""Core-tracking algorithms for vortex dynamics."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .._utils import XYConvention
from .methods import TRACKING_METHODS
from .models import TrajectoryResult

try:
    from scipy.optimize import curve_fit

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    curve_fit = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


def _normalize_tracking_input(data: np.ndarray, z_layer: int = -1) -> np.ndarray:
    """Normalize supported inputs to shape ``(Nt, Ny, Nx, 3)``."""
    arr = np.asarray(data, dtype=float)

    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr[np.newaxis, ...]

    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr

    if arr.ndim == 5 and arr.shape[-1] == 3:
        nz = arr.shape[1]
        if z_layer < 0:
            z_layer += nz
        if z_layer < 0 or z_layer >= nz:
            raise IndexError(f"z_layer index {z_layer} out of bounds for shape {arr.shape}")
        return arr[:, z_layer, ...]

    raise ValueError(
        "Unsupported magnetization shape for tracking. Expected "
        "(Ny,Nx,3), (Nt,Ny,Nx,3), or (Nt,Nz,Ny,Nx,3)."
    )


def _track_maximum(mz: np.ndarray, dx: float, dy: float) -> tuple[float, float, float]:
    """Track core as argmax of ``|m_z|``."""
    abs_mz = np.abs(mz)
    yi, xi = np.unravel_index(int(np.argmax(abs_mz)), abs_mz.shape)
    return float(xi) * dx, float(yi) * dy, float(abs_mz[yi, xi])


def _track_centroid(
    mz: np.ndarray,
    dx: float,
    dy: float,
    core_threshold: float,
) -> tuple[float, float, float]:
    """Track core using weighted centroid of the high-|mz| mask."""
    abs_mz = np.abs(mz)
    peak = float(np.max(abs_mz))
    if peak <= 0.0:
        return 0.0, 0.0, 0.0

    mask = abs_mz >= core_threshold * peak
    if not np.any(mask):
        return _track_maximum(mz, dx, dy)

    weights = np.where(mask, abs_mz**2, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return _track_maximum(mz, dx, dy)

    ny, nx = mz.shape
    x_idx = np.arange(nx, dtype=float)
    y_idx = np.arange(ny, dtype=float)
    x_grid, y_grid = np.meshgrid(x_idx, y_idx)

    x_pix = float(np.sum(weights * x_grid) / total)
    y_pix = float(np.sum(weights * y_grid) / total)
    confidence = float(np.mean(abs_mz[mask]))

    return x_pix * dx, y_pix * dy, confidence


def _gaussian_2d(
    coords: tuple[np.ndarray, np.ndarray],
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    offset: float,
) -> np.ndarray:
    """2D Gaussian surface used for sub-pixel fitting."""
    x, y = coords
    sx = max(abs(sigma_x), 1e-15)
    sy = max(abs(sigma_y), 1e-15)

    exponent = ((x - x0) ** 2) / (2.0 * sx**2) + ((y - y0) ** 2) / (2.0 * sy**2)
    return amplitude * np.exp(-exponent) + offset


def _track_gaussian(
    mz: np.ndarray,
    dx: float,
    dy: float,
    gaussian_roi: int,
    core_threshold: float,
) -> tuple[float, float, float, bool]:
    """Track core using local 2D Gaussian fit."""
    def _fallback_scaled(scale: float = 0.45) -> tuple[float, float, float, bool]:
        x, y, conf = _track_centroid(mz, dx, dy, core_threshold)
        return x, y, float(np.clip(conf * scale, 0.0, 1.0)), False

    if not SCIPY_AVAILABLE or curve_fit is None:
        return _fallback_scaled()

    abs_mz = np.abs(mz)
    yi0, xi0 = np.unravel_index(int(np.argmax(abs_mz)), abs_mz.shape)

    half = max(int(gaussian_roi) // 2, 1)
    y_start = max(0, yi0 - half)
    y_end = min(mz.shape[0], yi0 + half + 1)
    x_start = max(0, xi0 - half)
    x_end = min(mz.shape[1], xi0 + half + 1)

    roi = abs_mz[y_start:y_end, x_start:x_end]
    roi_touches_boundary = (
        y_start == 0 or x_start == 0 or y_end == mz.shape[0] or x_end == mz.shape[1]
    )

    # Near sample edges ROI is clipped, and unconstrained Gaussian fits are unstable.
    if roi_touches_boundary:
        return _fallback_scaled()

    if roi.size < 9:
        return _fallback_scaled()

    x_coords = np.arange(x_start, x_end, dtype=float) * dx
    y_coords = np.arange(y_start, y_end, dtype=float) * dy
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    coords = (x_grid.ravel(), y_grid.ravel())
    values = roi.ravel()

    peak = float(np.max(values))
    baseline = float(np.min(values))
    p0 = [
        max(peak - baseline, 1e-9),
        float(xi0) * dx,
        float(yi0) * dy,
        max(dx * half / 2.0, dx),
        max(dy * half / 2.0, dy),
        baseline,
    ]

    bounds = (
        [0.0, x_coords[0], y_coords[0], dx * 0.25, dy * 0.25, -1.0],
        [2.0, x_coords[-1], y_coords[-1], dx * 20.0, dy * 20.0, 1.0],
    )

    try:
        params, _ = curve_fit(
            _gaussian_2d,
            coords,
            values,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )
    except Exception:
        return _fallback_scaled()

    sigma_x = float(params[3])
    sigma_y = float(params[4])
    sigma_min_x = 0.3 * dx
    sigma_min_y = 0.3 * dy
    sigma_max_x = max(float(half) * dx, dx)
    sigma_max_y = max(float(half) * dy, dy)
    if not (sigma_min_x <= sigma_x <= sigma_max_x and sigma_min_y <= sigma_y <= sigma_max_y):
        return _fallback_scaled()

    fit_values = _gaussian_2d(coords, *params)
    residual_norm = float(np.linalg.norm(values - fit_values))
    signal_norm = float(np.linalg.norm(values))
    confidence = float(
        np.clip(1.0 - residual_norm / max(signal_norm, 1e-12), 0.0, 1.0)
    )

    return float(params[1]), float(params[2]), confidence, True


def _resolve_convention(convention: XYConvention | None) -> XYConvention:
    if convention is None:
        return XYConvention()
    return convention


def _down_to_physical_y(y_down: float, ny: int, dy: float, convention: XYConvention) -> float:
    if convention.y_axis == "up":
        return (float(ny - 1) * float(dy)) - float(y_down)
    return float(y_down)


def _polarity_from_position_down(
    mz: np.ndarray,
    x_down: float,
    y_down: float,
    dx: float,
    dy: float,
) -> int:
    """Get polarity sign from nearest-grid ``m_z`` at tracked down-axis position."""
    ny, nx = mz.shape
    xi = int(np.clip(round(x_down / dx), 0, nx - 1))
    yi = int(np.clip(round(y_down / dy), 0, ny - 1))
    return 1 if mz[yi, xi] >= 0.0 else -1


def _sample_core_signal(
    mz: np.ndarray,
    x_down: float,
    y_down: float,
    dx: float,
    dy: float,
    roi_pixels: int,
) -> float:
    """Average ``m_z`` in local ROI around tracked core position."""
    ny, nx = mz.shape
    xi = int(np.clip(round(x_down / dx), 0, nx - 1))
    yi = int(np.clip(round(y_down / dy), 0, ny - 1))
    radius = max(int(roi_pixels), 0)
    x0 = max(0, xi - radius)
    x1 = min(nx, xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(ny, yi + radius + 1)
    return float(np.mean(mz[y0:y1, x0:x1]))


def _extract_polarity_series(
    core_signal: np.ndarray,
    dt: float,
    *,
    threshold_up: float,
    threshold_down: float,
) -> tuple[np.ndarray, int, list[float]]:
    """Extract polarity time series with hysteresis thresholds."""
    values = np.asarray(core_signal, dtype=float)
    n = values.size
    if n == 0:
        return np.array([], dtype=int), 0, []

    if values[0] >= threshold_up:
        state = 1
    elif values[0] <= threshold_down:
        state = -1
    else:
        state = 1 if values[0] >= 0.0 else -1

    polarity = np.zeros(n, dtype=int)
    switch_times: list[float] = []
    switch_count = 0

    for idx, value in enumerate(values):
        if state > 0 and value <= threshold_down:
            state = -1
            switch_count += 1
            switch_times.append(float(idx) * float(dt))
        elif state < 0 and value >= threshold_up:
            state = 1
            switch_count += 1
            switch_times.append(float(idx) * float(dt))
        polarity[idx] = state

    return polarity, switch_count, switch_times


def track_core(
    data: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    *,
    method: str = "gaussian",
    z_layer: int = -1,
    core_threshold: float = 0.9,
    gaussian_roi: int = 7,
    convention: XYConvention | None = None,
    polarity_threshold_up: float = 0.3,
    polarity_threshold_down: float = -0.3,
    polarity_roi_pixels: int = 1,
    metadata: dict[str, Any] | None = None,
) -> TrajectoryResult:
    """Track vortex core position over time."""
    series = _normalize_tracking_input(data, z_layer=z_layer)
    nt = series.shape[0]
    conv = _resolve_convention(convention)

    requested_method = method.lower()
    effective_method = requested_method
    fallback_from: str | None = None

    if requested_method == "gaussian" and not SCIPY_AVAILABLE:
        warnings.warn(
            "SciPy is unavailable; falling back from 'gaussian' to 'centroid'.",
            RuntimeWarning,
            stacklevel=2,
        )
        effective_method = "centroid"
        fallback_from = "gaussian"

    if effective_method not in TRACKING_METHODS:
        raise ValueError(
            "Unknown tracking method. Use 'maximum', 'centroid', or 'gaussian'."
        )

    x_down = np.zeros(nt, dtype=float)
    y_down = np.zeros(nt, dtype=float)
    x_phys = np.zeros(nt, dtype=float)
    y_phys = np.zeros(nt, dtype=float)
    confidence = np.zeros(nt, dtype=float)
    core_signal = np.zeros(nt, dtype=float)

    gaussian_frame_fallbacks = 0

    for idx in range(nt):
        frame = series[idx]
        mz = frame[..., 2]

        if effective_method == "maximum":
            xi, yi, conf = _track_maximum(mz, dx, dy)
        elif effective_method == "centroid":
            xi, yi, conf = _track_centroid(mz, dx, dy, core_threshold)
        else:  # gaussian
            xi, yi, conf, success = _track_gaussian(
                mz,
                dx,
                dy,
                gaussian_roi,
                core_threshold,
            )
            if not success:
                gaussian_frame_fallbacks += 1

        x_down[idx] = xi
        y_down[idx] = yi
        x_phys[idx] = xi
        y_phys[idx] = _down_to_physical_y(yi, mz.shape[0], dy, conv)
        confidence[idx] = conf
        core_signal[idx] = _sample_core_signal(
            mz,
            xi,
            yi,
            dx,
            dy,
            roi_pixels=polarity_roi_pixels,
        )

    time = np.arange(nt, dtype=float) * float(dt)
    polarity, switch_count, switch_times = _extract_polarity_series(
        core_signal,
        dt,
        threshold_up=float(polarity_threshold_up),
        threshold_down=float(polarity_threshold_down),
    )

    result_metadata: dict[str, Any] = {
        "dx": float(dx),
        "dy": float(dy),
        "dt": float(dt),
        "n_frames": int(nt),
        "requested_method": requested_method,
        "gaussian_frame_fallbacks": int(gaussian_frame_fallbacks),
        "convention": conv.y_axis,
        "polarity_threshold_up": float(polarity_threshold_up),
        "polarity_threshold_down": float(polarity_threshold_down),
        "polarity_roi_pixels": int(max(0, polarity_roi_pixels)),
        "p_switch_count": int(switch_count),
        "switch_times_s": [float(v) for v in switch_times],
        "core_signal_mz": np.asarray(core_signal, dtype=float),
    }
    if metadata:
        result_metadata.update(metadata)
    if fallback_from is not None:
        result_metadata["fallback_from"] = fallback_from

    return TrajectoryResult(
        time=time,
        x=x_phys,
        y=y_phys,
        polarity=polarity,
        method=effective_method,
        confidence=np.clip(confidence, 0.0, 1.0),
        metadata=result_metadata,
    )


__all__ = ["SCIPY_AVAILABLE", "track_core"]
