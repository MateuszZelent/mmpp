# ruff: noqa: UP007
"""Dependency-light numerical primitives for skyrmion measurements."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .._coordinates import XYConvention
from .._field import select_snapshot
from .._topology import topological_density_fd
from .config import SizeFitConfig, SkyrmionTopologyConfig
from .models import SkyrmionSizeResult, SkyrmionTopologyResult


def _y_axis(convention: Optional[XYConvention]) -> str:
    return str(getattr(convention, "y_axis", "up")).lower()


def _prepare(
    m: np.ndarray,
    mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(m, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("Expected a two-dimensional field with shape (Ny, Nx, 3).")
    field = arr[..., :3]
    valid = np.all(np.isfinite(field), axis=-1)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != field.shape[:2]:
            raise ValueError("mask must have shape (Ny, Nx).")
        valid &= mask_arr
    norms = np.linalg.norm(field, axis=-1)
    # MuMax-style geometries commonly store zero vectors outside the material.
    # They are geometry, not a failed skyrmion measurement, so exclude them.
    valid &= norms > 1e-12
    if not np.any(valid):
        raise ValueError("The analysis mask contains no valid magnetisation vectors.")
    normalized = np.zeros_like(field)
    normalized[valid] = field[valid] / norms[valid, None]
    return normalized, valid


def _coordinates(
    shape: tuple[int, int], dx: float, dy: float, convention: Optional[XYConvention]
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    x = np.arange(nx, dtype=float)[None, :] * float(dx)
    y_index = np.arange(ny, dtype=float)[:, None]
    if _y_axis(convention) == "up":
        y = (float(ny - 1) - y_index) * float(dy)
    else:
        y = y_index * float(dy)
    return np.broadcast_to(x, shape), np.broadcast_to(y, shape)


def _background(
    mz: np.ndarray,
    valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Estimate background from the outer radial shell of valid material."""
    xmin, xmax = float(np.min(x[valid])), float(np.max(x[valid]))
    ymin, ymax = float(np.min(y[valid])), float(np.max(y[valid]))
    geometry_centre = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
    radial = np.hypot(x - geometry_centre[0], y - geometry_centre[1])
    shell_start = float(np.quantile(radial[valid], 0.8))
    values = mz[valid & (radial >= shell_start)]
    if values.size < 8:
        values = mz[valid]
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    return median, mad


def _centroid(
    weights: np.ndarray, x: np.ndarray, y: np.ndarray, valid: np.ndarray
) -> tuple[float, float]:
    w = np.where(valid, np.maximum(np.asarray(weights, dtype=float), 0.0), 0.0)
    total = float(np.sum(w))
    if total <= 1e-30:
        return float(np.mean(x[valid])), float(np.mean(y[valid]))
    return float(np.sum(w * x) / total), float(np.sum(w * y) / total)


def _q_density(
    field: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    convention: Optional[XYConvention],
) -> tuple[np.ndarray, float]:
    """Calculate Berg--Luscher charge density using pure NumPy/Python."""
    ny, nx, _ = field.shape
    density = np.zeros((ny, nx), dtype=float)
    sign = -1.0 if _y_axis(convention) == "up" else 1.0

    def charge(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        numerator = float(np.dot(a, np.cross(b, c)))
        denominator = float(1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a))
        return float(2.0 * np.arctan2(numerator, denominator) / (4.0 * np.pi))

    total = 0.0
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            if not (
                valid[iy, ix]
                and valid[iy, ix + 1]
                and valid[iy + 1, ix]
                and valid[iy + 1, ix + 1]
            ):
                continue
            a = field[iy, ix]
            b = field[iy, ix + 1]
            c = field[iy + 1, ix + 1]
            d = field[iy + 1, ix]
            q1 = sign * charge(a, b, c)
            q2 = sign * charge(a, c, d)
            total += q1 + q2
            density[iy, ix] += (q1 + q2) / 3.0
            density[iy, ix + 1] += q1 / 3.0
            density[iy + 1, ix + 1] += (q1 + q2) / 3.0
            density[iy + 1, ix] += q2 / 3.0
    area = float(dx) * float(dy)
    return density / max(area, 1e-30), float(total)


def _charge_density(
    field: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    convention: Optional[XYConvention],
    method: str,
) -> tuple[np.ndarray, float]:
    """Dispatch the requested topological-charge estimator."""
    selected = str(method).lower()
    if selected == "berg_luscher":
        return _q_density(field, valid, dx, dy, convention)
    if selected == "finite_diff":
        density, _ = topological_density_fd(
            field,
            dx,
            dy,
            convention=convention,
        )
        # The shared finite-difference primitive preserves the historical
        # vortex sign convention.  Align it with the Berg--Luscher skyrmion
        # convention without changing the existing vortex contract.
        density = np.asarray(density, dtype=float)
        if _y_axis(convention) == "up":
            density = -density

        # A derivative next to a geometry hole samples the zero-filled
        # exterior and creates a spurious charge sheet.  Keep only cells whose
        # first-order finite-difference stencil is fully inside the material.
        interior = np.zeros_like(valid, dtype=bool)
        if valid.shape[0] >= 3 and valid.shape[1] >= 3:
            interior[1:-1, 1:-1] = (
                valid[1:-1, 1:-1]
                & valid[:-2, 1:-1]
                & valid[2:, 1:-1]
                & valid[1:-1, :-2]
                & valid[1:-1, 2:]
            )
        density = np.where(interior, density, 0.0)
        total = float(np.sum(density) * float(dx) * float(dy))
        return density, total
    raise ValueError("method must be 'berg_luscher' or 'finite_diff'")


def _levels(r: np.ndarray, u: np.ndarray, target: float) -> Optional[float]:
    order = np.argsort(r)
    rr = np.asarray(r, dtype=float)[order]
    uu = np.asarray(u, dtype=float)[order]
    finite = np.isfinite(rr) & np.isfinite(uu)
    rr, uu = rr[finite], uu[finite]
    if rr.size < 2:
        return None
    # A short median filter suppresses one-cell spikes without SciPy.
    smooth = uu.copy()
    if uu.size >= 3:
        smooth[1:-1] = np.median(np.stack((uu[:-2], uu[1:-1], uu[2:])), axis=0)
    for i in range(1, smooth.size):
        if smooth[i - 1] >= target and smooth[i] <= target:
            denominator = smooth[i] - smooth[i - 1]
            fraction = (
                0.0
                if abs(denominator) <= 1e-30
                else (target - smooth[i - 1]) / denominator
            )
            return float(rr[i - 1] + fraction * (rr[i] - rr[i - 1]))
    index = int(np.argmin(np.abs(smooth - target)))
    if abs(float(smooth[index]) - target) < 0.08:
        return float(rr[index])
    return None


def _profile(
    mz: np.ndarray,
    valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    center: tuple[float, float],
    dx: float,
    dy: float,
    config: SizeFitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    radius = np.hypot(x - center[0], y - center[1])
    cell = max(min(abs(float(dx)), abs(float(dy))), 1e-30)
    bin_width = float(config.radial_bin_m or 0.5 * cell)
    bin_width = max(bin_width, 0.25 * cell)
    max_radius = float(np.quantile(radius[valid], config.edge_fraction))
    edges = np.arange(0.0, max_radius + bin_width, bin_width)
    if edges.size < config.min_profile_bins + 1:
        edges = np.linspace(0.0, max_radius, config.min_profile_bins + 1)
    indices = np.digitize(radius, edges) - 1
    angles = np.mod(np.arctan2(y - center[1], x - center[0]), 2.0 * np.pi)
    angular_sectors = 36
    r_out, z_out, n_out = [], [], []
    for i in range(edges.size - 1):
        take = valid & (indices == i)
        if np.count_nonzero(take) < 3:
            continue
        median_radius = float(np.median(radius[take]))
        if median_radius > 2.0 * cell:
            occupied = np.unique(
                np.floor(angles[take] * angular_sectors / (2.0 * np.pi)).astype(int)
            ).size
            expected = min(
                angular_sectors,
                max(4, int(round(2.0 * np.pi * median_radius / cell))),
            )
            angular_coverage = min(float(occupied) / float(expected), 1.0)
            if angular_coverage < float(config.min_angular_coverage):
                continue
        vals = mz[take]
        r_out.append(median_radius)
        z_out.append(float(np.median(vals)))
        n_out.append(float(vals.size))
    return np.asarray(r_out), np.asarray(z_out), np.asarray(n_out), max_radius


def _linear_parameters(
    basis: np.ndarray, values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    root = np.sqrt(np.maximum(weights, 1e-12))[:, None]
    design = basis * root
    target = values * root[:, 0]
    params, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = values - basis @ params
    return params, float(np.sum(weights * residual * residual))


def _ansatz_contrast(
    r: np.ndarray,
    radius: float,
    scale: float,
) -> np.ndarray:
    """Circular 360-degree domain-wall contrast profile."""
    safe_scale = max(float(scale), 1e-30)
    inner = np.clip((r - float(radius)) / safe_scale, -60.0, 60.0)
    outer = np.clip((r + float(radius)) / safe_scale, -60.0, 60.0)
    theta = 2.0 * np.arctan(np.exp(inner))
    theta += 2.0 * np.arctan(np.exp(outer))
    return 0.5 * (1.0 - np.cos(theta))


def _grid_fit(
    r: np.ndarray,
    z: np.ndarray,
    n: np.ndarray,
    background: float,
    contrast: float,
    kind: str,
    cell: float,
) -> tuple[dict[str, float], float, int]:
    """Real bounded least-squares fit using NumPy grid refinement."""
    if r.size == 0:
        return {}, float("inf"), 0
    weights = n / np.maximum(np.median(n), 1.0)
    rmax = max(float(np.max(r)), cell)
    if kind == "gaussian":
        best = (float("inf"), {})
        lower = max(cell * 0.25, rmax / 100.0)
        upper = max(rmax * 2.0, cell)
        for _ in range(5):
            sigma_grid = np.geomspace(lower, upper, 64)
            for sigma in sigma_grid:
                basis = np.column_stack(
                    (np.ones_like(r), np.exp(-0.5 * (r / sigma) ** 2))
                )
                params, loss = _linear_parameters(basis, z, weights)
                if loss < best[0]:
                    best = (
                        loss,
                        {
                            "offset": float(params[0]),
                            "amplitude": float(params[1]),
                            "sigma": float(sigma),
                        },
                    )
            center = best[1]["sigma"]
            lower = max(cell * 0.1, center / 1.35)
            upper = max(center * 1.35, lower * 1.01)
        return best[1], best[0], 3

    if abs(float(contrast)) <= 1e-30:
        return {}, float("inf"), 0
    r_center = _levels(r, (z - float(background)) / float(contrast), 0.5)
    center = float(r_center if r_center is not None else np.median(r))
    span = max(0.5 * rmax, 4.0 * cell)
    scale_best = max(cell, 0.1 * rmax)
    best = (float("inf"), {})
    for _ in range(4):
        r_grid = np.linspace(max(0.0, center - span), min(rmax, center + span), 24)
        s_grid = np.geomspace(
            max(cell * 0.2, scale_best / 4.0), max(scale_best * 4.0, cell), 22
        )
        for radius in r_grid:
            for scale in s_grid:
                if kind == "ansatz":
                    shape = _ansatz_contrast(r, radius, scale)
                else:
                    shape = np.tanh((r - radius) / scale)
                basis = np.column_stack((np.ones_like(r), shape))
                params, loss = _linear_parameters(basis, z, weights)
                if loss < best[0]:
                    best = (
                        loss,
                        {
                            "offset": float(params[0]),
                            "amplitude": float(params[1]),
                            "radius": float(radius),
                            "scale": float(scale),
                        },
                    )
        center = best[1]["radius"]
        scale_best = best[1]["scale"]
        span *= 0.35
    return best[1], best[0], 4


def _aicc(loss: float, n: int, k: int) -> float:
    if n <= k + 1 or not np.isfinite(loss) or loss < 0.0:
        return float("inf")
    safe_loss = max(float(loss), np.finfo(float).tiny)
    value = n * np.log(safe_loss / n) + 2.0 * k
    return float(value + (2.0 * k * (k + 1.0)) / (n - k - 1.0))


def _center_and_topology(
    field: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    convention: Optional[XYConvention],
    config: SkyrmionTopologyConfig,
) -> dict[str, Any]:
    mz = field[..., 2]
    x, y = _coordinates(mz.shape, dx, dy, convention)
    background, background_scatter = _background(mz, valid, x, y)
    contrast_field = mz - background
    minimum = float(np.min(contrast_field[valid]))
    maximum = float(np.max(contrast_field[valid]))
    direction = -1.0 if abs(minimum) >= abs(maximum) else 1.0
    positive_contrast = np.maximum(direction * contrast_field, 0.0)
    threshold = 0.2 * float(np.max(positive_contrast[valid]))
    contrast_center = _centroid(
        np.where(positive_contrast >= threshold, positive_contrast, 0.0), x, y, valid
    )
    q_density, q_total = _charge_density(
        field,
        valid,
        dx,
        dy,
        convention,
        config.method,
    )
    q_sign = 1.0 if q_total >= 0.0 else -1.0
    q_weights = np.maximum(q_sign * q_density, 0.0)
    peak_q = float(np.max(np.abs(q_density[valid])))
    if peak_q > 0.0 and float(config.q_threshold_fraction) > 0.0:
        q_weights = np.where(
            np.abs(q_density) >= float(config.q_threshold_fraction) * peak_q,
            q_weights,
            0.0,
        )
    q_center = _centroid(q_weights, x, y, valid)
    q_abs_integral = float(np.sum(np.abs(q_density[valid])) * dx * dy)
    q_purity = abs(q_total) / max(q_abs_integral, 1e-30)
    use_q = (
        abs(q_total) >= float(config.min_abs_q)
        and np.sum(q_weights) > 0.0
        and q_purity >= 0.25
    )
    center = q_center if use_q else contrast_center
    core_radius = max(2.0 * max(abs(dx), abs(dy)), 1e-30)
    core_take = valid & (np.hypot(x - center[0], y - center[1]) <= core_radius)
    core_mz = (
        float(np.median(mz[core_take]))
        if np.any(core_take)
        else float(
            mz[
                np.unravel_index(
                    np.argmax(np.where(valid, positive_contrast, -np.inf)),
                    mz.shape,
                )
            ]
        )
    )
    center_disagreement = float(
        np.hypot(q_center[0] - contrast_center[0], q_center[1] - contrast_center[1])
    )
    return {
        "x": x,
        "y": y,
        "background": background,
        "background_scatter": background_scatter,
        "contrast": contrast_field,
        "center": center,
        "q_density": q_density,
        "q_total": q_total,
        "q_abs_integral": q_abs_integral,
        "q_purity": q_purity,
        "q_center": q_center,
        "contrast_center": contrast_center,
        "center_disagreement": center_disagreement,
        "core_mz": core_mz,
        "valid": valid,
        "field": field,
    }


def detect_skyrmion(
    m: np.ndarray,
    dx: float,
    dy: float,
    *,
    frame: int = 0,
    z_layer: int = -1,
    mask: Optional[np.ndarray] = None,
    convention: Optional[XYConvention] = None,
    config: Optional[SkyrmionTopologyConfig] = None,
) -> SkyrmionTopologyResult:
    """Detect skyrmion charge, centre, polarity and background using NumPy."""
    if not np.isfinite(float(dx)) or not np.isfinite(float(dy)):
        raise ValueError("dx and dy must be finite.")
    if float(dx) <= 0.0 or float(dy) <= 0.0:
        raise ValueError("dx and dy must be positive.")
    snapshot = select_snapshot(m, frame=frame, z_layer=z_layer)
    field, valid = _prepare(snapshot, mask)
    cfg = config or SkyrmionTopologyConfig()
    selected_convention = convention or cfg.convention
    values = _center_and_topology(
        field,
        valid,
        float(dx),
        float(dy),
        selected_convention,
        cfg,
    )
    center = values["center"]
    x, y = values["x"], values["y"]
    radius = np.hypot(x - center[0], y - center[1])
    localized = float(
        np.sum(
            np.abs(values["q_density"])[
                valid
                & (radius <= max(4.0 * max(dx, dy), np.quantile(radius[valid], 0.5)))
            ]
            * dx
            * dy
        )
        / max(values["q_abs_integral"], 1e-30)
    )
    polarity = int(np.sign(values["core_mz"])) if abs(values["core_mz"]) >= 0.15 else 0
    background_sign = (
        int(np.sign(values["background"])) if abs(values["background"]) >= 0.15 else 0
    )
    flags = []
    if abs(values["core_mz"] - values["background"]) < float(cfg.min_contrast):
        flags.append("insufficient_contrast")
    if not (abs(values["q_total"]) >= float(cfg.min_abs_q)):
        flags.append("q_below_threshold")
    if values["q_purity"] < 0.25:
        flags.append("q_density_not_localized")
    if values["center_disagreement"] > 4.0 * max(dx, dy):
        flags.append("center_methods_disagree")
    confidence = float(
        np.clip(
            min(abs(values["q_total"]), 1.0)
            * values["q_purity"]
            * (
                1.0
                - min(
                    values["center_disagreement"] / max(8.0 * max(dx, dy), 1e-30), 1.0
                )
            ),
            0.0,
            1.0,
        )
    )
    reversed_core = (
        polarity != 0 and background_sign != 0 and polarity != background_sign
    )
    if not reversed_core:
        flags.append("core_not_reversed")
    valid_result = (
        np.isfinite(values["q_total"])
        and "insufficient_contrast" not in flags
        and "q_below_threshold" not in flags
        and "q_density_not_localized" not in flags
        and "core_not_reversed" not in flags
    )
    if (
        valid_result
        and abs(values["q_total"]) >= float(cfg.min_abs_q)
        and reversed_core
    ):
        state = "skyrmion"
    elif "insufficient_contrast" in flags:
        state = "uniform"
    else:
        state = "non_skyrmion"
    return SkyrmionTopologyResult(
        Q=float(values["q_total"]),
        center_xy_m=center,
        polarity=polarity,
        background_sign=background_sign,
        core_mz=float(values["core_mz"]),
        background_mz=float(values["background"]),
        contrast_mz=float(values["core_mz"] - values["background"]),
        q_density=values["q_density"],
        q_abs_integral=float(values["q_abs_integral"]),
        q_purity=float(values["q_purity"]),
        q_localized_fraction=localized,
        confidence=confidence,
        valid=valid_result,
        method=cfg.method,
        convention=_y_axis(selected_convention),
        state=state,
        flags=tuple(flags),
        metadata={
            "background_scatter_mz": values["background_scatter"],
            "q_center_xy_m": values["q_center"],
            "contrast_center_xy_m": values["contrast_center"],
            "center_disagreement_m": values["center_disagreement"],
            "valid_fraction": float(np.mean(valid)),
            "frame": int(frame),
            "z_layer": int(z_layer),
        },
    )


def _threshold_result(
    values: dict[str, Any], dx: float, dy: float, config: SizeFitConfig, fit_method: str
) -> SkyrmionSizeResult:
    r, z, n, _ = _profile(
        values["field"][..., 2],
        values["valid"],
        values["x"],
        values["y"],
        values["center"],
        dx,
        dy,
        config,
    )
    background = float(values["background"])
    core = float(values["core_mz"])
    contrast = core - background
    flags = []
    if abs(contrast) < float(config.min_contrast):
        flags.append("insufficient_contrast")
    if abs(contrast) <= 1e-30:
        u = np.full_like(z, np.nan)
    else:
        u = (z - background) / contrast
    r90 = _levels(r, u, 0.9)
    r50 = _levels(r, u, 0.5)
    r10 = _levels(r, u, 0.1)
    if r50 is None:
        flags.append("no_contrast50_crossing")
    if r90 is None or r10 is None:
        flags.append("no_wall_crossing")
    width = None if r90 is None or r10 is None else max(float(r10 - r90), 0.0)
    model_z = np.array([], dtype=float)
    return SkyrmionSizeResult(
        center_xy_m=values["center"],
        radius_m=r50,
        diameter_m=None if r50 is None else 2.0 * r50,
        wall_width_m=width,
        scale_m=None if width is None else width / 2.197224577,
        radius_90_m=r90,
        radius_50_m=r50,
        radius_10_m=r10,
        sigma_m=None,
        gaussian_fwhm_m=None,
        model="threshold",
        fit_method=fit_method,
        fit_success=r50 is not None and "insufficient_contrast" not in flags,
        background_mz=background,
        core_mz=core,
        contrast_mz=contrast,
        normalized_rmse=float("nan"),
        aicc=float("inf"),
        quality=(
            "invalid"
            if "insufficient_contrast" in flags
            else ("questionable" if flags else "good")
        ),
        requested_method="threshold",
        flags=tuple(flags),
        radial_r_m=r,
        radial_mz=z,
        model_mz=model_z,
        metadata={"sample_count_per_bin": n},
    )


def fit_skyrmion_size(
    m: np.ndarray,
    dx: float,
    dy: float,
    *,
    method: str = "auto",
    frame: int = 0,
    z_layer: int = -1,
    mask: Optional[np.ndarray] = None,
    convention: Optional[XYConvention] = None,
    config: Optional[SizeFitConfig] = None,
    topology: Optional[SkyrmionTopologyResult] = None,
) -> SkyrmionSizeResult:
    """Fit skyrmion size using bounded NumPy least squares and AICc selection."""
    if not np.isfinite(float(dx)) or not np.isfinite(float(dy)):
        raise ValueError("dx and dy must be finite.")
    if float(dx) <= 0.0 or float(dy) <= 0.0:
        raise ValueError("dx and dy must be positive.")
    selected = str(method).lower()
    allowed = {"auto", "domain_wall", "ansatz", "gaussian", "threshold"}
    if selected not in allowed:
        raise ValueError(
            "method must be 'auto', 'domain_wall', 'ansatz', "
            "'gaussian', or 'threshold'."
        )
    snapshot = select_snapshot(m, frame=frame, z_layer=z_layer)
    field, valid = _prepare(snapshot, mask)
    cfg = config or SizeFitConfig()
    if selected == "auto" and config is not None and cfg.method != "auto":
        selected = cfg.method
    selected_convention = convention or XYConvention()
    topo_cfg = SkyrmionTopologyConfig(
        min_contrast=cfg.min_contrast,
        convention=selected_convention,
    )
    values = _center_and_topology(
        field,
        valid,
        float(dx),
        float(dy),
        selected_convention,
        topo_cfg,
    )
    if topology is not None:
        values["center"] = topology.center_xy_m
        values["q_total"] = topology.Q
        values["q_density"] = topology.q_density
    threshold = _threshold_result(values, float(dx), float(dy), cfg, "numpy_crossing")
    threshold.requested_method = selected
    if selected == "threshold":
        return threshold

    r, z, n, max_radius = _profile(
        field[..., 2],
        valid,
        values["x"],
        values["y"],
        values["center"],
        float(dx),
        float(dy),
        cfg,
    )
    contrast = float(values["core_mz"] - values["background"])
    flags = []
    if abs(contrast) < float(cfg.min_contrast):
        flags.append("insufficient_contrast")
        threshold.flags = tuple(sorted(set(threshold.flags + tuple(flags))))
        threshold.quality = "invalid"
        threshold.fit_success = False
        return threshold

    cell = min(abs(float(dx)), abs(float(dy)))
    if r.size < int(cfg.min_profile_bins):
        threshold.flags = tuple(
            sorted(set(threshold.flags + ("insufficient_profile_bins",)))
        )
        threshold.quality = "invalid"
        threshold.fit_success = False
        return threshold

    if selected == "auto":
        kinds = ["domain_wall", "ansatz", "gaussian"]
    else:
        kinds = [selected]

    candidates = []
    candidate_diagnostics: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        params, loss, k = _grid_fit(
            r,
            z,
            n,
            float(values["background"]),
            contrast,
            kind,
            cell,
        )
        if not params:
            candidate_diagnostics[kind] = {"success": False}
            continue

        if kind == "gaussian":
            shape = np.exp(-0.5 * (r / params["sigma"]) ** 2)
            predicted = params["offset"] + params["amplitude"] * shape
            fit_r50 = float(params["sigma"] * np.sqrt(2.0 * np.log(2.0)))
            fit_r90 = float(params["sigma"] * np.sqrt(-2.0 * np.log(0.9)))
            fit_r10 = float(params["sigma"] * np.sqrt(-2.0 * np.log(0.1)))
            fit_width = None
            fit_scale = float(params["sigma"])
        elif kind == "ansatz":
            shape = _ansatz_contrast(r, params["radius"], params["scale"])
            predicted = params["offset"] + params["amplitude"] * shape
            dense_r = np.linspace(0.0, max(max_radius, float(np.max(r))), 4096)
            dense_u = _ansatz_contrast(
                dense_r,
                params["radius"],
                params["scale"],
            )
            fit_r90 = _levels(dense_r, dense_u, 0.9)
            fit_r50 = _levels(dense_r, dense_u, 0.5)
            fit_r10 = _levels(dense_r, dense_u, 0.1)
            fit_r50 = float(params["radius"] if fit_r50 is None else fit_r50)
            fit_width = (
                None if fit_r90 is None or fit_r10 is None else float(fit_r10 - fit_r90)
            )
            fit_scale = float(params["scale"])
        else:
            shape = np.tanh((r - params["radius"]) / params["scale"])
            predicted = params["offset"] + params["amplitude"] * shape
            fit_r50 = float(params["radius"])
            fit_width = float(2.197224577 * params["scale"])
            fit_r90 = fit_r50 - 0.5 * fit_width
            fit_r10 = fit_r50 + 0.5 * fit_width
            fit_scale = float(params["scale"])

        aicc = _aicc(loss, int(r.size), k)
        direct_rmse = float(
            np.sqrt(np.mean((z - predicted) ** 2)) / max(abs(contrast), 1e-30)
        )
        candidate_diagnostics[kind] = {
            "success": True,
            "aicc": float(aicc),
            "normalized_rmse": direct_rmse,
            "parameters": dict(params),
        }
        candidates.append(
            {
                "aicc": float(aicc),
                "kind": kind,
                "params": params,
                "predicted": predicted,
                "r50": float(fit_r50),
                "r90": None if fit_r90 is None else float(fit_r90),
                "r10": None if fit_r10 is None else float(fit_r10),
                "width": fit_width,
                "scale": fit_scale,
                "normalized_rmse": direct_rmse,
            }
        )

    if not candidates:
        threshold.flags = tuple(sorted(set(flags + ["fit_failed"])))
        threshold.quality = "invalid"
        threshold.fit_success = False
        threshold.candidate_diagnostics = candidate_diagnostics
        return threshold

    candidates.sort(key=lambda item: item["aicc"])
    selection_pool = candidates
    if selected == "auto":
        acceptable = [
            candidate
            for candidate in candidates
            if float(candidate["normalized_rmse"]) <= float(cfg.max_normalized_rmse)
        ]
        if acceptable:
            selection_pool = acceptable
    winner = selection_pool[0]
    if selected == "auto" and np.isfinite(float(winner["aicc"])):
        # Delta-AICc <= 2 means the models are statistically indistinguishable.
        # Prefer the physical circular-wall ansatz, then the simpler tanh wall,
        # over the heuristic Gaussian only inside that equivalence band.
        equivalent = [
            candidate
            for candidate in selection_pool
            if float(candidate["aicc"]) <= float(winner["aicc"]) + 2.0
        ]
        for preferred in ("ansatz", "domain_wall", "gaussian"):
            match = next(
                (
                    candidate
                    for candidate in equivalent
                    if candidate["kind"] == preferred
                ),
                None,
            )
            if match is not None:
                winner = match
                break
    chosen = str(winner["kind"])
    params = winner["params"]
    predicted = winner["predicted"]
    normalized_rmse = float(winner["normalized_rmse"])
    if normalized_rmse > float(cfg.max_normalized_rmse):
        flags.append("poor_model_residual")
    if values["center_disagreement"] > 4.0 * cell:
        flags.append("center_methods_disagree")

    radius_50 = (
        float(threshold.radius_50_m)
        if threshold.radius_50_m is not None
        else float(winner["r50"])
    )
    if threshold.radius_50_m is None:
        flags.append("radius_from_fit")
    radius_90 = (
        threshold.radius_90_m if threshold.radius_90_m is not None else winner["r90"]
    )
    radius_10 = (
        threshold.radius_10_m if threshold.radius_10_m is not None else winner["r10"]
    )

    if chosen == "gaussian":
        gaussian_sigma = float(params["sigma"])
        gaussian_fwhm = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * gaussian_sigma)
        domain_scale, wall_width = None, None
    else:
        gaussian_sigma, gaussian_fwhm = None, None
        domain_scale = float(winner["scale"])
        wall_width = (
            float(threshold.wall_width_m)
            if threshold.wall_width_m is not None
            else float(winner["width"])
        )
        if threshold.wall_width_m is None:
            flags.append("wall_width_from_fit")

    xmin = float(np.min(values["x"][valid]))
    xmax = float(np.max(values["x"][valid]))
    ymin = float(np.min(values["y"][valid]))
    ymax = float(np.max(values["y"][valid]))
    centre_x, centre_y = values["center"]
    edge_clearance = min(
        centre_x - xmin,
        xmax - centre_x,
        centre_y - ymin,
        ymax - centre_y,
    )
    if radius_50 + 0.5 * float(wall_width or 0.0) >= edge_clearance:
        flags.append("edge_limited")

    if not np.isfinite(normalized_rmse):
        quality = "invalid"
    elif not flags and normalized_rmse < 0.05:
        quality = "excellent"
    elif (
        normalized_rmse < float(cfg.max_normalized_rmse) and "edge_limited" not in flags
    ):
        quality = "good"
    else:
        quality = "questionable"

    return SkyrmionSizeResult(
        center_xy_m=values["center"],
        radius_m=radius_50,
        diameter_m=2.0 * radius_50,
        wall_width_m=wall_width,
        scale_m=domain_scale,
        radius_90_m=radius_90,
        radius_50_m=radius_50,
        radius_10_m=radius_10,
        sigma_m=gaussian_sigma,
        gaussian_fwhm_m=gaussian_fwhm,
        model=chosen,
        fit_method="numpy_grid_least_squares",
        fit_success=quality != "invalid",
        background_mz=float(values["background"]),
        core_mz=float(values["core_mz"]),
        contrast_mz=contrast,
        normalized_rmse=normalized_rmse,
        aicc=float(winner["aicc"]),
        quality=quality,
        requested_method=selected,
        flags=tuple(sorted(set(flags))),
        radial_r_m=r,
        radial_mz=z,
        model_mz=predicted,
        candidate_diagnostics=candidate_diagnostics,
        metadata={
            "sample_count_per_bin": n,
            "profile_bins": int(r.size),
            "max_profile_radius_m": max_radius,
            "center_disagreement_m": values["center_disagreement"],
            "edge_clearance_m": float(edge_clearance),
            "convention": _y_axis(selected_convention),
            "frame": int(frame),
            "z_layer": int(z_layer),
        },
    )


__all__ = ["detect_skyrmion", "fit_skyrmion_size"]
