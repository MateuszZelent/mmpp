"""High-level plotting and analytical overlays for vortex analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mmpp.analytical import ExternalField

from ._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)
from ._shared.models import TrajectoryResult
from .bridge.extract import AnalyticalParameterResolution, extract_model_defaults
from .nonlinear.slavin_tiberkevich import extract_st_parameters
from .trajectory.filtering import filter_trajectory
from .trajectory.orbit import fit_orbit_ellipse
from .trajectory.steady_state import extract_steady_state

if TYPE_CHECKING:
    from .interface import VortexInterface


def _trajectory_dt(trajectory: TrajectoryResult) -> float:
    time = np.asarray(trajectory.time, dtype=float)
    if time.size >= 2:
        dt = float(np.median(np.diff(time)))
        if np.isfinite(dt) and dt > 0.0:
            return dt
    return 1e-12


def _trajectory_center(trajectory: TrajectoryResult) -> tuple[float, float]:
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)
    return (
        float(np.mean(x)) if x.size else 0.0,
        float(np.mean(y)) if y.size else 0.0,
    )


def _trajectory_physical_coordinates(trajectory: TrajectoryResult) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates in the physical disk frame.

    Analytical overlays may be translated to the numerical center for display.
    Metrics that refer to distance from the disk center must undo that shift.
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


def _infer_disk_radius(vortex_interface: "VortexInterface", params: dict[str, Any] | None = None) -> float | None:
    if params is not None:
        if "R" in params:
            try:
                radius = float(params["R"])
                if np.isfinite(radius) and radius > 0.0:
                    return radius
            except Exception:
                pass
        if "D" in params:
            try:
                diameter = float(params["D"])
                if np.isfinite(diameter) and diameter > 0.0:
                    return 0.5 * diameter
            except Exception:
                pass
        if "Area" in params:
            try:
                area = float(params["Area"])
                if np.isfinite(area) and area > 0.0:
                    return float(np.sqrt(area / np.pi))
            except Exception:
                pass

    attrs = getattr(getattr(vortex_interface, "_job", None), "attrs", {}) or {}
    for key in ("R", "radius"):
        value = attrs.get(key)
        if value is None:
            continue
        try:
            radius = float(value)
        except Exception:
            continue
        if np.isfinite(radius) and radius > 0.0:
            return radius
    for key in ("D", "diameter"):
        value = attrs.get(key)
        if value is None:
            continue
        try:
            diameter = float(value)
        except Exception:
            continue
        if np.isfinite(diameter) and diameter > 0.0:
            return 0.5 * diameter
    value = attrs.get("Area")
    if value is not None:
        try:
            area = float(value)
            if np.isfinite(area) and area > 0.0:
                return float(np.sqrt(area / np.pi))
        except Exception:
            pass
    return None


def _draw_disk_outline(
    ax,
    *,
    radius: float | None,
    center: tuple[float, float] = (0.0, 0.0),
    color: str = "0.35",
    linestyle: str = ":",
    linewidth: float = 1.25,
    label: str = "nanodot radius",
) -> None:
    if radius is None or not np.isfinite(radius) or radius <= 0.0:
        return
    from matplotlib.patches import Circle

    existing_labels = {
        artist.get_label()
        for artist in getattr(ax, "patches", [])
        if hasattr(artist, "get_label")
    }
    patch_label = label if label not in existing_labels else "_nolegend_"
    ax.add_patch(
        Circle(
            (float(center[0]), float(center[1])),
            radius=float(radius),
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.9,
            label=patch_label,
            zorder=0,
        )
    )


def _resolve_plot_trajectory(
    vortex_interface: "VortexInterface",
    trajectory: str | TrajectoryResult = "steady_state",
    *,
    trajectory_kwargs: dict[str, Any] | None = None,
    tracking_source: str = "auto",
    tracking_method: str | None = None,
    tracking_kwargs: dict[str, Any] | None = None,
) -> tuple[TrajectoryResult, str]:
    kwargs = dict(trajectory_kwargs or {})
    track_kwargs = dict(tracking_kwargs or {})
    if isinstance(trajectory, TrajectoryResult):
        return trajectory, str(trajectory.metadata.get("selection", trajectory.method))

    source_token = str(tracking_source).strip().lower()
    if source_token not in {"auto", "table", "magnetization"}:
        raise ValueError("tracking_source must be one of {'auto', 'table', 'magnetization'}")

    method_token = None if tracking_method is None else str(tracking_method).strip().lower()
    if source_token == "table":
        resolved_tracking_method = "table"
    elif source_token == "magnetization":
        if method_token in {None, "", "auto", "table"}:
            cfg_method = str(vortex_interface.config.tracking.method).strip().lower()
            resolved_tracking_method = "gaussian" if cfg_method in {"", "auto", "table"} else cfg_method
        else:
            resolved_tracking_method = method_token
    else:
        resolved_tracking_method = None if method_token in {None, ""} else method_token

    raw = vortex_interface.core.track(method=resolved_tracking_method, **track_kwargs)
    source_label = source_token if source_token != "auto" else str(raw.metadata.get("source", raw.method))

    token = str(trajectory).strip().lower()
    if token == "raw":
        return raw, f"raw/{source_label}"
    if token == "steady_state":
        return extract_steady_state(raw, **kwargs), f"steady_state/{source_label}"
    if token == "filtered":
        method = kwargs.pop("method", None) or vortex_interface.config.trajectory.filter_method
        if "window" not in kwargs:
            kwargs["window"] = vortex_interface.config.trajectory.filter_window
        return filter_trajectory(raw, method=method, **kwargs), f"filtered/{source_label}"
    raise ValueError("trajectory must be one of {'raw', 'steady_state', 'filtered'} or a TrajectoryResult")


def _resolve_tracking_method_for_source(
    vortex_interface: "VortexInterface",
    *,
    tracking_source: str = "auto",
    tracking_method: str | None = None,
) -> str | None:
    source_token = str(tracking_source).strip().lower()
    if source_token not in {"auto", "table", "magnetization"}:
        raise ValueError("tracking_source must be one of {'auto', 'table', 'magnetization'}")

    method_token = None if tracking_method is None else str(tracking_method).strip().lower()
    if source_token == "table":
        return "table"
    if source_token == "magnetization":
        if method_token in {None, "", "auto", "table"}:
            cfg_method = str(vortex_interface.config.tracking.method).strip().lower()
            return "gaussian" if cfg_method in {"", "auto", "table"} else cfg_method
        return method_token
    return None if method_token in {None, ""} else method_token


def _resolve_analytical_initial_state(
    vortex_interface: "VortexInterface",
    numerical: TrajectoryResult,
    params: dict[str, Any],
    *,
    tracking_source: str = "auto",
    tracking_method: str | None = None,
    initial_condition: str = "auto",
) -> tuple[float, float, str]:
    disk_radius = max(float(params["R"]), 1e-18)

    def _fallback_perturbation() -> tuple[float, float, str]:
        return (1e-3 * disk_radius, 0.0, "perturbation")

    def _from_script() -> tuple[float, float, str] | None:
        if "x0" in params or "y0" in params:
            x0_val = float(params.get("x0", 0.0))
            y0_val = float(params.get("y0", 0.0))
            # Guard: origin is an unstable fixed point of the deterministic
            # Thiele ODE — the vortex can never start moving from (0, 0).
            # Fall back to a small perturbation so the orbit can develop.
            if float(np.hypot(x0_val, y0_val)) < 1e-18:
                return _fallback_perturbation()
            return (x0_val, y0_val, "script")
        return None

    def _from_trajectory(traj: TrajectoryResult, *, label: str) -> tuple[float, float, str]:
        center = _trajectory_center(traj)
        rel_x = float(traj.x[0] - center[0]) if np.asarray(traj.x).size else 0.0
        rel_y = float(traj.y[0] - center[1]) if np.asarray(traj.y).size else 0.0
        if float(np.hypot(rel_x, rel_y)) < 1e-18:
            return _fallback_perturbation()
        return (rel_x, rel_y, label)

    token = str(initial_condition).strip().lower()
    if token not in {"auto", "script", "trajectory", "raw", "perturbation"}:
        raise ValueError(
            "initial_condition must be one of "
            "{'auto', 'script', 'trajectory', 'raw', 'perturbation'}"
        )

    if token == "script":
        resolved = _from_script()
        if resolved is None:
            raise ValueError(
                "initial_condition='script' requested, but x0/y0 were not resolved "
                "from attrs/.mx3/params."
            )
        return resolved

    if token == "trajectory":
        return _from_trajectory(numerical, label="trajectory")

    if token == "raw":
        method = _resolve_tracking_method_for_source(
            vortex_interface,
            tracking_source=tracking_source,
            tracking_method=tracking_method,
        )
        raw = vortex_interface.core.track(method=method)
        return _from_trajectory(raw, label="raw")

    if token == "perturbation":
        return _fallback_perturbation()

    resolved_script = _from_script()
    if resolved_script is not None:
        return resolved_script
    if bool(numerical.metadata.get("steady_state")) or "steady_state" in str(numerical.method).lower():
        try:
            return _resolve_analytical_initial_state(
                vortex_interface,
                numerical,
                params,
                tracking_source=tracking_source,
                tracking_method=tracking_method,
                initial_condition="raw",
            )
        except Exception:
            return _fallback_perturbation()
    return _from_trajectory(numerical, label="trajectory")


def _translate_trajectory(
    trajectory: TrajectoryResult,
    *,
    shift: tuple[float, float],
    method_suffix: str = "",
    metadata: dict[str, Any] | None = None,
) -> TrajectoryResult:
    meta = dict(trajectory.metadata)
    if metadata:
        meta.update(metadata)
    return TrajectoryResult(
        time=np.asarray(trajectory.time, dtype=float),
        x=np.asarray(trajectory.x, dtype=float) + float(shift[0]),
        y=np.asarray(trajectory.y, dtype=float) + float(shift[1]),
        polarity=np.asarray(trajectory.polarity, dtype=int),
        method=f"{trajectory.method}{method_suffix}",
        confidence=np.asarray(trajectory.confidence, dtype=float),
        metadata=meta,
    )


def _resample_trajectory_to_reference(
    trajectory: TrajectoryResult,
    reference_time: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
) -> TrajectoryResult:
    ref_t = np.asarray(reference_time, dtype=float).reshape(-1)
    src_t = np.asarray(trajectory.time, dtype=float).reshape(-1)
    if ref_t.size == 0 or src_t.size == 0:
        return TrajectoryResult(
            time=ref_t,
            x=np.zeros_like(ref_t, dtype=float),
            y=np.zeros_like(ref_t, dtype=float),
            polarity=np.full(ref_t.shape, 1, dtype=int),
            method=f"{trajectory.method}+resampled",
            confidence=np.ones_like(ref_t, dtype=float),
            metadata={**dict(trajectory.metadata), **dict(metadata or {})},
        )

    dt_ref = float(np.median(np.diff(ref_t))) if ref_t.size >= 2 else 0.0
    if src_t.size == ref_t.size and np.allclose(src_t, ref_t, rtol=0.0, atol=max(dt_ref * 1e-6, 1e-18)):
        meta = dict(trajectory.metadata)
        if metadata:
            meta.update(metadata)
        return TrajectoryResult(
            time=ref_t,
            x=np.asarray(trajectory.x, dtype=float),
            y=np.asarray(trajectory.y, dtype=float),
            polarity=np.asarray(trajectory.polarity, dtype=int),
            method=trajectory.method,
            confidence=np.asarray(trajectory.confidence, dtype=float),
            metadata=meta,
        )

    x = np.interp(ref_t, src_t, np.asarray(trajectory.x, dtype=float))
    y = np.interp(ref_t, src_t, np.asarray(trajectory.y, dtype=float))
    confidence = np.interp(
        ref_t,
        src_t,
        np.asarray(trajectory.confidence, dtype=float),
        left=float(np.asarray(trajectory.confidence, dtype=float)[0]),
        right=float(np.asarray(trajectory.confidence, dtype=float)[-1]),
    )
    polarity_src = np.asarray(trajectory.polarity, dtype=float)
    polarity_val = 1 if polarity_src.size == 0 or float(np.mean(polarity_src)) >= 0.0 else -1
    meta = dict(trajectory.metadata)
    meta.update(
        {
            "resampled_to_reference": True,
            "reference_n_samples": int(ref_t.size),
        }
    )
    if metadata:
        meta.update(metadata)
    return TrajectoryResult(
        time=ref_t,
        x=x,
        y=y,
        polarity=np.full(ref_t.shape, polarity_val, dtype=int),
        method=f"{trajectory.method}+resampled",
        confidence=confidence,
        metadata=meta,
    )


def _tail_window_to_reference(
    trajectory: TrajectoryResult,
    reference_time: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
) -> TrajectoryResult:
    ref_t = np.asarray(reference_time, dtype=float).reshape(-1)
    n_ref = int(ref_t.size)
    if n_ref == 0:
        return _resample_trajectory_to_reference(trajectory, ref_t, metadata=metadata)

    src_t = np.asarray(trajectory.time, dtype=float).reshape(-1)
    if src_t.size < n_ref:
        return _resample_trajectory_to_reference(trajectory, ref_t, metadata=metadata)

    meta = dict(trajectory.metadata)
    meta.update(
        {
            "comparison_window": "tail",
            "tail_window_n_samples": n_ref,
        }
    )
    if metadata:
        meta.update(metadata)
    return TrajectoryResult(
        time=ref_t,
        x=np.asarray(trajectory.x, dtype=float)[-n_ref:],
        y=np.asarray(trajectory.y, dtype=float)[-n_ref:],
        polarity=np.asarray(trajectory.polarity, dtype=int)[-n_ref:],
        method=f"{trajectory.method}+tail",
        confidence=np.asarray(trajectory.confidence, dtype=float)[-n_ref:],
        metadata=meta,
    )


def _estimate_analytical_burn_in_duration(
    adapter: Any,
    *,
    resolution: AnalyticalParameterResolution,
    current_density: Any,
    rel_x: float,
    rel_y: float,
    window_duration: float,
    dt: float,
    initial_condition_source: str,
    numerical: TrajectoryResult,
) -> float:
    if initial_condition_source not in {"perturbation", "script", "raw"}:
        return 0.0
    if not bool(numerical.metadata.get("steady_state")):
        return 0.0
    if callable(current_density):
        return 0.0

    duration = max(float(window_duration), float(dt))
    model_kind = str(resolution.model_kind).strip().lower()
    model = getattr(adapter, "model", None)
    if model_kind != "cpp" or model is None:
        return 5.0 * duration

    try:
        J = float(current_density)
        disk_radius = max(float(resolution.resolved_params["R"]), 1e-18)
        u_init = max(float(np.hypot(rel_x, rel_y) / disk_radius), 1e-6)
        u_target = model.steady_state_u(J, allow_edge=True)
        if u_target is None:
            return 0.0
        u_target = float(np.clip(u_target, u_init * 1.05, 0.98))
        growth = float(model.chi(J) - model.d(u_init) * model.omega(u_init, J))
        if not np.isfinite(growth) or growth <= 0.0:
            return max(3.0 * duration, 0.0)
        est = float(np.log(max(u_target / max(u_init, 1e-12), 1.0001)) / growth)
        freq_hz = model.predict_frequency_dc(J, allow_edge=True)
        periods_term = 0.0
        if freq_hz is not None and np.isfinite(freq_hz) and float(freq_hz) > 0.0:
            periods_term = 6.0 / float(freq_hz)
        burn_in = max(3.0 * duration, 1.5 * est, periods_term)
        return max(float(burn_in), 0.0)
    except Exception:
        return 3.0 * duration


def _coerce_field(value: Any) -> ExternalField | None:
    if value is None:
        return None
    return ExternalField.from_any(value)


def _resolve_model_alignment_center(
    adapter: Any,
    trajectory: TrajectoryResult,
) -> tuple[float, float]:
    model = getattr(adapter, "model", None)
    if model is None or not hasattr(model, "s_eq"):
        return _trajectory_center(trajectory)

    try:
        field_state = getattr(model, "field", None)
        s_eq = np.asarray(model.s_eq(field_state=field_state), dtype=float).reshape(-1)
        if s_eq.size < 2:
            return _trajectory_center(trajectory)
        geom = getattr(model, "geom", None)
        disk_radius = float(getattr(geom, "R", 1.0))
        center = (float(s_eq[0] * disk_radius), float(s_eq[1] * disk_radius))
        if np.all(np.isfinite(center)):
            return center
    except Exception:
        pass
    return _trajectory_center(trajectory)


def _build_model_adapter(
    vortex_interface: "VortexInterface",
    resolution: AnalyticalParameterResolution,
):
    params = resolution.resolved_params
    material = {
        "Ms": float(params["Ms"]),
        "alpha": float(params["alpha"]),
        "P": float(params.get("P_model", params["P"])),
        "A": float(params.get("A", 1.3e-11)),
    }
    geom = {
        "R": float(params["R"]),
        "L": float(params["L"]),
    }
    field = _coerce_field(params.get("field"))

    if resolution.model_kind == "cpp":
        return vortex_interface.model.thiele.cpp(
            material=material,
            geom=geom,
            polarity=int(np.sign(float(params.get("polarity", 1))) or 1),
            omega0=float(params["omega0"]),
            N=float(params.get("N", 0.25)),
            domega0_dJ=float(params.get("domega0_dJ", 0.0)),
            field=field,
            chi_scale=float(params.get("chi_scale", 1.0)),
            torque_thickness=float(params.get("L_stt", params["L"])),
        )

    return vortex_interface.model.thiele.cip(
        material=material,
        geom=geom,
        polarity=int(np.sign(float(params.get("polarity", 1))) or 1),
        omega0=float(params["omega0"]),
        current_dir=tuple(params["current_dir"]),
        field=field,
    )


def _simulate_matching_trajectory(
    vortex_interface: "VortexInterface",
    numerical: TrajectoryResult,
    resolution: AnalyticalParameterResolution,
    *,
    tracking_source: str = "auto",
    tracking_method: str | None = None,
    initial_condition: str = "auto",
) -> tuple[TrajectoryResult, tuple[float, float], tuple[float, float]]:
    adapter = _build_model_adapter(vortex_interface, resolution)
    params = resolution.resolved_params

    center = _trajectory_center(numerical)
    disk_radius = max(float(params["R"]), 1e-18)
    rel_x, rel_y, init_source = _resolve_analytical_initial_state(
        vortex_interface,
        numerical,
        params,
        tracking_source=tracking_source,
        tracking_method=tracking_method,
        initial_condition=initial_condition,
    )

    current_density = params["current_density"]
    dt = _trajectory_dt(numerical)
    time = np.asarray(numerical.time, dtype=float)
    t0 = float(time[0]) if time.size else 0.0
    t1 = float(time[-1]) if time.size >= 2 else (t0 + dt)
    nominal_end = t0 + dt * max(int(time.size) - 1, 1)
    window_duration = max(float(t1 - t0), dt * max(int(time.size) - 1, 1), dt)
    burn_in_duration = _estimate_analytical_burn_in_duration(
        adapter,
        resolution=resolution,
        current_density=current_density,
        rel_x=rel_x,
        rel_y=rel_y,
        window_duration=window_duration,
        dt=dt,
        initial_condition_source=init_source,
        numerical=numerical,
    )
    sim_t0 = t0
    if burn_in_duration > 0.0:
        # For DC comparisons the analytical model can be safely prerun on a
        # synthetic pre-history. Clamping to t=0 truncates the burn-in and
        # biases near-threshold cases toward a false "stuck at the origin"
        # result when the numerical steady-state window starts too early.
        sim_t0 = float(t0 - burn_in_duration)
    sim_t1 = max(t1, nominal_end) + dt

    if resolution.model_kind == "cpp":
        analytical_full = adapter.simulate(
            t_span=(sim_t0, sim_t1),
            J_func=current_density,
            dt=dt,
            s0=(rel_x / disk_radius, rel_y / disk_radius),
        )
    else:
        analytical_full = adapter.simulate(
            t_span=(sim_t0, sim_t1),
            J_func=current_density,
            dt=dt,
            r0=(rel_x, rel_y),
        )

    analytical = _resample_trajectory_to_reference(
        analytical_full,
        time,
        metadata={
            "comparison_window": "reference",
            "reference_time_start": t0,
            "reference_time_end": t1,
            "simulation_time_start": sim_t0,
            "simulation_time_end": sim_t1,
            "burn_in_duration": float(max(t0 - sim_t0, 0.0)),
            "burn_in_applied": bool(sim_t0 < t0),
        },
    )

    raw_center = _trajectory_center(analytical)
    alignment_center = _resolve_model_alignment_center(adapter, analytical)
    shift = (center[0] - alignment_center[0], center[1] - alignment_center[1])
    aligned = _translate_trajectory(
        analytical,
        shift=shift,
        method_suffix="+aligned",
        metadata={
            "aligned_to_center": center,
            "alignment_shift": shift,
            "raw_center": raw_center,
            "alignment_reference_center": alignment_center,
            "comparison_model_kind": resolution.model_kind,
            "initial_condition_source": init_source,
            "initial_rel_x": float(rel_x),
            "initial_rel_y": float(rel_y),
            "simulation_time_start": sim_t0,
            "simulation_time_end": sim_t1,
            "burn_in_duration": float(max(t0 - sim_t0, 0.0)),
            "burn_in_applied": bool(sim_t0 < t0),
        },
    )
    return aligned, raw_center, alignment_center


def _resample_like(reference_time: np.ndarray, candidate: np.ndarray, candidate_time: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_time, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    cand_t = np.asarray(candidate_time, dtype=float)
    if ref.size == 0 or cand.size == 0 or cand_t.size == 0:
        return np.zeros_like(ref, dtype=float)
    if cand.size == ref.size and np.allclose(cand_t, ref, rtol=0.0, atol=max(_trajectory_dt_dummy(ref), 1e-18)):
        return cand
    return np.interp(ref, cand_t, cand, left=cand[0], right=cand[-1])


def _trajectory_dt_dummy(time: np.ndarray) -> float:
    if time.size >= 2:
        return float(np.median(np.diff(time)))
    return 0.0


@dataclass(frozen=True)
class VortexAnalyticalMetrics:
    """Comparison metrics between numerical and analytical vortex trajectories."""

    delta_radius_mean: float
    delta_radius_max: float
    delta_core_distance_mean: float
    delta_core_distance_max: float
    delta_freq_mean: float
    delta_center: float
    delta_eccentricity: float
    rms_xy_residual: float
    # Absolute values for summary display
    numerical_radius_nm: float = 0.0
    analytical_radius_nm: float = 0.0
    numerical_core_distance_nm: float = 0.0
    analytical_core_distance_nm: float = 0.0
    numerical_core_distance_max_nm: float = 0.0
    analytical_core_distance_max_nm: float = 0.0
    numerical_freq_ghz: float = 0.0
    analytical_freq_ghz: float = 0.0


@dataclass(frozen=True)
class VortexSTComparison:
    """Numerical vs analytical Slavin-Tiberkevich comparison."""

    numerical: Any
    analytical: Any
    delta: dict[str, float]


@dataclass(frozen=True)
class VortexForceBalanceComparison:
    """Numerical vs analytical Thiele-force decomposition comparison."""

    numerical: Any
    analytical: Any
    delta: dict[str, float]


def _compute_metrics(
    numerical: TrajectoryResult,
    analytical: TrajectoryResult,
    *,
    analytical_center_reference: tuple[float, float],
) -> VortexAnalyticalMetrics:
    numerical_fit = fit_orbit_ellipse(numerical)
    analytical_fit = fit_orbit_ellipse(analytical)

    ref_t = np.asarray(numerical.time, dtype=float)
    ana_x = _resample_like(ref_t, analytical.x, analytical.time)
    ana_y = _resample_like(ref_t, analytical.y, analytical.time)

    residual = np.sqrt(
        np.mean(
            (np.asarray(numerical.x, dtype=float) - ana_x) ** 2
            + (np.asarray(numerical.y, dtype=float) - ana_y) ** 2
        )
    )

    freq_num = float(np.mean(np.abs(np.asarray(numerical.instantaneous_frequency, dtype=float))) / (2.0 * np.pi))
    freq_ana = float(np.mean(np.abs(np.asarray(analytical.instantaneous_frequency, dtype=float))) / (2.0 * np.pi))
    num_center = _trajectory_center(numerical)

    radius_num = float(np.mean(numerical.r))
    radius_ana = float(np.mean(analytical.r))
    num_x_phys, num_y_phys = _trajectory_physical_coordinates(numerical)
    ana_x_phys, ana_y_phys = _trajectory_physical_coordinates(analytical)
    core_distance_num = np.hypot(num_x_phys, num_y_phys)
    core_distance_ana = np.hypot(ana_x_phys, ana_y_phys)
    core_distance_num_mean = float(np.mean(core_distance_num)) if core_distance_num.size else 0.0
    core_distance_ana_mean = float(np.mean(core_distance_ana)) if core_distance_ana.size else 0.0
    core_distance_num_max = float(np.max(core_distance_num)) if core_distance_num.size else 0.0
    core_distance_ana_max = float(np.max(core_distance_ana)) if core_distance_ana.size else 0.0

    return VortexAnalyticalMetrics(
        delta_radius_mean=abs(radius_num - radius_ana),
        delta_radius_max=abs(float(np.max(numerical.r)) - float(np.max(analytical.r))),
        delta_core_distance_mean=abs(core_distance_num_mean - core_distance_ana_mean),
        delta_core_distance_max=abs(core_distance_num_max - core_distance_ana_max),
        delta_freq_mean=abs(freq_num - freq_ana),
        delta_center=float(
            np.hypot(
                num_center[0] - analytical_center_reference[0],
                num_center[1] - analytical_center_reference[1],
            )
        ),
        delta_eccentricity=abs(float(numerical_fit.eccentricity) - float(analytical_fit.eccentricity)),
        rms_xy_residual=float(residual),
        numerical_radius_nm=radius_num * 1e9,
        analytical_radius_nm=radius_ana * 1e9,
        numerical_core_distance_nm=core_distance_num_mean * 1e9,
        analytical_core_distance_nm=core_distance_ana_mean * 1e9,
        numerical_core_distance_max_nm=core_distance_num_max * 1e9,
        analytical_core_distance_max_nm=core_distance_ana_max * 1e9,
        numerical_freq_ghz=freq_num * 1e-9,
        analytical_freq_ghz=freq_ana * 1e-9,
    )


def _compute_st_comparison(numerical: TrajectoryResult, analytical: TrajectoryResult) -> VortexSTComparison:
    numerical_st = extract_st_parameters(numerical)
    analytical_st = extract_st_parameters(analytical)
    delta = {
        "f_0_ghz": abs(float(numerical_st.f_0_ghz) - float(analytical_st.f_0_ghz)),
        "N": abs(float(numerical_st.N) - float(analytical_st.N)),
        "Gamma_G": abs(float(numerical_st.Gamma_G) - float(analytical_st.Gamma_G)),
        "Q": abs(float(numerical_st.Q) - float(analytical_st.Q)),
        "generation_power": abs(
            float(numerical_st.generation_power) - float(analytical_st.generation_power)
        ),
        "linewidth_hz": abs(float(numerical_st.linewidth_hz) - float(analytical_st.linewidth_hz)),
    }
    return VortexSTComparison(
        numerical=numerical_st,
        analytical=analytical_st,
        delta=delta,
    )


class VortexAnalyticalComparisonPlotAccessor:
    """Plot helpers for :class:`VortexAnalyticalComparison`."""

    def __init__(self, comparison: "VortexAnalyticalComparison"):
        self._comparison = comparison

    def orbit_overlay(self, *, ax=None, show_centers: bool = True, **kwargs):
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, default_figsize=(5.2, 4.6), figure_kwargs=figure_kwargs)
        self._comparison._draw_overlay(
            ax=ax,
            draw_numerical=True,
            show_centers=show_centers,
            analytical_kwargs=plot_kwargs,
        )
        apply_axes_style(ax, style_kwargs)
        return ax

    def time_traces(self, *, fig=None):
        import matplotlib.pyplot as plt

        if fig is None:
            fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=110)
        else:
            axes = fig.subplots(2, 2)

        numerical = self._comparison.numerical
        analytical = self._comparison.analytical

        axes[0, 0].plot(numerical.time, numerical.x, label="numerical")
        axes[0, 0].plot(analytical.time, analytical.x, "--", label="analytical")
        axes[0, 0].set_title("x(t)")
        axes[0, 0].set_xlabel("Time [s]")
        axes[0, 0].set_ylabel("x [m]")
        axes[0, 0].legend()

        axes[0, 1].plot(numerical.time, numerical.y, label="numerical")
        axes[0, 1].plot(analytical.time, analytical.y, "--", label="analytical")
        axes[0, 1].set_title("y(t)")
        axes[0, 1].set_xlabel("Time [s]")
        axes[0, 1].set_ylabel("y [m]")
        axes[0, 1].legend()

        axes[1, 0].plot(numerical.time, numerical.r * 1e9, label="numerical")
        axes[1, 0].plot(analytical.time, analytical.r * 1e9, "--", label="analytical")
        axes[1, 0].set_title("r(t)")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].set_ylabel("r [nm]")
        axes[1, 0].legend()

        freq_num = np.asarray(numerical.instantaneous_frequency, dtype=float) / (2.0 * np.pi * 1e9)
        freq_ana = np.asarray(analytical.instantaneous_frequency, dtype=float) / (2.0 * np.pi * 1e9)
        axes[1, 1].plot(numerical.time, freq_num, label="numerical")
        axes[1, 1].plot(analytical.time, freq_ana, "--", label="analytical")
        axes[1, 1].set_title("f(t)")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("Frequency [GHz]")
        axes[1, 1].legend()

        fig.tight_layout()
        return fig, axes

    def st_dashboard(self, *, fig=None):
        import matplotlib.pyplot as plt

        if fig is None:
            fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), dpi=110)
        else:
            axes = fig.subplots(2, 2)

        st = self._comparison.st
        entries = [
            ("f0 [GHz]", float(st.numerical.f_0_ghz), float(st.analytical.f_0_ghz)),
            ("Linewidth [MHz]", float(st.numerical.linewidth_hz) * 1e-6, float(st.analytical.linewidth_hz) * 1e-6),
            ("Power [a.u.]", float(st.numerical.generation_power), float(st.analytical.generation_power)),
            ("N [rad/s]", float(st.numerical.N), float(st.analytical.N)),
        ]
        flat_axes = axes.reshape(-1)
        for axis, (title, numerical_value, analytical_value) in zip(flat_axes, entries):
            axis.bar(["numerical", "analytical"], [numerical_value, analytical_value], color=["#1d4ed8", "#dc2626"])
            axis.set_title(title)
        fig.tight_layout()
        return fig, axes

    def force_balance(self, *, fig=None, as_norm: bool = True):
        import matplotlib.pyplot as plt

        balance = self._comparison.force_balance
        if fig is None:
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=110)
        else:
            axes = fig.subplots(1, 2)
        balance.numerical.plt.force_balance(ax=axes[0], as_norm=as_norm)
        axes[0].set_title("Numerical force balance")
        balance.analytical.plt.force_balance(ax=axes[1], as_norm=as_norm)
        axes[1].set_title("Analytical force balance")
        fig.tight_layout()
        return fig, axes


class VortexAnalyticalComparison:
    """Single-simulation comparison between numerical and analytical vortex motion."""

    def __init__(
        self,
        *,
        vortex_interface: "VortexInterface",
        numerical: TrajectoryResult,
        analytical: TrajectoryResult,
        resolution: AnalyticalParameterResolution,
        analytical_raw_center: tuple[float, float],
        analytical_center_reference: tuple[float, float],
        trajectory_source: str,
    ):
        self._vortex_interface = vortex_interface
        self.numerical = numerical
        self.analytical = analytical
        self.resolved_params = dict(resolution.resolved_params)
        self.param_sources = dict(resolution.param_sources)
        self.model_kind = resolution.model_kind
        self.trajectory_source = trajectory_source
        self.metrics = _compute_metrics(
            numerical,
            analytical,
            analytical_center_reference=analytical_center_reference,
        )
        self.st = _compute_st_comparison(numerical, analytical)
        self._analytical_raw_center = analytical_raw_center
        self._analytical_center_reference = analytical_center_reference
        self._force_balance_cache: VortexForceBalanceComparison | None = None
        self._autofit_result: Any = None  # VortexAutofitResult, set externally when fit=True

    @property
    def plt(self):
        return VortexAnalyticalComparisonPlotAccessor(self)

    @property
    def plot(self):
        return self.plt

    @property
    def autofit_result(self):
        """Autofit result, if this comparison was produced via ``fit=True``."""
        return self._autofit_result

    @property
    def force_balance(self) -> VortexForceBalanceComparison:
        if self._force_balance_cache is None:
            shared_kwargs = {
                "polarity": int(np.sign(float(self.resolved_params.get("polarity", 1))) or 1),
                "Ms": float(self.resolved_params["Ms"]),
                "thickness": float(self.resolved_params["L"]),
                "alpha": float(self.resolved_params["alpha"]),
            }
            numerical = self._vortex_interface.nonlinear.thiele.force_balance(
                trajectory=self.numerical,
                **shared_kwargs,
            )
            analytical = self._vortex_interface.nonlinear.thiele.force_balance(
                trajectory=self.analytical,
                **shared_kwargs,
            )
            delta = {
                "residual_ratio_mean": abs(
                    float(np.mean(numerical.residual_ratio)) - float(np.mean(analytical.residual_ratio))
                ),
                "residual_norm_mean": abs(
                    float(np.mean(numerical.residual_norm)) - float(np.mean(analytical.residual_norm))
                ),
                "gyro_norm_mean": abs(
                    float(np.mean(numerical.gyro_norm)) - float(np.mean(analytical.gyro_norm))
                ),
            }
            self._force_balance_cache = VortexForceBalanceComparison(
                numerical=numerical,
                analytical=analytical,
                delta=delta,
            )
        return self._force_balance_cache

    def _metrics_text(self) -> str:
        m = self.metrics
        lines = [
            f"Numerical\n"
            f"  orbit: {m.numerical_radius_nm:.1f} nm\n"
            f"  core:  {m.numerical_core_distance_nm:.1f} nm\n"
            f"  freq:  {m.numerical_freq_ghz:.4f} GHz\n"
            f"Analytical ({self.model_kind.upper()})\n"
            f"  orbit: {m.analytical_radius_nm:.1f} nm\n"
            f"  core:  {m.analytical_core_distance_nm:.1f} nm\n"
            f"  freq:  {m.analytical_freq_ghz:.4f} GHz\n"
            f"Difference\n"
            f"  \u0394f={m.delta_freq_mean * 1e-9:.4f} GHz\n"
            f"  \u0394orbit={m.delta_radius_mean * 1e9:.1f} nm\n"
            f"  \u0394core={m.delta_core_distance_mean * 1e9:.1f} nm\n"
            f"  RMS={m.rms_xy_residual * 1e9:.1f} nm"
        ]
        if bool(self.analytical.metadata.get("edge_limited")):
            hit_time = self.analytical.metadata.get("edge_hit_time")
            if hit_time is None:
                lines.append("State\n  edge-limited")
            else:
                lines.append(f"State\n  edge-limited @ {float(hit_time) * 1e9:.2f} ns")
        return "\n".join(lines)

    def _draw_overlay(
        self,
        *,
        ax,
        draw_numerical: bool,
        show_centers: bool,
        analytical_kwargs: dict[str, Any] | None = None,
    ) -> None:
        plot_kwargs = dict(analytical_kwargs or {})
        show_disk_radius = bool(plot_kwargs.pop("show_disk_radius", True))
        disk_radius = plot_kwargs.pop("disk_radius", None)
        if disk_radius is None:
            disk_radius = _infer_disk_radius(self._vortex_interface, self.resolved_params)
        disk_center = plot_kwargs.pop("disk_center", (0.0, 0.0))
        disk_color = plot_kwargs.pop("disk_color", "0.35")
        disk_linestyle = plot_kwargs.pop("disk_linestyle", ":")
        disk_linewidth = float(plot_kwargs.pop("disk_linewidth", 1.25))
        numerical_kwargs = {}
        if draw_numerical:
            numerical_kwargs = {
                "color": plot_kwargs.pop("numerical_color", "#1d4ed8"),
                "label": plot_kwargs.pop("numerical_label", "numerical"),
                "linewidth": plot_kwargs.pop("numerical_linewidth", 1.8),
            }
            ax.plot(self.numerical.x, self.numerical.y, **numerical_kwargs)

        analytical_style = {
            "color": plot_kwargs.pop("color", "#dc2626"),
            "linestyle": plot_kwargs.pop("linestyle", "--"),
            "linewidth": plot_kwargs.pop("linewidth", 1.8),
            "label": plot_kwargs.pop("label", f"analytical ({self.model_kind})"),
        }
        analytical_style.update(plot_kwargs)

        if show_disk_radius:
            _draw_disk_outline(
                ax,
                radius=disk_radius,
                center=disk_center,
                color=disk_color,
                linestyle=disk_linestyle,
                linewidth=disk_linewidth,
            )

        ax.plot(self.analytical.x, self.analytical.y, **analytical_style)

        if show_centers:
            num_center = _trajectory_center(self.numerical)
            ana_center = _trajectory_center(self.analytical)
            ax.scatter([num_center[0]], [num_center[1]], color=numerical_kwargs.get("color", "#1d4ed8"), s=18)
            ax.scatter([ana_center[0]], [ana_center[1]], color=analytical_style["color"], s=18, marker="x")

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Vortex orbit: numerical vs analytical")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.text(
            0.02,
            0.98,
            self._metrics_text(),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )


class VortexOrbitPlotHandle:
    """Plot handle returned by :meth:`VortexPlotInterface.orbit`."""

    def __init__(
        self,
        *,
        fig,
        ax,
        trajectory: TrajectoryResult,
        trajectory_source: str,
        tracking_source: str,
        tracking_method: str | None,
        vortex_interface: "VortexInterface",
    ):
        self.fig = fig
        self.ax = ax
        self.trajectory = trajectory
        self.trajectory_source = trajectory_source
        self.tracking_source = tracking_source
        self.tracking_method = tracking_method
        self.vortex_interface = vortex_interface
        self._comparison = None

    @property
    def comparison(self) -> VortexAnalyticalComparison | None:
        return self._comparison

    def add_analytics(
        self,
        *,
        params: str | dict[str, Any] = "auto",
        model: str = "auto",
        current: Any = None,
        param_keys: dict[str, Any] | None = None,
        initial_condition: str = "auto",
        fit: bool | str | dict[str, Any] = False,
        **overrides,
    ) -> VortexAnalyticalComparison:
        # If fit requested, delegate to autofit and use its comparison
        if fit is not False and fit is not None:
            from .autofit import AutofitConfig
            from .autofit.single import run_single_job_fit

            fit_kwargs: dict[str, Any] = {
                "trajectory": self.trajectory,
                "tracking_source": self.tracking_source,
                "tracking_method": self.tracking_method,
                "model": model,
                "params": params,
                "current": current,
                "initial_condition": initial_condition,
            }
            if isinstance(fit, dict):
                fit_kwargs.update(fit)

            config = AutofitConfig(**fit_kwargs)
            result = run_single_job_fit(self.vortex_interface, config)
            comparison = result.comparison
            comparison._autofit_result = result

            # Draw the fitted overlay
            comparison._draw_overlay(
                ax=self.ax,
                draw_numerical=False,
                show_centers=True,
            )
            self._comparison = comparison
            return comparison

        resolution = extract_model_defaults(
            vortex_interface=self.vortex_interface,
            trajectory=self.trajectory,
            params=params,
            model=model,
            current=current,
            param_keys=param_keys,
            **overrides,
        )
        analytical, raw_center, alignment_center = _simulate_matching_trajectory(
            self.vortex_interface,
            self.trajectory,
            resolution,
            tracking_source=self.tracking_source,
            tracking_method=self.tracking_method,
            initial_condition=initial_condition,
        )
        comparison = VortexAnalyticalComparison(
            vortex_interface=self.vortex_interface,
            numerical=self.trajectory,
            analytical=analytical,
            resolution=resolution,
            analytical_raw_center=raw_center,
            analytical_center_reference=alignment_center,
            trajectory_source=self.trajectory_source,
        )
        comparison._draw_overlay(
            ax=self.ax,
            draw_numerical=False,
            show_centers=True,
        )
        self._comparison = comparison
        return comparison

    def add_analitics(self, **kwargs) -> VortexAnalyticalComparison:
        """Backward-compatible typo alias for ``add_analytics``."""
        return self.add_analytics(**kwargs)


class VortexPlotInterface:
    """High-level plotting namespace for a single vortex job."""

    def __init__(self, vortex_interface: "VortexInterface"):
        self._interface = vortex_interface

    def orbit(
        self,
        *,
        trajectory: str | TrajectoryResult = "steady_state",
        trajectory_kwargs: dict[str, Any] | None = None,
        tracking_source: str = "auto",
        tracking_method: str | None = None,
        tracking_kwargs: dict[str, Any] | None = None,
        ax=None,
        show_center: bool = True,
        **kwargs,
    ) -> VortexOrbitPlotHandle:
        plot_kwargs = dict(kwargs)
        show_disk_radius = bool(plot_kwargs.pop("show_disk_radius", True))
        disk_radius = plot_kwargs.pop("disk_radius", None)
        if disk_radius is None:
            disk_radius = _infer_disk_radius(self._interface)
        disk_center = plot_kwargs.pop("disk_center", (0.0, 0.0))
        disk_color = plot_kwargs.pop("disk_color", "0.35")
        disk_linestyle = plot_kwargs.pop("disk_linestyle", ":")
        disk_linewidth = float(plot_kwargs.pop("disk_linewidth", 1.25))
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, default_figsize=(5.2, 4.6), figure_kwargs=figure_kwargs)

        resolved_trajectory, trajectory_source = _resolve_plot_trajectory(
            self._interface,
            trajectory,
            trajectory_kwargs=trajectory_kwargs,
            tracking_source=tracking_source,
            tracking_method=tracking_method,
            tracking_kwargs=tracking_kwargs,
        )
        line_kwargs = {
            "color": plot_kwargs.pop("color", "#1d4ed8"),
            "linewidth": plot_kwargs.pop("linewidth", 1.8),
            "label": plot_kwargs.pop("label", "numerical"),
        }
        line_kwargs.update(plot_kwargs)
        if show_disk_radius:
            _draw_disk_outline(
                ax,
                radius=disk_radius,
                center=disk_center,
                color=disk_color,
                linestyle=disk_linestyle,
                linewidth=disk_linewidth,
            )
        ax.plot(resolved_trajectory.x, resolved_trajectory.y, **line_kwargs)

        if show_center:
            center_x, center_y = _trajectory_center(resolved_trajectory)
            ax.scatter([center_x], [center_y], color=line_kwargs["color"], s=18, marker="o", label="center")
            ax.legend()

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(f"Vortex orbit ({trajectory_source})")
        ax.set_aspect("equal")
        apply_axes_style(ax, style_kwargs)

        return VortexOrbitPlotHandle(
            fig=ax.figure,
            ax=ax,
            trajectory=resolved_trajectory,
            trajectory_source=trajectory_source,
            tracking_source=tracking_source,
            tracking_method=tracking_method,
            vortex_interface=self._interface,
        )


__all__ = [
    "VortexPlotInterface",
    "VortexOrbitPlotHandle",
    "VortexAnalyticalComparison",
    "VortexAnalyticalComparisonPlotAccessor",
    "VortexAnalyticalMetrics",
    "VortexSTComparison",
    "VortexForceBalanceComparison",
]
