"""Numerical trajectory operations shared by solvers and plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._shared.models import TrajectoryResult

if TYPE_CHECKING:
    from ..interface import VortexInterface


def _trajectory_dt(trajectory: TrajectoryResult) -> float:
    time = np.asarray(trajectory.time, dtype=float)
    if time.size < 2:
        raise ValueError(
            "At least two trajectory timestamps are required to infer a sampling interval"
        )
    intervals = np.diff(time)
    if not np.all(np.isfinite(time)) or np.any(intervals <= 0.0):
        raise ValueError("Trajectory timestamps must be finite and strictly increasing")
    dt = float(np.median(intervals))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Trajectory sampling interval must be finite and positive")
    return dt


def _trajectory_center(trajectory: TrajectoryResult) -> tuple[float, float]:
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)
    return (
        float(np.mean(x)) if x.size else 0.0,
        float(np.mean(y)) if y.size else 0.0,
    )


def _resolve_tracking_method_for_source(
    vortex_interface: VortexInterface,
    *,
    tracking_source: str = "auto",
    tracking_method: str | None = None,
) -> str | None:
    source_token = str(tracking_source).strip().lower()
    if source_token not in {"auto", "table", "magnetization"}:
        raise ValueError(
            "tracking_source must be one of {'auto', 'table', 'magnetization'}"
        )

    method_token = (
        None if tracking_method is None else str(tracking_method).strip().lower()
    )
    if source_token == "table":
        return "table"
    if source_token == "magnetization":
        if method_token in {None, "", "auto", "table"}:
            cfg_method = str(vortex_interface.config.tracking.method).strip().lower()
            return "gaussian" if cfg_method in {"", "auto", "table"} else cfg_method
        return method_token
    return None if method_token in {None, ""} else method_token


def _resolve_analytical_initial_state(
    vortex_interface: VortexInterface,
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

    def _from_trajectory(
        traj: TrajectoryResult, *, label: str
    ) -> tuple[float, float, str]:
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
    if (
        bool(numerical.metadata.get("steady_state"))
        or "steady_state" in str(numerical.method).lower()
    ):
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
    if src_t.size == ref_t.size and np.allclose(
        src_t, ref_t, rtol=0.0, atol=max(dt_ref * 1e-6, 1e-18)
    ):
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
    polarity_val = (
        1 if polarity_src.size == 0 or float(np.mean(polarity_src)) >= 0.0 else -1
    )
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


__all__ = [
    "_resolve_analytical_initial_state",
    "_resolve_tracking_method_for_source",
    "_resample_trajectory_to_reference",
    "_trajectory_center",
    "_trajectory_dt",
    "_translate_trajectory",
]
