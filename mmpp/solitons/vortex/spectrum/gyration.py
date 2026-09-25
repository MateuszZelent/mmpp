"""Gyration spectrum computation from vortex core trajectory."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from mmpp._shared.spectral import compute_psd

from ..core.models import TrajectoryResult
from .models import VortexSpectrumResult


def _compute_scalar_spectrum(
    signal: np.ndarray,
    time: np.ndarray,
    *,
    method: str = "welch",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Compute scalar power spectrum using Welch or periodogram."""
    return compute_psd(
        np.asarray(signal, dtype=float),
        time=np.asarray(time, dtype=float),
        method=method,
        nperseg=nperseg,
        noverlap=noverlap,
    )


def _trajectory_provenance(trajectory: TrajectoryResult) -> dict[str, object]:
    """Return compact, serializable provenance for a tracked trajectory."""
    trajectory_metadata = trajectory.metadata
    job_result = trajectory_metadata.get("job_result")
    source_file = trajectory_metadata.get("source_file") or getattr(
        job_result, "path", None
    )
    provenance: dict[str, object] = {}
    if source_file:
        provenance["source_file"] = str(source_file)
        provenance["input_file_count"] = int(
            trajectory_metadata.get("input_file_count", 1)
        )
    else:
        provenance["input_file_count"] = trajectory_metadata.get("input_file_count")

    for key in (
        "source",
        "dataset",
        "slice_info",
        "z_layer",
        "magnetization_component",
        "x_column",
        "y_column",
        "time_column",
        "fallback_from",
    ):
        if key in trajectory_metadata:
            value = trajectory_metadata[key]
            if key == "slice_info" and value is not None:
                value = repr(value)
            provenance[key] = value

    frame_methods = trajectory_metadata.get("method_used", [])
    provenance.update(
        {
            "signal": "PSD(x_core) + PSD(y_core)",
            "trajectory_method": str(trajectory.method),
            "trajectory_requested_method": trajectory_metadata.get("requested_method"),
            "n_samples": int(trajectory.time.size),
            "tracking_frame_methods": dict(
                Counter(str(value) for value in frame_methods)
            ),
            "tracking_frame_fallbacks": int(
                trajectory_metadata.get("gaussian_frame_fallbacks", 0)
            ),
        }
    )
    return provenance


def compute_gyration_spectrum(
    trajectory: TrajectoryResult,
    *,
    method: str = "welch",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> VortexSpectrumResult:
    """Compute vortex gyration spectrum from tracked core coordinates."""
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)

    fx, pxx, used_x, meta = _compute_scalar_spectrum(
        x,
        trajectory.time,
        method=method,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    fy, pyy, used_y, _ = _compute_scalar_spectrum(
        y,
        trajectory.time,
        method=method,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    if fx.size == 0 or fy.size == 0:
        meta.update(_trajectory_provenance(trajectory))
        meta["status"] = "insufficient_samples"
        return VortexSpectrumResult(
            frequencies=np.array([], dtype=float),
            power=np.array([], dtype=float),
            method=method,
            metadata=meta,
        )

    size = min(fx.size, fy.size)
    frequencies = np.asarray(fx[:size], dtype=float)
    power = np.asarray(pxx[:size] + pyy[:size], dtype=float)
    used_method = used_x if used_x == used_y else "mixed"
    meta.update(_trajectory_provenance(trajectory))

    return VortexSpectrumResult(
        frequencies=np.asarray(frequencies, dtype=float),
        power=np.asarray(power, dtype=float),
        method=used_method,
        component="gyration",
        metadata=meta,
    )


def compute_breathing_spectrum(
    trajectory: TrajectoryResult,
    *,
    method: str = "welch",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> VortexSpectrumResult:
    """Compute breathing-mode spectrum from orbit radius signal ``r(t)``."""
    frequencies, power, used_method, meta = _compute_scalar_spectrum(
        trajectory.r,
        trajectory.time,
        method=method,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return VortexSpectrumResult(
        frequencies=np.asarray(frequencies, dtype=float),
        power=np.asarray(power, dtype=float),
        method=used_method,
        component="breathing",
        metadata=meta,
    )
