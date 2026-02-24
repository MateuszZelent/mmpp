"""High-level core-tracking API bound to MMPP datasets."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..._cache import InMemoryResultCache, build_cache_key
from ..._shared.models import TrajectoryResult
from ...config import VortexConfig
from .tracking import track_core, track_core_lazy


class CoreInterface:
    """Vortex core tracking namespace."""

    def __init__(
        self,
        job_result,
        dataset_name: str | None,
        slice_info: Any | None,
        config: VortexConfig,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config
        self._last_trajectory: TrajectoryResult | None = None
        self._cache = InMemoryResultCache(job_result, namespace="core")

    @property
    def dataset_name(self) -> str:
        if self._dataset_name is None:
            self._dataset_name = self._job.get_largest_m_dataset()
        return self._dataset_name

    def _resolve_dataset(self):
        dataset = getattr(self._job, self.dataset_name)
        if self._slice_info is not None:
            dataset = dataset[self._slice_info]
        return dataset

    def _resolve_data(self) -> np.ndarray:
        dataset = self._resolve_dataset()
        return np.asarray(dataset.numpy(copy=False), dtype=float)

    def _resolve_dt(self) -> float:
        dataset = self._resolve_dataset()
        try:
            return float(dataset.dt)
        except Exception:
            attrs = self._job.attrs
            return float(attrs.get("t_sampl", 1e-12))

    def _resolve_spacing(self) -> tuple[float, float]:
        attrs = self._job.attrs
        dx = attrs.get("dx", attrs.get("cellsize_x", 1.0))
        dy = attrs.get("dy", attrs.get("cellsize_y", 1.0))
        return float(dx), float(dy)

    def _resolve_lazy_dataset_source(self):
        """Return zarr-like source suitable for per-frame lazy reads, if available."""
        try:
            raw = self._job.get_raw(self.dataset_name)
        except Exception:
            return None

        if self._slice_info is None:
            return raw

        # If slicing was applied on DatasetAwareWrapper, laziness may be lost for
        # complex slices. Keep a fallback path to eager tracking for compatibility.
        try:
            return raw[self._slice_info]
        except Exception:
            return None

    def track(
        self,
        method: str | None = None,
        *,
        force: bool = False,
        **kwargs,
    ) -> TrajectoryResult:
        """Track core trajectory over time."""
        if not force and self._last_trajectory is not None and method is None and not kwargs:
            return self._last_trajectory

        dx, dy = self._resolve_spacing()
        dt = self._resolve_dt()

        cfg = self._config.tracking
        selected_method = method or cfg.method
        selected_z = kwargs.pop("z_layer", cfg.z_layer)
        selected_core_threshold = kwargs.pop("core_threshold", cfg.core_threshold)
        selected_gaussian_roi = kwargs.pop("gaussian_roi", cfg.gaussian_roi)
        selected_convention = kwargs.pop("convention", cfg.convention)
        selected_p_up = kwargs.pop("polarity_threshold_up", cfg.polarity_threshold_up)
        selected_p_down = kwargs.pop("polarity_threshold_down", cfg.polarity_threshold_down)
        selected_p_roi = kwargs.pop("polarity_roi_pixels", cfg.polarity_roi_pixels)
        selected_roi = kwargs.pop("roi", None)

        lazy_source = self._resolve_lazy_dataset_source()
        shape_for_key: tuple[int, ...] | None = None
        if lazy_source is not None:
            try:
                shape_for_key = tuple(int(v) for v in getattr(lazy_source, "shape", ()))
            except Exception:
                shape_for_key = None

        if shape_for_key is None:
            data = self._resolve_data()
            shape_for_key = tuple(int(v) for v in data.shape)
        else:
            data = None

        key, config_json = build_cache_key(
            selected_method,
            namespace="core_track",
            config_payload={
                "dataset_name": self.dataset_name,
                "slice_info": repr(self._slice_info),
                "dx": float(dx),
                "dy": float(dy),
                "dt": float(dt),
                "shape": shape_for_key,
                "params": {
                    "z_layer": int(selected_z),
                    "core_threshold": float(selected_core_threshold),
                    "gaussian_roi": int(selected_gaussian_roi),
                    "convention": getattr(selected_convention, "y_axis", "up"),
                    "polarity_threshold_up": float(selected_p_up),
                    "polarity_threshold_down": float(selected_p_down),
                    "polarity_roi_pixels": int(selected_p_roi),
                    "roi": selected_roi,
                    **{str(k): str(v) for k, v in kwargs.items()},
                },
            },
        )
        if not force and self._cache.has(key, config_json):
            return self._cache.get(key)

        common_kwargs = {
            "method": selected_method,
            "z_layer": selected_z,
            "core_threshold": selected_core_threshold,
            "gaussian_roi": selected_gaussian_roi,
            "convention": selected_convention,
            "polarity_threshold_up": selected_p_up,
            "polarity_threshold_down": selected_p_down,
            "polarity_roi_pixels": selected_p_roi,
            "roi": selected_roi,
            "metadata": {
                "dataset": self.dataset_name,
                "slice_info": self._slice_info,
                "job_result": self._job,
            },
        }

        # Prefer lazy per-frame reads for zarr arrays (stage-2 memory behavior).
        if lazy_source is not None and shape_for_key is not None and len(shape_for_key) in {4, 5}:
            result = track_core_lazy(
                lazy_source,
                dx,
                dy,
                dt,
                **common_kwargs,
            )
        else:
            if data is None:
                data = self._resolve_data()
            result = track_core(
                data,
                dx,
                dy,
                dt,
                **common_kwargs,
            )

        self._last_trajectory = result
        self._cache.put(key, result, config_json)
        return result

    def _require_trajectory(self) -> TrajectoryResult:
        if self._last_trajectory is None:
            self._last_trajectory = self.track()
        return self._last_trajectory

    def position(self, t: float | None = None) -> tuple[float, float] | np.ndarray:
        """Get trajectory position at time ``t`` or full array for all frames."""
        traj = self._require_trajectory()
        if t is None:
            return np.column_stack((traj.x, traj.y))

        idx = int(np.argmin(np.abs(traj.time - float(t))))
        return float(traj.x[idx]), float(traj.y[idx])

    def velocity(self, t: float | None = None) -> tuple[float, float] | np.ndarray:
        """Get velocity at time ``t`` or full velocity array for all frames."""
        traj = self._require_trajectory()
        vx, vy = traj.velocity

        if t is None:
            return np.column_stack((vx, vy))

        idx = int(np.argmin(np.abs(traj.time - float(t))))
        return float(vx[idx]), float(vy[idx])

    def _repr_html_(self) -> str:
        from html import escape as _esc

        dataset = _esc(str(self.dataset_name))
        methods = [
            (".track(method=..., **kw)", "Track core trajectory → TrajectoryResult"),
            (".position(t=None)", "Position at time t or full array"),
            (".velocity(t=None)", "Velocity at time t or full array"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        params = [
            ("method", "config", "'gaussian', 'centroid', 'maximum'"),
            ("z_layer", "config", "Z-layer for analysis"),
            ("roi", "None", "Optional ROI (x0,x1,y0,y1) in index coords"),
            ("core_threshold", "config", "Threshold for core detection"),
            ("gaussian_roi", "config", "ROI size for Gaussian fitting (pixels)"),
            ("force", "False", "Force recomputation (bypass cache)"),
        ]
        param_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#a5b4fc;'>{_esc(d)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
            for n, d, desc in params
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Core Tracking Interface</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            f"Vortex core position tracking · dataset: {dataset}</div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Arg</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{param_rows}</tbody></table></div></div>"
        )


__all__ = ["CoreInterface"]
