"""High-level core-tracking API bound to MMPP datasets."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._cache import InMemoryResultCache, build_cache_key
from ..config import VortexConfig
from .models import TrajectoryResult
from .tracking import track_core


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

        data = self._resolve_data()
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

        key, config_json = build_cache_key(
            selected_method,
            namespace="core_track",
            config_payload={
                "dataset_name": self.dataset_name,
                "slice_info": repr(self._slice_info),
                "dx": float(dx),
                "dy": float(dy),
                "dt": float(dt),
                "shape": tuple(int(v) for v in data.shape),
                "params": {
                    "z_layer": int(selected_z),
                    "core_threshold": float(selected_core_threshold),
                    "gaussian_roi": int(selected_gaussian_roi),
                    "convention": getattr(selected_convention, "y_axis", "up"),
                    "polarity_threshold_up": float(selected_p_up),
                    "polarity_threshold_down": float(selected_p_down),
                    "polarity_roi_pixels": int(selected_p_roi),
                    **{str(k): str(v) for k, v in kwargs.items()},
                },
            },
        )
        if not force and self._cache.has(key, config_json):
            return self._cache.get(key)

        result = track_core(
            data,
            dx,
            dy,
            dt,
            method=selected_method,
            z_layer=selected_z,
            core_threshold=selected_core_threshold,
            gaussian_roi=selected_gaussian_roi,
            convention=selected_convention,
            polarity_threshold_up=selected_p_up,
            polarity_threshold_down=selected_p_down,
            polarity_roi_pixels=selected_p_roi,
            metadata={
                "dataset": self.dataset_name,
                "slice_info": self._slice_info,
            },
            **kwargs,
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
