"""High-level core-tracking API bound to MMPP datasets."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from mmpp._repr_helpers import api_help_html, html_tabs

from ..._cache import InMemoryResultCache, build_cache_key
from ..._shared.models import TrajectoryResult
from ...config import VortexConfig
from .tracking import track_core, track_core_lazy

_POSITION_X_ALIASES = (
    "ext_coreposx",
    "coreposx",
    "core_pos_x",
    "core_x",
    "x_core",
)
_POSITION_Y_ALIASES = (
    "ext_coreposy",
    "coreposy",
    "core_pos_y",
    "core_y",
    "y_core",
)
_POLARITY_ALIASES = (
    "ext_coreposz",
    "coreposz",
    "core_pos_z",
    "core_z",
    "z_core",
    "mz",
)
_TIME_ALIASES = ("t", "time", "Time")


def _read_table_columns(job_result) -> dict[str, np.ndarray]:
    if "table" not in job_result:
        return {}
    table = job_result["table"]
    out: dict[str, np.ndarray] = {}
    for key in table.keys():
        try:
            arr = table[key]
            shape = tuple(getattr(arr, "shape", ()))
            if len(shape) != 1:
                continue
            out[str(key)] = np.asarray(arr[:], dtype=float).reshape(-1)
        except Exception:
            continue
    return out


def _resolve_column_name(
    columns: dict[str, np.ndarray], aliases: tuple[str, ...]
) -> str | None:
    lut = {name.lower(): name for name in columns}
    for alias in aliases:
        key = lut.get(alias.lower())
        if key is not None:
            return key
    return None


def _track_core_from_table(
    job_result,
    *,
    polarity_threshold_up: float,
    polarity_threshold_down: float,
    x_column: str | None = None,
    y_column: str | None = None,
    polarity_column: str | None = None,
) -> TrajectoryResult:
    columns = _read_table_columns(job_result)
    if not columns:
        raise ValueError("No readable 1D columns were found in the table group.")

    x_key = x_column or _resolve_column_name(columns, _POSITION_X_ALIASES)
    y_key = y_column or _resolve_column_name(columns, _POSITION_Y_ALIASES)
    z_key = polarity_column or _resolve_column_name(columns, _POLARITY_ALIASES)
    t_key = _resolve_column_name(columns, _TIME_ALIASES)

    if x_key is None or y_key is None:
        raise ValueError(
            "Table-driven tracking requires readable core-position columns "
            "(expected aliases like ext_coreposx/ext_coreposy)."
        )

    lengths = [int(columns[x_key].size), int(columns[y_key].size)]
    if t_key is not None:
        lengths.append(int(columns[t_key].size))
    if z_key is not None:
        lengths.append(int(columns[z_key].size))
    n = int(min(lengths))
    if n <= 0:
        raise ValueError("Table-driven tracking found zero samples.")

    attrs = getattr(job_result, "attrs", {})
    if t_key is not None:
        time = np.asarray(columns[t_key][:n], dtype=float)
    else:
        dt = float(attrs.get("t_sampl", attrs.get("sampling_interval", 1e-12)))
        time = np.arange(n, dtype=float) * dt

    if time.size >= 2:
        dt_est = float(np.median(np.diff(time)))
    else:
        dt_est = float(attrs.get("t_sampl", attrs.get("sampling_interval", 1e-12)))

    x = np.asarray(columns[x_key][:n], dtype=float)
    y = np.asarray(columns[y_key][:n], dtype=float)

    if z_key is not None:
        core_signal = np.asarray(columns[z_key][:n], dtype=float)
        polarity = np.zeros(n, dtype=int)
        state = 1 if float(core_signal[0]) >= 0.0 else -1
        switch_times: list[float] = []
        switch_count = 0
        for idx, value in enumerate(core_signal):
            if state > 0 and value <= float(polarity_threshold_down):
                state = -1
                switch_count += 1
                switch_times.append(float(time[idx]))
            elif state < 0 and value >= float(polarity_threshold_up):
                state = 1
                switch_count += 1
                switch_times.append(float(time[idx]))
            polarity[idx] = state
        confidence = np.clip(np.abs(core_signal), 0.0, 1.0)
    else:
        core_signal = None
        polarity = np.ones(n, dtype=int)
        switch_times = []
        switch_count = 0
        confidence = np.ones(n, dtype=float)

    metadata: dict[str, Any] = {
        "source": "table",
        "dt": float(dt_est),
        "n_frames": int(n),
        "requested_method": "table",
        "x_column": str(x_key),
        "y_column": str(y_key),
        "time_column": str(t_key) if t_key is not None else None,
        "polarity_column": str(z_key) if z_key is not None else None,
        "table_columns": sorted(columns.keys()),
        "method_used": ["table"] * int(n),
        "gaussian_frame_fallbacks": 0,
        "convention": "physical_table",
        "polarity_threshold_up": float(polarity_threshold_up),
        "polarity_threshold_down": float(polarity_threshold_down),
        "p_switch_count": int(switch_count),
        "switch_times_s": [float(v) for v in switch_times],
    }
    if core_signal is not None:
        metadata["core_signal_mz"] = np.asarray(core_signal, dtype=float)

    return TrajectoryResult(
        time=time,
        x=x,
        y=y,
        polarity=polarity,
        method="table",
        confidence=np.asarray(confidence, dtype=float),
        metadata=metadata,
    )


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
    def dataset_name(self) -> str | None:
        if self._dataset_name is None:
            candidate = self._job.get_largest_m_dataset()
            try:
                self._job._ensure_zarr_loaded()
                if candidate in self._job._z:
                    self._dataset_name = candidate
            except Exception:
                self._dataset_name = candidate
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

    def _table_tracking_available(
        self,
        *,
        x_column: str | None = None,
        y_column: str | None = None,
    ) -> bool:
        columns = _read_table_columns(self._job)
        x_key = x_column or _resolve_column_name(columns, _POSITION_X_ALIASES)
        y_key = y_column or _resolve_column_name(columns, _POSITION_Y_ALIASES)
        return x_key is not None and y_key is not None

    def _should_prefer_table_tracking(
        self,
        selected_method: str,
        *,
        x_column: str | None = None,
        y_column: str | None = None,
    ) -> bool:
        method_norm = str(selected_method).lower()
        if method_norm == "table":
            return True
        if method_norm != "auto":
            return False
        if not self._table_tracking_available(x_column=x_column, y_column=y_column):
            return False

        if self._dataset_name is None:
            for key in self._job.keys():
                if str(key) == "table":
                    continue
                try:
                    raw = self._job.get_raw(str(key))
                except Exception:
                    continue
                shape = tuple(int(v) for v in getattr(raw, "shape", ()))
                if len(shape) in {4, 5} and shape[-1] >= 3:
                    if int(shape[0]) > 1:
                        return False
            return True

        try:
            raw = self._job.get_raw(self.dataset_name)
        except Exception:
            return True

        shape = tuple(int(v) for v in getattr(raw, "shape", ()))
        if len(shape) not in {4, 5}:
            return True
        return int(shape[0]) <= 1

    def track(
        self,
        method: str | None = None,
        *,
        force: bool = False,
        **kwargs,
    ) -> TrajectoryResult:
        """Track core trajectory over time."""
        if (
            not force
            and self._last_trajectory is not None
            and method is None
            and not kwargs
        ):
            return self._last_trajectory

        cfg = self._config.tracking
        selected_method = method or cfg.method
        selected_z = kwargs.pop("z_layer", cfg.z_layer)
        selected_core_threshold = kwargs.pop("core_threshold", cfg.core_threshold)
        selected_gaussian_roi = kwargs.pop("gaussian_roi", cfg.gaussian_roi)
        selected_convention = kwargs.pop("convention", cfg.convention)
        selected_p_up = kwargs.pop("polarity_threshold_up", cfg.polarity_threshold_up)
        selected_p_down = kwargs.pop(
            "polarity_threshold_down", cfg.polarity_threshold_down
        )
        selected_p_roi = kwargs.pop("polarity_roi_pixels", cfg.polarity_roi_pixels)
        selected_roi = kwargs.pop("roi", None)
        selected_x_column = kwargs.pop("x_column", None)
        selected_y_column = kwargs.pop("y_column", None)
        selected_polarity_column = kwargs.pop("polarity_column", None)

        requested_method = str(selected_method).lower()
        if self._should_prefer_table_tracking(
            requested_method,
            x_column=selected_x_column,
            y_column=selected_y_column,
        ):
            preview = _track_core_from_table(
                self._job,
                polarity_threshold_up=float(selected_p_up),
                polarity_threshold_down=float(selected_p_down),
                x_column=selected_x_column,
                y_column=selected_y_column,
                polarity_column=selected_polarity_column,
            )
            key, config_json = build_cache_key(
                "table",
                namespace="core_track",
                config_payload={
                    "dataset_name": self._dataset_name,
                    "slice_info": repr(self._slice_info),
                    "params": {
                        "x_column": preview.metadata.get("x_column"),
                        "y_column": preview.metadata.get("y_column"),
                        "time_column": preview.metadata.get("time_column"),
                        "polarity_column": preview.metadata.get("polarity_column"),
                        "polarity_threshold_up": float(selected_p_up),
                        "polarity_threshold_down": float(selected_p_down),
                    },
                },
            )
            if not force and self._cache.has(key, config_json):
                return self._cache.get(key)

            preview.metadata.update(
                {
                    "dataset": self._dataset_name,
                    "slice_info": self._slice_info,
                    "job_result": self._job,
                    "requested_method": requested_method,
                }
            )
            self._last_trajectory = preview
            self._cache.put(key, preview, config_json)
            return preview

        if requested_method == "table":
            raise ValueError(
                "method='table' was requested but no usable table core-position columns "
                "were found."
            )

        dx, dy = self._resolve_spacing()
        dt = self._resolve_dt()

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

        effective_method = (
            "gaussian" if requested_method == "auto" else requested_method
        )

        key, config_json = build_cache_key(
            effective_method,
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
            "method": effective_method,
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
                "source": "dataset",
                "requested_method": requested_method,
            },
        }

        # Prefer lazy per-frame reads for zarr arrays (stage-2 memory behavior).
        if (
            lazy_source is not None
            and shape_for_key is not None
            and len(shape_for_key) in {4, 5}
        ):
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

        dataset = _esc(
            str(self._dataset_name if self._dataset_name is not None else "auto")
        )
        methods = [
            (".track(method=..., **kw)", "Track core trajectory from dataset or table"),
            (".position(t=None)", "Position at time t or full array"),
            (".velocity(t=None)", "Velocity at time t or full array"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        params = [
            ("method", "config", "'auto', 'table', 'gaussian', 'centroid', 'maximum'"),
            ("z_layer", "config", "Z-layer for magnetization-based analysis"),
            ("roi", "None", "Optional ROI (x0,x1,y0,y1) in index coords"),
            (
                "core_threshold",
                "config",
                "Threshold for centroid/Gaussian core detection",
            ),
            ("gaussian_roi", "config", "ROI size for Gaussian fitting (pixels)"),
            ("x_column", "None", "Override table X-position column name"),
            ("y_column", "None", "Override table Y-position column name"),
            ("polarity_column", "None", "Override table polarity/core-signal column"),
            ("force", "False", "Force recomputation (bypass cache)"),
        ]
        param_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#a5b4fc;'>{_esc(d)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
            for n, d, desc in params
        )
        overview = (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);">'
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
        api = api_help_html(
            self,
            title="Core tracking API help",
            prefix="vortex.core",
            methods=["track", "position", "velocity"],
            subtitle="Live public API for vortex-core trajectory tracking.",
            chrome=False,
        )
        return (
            '<div style=\'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);'>"
            + html_tabs(
                [("Overview", overview), ("API", api)],
                uid=f"mmpp-vortex-core-{uuid.uuid4().hex}",
            )
            + "</div>"
        )


__all__ = ["CoreInterface"]
