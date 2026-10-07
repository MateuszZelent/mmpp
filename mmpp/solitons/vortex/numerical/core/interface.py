"""High-level core-tracking API bound to MMPP datasets."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import numpy as np

from mmpp._repr_helpers import api_help_html, html_tabs

from ...._method_helpers import InteractiveNodeMixin
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
    selected_times: np.ndarray | None = None,
    time_selection: Any | None = None,
    source_time_size: int | None = None,
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

    aligned_keys = [x_key, y_key]
    if t_key is not None:
        aligned_keys.append(t_key)
    if z_key is not None:
        aligned_keys.append(z_key)
    lengths = {int(columns[key].size) for key in aligned_keys}
    if len(lengths) != 1:
        raise ValueError(
            "Table core position, time, and polarity columns must have equal "
            f"lengths; got {sorted(lengths)}."
        )
    n = next(iter(lengths))
    if n <= 0:
        raise ValueError("Table-driven tracking found zero samples.")

    attrs = getattr(job_result, "attrs", {})
    if t_key is not None:
        source_time = np.asarray(columns[t_key], dtype=float)
        indices = None
        if selected_times is not None:
            requested = np.asarray(selected_times, dtype=float).reshape(-1)
            if not np.isfinite(requested).all():
                raise ValueError("Selected dataset timestamps must be finite")
            if source_time.size > 1 and np.all(np.diff(source_time) > 0):
                right = np.searchsorted(source_time, requested).clip(
                    0, source_time.size - 1
                )
                left = np.maximum(right - 1, 0)
                indices = np.where(
                    np.abs(source_time[left] - requested)
                    <= np.abs(source_time[right] - requested),
                    left,
                    right,
                )
            elif source_time.size > 1 and np.all(np.diff(source_time) < 0):
                reversed_time = source_time[::-1]
                right = np.searchsorted(reversed_time, requested).clip(
                    0, source_time.size - 1
                )
                left = np.maximum(right - 1, 0)
                reversed_indices = np.where(
                    np.abs(reversed_time[left] - requested)
                    <= np.abs(reversed_time[right] - requested),
                    left,
                    right,
                )
                indices = source_time.size - 1 - reversed_indices
            else:
                indices = np.asarray(
                    [
                        int(np.argmin(np.abs(source_time - value)))
                        for value in requested
                    ],
                    dtype=int,
                )
            matched = source_time[indices]
            local_dt = (
                float(np.median(np.abs(np.diff(source_time))))
                if source_time.size > 1
                else 0.0
            )
            tolerance = max(local_dt * 1e-6, 1e-18)
            if np.any(np.abs(matched - requested) > tolerance):
                raise ValueError(
                    "Dataset timestamps could not be matched to table tracking "
                    "rows within tolerance."
                )
            if np.unique(indices).size != indices.size:
                raise ValueError(
                    "Selected dataset timestamps map to duplicate table rows"
                )
            columns = {
                key: values[indices] if values.size == source_time.size else values
                for key, values in columns.items()
            }
        elif time_selection is not None and source_time_size == n:
            columns = {
                key: values[time_selection]
                if values.size == source_time_size
                else values
                for key, values in columns.items()
            }
        time = np.asarray(columns[t_key], dtype=float)
        n = int(time.size)
    else:
        if selected_times is not None:
            time = np.asarray(selected_times, dtype=float).reshape(-1)
            if time.size != n:
                if time_selection is not None and source_time_size == n:
                    columns = {
                        key: values[time_selection]
                        if values.size == source_time_size
                        else values
                        for key, values in columns.items()
                    }
                    n = int(columns[x_key].size)
                if time.size != n:
                    raise ValueError(
                        "Table tracking has no time column and its row count does "
                        "not match the selected dataset time axis."
                    )
        else:
            raw_dt = attrs.get("t_sampl", attrs.get("sampling_interval"))
            try:
                dt = float(raw_dt)
            except (TypeError, ValueError) as exc:
                raise AttributeError(
                    "Table tracking requires a time column or sampling interval metadata"
                ) from exc
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError(f"Invalid table sampling interval: {raw_dt!r}")
            source_time = np.arange(n, dtype=float) * dt
            if time_selection is not None and source_time_size == n:
                columns = {
                    key: values[time_selection]
                    if values.size == source_time_size
                    else values
                    for key, values in columns.items()
                }
                time = np.asarray(source_time[time_selection], dtype=float).reshape(-1)
                n = int(time.size)
            else:
                time = source_time

    if time.size >= 2:
        dt_est = float(np.median(np.diff(time)))
    else:
        raw_dt = attrs.get("t_sampl", attrs.get("sampling_interval"))
        try:
            dt_est = float(raw_dt)
        except (TypeError, ValueError) as exc:
            raise AttributeError(
                "At least two table timestamps are needed to infer dt"
            ) from exc

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
        # A core location does not identify the sign of its out-of-plane
        # magnetization. Zero is the trajectory contract's explicit unknown.
        polarity = np.zeros(n, dtype=int)
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
        "polarity_status": "observed" if z_key is not None else "unavailable",
        "confidence_scope": (
            "core_polarity_signal_magnitude"
            if z_key is not None
            else "observed_table_position"
        ),
        "polarity_confidence": (
            np.asarray(confidence, dtype=float).copy()
            if z_key is not None
            else np.zeros(n, dtype=float)
        ),
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


class CoreInterface(InteractiveNodeMixin):
    """Vortex core tracking namespace."""

    _interactive_owner = "job[0].vortex.core"
    _interactive_nodes = frozenset({"track", "position", "velocity"})

    def __init__(
        self,
        job_result,
        dataset_name: str | None,
        slice_info: Any | None,
        config: VortexConfig,
        dataset_view: Any | None = None,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config
        self._dataset_view = dataset_view
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
        view = self._dataset_view
        if view is not None:
            shape = tuple(int(value) for value in getattr(view, "shape", ()))
            if shape and shape[-1] >= 3:
                return view
        dataset_name = self.dataset_name
        if dataset_name is None:
            raise ValueError("No magnetisation dataset is available for core tracking")
        dataset = getattr(self._job, dataset_name)
        if self._slice_info is not None:
            dataset = dataset[self._slice_info]
        return dataset

    def _resolve_data(self) -> np.ndarray:
        dataset = self._resolve_dataset()
        if hasattr(dataset, "numpy"):
            return np.asarray(dataset.numpy(copy=False, keepdims=True), dtype=float)
        return np.asarray(dataset, dtype=float)

    def _resolve_dt(self) -> float:
        dataset = self._resolve_dataset()
        try:
            return float(dataset.dt)
        except (AttributeError, TypeError, ValueError):
            times = self._resolve_time_axis()
            if times.size < 2:
                raise ValueError(
                    "At least two selected time samples are required"
                ) from None
            return float(np.mean(np.diff(times)))

    def _resolve_time_axis(self) -> np.ndarray:
        if self._dataset_view is not None:
            try:
                return np.asarray(self._dataset_view.time, dtype=float).reshape(-1)
            except (AttributeError, TypeError, ValueError):
                pass
        dataset = self._resolve_dataset()
        try:
            times = np.asarray(dataset.time, dtype=float).reshape(-1)
            if times.size:
                return times
        except (AttributeError, TypeError, ValueError):
            pass
        attrs = getattr(self._job, "attrs", {})
        raw_dt = attrs.get("t_sampl") if hasattr(attrs, "get") else None
        if raw_dt is None:
            raise AttributeError(
                "Core tracking requires a dataset time axis or positive t_sampl metadata"
            )
        try:
            dt = float(raw_dt)
        except (TypeError, ValueError) as exc:
            raise AttributeError(
                "Core tracking requires a dataset time axis or positive t_sampl metadata"
            ) from exc
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"Invalid t_sampl metadata: {raw_dt!r}")
        return np.arange(int(dataset.shape[0]), dtype=float) * dt

    def _resolve_spacing(self) -> tuple[float, float]:
        view = self._dataset_view
        if view is not None:
            axes = getattr(getattr(view, "geometry", None), "axes", {})
            x_axis, y_axis = axes.get("x"), axes.get("y")
            if x_axis is not None and y_axis is not None:
                return float(x_axis.cell_m), float(y_axis.cell_m)
        attrs = self._job.attrs
        dx = attrs.get("dx", attrs.get("cellsize_x", 1.0))
        dy = attrs.get("dy", attrs.get("cellsize_y", 1.0))
        return float(dx), float(dy)

    def _resolve_lazy_dataset_source(self):
        """Return zarr-like source suitable for per-frame lazy reads, if available."""
        if self._dataset_view is not None:
            shape = tuple(
                int(value) for value in getattr(self._dataset_view, "shape", ())
            )
            if not shape or shape[-1] < 3:
                return None
            if bool(getattr(self._dataset_view, "is_materialized", False)):
                return None
            return self._dataset_view
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
        if (
            method_norm == "auto"
            and self._dataset_view is not None
            and bool(getattr(self._dataset_view, "is_materialized", False))
        ):
            return False
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
        selected_times = None
        if self._dataset_view is not None or self.dataset_name is not None:
            try:
                selected_times = self._resolve_time_axis()
            except (AttributeError, TypeError, ValueError, IndexError):
                selected_times = None

        if (
            requested_method == "table"
            and self._dataset_view is not None
            and bool(getattr(self._dataset_view, "is_materialized", False))
        ):
            raise ValueError(
                "Table tracking cannot represent a materialized field view; use "
                "field tracking or pass a trajectory explicitly."
            )
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
                selected_times=selected_times,
                time_selection=(
                    self._slice_info[0]
                    if isinstance(self._slice_info, tuple) and self._slice_info
                    else None
                ),
                source_time_size=(
                    int(self._dataset_view._index_plan.source_shape[0])
                    if self._dataset_view is not None
                    and getattr(self._dataset_view, "_index_plan", None) is not None
                    else None
                ),
                polarity_column=selected_polarity_column,
            )
            selected_time_digest = (
                hashlib.blake2b(
                    np.ascontiguousarray(selected_times).tobytes(), digest_size=12
                ).hexdigest()
                if selected_times is not None
                else None
            )
            key, config_json = build_cache_key(
                "table",
                namespace="core_track",
                config_payload={
                    "dataset_name": self._dataset_name,
                    "slice_info": repr(self._slice_info),
                    "time_axis_digest": selected_time_digest,
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
                    "source_file": str(getattr(self._job, "path", "")) or None,
                    "input_file_count": 1,
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
        time_axis = self._resolve_time_axis()
        if time_axis.size < 2:
            raise ValueError("At least two selected time samples are required to track")
        dt = abs(float(np.mean(np.diff(time_axis))))

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
                "time_axis_digest": hashlib.blake2b(
                    np.ascontiguousarray(time_axis).tobytes(), digest_size=12
                ).hexdigest(),
                "shape": shape_for_key,
                "materialized_view": (
                    id(self._dataset_view)
                    if self._dataset_view is not None
                    and bool(getattr(self._dataset_view, "is_materialized", False))
                    else None
                ),
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
                "source_file": str(getattr(self._job, "path", "")) or None,
                "input_file_count": 1,
                "source": "dataset",
                "requested_method": requested_method,
                "z_layer": self._resolved_z_layer(selected_z, shape_for_key),
                "magnetization_component": "mz (m[..., 2])",
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
                time_axis=time_axis,
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
                time_axis=time_axis,
                **common_kwargs,
            )

        self._last_trajectory = result
        self._cache.put(key, result, config_json)
        return result

    @staticmethod
    def _resolved_z_layer(z_layer: int, shape: tuple[int, ...]) -> int | None:
        """Return the concrete z-layer index used for a 5D magnetization array."""
        if len(shape) != 5:
            return None
        nz = int(shape[1])
        index = int(z_layer)
        if index < 0:
            index += nz
        return index

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
