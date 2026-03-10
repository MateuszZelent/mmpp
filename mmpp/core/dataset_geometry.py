"""Geometry and slice helpers for dataset-backed array views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

OVERRIDABLE_ATTRS = {
    "dx",
    "dy",
    "dz",
    "Tx",
    "Ty",
    "Tz",
    "dt",
    "t_sampl",
    "fcut",
    "f_cut",
    "Nx",
    "Ny",
    "Nz",
    "cellsize_x",
    "cellsize_y",
    "cellsize_z",
    "total_time",
    "n_steps",
    "xmin",
    "xmax",
    "ymin",
    "ymax",
    "zmin",
    "zmax",
    "xbase",
    "ybase",
    "zbase",
    "x_name",
    "y_name",
    "z_name",
    "pmin",
    "pmax",
}


def normalize_index_key(
    key: Any,
    ndim: int,
    *,
    keep_dims: bool = True,
) -> tuple[Any, ...]:
    """Normalize indexing key to an explicit ndim-sized tuple."""
    if not isinstance(key, tuple):
        key = (key,)

    n_ellipsis = sum(1 for token in key if token is Ellipsis)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    if n_ellipsis == 1:
        ellipsis_idx = key.index(Ellipsis)
        n_explicit = len(key) - 1
        n_expand = max(0, ndim - n_explicit)
        key = key[:ellipsis_idx] + (slice(None),) * n_expand + key[ellipsis_idx + 1 :]

    if len(key) < ndim:
        key = key + (slice(None),) * (ndim - len(key))

    if len(key) != ndim:
        raise IndexError(f"too many indices for array: expected {ndim}, got {len(key)}")

    normalized: list[Any] = []
    for token in key:
        if keep_dims and isinstance(token, (int, np.integer)) and not isinstance(token, bool):
            idx = int(token)
            normalized.append(slice(idx, idx + 1 if idx != -1 else None))
        else:
            normalized.append(token)
    return tuple(normalized)


def _axis_length(size: int, token: Any) -> int:
    if isinstance(token, slice):
        start, stop, step = token.indices(int(size))
        return len(range(start, stop, step))
    if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
        return 1
    raise TypeError(f"Unsupported index token for shape resolution: {token!r}")


def shape_after_index(shape: tuple[int, ...], key: Any) -> tuple[int, ...]:
    """Return output shape after applying ``key`` with dimension preservation."""
    normalized = normalize_index_key(key, len(shape), keep_dims=True)
    return tuple(
        int(_axis_length(int(size), token))
        for size, token in zip(shape, normalized)
    )


def _is_simple_slice_token(token: Any) -> bool:
    return isinstance(token, slice) or (
        isinstance(token, (int, np.integer)) and not isinstance(token, bool)
    )


def has_only_simple_slices(key: Any, ndim: int) -> bool:
    try:
        normalized = normalize_index_key(key, ndim, keep_dims=True)
    except Exception:
        return False
    return all(_is_simple_slice_token(token) for token in normalized)


def _compose_slices(base: slice, child: slice, source_size: int) -> slice:
    base_start, base_stop, base_step = base.indices(int(source_size))
    base_len = len(range(base_start, base_stop, base_step))
    child_start, child_stop, child_step = child.indices(base_len)
    if len(range(child_start, child_stop, child_step)) == 0:
        return slice(0, 0, 1)
    return slice(
        base_start + child_start * base_step,
        base_start + child_stop * base_step,
        base_step * child_step,
    )


def compose_index_keys(
    base_key: Any,
    child_key: Any,
    source_shape: tuple[int, ...],
) -> tuple[Any, ...]:
    """Compose two ndim-preserving indexing keys into a single source key."""
    ndim = len(source_shape)
    normalized_base = normalize_index_key(
        (slice(None),) * ndim if base_key is None else base_key,
        ndim,
        keep_dims=True,
    )
    normalized_child = normalize_index_key(child_key, ndim, keep_dims=True)

    composed: list[Any] = []
    for size, base_token, child_token in zip(
        source_shape, normalized_base, normalized_child
    ):
        if not isinstance(base_token, slice) or not isinstance(child_token, slice):
            raise TypeError(
                "Only slice-based indexing can be composed without materialization"
            )
        composed.append(_compose_slices(base_token, child_token, int(size)))
    return tuple(composed)


def source_spatial_axes(
    source_shape: Optional[tuple[int, ...]],
) -> Optional[dict[str, int]]:
    """Infer spatial axis mapping for common mmpp dataset layouts."""
    if source_shape is None:
        return None

    ndim = len(source_shape)
    if ndim == 5:
        return {"x": 3, "y": 2, "z": 1}
    if ndim == 4:
        if int(source_shape[-1]) <= 4:
            return {"x": 2, "y": 1, "z": 0}
        return {"x": 3, "y": 2, "z": 1}
    if ndim == 3:
        return {"x": 2, "y": 1, "z": 0}
    return None


def _attr_triplet(attrs: Any, *, key: str) -> Optional[tuple[float, float, float]]:
    if not hasattr(attrs, "get"):
        return None
    raw = attrs.get(key, None)
    if raw is None:
        return None
    try:
        seq = tuple(float(v) for v in raw)
    except Exception:
        return None
    if len(seq) < 3:
        return None
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _axis_min_and_cell(
    *,
    attrs: Any,
    axis: str,
    total_n: int,
    default_cell_m: float,
) -> tuple[float, float]:
    axis_pos = {"x": 0, "y": 1, "z": 2}[axis]
    pmin_triplet = _attr_triplet(attrs, key="pmin")
    pmax_triplet = _attr_triplet(attrs, key="pmax")

    min_key = f"{axis}min"
    max_key = f"{axis}max"
    base_key = f"{axis}base"

    min_value = None
    max_value = None
    if pmin_triplet is not None:
        min_value = float(pmin_triplet[axis_pos])
    elif hasattr(attrs, "get"):
        raw = attrs.get(min_key, None)
        if raw is not None:
            min_value = float(raw)

    if pmax_triplet is not None:
        max_value = float(pmax_triplet[axis_pos])
    elif hasattr(attrs, "get"):
        raw = attrs.get(max_key, None)
        if raw is not None:
            max_value = float(raw)

    cell_m = float(default_cell_m)
    if min_value is not None and max_value is not None and int(total_n) > 0:
        span = float(max_value) - float(min_value)
        if np.isfinite(span) and span > 0.0:
            cell_m = span / float(total_n)

    if min_value is None and hasattr(attrs, "get"):
        raw = attrs.get(base_key, None)
        if raw is not None:
            min_value = float(raw) - 0.5 * cell_m

    if min_value is None and max_value is not None and int(total_n) > 0:
        min_value = float(max_value) - float(total_n) * cell_m

    if min_value is None:
        min_value = 0.0

    return float(min_value), float(cell_m)


def _axis_selection_geometry(
    *,
    total_n: int,
    token: Any,
    axis_min_m: float,
    cell_m: float,
    fallback_count: int,
) -> tuple[float, float, float]:
    count_target = max(int(fallback_count), 1)
    if token is None:
        pmin = float(axis_min_m)
        pmax = pmin + float(count_target) * float(cell_m)
        return pmin, pmax, float(cell_m)

    if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
        idx = int(token)
        if idx < 0:
            idx += int(total_n)
        idx = int(np.clip(idx, 0, max(int(total_n) - 1, 0)))
        pmin = float(axis_min_m) + float(idx) * float(cell_m)
        pmax = pmin + float(cell_m)
        return pmin, pmax, float(cell_m)

    if isinstance(token, slice):
        start, stop, step = token.indices(int(total_n))
        indices = list(range(start, stop, step))
        if not indices:
            pmin = float(axis_min_m)
            pmax = pmin + float(count_target) * float(cell_m)
            return pmin, pmax, float(cell_m)

        lo = min(indices)
        hi = max(indices)
        pmin = float(axis_min_m) + float(lo) * float(cell_m)
        pmax = float(axis_min_m) + float(hi + 1) * float(cell_m)
        cell_eff = (pmax - pmin) / float(count_target)
        return pmin, pmax, float(cell_eff)

    pmin = float(axis_min_m)
    pmax = pmin + float(count_target) * float(cell_m)
    return pmin, pmax, float(cell_m)


@dataclass(frozen=True)
class AxisGeometry:
    """Physical geometry for a single spatial axis of the current dataset view."""

    axis: str
    name: str
    index: int
    size: int
    min_m: float
    max_m: float
    cell_m: float

    @property
    def extent_m(self) -> float:
        return float(self.max_m - self.min_m)

    def center_index(self) -> int:
        if self.size <= 0:
            raise ValueError(f"Axis {self.axis!r} is empty")
        return int(np.clip(self.size // 2, 0, self.size - 1))

    def center_slice(self) -> slice:
        idx = self.center_index()
        return slice(idx, idx + 1)

    def nearest_index(self, value_m: float) -> int:
        if self.size <= 0:
            raise ValueError(f"Axis {self.axis!r} is empty")
        if not np.isfinite(value_m):
            raise ValueError(f"Selection on axis {self.axis!r} must be finite")
        raw = (float(value_m) - float(self.min_m)) / max(float(self.cell_m), 1e-30) - 0.5
        idx = int(np.floor(raw + 0.5))
        return int(np.clip(idx, 0, self.size - 1))

    def select_value(self, value_m: float) -> slice:
        idx = self.nearest_index(float(value_m))
        return slice(idx, idx + 1)

    def select_range(self, start_m: float, stop_m: float) -> slice:
        if self.size <= 0:
            return slice(0, 0, 1)
        low = float(min(start_m, stop_m))
        high = float(max(start_m, stop_m))
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(f"Selection on axis {self.axis!r} must be finite")
        cell = max(float(self.cell_m), 1e-30)
        # Treat the input as a point only when the requested span is tiny
        # relative to the physical cell size. NumPy's default absolute
        # tolerance (1e-08) is far too large for nanometre-scale selections.
        if abs(high - low) <= cell * 1e-9:
            return self.select_value(low)

        low_rel = (low - float(self.min_m)) / cell
        high_rel = (high - float(self.min_m)) / cell
        start_idx = int(np.floor(np.nextafter(low_rel, np.inf)))
        stop_idx = int(np.ceil(np.nextafter(high_rel, -np.inf)))
        start_idx = int(np.clip(start_idx, 0, self.size - 1))
        stop_idx = int(np.clip(stop_idx, start_idx + 1, self.size))
        return slice(start_idx, stop_idx)


@dataclass(frozen=True)
class DatasetGeometry:
    """Current-view geometry for dataset-backed fields."""

    shape: tuple[int, ...]
    spatial_axes: Optional[dict[str, int]]
    axes: dict[str, AxisGeometry]

    def canonical_axis(self, name: str) -> str:
        token = str(name).strip().lower()
        if token in self.axes:
            return token
        for axis, axis_geom in self.axes.items():
            if token == axis_geom.name.strip().lower():
                return axis
        raise KeyError(f"Unknown spatial axis {name!r}")

    @property
    def pmin(self) -> Optional[tuple[float, float, float]]:
        if not self.axes:
            return None
        return (
            float(self.axes["x"].min_m),
            float(self.axes["y"].min_m),
            float(self.axes["z"].min_m),
        )

    @property
    def pmax(self) -> Optional[tuple[float, float, float]]:
        if not self.axes:
            return None
        return (
            float(self.axes["x"].max_m),
            float(self.axes["y"].max_m),
            float(self.axes["z"].max_m),
        )

    def bounds_xyz_m(self) -> tuple[float, float, float, float, float, float]:
        return (
            float(self.axes["x"].min_m),
            float(self.axes["x"].max_m),
            float(self.axes["y"].min_m),
            float(self.axes["y"].max_m),
            float(self.axes["z"].min_m),
            float(self.axes["z"].max_m),
        )

    def cell_xyz_m(self) -> tuple[float, float, float]:
        return (
            float(self.axes["x"].cell_m),
            float(self.axes["y"].cell_m),
            float(self.axes["z"].cell_m),
        )

    def axis_names_xyz(self) -> tuple[str, str, str]:
        return (
            str(self.axes["x"].name),
            str(self.axes["y"].name),
            str(self.axes["z"].name),
        )

    def spatial_shape_zyx(self) -> Optional[tuple[int, int, int]]:
        if self.spatial_axes is None or not self.axes:
            return None
        return (
            int(self.axes["z"].size),
            int(self.axes["y"].size),
            int(self.axes["x"].size),
        )

    def sliced(self, key: Any) -> "DatasetGeometry":
        normalized = normalize_index_key(key, len(self.shape), keep_dims=True)
        new_shape = shape_after_index(self.shape, normalized)
        if self.spatial_axes is None or not self.axes:
            return DatasetGeometry(shape=new_shape, spatial_axes=None, axes={})

        new_axes: dict[str, AxisGeometry] = {}
        for axis, axis_geom in self.axes.items():
            token = normalized[axis_geom.index]
            size = int(new_shape[axis_geom.index])
            min_m, max_m, cell_m = _axis_selection_geometry(
                total_n=int(axis_geom.size),
                token=token,
                axis_min_m=float(axis_geom.min_m),
                cell_m=float(axis_geom.cell_m),
                fallback_count=size,
            )
            new_axes[axis] = AxisGeometry(
                axis=axis,
                name=axis_geom.name,
                index=axis_geom.index,
                size=size,
                min_m=min_m,
                max_m=max_m,
                cell_m=cell_m,
            )

        return DatasetGeometry(
            shape=new_shape,
            spatial_axes=dict(self.spatial_axes),
            axes=new_axes,
        )

    def resampled(self, new_shape: tuple[int, ...]) -> "DatasetGeometry":
        new_shape = tuple(int(v) for v in new_shape)
        if len(new_shape) != len(self.shape):
            raise ValueError(
                f"resampled shape ndim mismatch: {len(new_shape)} != {len(self.shape)}"
            )
        if self.spatial_axes is None or not self.axes:
            return DatasetGeometry(shape=new_shape, spatial_axes=None, axes={})

        new_axes: dict[str, AxisGeometry] = {}
        for axis, axis_geom in self.axes.items():
            size = int(new_shape[axis_geom.index])
            extent = float(axis_geom.max_m - axis_geom.min_m)
            if size > 0 and np.isfinite(extent) and extent > 0.0:
                cell_m = extent / float(size)
            else:
                cell_m = float(axis_geom.cell_m)
            new_axes[axis] = AxisGeometry(
                axis=axis,
                name=axis_geom.name,
                index=axis_geom.index,
                size=size,
                min_m=float(axis_geom.min_m),
                max_m=float(axis_geom.max_m),
                cell_m=float(cell_m),
            )

        return DatasetGeometry(
            shape=new_shape,
            spatial_axes=dict(self.spatial_axes),
            axes=new_axes,
        )


def _dataset_attrs(dataset_obj) -> Any:
    owner = getattr(dataset_obj, "job_result", None)
    if owner is None:
        owner = dataset_obj

    attrs: dict[str, Any] = {}

    for candidate in (owner, dataset_obj):
        if candidate is None:
            continue
        raw_attrs = getattr(candidate, "attrs", None)
        if hasattr(raw_attrs, "as_dict"):
            try:
                attrs.update(raw_attrs.as_dict())
            except Exception:
                pass
        elif hasattr(raw_attrs, "items"):
            try:
                attrs.update(dict(raw_attrs.items()))
            except Exception:
                pass
        elif isinstance(raw_attrs, dict):
            attrs.update(dict(raw_attrs))

        raw_attributes = getattr(candidate, "attributes", None)
        if isinstance(raw_attributes, dict):
            attrs.update(dict(raw_attributes))

    for candidate in (owner, dataset_obj):
        if candidate is None:
            continue
        raw_dict = getattr(candidate, "__dict__", {})
        for key in OVERRIDABLE_ATTRS:
            if key in raw_dict:
                attrs[key] = raw_dict[key]

    return attrs


def _raw_source_shape(dataset_obj) -> Optional[tuple[int, ...]]:
    materialized = getattr(dataset_obj, "_materialized_data", None)
    if materialized is not None:
        return tuple(int(v) for v in np.asarray(materialized).shape)

    source = getattr(dataset_obj, "zarr_array", None)
    shape = getattr(source, "shape", None)
    if shape is None:
        shape = getattr(dataset_obj, "shape", None)
    if shape is None:
        return None
    return tuple(int(v) for v in shape)


def resolve_dataset_geometry(
    dataset_obj: Any,
    *,
    include_slice: bool = True,
) -> DatasetGeometry:
    """Resolve physical geometry for the current dataset view."""
    override = getattr(dataset_obj, "_geometry_override", None)
    slice_info = getattr(dataset_obj, "slice_info", None)
    if isinstance(override, DatasetGeometry):
        if include_slice and slice_info is not None:
            return override.sliced(slice_info)
        return override

    source_shape = _raw_source_shape(dataset_obj)
    if source_shape is None:
        return DatasetGeometry(shape=tuple(), spatial_axes=None, axes={})

    spatial_axes = source_spatial_axes(source_shape)
    if spatial_axes is None:
        active_shape = (
            shape_after_index(source_shape, slice_info)
            if include_slice and slice_info is not None
            else source_shape
        )
        return DatasetGeometry(shape=active_shape, spatial_axes=None, axes={})

    attrs = _dataset_attrs(dataset_obj)
    if hasattr(attrs, "get"):
        dx_m = float(attrs.get("dx", 1e-9))
        dy_m = float(attrs.get("dy", 1e-9))
        dz_m = float(attrs.get("dz", 1e-9))
        axis_names = {
            "x": str(attrs.get("x_name", "x")),
            "y": str(attrs.get("y_name", "y")),
            "z": str(attrs.get("z_name", "z")),
        }
    else:
        dx_m = dy_m = dz_m = 1e-9
        axis_names = {"x": "x", "y": "y", "z": "z"}

    base_axis = {
        "x": _axis_min_and_cell(
            attrs=attrs,
            axis="x",
            total_n=int(source_shape[spatial_axes["x"]]),
            default_cell_m=dx_m,
        ),
        "y": _axis_min_and_cell(
            attrs=attrs,
            axis="y",
            total_n=int(source_shape[spatial_axes["y"]]),
            default_cell_m=dy_m,
        ),
        "z": _axis_min_and_cell(
            attrs=attrs,
            axis="z",
            total_n=int(source_shape[spatial_axes["z"]]),
            default_cell_m=dz_m,
        ),
    }

    normalized_slice = None
    if include_slice and slice_info is not None:
        normalized_slice = normalize_index_key(slice_info, len(source_shape), keep_dims=True)
        active_shape = shape_after_index(source_shape, normalized_slice)
    else:
        active_shape = source_shape

    axes: dict[str, AxisGeometry] = {}
    for axis in ("x", "y", "z"):
        axis_index = int(spatial_axes[axis])
        total_n = int(source_shape[axis_index])
        size = int(active_shape[axis_index])
        token = normalized_slice[axis_index] if normalized_slice is not None else slice(None)
        min_m, max_m, cell_m = _axis_selection_geometry(
            total_n=total_n,
            token=token,
            axis_min_m=float(base_axis[axis][0]),
            cell_m=float(base_axis[axis][1]),
            fallback_count=size,
        )
        axes[axis] = AxisGeometry(
            axis=axis,
            name=axis_names[axis],
            index=axis_index,
            size=size,
            min_m=min_m,
            max_m=max_m,
            cell_m=cell_m,
        )

    return DatasetGeometry(
        shape=tuple(int(v) for v in active_shape),
        spatial_axes=dict(spatial_axes),
        axes=axes,
    )
