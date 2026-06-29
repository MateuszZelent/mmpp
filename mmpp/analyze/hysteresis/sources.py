"""Data-source loaders for hysteresis analysis."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np

from .compute import segment_branches, validate_hysteresis_data
from .config import HysteresisConfig
from .result import HysteresisResult

_FIELD_ALIASES = [
    "B_extx",
    "B_exty",
    "B_extz",
    "B_ext",
    "B0",
    "Bext",
    "H",
    "Hext",
    "Hx",
    "Hy",
    "Hz",
]

_MAG_ALIASES = [
    "mx",
    "my",
    "mz",
    "m_full",
    "Mx",
    "My",
    "Mz",
]

_COMPONENT_KEYS = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def _resolve_config(config: HysteresisConfig | None) -> HysteresisConfig:
    return config if config is not None else HysteresisConfig()


def _available_table_arrays(table_group: Any) -> list[str]:
    names: list[str] = []
    for key in list(table_group.keys()):
        try:
            item = table_group[key]
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                names.append(str(key))
        except Exception:
            continue
    return sorted(set(names))


def _safe_read_1d(array_obj: Any, name: str) -> np.ndarray:
    try:
        shape = tuple(getattr(array_obj, "shape", ()))
        if len(shape) != 1:
            raise ValueError(f"Column '{name}' is not 1D (shape={shape})")
        if shape and int(shape[0]) == 0:
            return np.array([], dtype=float)

        values = np.asarray(array_obj[:], dtype=float).reshape(-1)
    except Exception as exc:
        raise ValueError(
            f"Failed reading table column '{name}'. The dataset may be empty/corrupt."
        ) from exc
    return values


def _resolve_column_name(
    available_columns: Iterable[str],
    requested: str | None,
    aliases: list[str],
    component: str | None = None,
) -> str | None:
    columns = list(available_columns)
    lower_to_original = {c.lower(): c for c in columns}

    if requested:
        exact = lower_to_original.get(str(requested).lower())
        if exact is not None:
            return exact

    candidates: list[str] = []
    if component:
        comp = str(component).lower()
        if aliases is _FIELD_ALIASES:
            candidates.extend([f"B_ext{comp}", f"H{comp}", f"B{comp}"])
        else:
            candidates.extend([f"m{comp}", f"M{comp}"])

    candidates.extend(aliases)

    for cand in candidates:
        found = lower_to_original.get(cand.lower())
        if found is not None:
            return found

    return None


def _resolve_roi_indices(
    roi: tuple[float, float, float, float] | None,
    *,
    roi_units: str,
    nx: int,
    ny: int,
    dx: float | None,
    dy: float | None,
) -> tuple[int, int, int, int]:
    if roi is None:
        return (0, nx, 0, ny)

    if len(roi) != 4:
        raise ValueError("roi must be a 4-tuple: (x0, x1, y0, y1)")

    x0, x1, y0, y1 = [float(v) for v in roi]
    units = str(roi_units).lower()

    if units == "nm":
        if dx is None or dy is None:
            raise ValueError("roi_units='nm' requires dx and dy attributes")
        dx_nm = float(dx) * 1e9
        dy_nm = float(dy) * 1e9
        if dx_nm <= 0 or dy_nm <= 0:
            raise ValueError("dx and dy must be positive to convert roi_units='nm'")
        x0, x1 = x0 / dx_nm, x1 / dx_nm
        y0, y1 = y0 / dy_nm, y1 / dy_nm
    elif units != "idx":
        raise ValueError("roi_units must be 'idx' or 'nm'")

    xi0, xi1 = sorted((int(np.floor(x0)), int(np.ceil(x1))))
    yi0, yi1 = sorted((int(np.floor(y0)), int(np.ceil(y1))))

    xi0 = int(np.clip(xi0, 0, nx))
    xi1 = int(np.clip(xi1, 0, nx))
    yi0 = int(np.clip(yi0, 0, ny))
    yi1 = int(np.clip(yi1, 0, ny))

    if xi1 <= xi0 or yi1 <= yi0:
        raise ValueError(
            f"ROI collapsed to empty range after clipping: ({xi0}, {xi1}, {yi0}, {yi1})"
        )

    return (xi0, xi1, yi0, yi1)


def _extract_table_arrays(
    job_result,
    *,
    field: str | None,
    magnetization: str | None,
    component: str | None,
) -> tuple[np.ndarray, np.ndarray, str, str, list[str]]:
    if "table" not in job_result:
        raise ValueError("No 'table' group found in job result")

    table = job_result["table"]
    columns = _available_table_arrays(table)
    if not columns:
        raise ValueError("Table group exists, but no readable 1D columns were found")

    field_col = _resolve_column_name(columns, field, _FIELD_ALIASES, component=component)
    mag_col = _resolve_column_name(
        columns,
        magnetization,
        _MAG_ALIASES,
        component=component,
    )

    if field_col is None or mag_col is None:
        raise ValueError(
            "Could not resolve table columns for field/magnetization. "
            f"Resolved field={field_col!r}, magnetization={mag_col!r}. "
            f"Available columns: {columns}"
        )

    field_arr = _safe_read_1d(table[field_col], field_col)
    mag_arr = _safe_read_1d(table[mag_col], mag_col)
    return field_arr, mag_arr, field_col, mag_col, columns


def from_arrays(
    field: np.ndarray,
    magnetization: np.ndarray,
    *,
    frame_index: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    config: HysteresisConfig | None = None,
    cloneflip: bool = False,
) -> HysteresisResult:
    """Build hysteresis result from explicit arrays."""
    cfg = _resolve_config(config)
    field_arr = np.asarray(field, dtype=float).reshape(-1)
    mag_arr = np.asarray(magnetization, dtype=float).reshape(-1)

    finite_mask = np.isfinite(field_arr) & np.isfinite(mag_arr)
    field_clean, mag_clean = validate_hysteresis_data(
        field_arr,
        mag_arr,
        require_non_monotonic=False,
    )

    if frame_index is None:
        frame_arr = np.arange(field_arr.size, dtype=int)[finite_mask]
    else:
        frame_full = np.asarray(frame_index, dtype=int).reshape(-1)
        if frame_full.size != field_arr.size:
            raise ValueError(
                f"frame_index size mismatch: expected {field_arr.size}, got {frame_full.size}"
            )
        frame_arr = frame_full[finite_mask]

    branches = segment_branches(field_clean)
    meta = dict(metadata or {})
    meta.setdefault("source_type", "arrays")

    result = HysteresisResult(
        field=field_clean,
        magnetization=mag_clean,
        branches=branches,
        frame_index=frame_arr,
        config=cfg,
        metadata=meta,
    )
    if cloneflip:
        from .compute import build_cloneflip_result
        result = build_cloneflip_result(result)
    return result


def from_table(
    job_result,
    *,
    field: str | None = None,
    magnetization: str | None = None,
    component: str | None = None,
    metadata: dict[str, Any] | None = None,
    config: HysteresisConfig | None = None,
    cloneflip: bool = False,
) -> HysteresisResult:
    """Read hysteresis signal from the table group."""
    cfg = _resolve_config(config)
    field_arr, mag_arr, field_col, mag_col, columns = _extract_table_arrays(
        job_result,
        field=field,
        magnetization=magnetization,
        component=component,
    )

    if field_arr.size != mag_arr.size:
        raise ValueError(
            f"Table column size mismatch: {field_col}({field_arr.size}) vs "
            f"{mag_col}({mag_arr.size}). Available columns: {columns}"
        )

    finite_mask = np.isfinite(field_arr) & np.isfinite(mag_arr)
    field_clean, mag_clean = validate_hysteresis_data(
        field_arr, mag_arr, require_non_monotonic=not cloneflip
    )

    meta = dict(metadata or {})
    meta.update(
        {
            "source_type": "table",
            "field_column": field_col,
            "magnetization_column": mag_col,
            "table_columns": columns,
            "job_result": job_result,
            "field_unit": meta.get("field_unit", "input"),
        }
    )

    frame_idx = np.arange(field_arr.size, dtype=int)[finite_mask]

    result = HysteresisResult(
        field=field_clean,
        magnetization=mag_clean,
        branches=segment_branches(field_clean),
        frame_index=frame_idx,
        config=cfg,
        metadata=meta,
    )
    if cloneflip:
        from .compute import build_cloneflip_result
        result = build_cloneflip_result(result)
    return result


def _extract_field_array(
    job_result,
    *,
    field: str | np.ndarray | None,
    component: str | None,
    expected_length: int,
) -> tuple[np.ndarray, str]:
    if field is None:
        try:
            field_arr, _mag_arr, field_col, _mag_col, _cols = _extract_table_arrays(
                job_result,
                field=None,
                magnetization=None,
                component=component,
            )
            source_name = field_col
        except Exception:
            field_arr = np.arange(expected_length, dtype=float)
            source_name = "index"
    elif isinstance(field, str):
        if "table" not in job_result:
            raise ValueError("field was provided as column name but no table group exists")
        table = job_result["table"]
        columns = _available_table_arrays(table)
        resolved = _resolve_column_name(columns, field, _FIELD_ALIASES, component=component)
        if resolved is None:
            raise ValueError(
                f"Could not find field column '{field}'. Available columns: {columns}"
            )
        field_arr = _safe_read_1d(table[resolved], resolved)
        source_name = resolved
    else:
        field_arr = np.asarray(field, dtype=float).reshape(-1)
        source_name = "array"

    if field_arr.size != expected_length:
        raise ValueError(
            f"Field length mismatch: expected {expected_length}, got {field_arr.size}"
        )

    return field_arr, source_name


def from_magnetization(
    job_result,
    *,
    dset: str = "m",
    component: str = "x",
    z_layer: int | str = 0,
    roi: tuple[float, float, float, float] | None = None,
    roi_units: str = "idx",
    field: str | np.ndarray | None = None,
    slice_info: Any | None = None,
    metadata: dict[str, Any] | None = None,
    config: HysteresisConfig | None = None,
    cloneflip: bool = False,
) -> HysteresisResult:
    """Build hysteresis loop from averaged magnetization dataset."""
    cfg = _resolve_config(config)

    try:
        dset_obj = job_result.get_raw(dset)
    except Exception as exc:
        raise ValueError(f"Dataset '{dset}' is not available") from exc

    try:
        # Prefer lazy per-frame reads to avoid materializing full 4D/5D arrays.
        source_data = dset_obj if slice_info is None else dset_obj[slice_info]
    except Exception as exc:
        raise ValueError(f"Failed reading dataset '{dset}'") from exc

    shape = tuple(getattr(source_data, "shape", ()))
    if len(shape) not in {4, 5}:
        raise ValueError(
            f"Dataset '{dset}' must be 4D or 5D magnetization, got ndim={len(shape)}"
        )

    component_key = str(component).lower()
    if component_key == "snapshot":
        component_key = "norm"
    if component_key not in {"x", "y", "z", "norm"}:
        raise ValueError("component must be one of: 'x', 'y', 'z', 'norm', 'snapshot'")

    if len(shape) == 5:
        # (t, z, y, x, c)
        t_count, z_count, ny, nx, c_count = shape
        if c_count < 1:
            raise ValueError(f"Dataset '{dset}' has no vector components")
        if z_layer == "all":
            z_idx = None
        else:
            z_idx = int(z_layer)
            if z_idx < 0 or z_idx >= z_count:
                raise ValueError(f"z_layer out of range: {z_idx} (valid 0..{z_count - 1})")
    else:
        # (t, y, x, c)
        t_count, ny, nx, c_count = shape
        if c_count < 1:
            raise ValueError(f"Dataset '{dset}' has no vector components")
        if z_layer != 0 and z_layer != "all":
            raise ValueError(
                "z_layer is not applicable for 4D dataset; use 0 or 'all'"
            )
        z_idx = None

    attrs = getattr(job_result, "attrs", {})
    dx = attrs.get("dx") if hasattr(attrs, "get") else None
    dy = attrs.get("dy") if hasattr(attrs, "get") else None

    x0, x1, y0, y1 = _resolve_roi_indices(
        roi,
        roi_units=roi_units,
        nx=nx,
        ny=ny,
        dx=float(dx) if dx is not None else None,
        dy=float(dy) if dy is not None else None,
    )

    n_components = max(1, min(3, int(c_count)))
    comp_idx = _COMPONENT_KEYS.get(component_key)
    if comp_idx is not None and comp_idx >= n_components:
        raise ValueError(
            f"Requested component '{component_key}' is not available in "
            f"dataset '{dset}' (components={n_components})"
        )

    mag_series = np.zeros(int(t_count), dtype=float)
    for t_idx in range(int(t_count)):
        if len(shape) == 5:
            if z_idx is None:
                # (z, y, x, c) -> mean over z
                frame_vec = np.asarray(
                    source_data[t_idx, :, y0:y1, x0:x1, :n_components],
                    dtype=float,
                )
                vec_roi = np.mean(frame_vec, axis=0)
            else:
                vec_roi = np.asarray(
                    source_data[t_idx, z_idx, y0:y1, x0:x1, :n_components],
                    dtype=float,
                )
        else:
            vec_roi = np.asarray(
                source_data[t_idx, y0:y1, x0:x1, :n_components],
                dtype=float,
            )

        if vec_roi.size == 0:
            raise ValueError("ROI produced empty magnetization selection")

        if comp_idx is not None:
            mag_series[t_idx] = float(np.mean(vec_roi[..., comp_idx]))
        else:
            mag_series[t_idx] = float(np.mean(np.linalg.norm(vec_roi, axis=-1)))

    field_arr, field_source = _extract_field_array(
        job_result,
        field=field,
        component=component_key,
        expected_length=t_count,
    )

    finite_mask = np.isfinite(field_arr) & np.isfinite(mag_series)
    field_clean, mag_clean = validate_hysteresis_data(
        field_arr,
        mag_series,
        require_non_monotonic=field_source != "index" and not cloneflip,
    )
    frame_idx = np.arange(t_count, dtype=int)[finite_mask]

    meta = dict(metadata or {})
    meta.update(
        {
            "source_type": "magnetization",
            "job_result": job_result,
            "dataset": dset,
            "slice_info": slice_info,
            "component": component_key,
            "z_layer": z_layer,
            "roi": (x0, x1, y0, y1),
            "roi_units": "idx",
            "field_source": field_source,
            "field_unit": meta.get("field_unit", "input"),
        }
    )

    result = HysteresisResult(
        field=field_clean,
        magnetization=mag_clean,
        branches=segment_branches(field_clean),
        frame_index=frame_idx,
        config=cfg,
        metadata=meta,
    )
    if cloneflip:
        from .compute import build_cloneflip_result
        result = build_cloneflip_result(result)
    return result


def _parse_field_from_key(key: str, key_prefix: str) -> float | None:
    """Extract float field value from a key like ``B-0.500000.6``.

    Supports patterns:

    - ``{prefix}{value}.{suffix}`` where suffix is digits (legacy form)
    - ``{prefix}0_{value}_{unit}`` used by some `B0` keyed sweeps
    - ``{prefix}{value}_{unit}`` with optional sign and unit suffix
    Returns ``None`` if the key does not match the pattern.
    """
    if key_prefix:
        if not key.startswith(key_prefix):
            return None
    else:
        return None

    # Fast path for legacy keys like ``B-0.500000.6``.
    pattern = rf"^{re.escape(key_prefix)}(.+)\.(\d+)$"
    m = re.match(pattern, key)
    if m is not None:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Generalized parse: strip the prefix and parse the remaining token.
    token = key[len(key_prefix) :]
    if not token:
        return None

    # Some sweep writers encode field keys as ``B0_-10_mT`` or ``B0-10_mT``.
    if token.startswith("0_") or token.startswith("0-"):
        token = token[2:]

    # Remove a trailing unit marker if present.
    token = re.sub(r"[_-][a-zA-Z]+$", "", token)

    # For older format like ``-0.500000.6`` keep the first two dot-separated
    # parts as the numeric value.
    if token.count(".") >= 2:
        token = ".".join(token.split(".")[:-1])

    token = token.strip("._-")
    if not token:
        return None

    try:
        return float(token)
    except ValueError:
        return None


def _extract_spatial_mean(
    array_obj,
    *,
    component: str,
    z_layer: int | str,
    roi: tuple[int, int, int, int] | None,
) -> float:
    """Average magnetization snapshot over spatial dims for one zarr array.

    Supports shapes:
    - ``(t, z, y, x, c)`` — 5-D (time/batch collapsed to first axis)
    - ``(z, y, x, c)``  — 4-D
    - ``(y, x, c)``     — 3-D

    Always picks the **last** time/batch index (settled state).
    """
    raw = np.asarray(array_obj[:], dtype=float)

    # squeeze leading size-1 axes until we have at most 5 dims
    while raw.ndim > 5 and raw.shape[0] == 1:
        raw = raw[0]

    # normalise to (y, x, c) by collapsing t, z
    if raw.ndim == 5:
        # (t, z, y, x, c) — take last t, select z
        t_idx = raw.shape[0] - 1
        if z_layer == "all":
            frame = raw[t_idx].mean(axis=0)          # (y, x, c)
        else:
            z_idx = int(np.clip(int(z_layer), 0, raw.shape[1] - 1))
            frame = raw[t_idx, z_idx]                # (y, x, c)
    elif raw.ndim == 4:
        # (z, y, x, c) — select z
        if z_layer == "all":
            frame = raw.mean(axis=0)
        else:
            z_idx = int(np.clip(int(z_layer), 0, raw.shape[0] - 1))
            frame = raw[z_idx]
    elif raw.ndim == 3:
        # (y, x, c) already
        frame = raw
    else:
        raise ValueError(f"Unsupported array shape for snapshot averaging: {raw.shape}")

    if frame.shape[-1] < 3:
        padded = np.zeros(frame.shape[:-1] + (3,), dtype=float)
        padded[..., : frame.shape[-1]] = frame
        frame = padded

    # apply ROI
    if roi is not None:
        x0, x1, y0, y1 = roi
        frame = frame[y0:y1, x0:x1, :]

    if frame.size == 0:
        return float("nan")

    comp = str(component).lower()
    _comp_map = {"x": 0, "y": 1, "z": 2}
    if comp in _comp_map:
        idx = _comp_map[comp]
        if idx >= frame.shape[-1]:
            return float("nan")
        spatial = frame[..., idx]
    else:
        # norm
        spatial = np.linalg.norm(frame, axis=-1)

    return float(np.nanmean(spatial))


def from_zarr_keys(
    job_result,
    *,
    key_prefix: str = "B",
    component: str = "x",
    z_layer: int | str = 0,
    roi: tuple[float, float, float, float] | None = None,
    roi_units: str = "idx",
    min_spatial_size: int = 50,
    metadata: dict[str, Any] | None = None,
    config: HysteresisConfig | None = None,
    cloneflip: bool = False,
) -> HysteresisResult:
    """Build hysteresis loop from named zarr arrays in the root group.

    This source is for simulations where each applied-field value is stored
    in a separate zarr array at the root level — e.g. ``B-0.500000.6``.

    The field value is parsed from the key name using the pattern::

        {key_prefix}{field_value}.{suffix}

    For example ``B-0.500000.6`` with ``key_prefix="B"`` yields ``B = -0.5 T``.

    Parameters
    ----------
    job_result:
        Any object whose root zarr group is accessible via ``job_result[key]``
        or ``job_result._z[key]``.  Works with ``ZarrJobResult`` from mmpp.
    key_prefix:
        Prefix to match keys with.  Default ``"B"`` matches ``B-0.500000.6``.
    component:
        Magnetization component to average: ``"x"``, ``"y"``, ``"z"`` or
        ``"norm"``.
    z_layer:
        Z-slice index (int) or ``"all"`` to average across all z layers.
    roi:
        ``(x0, x1, y0, y1)`` spatial region of interest.  Units controlled
        by ``roi_units``.
    roi_units:
        ``"idx"`` (default) or ``"nm"``.
    min_spatial_size:
        Skip arrays where ``y * x < min_spatial_size`` (tiny test/debug
        snapshots that should not enter the loop).
    metadata:
        Extra metadata attached to the result.
    config:
        Optional :class:`HysteresisConfig`.

    Returns
    -------
    HysteresisResult
        Sorted by field value, one point per matching key.
    """
    cfg = _resolve_config(config)

    # ── resolve the zarr group ──────────────────────────────────────────────
    if hasattr(job_result, "_z"):
        zgroup = job_result._z
    elif hasattr(job_result, "_zarr"):
        zgroup = job_result._zarr
    else:
        # assume job_result is itself zarr-like
        zgroup = job_result

    all_keys = list(zgroup.keys())

    # ── collect (field_value, key) pairs ───────────────────────────────────
    candidates: list[tuple[float, str]] = []
    for key in all_keys:
        fval = _parse_field_from_key(key, key_prefix)
        if fval is None:
            continue
        try:
            arr = zgroup[key]
            shape = tuple(getattr(arr, "shape", ()))
            # need at least 3 dims (y, x, c); skip tiny debug arrays
            if len(shape) < 3:
                continue
            y_size = shape[-3] if len(shape) >= 3 else 1
            x_size = shape[-2] if len(shape) >= 3 else 1
            if y_size * x_size < int(min_spatial_size):
                continue
        except Exception:
            continue
        candidates.append((fval, key))

    if not candidates:
        raise ValueError(
            f"No zarr keys matching prefix '{key_prefix}' with parseable field "
            f"values found.  Available keys: {sorted(all_keys)[:20]}"
        )

    # sort ascending by field value
    candidates.sort(key=lambda t: t[0])

    # ── resolve ROI in index units ─────────────────────────────────────────
    roi_idx: tuple[int, int, int, int] | None = None
    if roi is not None:
        # sample first array to get dimensions
        sample_arr = zgroup[candidates[0][1]]
        raw_sample = np.asarray(sample_arr[:], dtype=float)
        while raw_sample.ndim > 3 and raw_sample.shape[0] == 1:
            raw_sample = raw_sample[0]
        ny = int(raw_sample.shape[-3]) if raw_sample.ndim >= 3 else 1
        nx = int(raw_sample.shape[-2]) if raw_sample.ndim >= 3 else 1

        attrs = getattr(job_result, "attrs", {})
        dx = float(attrs.get("dx", 1e-9)) if hasattr(attrs, "get") else 1e-9
        dy = float(attrs.get("dy", 1e-9)) if hasattr(attrs, "get") else 1e-9

        roi_idx = _resolve_roi_indices(
            roi, roi_units=roi_units, nx=nx, ny=ny,
            dx=dx, dy=dy,
        )

    # ── average each snapshot ──────────────────────────────────────────────
    field_vals: list[float] = []
    mag_vals: list[float] = []
    frame_indices: list[int] = []

    for frame_idx, (fval, key) in enumerate(candidates):
        try:
            m_mean = _extract_spatial_mean(
                zgroup[key],
                component=component,
                z_layer=z_layer,
                roi=roi_idx,
            )
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Skipping key '{key}' (field={fval:.6g}): {exc}",
                stacklevel=2,
            )
            continue

        if not np.isfinite(m_mean):
            continue

        field_vals.append(fval)
        mag_vals.append(m_mean)
        frame_indices.append(frame_idx)

    if len(field_vals) < 3:
        raise ValueError(
            f"Only {len(field_vals)} valid points extracted from zarr keys "
            f"— need at least 3.  Check key_prefix, component, and roi."
        )

    field_arr = np.asarray(field_vals, dtype=float)
    mag_arr = np.asarray(mag_vals, dtype=float)
    frame_arr = np.asarray(frame_indices, dtype=int)

    from .compute import segment_branches as _seg

    meta = dict(metadata or {})
    # frame_keys: {frame_idx_value -> zarr_key}  (frame_idx_value == candidate index)
    frame_keys = {fi: candidates[fi][1] for fi in frame_indices}
    meta.update({
        "source_type": "zarr_keys",
        "key_prefix": key_prefix,
        "component": component,
        "z_layer": z_layer,
        "roi": roi_idx,
        "n_keys_scanned": len(candidates),
        "field_unit": "T",
        "job_result": job_result,
        "frame_keys": frame_keys,   # dict[int, str]: zarr key for each result frame
        "zarr_group": zgroup,       # the zarr root group for snapshot loading
    })

    result = HysteresisResult(
        field=field_arr,
        magnetization=mag_arr,
        branches=_seg(field_arr),
        frame_index=frame_arr,
        config=cfg,
        metadata=meta,
    )
    if cloneflip:
        from .compute import build_cloneflip_result
        result = build_cloneflip_result(result)
    return result


def resolve_auto_source(
    job_result,
    *,
    field: str | np.ndarray | None = None,
    magnetization: str | np.ndarray | None = None,
    component: str | None = None,
    dset: str | None = None,
    z_layer: int | str = 0,
    roi: tuple[float, float, float, float] | None = None,
    roi_units: str = "idx",
    slice_info: Any | None = None,
    config: HysteresisConfig | None = None,
) -> HysteresisResult:
    """Resolve best source automatically: table -> magnetization fallback."""
    cfg = _resolve_config(config)
    errors: list[str] = []

    if isinstance(field, np.ndarray) and isinstance(magnetization, np.ndarray):
        return from_arrays(
            field,
            magnetization,
            config=cfg,
            metadata={"source_type": "arrays"},
        )

    try:
        return from_table(
            job_result,
            field=field if isinstance(field, str) else None,
            magnetization=magnetization if isinstance(magnetization, str) else None,
            component=component,
            config=cfg,
        )
    except Exception as exc:
        errors.append(f"table: {exc}")

    try:
        return from_magnetization(
            job_result,
            dset=dset or cfg.default_m_dataset,
            component=component or cfg.default_component,
            z_layer=z_layer,
            roi=roi,
            roi_units=roi_units,
            field=field,
            slice_info=slice_info,
            config=cfg,
        )
    except Exception as exc:
        errors.append(f"magnetization: {exc}")

    try:
        return from_zarr_keys(
            job_result,
            component=component or cfg.default_component,
            z_layer=z_layer,
            roi=roi,
            roi_units=roi_units,
            config=cfg,
        )
    except Exception as exc:
        errors.append(f"zarr_keys: {exc}")

    raise ValueError(
        "Unable to resolve hysteresis source automatically. "
        f"Attempted table -> magnetization -> zarr_keys fallback. Details: {errors}"
    )
