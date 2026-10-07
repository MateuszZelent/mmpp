"""Extraction of time-resolved vortex energy channels from table data."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import EnergyTimeSeriesResult


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


def _resolve_time_array(
    columns: dict[str, np.ndarray], attrs: Any, n_samples: int
) -> np.ndarray:
    for name in ("t", "time", "Time"):
        if name in columns:
            time = np.asarray(columns[name], dtype=float)
            if int(time.size) != int(n_samples):
                raise ValueError(
                    f"Table time column {name!r} has {time.size} samples but "
                    f"selected channels have {n_samples}"
                )
            return time
    raw_dt = (
        attrs.get("t_sampl", attrs.get("sampling_interval"))
        if hasattr(attrs, "get")
        else None
    )
    try:
        dt = float(raw_dt)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Energy channels without a table time column require sampling interval metadata"
        ) from exc
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid energy sampling interval: {raw_dt!r}")
    return np.arange(int(n_samples), dtype=float) * dt


def _align_to_selected_times(
    source_time: np.ndarray,
    target_time: np.ndarray,
    values: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Select table rows by timestamp, rejecting length-only alignment."""
    source = np.asarray(source_time, dtype=float).reshape(-1)
    target = np.asarray(target_time, dtype=float).reshape(-1)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Energy and dataset timestamps must be finite")
    if source.size > 1:
        delta = np.diff(source)
        if not (np.all(delta > 0.0) or np.all(delta < 0.0)):
            raise ValueError("Energy table timestamps must be strictly monotonic")

    if source.size > 1 and np.all(np.diff(source) < 0.0):
        increasing = source[::-1]
        right = np.searchsorted(increasing, target).clip(0, source.size - 1)
        left = np.maximum(right - 1, 0)
        rev_idx = np.where(
            np.abs(increasing[left] - target) <= np.abs(increasing[right] - target),
            left,
            right,
        )
        indices = source.size - 1 - rev_idx
    elif source.size > 1:
        right = np.searchsorted(source, target).clip(0, source.size - 1)
        left = np.maximum(right - 1, 0)
        indices = np.where(
            np.abs(source[left] - target) <= np.abs(source[right] - target),
            left,
            right,
        )
    else:
        indices = np.zeros(target.size, dtype=int)

    matched = source[indices]
    dt = float(np.median(np.abs(np.diff(source)))) if source.size > 1 else 0.0
    tolerance = max(dt * 1e-6, 1e-18)
    if np.any(np.abs(matched - target) > tolerance):
        raise ValueError(
            "Energy table timestamps do not match the selected dataset time axis; "
            "matching by row count is not sufficient."
        )
    if np.unique(indices).size != indices.size:
        raise ValueError("Selected dataset times map to duplicate energy rows")
    return matched, {name: channel[indices] for name, channel in values.items()}


def extract_energy_time_series(
    job_result,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    prefixes: tuple[str, ...] = ("E_", "energy", "W_"),
    selected_times: np.ndarray | None = None,
    time_selection: Any | None = None,
    source_time_size: int | None = None,
) -> EnergyTimeSeriesResult:
    """Extract energy channels from the table group."""
    table_columns = _read_table_columns(job_result)
    if not table_columns:
        return EnergyTimeSeriesResult(
            time=np.array([], dtype=float),
            channels={},
            metadata={"status": "table_missing_or_unreadable"},
        )

    if columns is None:
        selected_names: list[str] = []
        for key in sorted(table_columns.keys()):
            key_norm = key.lower()
            if any(key_norm.startswith(prefix.lower()) for prefix in prefixes):
                selected_names.append(key)
        # Common explicit aliases even if they do not match prefix heuristics.
        for alias in ("E_ex", "E_demag", "E_Zeeman", "E_total"):
            if alias in table_columns and alias not in selected_names:
                selected_names.append(alias)
    else:
        selected_names = [str(name) for name in columns if str(name) in table_columns]

    if not selected_names:
        return EnergyTimeSeriesResult(
            time=np.array([], dtype=float),
            channels={},
            metadata={
                "status": "no_energy_columns",
                "available_columns": sorted(table_columns.keys()),
            },
        )

    lengths = {int(table_columns[name].size) for name in selected_names}
    if len(lengths) != 1:
        raise ValueError(
            f"Selected energy channels must have equal lengths; got {sorted(lengths)}."
        )
    n = next(iter(lengths))
    time = _resolve_time_array(table_columns, getattr(job_result, "attrs", {}), n)
    channels = {
        name: np.asarray(table_columns[name], dtype=float) for name in selected_names
    }
    alignment = "table_rows"

    if selected_times is not None:
        time, channels = _align_to_selected_times(time, selected_times, channels)
        alignment = "exact_timestamps"
    elif time_selection is not None and source_time_size == n:
        time = np.asarray(time[time_selection], dtype=float).reshape(-1)
        channels = {
            name: np.asarray(channel[time_selection], dtype=float).reshape(-1)
            for name, channel in channels.items()
        }
        alignment = "source_index_selection"

    return EnergyTimeSeriesResult(
        time=np.asarray(time, dtype=float),
        channels=channels,
        metadata={
            "status": "ok",
            "selected_columns": list(selected_names),
            "available_columns": sorted(table_columns.keys()),
            "time_alignment": alignment,
        },
    )


__all__ = ["extract_energy_time_series"]
