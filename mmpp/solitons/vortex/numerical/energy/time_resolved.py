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


def _resolve_time_array(columns: dict[str, np.ndarray], attrs: Any, n_samples: int) -> np.ndarray:
    for name in ("t", "time", "Time"):
        if name in columns and int(columns[name].size) == int(n_samples):
            return np.asarray(columns[name], dtype=float)
    dt = float(attrs.get("t_sampl", 1e-12)) if hasattr(attrs, "get") else 1e-12
    return np.arange(int(n_samples), dtype=float) * dt


def extract_energy_time_series(
    job_result,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    prefixes: tuple[str, ...] = ("E_", "energy", "W_"),
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

    n = int(min(table_columns[name].size for name in selected_names))
    time = _resolve_time_array(table_columns, getattr(job_result, "attrs", {}), n_samples=n)[:n]

    channels = {
        name: np.asarray(table_columns[name][:n], dtype=float)
        for name in selected_names
    }

    return EnergyTimeSeriesResult(
        time=np.asarray(time, dtype=float),
        channels=channels,
        metadata={
            "status": "ok",
            "selected_columns": list(selected_names),
            "available_columns": sorted(table_columns.keys()),
        },
    )


__all__ = ["extract_energy_time_series"]
