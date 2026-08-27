"""Batch soliton analysis helpers."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mmpp._repr_helpers import (
    NODE_COLOR_ANALYSIS,
    NODE_COLOR_COMPUTE,
    NODE_COLOR_PLOT,
    NODE_COLOR_UTIL,
    accessors_section_html,
    api_help_html,
    examples_section_html,
    metrics_section_html,
    node_card_html,
)
from mmpp._shared.spectral import compute_psd
from mmpp.cache.serializers import serialize_for_json

from .vortex._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)


def _coerce_numeric(value: Any, default: float = float("nan")) -> float:
    """Convert value to float when possible."""
    try:
        return float(value)
    except Exception:
        return float(default)


def _disk_radius_from_attrs(attrs: dict[str, Any]) -> float:
    """Infer disk radius from common metadata keys."""
    for key in ("D", "diameter", "disk_diameter", "pillar_diameter"):
        if key in attrs:
            diameter = _coerce_numeric(attrs[key])
            if np.isfinite(diameter) and diameter > 0.0:
                return 0.5 * diameter
    return float("nan")


def _coordinate_label(name: str) -> str:
    """Human-readable label for batch coordinate axes."""
    labels = {
        "i_pillar_ma": "Current [mA]",
        "ma": "Current [mA]",
        "Jdc": "Jdc [A/m^2]",
        "ni": "ni",
        "addoe": "addoe",
    }
    return labels.get(name, name)


def _resolved_spectrum_kwargs(
    trajectory,
    *,
    method: str,
    nperseg: int | None,
    noverlap: int | None,
) -> dict[str, Any]:
    """Adapt spectral parameters to the available trajectory length."""
    kwargs: dict[str, Any] = {"method": method}
    sample_count = int(len(getattr(trajectory, "time", [])))

    effective_nperseg = (
        None if nperseg is None else min(int(nperseg), max(sample_count, 1))
    )
    if effective_nperseg is not None:
        kwargs["nperseg"] = effective_nperseg

    if noverlap is not None:
        if effective_nperseg is None:
            kwargs["noverlap"] = int(max(noverlap, 0))
        else:
            kwargs["noverlap"] = int(
                min(max(noverlap, 0), max(effective_nperseg - 1, 0))
            )

    return kwargs


def _progress_iter(iterable, *, total: int, desc: str, enabled: bool):
    """Wrap iterable with tqdm when available and requested."""
    if not enabled or total <= 1:
        return iterable
    try:
        tqdm = import_module("tqdm.auto").tqdm
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit="result",
            leave=True,
        )
    except ImportError:
        return _text_progress_iter(iterable, total=total, desc=desc)


def _text_progress_iter(iterable, *, total: int, desc: str):
    """Minimal visible progress fallback when tqdm is unavailable."""
    print(f"{desc}: 0/{total}", end="", file=sys.stderr, flush=True)
    for index, item in enumerate(iterable, start=1):
        yield item
        print(f"\r{desc}: {index}/{total}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)


def _memory_mb() -> float:
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


def _result_cache_identity(result: Any, *, sort_by: str) -> dict[str, Any]:
    """Stable-ish identity for invalidating batch post-processing artifacts."""
    path = getattr(result, "path", None)
    attrs = getattr(result, "attrs", {}) or {}
    path_str = None if path is None else str(path)
    mtime_ns = None
    if path_str:
        try:
            mtime_ns = os.stat(path_str).st_mtime_ns
        except OSError:
            mtime_ns = None
    return {
        "path": path_str,
        "mtime_ns": mtime_ns,
        "sort_value": serialize_for_json(attrs.get(sort_by)),
        "attrs": serialize_for_json(dict(attrs) if hasattr(attrs, "items") else attrs),
    }


def _batch_postprocessing_root(
    mmpp_instance: Any | None,
    results: list[Any],
) -> Path | None:
    """Resolve the main sweep folder for persisted batch post-processing."""
    base_path = getattr(mmpp_instance, "base_path", None)
    if base_path:
        return Path(str(base_path)).expanduser().resolve()

    absolute_parents: list[Path] = []
    for result in results:
        raw_path = getattr(result, "path", None)
        if raw_path is None:
            continue
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            continue
        parent = path.parent if path.suffix == ".zarr" else path
        absolute_parents.append(parent.resolve())

    if not absolute_parents:
        return None

    try:
        return Path(os.path.commonpath([str(path) for path in absolute_parents]))
    except ValueError:
        return absolute_parents[0]


def _spectrum_map_cache_config(
    *,
    component: str,
    source: str,
    dataset_name: str | None,
    z_index: int | None,
    spatial_reduction: str,
    sort_by: str,
    steady_state: bool,
    spectrum_method: str,
    nperseg: int | None,
    noverlap: int | None,
    exclude_annihilated: bool,
    ordered_results: list[Any],
) -> dict[str, Any]:
    return {
        "kind": "vortex.spectrum_map",
        "version": 1,
        "component": component,
        "source": source,
        "dataset_name": dataset_name,
        "z_index": z_index,
        "spatial_reduction": spatial_reduction,
        "sort_by": sort_by,
        "steady_state": bool(steady_state),
        "spectrum_method": spectrum_method,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "exclude_annihilated": bool(exclude_annihilated),
        "results": [
            _result_cache_identity(result, sort_by=sort_by)
            for result in ordered_results
        ],
    }


def _spectrum_map_cache_key(config: dict[str, Any]) -> str:
    payload = json.dumps(serialize_for_json(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_spectrum_map_cache_dir(
    *,
    cache_dir: str | os.PathLike[str] | None,
    cache_key: str,
    mmpp_instance: Any | None,
    ordered_results: list[Any],
) -> Path | None:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()

    root = _batch_postprocessing_root(mmpp_instance, ordered_results)
    if root is None:
        return None
    return root / ".mmpp_postprocessing" / "vortex" / "spectrum_map" / cache_key


def _load_spectrum_map_cache(
    cache_path: Path,
    config: dict[str, Any],
) -> BatchVortexSpectrumMapResult | None:
    metadata_path = cache_path / "metadata.json"
    data_path = cache_path / "data.npz"
    if not metadata_path.exists() or not data_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config") != serialize_for_json(config):
            return None
        with np.load(data_path, allow_pickle=False) as data:
            coordinate = np.asarray(data["coordinate"], dtype=float)
            frequencies = np.asarray(data["frequencies"], dtype=float)
            power = np.asarray(data["power"], dtype=float)
        result_metadata = dict(metadata.get("result_metadata") or {})
        result_metadata["cache"] = {
            "status": "hit",
            "path": str(cache_path),
            "key": metadata.get("cache_key"),
            "created_at": metadata.get("created_at"),
            "force": False,
        }
        return BatchVortexSpectrumMapResult(
            coordinate=coordinate,
            frequencies=frequencies,
            power=power,
            component=str(metadata.get("component", config["component"])),
            coordinate_name=str(metadata.get("coordinate_name", config["sort_by"])),
            metadata=result_metadata,
        )
    except Exception:
        return None


def _store_spectrum_map_cache(
    result: BatchVortexSpectrumMapResult,
    *,
    cache_path: Path,
    cache_key: str,
    config: dict[str, Any],
) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path / "data.npz",
        coordinate=np.asarray(result.coordinate, dtype=float),
        frequencies=np.asarray(result.frequencies, dtype=float),
        power=np.asarray(result.power, dtype=float),
    )
    metadata = {
        "kind": "vortex.spectrum_map",
        "cache_key": cache_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "component": result.component,
        "coordinate_name": result.coordinate_name,
        "config": serialize_for_json(config),
        "result_metadata": serialize_for_json(
            {
                key: value
                for key, value in result.metadata.items()
                if key != "cache"
            }
        ),
    }
    (cache_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


_PROCESSED_SPECTRUM_COMPONENTS = {"gyration", "breathing"}
_MAGNETIZATION_COMPONENT_ALIASES = {
    "x": "mx",
    "mx": "mx",
    "m_x": "mx",
    "y": "my",
    "my": "my",
    "m_y": "my",
    "z": "mz",
    "mz": "mz",
    "m_z": "mz",
}
_MAGNETIZATION_COMPONENT_INDEX = {"mx": 0, "my": 1, "mz": 2}


def _normalize_spectrum_map_component(component: str) -> str:
    component_norm = str(component).lower()
    return _MAGNETIZATION_COMPONENT_ALIASES.get(component_norm, component_norm)


def _table_has_component(result: Any, component: str) -> bool:
    try:
        if "table" not in result:
            return False
        table = result["table"]
        aliases = _table_component_aliases(component)
        lower_to_name = {str(name).lower(): str(name) for name in table.keys()}
        return any(alias.lower() in lower_to_name for alias in aliases)
    except Exception:
        return False


def _resolve_spectrum_map_source(
    *,
    source: str,
    component: str,
    ordered_results: list[Any],
) -> str:
    source_norm = str(source or "auto").lower()
    component_norm = _normalize_spectrum_map_component(component)
    if source_norm in {"auto", "default"}:
        if component_norm in _PROCESSED_SPECTRUM_COMPONENTS:
            return "processed"
        if component_norm in _MAGNETIZATION_COMPONENT_INDEX:
            if any(_table_has_component(result, component_norm) for result in ordered_results):
                return "table"
            return "magnetization"
    if source_norm in {"processed", "trajectory", "motion", "core"}:
        return "processed"
    if source_norm in {"table", "tabular"}:
        return "table"
    if source_norm in {"magnetization", "m", "raw"}:
        return "magnetization"
    raise ValueError(
        "source must be 'auto', 'processed', 'table', or 'magnetization'"
    )


def _validate_spectrum_map_component(*, source: str, component: str) -> str:
    component_norm = _normalize_spectrum_map_component(component)
    if source == "processed":
        if component_norm not in _PROCESSED_SPECTRUM_COMPONENTS:
            raise ValueError(
                "processed spectrum_map component must be 'gyration' or 'breathing'"
            )
    elif component_norm not in _MAGNETIZATION_COMPONENT_INDEX:
        raise ValueError(
            "direct magnetization spectrum_map component must be 'mx', 'my', or 'mz'"
        )
    return component_norm


def _table_component_aliases(component: str) -> tuple[str, ...]:
    component_norm = _normalize_spectrum_map_component(component)
    if component_norm == "mx":
        return ("mx", "Mx", "m_x", "M_x", "x")
    if component_norm == "my":
        return ("my", "My", "m_y", "M_y", "y")
    if component_norm == "mz":
        return ("mz", "Mz", "m_z", "M_z", "z")
    return (component,)


def _read_table_trace(result: Any, *, component: str) -> tuple[np.ndarray, np.ndarray]:
    if "table" not in result:
        raise ValueError("table source requested but result has no 'table' group")
    table = result["table"]
    lower_to_name = {str(name).lower(): str(name) for name in table.keys()}

    value_key = None
    for alias in _table_component_aliases(component):
        key = lower_to_name.get(alias.lower())
        if key is not None:
            value_key = key
            break
    if value_key is None:
        raise ValueError(f"table has no {component!r} magnetization column")

    signal = np.asarray(table[value_key][:], dtype=float).reshape(-1)
    time_key = None
    for alias in ("t", "time", "Time"):
        key = lower_to_name.get(alias.lower())
        if key is not None:
            time_key = key
            break
    if time_key is not None:
        time = np.asarray(table[time_key][:], dtype=float).reshape(-1)
    else:
        attrs = getattr(result, "attrs", {}) or {}
        dt = _coerce_numeric(attrs.get("t_sampl", 1e-12), default=1e-12)
        time = np.arange(signal.size, dtype=float) * float(dt)

    n = min(int(time.size), int(signal.size))
    return time[:n], signal[:n]


def _resolve_magnetization_dataset_name(result: Any, dataset_name: str | None) -> str:
    if dataset_name:
        return str(dataset_name)
    if hasattr(result, "get_largest_m_dataset"):
        try:
            return str(result.get_largest_m_dataset())
        except Exception:
            pass
    return "m"


def _read_magnetization_trace(
    result: Any,
    *,
    component: str,
    dataset_name: str | None,
    z_index: int | None,
    spatial_reduction: str,
) -> tuple[np.ndarray, np.ndarray]:
    dset_name = _resolve_magnetization_dataset_name(result, dataset_name)
    raw = result.get_raw(dset_name)
    data = np.asarray(raw[:], dtype=float)
    if data.ndim < 2:
        raise ValueError("magnetization dataset must include time and component axes")
    comp_index = _MAGNETIZATION_COMPONENT_INDEX[component]
    if data.shape[-1] <= comp_index:
        raise ValueError(
            f"magnetization dataset does not contain component index {comp_index}"
        )
    values = data[..., comp_index]
    if z_index is not None and values.ndim >= 4:
        values = np.take(values, int(z_index), axis=1)
    if values.ndim == 1:
        signal = values
    else:
        reduction = str(spatial_reduction).lower()
        axes = tuple(range(1, values.ndim))
        if reduction == "mean":
            signal = np.nanmean(values, axis=axes)
        elif reduction == "sum":
            signal = np.nansum(values, axis=axes)
        elif reduction == "max":
            signal = np.nanmax(values, axis=axes)
        elif reduction == "min":
            signal = np.nanmin(values, axis=axes)
        else:
            raise ValueError("spatial_reduction must be 'mean', 'sum', 'max', or 'min'")

    attrs = getattr(result, "attrs", {}) or {}
    dt = _coerce_numeric(attrs.get("t_sampl", 1e-12), default=1e-12)
    time = np.arange(np.asarray(signal).size, dtype=float) * float(dt)
    return time, np.asarray(signal, dtype=float).reshape(-1)


def _resolve_direct_signal_trace(
    result: Any,
    *,
    component: str,
    source: str,
    dataset_name: str | None,
    z_index: int | None,
    spatial_reduction: str,
) -> tuple[np.ndarray, np.ndarray]:
    if source == "table":
        return _read_table_trace(result, component=component)
    return _read_magnetization_trace(
        result,
        component=component,
        dataset_name=dataset_name,
        z_index=z_index,
        spatial_reduction=spatial_reduction,
    )


def _compute_direct_signal_spectrum(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    method: str,
    nperseg: int | None,
    noverlap: int | None,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    kwargs: dict[str, Any] = {"method": method}
    if nperseg is not None:
        kwargs["nperseg"] = min(int(nperseg), max(int(np.asarray(signal).size), 1))
    if noverlap is not None:
        effective_nperseg = kwargs.get("nperseg")
        kwargs["noverlap"] = (
            int(max(noverlap, 0))
            if effective_nperseg is None
            else int(min(max(noverlap, 0), max(int(effective_nperseg) - 1, 0)))
        )
    return compute_psd(
        np.asarray(signal, dtype=float),
        time=np.asarray(time, dtype=float),
        **kwargs,
    )


def _compute_vortex_spectrum_map_one(
    payload: tuple[
        int,
        Any,
        str,
        bool,
        str,
        int | None,
        int | None,
        str,
        str,
        str | None,
        int | None,
        str,
        bool,
    ],
) -> tuple[float, np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
    (
        index,
        result,
        sort_by,
        steady_state,
        spectrum_method,
        nperseg,
        noverlap,
        component,
        source,
        dataset_name,
        z_index,
        spatial_reduction,
        exclude_annihilated,
    ) = payload
    attrs = getattr(result, "attrs", {}) or {}
    coord_value = _coerce_numeric(attrs.get(sort_by, index), default=float(index))

    try:
        if source in {"table", "magnetization"}:
            time, signal = _resolve_direct_signal_trace(
                result,
                component=component,
                source=source,
                dataset_name=dataset_name,
                z_index=z_index,
                spatial_reduction=spatial_reduction,
            )
            frequencies, power, _used_method, _meta = _compute_direct_signal_spectrum(
                time,
                signal,
                method=spectrum_method,
                nperseg=nperseg,
                noverlap=noverlap,
            )
            return coord_value, frequencies, power, None

        vortex = result.solitons.vortex
        trajectory = (
            vortex.trajectory.steady_state() if steady_state else vortex.trajectory.raw
        )

        try:
            from mmpp.solitons.vortex.health import check_core_health

            health = check_core_health(
                result,
                trajectory=trajectory,
            )
            if not health.is_healthy:
                if exclude_annihilated:
                    import warnings

                    path_label = str(getattr(result, "path", index))
                    warnings.warn(
                        f"spectrum_map: excluding {path_label} — "
                        + "; ".join(health.warnings),
                        UserWarning,
                        stacklevel=4,
                    )
                    return (
                        coord_value,
                        None,
                        None,
                        {
                            "index": index,
                            "path": getattr(result, "path", None),
                            "coordinate": coord_value,
                            "error": "excluded (annihilated): "
                            + "; ".join(health.warnings),
                        },
                    )
                health.issue_python_warnings()
        except Exception:
            pass

        spectrum_kwargs = _resolved_spectrum_kwargs(
            trajectory,
            method=spectrum_method,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        if component == "gyration":
            spectrum = vortex.spectrum.gyration(
                trajectory=trajectory,
                **spectrum_kwargs,
            )
        else:
            spectrum = vortex.spectrum.breathing(
                trajectory=trajectory,
                **spectrum_kwargs,
            )

        frequencies = np.asarray(spectrum.frequencies, dtype=float)
        power = np.asarray(spectrum.power, dtype=float)
        return coord_value, frequencies, power, None
    except Exception as exc:
        return (
            coord_value,
            None,
            None,
            {
                "index": index,
                "path": getattr(result, "path", None),
                "coordinate": coord_value,
                "error": str(exc),
            },
        )


_REGIME_ORDER = [
    "stable_gyro",
    "damped",
    "intermittent",
    "collision",
    "remagnetization",
    "error",
]

_REGIME_COLORS = {
    "stable_gyro": "tab:green",
    "damped": "tab:gray",
    "intermittent": "tab:orange",
    "collision": "tab:red",
    "remagnetization": "tab:purple",
    "error": "black",
}

_PHASE_METRICS = {
    "regime",
    "peak_gyr_ghz",
    "peak_breath_ghz",
    "peak_power_rel",
    "r_mean_nm",
    "r_max_nm",
    "r_max_rel",
    "n_p_switch",
    "n_gc_switch",
    "n_expulsion",
}


def _regime_codes(values) -> np.ndarray:
    """Encode regime labels as stable integer codes."""
    mapping = {name: idx for idx, name in enumerate(_REGIME_ORDER)}
    return np.asarray(
        [mapping.get(str(item), mapping["error"]) for item in values],
        dtype=float,
    )


def _phase_metric_is_categorical(metric: str) -> bool:
    return str(metric) == "regime"


def _mode_value(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return np.nan
    mode = values.mode()
    if mode.empty:
        return values.iloc[0]
    return mode.iloc[0]


def _aggregate_phase_frame(
    frame: pd.DataFrame,
    *,
    axes: tuple[str, str | None, str | None],
    metric: str,
    aggregate: str,
) -> pd.DataFrame:
    """Aggregate duplicate phase-diagram points."""
    axis_cols = [name for name in axes if name is not None]
    if not axis_cols or frame.empty:
        return frame.reset_index(drop=True)
    if not frame.duplicated(axis_cols, keep=False).any():
        return frame.reset_index(drop=True)

    aggregate_norm = str(aggregate).lower()
    if aggregate_norm not in {"first", "mean", "max", "min", "mode"}:
        raise ValueError("aggregate must be one of: first, mean, max, min, mode")

    def _aggregate_group(group: pd.DataFrame) -> pd.Series:
        row = group.iloc[0].copy()
        if aggregate_norm == "first":
            value = group[metric].iloc[0]
        elif aggregate_norm == "mode":
            value = _mode_value(group[metric])
        else:
            if _phase_metric_is_categorical(metric):
                value = _mode_value(group[metric])
            else:
                numeric = pd.to_numeric(group[metric], errors="coerce")
                if aggregate_norm == "mean":
                    value = float(numeric.mean())
                elif aggregate_norm == "max":
                    value = float(numeric.max())
                else:
                    value = float(numeric.min())
        row[metric] = value
        row["n_aggregated"] = int(len(group))
        return row

    rows = [
        _aggregate_group(group)
        for _, group in frame.groupby(axis_cols, dropna=False, sort=True)
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


@dataclass
class BatchVortexPhaseDiagramResult:
    """Phase diagram data for batch vortex dynamics."""

    frame: pd.DataFrame
    axes: tuple[str, str | None, str | None]
    metric: str = "regime"
    regime_order: tuple[str, ...] = tuple(_REGIME_ORDER)
    regime_colors: dict[str, str] = field(default_factory=lambda: dict(_REGIME_COLORS))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Number of active phase-diagram axes."""
        return sum(axis is not None for axis in self.axes)

    @property
    def plt(self):
        """Plotting accessor."""
        return BatchVortexPhaseDiagramPlotAccessor(self)

    def _repr_html_(self) -> str:
        api = api_help_html(
            self,
            title="Batch vortex phase-diagram API help",
            prefix="jobs.vortex.analyze.phase_diagram(...)",
            properties=[("plt", "Plot accessor for phase-diagram views")],
            methods=[],
            subtitle="Batch vortex regime or metric map over one to three swept parameters.",
            chrome=False,
        )
        axes_label = ", ".join(str(axis) for axis in self.axes if axis is not None)
        return node_card_html(
            self.__class__.__name__,
            icon="🧭",
            subtitle=f"{self.metric} over {axes_label or 'no axes'}",
            sections=[
                metrics_section_html(
                    [
                        ("points", str(len(self.frame)), NODE_COLOR_COMPUTE),
                        ("dimension", str(self.dimension), NODE_COLOR_ANALYSIS),
                        ("metric", str(self.metric), NODE_COLOR_UTIL),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Plot:",
                            [
                                (".plt.map()", NODE_COLOR_PLOT),
                                (".plt.scatter()", NODE_COLOR_PLOT),
                                (".plt.surface3d()", NODE_COLOR_PLOT),
                            ],
                        )
                    ]
                ),
            ],
            api=api,
            uid="batch-vortex-phase-diagram",
        )


class BatchVortexPhaseDiagramPlotAccessor:
    """Plot helpers for :class:`BatchVortexPhaseDiagramResult`."""

    def __init__(self, result: BatchVortexPhaseDiagramResult):
        self._result = result

    def _values(self) -> np.ndarray:
        if _phase_metric_is_categorical(self._result.metric):
            return _regime_codes(self._result.frame[self._result.metric])
        return pd.to_numeric(
            self._result.frame[self._result.metric],
            errors="coerce",
        ).to_numpy(dtype=float)

    def _add_regime_legend(self, ax) -> None:
        if not _phase_metric_is_categorical(self._result.metric):
            return
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=self._result.regime_colors.get(regime, "black"),
                label=regime,
            )
            for regime in self._result.regime_order
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8)

    def _colors(self) -> list[str] | np.ndarray:
        if not _phase_metric_is_categorical(self._result.metric):
            return self._values()
        return [
            self._result.regime_colors.get(str(regime), "black")
            for regime in self._result.frame[self._result.metric]
        ]

    def map(self, *, ax=None, **kwargs):
        """Plot an automatic 1D/2D/3D phase diagram."""
        if self._result.dimension == 3:
            return self.surface3d(ax=ax, **kwargs)
        if self._result.dimension == 2 and self._can_heatmap():
            return self._heatmap2d(ax=ax, **kwargs)
        return self.scatter(ax=ax, **kwargs)

    def scatter(self, *, ax=None, **kwargs):
        """Plot phase diagram points without interpolation."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, default_figsize=(6.5, 4.0), figure_kwargs=figure_kwargs)
        x_name, y_name, _z_name = self._result.axes
        x = self._result.frame[x_name].astype(float).to_numpy()
        colors = self._colors()

        if self._result.dimension == 1:
            if _phase_metric_is_categorical(self._result.metric):
                y = _regime_codes(self._result.frame[self._result.metric])
                ax.scatter(x, y, c=colors, s=70, **plot_kwargs)
                ax.set_yticks(np.arange(len(self._result.regime_order), dtype=float))
                ax.set_yticklabels(self._result.regime_order)
                self._add_regime_legend(ax)
            else:
                y = self._values()
                ax.plot(x, y, marker="o", linestyle="-", **plot_kwargs)
                ax.set_ylabel(self._result.metric)
        else:
            assert y_name is not None
            y = self._result.frame[y_name].astype(float).to_numpy()
            if _phase_metric_is_categorical(self._result.metric):
                ax.scatter(x, y, c=colors, s=70, **plot_kwargs)
                self._add_regime_legend(ax)
            else:
                sc = ax.scatter(x, y, c=colors, s=70, **plot_kwargs)
                ax.figure.colorbar(sc, ax=ax, label=self._result.metric)
            ax.set_ylabel(_coordinate_label(y_name))

        ax.set_xlabel(_coordinate_label(x_name))
        ax.set_title(f"Vortex phase diagram: {self._result.metric}")
        ax.grid(True, alpha=0.25)
        apply_axes_style(ax, style_kwargs)
        return ax

    def _can_heatmap(self) -> bool:
        x_name, y_name, _ = self._result.axes
        if y_name is None or self._result.frame.empty:
            return False
        x_unique = np.unique(self._result.frame[x_name].astype(float).to_numpy())
        y_unique = np.unique(self._result.frame[y_name].astype(float).to_numpy())
        return int(x_unique.size * y_unique.size) == int(len(self._result.frame))

    def _heatmap2d(self, *, ax=None, **kwargs):
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, default_figsize=(6.5, 4.8), figure_kwargs=figure_kwargs)
        x_name, y_name, _ = self._result.axes
        assert y_name is not None

        frame = self._result.frame.copy()
        value_col = "__phase_value__"
        frame[value_col] = self._values()
        pivot = frame.pivot(index=y_name, columns=x_name, values=value_col).sort_index()
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        mesh = ax.pcolormesh(
            pivot.columns.to_numpy(dtype=float),
            pivot.index.to_numpy(dtype=float),
            pivot.to_numpy(dtype=float),
            shading="auto",
            **plot_kwargs,
        )
        ax.set_xlabel(_coordinate_label(x_name))
        ax.set_ylabel(_coordinate_label(y_name))
        ax.set_title(f"Vortex phase diagram: {self._result.metric}")
        if _phase_metric_is_categorical(self._result.metric):
            self._add_regime_legend(ax)
        else:
            ax.figure.colorbar(mesh, ax=ax, label=self._result.metric)
        apply_axes_style(ax, style_kwargs)
        return ax

    def surface3d(self, *, ax=None, **kwargs):
        """Plot a 3D scatter phase diagram."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        if ax is None:
            import matplotlib.pyplot as plt

            fig = plt.figure(**figure_kwargs)
            ax = fig.add_subplot(111, projection="3d")
        x_name, y_name, z_name = self._result.axes
        if y_name is None or z_name is None:
            return self.scatter(ax=ax, **plot_kwargs)

        x = self._result.frame[x_name].astype(float).to_numpy()
        y = self._result.frame[y_name].astype(float).to_numpy()
        z = self._result.frame[z_name].astype(float).to_numpy()
        colors = self._colors()
        sc = ax.scatter(x, y, z, c=colors, s=70, **plot_kwargs)
        ax.set_xlabel(_coordinate_label(x_name))
        ax.set_ylabel(_coordinate_label(y_name))
        ax.set_zlabel(_coordinate_label(z_name))
        ax.set_title(f"Vortex phase diagram: {self._result.metric}")
        if _phase_metric_is_categorical(self._result.metric):
            self._add_regime_legend(ax)
        else:
            ax.figure.colorbar(sc, ax=ax, label=self._result.metric)
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        return api_help_html(
            self,
            title="Batch vortex phase diagram plot API help",
            prefix="phase_diagram.plt",
            methods=["map", "scatter", "surface3d"],
            subtitle="Plotting helper for batch vortex phase diagrams.",
        )


@dataclass
class BatchVortexSpectrumMapResult:
    """Spectrum matrix across a filtered batch of vortex simulations."""

    coordinate: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray
    component: str
    coordinate_name: str = "i_pillar_ma"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frequencies_ghz(self) -> np.ndarray:
        """Frequency axis in GHz."""
        return np.asarray(self.frequencies, dtype=float) * 1e-9

    @property
    def power_db(self) -> np.ndarray:
        """Power expressed as 10*log10(power)."""
        return 10.0 * np.log10(
            np.clip(np.asarray(self.power, dtype=float), 1e-30, None)
        )

    @property
    def plt(self):
        """Plotting accessor."""
        return BatchVortexSpectrumMapPlotAccessor(self)

    def _repr_html_(self) -> str:
        api = api_help_html(
            self,
            title="Batch vortex spectrum-map API help",
            prefix="batch.solitons.vortex.spectrum.map(...)",
            properties=[("plt", "Plot accessor for the batch spectrum map")],
            methods=[],
            subtitle="Batch result matrix containing one spectrum per sorted batch coordinate value.",
            chrome=False,
        )
        return node_card_html(
            self.__class__.__name__,
            icon="🗺️",
            subtitle=f"{self.component} spectrum map",
            sections=[
                metrics_section_html(
                    [
                        (
                            "n_runs",
                            str(int(np.asarray(self.coordinate).size)),
                            NODE_COLOR_COMPUTE,
                        ),
                        (
                            "n_freq",
                            str(int(np.asarray(self.frequencies).size)),
                            NODE_COLOR_ANALYSIS,
                        ),
                        ("coordinate", str(self.coordinate_name), NODE_COLOR_UTIL),
                    ]
                ),
                accessors_section_html(
                    [("Accessors:", [(".plt.heatmap()", NODE_COLOR_PLOT)])]
                ),
            ],
            api=api,
            uid="batch-vortex-spectrum-map",
        )


class BatchVortexSpectrumMapPlotAccessor:
    """Plot helpers for :class:`BatchVortexSpectrumMapResult`."""

    def __init__(self, result: BatchVortexSpectrumMapResult):
        self._result = result

    def heatmap(
        self, *, ax=None, as_ghz: bool = True, db_scale: bool = True, **kwargs
    ) -> Any:
        """Plot batch spectrum heatmap and return the matplotlib axis."""
        mesh_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(mesh_kwargs)
        figure_kwargs = pop_figure_kwargs(mesh_kwargs)
        colorbar = bool(mesh_kwargs.pop("colorbar", True))
        colorbar_options = mesh_kwargs.pop("colorbar_kwargs", {})
        colorbar_kwargs = {} if colorbar_options is None else dict(colorbar_options)

        ax = ensure_axis(ax, default_figsize=(6.5, 4.0), figure_kwargs=figure_kwargs)

        freqs = (
            self._result.frequencies_ghz
            if as_ghz
            else np.asarray(self._result.frequencies, dtype=float)
        )
        power = (
            self._result.power_db
            if db_scale
            else np.asarray(self._result.power, dtype=float)
        )

        mesh = ax.pcolormesh(
            np.asarray(self._result.coordinate, dtype=float),
            freqs,
            power.T,
            shading="auto",
            **mesh_kwargs,
        )
        ax.set_xlabel(_coordinate_label(self._result.coordinate_name))
        ax.set_ylabel("Frequency [GHz]" if as_ghz else "Frequency [Hz]")
        ax.set_title(f"Batch vortex {self._result.component} spectrum")

        if colorbar:
            label = "Power [dB]" if db_scale else "Power [a.u.]"
            colorbar_kwargs.setdefault("label", label)
            ax.figure.colorbar(mesh, ax=ax, **colorbar_kwargs)

        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        api = api_help_html(
            self,
            title="Batch vortex spectrum-map plot API help",
            prefix="map_result.plt",
            methods=["heatmap"],
            subtitle="Plotting helper for the batch vortex spectrum map.",
            chrome=False,
        )
        return node_card_html(
            self.__class__.__name__,
            icon="📊",
            subtitle="plot accessor",
            sections=[
                metrics_section_html([("methods", "heatmap()", NODE_COLOR_PLOT)]),
                examples_section_html(
                    "\n".join(
                        [
                            "result = batch.solitons.vortex.spectrum.map()",
                            "result.plt.heatmap()",
                        ]
                    )
                ),
            ],
            api=api,
            uid="batch-vortex-spectrum-map-plot",
        )


class BatchSolitonsInterface:
    """Batch entry point for soliton analysis namespaces."""

    def __init__(self, results: list[Any], mmpp_instance: Any | None = None):
        self._results = list(results)
        self._mmpp = mmpp_instance
        self._vortex = None
        self._skyrmion = None

    @property
    def vortex(self):
        """Batch vortex analysis namespace."""
        if self._vortex is None:
            self._vortex = BatchVortexInterface(self._results, self._mmpp)
        return self._vortex

    @property
    def skyrmion(self):
        """Batch skyrmion analysis namespace."""
        if self._skyrmion is None:
            from .skyrmion.batch import BatchSkyrmionInterface

            self._skyrmion = BatchSkyrmionInterface(self._results, self._mmpp)
        return self._skyrmion

    def __repr__(self) -> str:
        return f"BatchSolitonsInterface({len(self._results)} results)"

    def _repr_html_(self) -> str:
        import uuid as _uuid

        api = api_help_html(
            self,
            title="Batch solitons API help",
            prefix="job[:].solitons",
            properties=[
                ("vortex", "Batch vortex analysis namespace"),
                ("skyrmion", "Batch skyrmion analysis namespace"),
            ],
            methods=[],
            subtitle="Top-level batch entry point for soliton-related analysis namespaces.",
            chrome=False,
        )
        return node_card_html(
            self.__class__.__name__,
            icon="🧲",
            subtitle="batch soliton namespace",
            sections=[
                metrics_section_html(
                    [("n_results", str(len(self._results)), NODE_COLOR_COMPUTE)]
                ),
                accessors_section_html(
                    [("Namespaces:", [(".vortex", NODE_COLOR_ANALYSIS), (".skyrmion", NODE_COLOR_ANALYSIS)])]
                ),
            ],
            api=api,
            uid=f"batch-solitons-interface-{_uuid.uuid4().hex[:8]}",
        )


class BatchVortexSpectrumAccessor:
    """Batch spectrum namespace for vortex runs."""

    def __init__(self, interface: BatchVortexInterface):
        self._interface = interface

    def map(self, **kwargs) -> BatchVortexSpectrumMapResult:
        return self._interface.spectrum_map(**kwargs)

    def current_map(
        self,
        *,
        current: str = "i_pillar_ma",
        **kwargs,
    ) -> BatchVortexSpectrumMapResult:
        """Spectrum intensity map versus electric current.

        This is a convenience alias for ``spectrum_map(sort_by=current)``.
        The returned matrix has axes ``current`` × ``frequency`` and can be
        plotted with ``.plt.heatmap()``.
        """
        return self._interface.current_spectrum_map(current=current, **kwargs)

    def _repr_html_(self) -> str:
        api = api_help_html(
            self,
            title="Batch vortex spectrum API help",
            prefix="jobs.vortex.spectrum",
            methods=["map", "current_map"],
            subtitle="Namespace for building batch-wide vortex spectrum maps, exposed directly as jobs.vortex.spectrum.",
            chrome=False,
        )
        return node_card_html(
            self.__class__.__name__,
            icon="🌊",
            subtitle="batch vortex spectrum namespace",
            sections=[
                accessors_section_html(
                    [
                        (
                            "Methods:",
                            [
                                (
                                    ".map(sort_by='i_pillar_ma', ...)",
                                    NODE_COLOR_COMPUTE,
                                ),
                                (
                                    ".current_map(current='i_pillar_ma', ...)",
                                    NODE_COLOR_COMPUTE,
                                ),
                                (".heatmap()", NODE_COLOR_PLOT),
                            ],
                        )
                    ]
                ),
                examples_section_html(
                    "\n".join(
                        [
                            "spec = jobs.vortex.spectrum",
                            "result = spec.map(sort_by='i_pillar_ma', component='gyration')",
                            "mx_map = spec.map(sort_by='i_pillar_ma', source='table', component='mx')",
                            "fresh = spec.map(sort_by='i_pillar_ma', force=True)",
                            "current_map = spec.current_map(current='i_pillar_ma')",
                            "result.plt.heatmap()",
                        ]
                    )
                ),
            ],
            api=api,
            uid="batch-vortex-spectrum-accessor",
        )


class BatchVortexAnalyzeAccessor:
    """Batch vortex analysis namespace."""

    def __init__(self, interface: BatchVortexInterface):
        self._interface = interface

    def phase_diagram(
        self,
        *,
        x: str = "i_pillar_ma",
        y: str | None = None,
        z: str | None = None,
        metric: str = "regime",
        aggregate: str | None = None,
        show_progress: bool = True,
        **summary_kwargs,
    ) -> BatchVortexPhaseDiagramResult:
        """Build a vortex regime or metric phase diagram from batch summary."""
        return self._interface.phase_diagram(
            x=x,
            y=y,
            z=z,
            metric=metric,
            aggregate=aggregate,
            show_progress=show_progress,
            **summary_kwargs,
        )

    def _repr_html_(self) -> str:
        api = api_help_html(
            self,
            title="Batch vortex analyze API help",
            prefix="jobs.vortex.analyze",
            methods=["phase_diagram"],
            subtitle="Analysis namespace for batch vortex phase diagrams.",
            chrome=False,
        )
        return node_card_html(
            self.__class__.__name__,
            icon="🧭",
            subtitle="batch vortex analysis namespace",
            sections=[
                accessors_section_html(
                    [
                        (
                            "Methods:",
                            [(".phase_diagram(...)", NODE_COLOR_ANALYSIS)],
                        )
                    ]
                ),
                examples_section_html(
                    "\n".join(
                        [
                            "pdg = jobs.vortex.analyze.phase_diagram(x='i_pillar_ma')",
                            "pdg.plt.map()",
                            "pdg2 = jobs.vortex.analyze.phase_diagram(x='i_pillar_ma', y='epsilonprime')",
                        ]
                    )
                ),
            ],
            api=api,
            uid="batch-vortex-analyze-accessor",
        )


class BatchVortexInterface:
    """Batch helpers for vortex trajectory, spectrum and regime analysis."""

    def __init__(self, results: list[Any], mmpp_instance: Any | None = None):
        self._results = list(results)
        self._mmpp = mmpp_instance
        self._spectrum = None
        self._analyze = None

    @property
    def plt(self):
        """Plotting accessor for batch vortex summaries."""
        return BatchVortexPlotAccessor(self)

    @property
    def spectrum(self):
        """Batch spectrum namespace."""
        if self._spectrum is None:
            self._spectrum = BatchVortexSpectrumAccessor(self)
        return self._spectrum

    @property
    def analyze(self):
        """Batch vortex analysis namespace."""
        if self._analyze is None:
            self._analyze = BatchVortexAnalyzeAccessor(self)
        return self._analyze

    def _ordered_results(self, sort_by: str | None = "i_pillar_ma") -> list[Any]:
        if sort_by is None:
            return list(self._results)

        def key(result):
            attrs = getattr(result, "attrs", {}) or {}
            if sort_by in attrs:
                return (0, _coerce_numeric(attrs[sort_by], default=float("inf")))
            return (1, str(getattr(result, "path", "")))

        return sorted(self._results, key=key)

    @staticmethod
    def _classify_regime(
        row: pd.Series,
        *,
        power_floor_rel: float,
        radius_floor_nm: float,
        expulsion_ratio: float,
    ) -> str:
        if row.get("status") == "error":
            return "error"
        if int(row.get("n_p_switch", 0)) > 0:
            return "remagnetization"
        if int(row.get("n_expulsion", 0)) > 0:
            return "collision"
        radius_max_rel = _coerce_numeric(row.get("r_max_rel", np.nan))
        if np.isfinite(radius_max_rel) and radius_max_rel >= float(expulsion_ratio):
            return "collision"
        if int(row.get("n_gc_switch", 0)) > 0:
            return "intermittent"
        peak_power_rel = _coerce_numeric(row.get("peak_power_rel", np.nan))
        radius_mean_nm = _coerce_numeric(row.get("r_mean_nm", np.nan))
        if (
            np.isfinite(peak_power_rel) and peak_power_rel < float(power_floor_rel)
        ) or (np.isfinite(radius_mean_nm) and radius_mean_nm < float(radius_floor_nm)):
            return "damped"
        return "stable_gyro"

    def summary(
        self,
        *,
        sort_by: str | None = "i_pillar_ma",
        steady_state: bool = True,
        spectrum_method: str = "welch",
        nperseg: int | None = 512,
        noverlap: int | None = 256,
        radius_threshold: float = 0.6,
        expulsion_ratio: float = 0.95,
        power_floor_rel: float = 0.02,
        radius_floor_nm: float = 0.2,
        show_progress: bool = True,
        parallel: bool | int | str = False,
        max_workers: int | None = None,
        profile_memory: bool = False,
    ) -> pd.DataFrame:
        """Summarize vortex dynamics across the batch."""
        ordered_results = self._ordered_results(sort_by)
        mem_start = _memory_mb() if profile_memory else float("nan")

        def _summarize_one(index: int, result: Any) -> dict[str, Any]:
            attrs = getattr(result, "attrs", {}) or {}
            row: dict[str, Any] = {
                "index": index,
                "path": getattr(result, "path", None),
                "status": "ok",
                "error": None,
            }
            row.update(
                {
                    key: attrs.get(key)
                    for key in (
                        "i_pillar_ma",
                        "ma",
                        "Jdc",
                        "ni",
                        "addoe",
                        "EnableOersted",
                    )
                }
            )

            try:
                vortex = result.solitons.vortex
                trajectory = (
                    vortex.trajectory.steady_state()
                    if steady_state
                    else vortex.trajectory.raw
                )
                spectrum_kwargs = _resolved_spectrum_kwargs(
                    trajectory,
                    method=spectrum_method,
                    nperseg=nperseg,
                    noverlap=noverlap,
                )
                gyration = vortex.spectrum.gyration(
                    trajectory=trajectory,
                    **spectrum_kwargs,
                )
                breathing = vortex.spectrum.breathing(
                    trajectory=trajectory,
                    **spectrum_kwargs,
                )
                p_switches = vortex.events.polarity_switches(trajectory=trajectory)
                gc_switches = vortex.events.state_switches(
                    trajectory=trajectory,
                    radius_threshold=radius_threshold,
                )
                expulsions = vortex.events.core_expulsions(
                    trajectory=trajectory,
                    expulsion_ratio=expulsion_ratio,
                )

                peak_power = (
                    float(np.max(gyration.power))
                    if getattr(gyration.power, "size", 0)
                    else 0.0
                )
                disk_radius = _disk_radius_from_attrs(attrs)

                row.update(
                    {
                        "n_samples": int(len(trajectory.time)),
                        "peak_gyr_ghz": float(gyration.peak_frequency_ghz),
                        "peak_breath_ghz": float(breathing.peak_frequency_ghz),
                        "peak_power": peak_power,
                        "r_mean_nm": float(np.mean(trajectory.r) * 1e9),
                        "r_max_nm": float(np.max(trajectory.r) * 1e9),
                        "r_max_rel": float(np.max(trajectory.r) / disk_radius)
                        if np.isfinite(disk_radius) and disk_radius > 0.0
                        else float("nan"),
                        "n_p_switch": int(len(p_switches)),
                        "n_gc_switch": int(len(gc_switches)),
                        "n_expulsion": int(len(expulsions)),
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "error",
                        "error": str(exc),
                        "n_samples": 0,
                        "peak_gyr_ghz": float("nan"),
                        "peak_breath_ghz": float("nan"),
                        "peak_power": float("nan"),
                        "r_mean_nm": float("nan"),
                        "r_max_nm": float("nan"),
                        "r_max_rel": float("nan"),
                        "n_p_switch": 0,
                        "n_gc_switch": 0,
                        "n_expulsion": 0,
                    }
                )

            return row

        if parallel:
            workers = int(
                max_workers or (parallel if isinstance(parallel, int) else 0) or 4
            )
            workers = max(1, workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                mapped = executor.map(
                    lambda item: _summarize_one(*item), enumerate(ordered_results)
                )
                rows = list(
                    _progress_iter(
                        mapped,
                        total=len(ordered_results),
                        desc="Summarizing vortex batch",
                        enabled=show_progress,
                    )
                )
        else:
            iterator = _progress_iter(
                enumerate(ordered_results),
                total=len(ordered_results),
                desc="Summarizing vortex batch",
                enabled=show_progress,
            )
            rows = [_summarize_one(index, result) for index, result in iterator]

        if not rows:
            return pd.DataFrame(
                columns=[
                    "index",
                    "path",
                    "status",
                    "error",
                    "i_pillar_ma",
                    "ma",
                    "Jdc",
                    "ni",
                    "addoe",
                    "EnableOersted",
                    "n_samples",
                    "peak_gyr_ghz",
                    "peak_breath_ghz",
                    "peak_power",
                    "peak_power_rel",
                    "r_mean_nm",
                    "r_max_nm",
                    "r_max_rel",
                    "n_p_switch",
                    "n_gc_switch",
                    "n_expulsion",
                    "regime",
                ]
            )

        frame = pd.DataFrame(rows)
        peak_power_values = frame["peak_power"].to_numpy(dtype=float)
        finite_peak_power = peak_power_values[np.isfinite(peak_power_values)]
        peak_power_max = (
            float(np.max(finite_peak_power)) if finite_peak_power.size else 1.0
        )
        if peak_power_max <= 0.0:
            peak_power_max = 1.0
        frame["peak_power_rel"] = frame["peak_power"].astype(float) / float(
            peak_power_max
        )
        frame["regime"] = frame.apply(
            self._classify_regime,
            axis=1,
            power_floor_rel=power_floor_rel,
            radius_floor_nm=radius_floor_nm,
            expulsion_ratio=expulsion_ratio,
        )
        if profile_memory:
            mem_end = _memory_mb()
            frame.attrs["memory_profile"] = {
                "memory_start_mb": mem_start,
                "memory_end_mb": mem_end,
                "memory_delta_mb": mem_end - mem_start
                if np.isfinite(mem_start) and np.isfinite(mem_end)
                else float("nan"),
            }
        return frame

    def regimes(self, **kwargs) -> pd.DataFrame:
        """Alias for :meth:`summary` to emphasize regime classification."""
        return self.summary(**kwargs)

    def _axis_values_from_attrs(self, axis_name: str, ordered_results: list[Any]) -> dict[int, float]:
        """Return numeric axis values keyed by summary row index."""
        values: dict[int, float] = {}
        for index, result in enumerate(ordered_results):
            attrs = getattr(result, "attrs", {}) or {}
            if axis_name in attrs:
                values[index] = _coerce_numeric(attrs.get(axis_name))
        return values

    def _ensure_phase_axis(
        self,
        frame: pd.DataFrame,
        *,
        axis_name: str,
        ordered_results: list[Any],
        errors: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Ensure a requested phase-diagram axis is present and numeric."""
        if axis_name not in frame.columns:
            values = self._axis_values_from_attrs(axis_name, ordered_results)
            if not values:
                raise ValueError(
                    f"Phase-diagram axis {axis_name!r} is unavailable in summary or attrs"
                )
            frame = frame.copy()
            frame[axis_name] = frame["index"].map(values)

        numeric = pd.to_numeric(frame[axis_name], errors="coerce")
        missing = frame[numeric.isna()]
        if not missing.empty:
            for _, row in missing.iterrows():
                errors.append(
                    {
                        "index": int(row.get("index", -1)),
                        "path": row.get("path"),
                        "axis": axis_name,
                        "error": "missing or non-numeric axis value",
                    }
                )
        frame = frame.loc[~numeric.isna()].copy()
        frame[axis_name] = numeric.loc[frame.index].astype(float)
        return frame

    def phase_diagram(
        self,
        *,
        x: str = "i_pillar_ma",
        y: str | None = None,
        z: str | None = None,
        metric: str = "regime",
        aggregate: str | None = None,
        show_progress: bool = True,
        **summary_kwargs,
    ) -> BatchVortexPhaseDiagramResult:
        """Return a 1D/2D/3D phase diagram over batch parameters."""
        if not x:
            raise ValueError("phase_diagram requires x axis")
        metric_norm = str(metric)
        if metric_norm not in _PHASE_METRICS:
            allowed = ", ".join(sorted(_PHASE_METRICS))
            raise ValueError(f"metric must be one of: {allowed}")

        ordered_results = self._ordered_results(x)
        frame = self.summary(sort_by=x, show_progress=show_progress, **summary_kwargs)
        axes = (x, y, z)
        errors: list[dict[str, Any]] = []
        for axis_name in [name for name in axes if name is not None]:
            frame = self._ensure_phase_axis(
                frame,
                axis_name=axis_name,
                ordered_results=ordered_results,
                errors=errors,
            )
        if metric_norm not in frame.columns:
            raise ValueError(f"metric {metric_norm!r} is unavailable in summary")

        aggregate_mode = aggregate or ("first" if metric_norm == "regime" else "mean")
        frame = _aggregate_phase_frame(
            frame,
            axes=axes,
            metric=metric_norm,
            aggregate=aggregate_mode,
        )
        metadata = {
            "dimension": sum(axis is not None for axis in axes),
            "metric": metric_norm,
            "aggregate": aggregate_mode,
            "errors": errors,
            "n_input": int(len(ordered_results)),
            "n_points": int(len(frame)),
        }
        return BatchVortexPhaseDiagramResult(
            frame=frame,
            axes=axes,
            metric=metric_norm,
            metadata=metadata,
        )

    def current_phase_diagram(
        self,
        *,
        current: str = "i_pillar_ma",
        **kwargs,
    ) -> BatchVortexPhaseDiagramResult:
        """Convenience alias for a phase diagram over electric current."""
        return self.phase_diagram(x=current, **kwargs)

    def spectrum_map(
        self,
        *,
        component: str = "gyration",
        source: str = "auto",
        dataset_name: str | None = None,
        z_index: int | None = None,
        spatial_reduction: str = "mean",
        sort_by: str = "i_pillar_ma",
        steady_state: bool = True,
        spectrum_method: str = "welch",
        nperseg: int | None = 512,
        noverlap: int | None = 256,
        show_progress: bool = True,
        parallel: bool | int | str = False,
        max_workers: int | None = None,
        profile_memory: bool = False,
        exclude_annihilated: bool = False,
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
        force: bool = False,
        multiprocessing: bool = False,
    ) -> BatchVortexSpectrumMapResult:
        """Return batch spectrum matrix across the filtered simulations.

        Parameters
        ----------
        component : str
            ``gyration`` or ``breathing`` for processed trajectory spectra, or
            ``mx``/``my``/``mz`` for direct magnetization/table spectra.
        source : str
            ``auto`` (default), ``processed``, ``table`` or ``magnetization``.
            ``auto`` keeps old behavior for ``gyration``/``breathing`` and
            switches to direct magnetization for ``mx``/``my``/``mz``.
        dataset_name : str, optional
            Magnetization dataset for ``source='magnetization'``. When omitted,
            the largest ``m`` dataset is selected when possible.
        z_index : int, optional
            Optional z-layer index for raw magnetization datasets.
        spatial_reduction : str
            Spatial reduction for raw magnetization component traces:
            ``mean`` (default), ``sum``, ``max`` or ``min``.
        exclude_annihilated : bool
            When ``True``, simulations where core annihilation or polarity
            reversal is detected are silently skipped (their row is omitted
            from the result matrix).  A Python warning is still emitted for
            each excluded simulation.  Default ``False`` (include all, show
            warning annotation).
        cache : bool
            Persist the computed map in the sweep post-processing directory and
            reuse it on repeated calls with the same inputs. Default ``True``.
        cache_dir : path-like, optional
            Explicit artifact directory. When omitted, the cache is stored under
            ``<sweep root>/.mmpp_postprocessing/vortex/spectrum_map/<hash>/``.
        force : bool
            Recompute and overwrite the cached artifact even when it exists.
        multiprocessing : bool
            Use a process pool when combined with ``parallel=True``. Threads are
            the default because many notebook job objects are not picklable.
        """
        coordinate: list[float] = []
        power_rows: list[np.ndarray] = []
        frequency_ref: np.ndarray | None = None
        errors: list[dict[str, Any]] = []

        ordered_results = self._ordered_results(sort_by)
        source_norm = _resolve_spectrum_map_source(
            source=source,
            component=component,
            ordered_results=ordered_results,
        )
        component_norm = _validate_spectrum_map_component(
            source=source_norm,
            component=component,
        )
        cache_config = _spectrum_map_cache_config(
            component=component_norm,
            source=source_norm,
            dataset_name=dataset_name,
            z_index=z_index,
            spatial_reduction=spatial_reduction,
            sort_by=sort_by,
            steady_state=steady_state,
            spectrum_method=spectrum_method,
            nperseg=nperseg,
            noverlap=noverlap,
            exclude_annihilated=exclude_annihilated,
            ordered_results=ordered_results,
        )
        cache_key = _spectrum_map_cache_key(cache_config)
        cache_path = (
            _resolve_spectrum_map_cache_dir(
                cache_dir=cache_dir,
                cache_key=cache_key,
                mmpp_instance=self._mmpp,
                ordered_results=ordered_results,
            )
            if cache
            else None
        )
        if cache_path is not None and not force:
            cached = _load_spectrum_map_cache(cache_path, cache_config)
            if cached is not None:
                return cached

        mem_start = _memory_mb() if profile_memory else float("nan")

        work_items = [
            (
                index,
                result,
                sort_by,
                steady_state,
                spectrum_method,
                nperseg,
                noverlap,
                component_norm,
                source_norm,
                dataset_name,
                z_index,
                spatial_reduction,
                exclude_annihilated,
            )
            for index, result in enumerate(ordered_results)
        ]

        if parallel:
            workers = int(
                max_workers
                or (
                    parallel
                    if isinstance(parallel, int) and not isinstance(parallel, bool)
                    else 0
                )
                or 4
            )
            workers = max(1, workers)
            executor_cls = ProcessPoolExecutor if multiprocessing else ThreadPoolExecutor
            if isinstance(parallel, str) and parallel.lower() in {
                "process",
                "processes",
                "multiprocessing",
            }:
                executor_cls = ProcessPoolExecutor
            with executor_cls(max_workers=workers) as executor:
                mapped_iter = executor.map(_compute_vortex_spectrum_map_one, work_items)
                mapped = list(
                    _progress_iter(
                        mapped_iter,
                        total=len(ordered_results),
                        desc=f"Computing {source_norm}:{component_norm} spectrum map",
                        enabled=show_progress,
                    )
                )
        else:
            iterator = _progress_iter(
                work_items,
                total=len(ordered_results),
                desc=f"Computing {source_norm}:{component_norm} spectrum map",
                enabled=show_progress,
            )
            mapped = [
                _compute_vortex_spectrum_map_one(work_item) for work_item in iterator
            ]

        for coord_value, frequencies, power, error in mapped:
            if error is not None:
                errors.append(error)
                continue
            assert frequencies is not None and power is not None
            if frequency_ref is None:
                frequency_ref = frequencies
            elif frequencies.shape != frequency_ref.shape or not np.allclose(
                frequencies, frequency_ref
            ):
                power = np.interp(
                    frequency_ref, frequencies, power, left=0.0, right=0.0
                )
            coordinate.append(coord_value)
            power_rows.append(power)

        if frequency_ref is None or not power_rows:
            metadata = {
                "errors": errors,
                "steady_state": steady_state,
                "source": source_norm,
                "magnetization_component": component_norm
                if source_norm in {"table", "magnetization"}
                else None,
            }
            if profile_memory:
                mem_end = _memory_mb()
                metadata["memory_profile"] = {
                    "memory_start_mb": mem_start,
                    "memory_end_mb": mem_end,
                    "memory_delta_mb": mem_end - mem_start
                    if np.isfinite(mem_start) and np.isfinite(mem_end)
                        else float("nan"),
                }
            result = BatchVortexSpectrumMapResult(
                coordinate=np.asarray([], dtype=float),
                frequencies=np.asarray([], dtype=float),
                power=np.zeros((0, 0), dtype=float),
                component=f"{source_norm}:{component_norm}"
                if source_norm in {"table", "magnetization"}
                else component_norm,
                coordinate_name=sort_by,
                metadata=metadata,
            )
            if cache_path is not None:
                _store_spectrum_map_cache(
                    result,
                    cache_path=cache_path,
                    cache_key=cache_key,
                    config=cache_config,
                )
                result.metadata["cache"] = {
                    "status": "stored",
                    "path": str(cache_path),
                    "key": cache_key,
                    "force": bool(force),
                }
            return result

        metadata = {
            "errors": errors,
            "steady_state": steady_state,
            "source": source_norm,
            "magnetization_component": component_norm
            if source_norm in {"table", "magnetization"}
            else None,
        }
        if profile_memory:
            mem_end = _memory_mb()
            metadata["memory_profile"] = {
                "memory_start_mb": mem_start,
                "memory_end_mb": mem_end,
                "memory_delta_mb": mem_end - mem_start
                if np.isfinite(mem_start) and np.isfinite(mem_end)
                else float("nan"),
            }
        result = BatchVortexSpectrumMapResult(
            coordinate=np.asarray(coordinate, dtype=float),
            frequencies=np.asarray(frequency_ref, dtype=float),
            power=np.vstack(power_rows),
            component=f"{source_norm}:{component_norm}"
            if source_norm in {"table", "magnetization"}
            else component_norm,
            coordinate_name=sort_by,
            metadata=metadata,
        )
        if cache_path is not None:
            _store_spectrum_map_cache(
                result,
                cache_path=cache_path,
                cache_key=cache_key,
                config=cache_config,
            )
            result.metadata["cache"] = {
                "status": "stored",
                "path": str(cache_path),
                "key": cache_key,
                "force": bool(force),
            }
        return result

    def current_spectrum_map(
        self,
        *,
        current: str = "i_pillar_ma",
        **kwargs,
    ) -> BatchVortexSpectrumMapResult:
        """Return vortex spectrum intensity map versus electric current.

        Examples
        --------
        ``jobs.vortex.current_spectrum_map(current="i_pillar_ma")``
            returns a :class:`BatchVortexSpectrumMapResult`.
        ``jobs.vortex.current_spectrum_map().plt.heatmap()``
            plots intensity versus current and frequency.
        """
        return self.spectrum_map(sort_by=current, **kwargs)

    def frequency_sweep(
        self,
        *,
        current: str = "auto",
        method: str = "geometric",
        sort_by: str = "i_pillar_ma",
        steady_state: bool = False,
        t_min: float | None = None,
        transient_fraction: float | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Extract vortex gyrotropic frequency versus current across the batch."""
        ordered_results = self._ordered_results(sort_by)
        rows: list[dict[str, Any]] = []
        iterator = _progress_iter(
            enumerate(ordered_results),
            total=len(ordered_results),
            desc="Extracting vortex frequency sweep",
            enabled=show_progress,
        )
        for index, result in iterator:
            attrs = getattr(result, "attrs", {}) or {}
            row: dict[str, Any] = {
                "index": index,
                "path": getattr(result, "path", None),
                "status": "ok",
                "error": None,
            }
            try:
                if current == "auto":
                    current_ma = _coerce_numeric(
                        attrs.get("i_pillar_ma", attrs.get("ma", np.nan))
                    )
                    current_a = (
                        current_ma * 1e-3 if np.isfinite(current_ma) else float("nan")
                    )
                else:
                    raw = attrs.get(current, np.nan)
                    current_a = _coerce_numeric(raw)
                    current_ma = current_a * 1e3

                vortex = result.solitons.vortex
                trajectory = (
                    vortex.trajectory.steady_state()
                    if steady_state
                    else vortex.trajectory.raw
                )
                method_norm = method.lower()
                if method_norm == "geometric":
                    frequency_hz = vortex.trajectory.phase.mean_frequency(
                        center="mean",
                        t_min=t_min,
                        transient_fraction=transient_fraction,
                        unit="hz",
                    )
                    frequency_fft_hz = vortex.spectrum.gyration(
                        trajectory=trajectory,
                        method="periodogram",
                    ).peak_frequency_hz
                elif method_norm in {"fft", "spectrum"}:
                    frequency_fft_hz = vortex.spectrum.gyration(
                        trajectory=trajectory,
                        method="periodogram",
                    ).peak_frequency_hz
                    frequency_hz = frequency_fft_hz
                else:
                    raise ValueError("method must be 'geometric' or 'fft'")

                disk_radius = _disk_radius_from_attrs(attrs)
                row.update(
                    {
                        "I_A": float(current_a),
                        "I_mA": float(current_ma),
                        "frequency_geom_hz": float(frequency_hz),
                        "frequency_fft_hz": float(frequency_fft_hz),
                        "r_mean_nm": float(np.mean(trajectory.r) * 1e9),
                        "r_max_nm": float(np.max(trajectory.r) * 1e9),
                        "r_max_rel": float(np.max(trajectory.r) / disk_radius)
                        if np.isfinite(disk_radius) and disk_radius > 0.0
                        else float("nan"),
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "error",
                        "error": str(exc),
                        "I_A": float("nan"),
                        "I_mA": float("nan"),
                        "frequency_geom_hz": float("nan"),
                        "frequency_fft_hz": float("nan"),
                        "r_mean_nm": float("nan"),
                        "r_max_nm": float("nan"),
                        "r_max_rel": float("nan"),
                    }
                )
            rows.append(row)
        frame = pd.DataFrame(rows)
        if not frame.empty and "I_mA" in frame:
            frame = frame.sort_values("I_mA", kind="mergesort").reset_index(drop=True)
        return frame

    def interactive(
        self,
        index: int = 0,
        *,
        sort_by: str | None = "i_pillar_ma",
        figsize: tuple[float, float] = (10, 7),
        dpi: int = 100,
        trajectory_source: str = "magnetization",
        center_mode: str = "auto",
    ):
        """Open one vortex interactive dashboard from this batch.

        Batch-level interactive mode intentionally displays a single selected
        result. Rendering every run at once creates stacked notebook outputs and
        makes Matplotlib/ipympl backends duplicate canvases.

        Parameters
        ----------
        trajectory_source : {"magnetization", "table", "compare"}
            Default source selected in the Trajectory tab.
        center_mode : {"auto", "orbit", "disk", "raw"}
            Default centering mode for Core tracking and Trajectory plots.
        """
        ordered = self._ordered_results(sort_by)
        if not ordered:
            raise ValueError(
                "Cannot open batch vortex interactive dashboard for an empty batch"
            )

        selected = ordered[index]
        vortex = selected.solitons.vortex
        return vortex.interactive(
            figsize=figsize,
            dpi=dpi,
            trajectory_source=trajectory_source,
            center_mode=center_mode,
        )

    def __repr__(self) -> str:
        return f"BatchVortexInterface({len(self._results)} results)"

    def _repr_html_(self) -> str:
        api = api_help_html(
            self,
            title="Batch vortex API help",
            prefix="jobs.vortex",
            properties=[
                ("analyze", "Batch analysis namespace"),
                ("plt", "Batch plotting accessor"),
                ("spectrum", "Batch spectrum namespace"),
            ],
            methods=[
                "summary",
                "phase_diagram",
                "spectrum_map",
                "current_spectrum_map",
                "frequency_sweep",
                "interactive",
            ],
            subtitle="Batch helpers for vortex trajectory, spectrum and regime analysis, exposed directly as jobs.vortex.",
            chrome=False,
        )
        return node_card_html(
            self.__class__.__name__,
            icon="🌀",
            subtitle="batch vortex namespace exposed directly as jobs.vortex",
            sections=[
                metrics_section_html(
                    [
                        ("n_results", str(len(self._results)), NODE_COLOR_COMPUTE),
                        ("summary", "batch diagnostics", NODE_COLOR_ANALYSIS),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Methods:",
                            [
                                (".summary(...)", NODE_COLOR_COMPUTE),
                                (".phase_diagram(...)", NODE_COLOR_ANALYSIS),
                                (
                                    ".spectrum_map(sort_by='i_pillar_ma', ...)",
                                    NODE_COLOR_ANALYSIS,
                                ),
                                (
                                    ".current_spectrum_map(current='i_pillar_ma', ...)",
                                    NODE_COLOR_ANALYSIS,
                                ),
                                (".frequency_sweep(...)", NODE_COLOR_UTIL),
                                (".interactive(...)", NODE_COLOR_PLOT),
                            ],
                        ),
                        (
                            "Namespaces:",
                            [
                                (".analyze", NODE_COLOR_ANALYSIS),
                                (".plt", NODE_COLOR_PLOT),
                                (".spectrum", NODE_COLOR_ANALYSIS),
                            ],
                        ),
                    ]
                ),
                examples_section_html(
                    "\n".join(
                        [
                            "# Direct batch spectrum map",
                            "spec_map = jobs.vortex.spectrum_map(sort_by='i_pillar_ma')",
                            "spec_map.plt.heatmap()",
                            "mx = jobs.vortex.spectrum_map(sort_by='i_pillar_ma', source='table', component='mx')",
                            "mz = jobs.vortex.spectrum_map(sort_by='i_pillar_ma', source='magnetization', component='mz')",
                            "fresh = jobs.vortex.spectrum_map(sort_by='i_pillar_ma', force=True)",
                            "fast = jobs.vortex.spectrum_map(sort_by='i_pillar_ma', parallel=True, max_workers=4)",
                            "proc = jobs.vortex.spectrum_map(sort_by='i_pillar_ma', parallel='process', max_workers=4)",
                            "",
                            "# Phase diagram / regime map",
                            "pdg = jobs.vortex.analyze.phase_diagram(x='i_pillar_ma')",
                            "pdg.plt.map()",
                            "",
                            "# Shortcut for current-like coordinates",
                            "jobs.vortex.current_spectrum_map(current='i_pillar_ma').plt.heatmap()",
                        ]
                    ),
                    title="Batch Vortex Workflows",
                ),
            ],
            api=api,
            uid="batch-vortex-interface",
        )


class BatchVortexPlotAccessor:
    """Plot helpers for :class:`BatchVortexInterface`."""

    def __init__(self, interface: BatchVortexInterface):
        self._interface = interface

    def spectrum_map(self, **kwargs):
        """Compute and plot batch spectrum map."""
        map_result = self._interface.spectrum_map(**kwargs)
        return map_result.plt.heatmap()

    def current_spectrum_map(
        self,
        *,
        current: str = "i_pillar_ma",
        **kwargs,
    ):
        """Compute and plot spectrum intensity versus current and frequency."""
        map_result = self._interface.current_spectrum_map(
            current=current,
            **kwargs,
        )
        return map_result.plt.heatmap()

    def phase_diagram(self, **kwargs):
        """Compute and plot a batch vortex phase diagram."""
        result = self._interface.phase_diagram(**kwargs)
        return result.plt.map()

    def orbit_radius(
        self,
        *,
        sort_by: str = "i_pillar_ma",
        ax=None,
        show_progress: bool = True,
        save=None,
        **summary_kwargs,
    ):
        style_kwargs = pop_axes_style_kwargs(summary_kwargs)
        figure_kwargs = pop_figure_kwargs(summary_kwargs)
        axis = ensure_axis(ax, default_figsize=(7.0, 3.2), figure_kwargs=figure_kwargs)
        frame = self._interface.summary(
            sort_by=sort_by,
            show_progress=show_progress,
            **summary_kwargs,
        )
        if not frame.empty:
            x = frame[sort_by].astype(float).to_numpy()
            axis.plot(x, frame["r_mean_nm"], marker="o", label="mean")
            axis.plot(x, frame["r_max_nm"], marker="s", label="max")
            axis.legend(frameon=False)
        axis.set_xlabel(_coordinate_label(sort_by))
        axis.set_ylabel("Orbit radius [nm]")
        axis.set_title("Vortex orbit radius")
        apply_axes_style(axis, style_kwargs)
        if save is not None:
            axis.figure.savefig(save)
        return axis

    def regimes(
        self,
        *,
        sort_by: str = "i_pillar_ma",
        ax=None,
        show_progress: bool = True,
        **summary_kwargs,
    ):
        """Plot regime classification along the selected batch coordinate."""
        style_kwargs = pop_axes_style_kwargs(summary_kwargs)
        figure_kwargs = pop_figure_kwargs(summary_kwargs)
        axis = ensure_axis(ax, default_figsize=(7.0, 2.8), figure_kwargs=figure_kwargs)

        frame = self._interface.summary(
            sort_by=sort_by,
            show_progress=show_progress,
            **summary_kwargs,
        )
        if frame.empty:
            axis.set_title("No vortex results")
            axis.set_xlabel(_coordinate_label(sort_by))
            axis.set_ylabel("Regime")
            apply_axes_style(axis, style_kwargs)
            return axis

        mapping = {name: idx for idx, name in enumerate(_REGIME_ORDER)}
        x = frame[sort_by].astype(float).to_numpy()
        y = np.array(
            [mapping.get(item, mapping["error"]) for item in frame["regime"]],
            dtype=float,
        )
        colors = [_REGIME_COLORS.get(item, "black") for item in frame["regime"]]

        axis.scatter(x, y, c=colors, s=70, zorder=3)
        axis.plot(x, y, color="0.75", linewidth=1.0, zorder=1)
        axis.set_yticks(np.arange(len(_REGIME_ORDER), dtype=float))
        axis.set_yticklabels(_REGIME_ORDER)
        axis.set_xlabel(_coordinate_label(sort_by))
        axis.set_ylabel("Regime")
        axis.set_title("Vortex regime map")
        axis.grid(True, axis="x", alpha=0.25)
        apply_axes_style(axis, style_kwargs)
        return axis

    def frequency_vs_current(
        self,
        *,
        model_df: pd.DataFrame | None = None,
        ax=None,
        show_progress: bool = True,
        **sweep_kwargs,
    ):
        """Plot extracted gyrotropic frequency versus current with optional model overlay."""
        style_kwargs = pop_axes_style_kwargs(sweep_kwargs)
        figure_kwargs = pop_figure_kwargs(sweep_kwargs)
        axis = ensure_axis(ax, default_figsize=(7.0, 3.2), figure_kwargs=figure_kwargs)
        frame = self._interface.frequency_sweep(
            show_progress=show_progress,
            **sweep_kwargs,
        )
        if not frame.empty:
            axis.plot(
                frame["I_mA"].astype(float),
                frame["frequency_geom_hz"].astype(float) * 1e-9,
                marker="o",
                linestyle="",
                label="MuMax/core",
            )
        if model_df is not None and not model_df.empty:
            model_freq_col = (
                "frequency_geom_hz"
                if "frequency_geom_hz" in model_df
                else "frequency_hz"
            )
            axis.plot(
                model_df["I_mA"].astype(float),
                model_df[model_freq_col].astype(float) * 1e-9,
                linewidth=1.8,
                label="Thiele model",
            )
        axis.set_xlabel("Current [mA]")
        axis.set_ylabel("Frequency [GHz]")
        axis.set_title("Vortex gyrotropic frequency vs current")
        if len(axis.lines) > 1:
            axis.legend(frameon=False)
        apply_axes_style(axis, style_kwargs)
        return axis

    def dashboard(
        self,
        *,
        sort_by: str = "i_pillar_ma",
        show_progress: bool = True,
        figsize: tuple[float, float] = (11.5, 8.0),
        dpi: int | None = None,
        save=None,
        **summary_kwargs,
    ):
        """Plot a compact set of vortex batch diagnostics."""
        import matplotlib.pyplot as plt

        frame = self._interface.summary(
            sort_by=sort_by,
            show_progress=show_progress,
            **summary_kwargs,
        )
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, sharex=True)
        flat_axes = axes.ravel()

        if frame.empty:
            for axis in flat_axes:
                axis.set_axis_off()
            fig.suptitle("No vortex results")
            return fig, axes, frame

        x = frame[sort_by].astype(float).to_numpy()
        regime_colors = [_REGIME_COLORS.get(item, "black") for item in frame["regime"]]

        flat_axes[0].plot(x, frame["peak_gyr_ghz"], marker="o", label="gyration")
        flat_axes[0].plot(x, frame["peak_breath_ghz"], marker="s", label="breathing")
        flat_axes[0].set_ylabel("Frequency [GHz]")
        flat_axes[0].set_title("Dominant frequencies")
        flat_axes[0].legend(frameon=False)
        flat_axes[0].grid(True, alpha=0.25)

        flat_axes[1].plot(x, frame["r_mean_nm"], marker="o", label="mean radius")
        flat_axes[1].plot(x, frame["r_max_nm"], marker="s", label="max radius")
        flat_axes[1].set_ylabel("Radius [nm]")
        flat_axes[1].set_title("Orbit size")
        flat_axes[1].legend(frameon=False)
        flat_axes[1].grid(True, alpha=0.25)

        flat_axes[2].plot(x, frame["n_p_switch"], marker="o", label="polarity")
        flat_axes[2].plot(x, frame["n_gc_switch"], marker="s", label="G/C")
        flat_axes[2].plot(x, frame["n_expulsion"], marker="^", label="expulsion")
        flat_axes[2].set_xlabel(_coordinate_label(sort_by))
        flat_axes[2].set_ylabel("Count")
        flat_axes[2].set_title("Detected events")
        flat_axes[2].legend(frameon=False)
        flat_axes[2].grid(True, alpha=0.25)

        flat_axes[3].scatter(x, frame["peak_power_rel"], c=regime_colors, s=70)
        flat_axes[3].plot(x, frame["peak_power_rel"], color="0.75", linewidth=1.0)
        flat_axes[3].set_xlabel(_coordinate_label(sort_by))
        flat_axes[3].set_ylabel("Relative power")
        flat_axes[3].set_title("Mode strength / regime")
        flat_axes[3].grid(True, alpha=0.25)

        fig.tight_layout()
        if save is not None:
            fig.savefig(save)
        return fig, axes, frame

    def orbits(
        self,
        *,
        sort_by: str = "i_pillar_ma",
        ax=None,
        show_progress: bool = True,
        colorbar: bool = True,
        grid_alpha: float = 0.2,
        show_disk_boundary: bool = True,
        disk_radius: float | None = None,
        save=None,
        **kwargs,
    ):
        from matplotlib.patches import Circle

        del colorbar
        style_kwargs = pop_axes_style_kwargs(kwargs)
        figure_kwargs = pop_figure_kwargs(kwargs)
        axis = ensure_axis(ax, default_figsize=(5.0, 5.0), figure_kwargs=figure_kwargs)
        ordered = self._interface._ordered_results(sort_by)
        for result in ordered:
            trajectory = result.solitons.vortex.trajectory.steady_state()
            axis.plot(
                np.asarray(trajectory.x) * 1e9,
                np.asarray(trajectory.y) * 1e9,
                alpha=0.8,
            )
        radius = disk_radius
        if radius is None and ordered:
            radius = _disk_radius_from_attrs(getattr(ordered[0], "attrs", {}) or {})
        if show_disk_boundary and radius is not None and np.isfinite(radius):
            axis.add_patch(
                Circle(
                    (0.0, 0.0),
                    float(radius) * 1e9,
                    fill=False,
                    linestyle="--",
                    color="0.4",
                )
            )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [nm]")
        axis.set_ylabel("y [nm]")
        axis.grid(True, alpha=float(grid_alpha))
        apply_axes_style(axis, style_kwargs)
        if save is not None:
            axis.figure.savefig(save)
        return axis

    def orbits_grid(
        self,
        *,
        sort_by: str = "i_pillar_ma",
        show_progress: bool = True,
        colorbar: bool = True,
        grid_alpha: float = 0.2,
        show_disk_boundary: bool = True,
        disk_radius: float | None = None,
        ncols: int = 3,
        save=None,
        **kwargs,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        del show_progress, colorbar
        ordered = self._interface._ordered_results(sort_by)
        n = max(len(ordered), 1)
        cols = max(int(ncols), 1)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(
            rows, cols, squeeze=False, figsize=(3.2 * cols, 3.2 * rows)
        )
        flat = axes.ravel()
        for axis, result in zip(flat, ordered):
            trajectory = result.solitons.vortex.trajectory.steady_state()
            axis.plot(np.asarray(trajectory.x) * 1e9, np.asarray(trajectory.y) * 1e9)
            radius = disk_radius
            if radius is None:
                radius = _disk_radius_from_attrs(getattr(result, "attrs", {}) or {})
            if show_disk_boundary and radius is not None and np.isfinite(radius):
                axis.add_patch(
                    Circle(
                        (0.0, 0.0),
                        float(radius) * 1e9,
                        fill=False,
                        linestyle="--",
                        color="0.4",
                    )
                )
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, alpha=float(grid_alpha))
        for axis in flat[len(ordered) :]:
            axis.set_axis_off()
        fig.tight_layout()
        if save is not None:
            fig.savefig(save)
        return fig, axes

    def _repr_html_(self) -> str:
        api = api_help_html(
            self,
            title="Batch vortex plot API help",
            prefix="jobs.vortex.plt",
            methods=[
                "spectrum_map",
                "current_spectrum_map",
                "phase_diagram",
                "orbit_radius",
                "regimes",
                "frequency_vs_current",
                "dashboard",
                "orbits",
                "orbits_grid",
            ],
            subtitle="Plotting accessor for batch-level vortex diagnostics and summaries.",
            chrome=False,
        )
        return node_card_html(
            self.__class__.__name__,
            icon="📈",
            subtitle="batch vortex plot accessor",
            sections=[
                accessors_section_html(
                    [
                        (
                            "Plots:",
                            [
                                (
                                    ".spectrum_map(sort_by='i_pillar_ma', ...)",
                                    NODE_COLOR_ANALYSIS,
                                ),
                                (
                                    ".current_spectrum_map(current='i_pillar_ma', ...)",
                                    NODE_COLOR_ANALYSIS,
                                ),
                                (".phase_diagram(...)", NODE_COLOR_ANALYSIS),
                                (".regimes(...)", NODE_COLOR_COMPUTE),
                                (".dashboard(...)", NODE_COLOR_ANALYSIS),
                                (".orbits(...)", NODE_COLOR_PLOT),
                                (".orbits_grid(...)", NODE_COLOR_PLOT),
                                (".orbit_radius(...)", NODE_COLOR_UTIL),
                                (".frequency_vs_current(...)", NODE_COLOR_UTIL),
                            ],
                        )
                    ]
                ),
                examples_section_html(
                    "\n".join(
                        [
                            "plot = jobs.vortex.plt",
                            "plot.current_spectrum_map(current='i_pillar_ma')",
                            "plot.regimes(sort_by='i_pillar_ma')",
                            "plot.dashboard(sort_by='i_pillar_ma')",
                        ]
                    )
                ),
            ],
            api=api,
            uid="batch-vortex-plot-accessor",
        )
