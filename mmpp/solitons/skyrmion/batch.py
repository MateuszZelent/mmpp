# ruff: noqa: UP007
"""Batch helpers for skyrmion topology and size measurements."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from ._analysis import SIZE_METRIC_UNITS, analysis_catalog_rows, get_analysis_spec

_PARAMETER_PRIORITY = (
    "i_pillar_ma",
    "ma",
    "Jdc",
    "B_ext",
    "B0",
    "Bext",
    "H_ext",
    "H0",
    "field",
    "Dind",
    "D",
    "DMI",
    "Ku1",
    "Ku",
    "Aex",
    "Msat",
    "Ms",
    "alpha",
    "temperature",
    "f0",
    "amp",
    "thickness",
)
_PARAMETER_PRIORITY_INDEX = {
    name.casefold(): index for index, name in enumerate(_PARAMETER_PRIORITY)
}
_NON_SWEEP_ATTRIBUTES = {
    "dx",
    "dy",
    "dz",
    "nx",
    "ny",
    "nz",
    "dt",
    "t",
    "t_sampl",
    "time",
    "duration",
    "t_end",
    "nsteps",
    "steps",
    "index",
    "seed",
    "timestamp",
    "created_at",
    "updated_at",
    "path",
    "name",
    "version",
}


def _iter_progress(
    items: Iterable[Any], *, total: int, enabled: bool, desc: str
) -> Iterable[Any]:
    if not enabled or total <= 1:
        return items
    try:
        from tqdm.auto import tqdm

        return cast(
            Iterable[Any],
            tqdm(items, total=total, desc=desc, unit="result"),
        )
    except ImportError:
        return items


class BatchSkyrmionInterface:
    """Run skyrmion measurements across a batch of job results."""

    def __init__(self, results: list[Any], mmpp_instance: Any = None):
        self._results = list(results)
        self._mmpp = mmpp_instance

    @staticmethod
    def _attributes(result: Any) -> dict[str, Any]:
        """Return job attributes without assuming one concrete backend."""
        try:
            return dict(getattr(result, "attrs", {}) or {})
        except Exception:
            attributes = getattr(result, "attributes", {}) or {}
            try:
                return dict(attributes)
            except Exception:
                return {}

    @staticmethod
    def _numeric_scalar(value: Any) -> Optional[float]:
        """Normalize finite numeric scalar metadata, rejecting booleans/arrays."""
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return None
        normalized = float(value)
        return normalized if math.isfinite(normalized) else None

    def _candidate_rows(self) -> list[dict[str, Any]]:
        attributes = [self._attributes(result) for result in self._results]
        keys = sorted(
            {
                str(key)
                for mapping in attributes
                for key in mapping
                if not str(key).startswith("_")
            }
        )
        rows: list[dict[str, Any]] = []
        total = len(attributes)
        for key in keys:
            if key.casefold() in _NON_SWEEP_ATTRIBUTES:
                continue
            values = [self._numeric_scalar(mapping.get(key)) for mapping in attributes]
            available = [value for value in values if value is not None]
            unique = sorted(set(available))
            if len(unique) < 2:
                continue
            priority = _PARAMETER_PRIORITY_INDEX.get(key.casefold())
            rows.append(
                {
                    "parameter": key,
                    "available": len(available),
                    "total": total,
                    "coverage": len(available) / total if total else 0.0,
                    "n_unique": len(unique),
                    "minimum": unique[0],
                    "maximum": unique[-1],
                    "known_parameter": priority is not None,
                    "_priority": priority
                    if priority is not None
                    else len(_PARAMETER_PRIORITY),
                    "_values": tuple(values),
                }
            )
        rows.sort(
            key=lambda row: (
                -row["coverage"],
                row["_priority"],
                -row["n_unique"],
                row["parameter"].casefold(),
            )
        )
        return rows

    def parameter_candidates(self) -> pd.DataFrame:
        """List varying numeric attributes and mark the automatic choice.

        Geometry, sampling, timestamps, seeds, and other bookkeeping metadata
        are excluded.  The first row is the parameter that
        :meth:`size_vs_parameter` selects when ``parameter=None``.
        """
        rows = self._candidate_rows()
        public_rows = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in rows
        ]
        table = pd.DataFrame(public_rows)
        if not table.empty:
            table.insert(0, "recommended", False)
            table.loc[table.index[0], "recommended"] = True
            table.attrs["recommended_parameter"] = str(table.iloc[0]["parameter"])
        else:
            table = pd.DataFrame(
                columns=[
                    "recommended",
                    "parameter",
                    "available",
                    "total",
                    "coverage",
                    "n_unique",
                    "minimum",
                    "maximum",
                    "known_parameter",
                ]
            )
            table.attrs["recommended_parameter"] = None
        return table

    @staticmethod
    def available_analyses() -> pd.DataFrame:
        """Return the registered generic batch analyses and accepted aliases."""
        return pd.DataFrame(analysis_catalog_rows())

    def _resolve_parameter(
        self, parameter: Optional[str]
    ) -> tuple[str, list[dict[str, Any]], bool]:
        if not self._results:
            raise ValueError("Cannot build a skyrmion sweep from an empty batch.")

        rows = self._candidate_rows()
        if parameter is not None:
            parameter = str(parameter).strip()
            if not parameter:
                parameter = None
            else:
                values = [
                    self._numeric_scalar(self._attributes(result).get(parameter))
                    for result in self._results
                ]
                finite_values = [value for value in values if value is not None]
                if not finite_values:
                    available = ", ".join(row["parameter"] for row in rows) or "none"
                    raise ValueError(
                        f"Attribute {parameter!r} has no finite numeric values in "
                        f"this batch. Varying numeric candidates: {available}."
                    )
                if len(set(finite_values)) < 2:
                    raise ValueError(
                        f"Attribute {parameter!r} does not vary across this batch; "
                        "at least two distinct finite values are required."
                    )
                return parameter, rows, False

        if not rows:
            raise ValueError(
                "Could not auto-detect a varying numeric simulation parameter. "
                "Pass parameter='attribute_name' explicitly and make sure that "
                "attribute is stored in the jobs."
            )

        chosen = str(rows[0]["parameter"])
        chosen_signature = rows[0]["_values"]
        independent = [
            str(row["parameter"])
            for row in rows[1:]
            if row["_values"] != chosen_signature
        ]
        if independent:
            warnings.warn(
                "Several independent attributes vary across the skyrmion batch. "
                f"Automatically selected {chosen!r}; alternatives: "
                f"{', '.join(independent)}. Pass parameter=... to override.",
                UserWarning,
                stacklevel=3,
            )
        return chosen, rows, True

    @staticmethod
    def _namespace(result: Any, dataset_name: Optional[str]) -> Any:
        if dataset_name is None:
            return result.solitons.skyrmion
        return getattr(result, dataset_name).solitons.skyrmion

    @staticmethod
    def _identity(index: int, result: Any) -> dict[str, Any]:
        attrs = getattr(result, "attrs", {}) or {}
        return {
            "index": int(index),
            "name": getattr(result, "name", None),
            "path": getattr(result, "path", None),
            "status": "ok",
            "error": None,
            **{
                key: attrs.get(key)
                for key in ("i_pillar_ma", "ma", "Jdc", "B_ext", "field")
                if attrs.get(key) is not None
            },
        }

    def detect(
        self,
        *,
        dataset_name: Optional[str] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return one topology row per result, preserving failures as rows."""
        rows: list[dict[str, Any]] = []
        iterator = _iter_progress(
            enumerate(self._results),
            total=len(self._results),
            enabled=show_progress,
            desc="Detecting skyrmions",
        )
        for index, result in iterator:
            row = self._identity(index, result)
            try:
                topology = self._namespace(result, dataset_name).detect(**kwargs)
                row.update(
                    {
                        "Q": topology.Q,
                        "center_x_m": topology.center_xy_m[0],
                        "center_y_m": topology.center_xy_m[1],
                        "polarity": topology.polarity,
                        "background_sign": topology.background_sign,
                        "state": topology.state,
                        "confidence": topology.confidence,
                        "valid": topology.valid,
                        "flags": topology.flags,
                    }
                )
            except Exception as exc:
                row.update({"status": "error", "error": str(exc)})
            rows.append(row)
        return pd.DataFrame(rows)

    def measure_size(
        self,
        *,
        method: str = "auto",
        dataset_name: Optional[str] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit one skyrmion size per result and return a tidy table."""
        rows: list[dict[str, Any]] = []
        iterator = _iter_progress(
            enumerate(self._results),
            total=len(self._results),
            enabled=show_progress,
            desc="Measuring skyrmions",
        )
        for index, result in iterator:
            row = self._identity(index, result)
            try:
                namespace = self._namespace(result, dataset_name)
                topology_kwargs = {
                    key: kwargs[key]
                    for key in ("t", "frame", "z_layer", "mask", "convention", "force")
                    if key in kwargs
                }
                topology = namespace.detect(**topology_kwargs)
                size = namespace.fit_size(
                    method=method,
                    topology=topology,
                    **kwargs,
                )
                row.update(
                    {
                        "Q": topology.Q,
                        "state": topology.state,
                        "radius_m": size.radius_m,
                        "diameter_m": size.diameter_m,
                        "radius_nm": size.radius_nm,
                        "diameter_nm": size.diameter_nm,
                        "wall_width_m": size.wall_width_m,
                        "scale_m": size.scale_m,
                        "sigma_m": size.sigma_m,
                        "model": size.model,
                        "requested_method": size.requested_method,
                        "fit_success": size.fit_success,
                        "quality": size.quality,
                        "normalized_rmse": size.normalized_rmse,
                        "aicc": size.aicc,
                        "flags": size.flags,
                    }
                )
            except Exception as exc:
                row.update({"status": "error", "error": str(exc)})
            rows.append(row)
        return pd.DataFrame(rows)

    def _analyze_size(
        self,
        *,
        size_metric: str = "diameter_nm",
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, str, str]:
        metric = str(size_metric).strip()
        if metric not in SIZE_METRIC_UNITS:
            available = ", ".join(sorted(SIZE_METRIC_UNITS))
            raise ValueError(
                f"Unknown size_metric {size_metric!r}. Available metrics: {available}."
            )
        return self.measure_size(**kwargs), metric, SIZE_METRIC_UNITS[metric]

    def _analyze_charge(self, **kwargs: Any) -> tuple[pd.DataFrame, str, str]:
        return self.detect(**kwargs), "Q", "1"

    def _attach_parameter_axis(
        self,
        table: pd.DataFrame,
        *,
        analysis: str,
        value_column: str,
        value_unit: str,
        parameter: Optional[str],
        parameter_scale: float,
        parameter_unit: Optional[str],
        sort: bool,
    ) -> pd.DataFrame:
        scale = float(parameter_scale)
        if not math.isfinite(scale):
            raise ValueError("parameter_scale must be finite.")

        selected, candidates, auto_selected = self._resolve_parameter(parameter)
        values = [
            self._numeric_scalar(self._attributes(result).get(selected))
            for result in self._results
        ]
        scaled_values = [
            value * scale if value is not None else np.nan for value in values
        ]
        table.insert(3, "parameter_name", selected)
        table.insert(4, "parameter_value", scaled_values)
        table.insert(5, "parameter_available", [value is not None for value in values])
        reserved_columns = {
            "analysis",
            "observable_name",
            "observable_value",
            "observable_unit",
            "parameter_name",
            "parameter_value",
            "parameter_available",
        }
        if selected not in table.columns and selected not in reserved_columns:
            table.insert(6, selected, scaled_values)
        elif selected in table.columns and selected not in reserved_columns:
            table[selected] = scaled_values

        observable_values: Any
        if value_column in table.columns:
            observable_values = table[value_column]
        else:
            observable_values = np.full(len(table), np.nan)
        table.insert(6, "analysis", analysis)
        table.insert(7, "observable_name", value_column)
        table.insert(8, "observable_value", observable_values)
        table.insert(9, "observable_unit", value_unit)

        if sort:
            table = table.sort_values(
                by=["parameter_value", "index"],
                kind="stable",
                na_position="last",
            ).reset_index(drop=True)

        candidate_names = tuple(str(row["parameter"]) for row in candidates)
        label = selected if not parameter_unit else f"{selected} [{parameter_unit}]"
        table.attrs.update(
            {
                "analysis": analysis,
                "observable_name": value_column,
                "observable_unit": value_unit,
                "parameter": selected,
                "parameter_label": label,
                "parameter_scale": scale,
                "parameter_unit": parameter_unit,
                "parameter_source": "auto" if auto_selected else "manual",
                "parameter_candidates": candidate_names,
            }
        )
        if analysis == "size":
            table.attrs["size_columns"] = ("radius_nm", "diameter_nm")
        return table

    def size_vs_parameter(
        self,
        parameter: Optional[str] = None,
        *,
        parameter_scale: float = 1.0,
        parameter_unit: Optional[str] = None,
        sort: bool = True,
        method: str = "auto",
        dataset_name: Optional[str] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compatibility wrapper for ``analyze('size', ...)``.

        Parameters
        ----------
        parameter
            Attribute name used as the sweep axis.  ``None`` or an empty string
            selects the best varying numeric attribute automatically.
        parameter_scale
            Multiplicative conversion applied to the stored attribute values.
        parameter_unit
            Optional display unit recorded in ``DataFrame.attrs``.
        sort
            Sort rows by the physical parameter while preserving error rows.

        Returns
        -------
        pandas.DataFrame
            The regular batch size table plus ``parameter_name``,
            ``parameter_value``, and ``parameter_available`` columns.  Selection
            diagnostics are stored in ``DataFrame.attrs``.
        """
        return self.analyze(
            "size",
            parameter=parameter,
            parameter_scale=parameter_scale,
            parameter_unit=parameter_unit,
            sort=sort,
            method=method,
            dataset_name=dataset_name,
            show_progress=show_progress,
            **kwargs,
        )

    def fit_size(self, **kwargs: Any) -> pd.DataFrame:
        """Alias for :meth:`measure_size`."""
        return self.measure_size(**kwargs)

    def analyze(
        self,
        observable: Optional[str] = None,
        *,
        parameter: Optional[str] = None,
        parameter_scale: float = 1.0,
        parameter_unit: Optional[str] = None,
        sort: bool = True,
        dataset_name: Optional[str] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Analyze one observable across a parameter sweep.

        ``analyze("size", ...)`` measures a configurable size metric;
        ``analyze("charge", ...)`` measures topological charge ``Q``.  The
        sweep parameter can be named explicitly or auto-detected when omitted.

        Calling ``analyze()`` without an observable retains the historical
        combined topology-and-size batch table without requiring a sweep axis.
        """
        if observable is None:
            if (
                parameter is not None
                or parameter_scale != 1.0
                or parameter_unit is not None
                or not sort
            ):
                raise ValueError(
                    "A sweep axis requires an observable. Use "
                    "analyze('size', ...) or analyze('charge', ...)."
                )
            return self.measure_size(
                dataset_name=dataset_name,
                show_progress=show_progress,
                **kwargs,
            )

        spec = get_analysis_spec(observable)
        analysis = spec.name
        handler = getattr(self, spec.batch_handler)
        table, value_column, value_unit = handler(
            dataset_name=dataset_name,
            show_progress=show_progress,
            **kwargs,
        )
        return self._attach_parameter_axis(
            table,
            analysis=analysis,
            value_column=value_column,
            value_unit=value_unit,
            parameter=parameter,
            parameter_scale=parameter_scale,
            parameter_unit=parameter_unit,
            sort=sort,
        )

    def __repr__(self) -> str:
        return f"BatchSkyrmionInterface({len(self._results)} results)"

    def _repr_html_(self) -> str:
        import uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            accessors_section_html,
            api_help_html,
            metrics_section_html,
            node_card_html,
        )

        api = api_help_html(
            self,
            title="Batch skyrmion API help",
            prefix="job[:].solitons.skyrmion",
            methods=[
                "detect",
                "measure_size",
                "size_vs_parameter",
                "parameter_candidates",
                "available_analyses",
                "fit_size",
                "analyze",
            ],
            properties=[],
            chrome=False,
        )
        return str(
            node_card_html(
                "Batch Skyrmion Analysis",
                icon="🧲",
                subtitle="Batch topology and physical-size measurements",
                sections=[
                    metrics_section_html(
                        [("n_results", len(self._results), NODE_COLOR_COMPUTE)]
                    ),
                    accessors_section_html(
                        [
                            (
                                "Compute:",
                                [
                                    (".detect()", NODE_COLOR_ANALYSIS),
                                    (".measure_size()", NODE_COLOR_COMPUTE),
                                    (".analyze('size')", NODE_COLOR_ANALYSIS),
                                    (".analyze('charge')", NODE_COLOR_ANALYSIS),
                                ],
                            )
                        ]
                    ),
                ],
                api=api,
                uid=f"batch-skyrmion-{uuid.uuid4().hex[:8]}",
            )
        )


__all__ = ["BatchSkyrmionInterface"]
