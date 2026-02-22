"""Public interface for hysteresis analysis."""

from __future__ import annotations

import importlib
from html import escape as _esc
from typing import Any

import numpy as np

from .config import HysteresisConfig
from .result import Branch, HysteresisResult
from .sources import (
    from_arrays,
    from_magnetization,
    from_table,
    from_zarr_keys,
    resolve_auto_source,
)


class _HysteresisQuickPlot:
    """Quick plotting helper resolving sources automatically."""

    def __init__(self, interface: HysteresisInterface):
        self._interface = interface

    def _resolve_result(self, **kwargs) -> HysteresisResult:
        return self._interface._resolve_for_plot(**kwargs)

    def loop(self, **kwargs):
        """Compute/resolve source and draw static loop plot."""
        source_keys = {
            "field",
            "magnetization",
            "source",
            "component",
            "dset",
            "z_layer",
            "roi",
            "roi_units",
            "key_prefix",
            "min_spatial_size",
        }
        source_kwargs = {k: kwargs[k] for k in source_keys if k in kwargs}
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in source_keys}
        result = self._resolve_result(**source_kwargs)
        return result.plot.loop(**plot_kwargs)

    def interactive(self, **kwargs):
        """Compute/resolve source and open interactive explorer."""
        source_keys = {
            "field",
            "magnetization",
            "source",
            "component",
            "dset",
            "z_layer",
            "roi",
            "roi_units",
            "key_prefix",
            "min_spatial_size",
        }
        source_kwargs = {k: kwargs[k] for k in source_keys if k in kwargs}
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in source_keys}
        result = self._resolve_result(**source_kwargs)
        return result.plot.interactive(**plot_kwargs)

    def animation(self, **kwargs):
        """Compute/resolve source and create/export animation."""
        source_keys = {
            "field",
            "magnetization",
            "source",
            "component",
            "dset",
            "z_layer",
            "roi",
            "roi_units",
        }
        source_kwargs = {k: kwargs[k] for k in source_keys if k in kwargs}
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in source_keys}
        result = self._resolve_result(**source_kwargs)
        return result.plot.animation(**plot_kwargs)

    def __repr__(self) -> str:
        return "<HysteresisQuickPlot: .loop(), .interactive(), .animation()>"

    def _repr_html_(self) -> str:
        methods = [
            (".loop(...)", "Auto-resolve source and draw static loop"),
            (".interactive(...)", "Auto-resolve source and open explorer"),
            (".animation(...)", "Auto-resolve source and create/export animation"),
        ]
        rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(name)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
            for name, desc in methods
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;\">"
            "<div style='font-size:1.03em;font-weight:600;color:#f1f5f9;'>"
            "Hysteresis Quick Plot</div>"
            "<table style='width:100%;margin-top:8px;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:6px 8px;color:#e2e8f0;'>Method</th>"
            "<th style='padding:6px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _repr_mimebundle_(self, include=None, exclude=None):
        html = self._repr_html_()
        text = self.__repr__()
        if html:
            return {"text/html": html, "text/plain": text}
        return {"text/plain": text}


class HysteresisInterface:
    """Fluent entrypoint for hysteresis workflows."""

    def __init__(
        self,
        job_result,
        mmpp_instance: Any | None = None,
        dataset_name: str | None = None,
        slice_info: Any | None = None,
        config: HysteresisConfig | None = None,
    ):
        self._job = job_result
        self._mmpp = mmpp_instance
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config or HysteresisConfig()
        self._plot = None

    @property
    def config(self) -> HysteresisConfig:
        """Mutable configuration for this interface instance."""
        return self._config

    @property
    def plot(self) -> _HysteresisQuickPlot:
        """Quick plotting helper with auto-source resolution."""
        if self._plot is None:
            self._plot = _HysteresisQuickPlot(self)
        return self._plot

    def from_arrays(
        self,
        field: np.ndarray,
        magnetization: np.ndarray,
        *,
        frame_index: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HysteresisResult:
        """Build result directly from explicit arrays."""
        return from_arrays(
            field,
            magnetization,
            frame_index=frame_index,
            metadata=metadata,
            config=self._config,
        )

    def from_table(
        self,
        *,
        field: str | None = None,
        magnetization: str | None = None,
        component: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HysteresisResult:
        """Build result from ``job['table']`` data."""
        return from_table(
            self._job,
            field=field,
            magnetization=magnetization,
            component=component,
            metadata=metadata,
            config=self._config,
        )

    def from_magnetization(
        self,
        *,
        dset: str | None = None,
        component: str = "x",
        z_layer: int | str = 0,
        roi: tuple[float, float, float, float] | None = None,
        roi_units: str = "idx",
        field: str | np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HysteresisResult:
        """Build result from averaged magnetization dataset."""
        return from_magnetization(
            self._job,
            dset=dset or self._dataset_name or self._config.default_m_dataset,
            component=component,
            z_layer=z_layer,
            roi=roi,
            roi_units=roi_units,
            field=field,
            slice_info=self._slice_info,
            metadata=metadata,
            config=self._config,
        )

    def from_zarr_keys(
        self,
        *,
        key_prefix: str = "B",
        component: str = "x",
        z_layer: int | str = 0,
        roi: tuple[float, float, float, float] | None = None,
        roi_units: str = "idx",
        min_spatial_size: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> HysteresisResult:
        """Build hysteresis loop from named zarr arrays in the root group.

        For simulations where each applied-field value is stored as a
        separate zarr array — e.g. ``B-0.500000.6`` — rather than a
        single time-series dataset.

        Parameters
        ----------
        key_prefix:
            Common prefix of the field keys.  Default ``"B"`` matches
            keys like ``B-0.500000.6``.
        component:
            Magnetization component to spatially average: ``"x"``,
            ``"y"``, ``"z"`` or ``"norm"``.
        z_layer:
            Z-slice index or ``"all"`` to average all z layers.
        roi:
            ``(x0, x1, y0, y1)`` region of interest.
        roi_units:
            ``"idx"`` (pixel coords) or ``"nm"``.
        min_spatial_size:
            Skip arrays with fewer than this many spatial pixels
            (filters out small debug snapshots).
        """
        return from_zarr_keys(
            self._job,
            key_prefix=key_prefix,
            component=component,
            z_layer=z_layer,
            roi=roi,
            roi_units=roi_units,
            min_spatial_size=min_spatial_size,
            metadata=metadata,
            config=self._config,
        )

    def load(
        self,
        *,
        source: str,
        # ── source="table" ────────────────────────────────────────────────
        field: str | None = None,
        magnetization: str | None = None,
        # ── source="zarr_keys" ────────────────────────────────────────────
        key_prefix: str = "B",
        component: str = "x",
        z_layer: int | str = 0,
        roi: tuple[float, float, float, float] | None = None,
        roi_units: str = "idx",
        min_spatial_size: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> "HysteresisResult":
        """Unified entry point for loading hysteresis data.

        Parameters
        ----------
        source:
            ``"table"`` — read field & magnetization from the ``table/`` group
            (1-D scalar columns, one value per simulation step).

            ``"zarr_keys"`` — build loop from separately-stored spatial
            snapshots where the field value is encoded in the array name
            (e.g. ``B-0.025000.6`` → B = −0.025 T) and magnetization is
            computed as a spatial mean of the chosen component.

        source="table" specific
        -----------------------
        field : str
            Column name in ``table/``, e.g. ``"B_exty"``.
        magnetization : str
            Column name in ``table/``, e.g. ``"my"``.

        source="zarr_keys" specific
        ---------------------------
        key_prefix : str
            Prefix that all per-field arrays share, e.g. ``"B-"``.
            The field value is parsed from the remainder of the key name.
        component : str
            Which magnetization component to spatially average:
            ``"x"``, ``"y"``, ``"z"`` or ``"norm"``.
        z_layer : int or "all"
            Z slice index, or ``"all"`` to average every z layer.
        roi : tuple(x0, x1, y0, y1) or None
            Spatial region of interest.
        roi_units : str
            ``"idx"`` (pixel) or ``"nm"``.
        min_spatial_size : int
            Skip arrays with fewer spatial pixels (skip debug snapshots).

        Examples
        --------
        Table source — field and magnetization from scalar columns::

            result = job[0].analyze.hysteresis.load(
                source="table",
                field="B_exty",
                magnetization="my",
            )

        Zarr-keys source — field from array names, magnetization averaged spatially::

            result = job[0].analyze.hysteresis.load(
                source="zarr_keys",
                key_prefix="B-",
                component="y",
                z_layer=0,
            )
        """
        source_norm = source.strip().lower().replace("-", "_")

        if source_norm == "table":
            if field is None or magnetization is None:
                raise ValueError(
                    "source='table' requires both 'field' and 'magnetization' "
                    "as column name strings, e.g. field='B_exty', magnetization='my'."
                )
            return self.from_table(
                field=field,
                magnetization=magnetization,
                metadata=metadata,
            )

        if source_norm == "zarr_keys":
            return self.from_zarr_keys(
                key_prefix=key_prefix,
                component=component,
                z_layer=z_layer,
                roi=roi,
                roi_units=roi_units,
                min_spatial_size=min_spatial_size,
                metadata=metadata,
            )

        raise ValueError(
            f"Unknown source={source!r}. "
            "Use source='table' or source='zarr_keys'."
        )

    def _resolve_for_plot(
        self,
        *,
        field: str | np.ndarray | None = None,
        magnetization: str | np.ndarray | None = None,
        source: str | None = None,
        component: str | None = None,
        dset: str | None = None,
        z_layer: int | str | None = None,
        roi: tuple[float, float, float, float] | None = None,
        roi_units: str = "idx",
        key_prefix: str = "B",
        min_spatial_size: int = 50,
        **_kwargs,
    ) -> HysteresisResult:
        if source == "zarr_keys":
            return self.from_zarr_keys(
                key_prefix=key_prefix,
                component=component or self._config.default_component,
                z_layer=self._config.z_layer if z_layer is None else z_layer,
                roi=roi,
                roi_units=roi_units,
                min_spatial_size=min_spatial_size,
            )

        if source == "table":
            return self.from_table(
                field=field if isinstance(field, str) else None,
                magnetization=magnetization if isinstance(magnetization, str) else None,
                component=component,
            )

        if source == "magnetization":
            return self.from_magnetization(
                dset=dset,
                component=component or self._config.default_component,
                z_layer=self._config.z_layer if z_layer is None else z_layer,
                roi=roi,
                roi_units=roi_units,
                field=field,
            )

        if isinstance(field, np.ndarray) and isinstance(magnetization, np.ndarray):
            return self.from_arrays(field, magnetization)

        return resolve_auto_source(
            self._job,
            field=field,
            magnetization=magnetization,
            component=component,
            dset=dset or self._dataset_name,
            z_layer=self._config.z_layer if z_layer is None else z_layer,
            roi=roi,
            roi_units=roi_units,
            slice_info=self._slice_info,
            config=self._config,
        )

    def __repr__(self) -> str:
        return (
            "HysteresisInterface("
            f"dataset={self._dataset_name!r}, slice={self._slice_info!r})"
        )

    def _repr_html_(self) -> str:
        dataset = _esc(str(self._dataset_name)) if self._dataset_name is not None else "auto"
        slice_label = _esc(str(self._slice_info)) if self._slice_info is not None else "full"

        methods = [
            (".load(source='table', field=..., magnetization=...)", "Unified entry: read field & M from table/ columns"),
            (".load(source='zarr_keys', key_prefix=..., component=...)", "Unified entry: field from key names, M averaged spatially"),
            (".from_table(...)", "Direct: read from table/ columns"),
            (".from_magnetization(...)", "Direct: build loop from averaged m dataset"),
            (".from_zarr_keys(...)", "Direct: per-field zarr arrays (e.g. B-0.025000.6)"),
            (".from_arrays(...)", "Expert: explicit arrays"),
            (".plot.loop(...)", "Quick static plot"),
            (".plot.interactive(...)", "Quick interactive view"),
        ]
        rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(name)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
            for name, desc in methods
        )

        example = "\n".join(
            [
                "hys = job[0].analyze.hysteresis",
                "",
                "# Tryb 1: dane skalarne z kolumn tabeli",
                "res = hys.load(source='table', field='B_exty', magnetization='my')",
                "",
                "# Tryb 2: dane przestrzenne – pole z nazw kluczy, M uśredniona",
                "res = hys.load(source='zarr_keys', key_prefix='B-', component='y')",
                "",
                "res.plot.loop(show_hc=True)",
                "res.metrics.report()",
            ]
        )

        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:10px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 10px 22px rgba(0,0,0,0.28);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;'>Hysteresis Interface</div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-top:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='display:flex;gap:16px;flex-wrap:wrap;font-size:0.9em;'>"
            f"<div><span style='color:#94a3b8;'>Dataset:</span> <code style='color:#cbd5e1;'>{dataset}</code></div>"
            f"<div><span style='color:#94a3b8;'>Slice:</span> <code style='color:#cbd5e1;'>{slice_label}</code></div>"
            "</div></div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-top:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:6px 8px;color:#e2e8f0;'>Method</th>"
            "<th style='padding:6px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-top:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;border-radius:6px;"
            f"color:#e2e8f0;overflow-x:auto;font-size:0.85em;'><code>{_esc(example)}</code></pre>"
            "</div></div>"
        )

    def _repr_mimebundle_(self, include=None, exclude=None):
        html = self._repr_html_()
        text = self.__repr__()
        if html:
            return {"text/html": html, "text/plain": text}
        return {"text/plain": text}


__all__ = [
    "Branch",
    "HysteresisConfig",
    "HysteresisResult",
    "HysteresisInterface",
    "ComparisonAccessor",
    "HysteresisComparison",
    "HysteresisExporter",
    "plot",
]


def __getattr__(name: str):
    if name in {"ComparisonAccessor", "HysteresisComparison"}:
        from .comparison import ComparisonAccessor, HysteresisComparison

        mapping = {
            "ComparisonAccessor": ComparisonAccessor,
            "HysteresisComparison": HysteresisComparison,
        }
        return mapping[name]
    if name == "HysteresisExporter":
        from .export import HysteresisExporter

        return HysteresisExporter
    if name == "plot":
        return importlib.import_module(".plot", __name__)
    raise AttributeError(name)
