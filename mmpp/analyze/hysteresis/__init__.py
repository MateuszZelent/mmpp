"""Public interface for hysteresis analysis."""

from __future__ import annotations

import importlib
from html import escape as _esc
from typing import Any

import numpy as np

from mmpp._repr_helpers import (
    NODE_COLOR_ANALYSIS,
    NODE_COLOR_COMPUTE,
    NODE_COLOR_PLOT,
    NODE_COLOR_UTIL,
    api_help_html,
    accessors_section_html,
    examples_section_html,
    metrics_section_html,
    node_card_html,
)

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
        example = "\n".join(
            [
                "plot = job[0].analyze.hysteresis.plot",
                "plot.loop(source='table', field='B_extx', magnetization='mx')",
                "plot.interactive(source='zarr_keys', key_prefix='B-', component='y')",
            ]
        )
        api = api_help_html(
            self,
            title="Hysteresis quick plot API help",
            prefix="job[0].analyze.hysteresis.plot",
            methods=["loop", "interactive", "animation"],
            subtitle="Quick plot methods auto-resolve a hysteresis source before drawing.",
            chrome=False,
        )
        return node_card_html(
            "Hysteresis Quick Plot",
            icon="📈",
            subtitle="Plotting shortcuts that auto-resolve the hysteresis source before rendering.",
            sections=[
                metrics_section_html(
                    [
                        ("source mode", "auto-resolve", NODE_COLOR_UTIL),
                        ("target", "loop / explorer / animation", NODE_COLOR_ANALYSIS),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Plot:",
                            [
                                (".loop(...)", NODE_COLOR_COMPUTE),
                                (".interactive(...)", NODE_COLOR_ANALYSIS),
                                (".animation(...)", NODE_COLOR_PLOT),
                            ],
                        )
                    ]
                ),
                examples_section_html(example),
            ],
            api=api,
            uid="mmpp-hysteresis-quick-plot",
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
        cloneflip: bool = False,
    ) -> HysteresisResult:
        """Build result directly from explicit arrays."""
        return from_arrays(
            field,
            magnetization,
            frame_index=frame_index,
            metadata=metadata,
            config=self._config,
            cloneflip=cloneflip,
        )

    def from_table(
        self,
        *,
        field: str | None = None,
        magnetization: str | None = None,
        component: str | None = None,
        metadata: dict[str, Any] | None = None,
        cloneflip: bool = False,
    ) -> HysteresisResult:
        """Build result from ``job['table']`` data."""
        return from_table(
            self._job,
            field=field,
            magnetization=magnetization,
            component=component,
            metadata=metadata,
            config=self._config,
            cloneflip=cloneflip,
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
        cloneflip: bool = False,
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
            cloneflip=cloneflip,
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
        cloneflip: bool = False,
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
            cloneflip=cloneflip,
        )

    def load(
        self,
        *,
        source: str,
        # ── source="table" ────────────────────────────────────────────────
        field: str | None = None,
        magnetization: str | None = None,
        # ── source="magnetization" ────────────────────────────────────────
        dset: str | None = None,
        # ── source="zarr_keys" ────────────────────────────────────────────
        key_prefix: str = "B",
        component: str = "x",
        z_layer: int | str = 0,
        roi: tuple[float, float, float, float] | None = None,
        roi_units: str = "idx",
        min_spatial_size: int = 50,
        metadata: dict[str, Any] | None = None,
        cloneflip: bool = False,
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

            ``"magnetization"`` — build loop from a time/field series dataset
            such as ``m(t, z, y, x, c)`` by spatially averaging the chosen
            component and pairing it with a field column or explicit field array.

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

        source="magnetization" specific
        --------------------------------
        dset : str
            Magnetization dataset name, e.g. ``"m"``.
        component : str
            Which magnetization component to spatially average. Accepts
            ``"x"``, ``"y"``, ``"z"``, ``"mx"``, ``"my"``, ``"mz"`` or
            ``"norm"``.

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

        Magnetization source — field from table, magnetization from snapshots::

            result = job[0].analyze.hysteresis.load(
                source="magnetization",
                dset="m",
                field="B_exty",
                component="my",
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
                cloneflip=cloneflip,
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
                cloneflip=cloneflip,
            )

        if source_norm == "magnetization":
            return self.from_magnetization(
                dset=dset,
                component=component,
                z_layer=z_layer,
                roi=roi,
                roi_units=roi_units,
                field=field,
                metadata=metadata,
                cloneflip=cloneflip,
            )

        raise ValueError(
            f"Unknown source={source!r}. Use source='table', "
            "source='magnetization', or source='zarr_keys'."
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
        dataset = (
            _esc(str(self._dataset_name)) if self._dataset_name is not None else "auto"
        )
        slice_label = (
            _esc(str(self._slice_info)) if self._slice_info is not None else "full"
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
        api = api_help_html(
            self,
            title="Hysteresis API help",
            prefix="job[0].analyze.hysteresis",
            properties=[
                ("config", "Mutable configuration for this hysteresis namespace"),
                ("plot", "Quick plotting helper with auto-source resolution"),
            ],
            methods=[
                "load",
                "from_arrays",
                "from_table",
                "from_magnetization",
                "from_zarr_keys",
            ],
            subtitle="Live public API for loading hysteresis data and building result objects.",
            chrome=False,
        )
        return node_card_html(
            "Hysteresis Interface",
            icon="🧲",
            subtitle="Build hysteresis loops from table data, magnetization datasets, zarr-key sweeps or explicit arrays.",
            sections=[
                metrics_section_html(
                    [
                        ("dataset", dataset, NODE_COLOR_COMPUTE),
                        ("slice", slice_label, NODE_COLOR_PLOT),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Load:",
                            [
                                (".load(...)", NODE_COLOR_COMPUTE),
                                (".from_table(...)", NODE_COLOR_COMPUTE),
                                (".from_magnetization(...)", NODE_COLOR_ANALYSIS),
                                (".from_zarr_keys(...)", NODE_COLOR_ANALYSIS),
                                (".from_arrays(...)", NODE_COLOR_UTIL),
                            ],
                        ),
                        (
                            "Quick plot:",
                            [
                                (".plot.loop(...)", NODE_COLOR_COMPUTE),
                                (".plot.interactive(...)", NODE_COLOR_ANALYSIS),
                            ],
                        ),
                    ]
                ),
                examples_section_html(example),
            ],
            api=api,
            uid="mmpp-hysteresis",
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
