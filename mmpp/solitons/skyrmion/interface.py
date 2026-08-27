# ruff: noqa: UP007
"""Public dataset-aware skyrmion analysis namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ._analysis import analysis_catalog_rows, get_analysis_spec
from .config import SkyrmionConfig
from .models import SkyrmionAnalysisResult, SkyrmionSizeResult, SkyrmionTopologyResult

if TYPE_CHECKING:
    import pandas as pd


class SkyrmionInterface:
    """Topology and size analysis for isolated skyrmions."""

    def __init__(
        self,
        job_result: Any,
        dataset_name: Optional[str] = None,
        mmpp_instance: Any = None,
        slice_info: Any = None,
        config: Optional[SkyrmionConfig] = None,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._mmpp = mmpp_instance
        self._slice_info = slice_info
        self._config = config or SkyrmionConfig()
        self._topology = None
        self._size = None
        self._result_cache: dict[tuple[Any, ...], Any] = {}

    @property
    def dataset_name(self) -> Optional[str]:
        """Dataset used for analysis, auto-selected when omitted."""
        if self._dataset_name is None:
            candidate = self._job.get_largest_m_dataset()
            try:
                self._job._ensure_zarr_loaded()
                if candidate in self._job._z:
                    self._dataset_name = candidate
            except Exception:
                self._dataset_name = candidate
        return self._dataset_name

    @property
    def config(self) -> SkyrmionConfig:
        """Mutable skyrmion-analysis configuration."""
        return self._config

    @property
    def topology(self):
        """Topology detection namespace."""
        if self._topology is None:
            from .topology import SkyrmionTopologyInterface

            self._topology = SkyrmionTopologyInterface(self)
        return self._topology

    @property
    def size(self):
        """Radial size fitting namespace."""
        if self._size is None:
            from .size import SkyrmionSizeInterface

            self._size = SkyrmionSizeInterface(self)
        return self._size

    def _resolve_data(self) -> np.ndarray:
        name = self.dataset_name
        if name is None:
            raise ValueError(
                "No magnetisation dataset is available for skyrmion analysis."
            )
        dataset = getattr(self._job, name)
        if self._slice_info is not None:
            dataset = dataset[self._slice_info]
        if hasattr(dataset, "numpy"):
            return np.asarray(dataset.numpy(copy=False), dtype=float)
        return np.asarray(dataset, dtype=float)

    def _resolve_spacing(self) -> tuple[float, float]:
        attrs = self._job.attrs

        def resolve(axis: str, index: int) -> float:
            for key in (f"d{axis}", f"cellsize_{axis}", f"cell_size_{axis}"):
                value = attrs.get(key)
                if value is not None:
                    return float(value)
            for key in ("cellsize", "cell_size"):
                value = attrs.get(key)
                if value is not None:
                    values = np.asarray(value, dtype=float).reshape(-1)
                    if values.size > index:
                        return float(values[index])
            return 1.0

        return resolve("x", 0), resolve("y", 1)

    def _cache_key(self, namespace: str, **values: Any) -> tuple[Any, ...]:
        mask = values.pop("mask", None)
        mask_token = None
        if mask is not None:
            mask_array = np.asarray(mask, dtype=bool)
            mask_token = (mask_array.shape, hash(mask_array.tobytes()))
        return (
            namespace,
            self.dataset_name,
            repr(self._slice_info),
            tuple(sorted(values.items())),
            mask_token,
        )

    def detect(self, **kwargs: Any) -> SkyrmionTopologyResult:
        """Shortcut for :meth:`topology.detect`."""
        return self.topology.detect(**kwargs)

    def measure_size(self, **kwargs: Any) -> SkyrmionSizeResult:
        """Measure the contrast-crossing size without choosing a fit model."""
        kwargs.setdefault("method", "threshold")
        return self.size.fit(**kwargs)

    def fit_size(self, **kwargs: Any) -> SkyrmionSizeResult:
        """Fit the configured model or run intelligent automatic selection."""
        kwargs.setdefault("method", "auto")
        return self.size.fit(**kwargs)

    @staticmethod
    def available_analyses() -> pd.DataFrame:
        """Return registered analyses and accepted observable aliases."""
        import pandas as pd

        return pd.DataFrame(analysis_catalog_rows())

    def _analyze_size(self, **kwargs: Any) -> SkyrmionSizeResult:
        kwargs.pop("size_metric", None)
        return self.fit_size(**kwargs)

    def _analyze_charge(self, **kwargs: Any) -> SkyrmionTopologyResult:
        return self.detect(**kwargs)

    def analyze(self, observable: Optional[str] = None, **kwargs: Any) -> Any:
        """Dispatch one analysis or return the historical combined result.

        Use ``analyze("size")`` for a :class:`SkyrmionSizeResult` and
        ``analyze("charge")`` for a :class:`SkyrmionTopologyResult`.  Omitting
        the observable preserves the combined :class:`SkyrmionAnalysisResult`.
        """
        if observable is not None:
            spec = get_analysis_spec(observable)
            handler = getattr(self, spec.single_handler)
            return handler(**kwargs)

        topology_keys = {
            "t",
            "frame",
            "z_layer",
            "mask",
            "convention",
            "force",
        }
        topology_kwargs = {
            key: value for key, value in kwargs.items() if key in topology_keys
        }
        topology = self.detect(**topology_kwargs)
        size_kwargs = dict(kwargs)
        size_kwargs["topology"] = topology
        size = self.fit_size(**size_kwargs)
        return SkyrmionAnalysisResult(
            topology=topology,
            size=size,
            metadata={"dataset_name": self.dataset_name},
        )

    def __repr__(self) -> str:
        return (
            f"SkyrmionInterface(dataset={self.dataset_name!r}, "
            f"slice={self._slice_info!r})"
        )

    def _repr_html_(self) -> str:
        import uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        api = api_help_html(
            self,
            title="Skyrmion API help",
            prefix="job.solitons.skyrmion",
            properties=[
                ("topology", "Charge, centre, and polarity namespace"),
                ("size", "Radial size fitting namespace"),
                ("config", "Mutable numerical configuration"),
            ],
            methods=[
                "detect",
                "measure_size",
                "fit_size",
                "analyze",
                "available_analyses",
            ],
            chrome=False,
        )
        return node_card_html(
            "Skyrmion Analysis",
            icon="🧲",
            subtitle="Dedicated topology and physical-size workflow",
            sections=[
                metrics_section_html(
                    [
                        ("dataset", self.dataset_name or "auto", NODE_COLOR_COMPUTE),
                        ("slice", repr(self._slice_info), None),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Namespaces:",
                            [
                                (".topology", NODE_COLOR_ANALYSIS),
                                (".size", NODE_COLOR_ANALYSIS),
                            ],
                        ),
                        (
                            "Workflow:",
                            [
                                (".analyze('size')", NODE_COLOR_COMPUTE),
                                (".analyze('charge')", NODE_COLOR_ANALYSIS),
                                (".analyze()", NODE_COLOR_COMPUTE),
                            ],
                        ),
                    ]
                ),
                examples_section_html(
                    "size = job.m.skyrmion.analyze('size', method='auto')\n"
                    "charge = job.m.skyrmion.analyze('charge')\n"
                    "print(charge.Q, size.radius_nm, size.model)"
                ),
            ],
            api=api,
            uid=f"skyrmion-interface-{uuid.uuid4().hex[:8]}",
        )


__all__ = ["SkyrmionInterface"]
