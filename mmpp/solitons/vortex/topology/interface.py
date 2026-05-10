"""High-level topology API bound to an MMPP dataset context."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from mmpp._repr_helpers import api_help_html, html_tabs
from mmpp._shared.repr_html import make_simple_card

from .._cache import InMemoryResultCache, build_cache_key
from ..config import VortexConfig
from .detection import detect_topology
from .models import TopologyResult


class TopologyInterface:
    """Topology analysis for a dataset-backed vortex signal."""

    def __init__(
        self,
        job_result,
        dataset_name: str | None,
        slice_info: Any | None,
        config: VortexConfig,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config
        self._last_result: TopologyResult | None = None
        self._cache = InMemoryResultCache(job_result, namespace="topology")

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

    def _resolve_dataset_array(self) -> np.ndarray:
        dataset = getattr(self._job, self.dataset_name)
        if self._slice_info is not None:
            dataset = dataset[self._slice_info]
        return np.asarray(dataset.numpy(copy=False), dtype=float)

    def _resolve_spacing(self) -> tuple[float, float]:
        attrs = self._job.attrs
        dx = attrs.get("dx", attrs.get("cellsize_x", 1.0))
        dy = attrs.get("dy", attrs.get("cellsize_y", 1.0))
        return float(dx), float(dy)

    def detect(
        self,
        *,
        t: int | None = None,
        frame: int = 0,
        method: str | None = None,
        z_layer: int | None = None,
        polarity_threshold: float | None = None,
        chirality_ring_r: tuple[float, float] | None = None,
        convention=None,
        force: bool = False,
    ) -> TopologyResult:
        """Detect vortex topology for a selected frame."""
        if t is not None:
            frame = int(t)

        data = self._resolve_dataset_array()
        dx, dy = self._resolve_spacing()

        cfg = self._config.topology
        selected_method = method or cfg.method
        selected_z = cfg.z_layer if z_layer is None else z_layer
        selected_p = (
            cfg.polarity_threshold
            if polarity_threshold is None
            else float(polarity_threshold)
        )
        selected_ring = (
            cfg.chirality_ring_r if chirality_ring_r is None else chirality_ring_r
        )
        selected_convention = cfg.convention if convention is None else convention

        key, config_json = build_cache_key(
            selected_method,
            namespace="topology",
            config_payload={
                "dataset_name": self.dataset_name,
                "slice_info": repr(self._slice_info),
                "frame": int(frame),
                "z_layer": int(selected_z),
                "dx": float(dx),
                "dy": float(dy),
                "shape": tuple(int(v) for v in data.shape),
                "params": {
                    "polarity_threshold": float(selected_p),
                    "chirality_ring_r": None
                    if selected_ring is None
                    else tuple(float(v) for v in selected_ring),
                    "convention": getattr(selected_convention, "y_axis", "up"),
                },
            },
        )
        if not force and self._cache.has(key, config_json):
            return self._cache.get(key)

        result = detect_topology(
            data,
            dx,
            dy,
            method=selected_method,
            frame=frame,
            z_layer=selected_z,
            polarity_threshold=selected_p,
            chirality_ring_r=selected_ring,
            convention=selected_convention,
        )
        self._last_result = result
        self._cache.put(key, result, config_json)
        return result

    def polarity(self, **kwargs) -> int:
        """Return detected polarity."""
        return int(self.detect(**kwargs).polarity)

    def chirality(self, **kwargs) -> int:
        """Return detected chirality."""
        return int(self.detect(**kwargs).chirality)

    def winding_number(self, **kwargs) -> int:
        """Return detected vorticity/winding number sign."""
        return int(self.detect(**kwargs).vorticity)

    def topological_charge(self, **kwargs) -> float:
        """Return detected topological charge Q."""
        return float(self.detect(**kwargs).Q)

    def _repr_html_(self) -> str:
        methods = [
            (".detect(t=0, method='finite_diff')", "Detect full topology for frame"),
            (".polarity(...)", "Return p in {-1, +1}"),
            (".chirality(...)", "Return chirality C in {-1, 0, +1}"),
            (".winding_number(...)", "Return vorticity sign"),
            (".topological_charge(...)", "Return skyrmion number Q"),
        ]
        overview = make_simple_card(
            title="Topology Interface",
            subtitle=f"Snapshot topology analysis for dataset '{self.dataset_name}'",
            rows=methods,
        )
        api = api_help_html(
            self,
            title="Topology API help",
            prefix="vortex.topology",
            properties=[("dataset_name", "Resolved magnetization dataset name")],
            methods=[
                "detect",
                "polarity",
                "chirality",
                "winding_number",
                "topological_charge",
            ],
            subtitle="Live public API for snapshot topology analysis.",
            chrome=False,
        )
        return (
            '<div style=\'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;'>"
            + html_tabs(
                [("Overview", overview), ("API", api)],
                uid=f"mmpp-vortex-topology-{uuid.uuid4().hex}",
            )
            + "</div>"
        )


__all__ = ["TopologyInterface"]
