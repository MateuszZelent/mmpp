"""High-level topology API bound to an MMPP dataset context."""

from __future__ import annotations

from typing import Any

import numpy as np

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
    def dataset_name(self) -> str:
        if self._dataset_name is None:
            self._dataset_name = self._job.get_largest_m_dataset()
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
        selected_ring = cfg.chirality_ring_r if chirality_ring_r is None else chirality_ring_r
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
                    "convention": getattr(selected_convention, "y_axis", "down"),
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
        from html import escape as _esc

        dataset = _esc(str(self.dataset_name))
        methods = [
            (".detect(t=0, method=...)", "Full topology detection → TopologyResult"),
            (".polarity(**kw)", "Shortcut → detect().polarity"),
            (".chirality(**kw)", "Shortcut → detect().chirality"),
            (".winding_number(**kw)", "Shortcut → detect().vorticity"),
            (".topological_charge(**kw)", "Shortcut → detect().Q"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        params = [
            ("t / frame", "0", "Time step / frame index to analyze"),
            ("method", "config", "Detection method"),
            ("z_layer", "config", "Z-layer to analyze"),
            ("polarity_threshold", "config", "Threshold for polarity detection"),
            ("chirality_ring_r", "config", "Ring radii (r_min, r_max) for chirality"),
            ("force", "False", "Force recomputation (bypass cache)"),
        ]
        param_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#a5b4fc;'>{_esc(d)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
            for n, d, desc in params
        )
        example = (
            "topo = vortex.topology.detect()\n"
            "print(f'p={topo.polarity}, c={topo.chirality}, Q={topo.Q}')\n"
            "\n"
            "# Quick accessors\n"
            "p = vortex.topology.polarity()\n"
            "c = vortex.topology.chirality()"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Topology Interface</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            f"Topological charge detection · dataset: {dataset}</div>"
            # Methods
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
            # Params
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>"
            "Parameters <span style='color:#94a3b8;font-weight:400;'>(.detect)</span></div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Arg</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{param_rows}</tbody></table></div>"
            # Examples
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )
