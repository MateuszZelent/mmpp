# ruff: noqa: UP007
"""Dataset-bound skyrmion topology interface."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from .._method_helpers import InteractiveNodeMixin
from ._core import detect_skyrmion
from .models import SkyrmionTopologyResult


class SkyrmionTopologyInterface(InteractiveNodeMixin):
    """Detect topology for one snapshot of a dataset-backed field."""

    _interactive_owner = "job[0].skyrmion.topology"
    _interactive_nodes = frozenset({"detect", "topological_charge", "center"})

    def __init__(self, parent: Any):
        self._parent = parent
        self._last_result: Optional[SkyrmionTopologyResult] = None

    @property
    def last_result(self) -> Optional[SkyrmionTopologyResult]:
        """Most recently computed result, if any."""
        return self._last_result

    def detect(
        self,
        *,
        t: Optional[int] = None,
        frame: int = 0,
        z_layer: int = -1,
        method: Optional[str] = None,
        mask: Optional[np.ndarray] = None,
        convention: Any = None,
        force: bool = False,
    ) -> SkyrmionTopologyResult:
        """Detect charge, centre, polarity, and skyrmion state."""
        if t is not None:
            frame = int(t)
        cfg = self._parent.config.topology
        selected_method = cfg.method if method is None else str(method).lower()
        selected_convention = cfg.convention if convention is None else convention
        effective = replace(
            cfg,
            method=selected_method,
            convention=selected_convention,
        )
        key = self._parent._cache_key(
            "topology",
            frame=frame,
            z_layer=z_layer,
            method=selected_method,
            convention=getattr(selected_convention, "y_axis", "up"),
            mask=mask,
        )
        if not force:
            cached = self._parent._result_cache.get(key)
            if isinstance(cached, SkyrmionTopologyResult):
                self._last_result = cached
                return cached

        data = self._parent._resolve_data()
        dx, dy = self._parent._resolve_spacing()
        result = detect_skyrmion(
            data,
            dx,
            dy,
            frame=frame,
            z_layer=z_layer,
            mask=mask,
            convention=selected_convention,
            config=effective,
        )
        self._parent._result_cache[key] = result
        self._last_result = result
        return result

    def topological_charge(self, **kwargs: Any) -> float:
        """Return the integrated topological charge ``Q``."""
        return float(self.detect(**kwargs).Q)

    def center(self, **kwargs: Any) -> tuple[float, float]:
        """Return the topology-guided physical centre in metres."""
        return self.detect(**kwargs).center_xy_m

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
            title="Skyrmion topology API help",
            prefix="job.solitons.skyrmion.topology",
            methods=["detect", "topological_charge", "center"],
            properties=[("last_result", "Most recent topology result")],
            chrome=False,
        )
        return node_card_html(
            "Skyrmion Topology",
            icon="🧭",
            subtitle="Berg–Lüscher or finite-difference topology",
            sections=[
                metrics_section_html(
                    [
                        (
                            "dataset",
                            self._parent.dataset_name or "auto",
                            NODE_COLOR_COMPUTE,
                        ),
                        (
                            "method",
                            self._parent.config.topology.method,
                            NODE_COLOR_ANALYSIS,
                        ),
                    ]
                ),
                accessors_section_html(
                    [("Compute:", [(".detect(frame=0)", NODE_COLOR_COMPUTE)])]
                ),
            ],
            api=api,
            uid=f"skyrmion-topology-{uuid.uuid4().hex[:8]}",
        )


__all__ = ["SkyrmionTopologyInterface"]
