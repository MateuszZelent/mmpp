# ruff: noqa: UP007
"""Dataset-bound skyrmion size fitting interface."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from .._method_helpers import InteractiveNodeMixin
from ._core import fit_skyrmion_size
from .models import SkyrmionSizeResult, SkyrmionTopologyResult


class SkyrmionSizeInterface(InteractiveNodeMixin):
    """Measure a skyrmion radial profile and fit an effective size model."""

    _interactive_owner = "job[0].skyrmion.size"
    _interactive_nodes = frozenset({"fit", "measure"})

    def __init__(self, parent: Any):
        self._parent = parent
        self._last_result: Optional[SkyrmionSizeResult] = None

    @property
    def last_result(self) -> Optional[SkyrmionSizeResult]:
        """Most recently computed size result, if any."""
        return self._last_result

    def fit(
        self,
        *,
        method: Optional[str] = None,
        t: Optional[int] = None,
        frame: int = 0,
        z_layer: int = -1,
        mask: Optional[np.ndarray] = None,
        convention: Any = None,
        topology: Optional[SkyrmionTopologyResult] = None,
        force: bool = False,
    ) -> SkyrmionSizeResult:
        """Fit ``ansatz``, ``domain_wall``, ``gaussian``, or select automatically."""
        if t is not None:
            frame = int(t)
        cfg = self._parent.config.size
        selected_method = cfg.method if method is None else str(method).lower()
        effective = replace(cfg, method=selected_method)
        selected_convention = (
            self._parent.config.topology.convention
            if convention is None
            else convention
        )
        topology_token = None
        if topology is not None:
            topology_token = (
                float(topology.Q),
                tuple(float(value) for value in topology.center_xy_m),
            )
        key = self._parent._cache_key(
            "size",
            frame=frame,
            z_layer=z_layer,
            method=selected_method,
            convention=getattr(selected_convention, "y_axis", "up"),
            mask=mask,
            topology=topology_token,
        )
        if not force:
            cached = self._parent._result_cache.get(key)
            if isinstance(cached, SkyrmionSizeResult):
                self._last_result = cached
                return cached

        data = self._parent._resolve_data()
        dx, dy = self._parent._resolve_spacing()
        result = fit_skyrmion_size(
            data,
            dx,
            dy,
            method=selected_method,
            frame=frame,
            z_layer=z_layer,
            mask=mask,
            convention=selected_convention,
            config=effective,
            topology=topology,
        )
        self._parent._result_cache[key] = result
        self._last_result = result
        return result

    def measure(self, **kwargs: Any) -> SkyrmionSizeResult:
        """Measure the model-independent contrast-crossing radius."""
        kwargs.setdefault("method", "threshold")
        return self.fit(**kwargs)

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
            title="Skyrmion size API help",
            prefix="job.solitons.skyrmion.size",
            methods=["fit", "measure"],
            properties=[("last_result", "Most recent size result")],
            chrome=False,
        )
        return node_card_html(
            "Skyrmion Size",
            icon="📏",
            subtitle="Radial profile, physical ansatz, and Gaussian fitting",
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
                            self._parent.config.size.method,
                            NODE_COLOR_ANALYSIS,
                        ),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Compute:",
                            [
                                (".fit(method='auto')", NODE_COLOR_COMPUTE),
                                (".measure()", NODE_COLOR_ANALYSIS),
                            ],
                        )
                    ]
                ),
            ],
            api=api,
            uid=f"skyrmion-size-{uuid.uuid4().hex[:8]}",
        )


__all__ = ["SkyrmionSizeInterface"]
