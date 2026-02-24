"""Result containers for hysteresis analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from html import escape as _esc
from typing import Any

import numpy as np
import pandas as pd

from .config import HysteresisConfig


@dataclass
class Branch:
    """Monotonic branch segment on a hysteresis loop."""

    name: str
    start: int
    stop: int
    cycle_id: int = 0
    is_major: bool = True

    @property
    def slice(self) -> slice:
        """Slice view corresponding to the branch."""
        return slice(int(self.start), int(self.stop))

    @property
    def n_points(self) -> int:
        return max(0, int(self.stop) - int(self.start))


@dataclass
class HysteresisResult:
    """Container for processed hysteresis data and fluent accessors."""

    field: np.ndarray
    magnetization: np.ndarray
    branches: list[Branch]
    frame_index: np.ndarray | None
    config: HysteresisConfig = field(default_factory=HysteresisConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.field = np.asarray(self.field, dtype=float).reshape(-1)
        self.magnetization = np.asarray(self.magnetization, dtype=float).reshape(-1)
        if self.frame_index is not None:
            self.frame_index = np.asarray(self.frame_index, dtype=int).reshape(-1)

    @cached_property
    def metrics(self):
        """Lazy metrics accessor."""
        from .metrics import MetricsAccessor

        return MetricsAccessor(self)

    @property
    def plot(self):
        """Fluent plotting namespace."""
        from .plot.accessor import HysteresisPlotAccessor

        return HysteresisPlotAccessor(self)

    @property
    def compare(self):
        """Comparison namespace rooted at this result."""
        from .comparison import ComparisonAccessor

        return ComparisonAccessor(self)

    def export(self, path, fmt: str | None = None):
        """Export result data/metrics to a selected format."""
        from .export import HysteresisExporter

        return HysteresisExporter(self).export(path, fmt=fmt)

    def to_dataframe(self) -> pd.DataFrame:
        """Return loop samples as a tabular dataframe."""
        branch_labels = np.full(self.field.shape[0], "unlabeled", dtype=object)
        cycle_ids = np.full(self.field.shape[0], -1, dtype=int)

        for branch in self.branches:
            branch_labels[branch.slice] = branch.name
            cycle_ids[branch.slice] = int(branch.cycle_id)

        data = {
            "field": self.field,
            "magnetization": self.magnetization,
            "branch": branch_labels,
            "cycle_id": cycle_ids,
        }
        if self.frame_index is not None and self.frame_index.size == self.field.size:
            data["frame_index"] = self.frame_index
        return pd.DataFrame(data)

    def cloneflip(self) -> "HysteresisResult":
        """Build a symmetric full loop from a single monotonic sweep.

        Applies the centrosymmetric constraint **M(−B) = −M(B)**, valid for
        reversible micromagnetic simulations where only one field polarity was
        computed.  Both the field axis and the magnetization axis are reflected
        around the origin and appended to form a closed two-branch loop
        (ascending + descending) with the same |B| range on both polarities.

        The returned result contains ``4N − 1`` points, where *N* is the
        number of original samples.  For the interactive snapshot explorer,
        reflected points are mapped to the original snapshot at the mirrored
        field value.

        Returns
        -------
        HysteresisResult
            New result — the original is **not** modified.
            ``result.metadata["cloneflip"] = True`` marks the copy.

        Example
        -------
        >>> result = job[0].analyze.hysteresis.load(
        ...     source="zarr_keys", key_prefix="B", component="y"
        ... )
        >>> full_loop = result.cloneflip()
        >>> full_loop.plot.interactive()
        """
        from .compute import build_cloneflip_result

        return build_cloneflip_result(self)

    def __repr__(self) -> str:
        return (
            "HysteresisResult("
            f"n={self.field.size}, branches={len(self.branches)}, "
            f"source={self.metadata.get('source_type', 'unknown')!r})"
        )

    def _repr_html_(self) -> str:
        try:
            hc = self.metrics.coercive_field
        except Exception:
            hc = None
        try:
            mr = self.metrics.remanence
        except Exception:
            mr = None
        try:
            ms = self.metrics.saturation_points
        except Exception:
            ms = None

        def _fmt(value: Any) -> str:
            try:
                if value is None:
                    return "n/a"
                if isinstance(value, (float, int)):
                    if np.isnan(float(value)):
                        return "n/a"
                    return f"{float(value):.4g}"
            except Exception:
                pass
            return _esc(str(value))

        hc_mean = _fmt(getattr(hc, "mean", None))
        mr_mean = _fmt(getattr(mr, "mean", None))
        ms_mean = _fmt(getattr(ms, "ms_mean", None))
        n_major = sum(1 for b in self.branches if b.is_major)

        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;\">"
            "<div style='font-size:1.05em;font-weight:600;color:#f1f5f9;'>"
            "HysteresisResult</div>"
            f"<div style='color:#94a3b8;font-size:0.85em;margin-top:4px;'>"
            f"points: {self.field.size} · branches: {len(self.branches)} · major: {n_major}</div>"
            "<table style='width:100%;margin-top:10px;border-collapse:collapse;font-size:0.9em;'>"
            "<tr><td style='padding:4px 8px;color:#93c5fd;font-family:monospace;'>Hc mean</td>"
            f"<td style='padding:4px 8px;color:#e2e8f0;'>{hc_mean}</td></tr>"
            "<tr><td style='padding:4px 8px;color:#93c5fd;font-family:monospace;'>Mr mean</td>"
            f"<td style='padding:4px 8px;color:#e2e8f0;'>{mr_mean}</td></tr>"
            "<tr><td style='padding:4px 8px;color:#93c5fd;font-family:monospace;'>Ms mean</td>"
            f"<td style='padding:4px 8px;color:#e2e8f0;'>{ms_mean}</td></tr>"
            "</table></div>"
        )

    def _repr_mimebundle_(self, include=None, exclude=None):
        html = self._repr_html_()
        text = self.__repr__()
        if html:
            return {"text/html": html, "text/plain": text}
        return {"text/plain": text}
