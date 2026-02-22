"""Comparison helpers for hysteresis results."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape as _esc
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False


def _to_scalar(value: Any) -> float:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if hasattr(value, "mean"):
        mean_value = value.mean
        if isinstance(mean_value, (float, int, np.floating, np.integer)):
            return float(mean_value)
    return float("nan")


def _single_valued_curve(field: np.ndarray, magnetization: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(field, dtype=float))
    f_sorted = np.asarray(field, dtype=float)[order]
    m_sorted = np.asarray(magnetization, dtype=float)[order]

    unique_f, inverse = np.unique(f_sorted, return_inverse=True)
    sums = np.zeros(unique_f.size, dtype=float)
    counts = np.zeros(unique_f.size, dtype=float)
    for idx, inv in enumerate(inverse):
        sums[inv] += m_sorted[idx]
        counts[inv] += 1.0
    counts[counts == 0.0] = 1.0
    return unique_f, sums / counts


@dataclass
class HysteresisComparison:
    """Container for two hysteresis results and comparison utilities."""

    left: Any
    right: Any
    label: tuple[str, str]

    @property
    def plot(self):
        return HysteresisComparisonPlotAccessor(self)

    @property
    def delta_metrics(self) -> dict[str, tuple[float, float, float]]:
        names = [
            "loop_area",
            "squareness",
            "exchange_bias",
            "coercive_field",
            "remanence",
            "max_susceptibility",
        ]
        out: dict[str, tuple[float, float, float]] = {}
        for name in names:
            left_value = _to_scalar(getattr(self.left.metrics, name))
            right_value = _to_scalar(getattr(self.right.metrics, name))
            out[name] = (left_value, right_value, right_value - left_value)
        return out

    def __repr__(self) -> str:
        return f"<HysteresisComparison: {self.label[0]} vs {self.label[1]}>"

    def _repr_html_(self) -> str:
        left_label = _esc(self.label[0])
        right_label = _esc(self.label[1])
        rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(name)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{vals[0]:.6g}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{vals[1]:.6g}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{vals[2]:.6g}</td></tr>"
            for name, vals in self.delta_metrics.items()
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;\">"
            f"<div style='font-size:1.03em;font-weight:600;color:#f1f5f9;'>Comparison: {left_label} vs {right_label}</div>"
            "<table style='width:100%;margin-top:8px;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:6px 8px;color:#e2e8f0;'>Metric</th>"
            f"<th style='padding:6px 8px;color:#e2e8f0;'>{left_label}</th>"
            f"<th style='padding:6px 8px;color:#e2e8f0;'>{right_label}</th>"
            "<th style='padding:6px 8px;color:#e2e8f0;'>Δ</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )


class HysteresisComparisonPlotAccessor:
    """Plot namespace for :class:`HysteresisComparison`."""

    def __init__(self, comparison: HysteresisComparison):
        self._comparison = comparison

    def overlay(self, *, ax=None, figsize: tuple[float, float] = (7.5, 5.0), dpi: int = 120):
        if not _HAS_MPL:
            raise ImportError("Matplotlib is required for comparison plotting")

        left = self._comparison.left
        right = self._comparison.right
        label_left, label_right = self._comparison.label

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure
        ax.plot(left.field, left.magnetization, lw=1.8, alpha=0.9, label=label_left)
        ax.plot(right.field, right.magnetization, lw=1.8, alpha=0.9, label=label_right)
        ax.set_xlabel(f"Field [{left.metadata.get('field_unit', 'input')}]")
        ax.set_ylabel("Magnetization")
        ax.set_title("Hysteresis overlay")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax

    def difference(
        self,
        *,
        n_grid: int = 400,
        ax=None,
        figsize: tuple[float, float] = (7.5, 5.0),
        dpi: int = 120,
    ):
        if not _HAS_MPL:
            raise ImportError("Matplotlib is required for comparison plotting")

        left = self._comparison.left
        right = self._comparison.right
        label_left, label_right = self._comparison.label

        f1, m1 = _single_valued_curve(left.field, left.magnetization)
        f2, m2 = _single_valued_curve(right.field, right.magnetization)
        lo = max(float(np.min(f1)), float(np.min(f2)))
        hi = min(float(np.max(f1)), float(np.max(f2)))
        if hi <= lo:
            raise ValueError("Field ranges do not overlap - cannot compute M(B) difference")

        grid = np.linspace(lo, hi, int(max(50, n_grid)))
        interp1 = np.interp(grid, f1, m1)
        interp2 = np.interp(grid, f2, m2)
        diff = interp1 - interp2

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure
        ax.plot(grid, diff, color="#0891b2", lw=1.8)
        ax.axhline(0.0, color="#64748b", lw=1.0, ls="--")
        ax.set_xlabel(f"Field [{left.metadata.get('field_unit', 'input')}]")
        ax.set_ylabel(f"{label_left} - {label_right}")
        ax.set_title("Difference M(B)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig, ax, {"field": grid, "difference": diff}

    def metrics_table(self) -> pd.DataFrame:
        left_label, right_label = self._comparison.label
        left = self._comparison.left.metrics.report()[["metric", "value"]].copy()
        right = self._comparison.right.metrics.report()[["metric", "value"]].copy()
        left = left.rename(columns={"value": left_label}).drop_duplicates("metric")
        right = right.rename(columns={"value": right_label}).drop_duplicates("metric")
        merged = left.merge(right, on="metric", how="outer")

        lnum = pd.to_numeric(merged[left_label], errors="coerce")
        rnum = pd.to_numeric(merged[right_label], errors="coerce")
        merged["delta"] = rnum - lnum
        return merged


class ComparisonAccessor:
    """Namespace exposed as ``result.compare``."""

    def __init__(self, result):
        self._result = result

    def with_(
        self,
        other,
        *,
        label: tuple[str, str] = ("result_1", "result_2"),
    ) -> HysteresisComparison:
        return HysteresisComparison(left=self._result, right=other, label=label)

    def __repr__(self) -> str:
        return "<ComparisonAccessor: .with_(other, label=('a','b'))>"


__all__ = [
    "ComparisonAccessor",
    "HysteresisComparison",
    "HysteresisComparisonPlotAccessor",
]
