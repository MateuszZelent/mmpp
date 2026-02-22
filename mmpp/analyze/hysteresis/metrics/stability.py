"""Cycle-to-cycle stability analysis for hysteresis loops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..result import Branch, HysteresisResult

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False


def _cycle_ranges(result: HysteresisResult) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    cycle_ids = sorted({int(b.cycle_id) for b in result.branches})
    for cid in cycle_ids:
        segments = [b for b in result.branches if int(b.cycle_id) == cid]
        if not segments:
            continue
        start = min(int(b.start) for b in segments)
        stop = max(int(b.stop) for b in segments)
        if stop > start:
            out.append((cid, start, stop))
    return out


def _make_cycle_result(result: HysteresisResult, cid: int, start: int, stop: int) -> HysteresisResult:
    cycle_branches: list[Branch] = []
    for branch in result.branches:
        if int(branch.cycle_id) != cid:
            continue
        s = max(int(branch.start), start)
        e = min(int(branch.stop), stop)
        if e <= s:
            continue
        cycle_branches.append(
            Branch(
                name=branch.name,
                start=s - start,
                stop=e - start,
                cycle_id=0,
                is_major=True,
            )
        )

    frame_idx = None
    if result.frame_index is not None:
        frame_idx = np.asarray(result.frame_index[start:stop], dtype=int)

    meta = dict(result.metadata)
    meta["cycle_id"] = int(cid)

    return HysteresisResult(
        field=np.asarray(result.field[start:stop], dtype=float),
        magnetization=np.asarray(result.magnetization[start:stop], dtype=float),
        branches=cycle_branches if cycle_branches else [Branch("ascending", 0, stop - start, 0, True)],
        frame_index=frame_idx,
        config=result.config,
        metadata=meta,
    )


def _resample(values: np.ndarray, n_samples: int = 200) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(int(n_samples), dtype=float)
    if arr.size == 1:
        return np.full(int(n_samples), float(arr[0]), dtype=float)
    x = np.linspace(0.0, 1.0, arr.size)
    grid = np.linspace(0.0, 1.0, int(n_samples))
    return np.interp(grid, x, arr)


@dataclass
class CycleStabilityAnalysis:
    """Cycle stability diagnostics."""

    cycles: list[HysteresisResult]
    hc_drift: np.ndarray
    mr_drift: np.ndarray
    ms_drift: np.ndarray
    correlation_matrix: np.ndarray
    convergence_index: float

    def plot_drift(self):
        """Plot drift traces over cycle index."""
        if not _HAS_MPL:
            raise ImportError("Matplotlib is required for drift plotting")
        fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=120)
        x = np.arange(1, len(self.cycles) + 1)
        ax.plot(x, self.hc_drift, marker="o", label="ΔHc")
        ax.plot(x, self.mr_drift, marker="o", label="ΔMr")
        ax.plot(x, self.ms_drift, marker="o", label="ΔMs")
        ax.axhline(0.0, color="#64748b", ls="--", lw=1.0)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Drift vs cycle 1")
        ax.set_title("Cycle stability drift")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax

    def is_stable(self, threshold: float = 0.01) -> bool:
        """Return ``True`` when all drift magnitudes are below threshold."""
        th = float(abs(threshold))
        return bool(
            np.nanmax(np.abs(self.hc_drift)) <= th
            and np.nanmax(np.abs(self.mr_drift)) <= th
            and np.nanmax(np.abs(self.ms_drift)) <= th
        )


def analyze_cycle_stability(result: HysteresisResult) -> CycleStabilityAnalysis:
    """Compute cycle drift and convergence metrics."""
    ranges = _cycle_ranges(result)
    if not ranges:
        ranges = [(0, 0, int(result.field.size))]

    cycles = [_make_cycle_result(result, cid, start, stop) for cid, start, stop in ranges]

    hc = np.array([float(c.metrics.coercive_field.mean) for c in cycles], dtype=float)
    mr = np.array([float(c.metrics.remanence.mean) for c in cycles], dtype=float)
    ms = np.array([float(c.metrics.saturation_points.ms_mean) for c in cycles], dtype=float)

    hc_drift = hc - hc[0]
    mr_drift = mr - mr[0]
    ms_drift = ms - ms[0]

    curves = np.vstack([_resample(c.magnetization, n_samples=200) for c in cycles])
    if curves.shape[0] == 1:
        corr = np.array([[1.0]], dtype=float)
    else:
        corr = np.corrcoef(curves)

    if corr.size <= 1:
        convergence = 1.0
    else:
        upper = corr[np.triu_indices_from(corr, k=1)]
        convergence = float(np.nanmean(upper)) if upper.size else 1.0

    return CycleStabilityAnalysis(
        cycles=cycles,
        hc_drift=hc_drift,
        mr_drift=mr_drift,
        ms_drift=ms_drift,
        correlation_matrix=corr,
        convergence_index=convergence,
    )


__all__ = ["CycleStabilityAnalysis", "analyze_cycle_stability"]
