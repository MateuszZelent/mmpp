"""Multi-branch peak linking via Hungarian assignment.

Detects multiple dispersion branches simultaneously by:
1. Finding spectral peaks per k-bin (scipy.signal.find_peaks)
2. Linking peaks across adjacent k-bins via optimal assignment
   (scipy.optimize.linear_sum_assignment)
3. Handling branch birth/death when peaks appear/disappear
4. Optional Gaussian smoothing of final branches

Usage::

    result = job[0].fft.dispersion.filters(...).compute_1d(axis='x')
    branches = result.analyze.find_branches(n_branches=3)
    branches.plot()

"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ..models import DispersionResult1D

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Per‑column peak detection
# ──────────────────────────────────────────────────────────────────────


def _find_peaks_column(
    spectrum: np.ndarray,
    f_axis: np.ndarray,
    *,
    n_peaks: int = 5,
    min_prominence_rel: float = 0.02,
    min_distance_bins: int = 3,
    fmin_hz: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find up to *n_peaks* spectral peaks in a single k‑bin.

    Returns (f_peak_hz, amplitudes) sorted by amplitude descending.
    """
    # Restrict to positive / above fmin
    keep = f_axis >= fmin_hz
    spec = spectrum[keep]
    f_sub = f_axis[keep]

    if spec.size < 3:
        return np.array([]), np.array([])

    global_max = float(spec.max())
    if global_max <= 0:
        return np.array([]), np.array([])

    try:
        from scipy.signal import find_peaks as _scipy_find_peaks

        prominence = min_prominence_rel * global_max
        idx, props = _scipy_find_peaks(
            spec,
            distance=min_distance_bins,
            prominence=prominence,
        )
    except ImportError:
        # Fallback: simple local‑max scan
        idx = []
        for i in range(1, len(spec) - 1):
            if spec[i] > spec[i - 1] and spec[i] > spec[i + 1]:
                if spec[i] > min_prominence_rel * global_max:
                    idx.append(i)
        idx = np.asarray(idx, dtype=int)
        props = {}

    if len(idx) == 0:
        # If nothing found, take global max
        idx = np.array([int(np.argmax(spec))])

    # Sort by amplitude descending, keep top n_peaks
    order = np.argsort(spec[idx])[::-1][:n_peaks]
    idx = idx[order]

    return f_sub[idx], spec[idx]


# ──────────────────────────────────────────────────────────────────────
# Hungarian linking
# ──────────────────────────────────────────────────────────────────────

_UNLINKED_COST = 1e18


def _link_peaks(
    prev_f: np.ndarray,
    curr_f: np.ndarray,
    max_df_hz: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Link peaks between two adjacent k‑bins.

    Returns
    -------
    matches : list of (prev_idx, curr_idx) pairs
    unmatched_prev : indices of peaks in prev that got no match
    unmatched_curr : indices of peaks in curr that got no match (new branches)
    """
    n_prev = len(prev_f)
    n_curr = len(curr_f)

    if n_prev == 0:
        return [], [], list(range(n_curr))
    if n_curr == 0:
        return [], list(range(n_prev)), []

    # Build cost matrix |Δf|, with gating
    cost = np.full((n_prev, n_curr), _UNLINKED_COST)
    for i in range(n_prev):
        for j in range(n_curr):
            df = abs(prev_f[i] - curr_f[j])
            if df <= max_df_hz:
                cost[i, j] = df

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
    except ImportError:
        # Greedy fallback
        row_ind, col_ind = _greedy_assign(cost)

    matches = []
    matched_prev = set()
    matched_curr = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < _UNLINKED_COST:
            matches.append((int(r), int(c)))
            matched_prev.add(int(r))
            matched_curr.add(int(c))

    unmatched_prev = [i for i in range(n_prev) if i not in matched_prev]
    unmatched_curr = [j for j in range(n_curr) if j not in matched_curr]

    return matches, unmatched_prev, unmatched_curr


def _greedy_assign(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple greedy assignment fallback when scipy is unavailable."""
    rows, cols = [], []
    used_rows: set = set()
    used_cols: set = set()
    flat = np.argsort(cost, axis=None)
    n_prev, n_curr = cost.shape

    for idx in flat:
        r, c = divmod(int(idx), n_curr)
        if cost[r, c] >= _UNLINKED_COST:
            break
        if r not in used_rows and c not in used_cols:
            rows.append(r)
            cols.append(c)
            used_rows.add(r)
            used_cols.add(c)

    return np.array(rows, dtype=int), np.array(cols, dtype=int)


# ──────────────────────────────────────────────────────────────────────
# Full multi‑branch tracker
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TrackedBranch:
    """A single branch: parallel arrays of (k, f, amplitude)."""

    k: np.ndarray
    f_hz: np.ndarray
    amplitude: np.ndarray
    branch_id: int = 0

    @property
    def f_ghz(self) -> np.ndarray:
        return self.f_hz / 1e9

    @property
    def k_rad_um(self) -> np.ndarray:
        return self.k / 1e6

    def __len__(self) -> int:
        return len(self.k)


@dataclass
class BranchesResult:
    """Result of multi‑branch detection.

    Attributes
    ----------
    branches : list[TrackedBranch]
        All detected branches, sorted by mean frequency (ascending).
    result : DispersionResult1D
        Back-reference to the dispersion data.
    """

    branches: List[TrackedBranch]
    result: "DispersionResult1D"

    # ---- convenience ----

    def __len__(self) -> int:
        return len(self.branches)

    def __getitem__(self, idx: int) -> TrackedBranch:
        return self.branches[idx]

    def __iter__(self):
        return iter(self.branches)

    @property
    def plot(self) -> "BranchesPlotAccessor":
        return BranchesPlotAccessor(self)

    def __repr__(self) -> str:
        lines = [f"BranchesResult({len(self.branches)} branches):"]
        for br in self.branches:
            f_min = float(br.f_hz.min()) / 1e9
            f_max = float(br.f_hz.max()) / 1e9
            lines.append(
                f"  branch {br.branch_id}: "
                f"{len(br)} pts, f=[{f_min:.3f}..{f_max:.3f}] GHz"
            )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        from html import escape as _e

        rows = []
        for br in self.branches:
            f_min = float(br.f_hz.min()) / 1e9
            f_max = float(br.f_hz.max()) / 1e9
            k_lo = float(br.k.min()) / 1e6
            k_hi = float(br.k.max()) / 1e6
            rows.append(
                f"<tr>"
                f"<td style='padding:3px 10px;color:#93c5fd;font-weight:700;'>{br.branch_id}</td>"
                f"<td style='padding:3px 10px;'>{len(br)} pts</td>"
                f"<td style='padding:3px 10px;'>{f_min:.3f} – {f_max:.3f} GHz</td>"
                f"<td style='padding:3px 10px;'>{k_lo:.2f} – {k_hi:.2f} rad/μm</td>"
                f"</tr>"
            )
        tbody = "".join(rows)
        return (
            "<div style='font-family:sans-serif;border:2px solid #1e3a5f;"
            "border-left:4px solid #a78bfa;border-radius:10px;padding:14px;"
            "margin:6px 0;background:linear-gradient(135deg,#0f172a,#0c1a35);"
            "color:#e2e8f0;max-width:600px;'>"
            "<div style='font-weight:700;font-size:1.0em;color:#f1f5f9;"
            "margin-bottom:8px;'>🌊 BranchesResult</div>"
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<tr style='border-bottom:1px solid #334155;'>"
            f"<th style='text-align:left;padding:3px 10px;color:#94a3b8;font-size:.8em;'>ID</th>"
            f"<th style='text-align:left;padding:3px 10px;color:#94a3b8;font-size:.8em;'>Points</th>"
            f"<th style='text-align:left;padding:3px 10px;color:#94a3b8;font-size:.8em;'>f range</th>"
            f"<th style='text-align:left;padding:3px 10px;color:#94a3b8;font-size:.8em;'>k range</th>"
            f"</tr>{tbody}</table>"
            "<div style='margin-top:6px;font-size:.78em;color:#64748b;'>"
            "Use <code>.plot()</code> or <code>.plot.overlay(ax)</code> to visualize.</div>"
            "</div>"
        )


# ──────────────────────────────────────────────────────────────────────
# Main algorithm
# ──────────────────────────────────────────────────────────────────────


def find_branches(
    result: "DispersionResult1D",
    *,
    n_branches: int = 5,
    side: str = "both",
    min_prominence: float = 0.02,
    min_peak_distance: int = 3,
    max_df_ghz: float = 0.5,
    min_branch_length: int = 10,
    smooth_sigma: Optional[float] = 2.0,
    fmin_hz: Union[float, str, None] = "auto",
    k_min_rad_um: float = 0.0,
    k_max_rad_um: Optional[float] = None,
) -> BranchesResult:
    """Detect multiple dispersion branches via Hungarian peak linking.

    Algorithm
    ---------
    1. For each k-bin, detect up to *n_branches* spectral peaks.
    2. Walk along k (left→right) and link peaks between adjacent bins
       using the Hungarian algorithm (optimal assignment minimizing |Δf|).
    3. When a peak has no match, a new branch is born.
    4. When a branch loses its match for too many consecutive k-bins,
       it is terminated.
    5. Optionally smooth each branch with a Gaussian filter.
    6. Discard branches shorter than *min_branch_length*.

    Parameters
    ----------
    result : DispersionResult1D
        Dispersion data.
    n_branches : int
        Max peaks to detect per k-bin.
    side : ``"positive"`` | ``"negative"`` | ``"both"``
        Which k-half to search.
    min_prominence : float
        Rel. prominence threshold for peak detection (fraction of max).
    min_peak_distance : int
        Min frequency bins between peaks.
    max_df_ghz : float
        Max allowed frequency jump between adjacent k-bins [GHz].
    min_branch_length : int
        Discard branches shorter than this many k-bins.
    smooth_sigma : float or None
        Gaussian smoothing sigma (in k-bins) on final branches.
    fmin_hz : float, ``"auto"``, or None
        Min frequency cutoff.
    k_min_rad_um, k_max_rad_um : float
        k search window [rad/μm].

    Returns
    -------
    BranchesResult
    """
    S = result.S
    k_axis = result.k_axis
    f_axis = result.f_axis

    # Positive freq only
    pos_f = f_axis >= 0
    f_pos = f_axis[pos_f]
    S_pos = S[:, pos_f]

    # fmin cutoff
    if fmin_hz == "auto":
        fmin_cutoff = 0.05 * float(f_pos.max())
    elif fmin_hz is not None and fmin_hz > 0:
        fmin_cutoff = float(fmin_hz)
    else:
        fmin_cutoff = 0.0

    # k-side mask
    k_min_rm = k_min_rad_um * 1e6
    k_max_rm = (k_max_rad_um * 1e6) if k_max_rad_um is not None else np.inf

    if side == "positive":
        k_mask = (k_axis > k_min_rm) & (k_axis <= k_max_rm)
    elif side == "negative":
        k_mask = (k_axis < -k_min_rm) & (k_axis >= -k_max_rm)
    else:
        k_mask = (np.abs(k_axis) >= k_min_rm) & (np.abs(k_axis) <= k_max_rm)

    k_idx = np.where(k_mask)[0]
    if len(k_idx) == 0:
        raise ValueError(f"No k-bins for side={side!r}, k_min={k_min_rad_um}")

    # Sort k indices by k value (left→right)
    k_idx = k_idx[np.argsort(k_axis[k_idx])]

    max_df_hz = max_df_ghz * 1e9

    # ── Phase 1: per-column peak detection ──
    peaks_per_col: List[Tuple[np.ndarray, np.ndarray]] = []
    for ik in k_idx:
        fp, amp = _find_peaks_column(
            S_pos[ik],
            f_pos,
            n_peaks=n_branches,
            min_prominence_rel=min_prominence,
            min_distance_bins=min_peak_distance,
            fmin_hz=fmin_cutoff,
        )
        peaks_per_col.append((fp, amp))

    # ── Phase 2: Hungarian linking ──
    # active_branches: dict branch_id → {k_list, f_list, amp_list, last_f}
    active: dict[int, dict] = {}
    finished: list[dict] = []
    next_id = 0
    MAX_GAP = 5  # max consecutive misses before terminating a branch

    for col_i, ik in enumerate(k_idx):
        k_val = k_axis[ik]
        curr_f, curr_amp = peaks_per_col[col_i]

        if len(active) == 0:
            # Initialize all peaks as new branches
            for j in range(len(curr_f)):
                active[next_id] = {
                    "k": [k_val],
                    "f": [curr_f[j]],
                    "amp": [curr_amp[j]],
                    "last_f": curr_f[j],
                    "gap": 0,
                }
                next_id += 1
            continue

        # Build prev_f from active branches
        active_ids = list(active.keys())
        prev_f = np.array([active[bid]["last_f"] for bid in active_ids])

        matches, unmatched_prev, unmatched_curr = _link_peaks(
            prev_f, curr_f, max_df_hz
        )

        # Update matched branches
        matched_ids = set()
        for pi, ci in matches:
            bid = active_ids[pi]
            active[bid]["k"].append(k_val)
            active[bid]["f"].append(curr_f[ci])
            active[bid]["amp"].append(curr_amp[ci])
            active[bid]["last_f"] = curr_f[ci]
            active[bid]["gap"] = 0
            matched_ids.add(bid)

        # Increment gap for unmatched active branches
        for pi in unmatched_prev:
            bid = active_ids[pi]
            if bid not in matched_ids:
                active[bid]["gap"] += 1

        # Terminate branches with too many gaps
        to_remove = []
        for bid, br in active.items():
            if br["gap"] > MAX_GAP:
                to_remove.append(bid)
        for bid in to_remove:
            finished.append(active.pop(bid))

        # Start new branches for unmatched current peaks
        for ci in unmatched_curr:
            active[next_id] = {
                "k": [k_val],
                "f": [curr_f[ci]],
                "amp": [curr_amp[ci]],
                "last_f": curr_f[ci],
                "gap": 0,
            }
            next_id += 1

    # Move remaining active branches to finished
    for br in active.values():
        finished.append(br)

    # ── Phase 3: build TrackedBranch objects ──
    tracked: List[TrackedBranch] = []
    for br_data in finished:
        if len(br_data["k"]) < min_branch_length:
            continue

        k_arr = np.array(br_data["k"])
        f_arr = np.array(br_data["f"])
        amp_arr = np.array(br_data["amp"])

        # Optional smoothing
        if smooth_sigma and smooth_sigma > 0 and len(f_arr) > 3:
            try:
                from scipy.ndimage import gaussian_filter1d
                f_arr = gaussian_filter1d(f_arr, sigma=smooth_sigma)
            except ImportError:
                w = max(1, int(smooth_sigma * 2))
                kernel = np.ones(w) / w
                f_arr = np.convolve(f_arr, kernel, mode="same")

        tracked.append(TrackedBranch(
            k=k_arr,
            f_hz=f_arr,
            amplitude=amp_arr,
            branch_id=len(tracked),
        ))

    # Sort by mean frequency
    tracked.sort(key=lambda b: float(b.f_hz.mean()))
    for i, br in enumerate(tracked):
        br.branch_id = i

    logger.info(
        "Found %d branches (from %d candidates, min_length=%d)",
        len(tracked), len(finished), min_branch_length,
    )

    return BranchesResult(branches=tracked, result=result)


# ──────────────────────────────────────────────────────────────────────
# Plot accessor
# ──────────────────────────────────────────────────────────────────────

# Default distinct colors for branches
_BRANCH_COLORS = [
    "#f43f5e", "#3b82f6", "#22c55e", "#eab308", "#a855f7",
    "#06b6d4", "#f97316", "#ec4899", "#14b8a6", "#8b5cf6",
]


class BranchesPlotAccessor:
    """Plotting namespace for :class:`BranchesResult`."""

    def __init__(self, branches_result: BranchesResult) -> None:
        self._br = branches_result

    def __call__(self, **kwargs) -> Tuple["Figure", "Axes"]:
        """Default: overlay on heatmap."""
        return self.heatmap(**kwargs)

    def heatmap(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: Tuple[float, float] = (12, 8),
        dpi: Optional[int] = None,
        cmap: str = "cmc.davos",
        kscale: str = "rad_um",
        f_units: str = "GHz",
        fmax: Optional[float] = None,
        lognorm: bool = True,
        linewidth: float = 2.0,
        show_legend: bool = True,
        title: Optional[str] = None,
        save: Union[str, Any, bool, None] = None,
    ) -> Tuple["Figure", "Axes"]:
        """S(k,f) heatmap with all branches overlaid."""
        br = self._br

        fig, ax = br.result.plot.heatmap(
            ax=ax, figsize=figsize, dpi=dpi, cmap=cmap,
            kscale=kscale, f_units=f_units, fmax=fmax, lognorm=lognorm,
            title=title,
        )

        self.overlay(
            ax, kscale=kscale, f_units=f_units,
            linewidth=linewidth, show_legend=show_legend,
        )

        if save not in (None, False):
            br.result.plot._save_fig(fig, save, br.result)

        return fig, ax

    def overlay(
        self,
        ax: "Axes",
        *,
        kscale: str = "rad_um",
        f_units: str = "GHz",
        linewidth: float = 2.0,
        colors: Optional[List[str]] = None,
        show_legend: bool = True,
    ) -> None:
        """Overlay branch curves on existing axes."""
        colors = colors or _BRANCH_COLORS
        br = self._br

        for branch in br.branches:
            k_plot = branch.k.copy()
            f_plot = branch.f_hz.copy()

            if kscale == "rad_um":
                k_plot = k_plot / 1e6
            elif kscale == "meter":
                k_plot = k_plot / (2 * np.pi)

            if f_units == "GHz":
                f_plot = f_plot / 1e9

            color = colors[branch.branch_id % len(colors)]
            ax.plot(
                k_plot, f_plot,
                color=color,
                linewidth=linewidth,
                alpha=0.9,
                label=f"branch {branch.branch_id}",
            )

        if show_legend:
            ax.legend(fontsize=8, loc="upper right")

    def branches(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: Tuple[float, float] = (10, 5),
        dpi: Optional[int] = None,
        kscale: str = "rad_um",
        f_units: str = "GHz",
        title: Optional[str] = None,
    ) -> Tuple["Figure", "Axes"]:
        """Plot only the extracted branches (no heatmap background)."""
        import matplotlib.pyplot as plt

        br = self._br
        colors = _BRANCH_COLORS

        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize, **({} if dpi is None else {"dpi": dpi})
            )
        else:
            fig = ax.get_figure()

        for branch in br.branches:
            k_plot = branch.k.copy()
            f_plot = branch.f_hz.copy()

            if kscale == "rad_um":
                k_plot /= 1e6
            if f_units == "GHz":
                f_plot /= 1e9

            color = colors[branch.branch_id % len(colors)]
            ax.plot(
                k_plot, f_plot,
                color=color, linewidth=2.0,
                marker=".", markersize=3, alpha=0.8,
                label=f"branch {branch.branch_id} ({len(branch)} pts)",
            )

        k_label = {"rad_um": r"$k$ [rad/μm]", "meter": r"$k$ [m$^{-1}$]"}.get(
            kscale, r"$k$ [rad/m]"
        )
        f_label = "f [GHz]" if f_units == "GHz" else "f [Hz]"
        ax.set_xlabel(k_label)
        ax.set_ylabel(f_label)
        ax.set_title(title or f"Tracked Branches ({len(br)} found)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        try:
            fig.tight_layout()
        except Exception:
            pass

        return fig, ax

    def __repr__(self) -> str:
        return "<BranchesPlotAccessor: .heatmap(), .overlay(ax), .branches()>"
