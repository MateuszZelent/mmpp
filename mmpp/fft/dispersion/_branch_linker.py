"""Multi-branch peak linking via Hungarian assignment.

Detects multiple dispersion branches simultaneously by:
1. Finding spectral peaks per k-bin (scipy.signal.find_peaks)
2. SNR gating: skip k-bins with weak signal
3. Linking peaks across adjacent k-bins via optimal assignment
   (scipy.optimize.linear_sum_assignment) with amplitude-weighted cost
4. Handling branch birth/death when peaks appear/disappear
5. Quality filtering: discard noisy/short branches
6. Optional Gaussian smoothing of final branches

Usage::

    result = job[0].fft.dispersion.filters(...).compute_1d(axis='x')
    branches = result.analyze.find_branches(n_branches=3)
    branches.plot()

"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

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
    n_peaks: int = 3,
    min_prominence_log: float = 0.3,
    min_distance_bins: int = 5,
    fmin_hz: float = 0.0,
    noise_floor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Find up to *n_peaks* spectral peaks in a single k‑bin.

    Works in **log₁₀(S)** space so that peaks at high |k| (where S is
    orders of magnitude weaker) are detected just as reliably as at k≈0.

    Parameters
    ----------
    min_prominence_log : float
        Minimum prominence in log₁₀(S) units (default: 0.3 ≈ factor-of-2
        above the local baseline).  A value of 1.0 would require the peak
        to be 10× above its neighbours.
    noise_floor : float
        Absolute floor below which the spectrum is considered pure noise
        (in linear S units).  Columns whose max < noise_floor return no peaks.

    Returns (f_peak_hz, amplitudes) sorted by amplitude descending.
    """
    keep = f_axis >= fmin_hz
    spec = spectrum[keep]
    f_sub = f_axis[keep]

    if spec.size < 3:
        return np.array([]), np.array([])

    col_max = float(spec.max())
    if col_max <= 0 or col_max < noise_floor:
        return np.array([]), np.array([])

    # Work in log scale — key insight for wide dynamic range data
    spec_log = np.log10(np.maximum(spec, 1e-30))

    try:
        from scipy.signal import find_peaks as _scipy_find_peaks

        idx, props = _scipy_find_peaks(
            spec_log,
            distance=min_distance_bins,
            prominence=min_prominence_log,
        )
    except ImportError:
        # Fallback: simple local-max scan in log space
        idx = []
        for i in range(1, len(spec_log) - 1):
            if spec_log[i] > spec_log[i - 1] and spec_log[i] > spec_log[i + 1]:
                # Rough prominence check: compare to min of neighbours
                local_base = min(
                    spec_log[max(0, i - 3) : i].min(),
                    spec_log[i + 1 : min(len(spec_log), i + 4)].min(),
                )
                if spec_log[i] - local_base >= min_prominence_log:
                    idx.append(i)
        idx = np.asarray(idx, dtype=int)

    if len(idx) == 0:
        return np.array([]), np.array([])

    # Sort by amplitude descending (in linear space), keep top n_peaks
    order = np.argsort(spec[idx])[::-1][:n_peaks]
    idx = idx[order]

    return f_sub[idx], spec[idx]


# ──────────────────────────────────────────────────────────────────────
# Hungarian linking with amplitude‑weighted cost
# ──────────────────────────────────────────────────────────────────────

_UNLINKED_COST = 1e18


def _link_peaks(
    prev_f: np.ndarray,
    prev_amp: np.ndarray,
    curr_f: np.ndarray,
    curr_amp: np.ndarray,
    max_df_hz: float,
    amp_weight: float = 0.3,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Link peaks between two adjacent k‑bins.

    Cost = |Δf| / max_df - amp_weight * (amp_curr / amp_max)

    The amplitude term biases the assignment toward linking
    to stronger peaks, reducing jumps to noise.
    """
    n_prev = len(prev_f)
    n_curr = len(curr_f)

    if n_prev == 0:
        return [], [], list(range(n_curr))
    if n_curr == 0:
        return [], list(range(n_prev)), []

    # Normalise amplitudes for the weighting term
    all_amp = np.concatenate([prev_amp, curr_amp])
    amp_max = float(all_amp.max()) if len(all_amp) > 0 else 1.0

    cost = np.full((n_prev, n_curr), _UNLINKED_COST)
    for i in range(n_prev):
        for j in range(n_curr):
            df = abs(prev_f[i] - curr_f[j])
            if df <= max_df_hz:
                # Frequency distance (normalised) + penalty for weak peaks
                freq_cost = df / (max_df_hz + 1e-30)
                # Bonus for strong current peak (0..1)
                amp_bonus = amp_weight * (curr_amp[j] / (amp_max + 1e-30))
                cost[i, j] = freq_cost - amp_bonus

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
    except ImportError:
        row_ind, col_ind = _greedy_assign(cost)

    matches = []
    matched_prev: set = set()
    matched_curr: set = set()

    for r, c in zip(row_ind, col_ind, strict=False):
        if cost[r, c] < _UNLINKED_COST:
            matches.append((int(r), int(c)))
            matched_prev.add(int(r))
            matched_curr.add(int(c))

    unmatched_prev = [i for i in range(n_prev) if i not in matched_prev]
    unmatched_curr = [j for j in range(n_curr) if j not in matched_curr]

    return matches, unmatched_prev, unmatched_curr


def _greedy_assign(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
# Branch quality scoring
# ──────────────────────────────────────────────────────────────────────


def _branch_quality_metrics(
    k_arr: np.ndarray,
    f_arr: np.ndarray,
    amp_arr: np.ndarray,
    *,
    reference_k_axis: np.ndarray | None = None,
    noise_floor: float = 0.0,
    gap_count: int = 0,
) -> dict[str, float]:
    """Return normalized branch quality metrics.

    Components (all normalised to ~[0, 1]):
    - coverage: fraction of k-range covered (long branches score higher)
    - smoothness: 1 / (1 + relative variance of df/dk)
    - amplitude: mean amplitude relative to max
    - snr: bounded log-scaled peak/noise ratio
    - gaps: number of missing k-columns during linking
    - confidence: weighted aggregate used for ranking/filtering
    """
    n = len(k_arr)
    if n < 3:
        return {
            "coverage": 0.0,
            "smoothness": 0.0,
            "amplitude": 0.0,
            "length": 0.0,
            "snr": 0.0,
            "gaps": float(gap_count),
            "confidence": 0.0,
        }

    # Coverage: branch span relative to the searched k-window.
    dk_branch = float(k_arr.max() - k_arr.min())
    if reference_k_axis is not None and len(reference_k_axis) >= 2:
        ref = np.asarray(reference_k_axis, dtype=float)
        dk_total = float(np.nanmax(ref) - np.nanmin(ref))
    else:
        dk_total = dk_branch
    coverage = 1.0 if dk_total <= 0 else float(np.clip(dk_branch / dk_total, 0.0, 1.0))

    # Smoothness is based on the physical slope.  Using only diff(f) would
    # penalise a smooth branch merely because some k-columns were skipped.
    dk = np.diff(k_arr)
    if np.any(dk == 0):
        smoothness = 0.0
    else:
        slopes = np.diff(f_arr) / dk
        slope_scale = float(np.mean(np.abs(slopes)))
        span_scale = float(np.ptp(f_arr)) / max(float(np.ptp(k_arr)), 1e-30)
        roughness = float(np.std(slopes)) / max(slope_scale, span_scale, 1e-30)
        smoothness = 1.0 / (1.0 + 10.0 * roughness)

    # Amplitude: mean relative to max peak
    amp_max = float(amp_arr.max()) + 1e-30
    amp_score = float(amp_arr.mean()) / amp_max

    # Length bonus: log(n) to reward longer branches
    length_score = float(np.clip(math.log10(max(n, 1)) / 3.0, 0.0, 1.0))

    if noise_floor > 0:
        snr_linear = float(amp_arr.mean()) / float(noise_floor)
        snr_score = float(np.clip(math.log10(max(snr_linear, 1.0)) / 3.0, 0.0, 1.0))
    else:
        snr_score = 1.0 if float(amp_arr.max()) > 0 else 0.0

    gap_penalty = 1.0 / (1.0 + max(0, gap_count))
    confidence = float(
        0.25 * length_score
        + 0.25 * smoothness
        + 0.15 * amp_score
        + 0.20 * coverage
        + 0.15 * snr_score
    )
    confidence *= gap_penalty

    return {
        "coverage": float(coverage),
        "smoothness": float(smoothness),
        "amplitude": float(amp_score),
        "length": float(length_score),
        "snr": float(snr_score),
        "gaps": float(gap_count),
        "confidence": confidence,
    }


def _branch_quality(
    k_arr: np.ndarray,
    f_arr: np.ndarray,
    amp_arr: np.ndarray,
    *,
    reference_k_axis: np.ndarray | None = None,
) -> float:
    """Score a branch: higher = better. Used for final filtering."""
    return _branch_quality_metrics(
        k_arr,
        f_arr,
        amp_arr,
        reference_k_axis=reference_k_axis,
    )["confidence"]


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TrackedBranch:
    """A single branch: parallel arrays of (k, f, amplitude)."""

    k: np.ndarray
    f_hz: np.ndarray
    amplitude: np.ndarray
    branch_id: int = 0
    quality: float = 0.0
    quality_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.k = np.asarray(self.k, dtype=float)
        self.f_hz = np.asarray(self.f_hz, dtype=float)
        self.amplitude = np.asarray(self.amplitude, dtype=float)
        if any(value.ndim != 1 for value in (self.k, self.f_hz, self.amplitude)):
            raise ValueError("Tracked branch arrays must be one-dimensional")
        if not (self.k.size == self.f_hz.size == self.amplitude.size):
            raise ValueError("Tracked branch arrays must have matching lengths")
        if self.k.size == 0:
            raise ValueError("Tracked branch must contain at least one point")
        if not all(
            np.all(np.isfinite(value)) for value in (self.k, self.f_hz, self.amplitude)
        ):
            raise ValueError("Tracked branch arrays must contain finite values")
        if self.k.size > 1 and not np.all(np.diff(self.k) > 0):
            raise ValueError("Tracked branch k coordinates must be strictly increasing")
        self.quality = float(self.quality)
        if not np.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            raise ValueError("Tracked branch quality must be in [0, 1]")

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
        All detected branches, sorted by quality (best first).
    result : DispersionResult1D
        Back-reference to the dispersion data.
    """

    branches: list[TrackedBranch]
    result: DispersionResult1D
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.branches)

    def __getitem__(self, idx: int) -> TrackedBranch:
        return self.branches[idx]

    def __iter__(self):
        return iter(self.branches)

    @property
    def plot(self) -> BranchesPlotAccessor:
        return BranchesPlotAccessor(self)

    def __repr__(self) -> str:
        lines = [f"BranchesResult({len(self.branches)} branches):"]
        for br in self.branches:
            f_min = float(br.f_hz.min()) / 1e9
            f_max = float(br.f_hz.max()) / 1e9
            lines.append(
                f"  branch {br.branch_id}: "
                f"{len(br)} pts, f=[{f_min:.3f}..{f_max:.3f}] GHz, "
                f"quality={br.quality:.3f}"
            )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows = []
        for br in self.branches:
            f_min = float(br.f_hz.min()) / 1e9
            f_max = float(br.f_hz.max()) / 1e9
            k_lo = float(br.k.min()) / 1e6
            k_hi = float(br.k.max()) / 1e6
            q_bar = "█" * int(br.quality * 10) + "░" * (10 - int(br.quality * 10))
            rows.append(
                f"<tr>"
                f"<td style='padding:3px 10px;color:#93c5fd;font-weight:700;'>{br.branch_id}</td>"
                f"<td style='padding:3px 10px;'>{len(br)} pts</td>"
                f"<td style='padding:3px 10px;'>{f_min:.3f} – {f_max:.3f} GHz</td>"
                f"<td style='padding:3px 10px;'>{k_lo:.2f} – {k_hi:.2f} rad/μm</td>"
                f"<td style='padding:3px 10px;font-family:monospace;font-size:.75em;'>"
                f"<span style='color:#22c55e;'>{q_bar}</span> {br.quality:.2f}</td>"
                f"</tr>"
            )
        tbody = "".join(rows)
        return (
            "<div style='font-family:sans-serif;border:2px solid #1e3a5f;"
            "border-left:4px solid #a78bfa;border-radius:10px;padding:14px;"
            "margin:6px 0;background:linear-gradient(135deg,#0f172a,#0c1a35);"
            "color:#e2e8f0;max-width:700px;'>"
            "<div style='font-weight:700;font-size:1.0em;color:#f1f5f9;"
            "margin-bottom:8px;'>🌊 BranchesResult</div>"
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<tr style='border-bottom:1px solid #334155;'>"
            f"<th style='text-align:left;padding:3px 10px;color:#94a3b8;font-size:.8em;'>ID</th>"
            f"<th style='text-align:left;padding:3px 10px;color:#94a3b8;font-size:.8em;'>Points</th>"
            f"<th style='text-align:left;padding:3px 10px;color:#94a3b8;font-size:.8em;'>f range</th>"
            f"<th style='text-align:left;padding:3px 10px;color:#94a3b8;font-size:.8em;'>k range</th>"
            f"<th style='text-align:left;padding:3px 10px;color:#94a3b8;font-size:.8em;'>Quality</th>"
            f"</tr>{tbody}</table>"
            "<div style='margin-top:6px;font-size:.78em;color:#64748b;'>"
            "Use <code>.plot()</code> or <code>.plot.overlay(ax)</code> to visualize.</div>"
            "</div>"
        )


# ──────────────────────────────────────────────────────────────────────
# Main algorithm
# ──────────────────────────────────────────────────────────────────────


def find_branches(
    result: DispersionResult1D,
    *,
    n_branches: int = 3,
    side: str = "both",
    min_prominence_log: float = 0.3,
    min_peak_distance: int = 5,
    max_df_ghz: float = 0.5,
    min_branch_length: int = 20,
    noise_floor_percentile: float = 5.0,
    min_quality: float = 0.10,
    smooth_sigma: float | None = 3.0,
    fmin_hz: float | str | None = "auto",
    k_min_rad_um: float = 0.0,
    k_max_rad_um: float | None = None,
    analysis_source: str = "raw",
    positive_frequencies: bool = True,
) -> BranchesResult:
    """Detect multiple dispersion branches via Hungarian peak linking.

    Algorithm
    ---------
    1. Compute noise floor from the *noise_floor_percentile* of S(k,f).
       k-bins whose max < noise_floor are skipped.
    2. For each k-bin, detect up to *n_branches* spectral peaks in
       **log₁₀(S)** space (handles wide dynamic range naturally).
    3. Walk along k (left→right) and link peaks between adjacent bins
       using the Hungarian algorithm with amplitude-weighted cost.
    4. When a peak has no match, a new branch is born.
    5. When a branch loses its match for too many consecutive k-bins,
       it is terminated.
    6. Optionally smooth each branch with a Gaussian filter.
    7. Discard branches shorter than *min_branch_length*.
    8. Score each branch by quality (smoothness + amplitude + coverage)
       and discard those below *min_quality*.

    Parameters
    ----------
    result : DispersionResult1D
        Dispersion data.
    n_branches : int
        Max peaks to detect per k-bin (default: 3).
    side : ``"positive"`` | ``"negative"`` | ``"both"``
        Which k-half to search.
    min_prominence_log : float
        Minimum peak prominence in log₁₀(S) units (default: 0.3).
        0.3 ≈ factor-of-2 above local baseline;  1.0 = 10× above.
    min_peak_distance : int
        Min frequency bins between peaks (default: 5).
    max_df_ghz : float
        Max allowed frequency jump between adjacent k-bins [GHz] (default: 0.5).
    min_branch_length : int
        Discard branches shorter than this many k-bins (default: 20).
    noise_floor_percentile : float
        Percentile of S used as the absolute noise floor (default: 5).
        k-bins with max(S) below this level are skipped entirely.
    min_quality : float
        Discard branches with quality score below this (default: 0.10).
    smooth_sigma : float or None
        Gaussian smoothing sigma (in k-bins) on final branches (default: 3.0).
    fmin_hz : float, ``"auto"``, or None
        Min frequency cutoff.
    k_min_rad_um, k_max_rad_um : float
        k search window [rad/μm].
    positive_frequencies : bool
        Restrict branch tracking to f >= 0 by default. Set False with an
        explicit ``fmin_hz`` policy to analyze full signed-frequency spectra.

    Returns
    -------
    BranchesResult
    """
    if (
        isinstance(n_branches, (bool, np.bool_))
        or int(n_branches) != n_branches
        or int(n_branches) < 1
    ):
        raise ValueError("n_branches must be a positive integer")
    if side not in {"positive", "negative", "both"}:
        raise ValueError("side must be 'positive', 'negative', or 'both'")
    prominence = float(min_prominence_log)
    if not np.isfinite(prominence) or prominence < 0:
        raise ValueError("min_prominence_log must be finite and non-negative")
    if (
        isinstance(min_peak_distance, (bool, np.bool_))
        or int(min_peak_distance) != min_peak_distance
        or int(min_peak_distance) < 1
    ):
        raise ValueError("min_peak_distance must be a positive integer")
    max_df_value = float(max_df_ghz)
    if not np.isfinite(max_df_value) or max_df_value <= 0:
        raise ValueError("max_df_ghz must be finite and positive")
    if (
        isinstance(min_branch_length, (bool, np.bool_))
        or int(min_branch_length) != min_branch_length
        or int(min_branch_length) < 1
    ):
        raise ValueError("min_branch_length must be a positive integer")
    noise_percentile = float(noise_floor_percentile)
    if not np.isfinite(noise_percentile) or not 0 <= noise_percentile <= 100:
        raise ValueError("noise_floor_percentile must be in [0, 100]")
    quality_threshold = float(min_quality)
    if not np.isfinite(quality_threshold) or not 0 <= quality_threshold <= 1:
        raise ValueError("min_quality must be in [0, 1]")
    smooth_value = 0.0 if smooth_sigma is None else float(smooth_sigma)
    if not np.isfinite(smooth_value) or smooth_value < 0:
        raise ValueError("smooth_sigma must be finite and non-negative")
    k_min_value = float(k_min_rad_um)
    if not np.isfinite(k_min_value) or k_min_value < 0:
        raise ValueError("k_min_rad_um must be finite and non-negative")
    if k_max_rad_um is None:
        k_max_value = np.inf
    else:
        k_max_value = float(k_max_rad_um)
        if not np.isfinite(k_max_value) or k_max_value < k_min_value:
            raise ValueError(
                "k_max_rad_um must be finite and not smaller than k_min_rad_um"
            )

    if hasattr(result, "frequency_view"):
        S, k_axis, f_axis = result.frequency_view(
            positive_frequencies=positive_frequencies,
            analysis_source=analysis_source,
        )
    elif hasattr(result, "spectrum_for"):
        S = result.spectrum_for(analysis_source)
        k_axis = result.k_axis
        f_axis = result.f_axis
        if positive_frequencies:
            pos_f = f_axis >= 0
            S = S[:, pos_f]
            f_axis = f_axis[pos_f]
    else:
        S = result.S
        k_axis = result.k_axis
        f_axis = result.f_axis
        if positive_frequencies:
            pos_f = f_axis >= 0
            S = S[:, pos_f]
            f_axis = f_axis[pos_f]

    f_search = np.asarray(f_axis, dtype=float)
    S_search = np.asarray(S, dtype=float)
    if S_search.ndim != 2 or f_search.ndim != 1:
        raise ValueError("Branch tracking expects S(Nk, Nf) and a 1D f-axis")
    if f_search.size == 0 or S_search.shape[1] == 0:
        raise ValueError("No frequencies available for branch tracking")
    if S_search.shape != (len(k_axis), f_search.size):
        raise ValueError("Spectrum shape must match k and frequency axes")
    if not np.all(np.isfinite(S_search)) or not np.all(np.isfinite(f_search)):
        raise ValueError("Branch tracking requires finite spectrum and frequencies")
    if np.any(S_search < 0):
        raise ValueError("Branch tracking requires non-negative spectral power")
    frequency_order = np.argsort(f_search, kind="stable")
    f_search = f_search[frequency_order]
    S_search = S_search[:, frequency_order]
    if f_search.size > 1 and np.any(np.diff(f_search) <= 0):
        raise ValueError("Frequency axis must contain unique values")

    # fmin cutoff
    if fmin_hz == "auto":
        fmin_cutoff = (
            0.05 * float(f_search.max())
            if positive_frequencies
            else float(f_search.min())
        )
    elif fmin_hz is None:
        fmin_cutoff = 0.0 if positive_frequencies else float(f_search.min())
    else:
        try:
            fmin_cutoff = float(fmin_hz)
        except (TypeError, ValueError) as exc:
            raise ValueError("fmin_hz must be 'auto', None, or a number") from exc
        if not np.isfinite(fmin_cutoff):
            raise ValueError("fmin_hz must be finite")
        if positive_frequencies and fmin_cutoff < 0:
            raise ValueError("fmin_hz cannot be negative for positive-frequency search")

    # k-side mask
    k_min_rm = k_min_value * 1e6
    k_max_rm = k_max_value * 1e6

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

    max_df_hz = max_df_value * 1e9

    # ── Noise floor from percentile ──
    fmin_mask = f_search >= fmin_cutoff
    if not np.any(fmin_mask):
        raise ValueError("fmin_hz excludes every available frequency bin")
    S_for_snr = S_search[:, fmin_mask]
    positive_snr = S_for_snr[S_for_snr > 0]
    if positive_snr.size > 0:
        noise_floor = float(np.percentile(positive_snr, noise_percentile))
    else:
        raise ValueError("Dispersion spectrum has no positive spectral power")

    # Per-k noise gating
    row_max = (
        S_for_snr[k_idx].max(axis=1) if S_for_snr.shape[1] > 0 else np.zeros(len(k_idx))
    )
    snr_pass = row_max > noise_floor

    logger.info(
        "Branch search: %d k-bins, %d pass noise gate (floor=%.2e = P%.0f)",
        len(k_idx),
        int(snr_pass.sum()),
        noise_floor,
        noise_floor_percentile,
    )

    # ── Phase 1: per-column peak detection (log-scale) ──
    peaks_per_col: list[tuple[np.ndarray, np.ndarray]] = []
    for col_i, ik in enumerate(k_idx):
        if not snr_pass[col_i]:
            peaks_per_col.append((np.array([]), np.array([])))
            continue
        fp, amp = _find_peaks_column(
            S_search[ik],
            f_search,
            n_peaks=int(n_branches),
            min_prominence_log=prominence,
            min_distance_bins=int(min_peak_distance),
            fmin_hz=fmin_cutoff,
            noise_floor=noise_floor,
        )
        peaks_per_col.append((fp, amp))

    # ── Phase 2: Hungarian linking ──
    active: dict[int, dict] = {}
    finished: list[dict] = []
    next_id = 0
    MAX_GAP = 8  # max consecutive misses before terminating a branch

    for col_i, ik in enumerate(k_idx):
        k_val = k_axis[ik]
        curr_f, curr_amp = peaks_per_col[col_i]

        if len(curr_f) == 0:
            # SNR-gated or no peaks: increment gap on all active branches
            for bid in active:
                active[bid]["gap"] += 1
                active[bid]["gap_count"] += 1
            # Terminate branches with too many gaps
            to_remove = [bid for bid, br in active.items() if br["gap"] > MAX_GAP]
            for bid in to_remove:
                finished.append(active.pop(bid))
            continue

        if len(active) == 0:
            for j in range(len(curr_f)):
                active[next_id] = {
                    "source_id": next_id,
                    "k": [k_val],
                    "f": [curr_f[j]],
                    "amp": [curr_amp[j]],
                    "last_f": curr_f[j],
                    "last_amp": curr_amp[j],
                    "gap": 0,
                    "gap_count": 0,
                }
                next_id += 1
            continue

        active_ids = list(active.keys())
        prev_f = np.array([active[bid]["last_f"] for bid in active_ids])
        prev_amp = np.array([active[bid]["last_amp"] for bid in active_ids])

        matches, unmatched_prev, unmatched_curr = _link_peaks(
            prev_f, prev_amp, curr_f, curr_amp, max_df_hz
        )

        matched_ids = set()
        for pi, ci in matches:
            bid = active_ids[pi]
            active[bid]["k"].append(k_val)
            active[bid]["f"].append(curr_f[ci])
            active[bid]["amp"].append(curr_amp[ci])
            active[bid]["last_f"] = curr_f[ci]
            active[bid]["last_amp"] = curr_amp[ci]
            active[bid]["gap"] = 0
            matched_ids.add(bid)

        for pi in unmatched_prev:
            bid = active_ids[pi]
            if bid not in matched_ids:
                active[bid]["gap"] += 1
                active[bid]["gap_count"] += 1

        to_remove = [bid for bid, br in active.items() if br["gap"] > MAX_GAP]
        for bid in to_remove:
            finished.append(active.pop(bid))

        for ci in unmatched_curr:
            active[next_id] = {
                "source_id": next_id,
                "k": [k_val],
                "f": [curr_f[ci]],
                "amp": [curr_amp[ci]],
                "last_f": curr_f[ci],
                "last_amp": curr_amp[ci],
                "gap": 0,
                "gap_count": 0,
            }
            next_id += 1

    for br in active.values():
        finished.append(br)

    # ── Phase 3: build, score, filter ──
    tracked: list[TrackedBranch] = []
    rejected: list[dict[str, Any]] = []
    for br_data in finished:
        n_pts = len(br_data["k"])
        if n_pts < int(min_branch_length):
            rejected.append(
                {
                    "source_id": br_data.get("source_id"),
                    "reason": "min_branch_length",
                    "points": n_pts,
                    "threshold": int(min_branch_length),
                }
            )
            continue

        k_arr = np.array(br_data["k"])
        f_arr = np.array(br_data["f"])
        amp_arr = np.array(br_data["amp"])

        metrics = _branch_quality_metrics(
            k_arr,
            f_arr,
            amp_arr,
            reference_k_axis=k_axis[k_idx],
            noise_floor=noise_floor,
            gap_count=int(br_data.get("gap_count", 0)),
        )
        quality = metrics["confidence"]
        if quality < quality_threshold:
            rejected.append(
                {
                    "source_id": br_data.get("source_id"),
                    "reason": "min_quality",
                    "points": n_pts,
                    "quality": quality,
                    "threshold": quality_threshold,
                    "metrics": metrics,
                }
            )
            continue

        # Optional smoothing
        if smooth_value > 0 and len(f_arr) > 5:
            window = min(7, len(f_arr) if len(f_arr) % 2 else len(f_arr) - 1)
            half = window // 2
            smoothed = np.empty_like(f_arr)
            for index in range(len(f_arr)):
                start = max(0, min(index - half, len(f_arr) - window))
                stop = start + window
                local_k = k_arr[start:stop] - k_arr[index]
                coefficients = np.polyfit(local_k, f_arr[start:stop], deg=2)
                estimate = float(np.polyval(coefficients, 0.0))
                smoothed[index] = np.clip(
                    estimate,
                    float(np.min(f_arr[start:stop])),
                    float(np.max(f_arr[start:stop])),
                )
            f_arr = smoothed

        tracked.append(
            TrackedBranch(
                k=k_arr,
                f_hz=f_arr,
                amplitude=amp_arr,
                branch_id=len(tracked),
                quality=quality,
                quality_metrics=metrics,
            )
        )

    # Sort by quality (best first)
    tracked.sort(key=lambda b: b.quality, reverse=True)
    for i, tracked_branch in enumerate(tracked):
        tracked_branch.branch_id = i

    logger.info(
        "Found %d branches (from %d candidates, min_length=%d, min_quality=%.2f)",
        len(tracked),
        len(finished),
        min_branch_length,
        min_quality,
    )

    return BranchesResult(branches=tracked, result=result, rejected=rejected)


# ──────────────────────────────────────────────────────────────────────
# Plot accessor
# ──────────────────────────────────────────────────────────────────────

_BRANCH_COLORS = [
    "#f43f5e",
    "#3b82f6",
    "#22c55e",
    "#eab308",
    "#a855f7",
    "#06b6d4",
    "#f97316",
    "#ec4899",
    "#14b8a6",
    "#8b5cf6",
]


def _branch_plot_values(
    branch: TrackedBranch,
    *,
    kscale: str,
    f_units: str,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Convert one tracked branch to explicitly validated display units."""
    if kscale not in {"rad_um", "rad_m", "rad", "cycles_m", "meter"}:
        raise ValueError(
            "kscale must be 'rad_um', 'rad_m'/'rad', or 'cycles_m'/'meter'"
        )
    if f_units not in {"GHz", "Hz"}:
        raise ValueError("f_units must be 'GHz' or 'Hz'")

    k_plot = branch.k.copy()
    if kscale == "rad_um":
        k_plot /= 1e6
        k_label = r"$k$ [rad/μm]"
    elif kscale in {"cycles_m", "meter"}:
        k_plot /= 2 * np.pi
        k_label = r"$k$ [m$^{-1}$]"
    else:
        k_label = r"$k$ [rad/m]"

    f_plot = branch.f_hz.copy()
    if f_units == "GHz":
        f_plot /= 1e9
        f_label = "f [GHz]"
    else:
        f_label = "f [Hz]"
    return k_plot, f_plot, k_label, f_label


class BranchesPlotAccessor:
    """Plotting namespace for :class:`BranchesResult`."""

    def __init__(self, branches_result: BranchesResult) -> None:
        self._br = branches_result

    def __call__(self, **kwargs) -> tuple[Figure, Axes]:
        return self.heatmap(**kwargs)

    def heatmap(
        self,
        ax: Axes | None = None,
        *,
        figsize: tuple[float, float] = (12, 8),
        dpi: int | None = None,
        cmap: str = "cmc.davos",
        kscale: str = "rad_um",
        f_units: str = "GHz",
        fmax: float | None = None,
        lognorm: bool = True,
        linewidth: float = 2.0,
        show_legend: bool = True,
        title: str | None = None,
        save: str | Any | bool | None = None,
    ) -> tuple[Figure, Axes]:
        """S(k,f) heatmap with all branches overlaid."""
        br = self._br

        fig, ax = br.result.plot.heatmap(
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            cmap=cmap,
            kscale=kscale,
            f_units=f_units,
            fmax=fmax,
            lognorm=lognorm,
            title=title,
        )

        self.overlay(
            cast(Any, ax),
            kscale=kscale,
            f_units=f_units,
            linewidth=linewidth,
            show_legend=show_legend,
        )

        if save not in (None, False):
            br.result.plot._save_fig(fig, save, br.result)

        return fig, cast(Any, ax)

    def overlay(
        self,
        ax: Axes,
        *,
        kscale: str = "rad_um",
        f_units: str = "GHz",
        linewidth: float = 2.0,
        colors: list[str] | None = None,
        show_legend: bool = True,
    ) -> None:
        """Overlay branch curves on existing axes."""
        colors = colors or _BRANCH_COLORS
        width = float(linewidth)
        if not np.isfinite(width) or width <= 0:
            raise ValueError("linewidth must be finite and positive")
        br = self._br

        for branch in br.branches:
            k_plot, f_plot, _, _ = _branch_plot_values(
                branch, kscale=kscale, f_units=f_units
            )

            color = colors[branch.branch_id % len(colors)]
            ax.plot(
                k_plot,
                f_plot,
                color=color,
                linewidth=width,
                alpha=0.9,
                label=f"branch {branch.branch_id} (q={branch.quality:.2f})",
            )

        if show_legend:
            ax.legend(fontsize=8, loc="upper right")

    def branches(
        self,
        ax: Axes | None = None,
        *,
        figsize: tuple[float, float] = (10, 5),
        dpi: int | None = None,
        kscale: str = "rad_um",
        f_units: str = "GHz",
        title: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot only the extracted branches (no heatmap background)."""
        import matplotlib.pyplot as plt

        br = self._br
        colors = _BRANCH_COLORS

        if ax is None:
            fig, ax = cast(Any, plt.subplots)(
                figsize=figsize, **({} if dpi is None else {"dpi": dpi})
            )
        else:
            fig = ax.get_figure()

        for branch in br.branches:
            k_plot, f_plot, k_label, f_label = _branch_plot_values(
                branch, kscale=kscale, f_units=f_units
            )

            color = colors[branch.branch_id % len(colors)]
            ax.plot(
                k_plot,
                f_plot,
                color=color,
                linewidth=2.0,
                marker=".",
                markersize=3,
                alpha=0.8,
                label=f"branch {branch.branch_id} ({len(branch)} pts, q={branch.quality:.2f})",
            )

        if not br.branches:
            # Validate units and establish labels even for an empty result.
            placeholder = TrackedBranch(
                k=np.array([0.0]),
                f_hz=np.array([0.0]),
                amplitude=np.array([0.0]),
            )
            _, _, k_label, f_label = _branch_plot_values(
                placeholder, kscale=kscale, f_units=f_units
            )
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

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "BranchesPlotAccessor",
            [
                (
                    ".heatmap(cmap='cmc.davos', lognorm=True)",
                    "S(k,f) heatmap with all branches overlaid",
                    "kscale, f_units, fmax, linewidth, show_legend, save.",
                ),
                (
                    ".overlay(ax, kscale='rad_um')",
                    "Overlay branch curves on existing axes",
                    "f_units, linewidth, colors, show_legend.",
                ),
                (
                    ".branches(kscale='rad_um')",
                    "Plot only extracted branches (no heatmap)",
                    "f_units, dpi, title.",
                ),
            ],
        )
