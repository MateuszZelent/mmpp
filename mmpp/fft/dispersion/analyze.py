"""DispersionAnalyzeAccessor – analytical tools on DispersionResult1D.

Usage::

    result = job[0].fft.dispersion.filters(...).compute_1d(axis='x')
    lowest = result.analyze.find_lowest_possible_frequency()
    print(lowest)                    # → LowestFrequencyResult
    lowest.plot.heatmap()            # S(k,f) with marker at f_min
    lowest.plot.branch()             # f_peak(k) vs k curve
    lowest.plot()                    # alias for heatmap
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mmpp._repr_helpers import (
    NODE_COLOR_ANALYSIS,
    NODE_COLOR_COMPUTE,
    NODE_COLOR_PLOT,
    NODE_COLOR_UTIL,
    accessors_section_html,
    api_help_html,
    examples_section_html,
    metrics_section_html,
    node_card_html,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..models import DispersionResult1D
    from ._branch_linker import BranchesResult


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass
class LowestFrequencyResult:
    """Result of :meth:`DispersionAnalyzeAccessor.find_lowest_possible_frequency`.

    For spin waves with negative group velocity the minimum frequency on the
    dispersion branch does NOT occur at k=0 but at some finite k*.  This
    object captures that minimum and all associated quantities.

    Attributes
    ----------
    f_min_hz : float
        Minimum frequency on the branch [Hz].
    f_min_ghz : float
        Same in GHz.
    k_at_f_min : float
        Wave-vector at the frequency minimum [rad/m].
    k_at_f_min_um : float
        Wave-vector at the frequency minimum [rad/μm].
    f_at_k0_hz : float
        Frequency at k≈0 [Hz] – the uniform-mode (FMR) frequency.
    f_at_k0_ghz : float
        Same in GHz.
    group_velocity_at_min : float
        Estimated group velocity dω/dk = 2π·df/dk at k* [m/s].
    branch_f : np.ndarray
        f_peak(k) in Hz for each k on the axis.
    branch_k : np.ndarray
        Corresponding k values [rad/m].
    result : DispersionResult1D
        Back-reference to original dispersion data.
    side : str
        ``"positive"`` or ``"both"`` – which k-half was searched.
    """

    f_min_hz: float
    f_min_ghz: float
    k_at_f_min: float
    k_at_f_min_um: float
    f_at_k0_hz: float
    f_at_k0_ghz: float
    group_velocity_at_min: float
    branch_f: np.ndarray
    branch_k: np.ndarray
    result: DispersionResult1D
    side: str = "positive"

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    @property
    def plot(self) -> LowestFrequencyPlotAccessor:
        """Plotting namespace for this result."""
        return LowestFrequencyPlotAccessor(self)

    def __call__(self, **kwargs):
        """Shortcut: call on the result object renders the heatmap."""
        return self.plot.heatmap(**kwargs)

    def __repr__(self) -> str:
        return (
            f"LowestFrequencyResult(\n"
            f"  f_min     = {self.f_min_ghz:.4f} GHz  at  k* = {self.k_at_f_min_um:.3f} rad/μm\n"
            f"  f(k=0)   = {self.f_at_k0_ghz:.4f} GHz\n"
            f"  vg(k*)   = {self.group_velocity_at_min / 1e3:.2f} km/s\n"
            f"  Δf       = {(self.f_at_k0_ghz - self.f_min_ghz) * 1e3:.2f} MHz  "
            f"(k=0 is {'higher' if self.f_at_k0_ghz > self.f_min_ghz else 'lower'} than f_min)\n"
            f")"
        )

    def _repr_html_(self) -> str:
        delta_mhz = (self.f_at_k0_ghz - self.f_min_ghz) * 1e3

        if abs(delta_mhz) < 10:
            interpretation = (
                "Flat branch near k=0 — likely forward-volume or DEI geometry"
            )
            badge_color = "#a3e635"
        elif delta_mhz > 0:
            interpretation = (
                "Backward-volume: minimum at k>0 (negative group velocity region)"
            )
            badge_color = "#fb923c"
        else:
            interpretation = "Forward-volume: f increases with k from k=0"
            badge_color = "#34d399"
        api = api_help_html(
            self,
            title="Lowest frequency result API help",
            prefix="lowest",
            properties=[("plot", "Plotting accessor for heatmap and branch views")],
            methods=[],
            subtitle="Derived result object describing the minimum reachable frequency on the tracked dispersion branch.",
            chrome=False,
        )
        return node_card_html(
            "LowestFrequencyResult",
            icon="📉",
            subtitle=interpretation,
            badge=(self.side, badge_color),
            sections=[
                metrics_section_html(
                    [
                        ("f_min", f"{self.f_min_ghz:.4f} GHz", NODE_COLOR_COMPUTE),
                        (
                            "k* (at f_min)",
                            f"{self.k_at_f_min_um:.4f} rad/μm",
                            NODE_COLOR_ANALYSIS,
                        ),
                        ("f(k≈0)", f"{self.f_at_k0_ghz:.4f} GHz", NODE_COLOR_PLOT),
                        (
                            "v_g(k*)",
                            f"{self.group_velocity_at_min / 1e3:.3f} km/s",
                            NODE_COLOR_UTIL,
                        ),
                        ("Δf = f(k=0)-f_min", f"{delta_mhz:.2f} MHz", badge_color),
                        ("branch points", str(len(self.branch_k)), NODE_COLOR_UTIL),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Plot:",
                            [
                                (".plot()", NODE_COLOR_COMPUTE),
                                (".plot.heatmap(lognorm=True)", NODE_COLOR_ANALYSIS),
                                (".plot.branch(kscale='rad_um')", NODE_COLOR_PLOT),
                            ],
                        )
                    ]
                ),
                examples_section_html(
                    "\n".join(
                        [
                            "lowest = result.analyze.find_lowest_possible_frequency()",
                            "lowest.plot.heatmap(lognorm=True)",
                            "lowest.plot.branch(kscale='rad_um')",
                        ]
                    )
                ),
            ],
            api=api,
            uid="lowest-frequency-result",
        )


# ---------------------------------------------------------------------------
# Plot accessor for LowestFrequencyResult
# ---------------------------------------------------------------------------


class LowestFrequencyPlotAccessor:
    """Plotting namespace for :class:`LowestFrequencyResult`.

    Accessed via ``lowest.plot``.

    Methods
    -------
    heatmap(...)
        S(k,f) heatmap with a marker at the frequency minimum.
    branch(...)
        f_peak(k) vs k line plot with minimum marker.
    """

    def __init__(self, lowest: LowestFrequencyResult) -> None:
        self._lowest = lowest

    # convenience __call__ → heatmap
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
        vmin: float | None = None,
        vmax: float | None = None,
        k_xlim: tuple[float, float] | None = None,
        title: str | None = None,
        marker_color: str = "red",
        marker_size: int = 120,
        show_k0: bool = True,
        save: str | Path | bool | None = None,
    ) -> tuple[Figure, Axes]:
        """S(k,f) heatmap with the frequency minimum highlighted.

        Parameters
        ----------
        marker_color : str
            Color of the scatter marker at (k*, f_min).
        show_k0 : bool
            Also mark the k≈0 (FMR/uniform mode) point.
        """
        lowest = self._lowest
        if kscale not in {"rad_um", "rad_m", "rad", "cycles_m", "meter"}:
            raise ValueError(
                "kscale must be 'rad_um', 'rad_m'/'rad', or 'cycles_m'/'meter'"
            )
        if f_units not in {"GHz", "Hz"}:
            raise ValueError("f_units must be 'GHz' or 'Hz'")

        # Delegate base heatmap to DispersionPlotAccessor
        fig, ax = lowest.result.plot.heatmap(
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            cmap=cmap,
            kscale=kscale,
            f_units=f_units,
            fmax=fmax,
            lognorm=lognorm,
            vmin=vmin,
            vmax=vmax,
            k_xlim=k_xlim,
            title=title,
        )

        if ax is None:
            raise RuntimeError("Heatmap plotting did not return an axes object")
        # Convert coordinates to plot units
        k_min_plot = lowest.k_at_f_min
        f_min_plot = lowest.f_min_hz
        k0_plot = 0.0
        f_k0_plot = lowest.f_at_k0_hz

        if kscale == "rad_um":
            k_min_plot /= 1e6
            k0_plot = 0.0
        elif kscale in {"meter", "cycles_m"}:
            k_min_plot /= 2 * np.pi

        if f_units == "GHz":
            f_min_plot /= 1e9
            f_k0_plot /= 1e9

        # Draw minimum marker
        ax.scatter(
            [k_min_plot],
            [f_min_plot],
            s=marker_size,
            color=marker_color,
            marker="*",
            zorder=10,
            label=f"f_min = {lowest.f_min_ghz:.3f} GHz @ k* = {lowest.k_at_f_min_um:.2f} rad/μm",
        )

        # Draw horizontal line at f_min
        ax.axhline(
            f_min_plot,
            color=marker_color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )

        if show_k0:
            ax.scatter(
                [k0_plot],
                [f_k0_plot],
                s=80,
                color="cyan",
                marker="D",
                zorder=9,
                label=f"f(k=0) = {lowest.f_at_k0_ghz:.3f} GHz",
            )

        ax.legend(fontsize=8, loc="upper right")

        try:
            fig.tight_layout()
        except Exception:
            pass

        if save not in (None, False):
            lowest.result.plot._save_fig(fig, save, lowest.result)

        return fig, ax

    def branch(
        self,
        ax: Axes | None = None,
        *,
        figsize: tuple[float, float] = (10, 5),
        dpi: int | None = None,
        kscale: str = "rad_um",
        f_units: str = "GHz",
        title: str | None = None,
        marker_color: str = "red",
        save: str | Path | bool | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot f_peak(k) vs k with the minimum highlighted.

        This shows the extracted dispersion branch (one frequency per k)
        that was used to find the minimum.
        """
        import matplotlib.pyplot as plt

        lowest = self._lowest
        if kscale not in {"rad_um", "rad_m", "rad", "cycles_m", "meter"}:
            raise ValueError(
                "kscale must be 'rad_um', 'rad_m'/'rad', or 'cycles_m'/'meter'"
            )
        if f_units not in {"GHz", "Hz"}:
            raise ValueError("f_units must be 'GHz' or 'Hz'")

        k_data = lowest.branch_k.copy()
        f_data = lowest.branch_f.copy()

        if kscale == "rad_um":
            k_data = k_data / 1e6
            k_label = r"$k$ [rad/μm]"
        elif kscale in {"meter", "cycles_m"}:
            k_data = k_data / (2 * np.pi)
            k_label = r"$k$ [m$^{-1}$]"
        else:
            k_label = r"$k$ [rad/m]"

        if f_units == "GHz":
            f_data = f_data / 1e9
            f_label = "f [GHz]"
            f_min_plot = lowest.f_min_ghz
            f_k0_plot = lowest.f_at_k0_ghz
        else:
            f_label = "f [Hz]"
            f_min_plot = lowest.f_min_hz
            f_k0_plot = lowest.f_at_k0_hz

        k_min_plot = (
            lowest.k_at_f_min / 1e6
            if kscale == "rad_um"
            else lowest.k_at_f_min / (2 * np.pi)
            if kscale in {"meter", "cycles_m"}
            else lowest.k_at_f_min
        )

        if ax is None:
            fig, ax = cast(Any, plt.subplots)(
                figsize=figsize, **({"dpi": dpi} if dpi else {})
            )
        else:
            fig = cast(Figure, ax.get_figure())
        assert ax is not None

        ax.plot(k_data, f_data, linewidth=1.8, color="#60a5fa", label="f_peak(k)")

        # Min marker
        ax.scatter(
            [k_min_plot],
            [f_min_plot],
            s=150,
            color=marker_color,
            marker="*",
            zorder=10,
            label=f"f_min = {lowest.f_min_ghz:.3f} GHz",
        )
        ax.axhline(
            f_min_plot, color=marker_color, linestyle="--", linewidth=1.0, alpha=0.5
        )

        # k=0 marker
        ax.scatter(
            [0.0],
            [f_k0_plot],
            s=80,
            color="cyan",
            marker="D",
            zorder=9,
            label=f"f(k=0) = {lowest.f_at_k0_ghz:.3f} GHz",
        )

        ax.set_xlabel(k_label)
        ax.set_ylabel(f_label)
        ax.set_title(title or "Dispersion Branch: f_peak(k)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

        try:
            fig.tight_layout()
        except Exception:
            pass

        if save not in (None, False):
            lowest.result.plot._save_fig(fig, save, lowest.result)

        return fig, ax

    def __repr__(self) -> str:
        return "<LowestFrequencyPlotAccessor: .heatmap(), .branch()>"

    def _repr_html_(self) -> str:
        api = api_help_html(
            self,
            title="Lowest-frequency plot API help",
            prefix="lowest.plot",
            methods=["heatmap", "branch"],
            subtitle="Plotting helpers for the lowest-frequency analysis result.",
            chrome=False,
        )
        return node_card_html(
            "LowestFrequency Plot Accessor",
            icon="🖼️",
            subtitle="Heatmap and branch views focused on the minimum-frequency point of the tracked dispersion branch.",
            sections=[
                metrics_section_html(
                    [
                        ("owner", "LowestFrequencyResult.plot", NODE_COLOR_COMPUTE),
                        ("views", "heatmap / branch", NODE_COLOR_ANALYSIS),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Plot:",
                            [
                                (
                                    ".heatmap(lognorm=True, show_k0=True)",
                                    NODE_COLOR_COMPUTE,
                                ),
                                (".branch(kscale='rad_um')", NODE_COLOR_PLOT),
                            ],
                        )
                    ]
                ),
                examples_section_html(
                    "\n".join(
                        [
                            "lowest.plot.heatmap(lognorm=True, show_k0=True)",
                            "lowest.plot.branch(kscale='rad_um')",
                        ]
                    )
                ),
            ],
            api=api,
            uid="lowest-frequency-plot",
        )


# ---------------------------------------------------------------------------
# Analyze accessor  (attached to DispersionResult1D as .analyze)
# ---------------------------------------------------------------------------


class DispersionAnalyzeAccessor:
    """Analytical accessor on :class:`~mmpp.fft.dispersion.models.DispersionResult1D`.

    Accessed via ``result.analyze``.

    Methods
    -------
    find_lowest_possible_frequency(...)
        Locate the true minimum frequency on the dispersion branch, which
        for spin waves with negative group velocity lies at k ≠ 0.
    """

    def __init__(self, result: DispersionResult1D) -> None:
        self._result = result

    def find_lowest_possible_frequency(
        self,
        *,
        side: str = "positive",
        smooth_sigma: float | None = 2.0,
        min_snr: float = 0.05,
        k_min_rad_um: float = 0.0,
        k_max_rad_um: float | None = None,
        peak_method: str = "argmax",
        fmin_hz: float | str | None = "auto",
        analysis_source: str = "raw",
    ) -> LowestFrequencyResult:
        """Find the lowest frequency reachable on the spin-wave dispersion.

        For backward-volume and strongly hybridised spin-wave branches the
        minimum frequency is NOT at k = 0 but at some k* > 0.  This method:

        1. Extracts f_peak(k) – the spectral-peak frequency for every k bin.
        2. Optionally smooths the branch to suppress noise.
        3. Finds k* = argmin(f_peak) over the requested k-half.
        4. Estimates the group velocity at k* via ``np.gradient``.

        Parameters
        ----------
        side : ``"positive"`` | ``"negative"`` | ``"both"``
            Which half of the k-axis to search.  ``"positive"`` searches
            k > 0 only (default); ``"both"`` searches the full axis.
        smooth_sigma : float or None
            Gaussian smoothing width (in k-bins) applied to f_peak(k) before
            the minimum search.  Use *None* to skip smoothing.
        min_snr : float
            Fraction of global max-S used as minimum detectable signal.
            k-bins with ``S.max(axis=f) < min_snr * S.max()`` are excluded.
        k_min_rad_um, k_max_rad_um : float
            Search window in rad/μm.  Default: no restriction.
        peak_method : ``"argmax"`` | ``"centroid"``
            How to determine f_peak for each k bin.

            * ``"argmax"`` – index of maximum spectral density.
            * ``"centroid"`` – power-weighted centroid frequency.
        fmin_hz : float, ``"auto"``, or None
            Minimum frequency cutoff (Hz).  Frequencies below this value are
            excluded from the peak search to avoid DC / low-frequency artifacts.

            * ``"auto"`` (default) – skip the lowest 5 % of the positive
              frequency axis.  Equivalent to ``fmin_hz = 0.05 * f_axis.max()``.
            * ``None`` or ``0`` – no cutoff, use all positive frequencies.
            * Explicit ``float`` – use that value as the lower bound (Hz).
        analysis_source : ``"raw"`` | ``"display"``
            Spectrum used for quantitative extraction. Defaults to raw power;
            select ``"display"`` only when post-filtered values are intended
            to change the reported branch.

        Returns
        -------
        LowestFrequencyResult
        """
        result = self._result
        if side not in {"positive", "negative", "both"}:
            raise ValueError("side must be 'positive', 'negative', or 'both'")
        if peak_method not in {"argmax", "centroid"}:
            raise ValueError("peak_method must be 'argmax' or 'centroid'")
        snr_value = float(min_snr)
        if not np.isfinite(snr_value) or not 0.0 <= snr_value <= 1.0:
            raise ValueError("min_snr must be finite and in [0, 1]")
        k_min_value = float(k_min_rad_um)
        if not np.isfinite(k_min_value) or k_min_value < 0:
            raise ValueError("k_min_rad_um must be finite and non-negative")
        if k_max_rad_um is not None:
            k_max_value = float(k_max_rad_um)
            if not np.isfinite(k_max_value) or k_max_value < k_min_value:
                raise ValueError(
                    "k_max_rad_um must be finite and not smaller than k_min_rad_um"
                )
        else:
            k_max_value = np.inf
        if smooth_sigma is not None:
            smooth_value = float(smooth_sigma)
            if not np.isfinite(smooth_value) or smooth_value < 0:
                raise ValueError("smooth_sigma must be finite and non-negative")
        else:
            smooth_value = 0.0

        S = np.asarray(result.spectrum_for(analysis_source), dtype=float)
        if not np.all(np.isfinite(S)):
            raise ValueError("Selected dispersion spectrum contains non-finite values")
        if np.any(S < 0):
            raise ValueError(
                "Quantitative branch extraction requires non-negative power"
            )
        k_axis = result.k_axis  # rad/m
        f_axis = result.f_axis  # Hz

        # Restrict to positive frequencies. Prefer slice-based selection to
        # avoid materializing an (Nk, Nf_pos) copy via boolean indexing.
        if np.all(np.diff(f_axis) >= 0):
            f_selector: slice | np.ndarray = slice(
                int(np.searchsorted(f_axis, 0.0, side="left")), None
            )
            f_axis_pos = f_axis[f_selector]
        else:
            pos_idx = np.flatnonzero(f_axis >= 0)
            if pos_idx.size == 0:
                raise ValueError("Frequency axis does not contain non-negative values.")
            if int(pos_idx[-1] - pos_idx[0] + 1) == int(pos_idx.size):
                f_selector = slice(int(pos_idx[0]), int(pos_idx[-1]) + 1)
                f_axis_pos = f_axis[f_selector]
            else:
                # Fallback for non-monotonic/non-contiguous axes.
                f_selector = pos_idx
                f_axis_pos = f_axis[pos_idx]

        if f_axis_pos.size == 0:
            raise ValueError(
                "No non-negative frequencies available for branch extraction."
            )
        f_axis_monotonic = bool(np.all(np.diff(f_axis_pos) >= 0))

        # Apply fmin cutoff to avoid DC artifacts
        if fmin_hz == "auto":
            fmin_cutoff = 0.05 * float(f_axis_pos.max())
        elif fmin_hz is None:
            fmin_cutoff = 0.0
        else:
            try:
                fmin_cutoff = float(fmin_hz)
            except (TypeError, ValueError) as exc:
                raise ValueError("fmin_hz must be 'auto', None, or a number") from exc
            if not np.isfinite(fmin_cutoff) or fmin_cutoff < 0:
                raise ValueError("fmin_hz must be finite and non-negative")

        if fmin_cutoff > 0:
            if f_axis_monotonic:
                start_idx = int(np.searchsorted(f_axis_pos, fmin_cutoff, side="left"))
                if start_idx >= f_axis_pos.size:
                    raise ValueError(
                        f"fmin_hz={fmin_cutoff:.3e} exceeds available positive-frequency range "
                        f"[{float(f_axis_pos.min()):.3e}, {float(f_axis_pos.max()):.3e}] Hz.",
                    )
                f_axis_pos = f_axis_pos[start_idx:]
                if isinstance(f_selector, slice):
                    base_start = int(f_selector.start or 0)
                    f_selector = slice(
                        base_start + start_idx, f_selector.stop, f_selector.step
                    )
                else:
                    f_selector = f_selector[start_idx:]
            else:
                f_keep = f_axis_pos >= fmin_cutoff
                if not np.any(f_keep):
                    raise ValueError(
                        f"fmin_hz={fmin_cutoff:.3e} exceeds available positive-frequency values.",
                    )
                f_axis_pos = f_axis_pos[f_keep]
                if isinstance(f_selector, slice):
                    idx_full = np.arange(f_axis.shape[0], dtype=int)[f_selector]
                    f_selector = idx_full[f_keep]
                else:
                    f_selector = f_selector[f_keep]

        S_pos = S[:, f_selector]

        # Convert search window to rad/m
        k_min_rm = k_min_value * 1e6
        k_max_rm = k_max_value * 1e6

        # Build k-side mask
        if side == "positive":
            k_mask = (k_axis > k_min_rm) & (k_axis <= k_max_rm)
        elif side == "negative":
            k_mask = (k_axis < -k_min_rm) & (k_axis >= -k_max_rm)
        else:
            k_mask = (np.abs(k_axis) >= k_min_rm) & (np.abs(k_axis) <= k_max_rm)

        if not np.any(k_mask):
            raise ValueError(
                f"No k-bins found for side={side!r} with k_min={k_min_rad_um} "
                f"rad/μm.  Check k-axis range."
            )

        # SNR gate: exclude k-bins with very low total power
        global_max = float(S_pos.max())
        if global_max <= 0:
            raise ValueError("Dispersion spectrum has no positive spectral power")
        row_max = S_pos.max(axis=1)  # (Nk,)
        snr_mask = (row_max > 0) & (row_max >= snr_value * global_max)
        combined_mask = k_mask & snr_mask

        if not np.any(combined_mask):
            raise ValueError(
                "No k-bin in the requested window passes the min_snr threshold"
            )

        k_search = k_axis[combined_mask]
        S_search = S_pos[combined_mask, :]

        # Extract f_peak per k-bin
        if peak_method == "centroid":
            total_power = S_search.sum(axis=1, keepdims=True)
            f_peak_hz = (S_search * f_axis_pos[np.newaxis, :]).sum(
                axis=1
            ) / total_power[:, 0]
        else:  # argmax
            f_peak_idx = np.argmax(S_search, axis=1)
            f_peak_hz = f_axis_pos[f_peak_idx]

        # Smooth the branch
        if smooth_value > 0:
            try:
                from scipy.ndimage import gaussian_filter1d

                f_peak_hz = gaussian_filter1d(
                    f_peak_hz.astype(float), sigma=smooth_value
                )
            except ImportError:
                # Edge-padded box fallback with output length preserved.
                w = min(
                    f_peak_hz.size,
                    max(1, int(round(smooth_value * 2))),
                )
                if w % 2 == 0 and w > 1:
                    w -= 1
                kernel = np.ones(w) / w
                pad = w // 2
                f_peak_hz = np.convolve(
                    np.pad(f_peak_hz, pad, mode="edge"),
                    kernel,
                    mode="valid",
                )

        # Find minimum
        idx_min = int(np.argmin(f_peak_hz))
        k_star = float(k_search[idx_min])
        f_min = float(f_peak_hz[idx_min])

        # f at k≈0 (use full k-axis)
        idx_k0 = int(np.argmin(np.abs(k_axis)))
        k0_power = S_pos[idx_k0, :]
        k0_max = float(np.max(k0_power))
        if k0_max <= 0 or k0_max < snr_value * global_max:
            raise ValueError(
                "The k≈0 spectrum does not pass min_snr; lower min_snr if this "
                "weak uniform-mode estimate is intentional"
            )
        if peak_method == "centroid":
            f_k0 = float(np.sum(k0_power * f_axis_pos) / np.sum(k0_power))
        else:
            f_k0 = float(f_axis_pos[np.argmax(k0_power)])

        # Group velocity at k* via gradient on smoothed branch
        if k_search.size > 2:
            vg_arr = 2 * np.pi * np.gradient(f_peak_hz, k_search)
            vg_at_min = float(vg_arr[idx_min])
        else:
            vg_at_min = 0.0

        return LowestFrequencyResult(
            f_min_hz=f_min,
            f_min_ghz=f_min / 1e9,
            k_at_f_min=k_star,
            k_at_f_min_um=k_star / 1e6,
            f_at_k0_hz=f_k0,
            f_at_k0_ghz=f_k0 / 1e9,
            group_velocity_at_min=vg_at_min,
            branch_f=f_peak_hz,
            branch_k=k_search,
            result=result,
            side=side,
        )

    def find_branches(
        self,
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

        Peak detection works in **log₁₀(S)** space to handle the wide
        dynamic range of dispersion data (typically 5–7 orders of magnitude).
        Peaks are linked across k-bins using the Hungarian algorithm.

        Parameters
        ----------
        n_branches : int
            Maximum number of peaks to detect per k-bin.
        side : ``"positive"`` | ``"negative"`` | ``"both"``
            Which half of the k-axis to search.
        min_prominence_log : float
            Peak prominence in log₁₀(S) units (default: 0.3 ≈ factor-of-2).
        min_peak_distance : int
            Minimum frequency bins between detected peaks.
        max_df_ghz : float
            Maximum allowed frequency jump [GHz] between adjacent k-bins.
        min_branch_length : int
            Discard branches shorter than this many points.
        noise_floor_percentile : float
            Percentile of S used as noise floor (default: 5).
        min_quality : float
            Discard branches below this quality score (default: 0.10).
        smooth_sigma : float or None
            Gaussian smoothing sigma (k-bins) on final branches.
        fmin_hz : float, ``"auto"``, or None
            Lower frequency cutoff.
        k_min_rad_um, k_max_rad_um : float
            k-window [rad/μm].
        analysis_source : ``"raw"`` | ``"display"``
            Spectrum source used for branch extraction. Defaults to raw data so
            display filters do not change quantitative analysis unless asked.
        positive_frequencies : bool
            Restrict branch extraction to f >= 0. Set False for signed-frequency
            analysis when that is physically intended.

        Returns
        -------
        BranchesResult
            Object with ``.branches`` list and ``.plot`` accessor.

        Examples
        --------
        >>> br = result.analyze.find_branches(n_branches=3)
        >>> br.plot()                # heatmap + overlay
        >>> br.plot.branches()       # branches only
        >>> br.plot.overlay(ax)      # overlay on existing axes
        """
        from ._branch_linker import find_branches as _find_branches

        return _find_branches(
            self._result,
            n_branches=n_branches,
            side=side,
            min_prominence_log=min_prominence_log,
            min_peak_distance=min_peak_distance,
            max_df_ghz=max_df_ghz,
            min_branch_length=min_branch_length,
            noise_floor_percentile=noise_floor_percentile,
            min_quality=min_quality,
            smooth_sigma=smooth_sigma,
            fmin_hz=fmin_hz,
            k_min_rad_um=k_min_rad_um,
            k_max_rad_um=k_max_rad_um,
            analysis_source=analysis_source,
            positive_frequencies=positive_frequencies,
        )

    def scan(
        self,
        sources,
        param_values=None,
        param_label: str = "parameter",
        *,
        job_indices=None,
        param_attr: str | None = None,
        param_scale: float = 1.0,
        filters: dict | None = None,
        compute_kwargs: dict | None = None,
        find_kwargs: dict | None = None,
        slice_spec=None,
        verbose: bool = True,
        on_error: str = "warn",
    ):
        """Bulk scan of minimum frequency across multiple sources/jobs.

        Convenience wrapper around :func:`~mmpp.fft.dispersion.bulk.scan_minimum_frequency`.
        The current result's filters and compute settings are used as defaults
        (can be overridden via *filters* / *compute_kwargs*).

        Parameters
        ----------
        sources : mmpp.MMPP | list[ZarrJobResult] | list[DispersionResult1D]
            MMPP container, list of jobs, pre-computed results, or callables.
        param_values : sequence of float, optional
            One value per *selected* job.  ``None`` → read from *param_attr*
            or fall back to job indices.
        job_indices : sequence of int, optional
            Pick specific jobs from *sources* by index.
        param_attr : str, optional
            Attribute key on each job to use as the parameter value, e.g.
            ``param_attr="b"`` reads ``job.attrs["b"]``.
        param_scale : float
            Scale factor applied to the value read from *param_attr*
            (e.g. ``param_scale=1000`` converts T → mT).
        param_label : str
            Human-readable name for the scan parameter.
        filters, compute_kwargs, find_kwargs : dict, optional
            Override auto-detected settings from the current result.
        slice_spec
            Spatial slice applied to each job's magnetisation dataset.
        verbose : bool
            Print per-job progress.
        on_error : ``"warn"`` | ``"raise"`` | ``"skip"``
            How to handle per-job failures.

        Returns
        -------
        BulkMinimumFrequencyResult

        Examples
        --------
        >>> jobs = mmpp.MMPP("/path/to/sweep/", debug=False)
        >>> # Auto-read B field from attrs, convert T → mT:
        >>> bulk = result.analyze.scan(
        ...     jobs,
        ...     param_attr="b",
        ...     param_scale=1000,
        ...     param_label="B_ext [mT]",
        ...     filters=dict(remove_static=True),
        ...     slice_spec=(slice(800), Ellipsis, 2),
        ... )
        >>> bulk.plot.summary()
        """
        from .bulk import scan_minimum_frequency

        # Inherit config from current result where not overridden
        res = self._result
        _filters = filters or {}
        _find_kwargs = find_kwargs or {}
        _compute_kw = compute_kwargs or {"axis": getattr(res, "axis", "x")}

        return scan_minimum_frequency(
            sources,
            param_values=param_values,
            param_label=param_label,
            job_indices=job_indices,
            param_attr=param_attr,
            param_scale=param_scale,
            filters=_filters,
            compute_kwargs=_compute_kw,
            find_kwargs=_find_kwargs,
            slice_spec=slice_spec,
            verbose=verbose,
            on_error=on_error,
        )

    def __repr__(self) -> str:
        return "<DispersionAnalyzeAccessor: .find_lowest_possible_frequency(...)>"

    def _repr_html_(self) -> str:
        from html import escape as _esc

        HV = "onmouseover=\"this.style.background='#1e293b'\" onmouseout=\"this.style.background='transparent'\""

        methods = [
            (
                ".find_lowest_possible_frequency()",
                "→ LowestFrequencyResult",
                "Default: side='positive', smooth_sigma=2.0, peak_method='argmax'. "
                "Finds k* where f(k) is minimum on the dispersion branch.",
            ),
            (
                ".find_lowest_possible_frequency(side='both')",
                "Search full k-axis",
                "side='both' searches k<0 and k>0. Use for isotropic or symmetric systems.",
            ),
            (
                ".find_lowest_possible_frequency(smooth_sigma=None, peak_method='centroid')",
                "No smoothing, centroid peak",
                "smooth_sigma=None disables Gaussian smoothing. peak_method='centroid' uses power-weighted "
                "centroid frequency instead of argmax — more robust for broad peaks.",
            ),
            (
                ".find_lowest_possible_frequency(k_min_rad_um=0.5, k_max_rad_um=8.0)",
                "Restrict k search window",
                "k_min_rad_um / k_max_rad_um restrict the search range in rad/\u03bcm. Useful to exclude the k=0 region.",
            ),
            (
                ".find_lowest_possible_frequency(fmin_hz=2e9)",
                "Explicit fmin cutoff (Hz)",
                "fmin_hz='auto' (default) skips lowest 5% of spectrum. Set explicit Hz value, or None to disable.",
            ),
        ]
        rows = "".join(
            f"<tr {HV} title=\"{_esc(tip)}\" style='cursor:pointer;'>"
            f"<td style='padding:4px 10px;font-family:monospace;color:#34d399;font-size:.85em;'>{_esc(sig)}</td>"
            f"<td style='padding:4px 10px;color:#cbd5e1;font-size:.83em;'>{_esc(desc)}</td>"
            f"</tr>"
            for sig, desc, tip in methods
        )
        return (
            "<div style='font-family:-apple-system,sans-serif;border:2px solid #065f46;"
            "border-radius:10px;padding:12px;margin:6px 0;"
            "background:#0f172a;color:#e2e8f0;max-width:680px;'>"
            "<div style='font-weight:700;color:#34d399;margin-bottom:8px;'>"
            "DispersionAnalyzeAccessor"
            "<span style='font-size:.75em;color:#475569;font-weight:400;margin-left:8px;'>"
            "(hover rows for parameter details)</span></div>"
            f"<table style='width:100%;border-collapse:collapse;'>{rows}</table>"
            "<div style='margin-top:8px;font-size:.78em;color:#475569;'>"
            "Returns <code style='color:#6ee7b7;'>LowestFrequencyResult</code> with "
            "<code style='color:#6ee7b7;'>.plot.heatmap()</code> and "
            "<code style='color:#6ee7b7;'>.plot.branch()</code>"
            "</div></div>"
        )
