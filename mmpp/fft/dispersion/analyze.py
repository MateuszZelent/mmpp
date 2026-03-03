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
from typing import TYPE_CHECKING, Any, Optional, Union
from pathlib import Path

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ..models import DispersionResult1D


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
    result: "DispersionResult1D"
    side: str = "positive"

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    @property
    def plot(self) -> "LowestFrequencyPlotAccessor":
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
            f"  vg(k*)   = {self.group_velocity_at_min/1e3:.2f} km/s\n"
            f"  Δf       = {(self.f_at_k0_ghz - self.f_min_ghz)*1e3:.2f} MHz  "
            f"(k=0 is {'higher' if self.f_at_k0_ghz > self.f_min_ghz else 'lower'} than f_min)\n"
            f")"
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        HV = "onmouseover=\"this.style.background='#1e293b'\" onmouseout=\"this.style.background='transparent'\""
        delta_mhz = (self.f_at_k0_ghz - self.f_min_ghz) * 1e3
        delta_sign = "⬆" if delta_mhz > 0 else "⬇"

        if abs(delta_mhz) < 10:
            interpretation = "Flat branch near k=0 — likely forward-volume or DEI geometry"
            badge_color = "#a3e635"
        elif delta_mhz > 0:
            interpretation = "Backward-volume: minimum at k>0 (negative group velocity region)"
            badge_color = "#fb923c"
        else:
            interpretation = "Forward-volume: f increases with k from k=0"
            badge_color = "#34d399"

        stat_rows = [
            ("f_min", f"{self.f_min_ghz:.4f} GHz",
             "Minimum frequency on the branch",
             "True minimum — lowest frequency at which spin waves propagate in this geometry."),
            ("k* (at f_min)", f"{self.k_at_f_min_um:.4f} rad/μm",
             "Wave-vector where f is minimum",
             "For backward-volume SW, k* > 0. At this k, group velocity changes sign."),
            ("f(k≈0)", f"{self.f_at_k0_ghz:.4f} GHz",
             "Frequency at k≈0 (FMR / uniform mode)",
             "Spectral peak at the smallest k-bin — effectively the Kittel / FMR frequency."),
            ("v_g(k*)", f"{self.group_velocity_at_min/1e3:.3f} km/s",
             "Group velocity at the minimum",
             "dω/dk estimated via numpy.gradient. Should be ~0 at the true minimum."),
            (f"Δf = f(k=0)−f_min  {delta_sign}",
             f"{delta_mhz:.2f} MHz",
             "Frequency shift between k=0 and the minimum",
             interpretation),
            ("branch points", f"{len(self.branch_k)}",
             "k-bins used in the search",
             f"side={self.side!r} search over {len(self.branch_k)} k-bins after SNR gating."),
        ]
        stat_html = "".join(
            f"<tr {HV} title=\"{_esc(tip2)}\" style='cursor:help;'>"
            f"<td style='padding:3px 10px;font-family:monospace;color:#93c5fd;font-size:.86em;'>{_esc(k)}</td>"
            f"<td style='padding:3px 10px;color:#a5b4fc;font-weight:700;font-size:.9em;'>{_esc(v)}</td>"
            f"<td style='padding:3px 10px;color:#64748b;font-size:.8em;'>{_esc(desc)}</td>"
            f"</tr>"
            for k, v, desc, tip2 in stat_rows
        )

        plot_methods = [
            (".plot()", "Alias → .plot.heatmap()",
             "Shortcut: calling the result object directly renders the heatmap."),
            (".plot.heatmap(lognorm=True)",
             "S(k,f) with f_min marked",
             "S(k,f) heatmap with red star at (k*, f_min) and dashed line at f_min. Cyan diamond = f(k=0)."),
            (".plot.branch(kscale='rad_um')",
             "f_peak(k) curve with min marker",
             "Shows extracted branch f_peak(k) with minimum highlighted. Useful to verify k* identification."),
        ]
        plot_rows = "".join(
            f"<tr {HV} title=\"{_esc(tip2)}\" style='cursor:pointer;'>"
            f"<td style='padding:3px 10px;font-family:monospace;color:#f9a8d4;font-size:.85em;'>{_esc(sig)}</td>"
            f"<td style='padding:3px 10px;color:#cbd5e1;font-size:.83em;'>{_esc(desc)}</td>"
            f"</tr>"
            for sig, desc, tip2 in plot_methods
        )
        plot_section = (
            "<details style='margin-top:8px;'>"
            "<summary style='cursor:pointer;font-size:.83em;color:#f9a8d4;list-style:none;"
            "padding:3px 6px;background:#1e293b;border-radius:5px;' "
            "title='Expand to see plot methods'>"
            "▶ <code>.plot</code> — LowestFrequencyPlotAccessor</summary>"
            "<div style='margin-left:12px;margin-top:4px;'>"
            f"<table style='width:100%;border-collapse:collapse;'>{plot_rows}</table>"
            "</div></details>"
        )

        return (
            "<div style='font-family:-apple-system,BlinkMacSystemFont,sans-serif;"
            "border:2px solid #1e3a5f;border-left:4px solid "
            + badge_color
            + ";border-radius:10px;padding:14px;margin:6px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#0c1a35 100%);"
            "color:#e2e8f0;max-width:680px;'>"
            "<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>"
            "<span style='font-weight:700;font-size:1.0em;color:#f1f5f9;'>LowestFrequencyResult</span>"
            f"<span style='background:{badge_color}22;color:{badge_color};border-radius:4px;"
            f"padding:1px 8px;font-size:.72em;font-weight:600;'>{_esc(interpretation)}</span>"
            "</div>"
            f"<table style='width:100%;border-collapse:collapse;margin-bottom:4px;'>{stat_html}</table>"
            + plot_section
            + "</div>"
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
    def __call__(self, **kwargs) -> tuple["Figure", "Axes"]:
        return self.heatmap(**kwargs)

    def heatmap(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (12, 8),
        dpi: Optional[int] = None,
        cmap: str = "cmc.davos",
        kscale: str = "rad_um",
        f_units: str = "GHz",
        fmax: Optional[float] = None,
        lognorm: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        k_xlim: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        marker_color: str = "red",
        marker_size: int = 120,
        show_k0: bool = True,
        save: Union[str, "Path", bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """S(k,f) heatmap with the frequency minimum highlighted.

        Parameters
        ----------
        marker_color : str
            Color of the scatter marker at (k*, f_min).
        show_k0 : bool
            Also mark the k≈0 (FMR/uniform mode) point.
        """
        lowest = self._lowest

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

        # Convert coordinates to plot units
        k_min_plot = lowest.k_at_f_min
        f_min_plot = lowest.f_min_hz
        k0_plot = 0.0
        f_k0_plot = lowest.f_at_k0_hz

        if kscale == "rad_um":
            k_min_plot /= 1e6
            k0_plot = 0.0
        elif kscale == "meter":
            k_min_plot /= (2 * np.pi)

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
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (10, 5),
        dpi: Optional[int] = None,
        kscale: str = "rad_um",
        f_units: str = "GHz",
        title: Optional[str] = None,
        marker_color: str = "red",
        save: Union[str, "Path", bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot f_peak(k) vs k with the minimum highlighted.

        This shows the extracted dispersion branch (one frequency per k)
        that was used to find the minimum.
        """
        import matplotlib.pyplot as plt

        lowest = self._lowest

        k_data = lowest.branch_k.copy()
        f_data = lowest.branch_f.copy()

        if kscale == "rad_um":
            k_data = k_data / 1e6
            k_label = r"$k$ [rad/μm]"
        elif kscale == "meter":
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
            lowest.k_at_f_min / 1e6 if kscale == "rad_um"
            else lowest.k_at_f_min / (2 * np.pi) if kscale == "meter"
            else lowest.k_at_f_min
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

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
        ax.axhline(f_min_plot, color=marker_color, linestyle="--", linewidth=1.0, alpha=0.5)

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
        return "<LowestFrequencyPlotAccessor: .heatmap(...), .branch(...)>"


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

    def __init__(self, result: "DispersionResult1D") -> None:
        self._result = result

    def find_lowest_possible_frequency(
        self,
        *,
        side: str = "positive",
        smooth_sigma: Optional[float] = 2.0,
        min_snr: float = 0.05,
        k_min_rad_um: float = 0.0,
        k_max_rad_um: Optional[float] = None,
        peak_method: str = "argmax",
        fmin_hz: Optional[float] = "auto",
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

        Returns
        -------
        LowestFrequencyResult
        """
        result = self._result
        S = result.S          # (Nk, Nf)
        k_axis = result.k_axis  # rad/m
        f_axis = result.f_axis  # Hz

        # Restrict to positive frequencies
        pos_f = f_axis >= 0
        f_axis_pos = f_axis[pos_f]
        S_pos = S[:, pos_f]

        # Apply fmin cutoff to avoid DC artifacts
        if fmin_hz == "auto":
            fmin_cutoff = 0.05 * float(f_axis_pos.max())
        elif fmin_hz is not None and fmin_hz > 0:
            fmin_cutoff = float(fmin_hz)
        else:
            fmin_cutoff = 0.0

        if fmin_cutoff > 0:
            f_keep = f_axis_pos >= fmin_cutoff
            f_axis_pos = f_axis_pos[f_keep]
            S_pos = S_pos[:, f_keep]


        # Convert search window to rad/m
        k_min_rm = k_min_rad_um * 1e6
        k_max_rm = (k_max_rad_um * 1e6) if k_max_rad_um is not None else np.inf

        # Build k-side mask
        if side == "positive":
            k_mask = (k_axis > k_min_rm) & (k_axis <= k_max_rm)
        elif side == "negative":
            k_mask = (k_axis < -k_min_rm) & (k_axis >= -k_max_rm)
        else:  # "both"
            k_mask = (np.abs(k_axis) >= k_min_rm) & (np.abs(k_axis) <= k_max_rm)

        if not np.any(k_mask):
            raise ValueError(
                f"No k-bins found for side={side!r} with k_min={k_min_rad_um} "
                f"rad/μm.  Check k-axis range."
            )

        # SNR gate: exclude k-bins with very low total power
        global_max = float(S_pos.max())
        row_max = S_pos.max(axis=1)  # (Nk,)
        snr_mask = row_max >= min_snr * global_max
        combined_mask = k_mask & snr_mask

        if not np.any(combined_mask):
            # fall back without SNR gate
            combined_mask = k_mask

        k_search = k_axis[combined_mask]
        S_search = S_pos[combined_mask, :]

        # Extract f_peak per k-bin
        if peak_method == "centroid":
            total_power = S_search.sum(axis=1, keepdims=True) + 1e-30
            f_peak_hz = (S_search * f_axis_pos[np.newaxis, :]).sum(axis=1) / total_power[:, 0]
        else:  # argmax
            f_peak_idx = np.argmax(S_search, axis=1)
            f_peak_hz = f_axis_pos[f_peak_idx]

        # Smooth the branch
        if smooth_sigma and smooth_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter1d
                f_peak_hz = gaussian_filter1d(f_peak_hz.astype(float), sigma=smooth_sigma)
            except ImportError:
                # simple box smooth fallback
                w = max(1, int(smooth_sigma * 2))
                kernel = np.ones(w) / w
                f_peak_hz = np.convolve(f_peak_hz, kernel, mode="same")

        # Find minimum
        idx_min = int(np.argmin(f_peak_hz))
        k_star = float(k_search[idx_min])
        f_min = float(f_peak_hz[idx_min])

        # f at k≈0 (use full k-axis)
        idx_k0 = int(np.argmin(np.abs(k_axis)))
        f_k0 = float(f_axis_pos[np.argmax(S_pos[idx_k0, :])])

        # Group velocity at k* via gradient on smoothed branch
        if k_search.size > 2:
            dk = np.gradient(k_search)
            df = np.gradient(f_peak_hz)
            vg_arr = 2 * np.pi * (df / (dk + 1e-30))
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

    def scan(
        self,
        sources,
        param_values,
        param_label: str = "parameter",
        *,
        filters: Optional[dict] = None,
        compute_kwargs: Optional[dict] = None,
        find_kwargs: Optional[dict] = None,
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
        sources : iterable
            MMPP jobs, pre-computed :class:`DispersionResult1D` instances, or
            callables returning ``DispersionResult1D``.
        param_values : sequence of float
            One value per source item.
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
        >>> bulk = result.analyze.scan(
        ...     jobs,                         # mmpp.MMPP object — iterated automatically
        ...     param_values=[0, 10, 20, 30],
        ...     param_label="B_ext [mT]",
        ...     filters=dict(remove_static=True),
        ...     # slice_spec=(slice(None), Ellipsis, slice(0, 1))  # optional: first z-layer
        ... )
        >>> bulk.plot.summary()
        """
        from .bulk import scan_minimum_frequency

        # Inherit config from current result where not overridden
        res = self._result
        _filters       = filters       or {}
        _find_kwargs   = find_kwargs   or {}
        _compute_kw    = compute_kwargs or {"axis": getattr(res, "axis", "x")}

        return scan_minimum_frequency(
            sources,
            param_values=param_values,
            param_label=param_label,
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
            (".find_lowest_possible_frequency()",
             "→ LowestFrequencyResult",
             "Default: side='positive', smooth_sigma=2.0, peak_method='argmax'. "
             "Finds k* where f(k) is minimum on the dispersion branch."),
            (".find_lowest_possible_frequency(side='both')",
             "Search full k-axis",
             "side='both' searches k<0 and k>0. Use for isotropic or symmetric systems."),
            (".find_lowest_possible_frequency(smooth_sigma=None, peak_method='centroid')",
             "No smoothing, centroid peak",
             "smooth_sigma=None disables Gaussian smoothing. peak_method='centroid' uses power-weighted "
             "centroid frequency instead of argmax — more robust for broad peaks."),
            (".find_lowest_possible_frequency(k_min_rad_um=0.5, k_max_rad_um=8.0)",
             "Restrict k search window",
             "k_min_rad_um / k_max_rad_um restrict the search range in rad/\u03bcm. Useful to exclude the k=0 region."),
            (".find_lowest_possible_frequency(fmin_hz=2e9)",
             "Explicit fmin cutoff (Hz)",
             "fmin_hz='auto' (default) skips lowest 5% of spectrum. Set explicit Hz value, or None to disable."),
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
