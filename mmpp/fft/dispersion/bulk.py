"""Bulk / sweep analysis for spin-wave dispersion minimum frequency.

The core problem: storing full S(k,f) for dozens of simulations is expensive
(each array can be ~10–100 MB).  This module extracts only what is needed:

* Scalar summary per job: f_min, k*, vg(k*), f(k=0), Δf
* 1-D cross-section  : S(k) evaluated at f ≈ f_min  →  memory tiny
* 1-D branch         : f_peak(k)                     →  memory tiny

These are collected into a :class:`BulkMinimumFrequencyResult` that
supports heatmaps, line plots, and serialisation.

Typical usage::

    from mmpp.fft.dispersion.bulk import scan_minimum_frequency

    jobs = [mmpp.MMPP(path) for path in zarr_paths]

    bulk = scan_minimum_frequency(
        jobs,
        param_values=[0, 10, 20, 30, 40],     # e.g. field [mT]
        param_label="B_ext [mT]",
        filters=dict(remove_static=True, live={"gaussian_morph": {"enabled": True}}),
        find_kwargs=dict(side="positive", smooth_sigma=2.0),
        compute_kwargs=dict(axis="x", component="perp"),
    )

    bulk.plot.heatmap()            # S(k) at f_min vs param
    bulk.plot.f_min_vs_param()     # f_min(param) line plot
    bulk.plot.branches()           # all f_peak(k) curves stacked

Or from a single result via the .analyze accessor::

    bulk = result.analyze.scan([result1, result2, ...], params=[0, 10], ...)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from .models import DispersionResult1D
    from .analyze import LowestFrequencyResult, FindLowestKwargs


# ---------------------------------------------------------------------------
# Low-level extractor (memory-efficient)
# ---------------------------------------------------------------------------

def _extract_compact(
    result: "DispersionResult1D",
    find_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Compute LowestFrequencyResult and extract compact data from it.

    Does NOT store the full S(k,f).  Stores:
    - scalars: f_min, k_star, vg, f_k0
    - 1-D crosssection_at_fmin: S(k) at the f-bin nearest f_min
    - 1-D crosssection_at_fk0:  S(k) at the f-bin nearest f(k=0)
    - 1-D branch_f: f_peak(k) over searched k range
    - 1-D branch_k: k values for branch_f
    - k_axis: full k-axis [rad/m]
    """
    from .analyze import DispersionAnalyzeAccessor

    lowest = DispersionAnalyzeAccessor(result).find_lowest_possible_frequency(**find_kwargs)

    S = result.S          # (Nk, Nf)
    f_axis = result.f_axis
    k_axis = result.k_axis

    pos_f = f_axis >= 0
    f_pos = f_axis[pos_f]
    S_pos = S[:, pos_f]

    # Cross-section at f_min
    idx_fmin = int(np.abs(f_pos - lowest.f_min_hz).argmin())
    cs_at_fmin = S_pos[:, idx_fmin].copy()

    # Cross-section at f(k=0)
    idx_fk0 = int(np.abs(f_pos - lowest.f_at_k0_hz).argmin())
    cs_at_fk0 = S_pos[:, idx_fk0].copy()

    return {
        "f_min_hz":              lowest.f_min_hz,
        "k_star_rad_m":          lowest.k_at_f_min,
        "vg_at_min_m_s":         lowest.group_velocity_at_min,
        "f_at_k0_hz":            lowest.f_at_k0_hz,
        "crosssection_at_fmin":  cs_at_fmin,
        "crosssection_at_fk0":   cs_at_fk0,
        "branch_f":              lowest.branch_f.copy(),
        "branch_k":              lowest.branch_k.copy(),
        "k_axis":                k_axis.copy(),
        "f_axis_pos":            f_pos.copy(),
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BulkMinimumFrequencyResult:
    """Collection of minimum-frequency analysis results across a parameter sweep.

    Stores only compact data (scalars + 1-D cross-sections) — no full S(k,f).

    Attributes
    ----------
    param_values : np.ndarray
        Scan parameter values (shape: N).
    param_label : str
        Human-readable label for the scan parameter (e.g. ``"B_ext [mT]"``).
    f_min_hz : np.ndarray
        Minimum branch frequency for each scan point [Hz] (shape: N).
    k_star_rad_m : np.ndarray
        Wave-vector at f_min for each scan point [rad/m] (shape: N).
    vg_at_min : np.ndarray
        Group velocity at k* for each scan point [m/s] (shape: N).
    f_at_k0_hz : np.ndarray
        f(k≈0) / FMR frequency for each scan point [Hz] (shape: N).
    crosssections_at_fmin : list[np.ndarray]
        S(k) at f≈f_min for each scan point.  Each array has shape (Nk,).
    crosssections_at_fk0 : list[np.ndarray]
        S(k) at f≈f(k=0) for each scan point.
    branches_f : list[np.ndarray]
        f_peak(k) for each scan point.
    branches_k : list[np.ndarray]
        Corresponding k-axis for branches_f.
    k_axes : list[np.ndarray]
        Full k-axis [rad/m] for each scan point (may differ between jobs).
    errors : dict[int, str]
        ``{index: error_message}`` for jobs that failed to compute.
    meta : dict
        Arbitrary metadata (axis, component, filters_config, …).
    """

    param_values: np.ndarray
    param_label: str
    f_min_hz: np.ndarray
    k_star_rad_m: np.ndarray
    vg_at_min: np.ndarray
    f_at_k0_hz: np.ndarray
    crosssections_at_fmin: list[np.ndarray]
    crosssections_at_fk0: list[np.ndarray]
    branches_f: list[np.ndarray]
    branches_k: list[np.ndarray]
    k_axes: list[np.ndarray]
    errors: dict[int, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of scan points."""
        return len(self.param_values)

    @property
    def f_min_ghz(self) -> np.ndarray:
        return self.f_min_hz / 1e9

    @property
    def f_at_k0_ghz(self) -> np.ndarray:
        return self.f_at_k0_hz / 1e9

    @property
    def k_star_rad_um(self) -> np.ndarray:
        return self.k_star_rad_m / 1e6

    @property
    def delta_f_mhz(self) -> np.ndarray:
        """Δf = f(k=0) − f_min  [MHz]."""
        return (self.f_at_k0_hz - self.f_min_hz) / 1e6

    @property
    def plot(self) -> "BulkMinimumPlotAccessor":
        return BulkMinimumPlotAccessor(self)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        """Save to a compressed numpy archive (``.npz``).

        Parameters
        ----------
        path : path-like
            Output file path.  ``.npz`` extension added if absent.

        Returns
        -------
        Path
            Absolute path of the saved file.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, Any] = {
            "param_values":   self.param_values,
            "f_min_hz":       self.f_min_hz,
            "k_star_rad_m":   self.k_star_rad_m,
            "vg_at_min":      self.vg_at_min,
            "f_at_k0_hz":     self.f_at_k0_hz,
            "param_label":    np.array([self.param_label]),
        }
        for i, (cs_fmin, cs_fk0, bf, bk, ka) in enumerate(
            zip(
                self.crosssections_at_fmin,
                self.crosssections_at_fk0,
                self.branches_f,
                self.branches_k,
                self.k_axes,
            )
        ):
            arrays[f"cs_fmin_{i}"]   = cs_fmin
            arrays[f"cs_fk0_{i}"]    = cs_fk0
            arrays[f"branch_f_{i}"]  = bf
            arrays[f"branch_k_{i}"]  = bk
            arrays[f"k_axis_{i}"]    = ka

        arrays["errors_keys"]   = np.array(list(self.errors.keys()), dtype=int)
        arrays["errors_values"] = np.array(list(self.errors.values()), dtype=object)

        np.savez_compressed(str(path), **arrays)
        return path.resolve()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BulkMinimumFrequencyResult":
        """Load from a ``.npz`` file saved by :meth:`save`.

        Parameters
        ----------
        path : path-like
            Path to the ``.npz`` file.
        """
        path = Path(path)
        data = np.load(str(path), allow_pickle=True)

        param_values  = data["param_values"]
        n             = len(param_values)
        param_label   = str(data["param_label"][0])
        f_min_hz      = data["f_min_hz"]
        k_star_rad_m  = data["k_star_rad_m"]
        vg_at_min     = data["vg_at_min"]
        f_at_k0_hz    = data["f_at_k0_hz"]

        cs_fmin   = [data[f"cs_fmin_{i}"]  for i in range(n) if f"cs_fmin_{i}"  in data]
        cs_fk0    = [data[f"cs_fk0_{i}"]   for i in range(n) if f"cs_fk0_{i}"   in data]
        branches_f = [data[f"branch_f_{i}"] for i in range(n) if f"branch_f_{i}" in data]
        branches_k = [data[f"branch_k_{i}"] for i in range(n) if f"branch_k_{i}" in data]
        k_axes     = [data[f"k_axis_{i}"]   for i in range(n) if f"k_axis_{i}"   in data]

        keys   = list(data.get("errors_keys",   np.array([], dtype=int)))
        values = list(data.get("errors_values", np.array([], dtype=object)))
        errors = {int(k): str(v) for k, v in zip(keys, values)}

        return cls(
            param_values=param_values,
            param_label=param_label,
            f_min_hz=f_min_hz,
            k_star_rad_m=k_star_rad_m,
            vg_at_min=vg_at_min,
            f_at_k0_hz=f_at_k0_hz,
            crosssections_at_fmin=cs_fmin,
            crosssections_at_fk0=cs_fk0,
            branches_f=branches_f,
            branches_k=branches_k,
            k_axes=k_axes,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # DataFrame export
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """Export scalar results to a :class:`pandas.DataFrame`.

        Columns: ``param``, ``f_min_ghz``, ``k_star_rad_um``,
        ``vg_at_min_km_s``, ``f_at_k0_ghz``, ``delta_f_mhz``.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe().")
        return pd.DataFrame(
            {
                "param":            self.param_values,
                "f_min_ghz":        self.f_min_ghz,
                "k_star_rad_um":    self.k_star_rad_um,
                "vg_at_min_km_s":   self.vg_at_min / 1e3,
                "f_at_k0_ghz":      self.f_at_k0_ghz,
                "delta_f_mhz":      self.delta_f_mhz,
            }
        )

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ok = self.n - len(self.errors)
        return (
            f"BulkMinimumFrequencyResult("
            f"n={self.n}, ok={ok}, errors={len(self.errors)}, "
            f"param={self.param_label!r}, "
            f"f_min=[{self.f_min_ghz.min():.3f}..{self.f_min_ghz.max():.3f}] GHz)"
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        HV = "onmouseover=\"this.style.background='#1e293b'\" onmouseout=\"this.style.background='transparent'\""
        ok = self.n - len(self.errors)

        stat_rows = [
            ("scan points",   f"{self.n}  ({ok} OK, {len(self.errors)} errors)",
             "Total jobs in scan; errors shows jobs that raised exceptions."),
            ("parameter",     _esc(self.param_label),
             "Scan parameter label as provided to scan_minimum_frequency()."),
            ("f_min range",   f"{self.f_min_ghz.min():.4f} … {self.f_min_ghz.max():.4f} GHz",
             "Range of minimum branch frequency across all scan points."),
            ("k* range",      f"{self.k_star_rad_um.min():.3f} … {self.k_star_rad_um.max():.3f} rad/μm",
             "Range of wave-vector at f_min across scan points."),
            ("Δf range",      f"{self.delta_f_mhz.min():.2f} … {self.delta_f_mhz.max():.2f} MHz",
             "Range of Δf = f(k=0) − f_min. Positive = backward-volume character."),
            ("stored per pt", "S(k)|f_min  +  S(k)|f(k=0)  +  f_peak(k)",
             "Compact 1-D arrays only — full S(k,f) is NOT stored."),
        ]
        stat_html = "".join(
            f"<tr {HV} title=\"{_esc(tip)}\" style='cursor:help;'>"
            f"<td style='padding:3px 10px;font-family:monospace;color:#93c5fd;font-size:.86em;'>{_esc(k)}</td>"
            f"<td style='padding:3px 10px;color:#a5b4fc;font-size:.88em;font-weight:600;'>{v}</td>"
            f"</tr>"
            for k, v, tip in stat_rows
        )

        plot_methods = [
            (".plot.heatmap()",
             "2-D heatmap: S(k)|f_min vs param",
             "Color = spectral density at f_min, x = k [rad/μm], y = param. "
             "Shows how the k-profile at f_min changes with the scan parameter."),
            (".plot.f_min_vs_param()",
             "f_min(param) line plot",
             "Scatter/line of the minimum branch frequency vs scan parameter. "
             "Optionally overlays f(k=0)."),
            (".plot.k_star_vs_param()",
             "k*(param) line plot",
             "Wave-vector at f_min as a function of the scan parameter."),
            (".plot.delta_f_vs_param()",
             "Δf(param) = f(k=0) − f_min",
             "Shows how the frequency gap between k=0 and f_min varies with parameter."),
            (".plot.branches()",
             "f_peak(k) for all scan points",
             "Stacked line plot of all extracted f_peak(k) branches, coloured by param."),
            (".plot.vg_vs_param()",
             "v_g(k*) vs param",
             "Group velocity at k* as a function of the scan parameter [km/s]."),
        ]
        plot_html = "".join(
            f"<tr {HV} title=\"{_esc(tip)}\" style='cursor:pointer;'>"
            f"<td style='padding:3px 10px;font-family:monospace;color:#f9a8d4;font-size:.84em;'>{_esc(sig)}</td>"
            f"<td style='padding:3px 10px;color:#cbd5e1;font-size:.82em;'>{_esc(desc)}</td>"
            f"</tr>"
            for sig, desc, tip in plot_methods
        )

        error_section = ""
        if self.errors:
            err_rows = "".join(
                f"<tr><td style='padding:2px 8px;color:#fca5a5;font-family:monospace;font-size:.8em;'>{i}</td>"
                f"<td style='padding:2px 8px;color:#f87171;font-size:.78em;'>{_esc(str(msg)[:100])}</td></tr>"
                for i, msg in self.errors.items()
            )
            error_section = (
                f"<details style='margin-top:6px;'>"
                f"<summary style='cursor:pointer;color:#fca5a5;font-size:.8em;list-style:none;'>"
                f"&#9654; {len(self.errors)} error(s)</summary>"
                f"<table style='margin-top:4px;border-collapse:collapse;'>{err_rows}</table>"
                f"</details>"
            )

        return (
            "<div style='font-family:-apple-system,BlinkMacSystemFont,sans-serif;"
            "border:2px solid #334155;border-left:4px solid #6366f1;"
            "border-radius:10px;padding:14px;margin:6px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#0c1a35 100%);"
            "color:#e2e8f0;max-width:720px;'>"
            "<div style='font-weight:700;font-size:1.05em;color:#f1f5f9;margin-bottom:10px;'>"
            "BulkMinimumFrequencyResult"
            "<span style='background:#312e81;color:#c7d2fe;border-radius:4px;"
            "padding:1px 8px;font-size:.72em;font-weight:600;margin-left:8px;'>"
            f"n={self.n} scan points</span></div>"
            f"<table style='width:100%;border-collapse:collapse;margin-bottom:8px;'>{stat_html}</table>"
            + error_section
            + "<details style='margin-top:8px;'>"
            "<summary style='cursor:pointer;font-size:.83em;color:#f9a8d4;list-style:none;"
            "padding:3px 6px;background:#1e293b;border-radius:5px;'>"
            "▶ <code>.plot</code> — BulkMinimumPlotAccessor</summary>"
            "<div style='margin-left:12px;margin-top:4px;'>"
            f"<table style='width:100%;border-collapse:collapse;'>{plot_html}</table>"
            "</div></details>"
            "<div style='margin-top:8px;font-size:.77em;color:#475569;'>"
            "<code style='color:#a5b4fc;'>.save(path)</code> / "
            "<code style='color:#a5b4fc;'>.load(path)</code> — npz archive &nbsp;|&nbsp; "
            "<code style='color:#a5b4fc;'>.to_dataframe()</code> — pandas export"
            "</div>"
            "</div>"
        )


# ---------------------------------------------------------------------------
# Plot accessor
# ---------------------------------------------------------------------------

class BulkMinimumPlotAccessor:
    """Plotting namespace for :class:`BulkMinimumFrequencyResult`.

    Accessed via ``bulk.plot``.

    Methods
    -------
    heatmap()
        2-D heatmap of S(k) at f_min vs scan parameter.
    f_min_vs_param()
        f_min as a function of the scan parameter.
    k_star_vs_param()
        k* as a function of the scan parameter.
    delta_f_vs_param()
        Δf = f(k=0) − f_min as a function of the scan parameter.
    vg_vs_param()
        Group velocity at k* vs scan parameter.
    branches()
        All extracted f_peak(k) curves stacked by parameter.
    """

    def __init__(self, bulk: BulkMinimumFrequencyResult) -> None:
        self._bulk = bulk

    # ------------------------------------------------------------------

    def heatmap(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (12, 7),
        dpi: Optional[int] = None,
        which: str = "fmin",
        cmap: str = "cmc.davos",
        kscale: str = "rad_um",
        lognorm: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        k_xlim: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        annotate_fmin: bool = True,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """2-D heatmap of S(k) cross-sections stacked vs scan parameter.

        This is the key bulk plot: each row is S(k) at f≈f_min (or f≈f(k=0))
        for one scan point; rows are ordered by the parameter value.

        Parameters
        ----------
        which : ``"fmin"`` | ``"fk0"``
            Which cross-section to plot.  ``"fmin"`` uses S(k) at f=f_min;
            ``"fk0"`` uses S(k) at f=f(k=0).
        kscale : ``"rad_um"`` | ``"meter"``
            Wave-vector unit.
        lognorm : bool
            Logarithmic color scale.
        annotate_fmin : bool
            Draw scatter of k* positions on top of the heatmap.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        try:
            import cmcrameri  # noqa: F401
        except ImportError:
            cmap = "viridis"

        bulk = self._bulk

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        # Build matrix: (N_params, Nk) — interpolate to a common k-grid
        n = bulk.n
        # Use the k-axis of the first valid job as reference
        k_ref = bulk.k_axes[0]
        cs_list = (
            bulk.crosssections_at_fmin if which == "fmin"
            else bulk.crosssections_at_fk0
        )

        rows = []
        for i, (cs, ka) in enumerate(zip(cs_list, bulk.k_axes)):
            if ka.size == k_ref.size:
                rows.append(cs)
            else:
                # Interpolate to reference k-axis
                rows.append(np.interp(k_ref, ka, cs, left=0.0, right=0.0))
        matrix = np.array(rows)   # (N, Nk)

        k_plot = k_ref / 1e6 if kscale == "rad_um" else k_ref
        k_label = r"$k$ [rad/μm]" if kscale == "rad_um" else r"$k$ [rad/m]"
        params = bulk.param_values

        extent = [
            float(k_plot[0]), float(k_plot[-1]),
            float(params[0]),  float(params[-1]),
        ]

        norm = None
        if lognorm:
            s_min = matrix[matrix > 0].min() if np.any(matrix > 0) else 1e-12
            norm = LogNorm(
                vmin=vmin if vmin is not None else s_min,
                vmax=vmax if vmax is not None else matrix.max(),
            )
        elif vmin is not None or vmax is not None:
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=vmin, vmax=vmax)

        im = ax.imshow(
            matrix,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            origin="lower",
            extent=extent,
        )
        fig.colorbar(im, ax=ax, label="S(k)|f_min  [arb. units]")

        if annotate_fmin and which == "fmin":
            ks = bulk.k_star_rad_um if kscale == "rad_um" else bulk.k_star_rad_m
            ax.scatter(
                ks, params,
                s=50, color="red", marker="*", zorder=10,
                label="k* (f_min)",
            )
            ax.legend(fontsize=8)

        ax.set_xlabel(k_label)
        ax.set_ylabel(bulk.param_label)
        ax.set_title(title or f"S(k) at f_min  vs  {bulk.param_label}")

        if k_xlim is not None:
            ax.set_xlim(*k_xlim)

        try:
            fig.tight_layout()
        except Exception:
            pass

        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def f_min_vs_param(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (8, 5),
        dpi: Optional[int] = None,
        show_fk0: bool = True,
        show_delta_f: bool = False,
        f_units: str = "GHz",
        title: Optional[str] = None,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot f_min (and optionally f(k=0)) vs scan parameter.

        Parameters
        ----------
        show_fk0 : bool
            Overlay f(k=0) as a dashed cyan line.
        show_delta_f : bool
            Add a second y-axis showing Δf = f(k=0) − f_min.
        """
        import matplotlib.pyplot as plt

        bulk = self._bulk
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        p = bulk.param_values
        f_min  = bulk.f_min_ghz  if f_units == "GHz" else bulk.f_min_hz
        f_k0   = bulk.f_at_k0_ghz if f_units == "GHz" else bulk.f_at_k0_hz
        f_label = "f [GHz]"      if f_units == "GHz" else "f [Hz]"

        ax.plot(p, f_min, "o-", color="#f97316", linewidth=2, markersize=6, label="f_min")
        if show_fk0:
            ax.plot(p, f_k0, "s--", color="#22d3ee", linewidth=1.5, markersize=5, label="f(k=0)")
        ax.set_xlabel(bulk.param_label)
        ax.set_ylabel(f_label)
        ax.set_title(title or f"f_min  vs  {bulk.param_label}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

        if show_delta_f:
            ax2 = ax.twinx()
            df = bulk.delta_f_mhz
            ax2.plot(p, df, "^:", color="#a78bfa", linewidth=1.5, markersize=4, label="Δf [MHz]")
            ax2.set_ylabel("Δf = f(k=0) − f_min  [MHz]", color="#a78bfa")
            ax2.tick_params(axis="y", labelcolor="#a78bfa")

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def k_star_vs_param(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (8, 5),
        dpi: Optional[int] = None,
        kscale: str = "rad_um",
        title: Optional[str] = None,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot k* (wave-vector at f_min) vs scan parameter."""
        import matplotlib.pyplot as plt

        bulk = self._bulk
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        k_data  = bulk.k_star_rad_um if kscale == "rad_um" else bulk.k_star_rad_m
        k_label = r"$k^*$ [rad/μm]" if kscale == "rad_um" else r"$k^*$ [rad/m]"

        ax.plot(bulk.param_values, k_data, "D-", color="#4ade80", linewidth=2, markersize=6)
        ax.set_xlabel(bulk.param_label)
        ax.set_ylabel(k_label)
        ax.set_title(title or f"k* (wave-vector at f_min)  vs  {bulk.param_label}")
        ax.grid(True, alpha=0.25)

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def delta_f_vs_param(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (8, 5),
        dpi: Optional[int] = None,
        title: Optional[str] = None,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot Δf = f(k=0) − f_min  vs scan parameter [MHz]."""
        import matplotlib.pyplot as plt

        bulk = self._bulk
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        ax.plot(bulk.param_values, bulk.delta_f_mhz, "o-", color="#a78bfa", linewidth=2, markersize=6)
        ax.axhline(0, color="#475569", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xlabel(bulk.param_label)
        ax.set_ylabel("Δf = f(k=0) − f_min  [MHz]")
        ax.set_title(title or f"Δf  vs  {bulk.param_label}")
        ax.grid(True, alpha=0.25)

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def vg_vs_param(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (8, 5),
        dpi: Optional[int] = None,
        title: Optional[str] = None,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot group velocity at k* vs scan parameter [km/s]."""
        import matplotlib.pyplot as plt

        bulk = self._bulk
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        vg_km_s = self._bulk.vg_at_min / 1e3

        ax.plot(bulk.param_values, vg_km_s, "s-", color="#fb923c", linewidth=2, markersize=6)
        ax.axhline(0, color="#475569", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xlabel(bulk.param_label)
        ax.set_ylabel("v_g(k*)  [km/s]")
        ax.set_title(title or f"Group velocity at k*  vs  {bulk.param_label}")
        ax.grid(True, alpha=0.25)

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def branches(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (10, 6),
        dpi: Optional[int] = None,
        kscale: str = "rad_um",
        f_units: str = "GHz",
        cmap: str = "viridis",
        alpha: float = 0.7,
        linewidth: float = 1.5,
        title: Optional[str] = None,
        colorbar: bool = True,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot all extracted f_peak(k) branches, coloured by scan parameter.

        Parameters
        ----------
        cmap : str
            Colormap used to colour branches by parameter value.
        colorbar : bool
            Add a colorbar encoding the parameter value.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        bulk = self._bulk
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        params = bulk.param_values
        p_min, p_max = float(params.min()), float(params.max())
        cmap_obj = cm.get_cmap(cmap)
        norm = mcolors.Normalize(vmin=p_min, vmax=p_max)

        for i, (bk, bf, p) in enumerate(zip(bulk.branches_k, bulk.branches_f, params)):
            k_plot = bk / 1e6 if kscale == "rad_um" else bk
            f_plot = bf / 1e9 if f_units == "GHz" else bf
            color = cmap_obj(norm(float(p)))
            ax.plot(k_plot, f_plot, color=color, linewidth=linewidth, alpha=alpha)

        k_label = r"$k$ [rad/μm]" if kscale == "rad_um" else r"$k$ [rad/m]"
        f_label = "f [GHz]"      if f_units == "GHz"     else "f [Hz]"
        ax.set_xlabel(k_label)
        ax.set_ylabel(f_label)
        ax.set_title(title or f"f_peak(k) branches  —  {bulk.param_label}")
        ax.grid(True, alpha=0.2)

        if colorbar:
            sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=bulk.param_label, fraction=0.03)

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def summary(
        self,
        figsize: tuple[float, float] = (14, 10),
        dpi: Optional[int] = None,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", Any]:
        """4-panel summary figure: heatmap, f_min, k*, Δf."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize, **({"dpi": dpi} if dpi else {}))
        ax_hm, ax_fmin, ax_kstar, ax_df = axes.ravel()

        self.heatmap(ax=ax_hm)
        self.f_min_vs_param(ax=ax_fmin, show_fk0=True, show_delta_f=False)
        self.k_star_vs_param(ax=ax_kstar)
        self.delta_f_vs_param(ax=ax_df)

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, axes

    # ------------------------------------------------------------------

    def _save(self, fig: "Figure", save: Any) -> None:
        from datetime import datetime
        if isinstance(save, bool):
            ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            path = Path(f"bulk_dispersion_{ts}.png")
        else:
            path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")

    def __repr__(self) -> str:
        return (
            "<BulkMinimumPlotAccessor: .heatmap(), .f_min_vs_param(), "
            ".k_star_vs_param(), .delta_f_vs_param(), .vg_vs_param(), "
            ".branches(), .summary()>"
        )


# ---------------------------------------------------------------------------
# Main entry-point: scan_minimum_frequency
# ---------------------------------------------------------------------------

def scan_minimum_frequency(
    sources: Iterable[Any],
    param_values: Sequence[float],
    param_label: str = "parameter",
    *,
    filters: Optional[dict[str, Any]] = None,
    compute_kwargs: Optional[dict[str, Any]] = None,
    find_kwargs: Optional[dict[str, Any]] = None,
    slice_spec: Optional[Any] = None,
    dataset: str = "m",
    z_slice: Optional[Any] = None,
    verbose: bool = True,
    on_error: str = "warn",
) -> BulkMinimumFrequencyResult:
    """Scan the minimum branch frequency across a parameter sweep.

    For each source the function:

    1. Accesses the magnetisation dataset (``job[0].m[...]``).
    2. Applies ``filters`` via ``fft.dispersion.filters(...)``.
    3. Calls ``compute_1d(...)`` to get :class:`DispersionResult1D`.
    4. Extracts only compact data (scalars + 1-D cross-sections) — the
       full ``S(k,f)`` array is immediately discarded.

    Memory usage per job is therefore O(Nk) instead of O(Nk × Nf).

    Parameters
    ----------
    sources : iterable
        Each item must be either:

        * An ``mmpp.MMPP`` instance — accessed as ``src[0].m[slice_spec]``.
        * A precomputed :class:`DispersionResult1D` — used directly.
        * A callable ``() -> DispersionResult1D`` — called once per job.
    param_values : sequence of float
        One value per source item; used as the scan axis.
    param_label : str
        Human-readable label for the scan parameter (used in plots).
    filters : dict, optional
        Passed to ``fft.dispersion.filters(**filters)``.
    compute_kwargs : dict, optional
        Passed to ``DispersionFilterChain.compute_1d(**compute_kwargs)``.
    find_kwargs : dict, optional
        Passed to ``DispersionAnalyzeAccessor.find_lowest_possible_frequency(**find_kwargs)``.
    slice_spec : slice / index, optional
        Spatial slice applied to ``job[0].m[slice_spec]``.
        Default: ``[:, ..., 0:1]`` (first z-layer).
    dataset : str
        Dataset name on the simulation object (default ``"m"``).
    z_slice : optional
        Legacy alias for *slice_spec*.
    verbose : bool
        Print progress to stdout.
    on_error : ``"warn"`` | ``"raise"`` | ``"skip"``
        How to handle per-job exceptions.

    Returns
    -------
    BulkMinimumFrequencyResult
        Compact sweep result with ``.plot``, ``.save()``, ``.to_dataframe()``.

    Examples
    --------
    Sweep over applied field values::

        import mmpp
        from mmpp.fft.dispersion.bulk import scan_minimum_frequency

        jobs = [mmpp.MMPP(p) for p in sorted(zarr_paths)]
        bulk = scan_minimum_frequency(
            jobs,
            param_values=[0, 5, 10, 15, 20, 25, 30],
            param_label="B_ext [mT]",
            filters=dict(remove_static=True,
                         live={"gaussian_morph": {"enabled": True, "sigma_f": 1.0}}),
            compute_kwargs=dict(axis="x", save=True, force=False),
            find_kwargs=dict(side="positive", smooth_sigma=2.0),
        )

        bulk.plot.summary()
        bulk.save("sweep_field.npz")
    """
    sources = list(sources)
    param_values_arr = np.asarray(param_values, dtype=float)

    if len(sources) != len(param_values_arr):
        raise ValueError(
            f"sources has {len(sources)} items but param_values has "
            f"{len(param_values_arr)} — must match."
        )

    filters       = filters       or {}
    compute_kwargs = compute_kwargs or {}
    find_kwargs   = find_kwargs   or {}

    if z_slice is not None and slice_spec is None:
        slice_spec = z_slice
    if slice_spec is None:
        slice_spec = (slice(None), Ellipsis, slice(0, 1))

    f_min_arr      = np.full(len(sources), np.nan)
    k_star_arr     = np.full(len(sources), np.nan)
    vg_arr_out     = np.full(len(sources), np.nan)
    f_k0_arr       = np.full(len(sources), np.nan)
    cs_fmin_list:  list[np.ndarray] = []
    cs_fk0_list:   list[np.ndarray] = []
    branches_f:    list[np.ndarray] = []
    branches_k:    list[np.ndarray] = []
    k_axes:        list[np.ndarray] = []
    errors: dict[int, str] = {}

    for i, src in enumerate(sources):
        if verbose:
            print(f"[{i+1}/{len(sources)}]  {param_label}={param_values_arr[i]}", end="  ")
        try:
            # --- Resolve DispersionResult1D ---------------------------------
            if callable(src) and not hasattr(src, "fft"):
                result: "DispersionResult1D" = src()
            elif hasattr(src, "S") and hasattr(src, "k_axis"):
                # Already a DispersionResult1D
                result = src  # type: ignore[assignment]
            else:
                # Treat as MMPP job
                m_data = getattr(src[0], dataset)[slice_spec]
                chain = m_data.fft.dispersion.filters(**filters)
                result = chain.compute_1d(**compute_kwargs)

            # --- Extract compact data (S(k,f) freed immediately) ----------
            compact = _extract_compact(result, find_kwargs)

            f_min_arr[i]  = compact["f_min_hz"]
            k_star_arr[i] = compact["k_star_rad_m"]
            vg_arr_out[i] = compact["vg_at_min_m_s"]
            f_k0_arr[i]   = compact["f_at_k0_hz"]
            cs_fmin_list.append(compact["crosssection_at_fmin"])
            cs_fk0_list.append(compact["crosssection_at_fk0"])
            branches_f.append(compact["branch_f"])
            branches_k.append(compact["branch_k"])
            k_axes.append(compact["k_axis"])

            if verbose:
                print(f"  f_min={f_min_arr[i]/1e9:.4f} GHz  k*={k_star_arr[i]/1e6:.3f} rad/μm")

            # Free the large array immediately
            del result

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            errors[i] = msg
            # Insert placeholder arrays
            cs_fmin_list.append(np.array([]))
            cs_fk0_list.append(np.array([]))
            branches_f.append(np.array([]))
            branches_k.append(np.array([]))
            k_axes.append(np.array([]))
            if on_error == "raise":
                raise
            elif on_error == "warn":
                warnings.warn(f"Job {i} ({param_label}={param_values_arr[i]}) failed: {msg}")
            if verbose:
                print(f"  ERROR: {msg}")

    return BulkMinimumFrequencyResult(
        param_values=param_values_arr,
        param_label=param_label,
        f_min_hz=f_min_arr,
        k_star_rad_m=k_star_arr,
        vg_at_min=vg_arr_out,
        f_at_k0_hz=f_k0_arr,
        crosssections_at_fmin=cs_fmin_list,
        crosssections_at_fk0=cs_fk0_list,
        branches_f=branches_f,
        branches_k=branches_k,
        k_axes=k_axes,
        errors=errors,
        meta={
            "filters":         filters,
            "compute_kwargs":  compute_kwargs,
            "find_kwargs":     find_kwargs,
        },
    )
