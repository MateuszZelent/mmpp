"""DispersionPlotAccessor – plotting namespace on DispersionResult1D.

Usage::

    result = job[0].fft.dispersion.filters(...).compute_1d(axis='x')
    result.plot.heatmap(fmax=10, lognorm=True)
    result.plot.branch(branch)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ..models import DispersionBranch, DispersionResult1D


class DispersionPlotAccessor:
    """Plotting namespace for :class:`~mmpp.fft.dispersion.models.DispersionResult1D`.

    Accessed via ``result.plot``.

    Methods
    -------
    heatmap(...)
        S(k, f) heatmap (the main dispersion image).
    branch(branch, ...)
        Dispersion branch + group velocity panel.
    """

    def __init__(self, result: "DispersionResult1D") -> None:
        self._result = result

    # ------------------------------------------------------------------
    # primary plot: S(k,f) heatmap
    # ------------------------------------------------------------------

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
        lognorm: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        k_xlim: Optional[tuple[float, float]] = None,
        orth_index: Optional[int] = None,
        title: Optional[str] = None,
        live_filters: Optional[dict[str, Any]] = None,
        trim_0f: Optional[int] = None,
        save: Union[str, "Path", bool, None] = None,
        overlay_points: Optional[dict[str, Any]] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot S(k, f) dispersion heatmap.

        Parameters
        ----------
        ax : Axes, optional
            Target axes; creates a new figure when *None*.
        figsize, dpi
            Figure geometry (only used when *ax* is *None*).
        cmap : str
            Colormap; defaults to ``"cmc.davos"`` (crameri package required).
        kscale : ``"rad_um"`` | ``"rad"`` | ``"meter"``
            Wave-vector units.
        f_units : ``"GHz"`` | ``"Hz"``
            Frequency axis units.
        fmax : float, optional
            Clip display to this frequency (same units as *f_units*).
        lognorm : bool
            Use :class:`~matplotlib.colors.LogNorm` color scale.
        vmin, vmax : float, optional
            Manual color-scale limits.
        k_xlim : (float, float), optional
            Manual k-axis limits (in display units).
        orth_index : int, optional
            Select a single orthogonal slice from ``S_local``.
        title : str, optional
            Custom plot title.
        live_filters : dict, optional
            Post-processing filter dict applied at plot time to the cached S.
        trim_0f : int, optional
            Drop this many lowest-frequency bins (useful to hide DC artefacts).
        save : path-like or bool, optional
            Save figure to path or auto-generate filename.
        overlay_points : dict, optional
            ``{"k": array, "f": array, "style": dict}`` for arbitrary overlaid scatter.

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        try:
            import cmcrameri  # noqa: F401
        except ImportError:
            cmap = "viridis"

        result = self._result

        # --- Build figure/axes -----------------------------------------------
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        k_axis = result.k_axis.copy()
        f_axis = result.f_axis.copy()
        spectrum = result.S.copy()

        # Orthogonal slice selection
        if orth_index is not None:
            if result.S_local is None:
                raise ValueError(
                    "Result has no orthogonal slices; recompute with avg_over_orthogonal=False."
                )
            if not (0 <= orth_index < result.S_local.shape[0]):
                raise IndexError(
                    f"orth_index {orth_index} out of range "
                    f"(0..{result.S_local.shape[0] - 1})"
                )
            spectrum = result.S_local[orth_index]

        # Live filters
        if live_filters:
            try:
                from ..utils import apply_dispersion_post_filters

                spectrum = apply_dispersion_post_filters(
                    spectrum,
                    k_axis=result.k_axis,
                    f_axis=result.f_axis,
                    filters=live_filters,
                    include_live=True,
                )
            except Exception:
                pass  # degrade gracefully

        # Remove negative frequencies
        pos_mask = f_axis >= 0
        if pos_mask.sum() < f_axis.size:
            spectrum = spectrum[:, pos_mask]
            f_axis = f_axis[pos_mask]

        # Trim lowest bins
        if trim_0f and trim_0f > 0:
            spectrum = spectrum[:, trim_0f:]
            f_axis = f_axis[trim_0f:]

        # fmax clip (in Hz, then convert if needed)
        if fmax is not None and fmax > 0:
            fmax_hz = fmax * 1e9 if f_units == "GHz" else fmax
            mask_f = f_axis <= fmax_hz
            if np.any(mask_f):
                spectrum = spectrum[:, mask_f]
                f_axis = f_axis[mask_f]

        # Unit conversion
        if kscale == "rad_um":
            k_plot = k_axis / 1e6
            k_label = r"$k$ [rad/μm]"
            default_xlim: Optional[tuple[float, float]] = (-10.0, 10.0)
        elif kscale == "meter":
            k_plot = k_axis / (2 * np.pi)
            k_label = r"$k$ [m$^{-1}$]"
            default_xlim = (-20.0, 20.0)
        else:
            k_plot = k_axis
            k_label = r"$k$ [rad/m]"
            default_xlim = None

        if f_units == "GHz":
            f_plot = f_axis / 1e9
            f_label = "Frequency [GHz]"
        else:
            f_plot = f_axis
            f_label = "Frequency [Hz]"

        # Color normalisation
        norm = None
        if lognorm:
            s_min = spectrum[spectrum > 0].min() if np.any(spectrum > 0) else 1e-10
            norm = LogNorm(
                vmin=vmin if vmin is not None else s_min,
                vmax=vmax if vmax is not None else spectrum.max(),
            )
        elif vmin is not None or vmax is not None:
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=vmin, vmax=vmax)

        extent = (
            float(k_plot[0]),
            float(k_plot[-1]),
            float(f_plot[0]),
            float(f_plot[-1]),
        )

        im = ax.imshow(
            spectrum.T,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            origin="lower",
            extent=extent,
        )

        ax.set_xlabel(k_label)
        ax.set_ylabel(f_label)

        if k_xlim is not None:
            ax.set_xlim(*k_xlim)
        elif default_xlim is not None:
            ax.set_xlim(*default_xlim)

        if fmax is not None:
            ax.set_ylim(float(f_plot[0]), float(f_plot[-1]))

        if title is None:
            comp = getattr(result, "component", "")
            title = (
                f"Spin-Wave Dispersion S(k{result.axis}, f)"
                + (f" - {comp} component" if comp else "")
            )
        ax.set_title(title)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Power Spectral Density [arb. units]")

        # Optional scatter overlay
        if overlay_points is not None:
            ok = np.array(overlay_points.get("k", []), dtype=float)
            of = np.array(overlay_points.get("f", []), dtype=float)
            if ok.size and of.size:
                if kscale == "rad_um":
                    ok = ok / 1e6
                elif kscale == "meter":
                    ok = ok / (2 * np.pi)
                if f_units == "GHz":
                    of = of / 1e9
                style = dict(overlay_points.get("style", {}))
                style.setdefault("s", 40)
                style.setdefault("facecolors", "none")
                style.setdefault("edgecolors", "white")
                style.setdefault("linewidths", 1.5)
                ax.scatter(ok, of, **style)

        try:
            fig.tight_layout()
        except Exception:
            pass

        if save not in (None, False):
            self._save_fig(fig, save, result)

        return fig, ax

    # ------------------------------------------------------------------
    # branch + group velocity
    # ------------------------------------------------------------------

    def branch(
        self,
        branch: "DispersionBranch",
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (10, 6),
        kscale: str = "rad_um",
        f_units: str = "GHz",
        title: Optional[str] = None,
        save: Union[str, "Path", bool, None] = None,
    ) -> tuple["Figure", Any]:
        """Plot tracked dispersion branch and group velocity.

        Parameters
        ----------
        branch : DispersionBranch
            Pre-tracked branch from ``FFTDispersionInterface.track_branch()``.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_disp, ax_vg = axes

        k_data = branch.k_path.copy()
        f_data = branch.f_values.copy()

        if branch.group_velocity is None:
            branch.compute_group_velocity()
        vg_data = branch.group_velocity.copy()

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
        else:
            f_label = "f [Hz]"

        ax_disp.plot(k_data, f_data, "o-", linewidth=2, markersize=4)
        ax_disp.set_xlabel(k_label)
        ax_disp.set_ylabel(f_label)
        ax_disp.set_title(title or "Dispersion Branch")
        ax_disp.grid(True, alpha=0.3)

        ax_vg.plot(k_data, vg_data / 1e3, "s-", color="red", linewidth=2, markersize=4)
        ax_vg.set_xlabel(k_label)
        ax_vg.set_ylabel("Group Velocity [km/s]")
        ax_vg.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_vg.grid(True, alpha=0.3)

        try:
            fig.tight_layout()
        except Exception:
            pass

        if save not in (None, False):
            self._save_fig(fig, save, self._result)

        return fig, axes

    # ------------------------------------------------------------------
    # Analytical overlay
    # ------------------------------------------------------------------

    def add_analytics(
        self,
        ax: "Axes",
        *,
        model: str = "kalinikos",
        sw_config: str = "DE",
        n_modes: int = 1,
        color: str = "white",
        linestyle: str = "--",
        linewidth: float = 1.5,
        alpha: float = 0.8,
        label: Optional[str] = None,
        # Material parameter overrides (auto-detected from zarr if None):
        B: Optional[float] = None,
        Ms: Optional[float] = None,
        Aex: Optional[float] = None,
        d: Optional[float] = None,
        Ku: Optional[float] = None,
        g: float = 2.0,
        phi: Optional[float] = None,
        D: Optional[float] = None,
        k_points: int = 500,
        kscale: str = "rad_um",
        f_units: str = "GHz",
    ) -> tuple["Figure", "Axes"]:
        """Overlay analytical dispersion curve(s) on an existing axes.

        Material parameters are auto-extracted from zarr simulation attributes.
        Any parameter explicitly passed overrides the auto-detected value.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes (from a previous ``.heatmap()`` call).
        model : str
            Analytical model: ``"kalinikos"`` (default), ``"damon_eshbach"``,
            ``"backward_volume"``, ``"forward_volume"``, ``"bottcher"``,
            ``"kim"``, ``"cortes_ortuno"``.
        sw_config : str
            Spin-wave geometry preset:
            ``"DE"`` (k⊥M, φ=π/2), ``"BV"`` (k∥M, φ=0), ``"FV"`` (M⊥film).
        n_modes : int
            Number of PSSW modes to overlay (n=0, 1, …, n_modes-1).
        color, linestyle, linewidth, alpha
            Line styling for the overlay curves.
        label : str, optional
            Legend label. Auto-generated from model+sw_config if *None*.
        B, Ms, Aex, d, Ku, g, phi, D
            Material/geometry overrides. ``None`` = auto-detect from zarr.
        k_points : int
            Number of k-points for the analytical curve.
        kscale : str
            Must match the kscale used in heatmap (``"rad_um"`` | ``"rad"``).
        f_units : str
            Must match f_units used in heatmap (``"GHz"`` | ``"Hz"``).

        Returns
        -------
        fig, ax

        Examples
        --------
        >>> fig, ax = disp.plot.heatmap(fmax=10, lognorm=True)
        >>> disp.plot.add_analytics(ax, sw_config="DE")
        >>> disp.plot.add_analytics(ax, sw_config="BV", color="red")
        >>> disp.plot.add_analytics(ax, model="kalinikos", n_modes=3, sw_config="DE")
        """
        from ._analytics_overlay import (
            compute_analytical_dispersion,
            extract_material_params,
        )

        result = self._result
        fig = ax.get_figure()

        # Auto-extract params from zarr, then apply user overrides
        auto_params = extract_material_params(result)
        effective = {
            "B":   B   if B   is not None else auto_params.get("B"),
            "Ms":  Ms  if Ms  is not None else auto_params.get("Ms"),
            "Aex": Aex if Aex is not None else auto_params.get("Aex"),
            "d":   d   if d   is not None else auto_params.get("d"),
            "Ku":  Ku  if Ku  is not None else auto_params.get("Ku", 0.0),
            "g":   g,
        }

        # Validate required params
        missing = [k for k in ("B", "Ms", "Aex", "d") if effective[k] is None]
        if missing:
            raise ValueError(
                f"Cannot auto-detect material parameter(s): {missing}. "
                f"Please provide them explicitly, e.g.: "
                f"add_analytics(ax, B=0.1, Ms=8e5, Aex=13e-12, d=100e-9)"
            )

        # Get k-range from axes limits (in display units → back to rad/m)
        k_lo, k_hi = ax.get_xlim()
        if kscale == "rad_um":
            k_range = (k_lo * 1e6, k_hi * 1e6)  # rad/μm → rad/m
        elif kscale == "meter":
            k_range = (k_lo * 2 * np.pi, k_hi * 2 * np.pi)
        else:
            k_range = (k_lo, k_hi)

        # Compute analytical curves
        curves = compute_analytical_dispersion(
            k_range=k_range,
            model=model,
            sw_config=sw_config,
            n_modes=n_modes,
            k_points=k_points,
            phi=phi,
            D=D,
            B=effective["B"],
            Ms=effective["Ms"],
            d=effective["d"],
            Aex=effective["Aex"],
            Ku=effective["Ku"],
            g=effective["g"],
        )

        # Plot each mode
        for i, (k_arr, f_ghz, mode_label) in enumerate(curves):
            # Convert to display units
            if kscale == "rad_um":
                k_plot = k_arr / 1e6
            elif kscale == "meter":
                k_plot = k_arr / (2 * np.pi)
            else:
                k_plot = k_arr

            if f_units == "GHz":
                f_plot = f_ghz  # already in GHz
            else:
                f_plot = f_ghz * 1e9  # → Hz

            curve_label = label if label is not None else mode_label
            if n_modes > 1 and label is not None and i > 0:
                curve_label = f"{label} (n={i})"

            ax.plot(
                k_plot,
                f_plot,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha - 0.15 * i,  # lighter for higher modes
                label=curve_label,
            )

        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        return fig, ax

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _save_fig(self, fig: "Figure", save: Any, result: "DispersionResult1D") -> None:
        from datetime import datetime, timezone

        if isinstance(save, bool):
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            path = Path(f"dispersion_{result.axis}_{result.component}_{ts}.png")
        else:
            path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    def __repr__(self) -> str:
        return (
            "<DispersionPlotAccessor: .heatmap(...), .branch(branch, ...), .add_analytics(ax, ...)>"
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        HV = "onmouseover=\"this.style.background='#1e293b'\" onmouseout=\"this.style.background='transparent'\""

        methods = [
            (".heatmap(fmax=10, lognorm=True)",
             "S(k,f) power heatmap",
             "Main dispersion visualisation. Key params: fmax (GHz clip), lognorm (log color scale), "
             "kscale ('rad_um'|'meter'), cmap, vmin/vmax, k_xlim, orth_index, overlay_points, save."),
            (".heatmap(orth_index=0, lognorm=True)",
             "Single y-slice heatmap",
             "Select one orthogonal slice from S_local. Only available when result was computed with avg_over_orthogonal=False."),
            (".add_analytics(ax, sw_config='DE')",
             "Overlay analytical dispersion curve",
             "Auto-detects B, Ms, Aex, d from zarr attrs. sw_config: 'DE' (k⊥M), 'BV' (k∥M), 'FV' (M⊥film). "
             "model: 'kalinikos' (default), 'bottcher', 'kim', 'cortes_ortuno'. n_modes: PSSW mode count."),
            (".branch(branch, kscale='rad_um')",
             "Dispersion branch + v_g panel",
             "Two-panel plot: f(k) on left, group velocity dω/dk [km/s] on right. Pass a DispersionBranch from track_branch()."),
        ]
        rows = "".join(
            f"<tr {HV} title=\"{_esc(tip)}\" style='cursor:pointer;'>"
            f"<td style='padding:4px 10px;font-family:monospace;color:#93c5fd;font-size:.88em;'>{_esc(sig)}</td>"
            f"<td style='padding:4px 10px;color:#cbd5e1;font-size:.85em;'>{_esc(desc)}</td>"
            f"</tr>"
            for sig, desc, tip in methods
        )
        return (
            "<div style='font-family:-apple-system,sans-serif;border:2px solid #1d4ed8;"
            "border-radius:10px;padding:12px;margin:6px 0;background:#0f172a;"
            "color:#e2e8f0;max-width:680px;'>"
            "<div style='font-weight:700;color:#60a5fa;margin-bottom:8px;'>"
            "DispersionPlotAccessor"
            "<span style='font-size:.75em;color:#475569;font-weight:400;margin-left:8px;'>"
            "(hover rows for parameter details)</span></div>"
            f"<table style='width:100%;border-collapse:collapse;'>{rows}</table>"
            "<div style='margin-top:8px;font-size:.78em;color:#475569;'>"
            "All methods return <code style='color:#bae6fd;'>(fig, ax)</code> "
            "and accept <code style='color:#bae6fd;'>save=</code> path."
            "</div></div>"
        )

