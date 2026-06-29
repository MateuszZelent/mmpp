"""DispersionModesBridge – bridge between DispersionResult1D and mode extraction.

Accessed via ``result.modes``.

Usage::

    result = job[0].fft.dispersion.filters(...).compute_1d(axis='x')
    result.modes.interactive(lattice_constant_nm=470)
    result.modes.at(k_rad_um=2.3, f_ghz=5.0).plot.imshow()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .._json import json_safe

if TYPE_CHECKING:
    from ..models import DispersionResult1D


class DispersionModesBridge:
    """Modes namespace on :class:`~mmpp.fft.dispersion.models.DispersionResult1D`.

    This class mirrors the role of ``SpectrumModes`` in the FMR/spectrum
    module – it acts as the glue between the computed dispersion data and the
    mode-extraction engine (``FFTDispersionInterface``).

    Accessed via ``result.modes``.

    Methods
    -------
    interactive(lattice_constant_nm=470)
        Open the interactive dispersion-mode widget.
    at(k_rad_um, f_ghz, z_layer=0)
        Extract a single mode image at a specific (k, f) point.
    plot
        Static plotting accessor for mode images.
    """

    def __init__(self, result: "DispersionResult1D") -> None:
        self._result = result

    # ------------------------------------------------------------------
    # interactive widget
    # ------------------------------------------------------------------

    def interactive(
        self,
        lattice_constant_nm: float = 470.0,
        *,
        figsize: tuple[float, float] = (16, 9),
        kscale: str = "rad_um",
        f_units: str = "GHz",
        fmax: Optional[float] = None,
        lognorm: bool = True,
        component: Optional[str] = None,
        avg_over_orthogonal: bool = False,
        orthogonal_avg_mode: str = "fft_power",
        save: bool = False,
        force: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Launch the interactive dispersion-mode explorer widget.

        This delegates to the existing ``FFTDispersionInterface.dispersion_modes()``
        pipeline, preserving the full interactive experience while allowing the
        new fluent API.

        Parameters
        ----------
        lattice_constant_nm : float
            Lattice constant in nm used for mode reconstruction.
        figsize : (float, float)
            Figure size for the interactive widget.
        kscale : str
            Wave-vector unit scale (``"rad_um"``, ``"rad"``, ``"meter"``).
        f_units : str
            Frequency units (``"GHz"`` or ``"Hz"``).
        fmax : float, optional
            Frequency axis upper limit (in *f_units*).
        lognorm : bool
            Logarithmic color normalisation.
        component : str, optional
            Magnetization component to reconstruct (``None`` = auto).
        avg_over_orthogonal : bool
            Average over the orthogonal spatial direction.
        orthogonal_avg_mode : str
            Averaging method when *avg_over_orthogonal* is True.
        save, force : bool
            Passed to the underlying caching layer.
        **kwargs
            Forwarded to the interactive widget.

        Returns
        -------
        InteractiveDispersionModes widget.
        """
        show = bool(kwargs.pop("show", True))

        from .._interactive_viewer import DispersionInteractiveViewer

        viewer_kwargs = {
            "lattice_constant_nm": lattice_constant_nm,
            "figsize": figsize,
            "kscale": kscale,
            "f_units": f_units,
            "fmax": fmax,
            "lognorm": lognorm,
            "component": component,
            "avg_over_orthogonal": avg_over_orthogonal,
            "orthogonal_avg_mode": orthogonal_avg_mode,
            "save": save,
            "force": force,
            **kwargs,
        }

        if hasattr(self._result, "_interface"):
            iface = self._result._interface  # type: ignore[attr-defined]
            if not show:
                return DispersionInteractiveViewer.from_result(
                    self._result,
                    show=False,
                    can_reconstruct_modes=self._result.S_complex is not None,
                    **viewer_kwargs,
                )

            modes = iface.dispersion_modes(
                result=self._result,
                lattice_constant_nm=lattice_constant_nm,
                save=save,
                force=force,
            )
            modes.plot_interactive(
                result=self._result,
                figsize=figsize,
                lattice_constant_nm=lattice_constant_nm,
                fmax=fmax,
                f_units=f_units,
                lognorm=lognorm,
                **kwargs,
            )
            return modes

        else:
            return DispersionInteractiveViewer.from_result(
                self._result,
                show=show,
                can_reconstruct_modes=False,
                mode_unavailable_reason=(
                    "Mode reconstruction requires S_complex and source FFT context. "
                    "Use .dispersion_modes() on the original filter chain for full mode reconstruction."
                ),
                **viewer_kwargs,
            )

    # ------------------------------------------------------------------
    # single-mode extraction
    # ------------------------------------------------------------------

    def at(
        self,
        k_rad_um: float,
        f_ghz: float,
        *,
        z_layer: int = 0,
        component: Optional[str] = None,
    ) -> "DispersionModeResult":
        """Extract mode image at a specific (k, f) point.

        Parameters
        ----------
        k_rad_um : float
            Target wave-vector [rad/μm].
        f_ghz : float
            Target frequency [GHz].
        z_layer : int
            Z-layer index for 3D simulations.
        component : str, optional
            Magnetization component.

        Returns
        -------
        DispersionModeResult
        """
        if self._result.S_complex is None:
            raise ValueError(
                "No complex spectrum stored – recompute dispersion with "
                "``store_complex=True``."
            )

        k_target = k_rad_um * 1e6  # rad/m
        f_target = f_ghz * 1e9  # Hz

        # Nearest k-bin
        k_axis = self._result.k_axis
        idx_k = int(abs(k_axis - k_target).argmin())

        # Nearest f-bin (positive frequencies only)
        f_axis = self._result.f_axis
        pos_f = f_axis >= 0
        f_axis_pos = f_axis[pos_f]
        idx_f_rel = int(abs(f_axis_pos - f_target).argmin())
        idx_f = int(np.flatnonzero(pos_f)[idx_f_rel])

        # Reconstruct mode image from complex spectrum
        S_c = self._result.S_complex
        if S_c.ndim == 3:
            # (N_orth, Nk, Nf_complex)
            mode_data = S_c[:, idx_k, idx_f]
        else:
            mode_data = S_c[idx_k, idx_f]

        return DispersionModeResult(
            mode_data=mode_data,
            k_rad_um=float(k_axis[idx_k]) / 1e6,
            f_ghz=float(f_axis[idx_f]) / 1e9,
            z_layer=z_layer,
            component=component or self._result.component,
            result=self._result,
        )

    # ------------------------------------------------------------------
    # plot accessor for mode overview
    # ------------------------------------------------------------------

    @property
    def plot(self) -> "DispersionModesPlotAccessor":
        """Static plotting for extracted modes."""
        return DispersionModesPlotAccessor(self)

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "<DispersionModesBridge: "
            ".interactive(lattice_constant_nm=470), "
            ".at(k_rad_um, f_ghz), "
            ".plot>"
        )

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, html_tabs
        from html import escape as _esc

        HV = "onmouseover=\"this.style.background='#1e293b'\" onmouseout=\"this.style.background='transparent'\""

        methods = [
            (
                ".interactive(lattice_constant_nm=470)",
                "Open ipywidgets mode explorer",
                "Launches interactive widget. Click on any point in S(k,f) to reconstruct the spatial "
                "mode profile m(x,y) via inverse FFT. Requires Jupyter + ipywidgets.",
            ),
            (
                ".interactive(lattice_constant_nm=470, lognorm=True, fmax=10)",
                "With log color scale and f-axis clip",
                "lognorm=True uses LogNorm. fmax clips frequency display. "
                "lattice_constant_nm sets initial BZ folding parameter.",
            ),
            (
                ".at(k_rad_um=2.3, f_ghz=5.0)",
                "→ DispersionModeResult",
                "Extract single mode image at the nearest (k, f) bin. "
                "Requires S_complex to be stored (recompute with store_complex=True).",
            ),
            (
                ".at(...).plot.imshow(mode_type='abs')",
                "Mode spatial profile",
                "mode_type: 'abs' (amplitude), 'real', 'imag', 'phase'. Returns (fig, ax).",
            ),
        ]
        row_html = "".join(
            f"<tr {HV} title=\"{_esc(tip)}\" style='cursor:pointer;'>"
            f"<td style='padding:4px 10px;font-family:monospace;color:#fbbf24;font-size:.86em;'>{_esc(sig)}</td>"
            f"<td style='padding:4px 10px;color:#cbd5e1;font-size:.83em;'>{_esc(desc)}</td>"
            f"</tr>"
            for sig, desc, tip in methods
        )
        html = (
            "<div style='font-family:-apple-system,sans-serif;border:2px solid #78350f;"
            "border-radius:10px;padding:12px;margin:6px 0;"
            "background:#0f172a;color:#e2e8f0;max-width:680px;'>"
            "<div style='font-weight:700;color:#f59e0b;margin-bottom:8px;'>"
            "DispersionModesBridge"
            "<span style='font-size:.75em;color:#475569;font-weight:400;margin-left:8px;'>"
            "(hover rows for parameter details)</span></div>"
            f"<table style='width:100%;border-collapse:collapse;'>{row_html}</table>"
            "<div style='margin-top:8px;font-size:.78em;color:#475569;'>"
            "Requires <code style='color:#fcd34d;'>S_complex</code> for "
            "<code style='color:#fcd34d;'>.at()</code>; "
            "<code style='color:#fcd34d;'>_interface</code> back-ref for "
            "<code style='color:#fcd34d;'>.interactive()</code>."
            "</div></div>"
        )
        api_card = api_help_html(
            self,
            title="Dispersion modes bridge API help",
            prefix="disp_result.modes",
            properties=[
                ("plot", "Static plotting accessor for dispersion mode overview")
            ],
            methods=["interactive", "at"],
            subtitle="Live signatures for the modes namespace on DispersionResult1D.",
            chrome=False,
        )
        return (
            f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;">'
            + html_tabs(
                [("Overview", html), ("API", api_card)],
                uid=f"disp-modes-{str(_uuid.uuid4())[:8]}",
            )
            + "</div>"
        )


# ---------------------------------------------------------------------------
# DispersionModeResult  –  a single extracted mode image
# ---------------------------------------------------------------------------


class DispersionModeResult:
    """A single mode image extracted at a specific (k, f) point.

    Attributes
    ----------
    mode_data : np.ndarray
        Complex or real mode data array (spatial profile).
    k_rad_um : float
        Wave-vector of the mode [rad/μm].
    f_ghz : float
        Frequency of the mode [GHz].
    component : str
        Magnetization component.
    result : DispersionResult1D
        Back-reference to full dispersion result.
    """

    def __init__(
        self,
        mode_data: "Any",
        k_rad_um: float,
        f_ghz: float,
        z_layer: int,
        component: str,
        result: "DispersionResult1D",
    ) -> None:
        self.mode_data = mode_data
        self.k_rad_um = k_rad_um
        self.f_ghz = f_ghz
        self.z_layer = z_layer
        self.component = component
        self.result = result

    @property
    def plot(self) -> "DispersionModePlotAccessor":
        """Plotting namespace (.imshow, .phase, .interactive)."""
        return DispersionModePlotAccessor(self)

    def __repr__(self) -> str:
        return (
            f"DispersionModeResult(k={self.k_rad_um:.3f} rad/μm, "
            f"f={self.f_ghz:.3f} GHz, component={self.component!r})"
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        rows = [
            ("k", f"{self.k_rad_um:.3f} rad/μm"),
            ("f", f"{self.f_ghz:.3f} GHz"),
            ("component", self.component),
        ]
        row_html = "".join(
            f"<tr>"
            f"<td style='padding:2px 8px;color:#93c5fd;font-family:monospace;'>{_esc(k)}</td>"
            f"<td style='padding:2px 8px;color:#a5b4fc;'>{_esc(v)}</td>"
            f"</tr>"
            for k, v in rows
        )
        HV = "onmouseover=\"this.style.background='#1e293b'\" onmouseout=\"this.style.background='transparent'\""
        plot_methods = [
            (
                ".plot.imshow(mode_type='abs')",
                "Amplitude |\u03c8(x,y)|",
                "mode_type options: 'abs' (amplitude), 'real', 'imag', 'phase'",
            ),
            (
                ".plot.phase()",
                "Phase \u2220\u03c8(x,y) with hsv colormap",
                "Shortcut for .plot.imshow(mode_type='phase', cmap='hsv')",
            ),
            (
                ".plot.interactive(show=False)",
                "Headless single-mode viewer",
                "Returns a lightweight controller with .state, .show(), .close(), and _repr_html_().",
            ),
        ]
        plot_rows = "".join(
            f"<tr {HV} title=\"{_esc(tip)}\" style='cursor:pointer;'>"
            f"<td style='padding:3px 10px;font-family:monospace;color:#c4b5fd;font-size:.85em;'>{_esc(sig)}</td>"
            f"<td style='padding:3px 10px;color:#94a3b8;font-size:.83em;'>{_esc(desc)}</td>"
            f"</tr>"
            for sig, desc, tip in plot_methods
        )
        return (
            "<div style='font-family:-apple-system,sans-serif;border:2px solid #334155;"
            "border-radius:10px;padding:12px;margin:6px 0;background:#0f172a;"
            "color:#e2e8f0;max-width:600px;'>"
            "<div style='font-weight:700;color:#c4b5fd;margin-bottom:8px;'>DispersionModeResult</div>"
            f"<table style='border-collapse:collapse;margin-bottom:8px;'>{row_html}</table>"
            "<details><summary style='cursor:pointer;font-size:.82em;color:#a78bfa;"
            "list-style:none;padding:2px 4px;' title='Expand to see plot options'>"
            "&#9654; <code>.plot</code> — DispersionModePlotAccessor</summary>"
            "<div style='margin-left:12px;margin-top:4px;'>"
            f"<table style='width:100%;border-collapse:collapse;'>{plot_rows}</table>"
            "</div></details></div>"
        )


# ---------------------------------------------------------------------------
# DispersionModePlotAccessor  –  plot a single DispersionModeResult
# ---------------------------------------------------------------------------


class DispersionModePlotAccessor:
    """Plotting namespace for :class:`DispersionModeResult`."""

    def __init__(self, mode: DispersionModeResult) -> None:
        self._mode = mode

    def imshow(
        self,
        ax: "Optional[Any]" = None,
        *,
        figsize: tuple[float, float] = (6, 5),
        mode_type: str = "abs",
        cmap: str = "RdBu_r",
        title: Optional[str] = None,
    ) -> tuple:
        """Plot mode spatial profile.

        Parameters
        ----------
        mode_type : ``"abs"`` | ``"real"`` | ``"imag"`` | ``"phase"``
            Which part of the complex mode to display.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        mode = self._mode
        data = mode.mode_data

        if mode_type == "abs":
            img = np.abs(data)
        elif mode_type == "real":
            img = np.real(data)
        elif mode_type == "imag":
            img = np.imag(data)
        elif mode_type == "phase":
            img = np.angle(data)
        else:
            img = np.abs(data)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        im = ax.imshow(img, cmap=cmap, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_title(
            title or f"Mode |ψ| @ k={mode.k_rad_um:.2f} rad/μm, f={mode.f_ghz:.2f} GHz"
        )
        ax.set_xlabel("Position x")
        ax.set_ylabel("Position y (layer)")

        try:
            fig.tight_layout()
        except Exception:
            pass

        return fig, ax

    def phase(self, **kwargs) -> tuple:
        """Plot mode phase profile."""
        return self.imshow(mode_type="phase", cmap="hsv", **kwargs)

    def interactive(
        self,
        *,
        show: bool = True,
        mode_type: str = "abs",
        **kwargs: Any,
    ) -> "DispersionSingleModeInteractiveViewer":
        """Return a lightweight interactive controller for this single mode."""
        viewer = DispersionSingleModeInteractiveViewer(
            self._mode,
            show_requested=bool(show),
            mode_type=mode_type,
            options=dict(kwargs),
        )
        if show:
            viewer.show()
        return viewer

    def __repr__(self) -> str:
        return "<DispersionModePlotAccessor: .imshow(...), .phase(...), .interactive()>"

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, html_tabs, plot_accessor_html

        html = plot_accessor_html(
            "DispersionModePlotAccessor",
            [
                (
                    ".imshow(mode_type='abs', cmap='RdBu_r')",
                    "Mode spatial profile |ψ(x,y)|",
                    "mode_type: 'abs', 'real', 'imag', 'phase'. cmap, figsize, title.",
                ),
                (
                    ".interactive(show=False, mode_type='abs')",
                    "Headless single-mode viewer",
                    "Returns a lightweight controller for notebook display, presets, and tests.",
                ),
                (
                    ".phase(**kw)",
                    "Phase ∠ψ(x,y) with hsv colormap",
                    "Shortcut for .imshow(mode_type='phase', cmap='hsv').",
                ),
            ],
        )
        api_card = api_help_html(
            self,
            title="Dispersion mode plot API help",
            prefix="mode.plot",
            methods=["imshow", "phase", "interactive"],
            subtitle="Live signatures for plotting a single extracted dispersion mode.",
            chrome=False,
        )
        return (
            f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;">'
            + html_tabs(
                [("Overview", html), ("API", api_card)],
                uid=f"disp-mode-plot-{str(_uuid.uuid4())[:8]}",
            )
            + "</div>"
        )


class DispersionSingleModeInteractiveViewer:
    """Lightweight notebook controller for one extracted dispersion mode."""

    def __init__(
        self,
        mode: DispersionModeResult,
        *,
        show_requested: bool = True,
        mode_type: str = "abs",
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.mode = mode
        self.show_requested = bool(show_requested)
        self.mode_type = str(mode_type or "abs")
        self.options = dict(options or {})
        self._display_handle: Any = None

    def show(self) -> "DispersionSingleModeInteractiveViewer":
        """Display the lightweight notebook representation when IPython exists."""
        self.show_requested = True
        try:
            from IPython.display import display
        except ImportError:
            return self
        self._display_handle = display(self, display_id=True)
        return self

    def close(self) -> None:
        """Best-effort display cleanup hook."""
        if self._display_handle is not None and hasattr(self._display_handle, "update"):
            self._display_handle.update(None)
        self._display_handle = None
        self.show_requested = False

    @property
    def state(self) -> dict[str, Any]:
        """Serializable state for tests and notebooks."""
        data = self.mode.mode_data
        return {
            "show": self.show_requested,
            "mode_type": self.mode_type,
            "k_rad_um": float(self.mode.k_rad_um),
            "f_ghz": float(self.mode.f_ghz),
            "component": self.mode.component,
            "z_layer": int(self.mode.z_layer),
            "mode_shape": list(np.shape(data)),
            "mode_dtype": str(getattr(data, "dtype", type(data).__name__)),
            "options": json_safe(self.options),
        }

    def export_selection(self, **selection: Any) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of viewer state and selection."""
        return {"viewer": self.state, "selection": json_safe(selection)}

    def _repr_html_(self) -> str:
        from html import escape as _esc

        state = self.state
        rows = [
            ("mode_type", state["mode_type"]),
            ("k", f"{state['k_rad_um']:.3f} rad/μm"),
            ("f", f"{state['f_ghz']:.3f} GHz"),
            ("component", state["component"]),
            ("shape", state["mode_shape"]),
        ]
        rows_html = "".join(
            "<tr>"
            f"<td style='padding:2px 8px;color:#94a3b8;font-size:.85em;'>{_esc(str(key))}</td>"
            f"<td style='padding:2px 8px;color:#c4b5fd;font-family:monospace;'>{_esc(str(value))}</td>"
            "</tr>"
            for key, value in rows
        )
        return (
            "<div style='font-family:-apple-system,sans-serif;border:1px solid #6d28d9;"
            "border-radius:8px;padding:10px;background:#0f172a;color:#e2e8f0;'>"
            "<div style='font-weight:700;color:#c4b5fd;margin-bottom:6px;'>"
            "DispersionSingleModeInteractiveViewer</div>"
            f"<table style='border-collapse:collapse;'>{rows_html}</table>"
            "</div>"
        )


class DispersionModesAnimationViewer:
    """Lightweight controller for overview mode-animation requests."""

    def __init__(
        self,
        bridge: DispersionModesBridge,
        *,
        peaks: Optional[list[Any]] = None,
        show_requested: bool = True,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.bridge = bridge
        self.result = bridge._result
        self.peaks = list(peaks or [])
        self.show_requested = bool(show_requested)
        self.options = dict(options or {})
        self._display_handle: Any = None

    def show(self) -> "DispersionModesAnimationViewer":
        """Display a lightweight notebook representation when IPython exists."""
        self.show_requested = True
        try:
            from IPython.display import display
        except ImportError:
            return self
        self._display_handle = display(self, display_id=True)
        return self

    def close(self) -> None:
        """Best-effort display cleanup hook."""
        if self._display_handle is not None and hasattr(self._display_handle, "update"):
            self._display_handle.update(None)
        self._display_handle = None
        self.show_requested = False

    @property
    def state(self) -> dict[str, Any]:
        """Serializable animation request state."""
        return {
            "show": self.show_requested,
            "peaks": json_safe(self.peaks),
            "axis": self.result.axis,
            "component": self.result.component,
            "result_shape": list(self.result.S.shape),
            "has_complex": self.result.S_complex is not None,
            "has_local": self.result.S_local is not None,
            "options": json_safe(self.options),
        }

    def export_selection(self, **selection: Any) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of viewer state and selection."""
        return {"viewer": self.state, "selection": json_safe(selection)}

    def _repr_html_(self) -> str:
        from html import escape as _esc

        state = self.state
        rows = [
            ("peaks", state["peaks"]),
            ("shape", state["result_shape"]),
            ("complex", state["has_complex"]),
            ("local", state["has_local"]),
        ]
        rows_html = "".join(
            "<tr>"
            f"<td style='padding:2px 8px;color:#94a3b8;font-size:.85em;'>{_esc(str(key))}</td>"
            f"<td style='padding:2px 8px;color:#fbbf24;font-family:monospace;'>{_esc(str(value))}</td>"
            "</tr>"
            for key, value in rows
        )
        return (
            "<div style='font-family:-apple-system,sans-serif;border:1px solid #92400e;"
            "border-radius:8px;padding:10px;background:#0f172a;color:#e2e8f0;'>"
            "<div style='font-weight:700;color:#fbbf24;margin-bottom:6px;'>"
            "DispersionModesAnimationViewer</div>"
            f"<table style='border-collapse:collapse;'>{rows_html}</table>"
            "</div>"
        )


# ---------------------------------------------------------------------------
# DispersionModesPlotAccessor  –  overview plots for all modes
# ---------------------------------------------------------------------------


class DispersionModesPlotAccessor:
    """Static plotting for the full modes namespace (DispersionModesBridge.plot)."""

    def __init__(self, bridge: DispersionModesBridge) -> None:
        self._bridge = bridge

    def animation(
        self,
        peaks: Optional[list[Any]] = None,
        *,
        show: bool = True,
        **kwargs: Any,
    ) -> DispersionModesAnimationViewer:
        """Return a lightweight animation-request controller."""
        viewer = DispersionModesAnimationViewer(
            self._bridge,
            peaks=peaks,
            show_requested=bool(show),
            options=dict(kwargs),
        )
        if show:
            viewer.show()
        return viewer

    def __repr__(self) -> str:
        return "<DispersionModesPlotAccessor: .animation(peaks=[0,1])>"

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, html_tabs, plot_accessor_html

        html = plot_accessor_html(
            "DispersionModesPlotAccessor",
            [
                (
                    ".animation(peaks=[0,1], show=False)",
                    "Headless animation-request controller",
                    "peaks: list of peak indices to animate. Returns a lightweight controller.",
                ),
            ],
        )
        api_card = api_help_html(
            self,
            title="Dispersion modes plot API help",
            prefix="disp_result.modes.plot",
            methods=["animation"],
            subtitle="Live signatures for overview plots on DispersionModesBridge.plot.",
            chrome=False,
        )
        return (
            f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;">'
            + html_tabs(
                [("Overview", html), ("API", api_card)],
                uid=f"disp-modes-plot-{str(_uuid.uuid4())[:8]}",
            )
            + "</div>"
        )
