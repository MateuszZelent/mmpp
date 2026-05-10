"""Comprehensive interactive dashboard for vortex dynamics analysis.

Provides a professional ipywidgets-based UI that integrates all vortex analysis
modules: core tracking, topology, trajectory, spectrum, spectrogram, modes,
events, signals, and the analytical Thiele model.

Usage
-----
    dashboard = vortex.interactive()
    dashboard.show()

    # or directly:
    from mmpp.solitons.vortex.ui import VortexInteractiveDashboard
    db = VortexInteractiveDashboard(vortex_interface)
    db.show()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("mmpp.solitons.vortex.ui")

try:
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display
    from IPython.display import Image as IPyImage

    _HAS_WIDGETS = True
except ImportError:
    _HAS_WIDGETS = False
    widgets = None  # type: ignore

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# CSS / HTML helpers
# ---------------------------------------------------------------------------

_CSS = """
<style>
.vdash-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    color: #e94560;
    padding: 12px 18px;
    border-radius: 10px 10px 0 0;
    font-family: 'Segoe UI', monospace;
    font-size: 18px;
    font-weight: bold;
    letter-spacing: 1px;
    border-bottom: 2px solid #e94560;
}
.vdash-header span.sub {
    font-size: 11px;
    color: #a8b2d8;
    font-weight: normal;
    margin-left: 12px;
    letter-spacing: 0.5px;
}
.vdash-jobinfo {
    background: #0d1117;
    color: #58a6ff;
    padding: 6px 12px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 11px;
    margin-bottom: 4px;
    border: 1px solid #21262d;
}
.vdash-section {
    color: #e6edf3;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 0 2px 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 4px;
    font-family: 'Segoe UI', sans-serif;
}
.vdash-status-ok    { color: #3fb950; font-size: 11px; font-family: monospace; padding: 3px 8px; }
.vdash-status-warn  { color: #d29922; font-size: 11px; font-family: monospace; padding: 3px 8px; }
.vdash-status-error { color: #f85149; font-size: 11px; font-family: monospace; padding: 3px 8px; }
.vdash-status-info  { color: #79c0ff; font-size: 11px; font-family: monospace; padding: 3px 8px; }
</style>
"""

_PANEL_LAYOUT = {
    "width": "290px",
    "min_width": "290px",
    "padding": "6px",
    "border": "1px solid #21262d",
    "background_color": "#0d1117",
    "overflow_y": "auto",
}
_CTRL_STYLE = {"description_width": "110px"}
_CTRL_LAYOUT = widgets.Layout(width="270px") if _HAS_WIDGETS else None
_BTN_LAYOUT = widgets.Layout(width="270px", margin="4px 0px") if _HAS_WIDGETS else None
_OUTPUT_LAYOUT = {
    "min_width": "700px",
    "flex": "1",
    "border": "1px solid #21262d",
}


def _slider(desc, val, lo, hi, step, **kw):
    style = dict(_CTRL_STYLE)
    style.update(kw.pop("style", {}))
    return widgets.FloatSlider(
        description=desc,
        value=val,
        min=lo,
        max=hi,
        step=step,
        continuous_update=False,
        style=style,
        layout=widgets.Layout(width="270px"),
        **kw,
    )


def _int_slider(desc, val, lo, hi, step=1, **kw):
    style = dict(_CTRL_STYLE)
    style.update(kw.pop("style", {}))
    return widgets.IntSlider(
        description=desc,
        value=val,
        min=lo,
        max=hi,
        step=step,
        continuous_update=False,
        style=style,
        layout=widgets.Layout(width="270px"),
        **kw,
    )


def _dropdown(desc, options, value=None, **kw):
    style = dict(_CTRL_STYLE)
    style.update(kw.pop("style", {}))

    # When options are (label, value) tuples, extract the raw value
    def _extract(v):
        return v[1] if isinstance(v, tuple) and len(v) == 2 else v

    if value is None:
        resolved = _extract(options[0]) if options else None
    else:
        resolved = _extract(value)

    return widgets.Dropdown(
        description=desc,
        options=options,
        value=resolved,
        style=style,
        layout=widgets.Layout(width="270px"),
        **kw,
    )


def _checkbox(desc, value=True, **kw):
    return widgets.Checkbox(
        description=desc,
        value=value,
        indent=False,
        layout=widgets.Layout(width="270px"),
        **kw,
    )


def _btn(desc, style="primary", icon=""):
    return widgets.Button(
        description=desc,
        button_style=style,
        icon=icon,
        layout=widgets.Layout(width="270px", height="30px"),
    )


def _section(text):
    return widgets.HTML(f"<div class='vdash-section'>{text}</div>")


# ---------------------------------------------------------------------------
# Dashboard state container
# ---------------------------------------------------------------------------


@dataclass
class _DashboardState:
    """Cached computation results shared between tabs."""

    trajectory: Any = None
    topology_result: Any = None
    orbit_fit: Any = None
    gyration_spectrum: Any = None
    breathing_spectrum: Any = None
    spectrogram: Any = None
    mode_result: Any = None
    events_result: Any = None
    signals_result: Any = None
    nonlinear_result: Any = None


# ---------------------------------------------------------------------------
# Main dashboard class
# ---------------------------------------------------------------------------


class VortexInteractiveDashboard:
    """Professional interactive dashboard for vortex dynamics analysis.

    Integrates all vortex analysis modules in a single beautiful ipywidgets UI.

    Parameters
    ----------
    vortex_interface : VortexInterface
        The vortex analysis namespace (``job[i].solitons.vortex``).
    figsize : tuple
        Figure size for plot panels (width, height).
    dpi : int
        Plot resolution.
    """

    def __init__(self, vortex_interface, figsize=(10, 7), dpi=100):
        if not _HAS_WIDGETS:
            raise ImportError("ipywidgets is required: pip install ipywidgets")
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required: pip install matplotlib")

        self._vx = vortex_interface
        self.figsize = figsize
        self.dpi = dpi
        self._state = _DashboardState()
        self._fig: Figure | None = None
        self._output: Any = None
        self._status: Any = None
        self._health_widget: Any = None  # HTML widget showing health status
        self._controls: dict[str, Any] = {}
        self._display_handle: Any = None
        self._css_displayed = False
        self._built = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self) -> Any:
        """Build and display the interactive dashboard."""
        self._build()
        plt.ioff()  # prevent %matplotlib widget from auto-displaying figures to cell output
        if not self._css_displayed:
            display(HTML(_CSS))
            self._css_displayed = True
        if self._display_handle is None:
            self._display_handle = display(self._root, display_id=True)
        else:
            self._display_handle.update(self._root)
        return self

    # ------------------------------------------------------------------
    # Build layout
    # ------------------------------------------------------------------

    def _build(self):
        if self._built:
            return
        self._built = True

        # Global output widget (right panel)
        self._output = widgets.Output(layout=widgets.Layout(**_OUTPUT_LAYOUT))
        # Status bar
        self._status = widgets.HTML(
            value=self._fmt_status("Ready — select a module and click Run", "info")
        )
        # Health status banner (updated on each computation)
        self._health_widget = widgets.HTML(value="")

        # ---- Assemble tabs --------------------------------------------------
        tab_items = [
            ("🎯 Core", self._build_tab_core()),
            ("🌀 Topology", self._build_tab_topology()),
            ("📐 Trajectory", self._build_tab_trajectory()),
            ("📊 Spectrum", self._build_tab_spectrum()),
            ("🌈 Spectrogram", self._build_tab_spectrogram()),
            ("🎭 Modes", self._build_tab_modes()),
            ("⚡ Events", self._build_tab_events()),
            ("📡 Signals", self._build_tab_signals()),
            ("🔬 Thiele", self._build_tab_thiele()),
        ]

        tab = widgets.Tab(
            children=[item[1] for item in tab_items],
            layout=widgets.Layout(width="290px"),
        )
        for i, (title, _) in enumerate(tab_items):
            tab.set_title(i, title)

        # ---- Left panel (info + presets + tabs) -----------------------------
        job_info = self._build_job_info()
        preset_row = self._build_preset_row()

        left_panel = widgets.VBox(
            [
                job_info,
                self._health_widget,
                preset_row,
                tab,
            ],
            layout=widgets.Layout(**_PANEL_LAYOUT),
        )

        # ---- Main HBox ------------------------------------------------------
        main_area = widgets.HBox(
            [left_panel, self._output],
            layout=widgets.Layout(width="100%", align_items="stretch"),
        )

        # ---- Header ---------------------------------------------------------
        header = widgets.HTML(
            "<div class='vdash-header'>"
            "🌀 Vortex Dynamics <em>Interactive Dashboard</em>"
            "<span class='sub'>All modules · mmpp</span>"
            "</div>"
        )

        self._root = widgets.VBox(
            [header, main_area, self._status],
            layout=widgets.Layout(width="100%"),
        )

    # ------------------------------------------------------------------
    # Job info widget
    # ------------------------------------------------------------------

    def _build_job_info(self):
        try:
            job = self._vx._job
            path = getattr(job, "zarr_path", getattr(job, "_path", "?"))
            dset = self._vx.dataset_name or "auto"
            info_html = (
                f"<div class='vdash-jobinfo'>"
                f"📂 {str(path)[-42:]}<br>"
                f"📦 dataset: <b>{dset}</b>"
                f"</div>"
            )
        except Exception:
            info_html = "<div class='vdash-jobinfo'>Job: —</div>"
        return widgets.HTML(info_html)

    # ------------------------------------------------------------------
    # Preset row
    # ------------------------------------------------------------------

    def _build_preset_row(self):
        self._w_preset_name = widgets.Text(
            placeholder="preset name",
            layout=widgets.Layout(width="155px"),
        )
        self._w_preset_save = widgets.Button(
            description="💾",
            button_style="",
            layout=widgets.Layout(width="40px"),
            tooltip="Save preset",
        )
        self._w_preset_load = widgets.Dropdown(
            options=["— load preset —"],
            layout=widgets.Layout(width="270px"),
        )
        self._w_preset_save.on_click(self._on_save_preset)
        self._w_preset_load.observe(self._on_load_preset, names="value")
        row = widgets.VBox(
            [
                widgets.HBox([self._w_preset_name, self._w_preset_save]),
                self._w_preset_load,
            ]
        )
        self._refresh_preset_list()
        return row

    # ------------------------------------------------------------------
    # ───────────────────────── TAB: CORE ──────────────────────────────
    # ------------------------------------------------------------------

    def _build_tab_core(self):
        c = {}
        c["threshold"] = _slider(
            "Threshold",
            0.5,
            0.0,
            1.0,
            0.01,
            description_tooltip="Skyrmion-number threshold for core detection",
        )
        c["method"] = _dropdown(
            "Method",
            [("Center-of-mass", "com"), ("Peak", "peak"), ("Fit", "fit")],
            ("Center-of-mass", "com"),
        )
        c["t_start"] = _int_slider(
            "t start [idx]",
            0,
            0,
            5000,
            description_tooltip="First time step to include",
        )
        c["t_end"] = _int_slider(
            "t end [idx]", 0, 0, 5000, description_tooltip="Last time step (0 = all)"
        )
        c["component"] = _dropdown(
            "Component", [("mz (default)", "z"), ("mx", "x"), ("my", "y")]
        )
        c["smooth"] = _checkbox("Smooth trajectory", True)
        c["smooth_window"] = _int_slider("Smooth window", 5, 1, 51, 2)
        c["show_orbit"] = _checkbox("Show orbit", True)
        c["cmap"] = _dropdown(
            "Colormap", ["viridis", "plasma", "inferno", "cividis", "turbo"]
        )

        btn = _btn("▶  Run Core Tracking", "success", "play")
        btn.on_click(lambda _: self._run_core(c))

        self._controls["core"] = c
        return widgets.VBox(
            [
                _section("📌 Detection Parameters"),
                c["threshold"],
                c["method"],
                c["component"],
                _section("🕐 Time Range"),
                c["t_start"],
                c["t_end"],
                _section("✨ Visualization"),
                c["smooth"],
                c["smooth_window"],
                c["show_orbit"],
                c["cmap"],
                btn,
            ]
        )

    # ------------------------------------------------------------------
    # ─────────────────────── TAB: TOPOLOGY ────────────────────────────
    # ------------------------------------------------------------------

    def _build_tab_topology(self):
        c = {}
        c["component"] = _dropdown("Component", [("mz", "z"), ("mx", "x"), ("my", "y")])
        c["t_index"] = _int_slider(
            "Frame index", 0, 0, 5000, description_tooltip="Time frame to analyze"
        )
        c["threshold"] = _slider("Threshold", 0.5, 0.01, 1.0, 0.01)
        c["method"] = _dropdown(
            "Topo. method", [("Discrete", "discrete"), ("Continuous", "continuous")]
        )
        c["cmap"] = _dropdown(
            "Colormap", ["RdBu_r", "seismic", "coolwarm", "bwr", "twilight"]
        )
        c["show_charge"] = _checkbox("Annotate charge", True)

        btn = _btn("▶  Detect Topology", "success", "play")
        btn.on_click(lambda _: self._run_topology(c))

        self._controls["topology"] = c
        return widgets.VBox(
            [
                _section("🔬 Detection"),
                c["component"],
                c["t_index"],
                c["threshold"],
                c["method"],
                _section("✨ Display"),
                c["cmap"],
                c["show_charge"],
                btn,
            ]
        )

    # ------------------------------------------------------------------
    # ─────────────────────── TAB: TRAJECTORY ──────────────────────────
    # ------------------------------------------------------------------

    def _build_tab_trajectory(self):
        c = {}
        c["fit_orbit"] = _checkbox("Fit orbit (ellipse)", True)
        c["detrend"] = _checkbox("Detrend (remove drift)", False)
        c["show_velocity"] = _checkbox("Show velocity", False)
        c["show_stats"] = _checkbox("Show statistics", True)
        c["color_by"] = _dropdown(
            "Color by",
            [
                ("time", "time"),
                ("speed", "speed"),
                ("radius", "radius"),
                ("none", "none"),
            ],
        )
        c["cmap"] = _dropdown(
            "Colormap", ["plasma", "viridis", "inferno", "magma", "turbo"]
        )

        btn = _btn("▶  Analyze Trajectory", "success", "play")
        btn.on_click(lambda _: self._run_trajectory(c))

        self._controls["trajectory"] = c
        return widgets.VBox(
            [
                _section("📐 Analysis Options"),
                c["fit_orbit"],
                c["detrend"],
                _section("✨ Visualization"),
                c["show_velocity"],
                c["show_stats"],
                c["color_by"],
                c["cmap"],
                btn,
            ]
        )

    # ------------------------------------------------------------------
    # ─────────────────────── TAB: SPECTRUM ────────────────────────────
    # ------------------------------------------------------------------

    def _build_tab_spectrum(self):
        c = {}
        c["mode"] = _dropdown(
            "Spectrum",
            [("Gyration", "gyration"), ("Breathing", "breathing"), ("Both", "both")],
        )
        c["method"] = _dropdown(
            "PSD method", [("Welch", "welch"), ("Periodogram", "periodogram")]
        )
        c["nperseg"] = _int_slider("nperseg", 256, 32, 4096, 32)
        c["f_max_ghz"] = _slider("f max [GHz]", 10.0, 0.1, 50.0, 0.1)
        c["log_scale"] = _checkbox("Log Y scale", False)
        c["show_peaks"] = _checkbox("Show peaks", True)
        c["peak_prom"] = _slider("Peak prominence", 0.05, 0.0, 1.0, 0.01)
        c["normalize"] = _checkbox("Normalize", True)

        btn = _btn("▶  Compute Spectrum", "success", "play")
        btn.on_click(lambda _: self._run_spectrum(c))

        self._controls["spectrum"] = c
        return widgets.VBox(
            [
                _section("📊 Spectral Analysis"),
                c["mode"],
                c["method"],
                c["nperseg"],
                _section("🎛️ Display"),
                c["f_max_ghz"],
                c["log_scale"],
                c["show_peaks"],
                c["peak_prom"],
                c["normalize"],
                btn,
            ]
        )

    # ------------------------------------------------------------------
    # ─────────────────────── TAB: SPECTROGRAM ─────────────────────────
    # ------------------------------------------------------------------

    def _build_tab_spectrogram(self):
        c = {}
        c["component"] = _dropdown(
            "Signal", [("Radius", "radius"), ("X", "x"), ("Y", "y"), ("Angle", "angle")]
        )
        c["method"] = _dropdown("Method", [("STFT", "stft"), ("Welch", "welch")])
        c["nperseg"] = _int_slider("nperseg", 128, 32, 1024, 32)
        c["noverlap_pct"] = _int_slider("Overlap %", 75, 0, 99, 5)
        c["f_max_ghz"] = _slider("f max [GHz]", 10.0, 0.1, 50.0, 0.1)
        c["cmap"] = _dropdown(
            "Colormap", ["viridis", "plasma", "inferno", "hot", "turbo"]
        )
        c["log_power"] = _checkbox("Log power", True)

        btn = _btn("▶  Compute Spectrogram", "success", "play")
        btn.on_click(lambda _: self._run_spectrogram(c))

        self._controls["spectrogram"] = c
        return widgets.VBox(
            [
                _section("🌈 Spectrogram Config"),
                c["component"],
                c["method"],
                c["nperseg"],
                c["noverlap_pct"],
                _section("🎛️ Display"),
                c["f_max_ghz"],
                c["cmap"],
                c["log_power"],
                btn,
            ]
        )

    # ------------------------------------------------------------------
    # ─────────────────────── TAB: MODES ───────────────────────────────
    # ------------------------------------------------------------------

    def _build_tab_modes(self):
        c = {}
        c["n_modes"] = _int_slider("Num. modes", 3, 1, 12)
        c["component"] = _dropdown(
            "Component", [("mz", "z"), ("mx", "x"), ("my", "y"), ("|m|", "amplitude")]
        )
        c["z_layer"] = _int_slider("Z layer", -1, -20, 20)
        c["cmap_real"] = _dropdown("Cmap (real)", ["RdBu_r", "seismic", "coolwarm"])
        c["cmap_amp"] = _dropdown("Cmap (amp)", ["viridis", "plasma", "hot", "inferno"])
        c["normalize_modes"] = _checkbox("Normalize modes", True)
        c["show_colorbar"] = _checkbox("Show colorbars", True)

        btn = _btn("▶  Compute Modes", "success", "play")
        btn.on_click(lambda _: self._run_modes(c))

        self._controls["modes"] = c
        return widgets.VBox(
            [
                _section("🎭 Mode Parameters"),
                c["n_modes"],
                c["component"],
                c["z_layer"],
                _section("✨ Display"),
                c["cmap_real"],
                c["cmap_amp"],
                c["normalize_modes"],
                c["show_colorbar"],
                btn,
            ]
        )

    # ------------------------------------------------------------------
    # ─────────────────────── TAB: EVENTS ──────────────────────────────
    # ------------------------------------------------------------------

    def _build_tab_events(self):
        c = {}
        c["event_type"] = _dropdown(
            "Event type",
            [
                ("All", "all"),
                ("Annihilation", "annihilation"),
                ("Creation", "creation"),
                ("Switching", "switching"),
            ],
        )
        c["threshold"] = _slider("Threshold", 0.5, 0.0, 1.0, 0.01)
        c["min_duration"] = _int_slider("Min duration [fs]", 10, 1, 500)
        c["show_on_trajectory"] = _checkbox("Overlay on trajectory", True)

        btn = _btn("▶  Detect Events", "success", "play")
        btn.on_click(lambda _: self._run_events(c))

        self._controls["events"] = c
        return widgets.VBox(
            [
                _section("⚡ Event Detection"),
                c["event_type"],
                c["threshold"],
                c["min_duration"],
                _section("✨ Display"),
                c["show_on_trajectory"],
                btn,
            ]
        )

    # ------------------------------------------------------------------
    # ─────────────────────── TAB: SIGNALS ─────────────────────────────
    # ------------------------------------------------------------------

    def _build_tab_signals(self):
        c = {}
        c["signal_type"] = _dropdown(
            "Signal type",
            [("TMR proxy", "tmr"), ("MR proxy", "mr"), ("Voltage", "voltage")],
        )
        c["polarizer_angle"] = _slider("Polarizer angle [°]", 0.0, -180.0, 180.0, 5.0)
        c["chirality"] = _dropdown("Chirality", [("CCW (+1)", 1), ("CW (-1)", -1)])
        c["show_psd"] = _checkbox("Show PSD", True)
        c["method"] = _dropdown("PSD method", [("Welch", "welch"), ("FFT", "fft")])
        c["f_max_ghz"] = _slider("f max [GHz]", 10.0, 0.1, 50.0, 0.1)

        btn = _btn("▶  Compute Signal", "success", "play")
        btn.on_click(lambda _: self._run_signals(c))

        self._controls["signals"] = c
        return widgets.VBox(
            [
                _section("📡 Signal Parameters"),
                c["signal_type"],
                c["polarizer_angle"],
                c["chirality"],
                _section("📊 Power Spectral Density"),
                c["show_psd"],
                c["method"],
                c["f_max_ghz"],
                btn,
            ]
        )

    # ------------------------------------------------------------------
    # ─────────────────────── TAB: THIELE ──────────────────────────────
    # ------------------------------------------------------------------

    def _build_tab_thiele(self):
        c = {}
        c["model"] = _dropdown(
            "Model", [("CPP (Slonczewski)", "CPP"), ("CIP (Zhang-Li)", "CIP")]
        )
        c["diameter_nm"] = _slider("Diameter [nm]", 250.0, 50.0, 1000.0, 5.0)
        c["thickness_nm"] = _slider("Thickness [nm]", 10.0, 2.0, 100.0, 0.5)
        c["ms_kA"] = _slider("M_s [kA/m]", 800.0, 100.0, 2000.0, 10.0)
        c["alpha"] = widgets.FloatLogSlider(
            description="Damping α",
            value=0.01,
            base=10,
            min=-4,
            max=-1,
            step=0.01,
            continuous_update=False,
            style=_CTRL_STYLE,
            layout=widgets.Layout(width="270px"),
        )
        c["current_mA"] = _slider("Current I [mA]", 6.0, -30.0, 30.0, 0.1)
        c["polarization"] = _slider("Polarization P", 0.3, 0.0, 1.0, 0.01)
        c["bz_mT"] = _slider("B_z [mT]", 0.0, -500.0, 500.0, 5.0)
        c["fit_to_data"] = _checkbox("Fit to simulation data", True)
        c["show_psd"] = _checkbox("Show PSD comparison", True)

        btn_thiele = _btn("▶  Run Thiele Dashboard", "warning", "cog")
        btn_thiele.on_click(lambda _: self._run_thiele_full())

        btn_quick = _btn("▶  Quick Trajectory Fit", "success", "play")
        btn_quick.on_click(lambda _: self._run_thiele_quick(c))

        self._controls["thiele"] = c
        return widgets.VBox(
            [
                _section("🔬 Geometry & Material"),
                c["model"],
                c["diameter_nm"],
                c["thickness_nm"],
                c["ms_kA"],
                c["alpha"],
                _section("⚡ STT & Field"),
                c["current_mA"],
                c["polarization"],
                c["bz_mT"],
                _section("🔧 Analysis"),
                c["fit_to_data"],
                c["show_psd"],
                btn_quick,
                btn_thiele,
            ]
        )

    # ------------------------------------------------------------------
    # ─────────────────── COMPUTE HANDLERS ─────────────────────────────
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, kind: str = "info"):
        if self._status is not None:
            self._status.value = self._fmt_status(msg, kind)

    @staticmethod
    def _fmt_status(msg: str, kind: str = "info") -> str:
        return f"<div class='vdash-status-{kind}'>● {msg}</div>"

    def _clear_plot(self):
        with self._output:
            clear_output(wait=True)

    def _get_health(self, force: bool = False):
        """Return cached CoreHealthStatus, running the check on first call."""
        try:
            status = self._vx.check_health(force=force)
            self._update_health_widget(status)
            return status
        except Exception:
            return None

    def _update_health_widget(self, status) -> None:
        """Update the health banner HTML widget."""
        if self._health_widget is None:
            return
        if status is None:
            self._health_widget.value = ""
            return
        if status.is_healthy:
            html = (
                "<div style='background:#052e16;border:1px solid #166534;"
                "border-radius:5px;padding:3px 8px;margin:2px 0;"
                "font-family:monospace;font-size:10px;color:#86efac;'>"
                "&#10003; Core health: OK"
                "</div>"
            )
        else:
            problems = ", ".join(status.warnings)
            html = (
                "<div style='background:#431407;border:1px solid #f97316;"
                "border-radius:5px;padding:4px 8px;margin:2px 0;"
                "font-family:monospace;font-size:10px;color:#fdba74;"
                "word-break:break-word;'>"
                f"&#9888; {problems}"
                "</div>"
            )
        self._health_widget.value = html

    def _show_figure(self, fig, health=None):
        """Render figure to PNG and display in Output widget.

        Using PNG avoids all ipympl canvas-widget duplication issues
        regardless of the active matplotlib backend.

        Parameters
        ----------
        fig : matplotlib Figure
        health : CoreHealthStatus or None
            When provided and not healthy, a warning annotation is drawn on
            the figure before it is rasterised.
        """
        import io

        # Attach health warning annotation to the first axes of the figure
        if health is not None and not health.is_healthy:
            try:
                if fig.axes:
                    health.warn_on_plot(fig.axes[0])
            except Exception:
                pass

        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        buf.seek(0)
        img_data = buf.read()
        plt.close(fig)
        if self._fig is not None:
            try:
                plt.close(self._fig)
            except Exception:
                pass
            self._fig = None
        with self._output:
            clear_output(wait=True)
            display(IPyImage(data=img_data, format="png"))

    # ---- CORE -------------------------------------------------------

    def _run_core(self, c: dict):
        self._set_status("Computing core trajectory…", "info")
        try:
            traj = self._vx.core.track()
            self._state.trajectory = traj

            # Check simulation health (annihilation / boundary collision)
            health = self._get_health()

            t_start_idx = int(c["t_start"].value) or None
            t_end_idx = int(c["t_end"].value) or None
            sl = (
                slice(t_start_idx, t_end_idx)
                if (t_start_idx or t_end_idx)
                else slice(None)
            )

            fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
            ax_xy, ax_t = axes

            x_nm = np.asarray(traj.x)[sl] * 1e9
            y_nm = np.asarray(traj.y)[sl] * 1e9
            t_ns = np.asarray(traj.time)[sl] * 1e9

            cmap_name = c["cmap"].value
            color_by_time = t_ns if c["show_orbit"].value else None

            if color_by_time is not None:
                sc = ax_xy.scatter(
                    x_nm, y_nm, c=t_ns, cmap=cmap_name, s=2, lw=0, alpha=0.8
                )
                fig.colorbar(sc, ax=ax_xy, label="t [ns]", shrink=0.8)
            else:
                ax_xy.plot(x_nm, y_nm, "o", ms=1.5, alpha=0.6, color="#58a6ff")

            ax_xy.set_xlabel("x [nm]", fontsize=10)
            ax_xy.set_ylabel("y [nm]", fontsize=10)
            ax_xy.set_title("Core trajectory (xy)", fontsize=11)
            ax_xy.set_aspect("equal")
            ax_xy.grid(True, alpha=0.25)

            if c["smooth"].value:
                w = max(int(c["smooth_window"].value) | 1, 3)
                from scipy.signal import savgol_filter

                try:
                    x_s = savgol_filter(x_nm, w, 2)
                    y_s = savgol_filter(y_nm, w, 2)
                    ax_xy.plot(
                        x_s,
                        y_s,
                        "-",
                        lw=1.2,
                        color="#e94560",
                        alpha=0.9,
                        label="smoothed",
                    )
                    ax_xy.legend(fontsize=8)
                except Exception:
                    pass

            r_nm = np.sqrt(x_nm**2 + y_nm**2)
            ax_t.plot(t_ns, r_nm, lw=1.2, color="#58a6ff")
            ax_t.set_xlabel("t [ns]", fontsize=10)
            ax_t.set_ylabel("r [nm]", fontsize=10)
            ax_t.set_title("Core radius vs time", fontsize=11)
            ax_t.grid(True, alpha=0.25)

            fig.suptitle("🎯 Core Tracking", fontsize=12, color="#e94560")
            self._show_figure(fig, health=health)
            self._set_status(
                f"Core tracked: {len(x_nm)} steps, ⟨r⟩ = {np.mean(r_nm):.1f} nm", "ok"
            )
        except Exception as exc:
            self._set_status(f"Core tracking failed: {exc}", "error")
            log.exception("Core tracking error")

    # ---- TOPOLOGY ---------------------------------------------------

    def _run_topology(self, c: dict):
        self._set_status("Computing topology…", "info")
        try:
            health = self._get_health()
            t_idx = int(c["t_index"].value)
            result = self._vx.topology.detect(frame=t_idx)
            self._state.topology_result = result

            polarity = int(getattr(result, "polarity", 0))
            vorticity = int(getattr(result, "vorticity", 0))
            chirality = int(getattr(result, "chirality", 0))
            Q = float(getattr(result, "Q", 0.0))
            confidence = float(getattr(result, "chirality_confidence", 0.0))

            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.axis("off")

            summary_text = (
                f"Frame:          {t_idx}\n"
                f"Polarity (p):   {polarity:+d}\n"
                f"Vorticity (c):  {vorticity:+d}\n"
                f"Chirality:      {chirality:+d}\n"
                f"Topological Q:  {Q:.4f}\n"
                f"Chirality conf: {confidence:.3f}"
            )
            ax.text(
                0.5,
                0.5,
                summary_text,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                family="monospace",
                color="#e2e8f0",
                bbox={
                    "boxstyle": "round,pad=1.0",
                    "facecolor": "#1e293b",
                    "edgecolor": "#334155",
                    "linewidth": 2,
                },
            )
            ax.set_title(
                f"🌀 Vortex Topology  ·  frame {t_idx}  ·  Q = {Q:.4f}",
                fontsize=12,
                color="#e94560",
            )
            self._show_figure(fig, health=health)
            self._set_status(
                f"Topology | p={polarity:+d}, c={vorticity:+d}, Q={Q:.4f}", "ok"
            )
        except Exception as exc:
            self._set_status(f"Topology failed: {exc}", "error")
            log.exception("Topology error")

    # ---- TRAJECTORY -------------------------------------------------

    def _run_trajectory(self, c: dict):
        self._set_status("Analyzing trajectory…", "info")
        try:
            health = self._get_health()
            traj = self._state.trajectory
            if traj is None:
                traj = self._vx.core.track()
                self._state.trajectory = traj

            x_nm = np.asarray(traj.x) * 1e9
            y_nm = np.asarray(traj.y) * 1e9
            t_ns = np.asarray(traj.time) * 1e9

            color_by = c["color_by"].value
            cmap = c["cmap"].value

            fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
            ax_xy, ax_r = axes

            if color_by == "time":
                clr = t_ns
                clabel = "t [ns]"
            elif color_by == "speed":
                vx = np.gradient(x_nm, t_ns)
                vy = np.gradient(y_nm, t_ns)
                clr = np.sqrt(vx**2 + vy**2)
                clabel = "speed [nm/ns]"
            elif color_by == "radius":
                clr = np.sqrt(x_nm**2 + y_nm**2)
                clabel = "r [nm]"
            else:
                clr = None
                clabel = ""

            if clr is not None:
                sc = ax_xy.scatter(x_nm, y_nm, c=clr, cmap=cmap, s=2, lw=0, alpha=0.8)
                fig.colorbar(sc, ax=ax_xy, label=clabel, shrink=0.8)
            else:
                ax_xy.plot(x_nm, y_nm, ".", ms=1.5, alpha=0.5, color="#58a6ff")

            if c["fit_orbit"].value:
                try:
                    orbit = self._vx.trajectory.orbit.fit()
                    self._state.orbit_fit = orbit
                    theta = np.linspace(0, 2 * np.pi, 360)
                    cx = float(orbit.center[0]) * 1e9
                    cy = float(orbit.center[1]) * 1e9
                    a_nm = float(getattr(orbit, "semi_major", 50.0)) * 1e9
                    b_nm = float(getattr(orbit, "semi_minor", a_nm)) * 1e9
                    angle = float(getattr(orbit, "tilt_angle", 0.0))
                    xe = (
                        cx
                        + a_nm * np.cos(theta) * np.cos(angle)
                        - b_nm * np.sin(theta) * np.sin(angle)
                    )
                    ye = (
                        cy
                        + a_nm * np.cos(theta) * np.sin(angle)
                        + b_nm * np.sin(theta) * np.cos(angle)
                    )
                    ax_xy.plot(xe, ye, "--", lw=2, color="#e94560", label="orbit fit")
                    ax_xy.legend(fontsize=8)
                except Exception:
                    pass

            ax_xy.set_aspect("equal")
            ax_xy.set_xlabel("x [nm]")
            ax_xy.set_ylabel("y [nm]")
            ax_xy.set_title("Core trajectory")
            ax_xy.grid(True, alpha=0.25)

            r_nm = np.sqrt(x_nm**2 + y_nm**2)
            ax_r.plot(t_ns, r_nm, lw=1.2, color="#58a6ff", label="radius")
            if c["show_velocity"].value:
                vr = np.abs(np.gradient(r_nm, t_ns))
                ax2 = ax_r.twinx()
                ax2.plot(t_ns, vr, lw=1.0, color="#e94560", alpha=0.7, label="vr")
                ax2.set_ylabel("|dr/dt| [nm/ns]", color="#e94560")
            ax_r.set_xlabel("t [ns]")
            ax_r.set_ylabel("r [nm]")
            ax_r.set_title("Orbit radius")
            ax_r.grid(True, alpha=0.25)

            if c["show_stats"].value:
                stats = (
                    f"N = {len(x_nm)}  |  "
                    f"⟨r⟩ = {np.mean(r_nm):.1f} nm  |  "
                    f"max(r) = {np.max(r_nm):.1f} nm"
                )
                fig.suptitle(f"📐 Trajectory  ·  {stats}", fontsize=10, color="#a8b2d8")

            self._show_figure(fig, health=health)
            self._set_status(
                f"Trajectory done | ⟨r⟩={np.mean(r_nm):.1f} nm, max={np.max(r_nm):.1f} nm",
                "ok",
            )
        except Exception as exc:
            self._set_status(f"Trajectory failed: {exc}", "error")
            log.exception("Trajectory error")

    # ---- SPECTRUM ---------------------------------------------------

    def _run_spectrum(self, c: dict):
        self._set_status("Computing spectrum…", "info")
        try:
            health = self._get_health()
            traj = self._state.trajectory
            if traj is None:
                traj = self._vx.core.track()
                self._state.trajectory = traj

            method = c["method"].value
            nperseg = int(c["nperseg"].value)
            mode = c["mode"].value

            specs = {}
            if mode in ("gyration", "both"):
                specs["gyration"] = self._vx.spectrum.gyration(
                    method=method, trajectory=traj, nperseg=nperseg
                )
                self._state.gyration_spectrum = specs["gyration"]
            if mode in ("breathing", "both"):
                specs["breathing"] = self._vx.spectrum.breathing(
                    method=method, trajectory=traj, nperseg=nperseg
                )
                self._state.breathing_spectrum = specs["breathing"]

            n = len(specs)
            fig, axes = plt.subplots(
                1, n, figsize=self.figsize, dpi=self.dpi, squeeze=False
            )

            colors = {"gyration": "#58a6ff", "breathing": "#3fb950"}
            f_max = float(c["f_max_ghz"].value)

            for idx, (name, spec) in enumerate(specs.items()):
                ax = axes[0, idx]
                f_hz = np.asarray(spec.frequencies)
                psd = np.asarray(spec.power if hasattr(spec, "power") else spec.psd)
                f_ghz = f_hz / 1e9

                mask = f_ghz <= f_max
                f_p = f_ghz[mask]
                p_p = psd[mask]

                if c["normalize"].value and p_p.max() > 0:
                    p_p = p_p / p_p.max()

                if c["log_scale"].value:
                    ax.semilogy(
                        f_p, p_p + 1e-12, color=colors.get(name, "#e94560"), lw=1.4
                    )
                else:
                    ax.plot(f_p, p_p, color=colors.get(name, "#e94560"), lw=1.4)
                    ax.fill_between(
                        f_p, p_p, alpha=0.2, color=colors.get(name, "#e94560")
                    )

                if c["show_peaks"].value:
                    try:
                        from scipy.signal import find_peaks

                        min_prom = float(c["peak_prom"].value) * (p_p.max() or 1.0)
                        peak_idx, _ = find_peaks(p_p, prominence=min_prom)
                        if peak_idx.size:
                            ax.plot(
                                f_p[peak_idx],
                                p_p[peak_idx],
                                "v",
                                ms=8,
                                color="#e94560",
                                zorder=5,
                                label=f"peaks: {', '.join(f'{f_p[i]:.2f}' for i in peak_idx[:5])} GHz",
                            )
                            ax.legend(fontsize=8)
                    except Exception:
                        pass

                ax.set_xlabel("f [GHz]", fontsize=10)
                ax.set_ylabel(
                    "PSD (norm.)" if c["normalize"].value else "PSD", fontsize=10
                )
                ax.set_title(f"{name.title()} spectrum", fontsize=11)
                ax.grid(True, alpha=0.25)

            fig.suptitle("📊 Vortex Spectrum", fontsize=12, color="#e94560")
            self._show_figure(fig, health=health)
            self._set_status("Spectrum computed", "ok")
        except Exception as exc:
            self._set_status(f"Spectrum failed: {exc}", "error")
            log.exception("Spectrum error")

    # ---- SPECTROGRAM ------------------------------------------------

    def _run_spectrogram(self, c: dict):
        self._set_status("Computing spectrogram…", "info")
        try:
            health = self._get_health()
            traj = self._state.trajectory
            if traj is None:
                traj = self._vx.core.track()
                self._state.trajectory = traj

            component = c["component"].value
            nperseg = int(c["nperseg"].value)
            noverlap = int(nperseg * c["noverlap_pct"].value / 100)

            sgram = self._vx.spectrum.spectrogram(
                component=component,
                trajectory=traj,
                nperseg=nperseg,
                noverlap=noverlap,
            )
            self._state.spectrogram = sgram

            f_ghz = np.asarray(sgram.frequencies) / 1e9
            t_ns = np.asarray(sgram.times) * 1e9
            S = np.asarray(sgram.power)

            f_max = float(c["f_max_ghz"].value)
            mask = f_ghz <= f_max
            f_p = f_ghz[mask]
            S_p = S[mask, :]

            if c["log_power"].value:
                S_p = 10 * np.log10(S_p + 1e-30)

            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            im = ax.pcolormesh(t_ns, f_p, S_p, cmap=c["cmap"].value, shading="auto")
            fig.colorbar(
                im,
                ax=ax,
                label="Power [dB]" if c["log_power"].value else "Power",
                shrink=0.8,
            )
            ax.set_xlabel("t [ns]", fontsize=10)
            ax.set_ylabel("f [GHz]", fontsize=10)
            ax.set_title(f"🌈 Spectrogram · {component} component", fontsize=11)
            ax.grid(False)

            self._show_figure(fig, health=health)
            self._set_status("Spectrogram done", "ok")
        except Exception as exc:
            self._set_status(f"Spectrogram failed: {exc}", "error")
            log.exception("Spectrogram error")

    # ---- MODES ------------------------------------------------------

    def _run_modes(self, c: dict):
        self._set_status("Computing vortex modes…", "info")
        try:
            health = self._get_health()
            n = int(c["n_modes"].value)
            mode_list = self._vx.modes.classify_all(max_modes=n)
            self._state.mode_result = mode_list

            if not mode_list:
                fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
                ax.text(
                    0.5,
                    0.5,
                    "No modes detected",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                    color="#a8b2d8",
                )
                ax.axis("off")
                self._show_figure(fig, health=health)
                self._set_status("No modes detected", "warn")
                return

            labels = [m.label for m in mode_list]
            freqs = [float(m.frequency_ghz) for m in mode_list]
            powers = [float(m.power) for m in mode_list]
            x_pos = np.arange(len(labels))

            fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

            ax_f.bar(x_pos, freqs, color="#58a6ff", alpha=0.85, edgecolor="#1e293b")
            ax_f.set_xticks(x_pos)
            ax_f.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax_f.set_ylabel("Frequency [GHz]", fontsize=10)
            ax_f.set_title("Mode Frequencies", fontsize=11)
            ax_f.grid(True, alpha=0.25, axis="y")

            ax_p.bar(x_pos, powers, color="#3fb950", alpha=0.85, edgecolor="#1e293b")
            ax_p.set_xticks(x_pos)
            ax_p.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax_p.set_ylabel("Relative power", fontsize=10)
            ax_p.set_title("Mode Powers", fontsize=11)
            ax_p.grid(True, alpha=0.25, axis="y")

            fig.suptitle("🎭 Vortex Modes", fontsize=12, color="#e94560")
            self._show_figure(fig, health=health)
            self._set_status(f"Modes: {len(mode_list)} found", "ok")
        except Exception as exc:
            self._set_status(f"Modes failed: {exc}", "error")
            log.exception("Modes error")

    # ---- EVENTS -----------------------------------------------------

    def _run_events(self, c: dict):
        self._set_status("Detecting events…", "info")
        try:
            health = self._get_health()
            traj = self._state.trajectory
            if traj is None:
                traj = self._vx.core.track()
                self._state.trajectory = traj

            event_type = c["event_type"].value
            threshold = float(c["threshold"].value)
            if event_type in ("all", "switching"):
                events = self._vx.events.polarity_switches(threshold=threshold)
            elif event_type == "annihilation":
                events = self._vx.events.core_expulsions()
            elif event_type == "creation":
                events = self._vx.events.state_switches()
            else:
                events = self._vx.events.polarity_switches(threshold=threshold)
            self._state.events_result = events

            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

            x_nm = np.asarray(traj.x) * 1e9
            y_nm = np.asarray(traj.y) * 1e9
            t_ns = np.asarray(traj.time) * 1e9
            r_nm = np.sqrt(x_nm**2 + y_nm**2)

            ax.plot(t_ns, r_nm, lw=1.2, color="#58a6ff", label="radius", zorder=1)

            ev_colors = {
                "annihilation": "#f85149",
                "creation": "#3fb950",
                "switching": "#d29922",
            }
            for ev in events:
                t_ev = float(getattr(ev, "time", 0.0)) * 1e9
                from_v = getattr(ev, "from_p", getattr(ev, "from_state", "?"))
                to_v = getattr(ev, "to_p", getattr(ev, "to_state", "?"))
                color = ev_colors.get(event_type, "#e94560")
                ax.axvline(
                    t_ev,
                    color=color,
                    lw=1.5,
                    alpha=0.8,
                    label=f"{from_v}→{to_v} @ {t_ev:.1f} ns",
                )

            ax.set_xlabel("t [ns]", fontsize=10)
            ax.set_ylabel("r [nm]", fontsize=10)
            ax.set_title(
                f"⚡ Event Detection  ·  {len(events)} events found", fontsize=11
            )
            ax.grid(True, alpha=0.25)
            if events:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=8)

            self._show_figure(fig, health=health)
            self._set_status(f"Events: {len(events)} found", "ok")
        except Exception as exc:
            self._set_status(f"Event detection failed: {exc}", "error")
            log.exception("Events error")

    # ---- SIGNALS ----------------------------------------------------

    def _run_signals(self, c: dict):
        self._set_status("Computing synthetic signal…", "info")
        try:
            health = self._get_health()
            traj = self._state.trajectory
            if traj is None:
                traj = self._vx.core.track()
                self._state.trajectory = traj

            from mmpp.solitons.vortex.nonlinear.interactive import (
                proxy_psd,
                proxy_signal_from_trajectory,
            )

            chirality = int(c["chirality"].value)
            angle = float(c["polarizer_angle"].value)
            sig = proxy_signal_from_trajectory(
                traj,
                polarizer_angle_deg=angle,
                chirality=chirality,
            )
            self._state.signals_result = sig

            dt = float(np.mean(np.diff(traj.time))) if len(traj.time) > 1 else 1e-12
            t_ns = np.asarray(traj.time) * 1e9

            if c["show_psd"].value:
                f_max = float(c["f_max_ghz"].value)
                method = c["method"].value
                f_hz, psd = proxy_psd(sig, dt=dt, method=method)
                f_ghz = f_hz / 1e9
                mask = f_ghz <= f_max

                fig, (ax_t, ax_f) = plt.subplots(
                    1, 2, figsize=self.figsize, dpi=self.dpi
                )
                ax_t.plot(t_ns, sig, lw=1.0, color="#58a6ff")
                ax_t.set_xlabel("t [ns]")
                ax_t.set_ylabel("TMR signal (a.u.)")
                ax_t.set_title("📡 Proxy signal", fontsize=11)
                ax_t.grid(True, alpha=0.25)

                ax_f.plot(f_ghz[mask], psd[mask], lw=1.4, color="#3fb950")
                ax_f.fill_between(f_ghz[mask], psd[mask], alpha=0.2, color="#3fb950")
                ax_f.set_xlabel("f [GHz]")
                ax_f.set_ylabel("PSD")
                ax_f.set_title("📡 Signal PSD", fontsize=11)
                ax_f.grid(True, alpha=0.25)
            else:
                fig, ax_t = plt.subplots(figsize=self.figsize, dpi=self.dpi)
                ax_t.plot(t_ns, sig, lw=1.0, color="#58a6ff")
                ax_t.set_xlabel("t [ns]")
                ax_t.set_ylabel("TMR signal (a.u.)")
                ax_t.set_title("📡 Proxy MTJ signal", fontsize=11)
                ax_t.grid(True, alpha=0.25)

            fig.suptitle("📡 Synthetic Signal", fontsize=12, color="#e94560")
            self._show_figure(fig, health=health)
            self._set_status("Signal computed", "ok")
        except Exception as exc:
            self._set_status(f"Signal failed: {exc}", "error")
            log.exception("Signals error")

    # ---- THIELE QUICK -----------------------------------------------

    def _run_thiele_quick(self, c: dict):
        self._set_status("Running Thiele quick fit…", "info")
        try:
            import math

            from mmpp.analytical.thiele import (
                CIPThieleModel,
                CPPThieleModel,
                DiskGeometry,
                ExternalField,
                MaterialParams,
                current_dc,
                field_dc,
                omega0_novosad,
            )

            R_m = float(c["diameter_nm"].value) * 1e-9 / 2.0
            L_m = float(c["thickness_nm"].value) * 1e-9
            Ms = float(c["ms_kA"].value) * 1e3
            alpha = float(c["alpha"].value)
            I_mA = float(c["current_mA"].value)
            P = float(c["polarization"].value)
            Bz = float(c["bz_mT"].value) * 1e-3
            model_type = c["model"].value

            geom = DiskGeometry(R=R_m, L=L_m)
            mat = MaterialParams(Ms=Ms, alpha=alpha, P=P)
            b_field = ExternalField(Bz_T=Bz)
            omega0 = omega0_novosad(mat, geom)

            # Convert mA to A/m² via pillar cross-section area
            area = math.pi * R_m**2
            J_dc = (I_mA * 1e-3) / max(area, 1e-20)
            J_func = current_dc(J_dc)
            B_func = field_dc(b_field)

            if model_type == "CPP":
                model = CPPThieleModel(
                    material=mat,
                    geom=geom,
                    omega0=omega0,
                    polarity=1,
                    field=b_field,
                )
            else:
                model = CIPThieleModel(
                    material=mat,
                    geom=geom,
                    omega0=omega0,
                    polarity=1,
                    field=b_field,
                )

            result = model.simulate(
                t_span=(0.0, 10e-9),
                r0=(R_m * 0.1, 0.0),
                J_func=J_func,
                B_func=B_func,
                dt=5e-12,
            )
            self._state.nonlinear_result = result

            fig, (ax_orbit, ax_r) = plt.subplots(
                1, 2, figsize=self.figsize, dpi=self.dpi
            )

            x_nm = result.x * 1e9
            y_nm = result.y * 1e9
            t_ns = result.t * 1e9
            r_nm = result.r * 1e9

            ax_orbit.plot(x_nm, y_nm, lw=0.8, color="#58a6ff", alpha=0.85)
            ax_orbit.set_xlabel("x [nm]", fontsize=10)
            ax_orbit.set_ylabel("y [nm]", fontsize=10)
            ax_orbit.set_aspect("equal")
            ax_orbit.set_title(f"{model_type} orbit", fontsize=11)
            ax_orbit.grid(True, alpha=0.25)

            ax_r.plot(t_ns, r_nm, lw=1.2, color="#3fb950")
            ax_r.set_xlabel("t [ns]", fontsize=10)
            ax_r.set_ylabel("r [nm]", fontsize=10)
            ax_r.set_title("Orbit radius", fontsize=11)
            ax_r.grid(True, alpha=0.25)

            f_val = result.dominant_frequency_ghz
            r_val = result.steady_state_radius_m * 1e9
            fig.suptitle(
                f"🔬 Thiele ({model_type}) | f = {f_val:.3f} GHz, r = {r_val:.1f} nm",
                fontsize=11,
                color="#e94560",
            )
            # Note: Thiele quick tab uses analytical model, not the simulation
            # data — no health annotation needed here (skip)
            self._show_figure(fig)
            self._set_status(
                f"Thiele ({model_type}): f = {f_val:.3f} GHz, r = {r_val:.1f} nm", "ok"
            )
        except Exception as exc:
            self._set_status(f"Thiele failed: {exc}", "error")
            log.exception("Thiele error")

    # ---- THIELE FULL DASHBOARD --------------------------------------

    def _run_thiele_full(self):
        self._set_status("Launching Thiele interactive dashboard…", "info")
        try:
            from mmpp.solitons.vortex.nonlinear.interactive import (
                ThieleInteractiveDashboard,
            )

            db = ThieleInteractiveDashboard()
            with self._output:
                clear_output(wait=True)
                db.show()
            self._set_status("Thiele dashboard active", "ok")
        except Exception as exc:
            self._set_status(f"Thiele dashboard failed: {exc}", "error")
            log.exception("Thiele dashboard error")

    # ------------------------------------------------------------------
    # Preset management
    # ------------------------------------------------------------------

    def _collect_state(self) -> dict:
        """Collect all widget values into a serializable dict."""
        import json

        state: dict = {}
        for tab_name, controls in self._controls.items():
            state[tab_name] = {}
            for key, w in controls.items():
                try:
                    val = w.value
                    json.dumps(val)  # ensure serializable
                    state[tab_name][key] = val
                except Exception:
                    pass
        return state

    def _apply_state(self, state: dict):
        """Apply a serialized state dict to all widgets."""
        for tab_name, values in state.items():
            if tab_name not in self._controls:
                continue
            for key, val in values.items():
                if key in self._controls[tab_name]:
                    try:
                        self._controls[tab_name][key].value = val
                    except Exception:
                        pass

    def _get_presets_dir(self):
        import os
        from pathlib import Path

        cwd = Path(os.getcwd())
        d = cwd / ".mmpp_presets" / "vortex_dashboard"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _refresh_preset_list(self):
        try:
            presets_dir = self._get_presets_dir()
            names = ["— load preset —"] + sorted(
                p.stem for p in presets_dir.glob("*.json")
            )
            self._w_preset_load.options = names
        except Exception:
            pass

    def _on_save_preset(self, _):
        import json

        name = self._w_preset_name.value.strip()
        if not name:
            self._set_status("Enter a preset name first", "warn")
            return
        try:
            presets_dir = self._get_presets_dir()
            path = presets_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(self._collect_state(), f, indent=2)
            self._refresh_preset_list()
            self._set_status(f"Preset '{name}' saved", "ok")
        except Exception as exc:
            self._set_status(f"Save failed: {exc}", "error")

    def _on_load_preset(self, change):
        import json

        name = change["new"]
        if name.startswith("—"):
            return
        try:
            presets_dir = self._get_presets_dir()
            path = presets_dir / f"{name}.json"
            with open(path) as f:
                state = json.load(f)
            self._apply_state(state)
            self._set_status(f"Preset '{name}' loaded", "ok")
        except Exception as exc:
            self._set_status(f"Load failed: {exc}", "error")

    # ------------------------------------------------------------------
    # HTML repr for notebook
    # ------------------------------------------------------------------

    def _repr_html_(self) -> str:
        return (
            "<b>VortexInteractiveDashboard</b> — call <code>.show()</code> to display"
        )
