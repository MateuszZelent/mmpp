"""Interactive dashboard helpers for Thiele vortex analysis."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from mmpp._shared.spectral import compute_psd
from mmpp.analytical.thiele import (
    CIPThieleModel,
    CPPThieleModel,
    DiskGeometry,
    ExternalField,
    FieldCalibration,
    MaterialParams,
    current_dc,
    ellipse_area,
    field_ac,
    field_dc,
    omega0_novosad,
    slonczewski_mtj_efficiency,
)

try:
    import ipywidgets as widgets
    from IPython.display import HTML

    _HAS_WIDGETS = True
except ImportError:
    widgets = None  # type: ignore[misc, assignment]
    HTML = None  # type: ignore[misc, assignment]
    _HAS_WIDGETS = False

try:
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
    # Wyciszamy irytujące ostrzeżenia o tight_layout podczas szybkiej interakcji
    warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
except ImportError:
    plt = None  # type: ignore[misc, assignment]
    _HAS_MATPLOTLIB = False


def _normalize_in_plane_polarizer(angle_deg: float) -> tuple[float, float]:
    """Zwraca znormalizowany wektor polaryzatora w płaszczyźnie (in-plane)."""
    rad = math.radians(angle_deg)
    return math.cos(rad), math.sin(rad)


def proxy_signal_from_trajectory(
    trajectory: Any,
    *,
    disk_radius: float | None = None,
    polarizer_angle_deg: float = 0.0,
    polarizer: tuple[float, float, float] | tuple[float, float] | None = None,
    center: tuple[float, float] | None = None,
    cubic: float = 0.0,
    chirality: int = 1,
) -> np.ndarray:
    """Build MTJ readout proxy signal (TMR) from vortex core trajectory.

    Physics note:
    For a magnetic vortex, displacing the core by (X, Y) induces an average
    in-plane magnetization perpendicular to the displacement:
        <M_x> / M_s = - c * xi * (Y / R)
        <M_y> / M_s =   c * xi * (X / R)
    where c is the chirality (+1 CCW, -1 CW) and xi ≈ 2/3 is a shape factor.
    """
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)

    if center is None:
        x0 = float(np.mean(x)) if x.size else 0.0
        y0 = float(np.mean(y)) if y.size else 0.0
    else:
        x0, y0 = float(center[0]), float(center[1])

    if disk_radius is None:
        disk_radius = float(getattr(trajectory, "disk_radius", 100e-9))
    if not np.isfinite(disk_radius) or float(disk_radius) <= 0.0:
        radii = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        disk_radius = max(float(np.percentile(radii, 95)) * 1.1, 1e-12)

    r_norm = float(disk_radius)

    x_reduced = (x - x0) / r_norm
    y_reduced = (y - y0) / r_norm

    # Transformacja na średnią magnetyzację (uwzględniając chiralność wira)
    c = 1.0 if chirality >= 0 else -1.0
    xi = 2.0 / 3.0
    mx_avg = -c * xi * y_reduced
    my_avg = c * xi * x_reduced

    if polarizer is not None:
        polarizer_vec = np.asarray(polarizer, dtype=float).reshape(-1)
        if polarizer_vec.size < 2:
            raise ValueError("polarizer must provide at least x and y components")
        px_raw = float(polarizer_vec[0])
        py_raw = float(polarizer_vec[1])
        norm = float(np.hypot(px_raw, py_raw))
        if norm <= 1e-30:
            raise ValueError("polarizer x/y components cannot both be zero")
        px, py = px_raw / norm, py_raw / norm
    else:
        px, py = _normalize_in_plane_polarizer(polarizer_angle_deg)

    # Sygnał TMR jest proporcjonalny do rzutu <M> na oś polaryzatora w płaszczyźnie
    base_signal = px * mx_avg + py * my_avg
    signal = base_signal + float(cubic) * (base_signal**3)
    return np.asarray(signal, dtype=float)


def proxy_psd(
    signal: np.ndarray,
    *,
    dt: float,
    method: str = "welch",
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD for proxy MTJ signal."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    sig = np.asarray(signal, dtype=float)
    if sig.size < 4:
        return np.array([], dtype=float), np.array([], dtype=float)

    sig = sig - float(np.mean(sig))
    method_norm = str(method).lower()
    if method_norm not in {"welch", "fft"}:
        raise ValueError("method must be 'welch' or 'fft'")

    nperseg_eff = min(
        sig.size,
        int(nperseg) if nperseg is not None else max(64, sig.size // 4),
    )
    f, p, _, _ = compute_psd(
        sig,
        dt=float(dt),
        method="periodogram" if method_norm == "fft" else method_norm,
        nperseg=nperseg_eff,
        scaling="density",
    )
    return np.asarray(f, dtype=float), np.asarray(p, dtype=float)


@dataclass
class ThieleInteractiveDashboard:
    """Professional Ipywidgets dashboard for Vortex Dynamics."""

    def __init__(self, analyzer: Any = None):
        self._require_widgets()
        self._state: dict[str, Any] = {
            "fig": None,
            "axes": None,
            "lines": {},
            "patches": {},
            "texts": {},
            "mode": None,
        }

    def _require_widgets(self) -> None:
        if not _HAS_WIDGETS or not _HAS_MATPLOTLIB:
            raise ImportError("ipywidgets and matplotlib are required.")

    @staticmethod
    def _update_line(line, xdata, ydata):
        line.set_xdata(xdata)
        line.set_ydata(ydata)

    @staticmethod
    def _relim_and_rescale(ax):
        ax.relim()
        ax.autoscale_view()

    def show(self, figsize=(14, 4.5), dpi=100, fast_mode: bool = False):
        _ = fast_mode
        self._require_widgets()
        self.dpi = dpi

        style = {"description_width": "160px"}

        def slider(desc, val, min_v, max_v, step, unit="", tooltip=""):
            label = f"{desc} [{unit}]" if unit else desc
            # KRYTYCZNA ZMIANA: Tworzymy nowy obiekt Layout dla każdego suwaka z osobna!
            return widgets.FloatSlider(
                description=label,
                value=val,
                min=min_v,
                max=max_v,
                step=step,
                continuous_update=False,
                style=style,
                layout=widgets.Layout(width="95%"),
                description_tooltip=tooltip,
            )

        # =========================================================
        # ZAKŁADKA 0: Model Selection
        # =========================================================
        self.w_model_type = widgets.ToggleButtons(
            description="Physics Model:",
            options=[
                ("CPP (Vortex STNO - Slonczewski STT)", "CPP"),
                ("CIP (In-Plane Current - Zhang-Li STT)", "CIP"),
            ],
            value="CPP",
            style=style,
        )

        # --- TAB 1: Geometry & Material ---
        self.w_geom_mode = widgets.ToggleButtons(
            description="Shape:",
            options=[("Disk", "disk"), ("Ellipse→Eq", "ellipse_eq")],
            value="disk",
            style=style,
        )
        self.w_disk_d = slider(
            "Diameter D",
            250.0,
            50.0,
            1000.0,
            5.0,
            "nm",
            "Physical diameter of the nanodisk",
        )
        self.w_size_x = slider("Ellipse X", 300.0, 50.0, 1000.0, 5.0, "nm")
        self.w_size_y = slider("Ellipse Y", 200.0, 50.0, 1000.0, 5.0, "nm")
        self.w_thick = slider(
            "Thickness L", 10.0, 2.0, 50.0, 0.5, "nm", "Magnetic free layer thickness"
        )
        self.w_ms = slider(
            "Magnetization M_s",
            800.0,
            100.0,
            2000.0,
            10.0,
            "kA/m",
            "Saturation magnetization (Permalloy ≈ 800)",
        )

        # KRYTYCZNA ZMIANA 2: Suwak logarytmiczny też dostaje nową instancję Layoutu!
        self.w_alpha = widgets.FloatLogSlider(
            description="Damping α:",
            value=0.01,
            base=10,
            min=-4,
            max=-1,
            step=0.01,
            continuous_update=False,
            style=style,
            layout=widgets.Layout(width="95%"),
        )

        tab_geom = widgets.VBox(
            [
                widgets.HTML("<b>Free Layer Geometry</b>"),
                self.w_geom_mode,
                self.w_disk_d,
                self.w_size_x,
                self.w_size_y,
                self.w_thick,
                widgets.HTML("<hr><b>Material Properties</b>"),
                self.w_ms,
                self.w_alpha,
            ]
        )

        # --- TAB 2: STT & Readout ---
        self.w_current = slider(
            "Current I_dc",
            6.0,
            -30.0,
            30.0,
            0.1,
            "mA",
            "DC current injected into the system",
        )
        self.w_pol = slider("Spin Polarization P", 0.3, 0.0, 1.0, 0.02)

        # CPP specific
        self.w_lambda = slider(
            "Slonczewski Λ (CPP)",
            1.0,
            0.5,
            5.0,
            0.05,
            "",
            "Asymmetry parameter for MTJ STT",
        )
        self.w_pz = slider(
            "Polarizer P_z (CPP)",
            1.0,
            -1.0,
            1.0,
            0.05,
            "",
            "Out-of-plane component (excites vortex)",
        )

        # CIP specific
        self.w_beta = slider(
            "Non-adiabatic β (CIP)",
            0.02,
            0.0,
            0.2,
            0.005,
            "",
            "Zhang-Li non-adiabatic parameter",
        )
        self.w_cip_angle = slider(
            "Current Angle (CIP)",
            0.0,
            -180.0,
            180.0,
            5.0,
            "deg",
            "Direction of current in the plane",
        )

        self.w_p = widgets.ToggleButtons(
            description="Core Polarity (p):",
            options=[("Up (+1)", 1), ("Down (-1)", -1)],
            value=1,
            style=style,
        )
        self.w_c = widgets.ToggleButtons(
            description="Chirality (c):",
            options=[("CCW (+1)", 1), ("CW (-1)", -1)],
            value=1,
            style=style,
        )
        self.w_angle = slider("TMR Readout Angle", 0.0, -180.0, 180.0, 5.0, "deg")

        tab_stt = widgets.VBox(
            [
                widgets.HTML("<b>Spin Transfer Torque (STT)</b>"),
                self.w_current,
                self.w_pol,
                self.w_lambda,
                self.w_pz,  # Hideable CPP params
                self.w_beta,
                self.w_cip_angle,  # Hideable CIP params
                self.w_p,
                self.w_c,
                widgets.HTML("<hr><b>MTJ Readout (TMR)</b>"),
                self.w_angle,
            ]
        )

        # --- TAB 3: Magnetic Fields ---
        self.w_bx = slider("B_x (IP)", 50.0, -200.0, 200.0, 1.0, "mT")
        self.w_by = slider("B_y (IP)", 0.0, -200.0, 200.0, 1.0, "mT")
        self.w_bz = slider("B_z (OOP)", 0.0, -500.0, 500.0, 5.0, "mT")
        self.w_oersted = slider(
            "Oersted Shift (CPP)", -15.0, -50.0, 50.0, 0.5, "MHz per 10 MA/cm²"
        )

        self.w_field_mode = widgets.ToggleButtons(
            description="Field Mode:", options=["DC", "AC"], value="DC", style=style
        )
        self.w_bac_freq = slider("AC Freq", 1.0, 0.01, 10.0, 0.01, "GHz")
        self.w_bac_phase = slider("AC Phase", 0.0, -180.0, 180.0, 5.0, "deg")

        tab_field = widgets.VBox(
            [
                widgets.HTML(
                    "<b>External Magnetic Field</b><br/><i>In-plane fields (Bx, By) shift the core equilibrium based on 's_eq' parameter. OOP field (Bz) shifts frequency based on 'dω₀/dBz'.</i>"
                ),
                self.w_bx,
                self.w_by,
                self.w_bz,
                widgets.HTML("<hr><b>Self-induced Field (CPP Only)</b>"),
                self.w_oersted,
                widgets.HTML("<hr><b>AC Excitation</b>"),
                self.w_field_mode,
                self.w_bac_freq,
                self.w_bac_phase,
            ]
        )

        # --- TAB 4: Thiele Physics Calibration ---
        self.w_auto_w0 = widgets.Checkbox(
            description="Auto-calculate ω₀", value=True, style=style
        )
        self.w_omega0 = slider("Override Base f₀", 0.395, 0.05, 5.0, 0.01, "GHz")
        self.w_n = slider(
            "Nonlinearity N (CPP)",
            0.25,
            -1.0,
            2.0,
            0.05,
            "",
            "Frequency blue-shift/red-shift coefficient",
        )

        # ZMIANA: Niezereowe wartości domyślne aby pole magnetyczne działało "Out of the box"
        self.w_domega0_dBz = slider("Zeeman dω₀/dB_z", 2.8, -10.0, 10.0, 0.1, "GHz/T")
        self.w_seq_per_T = slider(
            "IP Shift s_eq",
            1.5,
            -5.0,
            5.0,
            0.05,
            "1/T",
            "Equilibrium shift per 1T of in-plane field",
        )

        tab_model = widgets.VBox(
            [
                widgets.HTML("<b>Gyrotropic Mode Tuning</b>"),
                self.w_auto_w0,
                self.w_omega0,
                widgets.HTML(
                    "<hr><b>Phenomenological Coupling</b><br/><i>Must be != 0 for fields to have an effect. Fit these to mumax3!</i>"
                ),
                self.w_n,
                self.w_domega0_dBz,
                self.w_seq_per_T,
            ]
        )

        # --- TAB 5: Solver & Noise ---
        self.w_sim_mode = widgets.ToggleButtons(
            description="Sim Mode:",
            options=["Fast (Analytical f(J))", "Full Trajectory"],
            value="Full Trajectory",
            style=style,
        )
        self.w_tend = slider("Sim. Time", 200.0, 10.0, 1000.0, 10.0, "ns")
        self.w_dt = slider("Time Step dt", 5.0, 0.5, 50.0, 0.5, "ps")
        self.w_sde = widgets.Checkbox(
            description="Enable Thermal Noise (CPP SDE)", value=False, style=style
        )
        self.w_temp = slider("Temperature", 300.0, 0.0, 600.0, 5.0, "K")
        self.w_noise = slider("Noise Multiplier", 1.0, 0.0, 5.0, 0.1, "")
        self.w_rand_seed = widgets.Checkbox(
            description="Randomize Seed", value=False, style=style
        )
        self.w_seed = widgets.IntText(
            description="RNG Seed:",
            value=42,
            style=style,
            layout=widgets.Layout(width="200px"),
        )

        tab_solver = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Integration Control</b><br/><i>SDE is supported only in the CPP model.</i>"
                ),
                self.w_sim_mode,
                self.w_tend,
                self.w_dt,
                widgets.HTML("<hr><b>Stochastic Parameters (CPP)</b>"),
                self.w_sde,
                self.w_temp,
                self.w_noise,
                widgets.HBox([self.w_seed, self.w_rand_seed]),
            ]
        )

        # Złożenie Tabs
        self.tabs = widgets.Tab(
            children=[tab_geom, tab_stt, tab_field, tab_model, tab_solver]
        )
        self.tabs.set_title(0, "📏 Geom & Mat")
        self.tabs.set_title(1, "⚡ STT")
        self.tabs.set_title(2, "🧲 Fields")
        self.tabs.set_title(3, "⚙️ Calib.")
        self.tabs.set_title(4, "⏱️ Solver")

        # Live Info Banner & Outputs
        self.w_hud = widgets.HTML(value="")
        self.w_status = widgets.HTML(value="")
        self.out = widgets.Output()

        self.figsize = figsize

        self._wire_events()
        self._sync_disabled()

        # Pusty wykres startowy
        self._ensure_figure(self.w_sim_mode.value)
        self.w_status.value = "<b style='color:green;'>Status: Ready</b>"

        # Przycisk uruchamiania
        btn_run = widgets.Button(
            description=" ▶ RUN SIMULATION",
            button_style="success",
            layout=widgets.Layout(width="200px", height="40px"),
        )
        btn_run.on_click(self._render)

        header = widgets.HBox(
            [btn_run, self.w_status],
            layout=widgets.Layout(
                align_items="center", gap="20px", margin="10px 0 10px 0"
            ),
        )

        return widgets.VBox(
            [
                widgets.HTML(
                    "<h2 style='color:#334155; margin-bottom:0;'>🌪️ Vortex Dynamics Dashboard</h2>"
                ),
                self.w_model_type,
                self.w_hud,
                header,
                self.tabs,
                self.out,
            ]
        )

    def _ensure_figure(self, mode: str):
        if self._state["mode"] == mode and self._state["fig"] is not None:
            return

        if self._state["fig"] is not None:
            plt.close(self._state["fig"])

        self._state["lines"].clear()
        self._state["patches"].clear()

        if mode == "Fast (Analytical f(J))":
            fs = (10.0, 4.0) if self.figsize is None else self.figsize
            fig, ax = plt.subplots(1, 1, figsize=fs, dpi=self.dpi, layout="tight")
            (line_fj,) = ax.plot([], [], "o-", color="tab:blue", markersize=3)
            vline = ax.axvline(0, color="tab:red", linestyle="--", alpha=0.6)
            ax.set_xlabel("Current Density J [MA/cm²]")
            ax.set_ylabel("Frequency [GHz]")
            ax.set_title("Analytic Steady-State Frequency vs Current (CPP Only)")
            ax.grid(True, alpha=0.3)

            self._state["lines"]["fj"] = line_fj
            self._state["lines"]["vline"] = vline
            self._state["axes"] = [ax]
        else:
            fs = (14.0, 4.0) if self.figsize is None else self.figsize
            fig, axes = plt.subplots(1, 3, figsize=fs, dpi=self.dpi, layout="tight")

            # Orbit
            (line_orbit,) = axes[0].plot(
                [], [], "-", color="tab:blue", lw=1.5, zorder=2
            )
            (line_point,) = axes[0].plot([], [], "ro", markersize=5, zorder=3)
            circle = plt.Circle(
                (0, 0),
                1,
                fill=False,
                color="tab:gray",
                linestyle="--",
                lw=1.2,
                alpha=0.9,
                zorder=1,
            )
            axes[0].add_patch(circle)
            axes[0].set_xlabel("X [nm]")
            axes[0].set_ylabel("Y [nm]")
            axes[0].set_title("Core Trajectory")
            axes[0].set_aspect("equal", adjustable="datalim")
            axes[0].grid(True, alpha=0.3)

            # Signal
            (line_sig,) = axes[1].plot([], [], color="tab:orange", lw=1.0)
            axes[1].set_xlabel("Time [ns]")
            axes[1].set_ylabel("TMR Proxy [a.u.]")
            axes[1].set_title("Time-domain Signal")
            axes[1].grid(True, alpha=0.3)

            # PSD
            (line_psd,) = axes[2].plot([], [], color="tab:green", lw=1.5)
            axes[2].set_xlabel("Frequency [GHz]")
            axes[2].set_ylabel("PSD [dB]")
            axes[2].set_title("Power Spectrum")
            axes[2].grid(True, alpha=0.3)

            suptitle = fig.suptitle("", fontsize=10)

            self._state["lines"]["orbit"] = line_orbit
            self._state["lines"]["point"] = line_point
            self._state["patches"]["circle"] = circle
            self._state["lines"]["signal"] = line_sig
            self._state["lines"]["psd"] = line_psd
            self._state["texts"]["suptitle"] = suptitle
            self._state["axes"] = axes

        self._state["fig"] = fig
        self._state["mode"] = mode

    def _sync_disabled(self, *_args):
        # Aktywacja w zaleznosci od modelu CPP/CIP
        is_cpp = self.w_model_type.value == "CPP"
        is_cip = not is_cpp

        # POPRAWKA: Używamy `None` zamiast `'flex'`, aby nie psuć wewnętrznego CSS suwaka
        self.w_lambda.layout.display = None if is_cpp else "none"
        self.w_pz.layout.display = None if is_cpp else "none"
        self.w_oersted.layout.display = None if is_cpp else "none"
        self.w_n.layout.display = None if is_cpp else "none"
        self.w_sde.layout.display = None if is_cpp else "none"

        self.w_beta.layout.display = None if is_cip else "none"
        self.w_cip_angle.layout.display = None if is_cip else "none"

        if is_cip and self.w_sim_mode.value == "Fast (Analytical f(J))":
            self.w_sim_mode.value = "Full Trajectory"
        self.w_sim_mode.disabled = is_cip  # CIP obsługuje tylko pełne trajektorie ODE

        # Pozostale zaleznosci
        is_disk = self.w_geom_mode.value == "disk"
        self.w_disk_d.disabled = not is_disk
        self.w_size_x.disabled = is_disk
        self.w_size_y.disabled = is_disk

        is_ac = self.w_field_mode.value == "AC"
        self.w_bac_freq.disabled = not is_ac
        self.w_bac_phase.disabled = not is_ac

        is_sde = self.w_sde.value and is_cpp
        self.w_temp.disabled = not is_sde
        self.w_noise.disabled = not is_sde
        self.w_seed.disabled = (not is_sde) or self.w_rand_seed.value

        self.w_omega0.disabled = self.w_auto_w0.value

    def _update_hud(self, *_args):
        """Aktualizuje statystyki fizyczne w locie bez całkowania równań ODE."""
        try:
            if self.w_geom_mode.value == "disk":
                radius_eq = 0.5 * self.w_disk_d.value * 1e-9
                geom_desc = f"Disk D={self.w_disk_d.value:.0f} nm"
            else:
                area_ell = ellipse_area(
                    self.w_size_x.value * 1e-9, self.w_size_y.value * 1e-9
                )
                radius_eq = math.sqrt(area_ell / math.pi)
                geom_desc = (
                    f"Ellipse {self.w_size_x.value:.0f}x{self.w_size_y.value:.0f} nm"
                )

            area = math.pi * (radius_eq**2)
            is_cpp = self.w_model_type.value == "CPP"

            peff = (
                slonczewski_mtj_efficiency(
                    self.w_pol.value, self.w_lambda.value, self.w_pz.value
                )
                if is_cpp
                else self.w_pol.value
            )

            current_density_A_m2 = (self.w_current.value * 1e-3) / area
            J_MA_cm2 = current_density_A_m2 / 1e10

            mat = MaterialParams(
                Ms=self.w_ms.value * 1e3,
                alpha=self.w_alpha.value,
                P=peff,
                beta_nonadiabatic=self.w_beta.value if not is_cpp else None,
            )
            geo = DiskGeometry(R=radius_eq, L=self.w_thick.value * 1e-9)

            w0_novo = omega0_novosad(mat, geo)
            if self.w_auto_w0.value:
                self.w_omega0.unobserve(self._update_hud, names="value")
                self.w_omega0.value = w0_novo / (2.0 * math.pi * 1e9)
                self.w_omega0.observe(self._update_hud, names="value")

            omega0_rad = 2.0 * math.pi * self.w_omega0.value * 1e9
            ext_field = ExternalField(
                Bx_T=self.w_bx.value * 1e-3,
                By_T=self.w_by.value * 1e-3,
                Bz_T=self.w_bz.value * 1e-3,
            )
            fcal = FieldCalibration(
                domega0_dBz=self.w_domega0_dBz.value * 2.0 * math.pi * 1e9,
                seq_per_T=self.w_seq_per_T.value,
                chirality=self.w_c.value,
            )

            if is_cpp:
                # CPP HUD Logic
                oe_rad_per_Am2 = (self.w_oersted.value * 1e6 * 2.0 * math.pi) / 1e11

                model = CPPThieleModel(
                    material=mat,
                    geom=geo,
                    omega0=omega0_rad,
                    N=self.w_n.value,
                    polarity=self.w_p.value,
                    domega0_dJ=oe_rad_per_Am2,
                    field=ext_field,
                    field_cal=fcal,
                )
                j_th = model.threshold_current_dc()
                j_th_MA = j_th / 1e10 if j_th != float("inf") else float("inf")

                f_pred = model.predict_frequency_dc(
                    current_density_A_m2, allow_edge=True
                )

                is_osc = abs(current_density_A_m2) > j_th and j_th != float("inf")
                status_color = "#10b981" if is_osc else "#ef4444"
                status_text = (
                    "AUTO-OSCILLATION (CPP)" if is_osc else "DAMPED (Sub-threshold)"
                )
                f_disp = (
                    f"<span style='color:{status_color}; font-weight:bold;'>{f_pred * 1e-9:.3f} GHz</span>"
                    if f_pred
                    else "<span style='color:#ef4444;'>Damped (J < J_th)</span>"
                )
                hud_details = f"<b>⚡ CPP Current :</b> <span style='color:{status_color}'>{J_MA_cm2:.2f} MA/cm²</span> (Threshold: {j_th_MA:.2f} MA/cm²)<br>"
            else:
                # CIP HUD Logic
                j_th_MA = float("nan")
                status_color = "#3b82f6"
                status_text = "CIP DYNAMICS (Core Displacement)"
                f_disp = "<span style='color:#94a3b8;'>N/A (CIP Model)</span>"
                hud_details = f"<b>⚡ CIP Current :</b> <span style='color:{status_color}'>{J_MA_cm2:.2f} MA/cm²</span> (Flow angle: {self.w_cip_angle.value}°)<br>"

            model_badge = f"<span style='background-color:#334155; padding:2px 6px; border-radius:4px;'>{self.w_model_type.value}</span>"
            hud_html = f"""
            <div style="background: #0f172a; color: #cbd5e1; padding: 12px; border-radius: 8px; border-left: 6px solid {status_color}; font-family: monospace; font-size: 1.1em; line-height: 1.5;">
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                    <div style="flex: 1; min-width: 200px;">
                        <b>{model_badge} 📏 Geometry:</b> {geom_desc} | R_eq = {radius_eq * 1e9:.1f} nm<br>
                        {hud_details}
                    </div>
                    <div style="flex: 1; min-width: 200px;">
                        <b>⚙️ Base f₀:</b> {self.w_omega0.value:.3f} GHz (Analytic Novosad = {w0_novo / (2 * math.pi * 1e9):.3f} GHz)<br>
                        <b>🎯 Pred f(J):</b> {f_disp}<br>
                        <b>🏷️ Status :</b> <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
                    </div>
                </div>
            </div>
            """
            self.w_hud.value = hud_html

        except Exception:
            pass

    def _wire_events(self):
        for w in [
            self.w_model_type,
            self.w_geom_mode,
            self.w_field_mode,
            self.w_sde,
            self.w_auto_w0,
            self.w_rand_seed,
        ]:
            w.observe(self._sync_disabled, names="value")

        widgets_to_hud = [
            self.w_model_type,
            self.w_geom_mode,
            self.w_disk_d,
            self.w_size_x,
            self.w_size_y,
            self.w_thick,
            self.w_ms,
            self.w_alpha,
            self.w_beta,
            self.w_current,
            self.w_pol,
            self.w_lambda,
            self.w_pz,
            self.w_p,
            self.w_c,
            self.w_bx,
            self.w_by,
            self.w_bz,
            self.w_oersted,
            self.w_auto_w0,
            self.w_omega0,
            self.w_n,
            self.w_domega0_dBz,
            self.w_seq_per_T,
            self.w_cip_angle,
        ]
        for w in widgets_to_hud:
            w.observe(self._update_hud, names="value")

    def _render(self, *_args):
        self.w_status.value = (
            "<b><span style='color:orange'>⚙️ Calculating ODE/SDE...</span></b>"
        )

        with self.out:
            self.out.clear_output(wait=True)
            try:
                is_cpp = self.w_model_type.value == "CPP"

                if self.w_geom_mode.value == "disk":
                    radius_eq = 0.5 * self.w_disk_d.value * 1e-9
                else:
                    area_ell = ellipse_area(
                        self.w_size_x.value * 1e-9, self.w_size_y.value * 1e-9
                    )
                    radius_eq = math.sqrt(area_ell / math.pi)

                area = math.pi * (radius_eq**2)
                peff = (
                    slonczewski_mtj_efficiency(
                        self.w_pol.value, self.w_lambda.value, self.w_pz.value
                    )
                    if is_cpp
                    else self.w_pol.value
                )

                current_density_A_m2 = (self.w_current.value * 1e-3) / area
                J_MA_cm2 = current_density_A_m2 / 1e10

                mat = MaterialParams(
                    Ms=self.w_ms.value * 1e3,
                    alpha=self.w_alpha.value,
                    P=peff,
                    beta_nonadiabatic=self.w_beta.value if not is_cpp else None,
                )
                geo = DiskGeometry(R=radius_eq, L=self.w_thick.value * 1e-9)
                omega0_rad = 2.0 * math.pi * self.w_omega0.value * 1e9

                ext_field = ExternalField(
                    Bx_T=self.w_bx.value * 1e-3,
                    By_T=self.w_by.value * 1e-3,
                    Bz_T=self.w_bz.value * 1e-3,
                )
                fcal = FieldCalibration(
                    domega0_dBz=self.w_domega0_dBz.value * 2.0 * math.pi * 1e9,
                    seq_per_T=self.w_seq_per_T.value,
                    chirality=self.w_c.value,
                )

                if self.w_field_mode.value == "AC":
                    b_func = field_ac(
                        ext_field,
                        f_hz=self.w_bac_freq.value * 1e9,
                        phase=math.radians(self.w_bac_phase.value),
                    )
                else:
                    b_func = field_dc(ext_field)

                model: Any
                if is_cpp:
                    oe_rad_per_Am2 = (self.w_oersted.value * 1e6 * 2.0 * math.pi) / 1e11
                    model = CPPThieleModel(
                        material=mat,
                        geom=geo,
                        omega0=omega0_rad,
                        N=self.w_n.value,
                        polarity=self.w_p.value,
                        domega0_dJ=oe_rad_per_Am2,
                        field=ext_field,
                        field_cal=fcal,
                    )
                else:
                    rad_cip = math.radians(self.w_cip_angle.value)
                    model = CIPThieleModel(
                        material=mat,
                        geom=geo,
                        omega0=omega0_rad,
                        polarity=self.w_p.value,
                        current_dir=(math.cos(rad_cip), math.sin(rad_cip)),
                        field=ext_field,
                        field_cal=fcal,
                    )

                t_span = (0.0, self.w_tend.value * 1e-9)
                dt = self.w_dt.value * 1e-12

                # ── FAST MODE (Only for CPP) ──
                if is_cpp and self.w_sim_mode.value == "Fast (Analytical f(J))":
                    self._ensure_figure("Fast (Analytical f(J))")
                    axes = self._state["axes"][0]

                    j_th = model.threshold_current_dc()
                    j_max_scan = (
                        max(10.0, (j_th / 1e10) * 4.0) if j_th != float("inf") else 10.0
                    )
                    j_grid = np.linspace(0.0, j_max_scan, 200)
                    f_grid = []

                    for j in j_grid:
                        f = model.predict_frequency_dc(j * 1e10, allow_edge=True)
                        f_grid.append(f * 1e-9 if f is not None else np.nan)

                    self._update_line(self._state["lines"]["fj"], j_grid, f_grid)
                    self._state["lines"]["vline"].set_xdata([J_MA_cm2, J_MA_cm2])

                    self._relim_and_rescale(axes)

                # ── FULL ODE/SDE MODE ──
                else:
                    self._ensure_figure("Full Trajectory")
                    axes = self._state["axes"]

                    if is_cpp:
                        if self.w_sde.value and self.w_temp.value > 0:
                            seed = (
                                None
                                if self.w_rand_seed.value
                                else int(self.w_seed.value)
                            )
                            traj = model.simulate_sde(
                                t_span=t_span,
                                s0=(1e-3, 0.0),
                                J_func=current_dc(current_density_A_m2),
                                B_func=b_func,
                                dt=dt,
                                temperature_k=self.w_temp.value,
                                noise_scale=self.w_noise.value,
                                seed=seed,
                            )
                        else:
                            traj = model.simulate(
                                t_span=t_span,
                                s0=(1e-3, 0.0),
                                J_func=current_dc(current_density_A_m2),
                                B_func=b_func,
                                dt=dt,
                            )
                    else:
                        # W modelu CIP symulujemy z dużą dokładnością omijając szum numeryczny punktu zerowego
                        kwargs: dict[str, Any] = {"atol": 1e-15, "rtol": 1e-10}
                        traj = model.simulate(
                            t_span=t_span,
                            r0=(1e-3 * radius_eq, 0.0),
                            J_func=current_dc(current_density_A_m2),
                            B_func=b_func,
                            dt=dt,
                            **kwargs,
                        )

                    # TMR Proxy Signal (Uwzględnia chiralność!)
                    signal = proxy_signal_from_trajectory(
                        traj,
                        disk_radius=radius_eq,
                        polarizer_angle_deg=self.w_angle.value,
                        chirality=self.w_c.value,
                    )

                    # FFT / Welch
                    freqs, psd = proxy_psd(
                        signal,
                        dt=dt,
                        method="welch" if (is_cpp and self.w_sde.value) else "fft",
                    )

                    x_nm, y_nm = np.asarray(traj.x) * 1e9, np.asarray(traj.y) * 1e9
                    R_nm = radius_eq * 1e9

                    # Plot 1: Orbit
                    self._update_line(self._state["lines"]["orbit"], x_nm, y_nm)
                    if len(x_nm) > 0:
                        self._update_line(
                            self._state["lines"]["point"], [x_nm[-1]], [y_nm[-1]]
                        )
                    self._state["patches"]["circle"].set_radius(R_nm)

                    # Centrowanie na punkcie równowagi z uwzględnieniem modelu
                    eq_nm = (0.0, 0.0)
                    if is_cpp:
                        eq = model.s_eq(ext_field)
                        eq_nm = (eq[0] * R_nm, eq[1] * R_nm)
                    else:
                        sx, sy = fcal.s_eq(field_state=ext_field)
                        eq_nm = (sx * R_nm, sy * R_nm)

                    lim = (
                        max(
                            R_nm,
                            np.max(np.abs(x_nm - eq_nm[0])) if len(x_nm) > 0 else 0,
                            np.max(np.abs(y_nm - eq_nm[1])) if len(y_nm) > 0 else 0,
                        )
                        * 1.05
                    )
                    axes[0].set_xlim(eq_nm[0] - lim, eq_nm[0] + lim)
                    axes[0].set_ylim(eq_nm[1] - lim, eq_nm[1] + lim)

                    # Plot 2: Time Signal
                    show_idx = max(
                        0, len(traj.t) - int(30e-9 / dt)
                    )  # Zoom na ostatnie 30 ns
                    self._update_line(
                        self._state["lines"]["signal"],
                        traj.t[show_idx:] * 1e9,
                        signal[show_idx:],
                    )
                    self._relim_and_rescale(axes[1])

                    # Plot 3: PSD in dB scale
                    if len(psd) > 0:
                        psd_db = 10 * np.log10(np.maximum(psd, 1e-20))
                        self._update_line(
                            self._state["lines"]["psd"], freqs * 1e-9, psd_db
                        )
                        axes[2].set_xlim(0, max(self.w_omega0.value * 2.5, 2.0))
                        self._relim_and_rescale(axes[2])

                    f_dom = (
                        traj.dominant_frequency_ghz
                        if hasattr(traj, "dominant_frequency_ghz")
                        else 0.0
                    )
                    lw = (
                        traj.linewidth_ghz * 1000
                        if hasattr(traj, "linewidth_ghz")
                        else 0.0
                    )

                    if is_cpp:
                        u_ss = (
                            traj.steady_state_radius_m / radius_eq
                            if hasattr(traj, "steady_state_radius_m")
                            else 0.0
                        )
                        info = f"Model: {self.w_model_type.value} | p={model.polarity:+d} | c={self.w_c.value:+d} | Radius u={u_ss:.3f} | f_peak={f_dom:.3f} GHz | Δf={lw:.1f} MHz"
                    else:
                        x_ss = x_nm[-1] if len(x_nm) > 0 else 0.0
                        y_ss = y_nm[-1] if len(y_nm) > 0 else 0.0
                        info = f"Model: CIP | p={model.polarity:+d} | c={self.w_c.value:+d} | Final Pos: ({x_ss:.1f}, {y_ss:.1f}) nm"

                    self._state["texts"]["suptitle"].set_text(info)

                self._state["fig"].canvas.draw_idle()
                self._state["fig"].canvas.flush_events()
                self.w_status.value = "<b style='color:green;'>Status: Completed ✅</b>"

            except Exception:
                import traceback

                self.w_status.value = f"<b style='color:red;'>Simulation Error:</b><br><pre>{traceback.format_exc()}</pre>"

        self._update_hud()


def build_thiele_dashboard(analyzer: Any | None = None, **kwargs):
    _ = analyzer
    dash = ThieleInteractiveDashboard()
    return dash.show(**kwargs)


__all__ = [
    "ThieleInteractiveDashboard",
    "build_thiele_dashboard",
    "proxy_signal_from_trajectory",
    "proxy_psd",
]
