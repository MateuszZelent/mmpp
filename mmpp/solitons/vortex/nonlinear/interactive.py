"""Interactive dashboard helpers for Thiele vortex analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mmpp.analytical import ellipse_area, slonczewski_mtj_efficiency

from ..core.models import TrajectoryResult

try:
    import ipywidgets as widgets

    _HAS_WIDGETS = True
except ImportError:  # pragma: no cover - optional dependency
    widgets = None  # type: ignore[assignment]
    _HAS_WIDGETS = False

try:
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    _HAS_MATPLOTLIB = False


def _normalize_polarizer(polarizer: tuple[float, float, float]) -> tuple[float, float, float]:
    px, py, pz = float(polarizer[0]), float(polarizer[1]), float(polarizer[2])
    norm = float(np.sqrt(px * px + py * py + pz * pz))
    if norm <= 1e-30:
        raise ValueError("polarizer vector norm must be positive")
    return px / norm, py / norm, pz / norm


def proxy_signal_from_trajectory(
    trajectory: TrajectoryResult,
    *,
    disk_radius: float | None = None,
    polarizer: tuple[float, float, float] = (1.0, 0.0, 0.0),
    center: tuple[float, float] | None = None,
    cubic: float = 0.0,
) -> np.ndarray:
    """Build MTJ readout proxy signal from core trajectory."""
    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)

    if center is None:
        x0 = float(np.mean(x)) if x.size else 0.0
        y0 = float(np.mean(y)) if y.size else 0.0
    else:
        x0 = float(center[0])
        y0 = float(center[1])

    if disk_radius is None:
        disk_radius = float(trajectory.metadata.get("disk_radius", np.nan))
    if not np.isfinite(disk_radius) or float(disk_radius) <= 0.0:
        radii = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        disk_radius = max(float(np.percentile(radii, 95)) * 1.1, 1e-12)
    r_norm = float(disk_radius)

    x_reduced = (x - x0) / r_norm
    y_reduced = (y - y0) / r_norm

    px, py, _ = _normalize_polarizer(polarizer)
    base = px * x_reduced + py * y_reduced
    signal = base + float(cubic) * (base**3)
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

    if method_norm == "welch":
        try:
            from scipy.signal import welch

            nperseg_eff = min(sig.size, int(nperseg) if nperseg is not None else max(32, sig.size // 8))
            f, p = welch(sig, fs=1.0 / dt, nperseg=nperseg_eff, detrend="constant", scaling="density")
            return np.asarray(f, dtype=float), np.asarray(p, dtype=float)
        except Exception:
            method_norm = "fft"

    if method_norm == "fft":
        n = sig.size
        fft = np.fft.rfft(sig)
        f = np.fft.rfftfreq(n, d=dt)
        p = (np.abs(fft) ** 2) / max(float(n), 1.0)
        return np.asarray(f, dtype=float), np.asarray(p, dtype=float)

    return np.array([], dtype=float), np.array([], dtype=float)


@dataclass
class ThieleInteractiveDashboard:
    """Ipywidgets dashboard for fast/full CPP Thiele exploration.

    Uses a **persistent** matplotlib figure with in-place data updates
    (``set_xdata`` / ``set_ydata``) to avoid flickering.  The figure is
    created once on ``show()`` and then mutated by the ``_render``
    callback.
    """

    analyzer: Any

    def _require_widgets(self) -> None:
        if not _HAS_WIDGETS:
            raise ImportError("ipywidgets is required for interactive dashboard")
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for interactive dashboard")

    # ------------------------------------------------------------------
    # Smooth-update helper
    # ------------------------------------------------------------------

    @staticmethod
    def _update_line(line, xdata, ydata):
        """Update an existing Line2D in place."""
        line.set_xdata(xdata)
        line.set_ydata(ydata)

    @staticmethod
    def _relim_and_rescale(ax):
        """Recompute limits from data and rescale."""
        ax.relim()
        ax.autoscale_view()

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def show(
        self,
        *,
        geometry_mode: str = "disk",
        disk_diameter_nm: float | None = None,
        size_x_nm: float = 220.0,
        size_y_nm: float = 120.0,
        thickness_nm: float = 20.0,
        ms_kA_per_m: float = 800.0,
        alpha: float = 0.01,
        pol: float = 0.56,
        lambd: float = 1.2,
        cos_theta_eff: float = 0.5,
        angle_deg: float = 20.0,
        current_mA: float = 8.0,
        b_ext_mt: float = 0.0,
        omega0_ghz: float = 0.9,
        N: float = 0.25,
        temperature_k: float = 300.0,
        noise_scale: float = 1.0,
        t_end_ns: float = 120.0,
        dt_ps: float = 10.0,
        fast_mode: bool = False,
        use_sde: bool = True,
        sde_seed: int | None = 0,
        figsize: tuple[float, float] | None = None,
        dpi: int = 100,
    ):
        """Create and return interactive widget panel."""
        self._require_widgets()

        dpi_value = int(dpi)
        if dpi_value <= 0:
            raise ValueError("dpi must be positive")

        if figsize is not None:
            if len(figsize) != 2:
                raise ValueError("figsize must be a tuple of (width, height)")
            figsize_value = (float(figsize[0]), float(figsize[1]))
            if figsize_value[0] <= 0.0 or figsize_value[1] <= 0.0:
                raise ValueError("figsize values must be positive")
        else:
            figsize_value = None

        mode_value = str(geometry_mode).strip().lower()
        if mode_value not in {"disk", "ellipse_eq"}:
            raise ValueError("geometry_mode must be 'disk' or 'ellipse_eq'")
        disk_diam_init = float(200.0 if disk_diameter_nm is None else disk_diameter_nm)
        if disk_diam_init <= 0.0:
            raise ValueError("disk_diameter_nm must be positive")

        # Geometry/material controls
        w_geom_mode = widgets.ToggleButtons(
            description="geom",
            options=[("disk", "disk"), ("ellipse→eq", "ellipse_eq")],
            value=mode_value,
        )
        w_disk_d = widgets.FloatSlider(description="D [nm]", value=disk_diam_init, min=40.0, max=500.0, step=5.0)
        w_size_x = widgets.FloatSlider(description="sizeX [nm]", value=size_x_nm, min=40.0, max=500.0, step=5.0)
        w_size_y = widgets.FloatSlider(description="sizeY [nm]", value=size_y_nm, min=40.0, max=500.0, step=5.0)
        w_thick = widgets.FloatSlider(description="L [nm]", value=thickness_nm, min=2.0, max=80.0, step=1.0)
        w_ms = widgets.FloatSlider(description="Ms [kA/m]", value=ms_kA_per_m, min=200.0, max=1600.0, step=10.0)
        w_alpha = widgets.FloatSlider(description="alpha", value=alpha, min=0.001, max=0.1, step=0.001)

        # STT / readout controls
        w_pol = widgets.FloatSlider(description="Pol", value=pol, min=0.05, max=0.95, step=0.01)
        w_lambda = widgets.FloatSlider(description="Lambda", value=lambd, min=0.5, max=3.0, step=0.05)
        w_cos = widgets.FloatSlider(description="cosθ_eff", value=cos_theta_eff, min=-1.0, max=1.0, step=0.01)
        w_angle = widgets.FloatSlider(description="angle [deg]", value=angle_deg, min=-180.0, max=180.0, step=1.0)
        w_current = widgets.FloatSlider(description="I [mA]", value=current_mA, min=-30.0, max=30.0, step=0.1)
        w_bext = widgets.FloatSlider(description="B_ext [mT]", value=b_ext_mt, min=-500.0, max=500.0, step=1.0)

        # Model controls
        w_omega0 = widgets.FloatSlider(description="omega0 [GHz]", value=omega0_ghz, min=0.1, max=20.0, step=0.05)
        w_n = widgets.FloatSlider(description="N", value=N, min=-2.0, max=3.0, step=0.01)
        w_temp = widgets.FloatSlider(description="T [K]", value=temperature_k, min=0.0, max=800.0, step=5.0)
        w_noise = widgets.FloatSlider(description="noise", value=noise_scale, min=0.0, max=8.0, step=0.05)
        w_tend = widgets.FloatSlider(description="t_end [ns]", value=t_end_ns, min=10.0, max=400.0, step=5.0)
        w_dt = widgets.FloatSlider(description="dt [ps]", value=dt_ps, min=1.0, max=100.0, step=1.0)
        w_fast = widgets.Checkbox(description="fast mode", value=bool(fast_mode))
        w_sde = widgets.Checkbox(description="use SDE", value=bool(use_sde))
        w_seed = widgets.IntText(description="seed", value=0 if sde_seed is None else int(sde_seed))

        out = widgets.Output()

        # ── Persistent figure state ──────────────────────────────
        # We create one figure for "fast" mode and one for "full" mode.
        # Each is lazily initialised and then updated in-place.
        _state: dict[str, Any] = {
            "fig": None,
            "axes": None,
            "lines": {},          # {name: Line2D}
            "patches": {},        # {name: patch}
            "texts": {},          # {name: Text}
            "mode": None,         # "fast" or "full"
        }

        def _ensure_figure(mode: str):
            """Create or recycle the persistent figure.

            If the mode switches (fast↔full) we tear down and rebuild,
            but within the same mode we only update data in-place —
            no flicker.
            """
            if _state["mode"] == mode and _state["fig"] is not None:
                return  # reuse current figure

            # Tear down previous figure
            if _state["fig"] is not None:
                plt.close(_state["fig"])

            _state["lines"].clear()
            _state["patches"].clear()
            _state["texts"].clear()

            if mode == "fast":
                fs = (12.0, 4.0) if figsize_value is None else figsize_value
                fig, axes = plt.subplots(1, 2, figsize=fs, dpi=dpi_value)
                (line_fj,) = axes[0].plot([], [], color="tab:blue")
                vline = axes[0].axvline(0, color="tab:red", linestyle="--", alpha=0.6)
                axes[0].set_xlabel("J [GA/m²]")
                axes[0].set_ylabel("f [GHz]")
                axes[0].set_title("Fast prediction: f(J)")
                axes[0].grid(True, alpha=0.3)
                axes[1].axis("off")
                info_text = axes[1].text(0.0, 1.0, "", va="top", family="monospace",
                                         transform=axes[1].transAxes)
                _state["lines"]["fj"] = line_fj
                _state["lines"]["vline"] = vline
                _state["texts"]["info"] = info_text
            else:
                fs = (15.0, 4.5) if figsize_value is None else figsize_value
                fig, axes = plt.subplots(1, 3, figsize=fs, dpi=dpi_value)
                # Orbit panel
                (line_orbit,) = axes[0].plot([], [], color="tab:blue", zorder=2)
                circle = plt.Circle((0, 0), 1, fill=False, color="tab:gray",
                                    linestyle="--", linewidth=1.2, alpha=0.9, zorder=1)
                axes[0].add_patch(circle)
                axes[0].set_xlabel("X [nm]")
                axes[0].set_ylabel("Y [nm]")
                axes[0].set_title("Core orbit")
                axes[0].set_aspect("equal", adjustable="box")
                axes[0].grid(True, alpha=0.3)
                # Signal panel
                (line_sig,) = axes[1].plot([], [], color="tab:orange")
                axes[1].set_xlabel("t [ns]")
                axes[1].set_ylabel("proxy signal [a.u.]")
                axes[1].set_title("MTJ readout proxy")
                axes[1].grid(True, alpha=0.3)
                # PSD panel
                (line_psd,) = axes[2].plot([], [], color="tab:green")
                axes[2].set_xlabel("f [GHz]")
                axes[2].set_ylabel("PSD [a.u.]")
                axes[2].set_title("Proxy PSD")
                axes[2].grid(True, alpha=0.3)

                suptitle = fig.suptitle("", fontsize=10)

                _state["lines"]["orbit"] = line_orbit
                _state["patches"]["circle"] = circle
                _state["lines"]["signal"] = line_sig
                _state["lines"]["psd"] = line_psd
                _state["texts"]["suptitle"] = suptitle

            fig.tight_layout()
            _state["fig"] = fig
            _state["axes"] = axes if hasattr(axes, '__len__') else [axes]
            _state["mode"] = mode

        def _sync_geometry_controls() -> None:
            disk_mode = w_geom_mode.value == "disk"
            w_disk_d.disabled = not disk_mode
            w_size_x.disabled = disk_mode
            w_size_y.disabled = disk_mode

        def _sync_sde_controls() -> None:
            w_seed.disabled = not bool(w_sde.value)

        def _render(*_args):
            with out:
                # ── Compute parameters ───────────────────────────
                if w_geom_mode.value == "disk":
                    radius_eq = 0.5 * w_disk_d.value * 1e-9
                    area = math.pi * (radius_eq**2)
                    geom_desc = f"disk D={w_disk_d.value:.1f} nm"
                else:
                    area = ellipse_area(w_size_x.value * 1e-9, w_size_y.value * 1e-9)
                    radius_eq = math.sqrt(area / math.pi)
                    geom_desc = f"ellipse→eq ({w_size_x.value:.1f} x {w_size_y.value:.1f} nm)"
                peff = slonczewski_mtj_efficiency(w_pol.value, w_lambda.value, w_cos.value)
                current_density = (w_current.value * 1e-3) / area

                angle_rad = math.radians(w_angle.value)
                polarizer = (math.cos(angle_rad), math.sin(angle_rad), 0.0)

                material = {
                    "Ms": w_ms.value * 1e3,
                    "alpha": w_alpha.value,
                    "P": peff,
                }
                geometry = {
                    "R": radius_eq,
                    "L": w_thick.value * 1e-9,
                }
                b_ext_tesla = w_bext.value * 1e-3

                omega0 = 2.0 * math.pi * w_omega0.value * 1e9
                j_th = self.analyzer.threshold_current_dc(
                    material=material,
                    geometry=geometry,
                    omega0=omega0,
                    N=w_n.value,
                    B_ext=b_ext_tesla,
                )
                f_pred = self.analyzer.predict_frequency_dc(
                    current_density,
                    material=material,
                    geometry=geometry,
                    omega0=omega0,
                    N=w_n.value,
                    allow_edge=True,
                    B_ext=b_ext_tesla,
                )
                f_target = f_pred if f_pred is not None else abs(w_omega0.value * 1e9)
                opt = self.analyzer.optimize_current_for_target_frequency(
                    f_target,
                    material=material,
                    geometry=geometry,
                    omega0=omega0,
                    N=w_n.value,
                    J_bounds=(1.01 * j_th, 8.0 * j_th),
                    allow_edge=True,
                    B_ext=b_ext_tesla,
                )

                # ── Fast mode ────────────────────────────────────
                if w_fast.value:
                    _ensure_figure("fast")
                    fig = _state["fig"]
                    axes = _state["axes"]

                    j_min = 1.01 * j_th
                    j_max = 4.0 * j_th
                    j_grid = np.linspace(j_min, j_max, 160)
                    f_grid = np.array(
                        [
                            self.analyzer.predict_frequency_dc(
                                val,
                                material=material,
                                geometry=geometry,
                                omega0=omega0,
                                N=w_n.value,
                                allow_edge=True,
                                B_ext=b_ext_tesla,
                            )
                            for val in j_grid
                        ],
                        dtype=float,
                    )

                    # Update f(J) line in-place
                    self._update_line(_state["lines"]["fj"],
                                      j_grid * 1e-9, f_grid * 1e-9)
                    # Update vertical marker
                    _state["lines"]["vline"].set_xdata([current_density * 1e-9,
                                                         current_density * 1e-9])
                    self._relim_and_rescale(axes[0])

                    # Update info text
                    text_lines = [
                        f"Geometry: {geom_desc}",
                        f"Area: {area:.3e} m²",
                        f"R_eq: {radius_eq*1e9:.2f} nm",
                        f"B_ext: {w_bext.value:.1f} mT",
                        f"P_eff: {peff:.4f}",
                        f"J_dc: {current_density*1e-9:.3f} GA/m²",
                        f"J_th: {j_th*1e-9:.3f} GA/m²",
                        f"f_pred: {('n/a' if f_pred is None else f'{f_pred*1e-9:.3f} GHz')}",
                        f"J_opt(target=f_pred): {opt.current_density_ga_per_m2:.3f} GA/m²",
                    ]
                    _state["texts"]["info"].set_text("\n".join(text_lines))

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    return

                # ── Full simulation mode ─────────────────────────
                _ensure_figure("full")
                fig = _state["fig"]
                axes = _state["axes"]

                t_span = (0.0, w_tend.value * 1e-9)
                dt = w_dt.value * 1e-12
                if w_sde.value:
                    traj = self.analyzer.simulate_cpp_sde(
                        material=material,
                        geometry=geometry,
                        omega0=omega0,
                        N=w_n.value,
                        current_density=current_density,
                        t_span=t_span,
                        dt=dt,
                        temperature_k=w_temp.value,
                        noise_scale=w_noise.value,
                        seed=int(w_seed.value),
                        s0=(0.0, 0.0),
                        B_ext=b_ext_tesla,
                    )
                else:
                    traj = self.analyzer.simulate_cpp(
                        material=material,
                        geometry=geometry,
                        omega0=omega0,
                        N=w_n.value,
                        current_density=current_density,
                        t_span=t_span,
                        dt=dt,
                        s0=(1e-3, 0.0),
                        B_ext=b_ext_tesla,
                    )

                signal = proxy_signal_from_trajectory(
                    traj,
                    disk_radius=radius_eq,
                    polarizer=polarizer,
                )
                freq, psd = proxy_psd(signal, dt=dt, method="welch")

                x_nm = np.asarray(traj.x, dtype=float) * 1e9
                y_nm = np.asarray(traj.y, dtype=float) * 1e9
                disk_radius_nm = radius_eq * 1e9
                if w_geom_mode.value == "disk":
                    orbit_limit_nm = max(disk_radius_nm, 1e-12)
                else:
                    orbit_limit_nm = max(disk_radius_nm, 1e-12) * 1.05

                # Update orbit line
                self._update_line(_state["lines"]["orbit"], x_nm, y_nm)
                # Update disk circle radius
                _state["patches"]["circle"].set_radius(disk_radius_nm)
                axes[0].set_xlim(-orbit_limit_nm, orbit_limit_nm)
                axes[0].set_ylim(-orbit_limit_nm, orbit_limit_nm)

                # Update signal line
                self._update_line(_state["lines"]["signal"],
                                  traj.t * 1e9, signal)
                self._relim_and_rescale(axes[1])

                # Update PSD line
                self._update_line(_state["lines"]["psd"],
                                  freq * 1e-9, psd)
                self._relim_and_rescale(axes[2])

                # Update suptitle
                info = (
                    f"{geom_desc}  "
                    f"B_ext={w_bext.value:.1f} mT  "
                    f"J_dc={current_density*1e-9:.3f} GA/m²  "
                    f"J_th={j_th*1e-9:.3f} GA/m²  "
                    f"f_pred={('n/a' if f_pred is None else f'{f_pred*1e-9:.3f} GHz')}  "
                    f"J_opt={opt.current_density_ga_per_m2:.3f} GA/m²"
                )
                _state["texts"]["suptitle"].set_text(info)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

        for widget in (
            w_geom_mode,
            w_disk_d,
            w_size_x,
            w_size_y,
            w_thick,
            w_ms,
            w_alpha,
            w_pol,
            w_lambda,
            w_cos,
            w_angle,
            w_current,
            w_bext,
            w_omega0,
            w_n,
            w_temp,
            w_noise,
            w_tend,
            w_dt,
            w_fast,
            w_sde,
            w_seed,
        ):
            widget.observe(_render, names="value")

        w_geom_mode.observe(lambda *_: _sync_geometry_controls(), names="value")
        w_sde.observe(lambda *_: _sync_sde_controls(), names="value")

        controls = widgets.VBox(
            [
                widgets.HBox([w_geom_mode, w_disk_d, w_size_x, w_size_y, w_thick]),
                widgets.HBox([w_ms, w_alpha, w_bext]),
                widgets.HBox([w_pol, w_lambda, w_cos, w_angle, w_current]),
                widgets.HBox([w_omega0, w_n, w_temp, w_noise, w_seed]),
                widgets.HBox([w_tend, w_dt, w_fast, w_sde]),
            ]
        )
        root = widgets.VBox([controls, out])
        _sync_geometry_controls()
        _sync_sde_controls()
        _render()
        return root


def build_thiele_dashboard(analyzer: Any, **kwargs):
    """Build and return interactive dashboard widget for Thiele analysis."""
    dash = ThieleInteractiveDashboard(analyzer=analyzer)
    return dash.show(**kwargs)


__all__ = [
    "ThieleInteractiveDashboard",
    "build_thiele_dashboard",
    "proxy_signal_from_trajectory",
    "proxy_psd",
]
