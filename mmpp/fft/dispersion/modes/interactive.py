"""
Interactive Jupyter widget interface for dispersion mode analysis.

Provides an ipywidgets-based interactive exploration of BZ-folded
dispersion relations with real-time parameter adjustment.

Features:
- Stacked layout: dispersion plot on top, mode visualization (2D spatial) below
- Click on dispersion to select mode → shows marker and mask positions
- Slider for N_BZ (number of Brillouin zones in mask)
- Option for +k/-k direction selection
- Real-time 2D spatial mode visualization m(x, y) using pre-computed S_complex

Algorithm for spatial mode reconstruction (following Rychły et al.):
1. Dispersion computation stores both S (power spectrum) and S_complex (phase-preserving)
2. User selects (k, f) on dispersion plot
3. Create BZ mask for k_0 ± n·G (all periodic copies)
4. Extract S_complex at f_0 and apply mask
5. IFFT only over k → propagation axis (FAST - no re-computation!)
6. Result: Spatial mode profile m(x, y) at selected (k_0, f_0)

Performance: Mode visualization is INSTANT because S_complex is pre-computed and cached.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

# Check for interactive dependencies
_HAS_WIDGETS = False
_HAS_MATPLOTLIB = False

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    _HAS_WIDGETS = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    import matplotlib

    _HAS_MATPLOTLIB = True
except ImportError:
    pass

if TYPE_CHECKING:
    from ..models import DispersionResult1D
    from ..interface import FFTDispersionInterface
    from .models import FoldedDispersionResult
    from .folding import BrillouinZoneFolding
    from .detection import BrillouinZoneDetector

from .profile import ModeProfile


class InteractiveDispersionModes:
    """
    Interactive widget for exploring Brillouin zone folded dispersion.

    Features:
    - Stacked layout: dispersion on top, mode visualization below
    - Click to select mode → visualize m(y, t) heatmap
    - Markers showing k and all k ± n×BZ positions included in mask
    - Slider for number of BZ in mask
    - Option for +k/-k direction selection
    """

    def __init__(self, dispersion_interface: FFTDispersionInterface):
        """Initialize the interactive modes analysis."""
        self.interface = dispersion_interface
        self.result: DispersionResult1D | None = None
        self.folded: FoldedDispersionResult | None = None

        # Lazy import components
        self._detector: BrillouinZoneDetector | None = None
        self._folder: BrillouinZoneFolding | None = None

        # Widget state
        self._widgets_created = False
        self._output: widgets.Output | None = None
        self._fig: Figure | None = None
        self._ax_disp: Axes | None = None
        self._ax_mode: Axes | None = None
        self._colorbar_disp = None
        self._colorbar_mode = None

        # Selection state
        self._selected_k = None
        self._selected_f = None
        self._mask_markers = []  # Artists for mask position markers
        
        # Animation state
        self._animation = None  # FuncAnimation object
        self._is_animating = False

        # Default parameters
        self._default_params = {
            "lattice_nm": 470.0,
            "n_bz_mask": 3,  # Number of BZ to include in mask
            "f_min_ghz": 0.0,
            "f_max_ghz": 10.0,
            "threshold": 0.01,
            "k_direction": "both",  # 'both', 'positive', 'negative'
            "cmap_disp": "viridis",
            "cmap_mode": "RdBu_r",
        }

        # Figure settings
        self._dpi = 150
        self._figsize = (10, 10)

    @property
    def detector(self) -> BrillouinZoneDetector:
        """Lazy-load detector."""
        if self._detector is None:
            from .detection import BrillouinZoneDetector

            self._detector = BrillouinZoneDetector()
        return self._detector

    def _get_folder(self, lattice_constant: float, n_periods: int) -> BrillouinZoneFolding:
        """Get or create folder with specified parameters."""
        from .folding import BrillouinZoneFolding

        return BrillouinZoneFolding(lattice_constant, n_periods)

    def fold(
        self,
        result: DispersionResult1D | None = None,
        lattice_constant: float | None = None,
        n_periods: int = 3,
        peak_threshold: float = 0.01,
        **compute_kwargs,
    ) -> FoldedDispersionResult:
        """Fold dispersion to first Brillouin zone (non-interactive)."""
        if result is None:
            result = self.interface.compute_1d(**compute_kwargs)
        self.result = result

        if lattice_constant is None:
            lattice_constant = 470e-9
            logger.info("Using default lattice constant: %.1f nm", lattice_constant * 1e9)

        folder = self._get_folder(lattice_constant, n_periods)
        self.folded = folder.fold_dispersion(result, peak_threshold)

        return self.folded

    def plot_interactive(
        self,
        result: DispersionResult1D | None = None,
        figsize: tuple[float, float] = (10, 10),
        dpi: int = 150,
        lattice_constant_nm: float | None = None,
        **compute_kwargs,
    ):
        """
        Launch interactive widget for dispersion mode exploration.

        Parameters
        ----------
        result : DispersionResult1D, optional
            Pre-computed dispersion. If None, uses self.result or computes automatically.
        figsize : tuple
            Figure size (width, height) in inches
        dpi : int
            Figure DPI (default 150)
        lattice_constant_nm : float, optional
            Initial lattice constant in nm. If None, uses default from dispersion_modes().
        **compute_kwargs : dict
            Extra kwargs passed to compute_1d if result needs to be computed.
        """
        if not _HAS_WIDGETS:
            raise ImportError(
                "ipywidgets is required for interactive mode. "
                "Install with: pip install ipywidgets"
            )
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting.")

        self._dpi = dpi
        self._figsize = figsize

        # Priority: explicit result > self.result > compute
        if result is not None:
            self.result = result
        elif self.result is None:
            logger.info("Computing dispersion result...")
            self.result = self.interface.compute_1d(**compute_kwargs)

        # Set lattice constant (use provided or keep default from dispersion_modes())
        if lattice_constant_nm is not None:
            self._default_params["lattice_nm"] = lattice_constant_nm

        # Set initial parameters from result
        f_max_ghz = self.result.f_axis.max() / 1e9
        self._default_params["f_max_ghz"] = min(f_max_ghz, 20.0)

        # Create widgets
        self._create_widgets()

        # Create the main layout
        main_layout = self._create_layout()

        # Display
        display(main_layout)

        # Initial plot
        self._initialize_figure()
        self._update_dispersion_plot()

    def _create_widgets(self):
        """Create all interactive widgets."""
        params = self._default_params

        # === Lattice parameters ===
        self.w_lattice = widgets.FloatSlider(
            value=params["lattice_nm"],
            min=50,
            max=2000,
            step=5,
            description="a [nm]:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        self.w_n_bz_mask = widgets.IntSlider(
            value=params["n_bz_mask"],
            min=1,
            max=10,
            step=1,
            description="N_BZ mask:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        # === Frequency range ===
        self.w_fmin = widgets.FloatSlider(
            value=params["f_min_ghz"],
            min=0,
            max=params["f_max_ghz"],
            step=0.1,
            description="f min [GHz]:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        self.w_fmax = widgets.FloatSlider(
            value=params["f_max_ghz"],
            min=0.1,
            max=params["f_max_ghz"] * 1.5,
            step=0.1,
            description="f max [GHz]:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        # === k-direction selection ===
        self.w_k_direction = widgets.Dropdown(
            options=[
                ("Both ±k", "both"),
                ("Only +k", "positive"),
                ("Only -k", "negative"),
            ],
            value=params["k_direction"],
            description="k-dirs:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        # === Mode visualization type ===
        self.w_mode_type = widgets.Dropdown(
            options=[
                ("Real part", "real"),
                ("Imaginary part", "imag"),
                ("Amplitude |M|", "abs"),
                ("Phase φ[M]", "phase"),
                ("Ampl×Phase", "ampl_phase"),
            ],
            value="real",
            description="Mode type:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        # === Colormaps ===
        self.w_cmap_disp = widgets.Dropdown(
            options=["viridis", "plasma", "cividis", "turbo", "inferno"],
            value=params["cmap_disp"],
            description="Cmap disp:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        self.w_cmap_mode = widgets.Dropdown(
            options=["RdBu_r", "seismic", "coolwarm", "bwr", "PiYG"],
            value="RdBu_r",
            description="Cmap mode:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        # === Buttons ===
        self.w_update = widgets.Button(
            description="🔄 Update",
            button_style="success",
            layout=widgets.Layout(width="95%"),
        )

        self.w_auto_detect = widgets.Button(
            description="🔍 Auto-detect a",
            button_style="info",
            layout=widgets.Layout(width="95%"),
        )

        self.w_animate = widgets.Button(
            description="🎬 Animate Mode",
            button_style="warning",
            layout=widgets.Layout(width="95%"),
            tooltip="Toggle mode oscillation animation (full 2π cycle)",
        )

        # === Info display ===
        self.w_info = widgets.HTML(
            value="<small>Click on dispersion to select mode (k, f)</small>",
        )

        self.w_mode_info = widgets.HTML(
            value="<small>No mode selected</small>",
        )

        self._output = widgets.Output(layout=widgets.Layout(width="100%", height="auto"))

        # === Connect callbacks ===
        self.w_update.on_click(self._on_update)
        self.w_auto_detect.on_click(self._on_auto_detect)
        self.w_animate.on_click(self._on_animate)

        # Connect changes that should update immediately
        # w_lattice also updates BZ lines on dispersion plot
        for w in [self.w_cmap_disp, self.w_fmin, self.w_fmax, self.w_lattice]:
            w.observe(self._on_display_param_change, names="value")

        # Connect mode visualization params
        # w_lattice affects BZ mask positions for mode extraction
        for w in [self.w_n_bz_mask, self.w_k_direction, self.w_mode_type, self.w_cmap_mode, self.w_lattice]:
            w.observe(self._on_mode_param_change, names="value")
        
        # Watch n_bz to show/hide k-direction widget
        self.w_n_bz_mask.observe(self._on_n_bz_change, names="value")
        self._update_k_direction_visibility()

        self._widgets_created = True

    def _create_layout(self) -> widgets.Widget:
        """Create layout with controls on left, stacked plots on right."""

        # Left panel: controls
        left_panel = widgets.VBox(
            [
                widgets.HTML("<b>🌊 BZ Mode Analysis</b>"),
                widgets.HTML("<hr style='margin:2px'>"),
                widgets.HTML("<small><b>Lattice</b></small>"),
                self.w_lattice,
                self.w_auto_detect,
                widgets.HTML("<small><b>Mask Settings</b></small>"),
                self.w_n_bz_mask,
                self.w_k_direction,
                widgets.HTML("<small><b>Mode Visualization</b></small>"),
                self.w_mode_type,
                widgets.HTML("<small><b>Frequency Range</b></small>"),
                self.w_fmin,
                self.w_fmax,
                widgets.HTML("<small><b>Display</b></small>"),
                self.w_cmap_disp,
                self.w_cmap_mode,
                self.w_update,
                self.w_animate,
                widgets.HTML("<hr style='margin:5px'>"),
                self.w_info,
                widgets.HTML("<hr style='margin:5px'>"),
                widgets.HTML("<small><b>Selected Mode</b></small>"),
                self.w_mode_info,
            ],
            layout=widgets.Layout(
                width="200px",
                padding="5px",
                border="1px solid #ddd",
            ),
        )

        # Right panel: stacked plots
        right_panel = widgets.VBox(
            [
                self._output,
            ],
            layout=widgets.Layout(
                width="calc(100% - 220px)",
                min_width="700px",
            ),
        )

        # Main layout
        main = widgets.HBox(
            [
                left_panel,
                right_panel,
            ],
            layout=widgets.Layout(
                width="100%",
            ),
        )

        return main

    def _initialize_figure(self):
        """Create the matplotlib figure with two subplots."""
        with self._output:
            clear_output(wait=True)

            plt.ioff()

            # Create figure with 2 rows: dispersion on top, mode viz below
            self._fig, (self._ax_disp, self._ax_mode) = plt.subplots(
                2,
                1,
                figsize=self._figsize,
                dpi=self._dpi,
                gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.25},
            )

            # Connect click events
            self._fig.canvas.mpl_connect("button_press_event", self._on_click)

            plt.ion()
            plt.show()

    def _on_click(self, event):
        """Handle mouse click on the dispersion plot."""
        if event.inaxes != self._ax_disp:
            return

        k_clicked = event.xdata * 1e6  # μm^-1 → m^-1
        f_clicked = event.ydata * 1e9  # GHz → Hz

        self._selected_k = k_clicked
        self._selected_f = f_clicked

        # Update mode info
        self.w_mode_info.value = (
            f"<small><b>k</b> = {k_clicked/1e6:.2f} rad/μm<br>"
            f"<b>f</b> = {f_clicked/1e9:.2f} GHz</small>"
        )

        # Update visualization
        self._update_mode_visualization()

    def _on_update(self, _):
        """Handle update button click."""
        self._update_dispersion_plot()
        if self._selected_k is not None:
            self._update_mode_visualization()

    def _on_display_param_change(self, change):
        """Handle display parameter changes."""
        self._update_dispersion_plot()

    def _on_mode_param_change(self, change):
        """Handle mode visualization parameter changes."""
        if self._selected_k is not None:
            self._update_mode_visualization()
    
    def _on_n_bz_change(self, change):
        """Handle N_BZ slider change - update k-direction visibility."""
        self._update_k_direction_visibility()
        if self._selected_k is not None:
            self._update_mode_visualization()
    
    def _update_k_direction_visibility(self):
        """Show/hide k-direction dropdown based on N_BZ value.
        
        When N_BZ=1, only one k-point is selected so k-direction is meaningless.
        """
        n_bz = self.w_n_bz_mask.value
        if n_bz <= 1:
            # Hide k-direction - only one point, direction doesn't matter
            self.w_k_direction.layout.display = 'none'
        else:
            # Show k-direction - multiple BZ so direction matters
            self.w_k_direction.layout.display = ''

    def _on_auto_detect(self, _):
        """Handle auto-detect button click."""
        if self.result is None:
            return

        methods = ["autocorrelation", "fft", "peak_spacing"]
        results = []

        for method in methods:
            try:
                a = self.detector.detect_lattice_constant(self.result, method=method)
                if 50e-9 < a < 5000e-9:
                    results.append(a)
            except Exception:
                pass

        if results:
            detected_a = np.median(results)
            self.w_lattice.value = detected_a * 1e9
            self.w_info.value = f"<small style='color:green'>Detected: {detected_a*1e9:.0f} nm</small>"
        else:
            self.w_info.value = "<small style='color:orange'>Detection failed</small>"

    def _ensure_animation_state(self):
        """Backfill animation attributes for legacy/stale live instances."""
        if not hasattr(self, "_animation"):
            self._animation = None
        if not hasattr(self, "_is_animating"):
            self._is_animating = False

    def _on_animate(self, _):
        """Toggle animation of selected mode in the mode visualization panel."""
        self._ensure_animation_state()

        if self._selected_k is None or self._selected_f is None:
            self.w_info.value = "<small style='color:red'>⚠️ Select a mode first (click on dispersion)</small>"
            return
        
        # Toggle animation on/off
        if self._is_animating:
            # Stop animation
            self._stop_animation()
            self.w_info.value = "<small>Animation stopped</small>"
            self.w_animate.description = "🎬 Animate Mode"
            self.w_animate.button_style = "warning"
            # Restore static view
            self._update_mode_visualization()
            return
        
        try:
            from matplotlib.animation import FuncAnimation
            
            # Get parameters
            a = self.w_lattice.value * 1e-9
            n_bz = self.w_n_bz_mask.value
            k_direction = self.w_k_direction.value
            mode_type = self.w_mode_type.value
            
            # Extract complex mode data
            x_axis, y_axis, mode_2d_complex = self._extract_mode_2d_custom(
                k_0=self._selected_k,
                f_0=self._selected_f,
                lattice_constant=a,
                n_bz=n_bz,
                k_direction=k_direction,
            )
            
            # Time parameters for full 2π cycle
            period_s = 1.0 / self._selected_f  # Full period
            omega = 2 * np.pi * self._selected_f
            n_frames = 60  # Smooth animation
            fps = 30
            
            # Time array for one complete cycle (0 to T)
            time_array = np.linspace(0, period_s, n_frames, endpoint=False)
            
            # Pre-compute all frames for selected visualization mode
            mode_labels = {
                'real': 'Re[M]',
                'imag': 'Im[M]',
                'abs': '|M|',
                'phase': 'φ[M]',
                'ampl_phase': 'Ampl×Phase',
            }
            mode_label = mode_labels.get(mode_type, mode_type)
            is_rgb = (mode_type == 'ampl_phase')

            if mode_type in ['real', 'imag']:
                frames = []
                for t in time_array:
                    m_t_complex = mode_2d_complex * np.exp(-1j * omega * t)
                    if mode_type == 'real':
                        m_t = np.real(m_t_complex)
                    else:
                        m_t = np.imag(m_t_complex)
                    frames.append(m_t)

                frames = np.array(frames)  # (n_frames, N_y, N_x)
                vmax = np.max(np.abs(frames))
                if vmax < 1e-20:
                    vmax = 1.0
                vmin = -vmax
                cmap = self.w_cmap_mode.value
                cbar_label = "Re[M(t)]" if mode_type == 'real' else "Im[M(t)]"

            elif mode_type == 'abs':
                # |M| is the spatial envelope and stays constant in time for a pure harmonic mode.
                amplitude = np.abs(mode_2d_complex)
                frames = np.repeat(amplitude[np.newaxis, :, :], n_frames, axis=0)
                vmin = 0.0
                vmax = np.max(amplitude)
                if vmax < 1e-20:
                    vmax = 1.0
                cmap = 'hot'
                cbar_label = "|M|"

            elif mode_type == 'phase':
                frames = []
                for t in time_array:
                    m_t_complex = mode_2d_complex * np.exp(-1j * omega * t)
                    frames.append(np.angle(m_t_complex))

                frames = np.array(frames)
                vmin = -np.pi
                vmax = np.pi
                cmap = 'hsv'
                cbar_label = "φ[M(t)] [rad]"

            elif mode_type == 'ampl_phase':
                from ..utils import create_amplitude_phase_colormap

                amplitude_ref = np.abs(mode_2d_complex)
                amp_min = float(amplitude_ref.min())
                amp_max = float(amplitude_ref.max())

                frames = []
                for t in time_array:
                    m_t_complex = mode_2d_complex * np.exp(-1j * omega * t)
                    frames.append(
                        create_amplitude_phase_colormap(
                            m_t_complex,
                            amp_min=amp_min,
                            amp_max=amp_max,
                        )
                    )

                frames = np.array(frames)  # (n_frames, N_y, N_x, 3)
                vmin = None
                vmax = None
                cmap = None
                cbar_label = None

            else:
                raise ValueError(f"Unknown mode_type='{mode_type}'")
            
            # Setup axes
            ax = self._ax_mode
            ax.clear()
            
            # Remove old mode colorbar
            if self._colorbar_mode is not None:
                try:
                    self._colorbar_mode.remove()
                except Exception:
                    pass
                self._colorbar_mode = None
            
            # Extent
            x_um = x_axis * 1e6
            y_um = y_axis * 1e6
            extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]
            
            # Initial frame
            if is_rgb:
                im = ax.imshow(
                    frames[0],
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    interpolation="bilinear",
                )
            else:
                im = ax.imshow(
                    frames[0],
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="bilinear",
                )
            
            # Colorbar only for scalar data
            if not is_rgb:
                self._colorbar_mode = self._fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                self._colorbar_mode.set_label(cbar_label, fontsize=9)
            
            # Labels
            ax.set_xlabel("x [μm]", fontsize=10)
            ax.set_ylabel("y [μm]", fontsize=10)
            
            # Title (will be updated each frame)
            k_str = f"k = {self._selected_k/1e6:.2f} rad/μm"
            f_str = f"f = {self._selected_f/1e9:.2f} GHz"
            title = ax.set_title(
                f"{mode_label} Mode | {k_str}, {f_str} | t=0.00 ns | φ=0.00°",
                fontsize=11
            )
            ax.tick_params(labelsize=9)
            
            # Animation update function
            def update(frame_idx):
                im.set_data(frames[frame_idx])
                t_ns = time_array[frame_idx] * 1e9
                phase_deg = (time_array[frame_idx] / period_s) * 360
                title.set_text(
                    f"{mode_label} Mode | {k_str}, {f_str} | t={t_ns:.2f} ns | φ={phase_deg:.0f}°"
                )
                return [im, title]
            
            # Create animation
            interval = 1000 / fps  # ms between frames
            self._animation = FuncAnimation(
                self._fig,
                update,
                frames=n_frames,
                interval=interval,
                blit=True,
                repeat=True,
            )
            
            self._is_animating = True
            self.w_animate.description = "⏸️ Stop Animation"
            self.w_animate.button_style = "danger"
            abs_note = " | |M| is time-invariant for harmonic modes" if mode_type == 'abs' else ""
            self.w_info.value = (
                f"<small style='color:green'>🎬 Animating: {n_frames} frames, "
                f"T={period_s*1e9:.2f} ns (1 period = 2π), mode={mode_label}{abs_note}</small>"
            )
            
            # Redraw
            self._fig.canvas.draw_idle()
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Animation failed:\n{tb}")
            self.w_info.value = f"<small style='color:red'>❌ Animation error: {str(e)[:50]}</small>"
            self._is_animating = False
            self.w_animate.description = "🎬 Animate Mode"
            self.w_animate.button_style = "warning"
    
    def _stop_animation(self):
        """Stop the current animation."""
        self._ensure_animation_state()

        if self._animation is not None:
            self._animation.event_source.stop()
            self._animation = None
        self._is_animating = False

    def _update_dispersion_plot(self):
        """Update the dispersion heatmap."""
        if self.result is None or self._ax_disp is None:
            return

        ax = self._ax_disp

        # Clear axes
        ax.clear()

        # Remove old colorbar
        if self._colorbar_disp is not None:
            try:
                self._colorbar_disp.remove()
            except Exception:
                pass
            self._colorbar_disp = None

        # Get data
        S = self.result.S.T  # (Nf, Nk)
        k_axis = self.result.k_axis / 1e6  # rad/μm
        f_axis = self.result.f_axis / 1e9  # GHz

        # Apply frequency limits
        f_min = self.w_fmin.value
        f_max = self.w_fmax.value
        f_mask = (f_axis >= f_min) & (f_axis <= f_max)

        if np.sum(f_mask) < 2:
            ax.text(0.5, 0.5, "No data in frequency range", transform=ax.transAxes, ha="center")
            self._fig.canvas.draw_idle()
            return

        S = S[f_mask, :]
        f_axis_plot = f_axis[f_mask]

        extent = [k_axis[0], k_axis[-1], f_axis_plot[0], f_axis_plot[-1]]

        # Plot heatmap
        im = ax.imshow(
            np.log10(S + 1e-20),
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=self.w_cmap_disp.value,
            interpolation="bilinear",
        )

        self._colorbar_disp = self._fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        self._colorbar_disp.set_label("log₁₀(S)", fontsize=9)

        # Add BZ boundary lines (reciprocal lattice vectors G = 2π/a)
        a = self.w_lattice.value * 1e-9
        G = 2 * np.pi / a / 1e6  # rad/μm (reciprocal lattice vector)
        
        # Show multiple BZ boundaries within k-range
        k_range = ax.get_xlim()
        k_max = max(abs(k_range[0]), abs(k_range[1]))
        n_zones = int(np.ceil(k_max / G)) + 1
        
        for n in range(-n_zones, n_zones + 1):
            if n == 0:
                continue
            k_line = n * G
            if abs(k_line) <= k_max * 1.1:  # Show if within range
                alpha = 0.8 if abs(n) == 1 else 0.4  # Emphasize ±1G
                ax.axvline(k_line, color="red", linestyle="--",
                          linewidth=1.5 if abs(n) == 1 else 1.0, alpha=alpha)
        
        # Add k=0 line and legend
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.legend([f"BZ boundaries (G = {G:.1f} rad/μm)"], loc="upper right", fontsize=8)

        # Labels
        ax.set_xlabel(r"$k$ [rad/μm]", fontsize=10)
        ax.set_ylabel("f [GHz]", fontsize=10)
        ax.set_title(f"Dispersion S(k, f) | a = {a*1e9:.0f} nm | Click to select mode", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=9)

        # Set default k-axis limits to ±2 Brillouin zones (from -2G to +2G)
        k_limit = 2 * G  # ±2 zones (each zone spans G)
        ax.set_xlim(-k_limit, k_limit)

        # Redraw selection marker if exists
        if self._selected_k is not None and self._selected_f is not None:
            self._draw_selection_markers(ax)

        self._fig.canvas.draw_idle()

    def _draw_selection_markers(self, ax: Axes):
        """Draw markers showing selected (k, f) and all mask positions."""
        if self._selected_k is None:
            return

        k_sel = self._selected_k / 1e6  # rad/μm
        f_sel = self._selected_f / 1e9  # GHz

        a = self.w_lattice.value * 1e-9
        G = 2 * np.pi / a / 1e6  # rad/μm
        n_bz = self.w_n_bz_mask.value
        k_direction = self.w_k_direction.value

        # Draw main selection marker (red square)
        ax.plot(
            k_sel,
            f_sel,
            "rs",
            markersize=12,
            markerfacecolor="none",
            markeredgewidth=2,
            label="Selected",
        )

        # Draw all mask positions as circles
        k_axis = self.result.k_axis / 1e6
        k_min, k_max = k_axis.min(), k_axis.max()

        for n in range(-n_bz, n_bz + 1):
            if n == 0:
                continue  # Skip main selection

            k_copy = k_sel + n * G

            # Check k-direction filter
            if k_direction == "positive" and k_copy < 0:
                continue
            if k_direction == "negative" and k_copy > 0:
                continue

            # Check if within k-axis range
            if k_min <= k_copy <= k_max:
                ax.plot(
                    k_copy,
                    f_sel,
                    "o",
                    markersize=8,
                    markerfacecolor="none",
                    markeredgecolor="lime",
                    markeredgewidth=1.5,
                )

                # Draw connecting line
                ax.plot(
                    [k_sel, k_copy],
                    [f_sel, f_sel],
                    "g--",
                    linewidth=0.8,
                    alpha=0.5,
                )

        # Update info with mask positions
        mask_positions = []
        for n in range(-n_bz, n_bz + 1):
            k_copy = self._selected_k + n * (2 * np.pi / a)
            if k_direction == "positive" and k_copy < 0:
                continue
            if k_direction == "negative" and k_copy > 0:
                continue
            if self.result.k_axis.min() <= k_copy <= self.result.k_axis.max():
                mask_positions.append(k_copy / 1e6)

        self.w_info.value = (
            f"<small>Mask includes <b>{len(mask_positions)}</b> k-positions<br>"
            f"k = {', '.join([f'{k:.1f}' for k in mask_positions[:5]])}"
            f"{'...' if len(mask_positions) > 5 else ''} rad/μm</small>"
        )

    def _update_mode_visualization(self):
        """Update the 2D spatial mode visualization m(x, y)."""
        if self.result is None or self._ax_mode is None:
            return
        if self._selected_k is None or self._selected_f is None:
            return

        ax = self._ax_mode
        ax.clear()

        # Remove old colorbar
        if self._colorbar_mode is not None:
            try:
                self._colorbar_mode.remove()
            except Exception:
                pass
            self._colorbar_mode = None

        try:
            # Get parameters
            a = self.w_lattice.value * 1e-9
            n_bz = self.w_n_bz_mask.value
            k_direction = self.w_k_direction.value
            mode_type = self.w_mode_type.value  # 'real', 'imag', 'abs', 'phase'

            # Extract spatial mode profile m(x, y) - returns COMPLEX data
            x_axis, y_axis, mode_2d_complex = self._extract_mode_2d_custom(
                k_0=self._selected_k,
                f_0=self._selected_f,
                lattice_constant=a,
                n_bz=n_bz,
                k_direction=k_direction,
            )
            
            # Extract requested component
            if mode_type == 'real':
                mode_2d = np.real(mode_2d_complex)
                cmap = self.w_cmap_mode.value
                cbar_label = "Re[M]"
                use_rgb = False
            elif mode_type == 'imag':
                mode_2d = np.imag(mode_2d_complex)
                cmap = self.w_cmap_mode.value
                cbar_label = "Im[M]"
                use_rgb = False
            elif mode_type == 'abs':
                mode_2d = np.abs(mode_2d_complex)
                cmap = 'hot'  # Better for amplitude
                cbar_label = "|M|"
                use_rgb = False
            elif mode_type == 'phase':
                mode_2d = np.angle(mode_2d_complex)
                cmap = 'hsv'  # Cyclic colormap for phase
                cbar_label = "φ[M] [rad]"
                use_rgb = False
            elif mode_type == 'ampl_phase':
                # Use RGB colormap: hue=phase, brightness=amplitude
                from ..utils import create_amplitude_phase_colormap
                mode_2d = create_amplitude_phase_colormap(mode_2d_complex)
                cmap = None  # RGB data doesn't use colormap
                cbar_label = "Ampl×Phase"
                use_rgb = True

            # Convert axes to convenient units
            x_um = x_axis * 1e6  # μm
            y_um = y_axis * 1e6  # μm

            # Plot 2D spatial heatmap
            extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]

            # Auto color limits based on mode type
            if mode_type in ['real', 'imag']:
                # Symmetric for real/imag
                vmax = np.max(np.abs(mode_2d))
                if vmax < 1e-20:
                    vmax = 1.0
                vmin = -vmax
            elif mode_type == 'abs':
                # Always positive
                vmin = 0
                vmax = np.max(mode_2d)
                if vmax < 1e-20:
                    vmax = 1.0
            elif mode_type == 'phase':
                # Phase range
                vmin = -np.pi
                vmax = np.pi
            elif mode_type == 'ampl_phase':
                # RGB image - no vmin/vmax needed
                vmin = None
                vmax = None

            if use_rgb:
                # RGB image - no colormap
                im = ax.imshow(
                    mode_2d,
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    interpolation="bilinear",
                )
            else:
                # Scalar data with colormap
                im = ax.imshow(
                    mode_2d,
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="bilinear",
                )

            # Only add colorbar for scalar data
            if not use_rgb:
                self._colorbar_mode = self._fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                self._colorbar_mode.set_label(cbar_label, fontsize=9)

            # Labels
            ax.set_xlabel("x [μm]", fontsize=10)
            ax.set_ylabel("y [μm]", fontsize=10)

            # Title with mode type
            mode_type_labels = {
                'real': 'Re[M]',
                'imag': 'Im[M]',
                'abs': '|M|',
                'phase': 'φ[M]',
                'ampl_phase': 'Ampl×Phase',
            }
            mode_label = mode_type_labels.get(mode_type, mode_type)
            k_str = f"k = {self._selected_k/1e6:.2f} rad/μm"
            f_str = f"f = {self._selected_f/1e9:.2f} GHz"
            ax.set_title(f"{mode_label} Mode | {k_str}, {f_str}", fontsize=11)
            ax.tick_params(labelsize=9)

            # Redraw dispersion with markers
            self._update_dispersion_plot()

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error extracting mode:\n{e}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="red",
            )
            logger.exception("Mode extraction failed")

        self._fig.canvas.draw_idle()

    def _extract_mode_2d_custom(
        self,
        k_0: float,
        f_0: float,
        lattice_constant: float,
        n_bz: int,
        k_direction: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract 2D spatial mode profile m(x, y) using pre-computed S_complex.
        
        Algorithm (following Rychły et al.):
        1. Use S_complex from dispersion result (already FFT'd!)
        2. Select frequency f_0 and create mask for k_0 ± n·G (BZ replicas)
        3. IFFT only over k → propagation axis (phase preserved!)
        4. Result: M(x, y) spatial profile of the mode
        
        This is FAST - no re-computation of FFT! Uses cached S_complex.
        
        Returns x_axis, y_axis, mode_2d(x, y).
        """
        # Check if we have complex spectrum
        if self.result.S_complex is None:
            raise ValueError(
                "Mode visualization requires complex spectrum S_complex.\n"
                "This should be automatically computed with dispersion.\n"
                "Try recomputing with force=True."
            )
        
        # Get axes and data
        axis = self.result.axis  # 'x' or 'y'
        k_axis = self.result.k_axis.copy()
        f_axis = self.result.f_axis.copy()
        S_complex = self.result.S_complex  # Shape: (Nk, Nf) or (N_orth, Nk, Nf)
        
        # Get grid spacings from result
        dx = self.result.dx if self.result.dx > 0 else 1e-9
        
        # Determine if we have orthogonal spectra
        if S_complex.ndim == 3:
            # Shape: (N_orth, Nk, Nf) - we have spatial variation in orthogonal direction
            N_orth, N_k, N_f = S_complex.shape
            has_orth = True
            logger.info(f"Using orthogonal spectra: {N_orth} positions")
        else:
            # Shape: (Nk, Nf) - averaged over orthogonal direction
            N_k, N_f = S_complex.shape
            N_orth = 1
            has_orth = False
            logger.info("Using averaged spectrum (no orthogonal variation)")
        
        # ===== STEP 1: Select frequency f_0 =====
        idx_f = np.argmin(np.abs(f_axis - f_0))
        f_selected = f_axis[idx_f]
        
        logger.info(f"Selected frequency: f={f_selected/1e9:.3f} GHz (requested: {f_0/1e9:.3f} GHz)")
        
        # ===== STEP 2: Create BZ mask for k_0 ± n·G =====
        G = 2 * np.pi / lattice_constant
        dk = np.abs(k_axis[1] - k_axis[0]) if len(k_axis) > 1 else 1.0
        
        # Mask width: 2 k-bins
        delta_k = dk * 2
        
        mask = np.zeros(len(k_axis), dtype=bool)
        
        for n in range(-n_bz, n_bz + 1):
            k_target = k_0 + n * G
            
            # Apply k-direction filter
            if k_direction == "positive" and k_target < 0:
                continue
            if k_direction == "negative" and k_target > 0:
                continue
            
            # Find k-values within delta_k of target
            mask |= np.abs(k_axis - k_target) < delta_k
            
        logger.info(f"BZ mask: {np.sum(mask)} k-points selected out of {len(k_axis)}")
        
        # ===== STEP 3: Extract slice at f_0 and apply mask =====
        if has_orth:
            # S_complex shape: (N_orth, Nk, Nf)
            # Extract at f_0: (N_orth, Nk)
            S_at_f = S_complex[:, :, idx_f]
            
            # Apply mask: zero non-selected k
            S_filtered = S_at_f.copy()
            S_filtered[:, ~mask] = 0
            
            # S_filtered shape: (N_orth, Nk)
        else:
            # S_complex shape: (Nk, Nf)
            # Extract at f_0: (Nk,)
            S_at_f = S_complex[:, idx_f]
            
            # Apply mask
            S_filtered = S_at_f.copy()
            S_filtered[~mask] = 0
            
            # S_filtered shape: (Nk,)
            
        # ===== STEP 4: IFFT over k → spatial axis =====
        # Undo fftshift before IFFT
        if has_orth:
            S_unshift = np.fft.ifftshift(S_filtered, axes=1)  # Unshift k-axis (axis=1)
            M_mode = np.fft.ifft(S_unshift, axis=1)  # IFFT along k
            # M_mode shape: (N_orth, N_prop) where N_prop is propagation axis length
        else:
            S_unshift = np.fft.ifftshift(S_filtered)
            M_mode = np.fft.ifft(S_unshift)
            # M_mode shape: (N_prop,)
            # Expand to 2D for consistency
            M_mode = M_mode[np.newaxis, :]  # → (1, N_prop)
            
        # ===== STEP 5: Construct spatial axes =====
        N_prop = N_k  # Propagation axis length = k-axis length
        
        # Propagation axis
        L_prop = 2 * np.pi / dk if dk > 0 else N_prop * dx
        prop_axis = np.linspace(0, L_prop, N_prop, endpoint=False)
        
        # Orthogonal axis
        if has_orth and self.result.orth_axis is not None:
            orth_axis = self.result.orth_axis
        else:
            # Fallback - assume same spacing
            orth_axis = np.arange(N_orth) * dx
            
        # ===== STEP 6: Assign to x, y based on propagation axis =====
        # M_mode shape: (N_orth, N_prop)
        # For axis='x': N_orth=N_y, N_prop=N_x → M_mode is (N_y, N_x) ✓
        # For axis='y': N_orth=N_x, N_prop=N_y → M_mode is (N_x, N_y), need transpose
        if axis == "x":
            x_axis = prop_axis
            y_axis = orth_axis
            # M_mode shape: (N_y, N_x) - already correct for m[y, x] indexing
            mode_2d = M_mode
        else:  # axis == 'y'
            x_axis = orth_axis
            y_axis = prop_axis
            # M_mode shape: (N_x, N_y) - need transpose to (N_y, N_x) for m[y, x]
            mode_2d = M_mode.T
            
        # Return COMPLEX mode (caller decides whether to take real, imag, abs, phase)
        logger.info(
            f"Mode profile shape: {mode_2d.shape} (complex), "
            f"x: {x_axis.min()*1e6:.1f}-{x_axis.max()*1e6:.1f} μm, "
            f"y: {y_axis.min()*1e6:.1f}-{y_axis.max()*1e6:.1f} μm, "
            f"|M|_max: {np.abs(mode_2d).max():.2e}"
        )

        return x_axis, y_axis, mode_2d

    # =========================================================================
    # Non-interactive methods
    # =========================================================================

    def mode(
        self,
        k: float,
        f: float,
        lattice_constant_nm: float | None = None,
        n_bz: int = 3,
        k_direction: str = "both",
    ) -> ModeProfile:
        """
        Extract 2D spatial mode profile m(x, y) and return visualization object.
        
        Returns a ModeProfile object with plotting and animation capabilities.
        Uses the Rychły et al. algorithm to reconstruct spatial mode from
        pre-computed S_complex data.
        
        Parameters
        ----------
        k : float
            Wave vector in rad/μm
        f : float
            Frequency in GHz
        lattice_constant_nm : float, optional
            Lattice constant in nm. If None, uses default from widget (470 nm).
        n_bz : int, default=3
            Number of Brillouin zones to include in mask (±n_bz around k_0).
        k_direction : str, default='both'
            Direction filter: 'both' (±k), 'positive' (+k only), 'negative' (-k only).
            
        Returns
        -------
        ModeProfile
            Object containing complex mode profile M(x,y) with methods:
            - .plot(mode_type='real'|'imag'|'abs'|'phase', cmap=..., dpi=...)
            - .animate(duration_ns=..., n_frames=..., fps=...)
            - .get_components() → dict with all components
            - .to_dict() → legacy dict format
            
        Examples
        --------
        >>> modes = job[0].m_layer13[...].fft.dispersion.dispersion_modes(save=True)
        >>> 
        >>> # Get mode object
        >>> mode = modes.mode(k=2.30, f=1.12)
        >>> 
        >>> # Plot different components
        >>> mode.plot(mode_type='abs', cmap='hot', dpi=150)
        >>> mode.plot(mode_type='phase', cmap='hsv')
        >>> mode.plot(mode_type='real', figsize=(12, 8))
        >>> 
        >>> # Animate
        >>> anim = mode.animate(duration_ns=10, n_frames=100, fps=30)
        >>> anim.save('mode.gif', writer='pillow')
        >>> 
        >>> # Access data
        >>> print(mode.m_xy.shape)  # Complex array (N_y, N_x)
        >>> components = mode.get_components()  # dict with real, imag, abs, phase
        """
        if self.result is None:
            raise ValueError(
                "No dispersion result available. "
                "Run dispersion_modes() with save=True first."
            )
            
        # Convert to SI units
        k_si = k * 1e6  # rad/μm → rad/m
        f_si = f * 1e9  # GHz → Hz
        
        # Get lattice constant
        if lattice_constant_nm is None:
            lattice_constant_nm = self._default_params.get("lattice_nm", 470.0)
        a = lattice_constant_nm * 1e-9  # nm → m
        
        # Extract mode using internal method (returns COMPLEX data!)
        x_axis, y_axis, mode_2d_complex = self._extract_mode_2d_custom(
            k_0=k_si,
            f_0=f_si,
            lattice_constant=a,
            n_bz=n_bz,
            k_direction=k_direction,
        )
        
        # Build metadata
        info = {
            'k_rad_um': k,
            'f_GHz': f,
            'lattice_constant_nm': lattice_constant_nm,
            'n_bz': n_bz,
            'k_direction': k_direction,
            'shape': mode_2d_complex.shape,
            'amplitude_max': float(np.abs(mode_2d_complex).max()),
        }
        
        # Return ModeProfile object
        return ModeProfile(
            m_xy=mode_2d_complex,
            x=x_axis,
            y=y_axis,
            k=k_si,
            f=f_si,
            info=info,
        )

    def extract_mode_profile(
        self,
        k: float,
        f: float,
        lattice_constant: float | None = None,
        n_periods: int = 3,
        delta_k: float | None = None,
        delta_f: float | None = None,
        result: DispersionResult1D | None = None,
        **compute_kwargs,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Extract the spatial profile of a specific mode."""
        if result is None:
            if self.result is not None:
                result = self.result
            else:
                result = self.interface.compute_1d(**compute_kwargs)

        if lattice_constant is None:
            lattice_constant = 470e-9

        folder = self._get_folder(lattice_constant, n_periods)
        return folder.extract_mode_profile(
            result=result,
            k_0=k,
            f_0=f,
            delta_k=delta_k,
            delta_f=delta_f,
        )

    def extract_mode_evolution(
        self,
        k: float,
        f: float,
        lattice_constant: float | None = None,
        n_periods: int = 3,
        delta_k: float | None = None,
        delta_f: float | None = None,
        result: DispersionResult1D | None = None,
        **compute_kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract full time-space evolution of a mode."""
        if result is None:
            if self.result is not None:
                result = self.result
            else:
                result = self.interface.compute_1d(**compute_kwargs)

        if lattice_constant is None:
            lattice_constant = 470e-9

        folder = self._get_folder(lattice_constant, n_periods)
        return folder.extract_mode_time_evolution(
            result=result,
            k_0=k,
            f_0=f,
            delta_k=delta_k,
            delta_f=delta_f,
        )

    def animate_mode(
        self,
        k: float,
        f: float,
        lattice_constant: float | None = None,
        n_periods: int = 3,
        n_frames: int = 120,
        delta_k: float | None = None,
        delta_f: float | None = None,
        damping_time: float | None = None,
        save_path: str | None = None,
        result: DispersionResult1D | None = None,
        **animate_kwargs,
    ):
        """Animate a specific spin wave mode."""
        from .animation import SpinWaveModeAnimator, extract_amplitude_phase

        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for animation")

        if result is None:
            if self.result is not None:
                result = self.result
            else:
                result = self.interface.compute_1d()

        if lattice_constant is None:
            lattice_constant = 470e-9

        folder = self._get_folder(lattice_constant, n_periods)
        y_axis, profile_complex, info = folder.extract_mode_profile(
            result=result,
            k_0=k,
            f_0=f,
            delta_k=delta_k,
            delta_f=delta_f,
            return_complex=True,
        )

        amplitude, phase = extract_amplitude_phase(profile_complex)

        animator = SpinWaveModeAnimator(
            y_axis=y_axis,
            amplitude=amplitude,
            phase=phase,
            frequency_hz=f,
            k_value=k,
        )

        animator.animate(
            n_frames=n_frames,
            n_periods=n_periods,
            damping_time=damping_time,
            save_path=save_path,
            **animate_kwargs,
        )

        return animator

    def plot_static(
        self,
        result: DispersionResult1D | None = None,
        lattice_constant: float | None = None,
        n_periods: int = 3,
        peak_threshold: float = 0.01,
        f_min: float = 0,
        f_max: float = np.inf,
        figsize: tuple[float, float] = (12, 7),
        dpi: int = 150,
        show_heatmap: bool = True,
        cmap: str = "viridis",
        ax: Axes | None = None,
        **compute_kwargs,
    ) -> tuple[Figure, Axes, FoldedDispersionResult]:
        """Create a static (non-interactive) folded dispersion plot."""
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        if result is None:
            result = self.interface.compute_1d(**compute_kwargs)
        self.result = result

        if lattice_constant is None:
            lattice_constant = 470e-9

        folder = self._get_folder(lattice_constant, n_periods)
        folded = folder.fold_dispersion(result, peak_threshold)
        folded = folded.filter_frequency(f_min * 1e9, f_max * 1e9)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        # Plot heatmap
        if show_heatmap:
            S = result.S.T
            k_axis = result.k_axis / 1e6
            f_axis = result.f_axis / 1e9

            f_mask = (f_axis >= f_min) & (f_axis <= (f_max if f_max < np.inf else f_axis.max()))
            S = S[f_mask, :]
            f_axis = f_axis[f_mask]

            extent = [k_axis[0], k_axis[-1], f_axis[0], f_axis[-1]]
            ax.imshow(
                np.log10(S + 1e-20),
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap=cmap,
                alpha=0.8,
            )

        # BZ boundary lines (reciprocal lattice vectors G = 2π/a)
        G = 2 * np.pi / lattice_constant / 1e6  # rad/μm
        ax.axvline(-G, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(G, color="red", linestyle="--", linewidth=1.5, label=f"±G = ±{G:.1f}")

        # Show ±2G as well if in range
        k_range = ax.get_xlim()
        k_max = max(abs(k_range[0]), abs(k_range[1]))
        if 2*G <= k_max:
            ax.axvline(-2*G, color="red", linestyle="--", linewidth=1.0, alpha=0.4)
            ax.axvline(2*G, color="red", linestyle="--", linewidth=1.0, alpha=0.4)

        ax.set_xlabel(r"$k$ [rad/μm]", fontsize=10)
        ax.set_ylabel("f [GHz]", fontsize=10)
        ax.set_title(f"Dispersion | a = {lattice_constant*1e9:.0f} nm", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")

        return fig, ax, folded

    def __repr__(self) -> str:
        status = "result loaded" if self.result is not None else "no result"
        return f"InteractiveDispersionModes({status})"
