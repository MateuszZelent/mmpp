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

        # Connect changes that should update immediately
        for w in [self.w_cmap_disp, self.w_fmin, self.w_fmax]:
            w.observe(self._on_display_param_change, names="value")

        # Connect mode visualization params
        for w in [self.w_n_bz_mask, self.w_k_direction, self.w_cmap_mode]:
            w.observe(self._on_mode_param_change, names="value")

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
                widgets.HTML("<small><b>Frequency Range</b></small>"),
                self.w_fmin,
                self.w_fmax,
                widgets.HTML("<small><b>Display</b></small>"),
                self.w_cmap_disp,
                self.w_cmap_mode,
                self.w_update,
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

        # Add FBZ lines
        a = self.w_lattice.value * 1e-9
        k_bz = np.pi / a / 1e6  # rad/μm
        ax.axvline(-k_bz, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(k_bz, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label=f"±π/a = ±{k_bz:.1f}")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.legend(loc="upper right", fontsize=8)

        # Labels
        ax.set_xlabel(r"$k$ [rad/μm]", fontsize=10)
        ax.set_ylabel("f [GHz]", fontsize=10)
        ax.set_title(f"Dispersion S(k, f) | a = {a*1e9:.0f} nm | Click to select mode", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=9)

        # Set default k-axis limits to ±2 BZ (can be zoomed out manually)
        k_limit = 2 * k_bz  # ±2 Brillouin zones
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

            # Extract spatial mode profile m(x, y)
            x_axis, y_axis, mode_2d = self._extract_mode_2d_custom(
                k_0=self._selected_k,
                f_0=self._selected_f,
                lattice_constant=a,
                n_bz=n_bz,
                k_direction=k_direction,
            )

            # Convert axes to convenient units
            x_um = x_axis * 1e6  # μm
            y_um = y_axis * 1e6  # μm

            # Plot 2D spatial heatmap
            extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]

            # Use symmetric colormap for real part (shows oscillation structure)
            vmax = np.max(np.abs(mode_2d))
            if vmax < 1e-20:
                vmax = 1.0
            vmin = -vmax

            im = ax.imshow(
                mode_2d,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="RdBu_r",  # Symmetric diverging colormap for real part
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
            )

            self._colorbar_mode = self._fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            self._colorbar_mode.set_label("Re[m]", fontsize=9)

            # Labels
            ax.set_xlabel("x [μm]", fontsize=10)
            ax.set_ylabel("y [μm]", fontsize=10)

            k_str = f"k = {self._selected_k/1e6:.2f} rad/μm"
            f_str = f"f = {self._selected_f/1e9:.2f} GHz"
            ax.set_title(f"Mode Profile m(x, y) | {k_str}, {f_str}", fontsize=11)
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
        if axis == "x":
            x_axis = prop_axis
            y_axis = orth_axis
            # M_mode shape: (N_y, N_x) - transpose to (N_x, N_y) for imshow
            mode_2d = M_mode.T
        else:  # axis == 'y'
            x_axis = orth_axis
            y_axis = prop_axis
            # M_mode shape: (N_y, N_x) already correct
            mode_2d = M_mode.T
            
        # Take real part (shows oscillation structure)
        mode_2d = np.real(mode_2d)
        
        logger.info(f"Mode profile shape: {mode_2d.shape}, x: {x_axis.min()*1e6:.1f}-{x_axis.max()*1e6:.1f} μm, y: {y_axis.min()*1e6:.1f}-{y_axis.max()*1e6:.1f} μm")

        return x_axis, y_axis, mode_2d

    # =========================================================================
    # Non-interactive methods
    # =========================================================================

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

        # FBZ lines
        k_bz = np.pi / lattice_constant / 1e6
        ax.axvline(-k_bz, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(k_bz, color="red", linestyle="--", linewidth=1.5, label=f"FBZ ±{k_bz:.1f}")

        ax.set_xlabel(r"$k$ [rad/μm]", fontsize=10)
        ax.set_ylabel("f [GHz]", fontsize=10)
        ax.set_title(f"Dispersion | a = {lattice_constant*1e9:.0f} nm", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")

        return fig, ax, folded

    def __repr__(self) -> str:
        status = "result loaded" if self.result is not None else "no result"
        return f"InteractiveDispersionModes({status})"
