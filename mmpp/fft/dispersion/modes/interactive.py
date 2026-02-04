"""
Interactive Jupyter widget interface for dispersion mode analysis.

Provides an ipywidgets-based interactive exploration of BZ-folded
dispersion relations with real-time parameter adjustment.

Features:
- Side-by-side layout (controls | plot)
- In-place plot updates
- Click to select mode (single click) or find branch (double click)
- Mode visualization panel
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

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
    from .animation import SpinWaveModeAnimator


class InteractiveDispersionModes:
    """
    Interactive widget for exploring Brillouin zone folded dispersion.
    
    Provides real-time adjustment of:
    - Lattice constant (period)
    - Number of BZ periods
    - Frequency range
    - Peak detection threshold
    - Visualization options
    
    Usage
    -----
    >>> # From dispersion interface
    >>> job[0].fft.dispersion.dispersion_modes.plot_interactive()
    
    >>> # With pre-computed result
    >>> disp_result = job[0].fft.dispersion.compute_1d(axis="x")
    >>> job[0].fft.dispersion.dispersion_modes.plot_interactive(result=disp_result)
    
    >>> # Programmatic access (no widgets)
    >>> modes = job[0].fft.dispersion.dispersion_modes
    >>> folded = modes.fold(disp_result, lattice_constant=470e-9)
    >>> print(folded.summary())
    """
    
    def __init__(self, dispersion_interface: "FFTDispersionInterface"):
        """
        Initialize the interactive modes analysis.
        
        Parameters
        ----------
        dispersion_interface : FFTDispersionInterface
            Parent dispersion interface for accessing compute methods
        """
        self.interface = dispersion_interface
        self.result: Optional["DispersionResult1D"] = None
        self.folded: Optional["FoldedDispersionResult"] = None
        
        # Lazy import components
        self._detector: Optional["BrillouinZoneDetector"] = None
        self._folder: Optional["BrillouinZoneFolding"] = None
        
        # Widget state
        self._widgets_created = False
        self._output: Optional[Any] = None
        self._mode_output: Optional[Any] = None
        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None
        self._colorbar = None
        self._scatter = None
        
        # Click state
        self._last_click_time = 0.0
        self._selected_mode = None
        self._selected_branch = None
        
        # Default parameters
        self._default_params = {
            "lattice_nm": 470.0,  # nm - sensible default for magnonic crystals
            "n_periods": 3,
            "f_min_ghz": 0.0,
            "f_max_ghz": 10.0,
            "threshold": 0.01,
            "show_origin": True,
            "show_fbz": True,
            "show_heatmap": True,
            "cmap": "viridis",
        }
        
        # Figure settings
        self._dpi = 150
        self._figsize = (10, 6)
    
    @property
    def detector(self) -> "BrillouinZoneDetector":
        """Lazy-load detector."""
        if self._detector is None:
            from .detection import BrillouinZoneDetector
            self._detector = BrillouinZoneDetector()
        return self._detector
    
    def _get_folder(self, lattice_constant: float, n_periods: int) -> "BrillouinZoneFolding":
        """Get or create folder with specified parameters."""
        from .folding import BrillouinZoneFolding
        return BrillouinZoneFolding(lattice_constant, n_periods)
    
    def fold(
        self,
        result: Optional["DispersionResult1D"] = None,
        lattice_constant: Optional[float] = None,
        n_periods: int = 3,
        peak_threshold: float = 0.01,
        **compute_kwargs,
    ) -> "FoldedDispersionResult":
        """
        Fold dispersion to first Brillouin zone (non-interactive).
        
        Parameters
        ----------
        result : DispersionResult1D, optional
            Pre-computed dispersion result. If None, computes using interface.
        lattice_constant : float, optional
            Lattice constant [m]. If None, uses default 470nm.
        n_periods : int
            Number of BZ periods to consider
        peak_threshold : float
            Relative peak detection threshold
        **compute_kwargs : dict
            Extra kwargs for compute_1d if result is None
            
        Returns
        -------
        FoldedDispersionResult
            Folded dispersion with mode tracking
        """
        # Get or compute dispersion result
        if result is None:
            result = self.interface.compute_1d(**compute_kwargs)
        self.result = result
        
        # Use default if not provided (auto-detect is unreliable)
        if lattice_constant is None:
            lattice_constant = 470e-9  # Default for magnonic crystals
            logger.info("Using default lattice constant: %.1f nm", lattice_constant * 1e9)
        
        # Create folder and fold
        folder = self._get_folder(lattice_constant, n_periods)
        self.folded = folder.fold_dispersion(result, peak_threshold)
        
        return self.folded
    
    def plot_interactive(
        self,
        result: Optional["DispersionResult1D"] = None,
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 150,
        lattice_constant_nm: float = 470.0,
        **compute_kwargs,
    ):
        """
        Launch interactive widget for dispersion mode exploration.
        
        Parameters
        ----------
        result : DispersionResult1D, optional
            Pre-computed dispersion. If None, computes automatically.
        figsize : tuple
            Figure size (width, height) in inches
        dpi : int
            Figure DPI (default 150)
        lattice_constant_nm : float
            Initial lattice constant in nm (default 470)
        **compute_kwargs : dict
            Extra kwargs passed to compute_1d if result is None
        """
        if not _HAS_WIDGETS:
            raise ImportError(
                "ipywidgets is required for interactive mode. "
                "Install with: pip install ipywidgets"
            )
        if not _HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )
        
        self._dpi = dpi
        self._figsize = figsize
        
        # Compute result if needed
        if result is None:
            logger.info("Computing dispersion result...")
            result = self.interface.compute_1d(**compute_kwargs)
        self.result = result
        
        # Set initial parameters
        f_max_ghz = result.f_axis.max() / 1e9
        self._default_params["lattice_nm"] = lattice_constant_nm
        self._default_params["f_max_ghz"] = min(f_max_ghz, 20.0)
        
        # Create widgets
        self._create_widgets()
        
        # Create the main layout (side by side)
        main_layout = self._create_side_by_side_layout()
        
        # Display
        display(main_layout)
        
        # Initial plot
        self._initialize_figure()
        self._update_plot_inplace()
    
    def _create_widgets(self):
        """Create all interactive widgets."""
        params = self._default_params
        
        # === Lattice parameters ===
        self.w_lattice = widgets.FloatSlider(
            value=params["lattice_nm"],
            min=50, max=2000, step=5,
            description='a [nm]:',
            layout=widgets.Layout(width='95%'),
            style={'description_width': '60px'},
            continuous_update=False,
        )
        
        self.w_n_periods = widgets.IntSlider(
            value=params["n_periods"],
            min=1, max=10, step=1,
            description='N BZ:',
            layout=widgets.Layout(width='95%'),
            style={'description_width': '60px'},
            continuous_update=False,
        )
        
        # === Frequency range ===
        self.w_fmin = widgets.FloatSlider(
            value=params["f_min_ghz"],
            min=0, max=params["f_max_ghz"], step=0.1,
            description='f min:',
            layout=widgets.Layout(width='95%'),
            style={'description_width': '60px'},
            continuous_update=False,
        )
        
        self.w_fmax = widgets.FloatSlider(
            value=params["f_max_ghz"],
            min=0.1, max=params["f_max_ghz"] * 1.5, step=0.1,
            description='f max:',
            layout=widgets.Layout(width='95%'),
            style={'description_width': '60px'},
            continuous_update=False,
        )
        
        self.w_threshold = widgets.FloatLogSlider(
            value=params["threshold"],
            base=10, min=-4, max=0, step=0.1,
            description='Thresh:',
            layout=widgets.Layout(width='95%'),
            style={'description_width': '60px'},
            continuous_update=False,
        )
        
        # === Visualization options ===
        self.w_show_heatmap = widgets.Checkbox(
            value=params["show_heatmap"],
            description='Heatmap',
            layout=widgets.Layout(width='95%'),
        )
        
        self.w_show_origin = widgets.Checkbox(
            value=params["show_origin"],
            description='Color by BZ',
            layout=widgets.Layout(width='95%'),
        )
        
        self.w_show_fbz = widgets.Checkbox(
            value=params["show_fbz"],
            description='FBZ lines',
            layout=widgets.Layout(width='95%'),
        )
        
        self.w_cmap = widgets.Dropdown(
            options=['viridis', 'plasma', 'cividis', 'turbo', 'coolwarm'],
            value=params["cmap"],
            description='Cmap:',
            layout=widgets.Layout(width='95%'),
            style={'description_width': '60px'},
        )
        
        # === Buttons ===
        self.w_update = widgets.Button(
            description='🔄 Update',
            button_style='success',
            layout=widgets.Layout(width='95%'),
        )
        
        self.w_auto_detect = widgets.Button(
            description='🔍 Auto-detect a',
            button_style='info',
            layout=widgets.Layout(width='95%'),
        )
        
        # === Info and mode output ===
        self.w_info = widgets.HTML(
            value="<small>Click plot to select mode</small>",
        )
        
        self._output = widgets.Output(
            layout=widgets.Layout(width='100%', height='auto')
        )
        
        self._mode_output = widgets.Output(
            layout=widgets.Layout(width='100%', height='200px', overflow='auto')
        )
        
        # === Connect callbacks ===
        self.w_update.on_click(self._on_update)
        self.w_auto_detect.on_click(self._on_auto_detect)
        
        # Connect slider changes to update
        for w in [self.w_show_heatmap, self.w_show_origin, self.w_show_fbz, self.w_cmap]:
            w.observe(self._on_checkbox_change, names='value')
        
        self._widgets_created = True
    
    def _create_side_by_side_layout(self) -> widgets.Widget:
        """Create side-by-side layout with controls on left, plot on right."""
        
        # Left panel: controls
        left_panel = widgets.VBox([
            widgets.HTML("<b>🌊 BZ Folding</b>"),
            widgets.HTML("<hr style='margin:2px'>"),
            widgets.HTML("<small><b>Lattice</b></small>"),
            self.w_lattice,
            self.w_n_periods,
            self.w_auto_detect,
            widgets.HTML("<small><b>Frequency</b></small>"),
            self.w_fmin,
            self.w_fmax,
            self.w_threshold,
            widgets.HTML("<small><b>Display</b></small>"),
            self.w_show_heatmap,
            self.w_show_origin,
            self.w_show_fbz,
            self.w_cmap,
            self.w_update,
            widgets.HTML("<hr style='margin:5px'>"),
            self.w_info,
            widgets.HTML("<small><b>Mode Info</b></small>"),
            self._mode_output,
        ], layout=widgets.Layout(
            width='220px',
            padding='5px',
            border='1px solid #ddd',
        ))
        
        # Right panel: plot
        right_panel = widgets.VBox([
            self._output,
        ], layout=widgets.Layout(
            width='calc(100% - 240px)',
            min_width='600px',
        ))
        
        # Main layout
        main = widgets.HBox([
            left_panel,
            right_panel,
        ], layout=widgets.Layout(
            width='100%',
        ))
        
        return main
    
    def _initialize_figure(self):
        """Create the matplotlib figure once."""
        with self._output:
            clear_output(wait=True)
            
            # Use widget backend for interactivity
            plt.ioff()
            
            self._fig, self._ax = plt.subplots(
                figsize=self._figsize,
                dpi=self._dpi,
            )
            
            # Connect click events
            self._fig.canvas.mpl_connect('button_press_event', self._on_click)
            
            plt.ion()
            plt.show()
    
    def _on_click(self, event):
        """Handle mouse click on the plot."""
        if event.inaxes != self._ax:
            return
        
        k_clicked = event.xdata * 1e6  # μm^-1 → m^-1
        f_clicked = event.ydata * 1e9  # GHz → Hz
        
        current_time = time.time()
        time_since_last = current_time - self._last_click_time
        self._last_click_time = current_time
        
        if time_since_last < 0.3:
            # Double click: find nearest branch
            self._on_double_click(k_clicked, f_clicked)
        else:
            # Single click: visualize mode at this location
            self._on_single_click(k_clicked, f_clicked)
    
    def _on_single_click(self, k: float, f: float):
        """Handle single click - visualize mode at clicked location."""
        with self._mode_output:
            clear_output(wait=True)
            
            print(f"📍 Selected: k={k/1e6:.2f} rad/μm, f={f/1e9:.2f} GHz")
            
            # Store selection
            self._selected_mode = (k, f)
            
            # Get parameters
            a = self.w_lattice.value * 1e-9
            
            # Visualize mode profile
            try:
                self._visualize_mode_at_point(k, f, a)
            except Exception as e:
                print(f"⚠️ Error: {e}")
    
    def _on_double_click(self, k: float, f: float):
        """Handle double click - find nearest branch."""
        if self.folded is None or len(self.folded.modes) == 0:
            return
        
        with self._mode_output:
            clear_output(wait=True)
            
            # Find nearest mode
            nearest = self.folded.find_mode_nearest(k, f)
            
            if nearest is not None:
                print(f"🎯 Nearest branch: {nearest.branch_index}")
                print(f"   k = {nearest.k/1e6:.3f} rad/μm")
                print(f"   f = {nearest.omega/1e9:.3f} GHz")
                print(f"   Origin BZ: {nearest.origin_BZ}")
                print(f"   Intensity: {nearest.intensity:.2e}")
                
                self._selected_branch = nearest.branch_index
                
                # Highlight this branch on plot
                self._highlight_branch(nearest.branch_index)
    
    def _visualize_mode_at_point(self, k: float, f: float, a: float):
        """Visualize mode profile at arbitrary (k, f) point."""
        if self.result is None:
            print("No dispersion data")
            return
        
        # Create folder for this lattice constant
        folder = self._get_folder(a, self.w_n_periods.value)
        
        try:
            # Extract profile from full spectrum (not just found modes)
            y_axis, profile_complex, info = folder.extract_mode_profile(
                result=self.result,
                k_0=k,
                f_0=f,
                return_complex=True,
            )
            
            from .animation import extract_amplitude_phase
            amplitude, phase = extract_amplitude_phase(profile_complex)
            
            # Quick inline plot
            fig_mode, ax_mode = plt.subplots(figsize=(4, 2), dpi=100)
            y_um = y_axis * 1e6
            ax_mode.plot(y_um, amplitude, 'b-', linewidth=1)
            ax_mode.fill_between(y_um, 0, amplitude, alpha=0.3)
            ax_mode.set_xlabel('y [μm]', fontsize=8)
            ax_mode.set_ylabel('|m|', fontsize=8)
            ax_mode.set_title(f'Mode @ k={k/1e6:.1f}, f={f/1e9:.1f} GHz', fontsize=9)
            ax_mode.tick_params(labelsize=7)
            plt.show()
            
        except Exception as e:
            print(f"Cannot extract profile: {e}")
    
    def _highlight_branch(self, branch_index: int):
        """Highlight a specific branch on the plot."""
        if self.folded is None:
            return
        
        branch_modes = self.folded.get_branch(branch_index)
        if not branch_modes:
            return
        
        k_branch = np.array([m.k for m in branch_modes]) / 1e6
        f_branch = np.array([m.omega for m in branch_modes]) / 1e9
        
        # Add highlight to existing plot
        self._ax.scatter(
            k_branch, f_branch,
            s=150, facecolors='none', edgecolors='lime',
            linewidth=2, zorder=10, label=f'Branch {branch_index}'
        )
        self._ax.legend(loc='upper right', fontsize=8)
        self._fig.canvas.draw_idle()
    
    def _on_auto_detect(self, _):
        """Handle auto-detect button click."""
        if self.result is None:
            return
        
        # Try multiple methods and pick most reasonable
        methods = ['autocorrelation', 'fft', 'peak_spacing']
        results = []
        
        for method in methods:
            try:
                a = self.detector.detect_lattice_constant(self.result, method=method)
                if 50e-9 < a < 5000e-9:  # Sanity check: 50nm to 5μm
                    results.append(a)
            except Exception:
                pass
        
        if results:
            # Use median of valid results
            detected_a = np.median(results)
            self.w_lattice.value = detected_a * 1e9
            
            self.w_info.value = f"<small style='color:green'>Detected: {detected_a*1e9:.0f} nm</small>"
        else:
            self.w_info.value = "<small style='color:orange'>Detection failed, set manually</small>"
    
    def _on_checkbox_change(self, change):
        """Handle checkbox/dropdown change - update immediately."""
        self._update_plot_inplace()
    
    def _on_update(self, _):
        """Handle update button click."""
        self._update_plot_inplace()
    
    def _update_plot_inplace(self):
        """Update the existing plot without creating a new figure."""
        if self.result is None or self._ax is None:
            return
        
        try:
            # Get parameters from widgets
            a = self.w_lattice.value * 1e-9  # nm → m
            n_periods = self.w_n_periods.value
            f_min = self.w_fmin.value * 1e9  # GHz → Hz
            f_max = self.w_fmax.value * 1e9
            threshold = self.w_threshold.value
            
            # Perform folding
            folder = self._get_folder(a, n_periods)
            self.folded = folder.fold_dispersion(self.result, threshold)
            
            # Filter by frequency
            filtered = self.folded.filter_frequency(f_min, f_max)
            
            # Clear axes but keep figure
            self._ax.clear()
            
            # Remove old colorbar if exists
            if self._colorbar is not None:
                try:
                    self._colorbar.remove()
                except Exception:
                    pass
                self._colorbar = None
            
            # Plot
            self._render_dispersion(filtered, a)
            
            # Redraw
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            
            # Update info
            k_bz = np.pi / a / 1e6
            self.w_info.value = (
                f"<small><b>{filtered.n_modes}</b> modes, "
                f"<b>{filtered.n_branches}</b> branches<br>"
                f"BZ: ±{k_bz:.1f} rad/μm</small>"
            )
            
        except Exception as e:
            logger.exception("Error updating plot")
            self.w_info.value = f"<small style='color:red'>Error: {e}</small>"
    
    def _render_dispersion(self, folded: "FoldedDispersionResult", a: float):
        """Render the folded dispersion on self._ax."""
        ax = self._ax
        fig = self._fig
        
        # Show heatmap background if requested
        if self.w_show_heatmap.value and self.result is not None:
            self._plot_heatmap_background(ax)
        
        # Get data
        k_vals = folded.k_values / 1e6  # rad/m → rad/μm
        f_vals = folded.f_values / 1e9  # Hz → GHz
        intensities = folded.intensities
        origins = folded.origins
        
        if len(k_vals) == 0:
            ax.text(0.5, 0.5, 'No modes found above threshold',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='red')
        else:
            # Normalize intensities for marker size
            int_norm = intensities / (np.max(intensities) + 1e-20)
            marker_sizes = int_norm * 60 + 8
            
            # Choose coloring
            if self.w_show_origin.value:
                scatter = ax.scatter(
                    k_vals, f_vals,
                    c=origins,
                    s=marker_sizes,
                    cmap=self.w_cmap.value,
                    alpha=0.8,
                    edgecolors='white',
                    linewidth=0.3,
                )
                self._colorbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
                self._colorbar.set_label('Origin BZ', fontsize=9)
            else:
                scatter = ax.scatter(
                    k_vals, f_vals,
                    c=np.log10(intensities + 1e-20),
                    s=marker_sizes,
                    cmap=self.w_cmap.value,
                    alpha=0.8,
                    edgecolors='white',
                    linewidth=0.3,
                )
                self._colorbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
                self._colorbar.set_label('log₁₀(I)', fontsize=9)
            
            self._scatter = scatter
        
        # FBZ boundaries
        if self.w_show_fbz.value:
            k_bz = np.pi / a / 1e6  # rad/μm
            ax.axvline(-k_bz, color='red', linestyle='--', linewidth=1.5)
            ax.axvline(k_bz, color='red', linestyle='--', linewidth=1.5,
                       label=f'FBZ ±{k_bz:.1f}')
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.legend(loc='upper right', fontsize=8)
        
        # Labels
        ax.set_xlabel(r'$k$ [rad/μm]', fontsize=10)
        ax.set_ylabel('f [GHz]', fontsize=10)
        ax.set_title(f'BZ-Folded Dispersion | a = {a*1e9:.0f} nm', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.tick_params(labelsize=9)
        
        # Set limits
        f_min = self.w_fmin.value
        f_max = self.w_fmax.value
        ax.set_ylim(f_min, f_max)
        
        # K limits to show heatmap fully  
        if self.result is not None:
            k_range = self.result.k_axis
            ax.set_xlim(k_range[0] / 1e6, k_range[-1] / 1e6)
    
    def _plot_heatmap_background(self, ax: Axes):
        """Plot the original dispersion as a faint heatmap background."""
        if self.result is None:
            return
        
        S = self.result.S.T  # (Nf, Nk)
        k_axis = self.result.k_axis / 1e6  # rad/μm
        f_axis = self.result.f_axis / 1e9  # GHz
        
        # Only positive frequencies
        f_mask = f_axis >= 0
        S = S[f_mask, :]
        f_axis = f_axis[f_mask]
        
        # Apply frequency limits
        f_min = self.w_fmin.value
        f_max = self.w_fmax.value
        f_range_mask = (f_axis >= f_min) & (f_axis <= f_max)
        
        if np.sum(f_range_mask) < 2:
            return
        
        S = S[f_range_mask, :]
        f_axis = f_axis[f_range_mask]
        
        extent = [k_axis[0], k_axis[-1], f_axis[0], f_axis[-1]]
        
        ax.imshow(
            np.log10(S + 1e-20),
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='Greys',
            alpha=0.4,
            interpolation='bilinear',
        )
    
    # =========================================================================
    # Non-interactive methods (kept from original)
    # =========================================================================
    
    def extract_mode_profile(
        self,
        k: float,
        f: float,
        lattice_constant: Optional[float] = None,
        n_periods: int = 3,
        delta_k: Optional[float] = None,
        delta_f: Optional[float] = None,
        result: Optional["DispersionResult1D"] = None,
        **compute_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Extract the spatial profile of a specific mode.
        
        Parameters
        ----------
        k : float
            Mode wave vector in FBZ [rad/m]
        f : float
            Mode frequency [Hz]
        lattice_constant : float, optional
            Lattice constant [m]. Default: 470nm
        n_periods : int
            Number of BZ periods for mask
        delta_k, delta_f : float, optional
            Filter widths
        result : DispersionResult1D, optional
            Pre-computed dispersion
            
        Returns
        -------
        y_axis, mode_profile, info
        """
        if result is None:
            if self.result is not None:
                result = self.result
            else:
                result = self.interface.compute_1d(**compute_kwargs)
        
        if lattice_constant is None:
            lattice_constant = 470e-9  # Default
        
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
        lattice_constant: Optional[float] = None,
        n_periods: int = 3,
        delta_k: Optional[float] = None,
        delta_f: Optional[float] = None,
        result: Optional["DispersionResult1D"] = None,
        **compute_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        lattice_constant: Optional[float] = None,
        n_periods: int = 3,
        n_frames: int = 120,
        delta_k: Optional[float] = None,
        delta_f: Optional[float] = None,
        damping_time: Optional[float] = None,
        save_path: Optional[str] = None,
        result: Optional["DispersionResult1D"] = None,
        **animate_kwargs,
    ) -> "SpinWaveModeAnimator":
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
        result: Optional["DispersionResult1D"] = None,
        lattice_constant: Optional[float] = None,
        n_periods: int = 3,
        peak_threshold: float = 0.01,
        f_min: float = 0,
        f_max: float = np.inf,
        figsize: Tuple[float, float] = (12, 7),
        dpi: int = 150,
        show_heatmap: bool = True,
        show_origin: bool = True,
        show_fbz: bool = True,
        cmap: str = "viridis",
        ax: Optional[Axes] = None,
        **compute_kwargs,
    ) -> Tuple[Figure, Axes, "FoldedDispersionResult"]:
        """
        Create a static (non-interactive) folded dispersion plot.
        """
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
        
        # Dummy widget values for render
        class DummyWidget:
            def __init__(self, val):
                self.value = val
        
        self.w_show_heatmap = DummyWidget(show_heatmap)
        self.w_show_origin = DummyWidget(show_origin)
        self.w_show_fbz = DummyWidget(show_fbz)
        self.w_cmap = DummyWidget(cmap)
        self.w_fmin = DummyWidget(f_min)
        self.w_fmax = DummyWidget(f_max if f_max < np.inf else result.f_axis.max() / 1e9)
        
        self._fig = fig
        self._ax = ax
        self._colorbar = None
        
        self._render_dispersion(folded, lattice_constant)
        
        return fig, ax, folded
    
    def __repr__(self) -> str:
        status = "result loaded" if self.result is not None else "no result"
        return f"InteractiveDispersionModes({status})"
