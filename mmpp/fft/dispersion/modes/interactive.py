"""
Interactive Jupyter widget interface for dispersion mode analysis.

Provides an ipywidgets-based interactive exploration of BZ-folded
dispersion relations with real-time parameter adjustment.
"""

from __future__ import annotations
import logging
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
        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None
        
        # Default parameters
        self._default_params = {
            "lattice_nm": 470.0,  # nm
            "n_periods": 3,
            "f_min_ghz": 0.0,
            "f_max_ghz": 10.0,
            "threshold": 0.01,
            "show_origin": True,
            "show_fbz": True,
            "show_heatmap": True,
            "cmap": "viridis",
        }
    
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
            Lattice constant [m]. If None, auto-detects from data.
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
        
        # Auto-detect lattice constant if not provided
        if lattice_constant is None:
            lattice_constant = self.detector.detect_lattice_constant(result)
            logger.info("Auto-detected lattice constant: %.1f nm", lattice_constant * 1e9)
        
        # Create folder and fold
        folder = self._get_folder(lattice_constant, n_periods)
        self.folded = folder.fold_dispersion(result, peak_threshold)
        
        return self.folded
    
    def plot_interactive(
        self,
        result: Optional["DispersionResult1D"] = None,
        figsize: Tuple[float, float] = (14, 8),
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
        
        # Compute result if needed
        if result is None:
            logger.info("Computing dispersion result...")
            result = self.interface.compute_1d(**compute_kwargs)
        self.result = result
        
        # Auto-detect initial parameters
        detected_a = self.detector.detect_lattice_constant(result)
        suggested_n = self.detector.suggest_n_periods(result.k_axis, detected_a)
        f_max_ghz = result.f_axis.max() / 1e9
        
        # Update defaults
        self._default_params["lattice_nm"] = detected_a * 1e9
        self._default_params["n_periods"] = suggested_n
        self._default_params["f_max_ghz"] = min(f_max_ghz, 20.0)
        
        # Create widgets
        self._create_widgets()
        
        # Layout
        controls = self._create_layout()
        
        # Display
        display(controls)
        display(self._output)
        
        # Initial plot
        self._update_plot(figsize)
    
    def _create_widgets(self):
        """Create all interactive widgets."""
        params = self._default_params
        
        # === Row 1: Lattice parameters ===
        self.w_lattice = widgets.FloatSlider(
            value=params["lattice_nm"],
            min=50, max=2000, step=5,
            description='Period a [nm]:',
            layout=widgets.Layout(width='380px'),
            style={'description_width': '110px'},
            continuous_update=False,
        )
        
        self.w_n_periods = widgets.IntSlider(
            value=params["n_periods"],
            min=1, max=10, step=1,
            description='N periods:',
            layout=widgets.Layout(width='280px'),
            style={'description_width': '100px'},
            continuous_update=False,
        )
        
        self.w_auto_detect = widgets.Button(
            description='🔍 Auto-detect',
            button_style='info',
            tooltip='Automatically detect lattice constant from data',
            layout=widgets.Layout(width='120px'),
        )
        
        # === Row 2: Frequency range ===
        self.w_fmin = widgets.FloatSlider(
            value=params["f_min_ghz"],
            min=0, max=params["f_max_ghz"], step=0.1,
            description='f min [GHz]:',
            layout=widgets.Layout(width='340px'),
            style={'description_width': '100px'},
            continuous_update=False,
        )
        
        self.w_fmax = widgets.FloatSlider(
            value=params["f_max_ghz"],
            min=0.1, max=params["f_max_ghz"] * 1.5, step=0.1,
            description='f max [GHz]:',
            layout=widgets.Layout(width='340px'),
            style={'description_width': '100px'},
            continuous_update=False,
        )
        
        self.w_threshold = widgets.FloatLogSlider(
            value=params["threshold"],
            base=10, min=-4, max=0, step=0.1,
            description='Threshold:',
            layout=widgets.Layout(width='340px'),
            style={'description_width': '100px'},
            continuous_update=False,
        )
        
        # === Row 3: Visualization options ===
        self.w_show_heatmap = widgets.Checkbox(
            value=params["show_heatmap"],
            description='Show heatmap background',
            layout=widgets.Layout(width='200px'),
        )
        
        self.w_show_origin = widgets.Checkbox(
            value=params["show_origin"],
            description='Color by BZ origin',
            layout=widgets.Layout(width='160px'),
        )
        
        self.w_show_fbz = widgets.Checkbox(
            value=params["show_fbz"],
            description='Show FBZ boundaries',
            layout=widgets.Layout(width='160px'),
        )
        
        self.w_cmap = widgets.Dropdown(
            options=['viridis', 'plasma', 'cividis', 'magma', 'inferno', 
                     'coolwarm', 'RdYlBu', 'Spectral', 'turbo'],
            value=params["cmap"],
            description='Colormap:',
            layout=widgets.Layout(width='180px'),
            style={'description_width': '80px'},
        )
        
        self.w_update = widgets.Button(
            description='🔄 Update',
            button_style='success',
            tooltip='Refresh the plot with current parameters',
            layout=widgets.Layout(width='100px'),
        )
        
        # === Output area ===
        self._output = widgets.Output()
        
        # === Connect callbacks ===
        self.w_auto_detect.on_click(self._on_auto_detect)
        self.w_update.on_click(self._on_update)
        
        # Auto-update on value change
        for w in [self.w_lattice, self.w_n_periods, self.w_fmin, self.w_fmax,
                  self.w_threshold, self.w_show_heatmap, self.w_show_origin,
                  self.w_show_fbz, self.w_cmap]:
            w.observe(self._on_param_change, names='value')
        
        self._widgets_created = True
    
    def _create_layout(self) -> widgets.Widget:
        """Create widget layout."""
        row1 = widgets.HBox([
            self.w_lattice, 
            self.w_n_periods, 
            self.w_auto_detect,
        ])
        
        row2 = widgets.HBox([
            self.w_fmin, 
            self.w_fmax, 
            self.w_threshold,
        ])
        
        row3 = widgets.HBox([
            self.w_show_heatmap,
            self.w_show_origin, 
            self.w_show_fbz, 
            self.w_cmap,
            self.w_update,
        ])
        
        # Info label
        self.w_info = widgets.HTML(
            value="<i>Adjust parameters and click Update or change sliders</i>",
            layout=widgets.Layout(margin='5px 0'),
        )
        
        return widgets.VBox([
            widgets.HTML("<h3>🌊 Interactive Dispersion Mode Analysis</h3>"),
            row1, row2, row3,
            self.w_info,
        ])
    
    def _on_auto_detect(self, _):
        """Handle auto-detect button click."""
        if self.result is None:
            return
        
        detected_a = self.detector.detect_lattice_constant(self.result)
        self.w_lattice.value = detected_a * 1e9
        
        suggested_n = self.detector.suggest_n_periods(self.result.k_axis, detected_a)
        self.w_n_periods.value = suggested_n
        
        self.w_info.value = (
            f"<b style='color:green'>✓ Auto-detected: a = {detected_a*1e9:.1f} nm, "
            f"n_periods = {suggested_n}</b>"
        )
    
    def _on_param_change(self, change):
        """Handle parameter change."""
        # Debounce by only updating on button or checkboxes
        if isinstance(change['owner'], (widgets.Checkbox, widgets.Dropdown)):
            self._update_plot()
    
    def _on_update(self, _):
        """Handle update button click."""
        self._update_plot()
    
    def _update_plot(self, figsize: Tuple[float, float] = (14, 8)):
        """Update the plot with current widget values."""
        if self.result is None:
            return
        
        with self._output:
            clear_output(wait=True)
            
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
                
                # Create figure
                fig, ax = plt.subplots(figsize=figsize)
                
                # Plot
                self._plot_folded_dispersion(fig, ax, filtered, a)
                
                plt.tight_layout()
                plt.show()
                
                self._fig = fig
                self._ax = ax
                
                # Update info
                self.w_info.value = (
                    f"<b>Found {filtered.n_modes} modes in {filtered.n_branches} branches</b> "
                    f"| BZ: ±{np.pi/a/1e6:.2f} rad/μm"
                )
                
            except Exception as e:
                logger.exception("Error updating plot")
                self.w_info.value = f"<b style='color:red'>Error: {e}</b>"
    
    def _plot_folded_dispersion(
        self,
        fig: Figure,
        ax: Axes,
        folded: "FoldedDispersionResult",
        a: float,
    ):
        """Render the folded dispersion plot."""
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
                    fontsize=14, color='red')
            ax.set_xlabel(r'Wave vector $k$ [rad/μm]')
            ax.set_ylabel('Frequency [GHz]')
            return
        
        # Normalize intensities for marker size
        int_norm = intensities / (np.max(intensities) + 1e-20)
        marker_sizes = int_norm * 80 + 10
        
        # Choose coloring
        if self.w_show_origin.value:
            # Color by BZ origin
            scatter = ax.scatter(
                k_vals, f_vals,
                c=origins,
                s=marker_sizes,
                cmap=self.w_cmap.value,
                alpha=0.8,
                edgecolors='white',
                linewidth=0.3,
            )
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Origin BZ index')
        else:
            # Color by intensity
            scatter = ax.scatter(
                k_vals, f_vals,
                c=np.log10(intensities + 1e-20),
                s=marker_sizes,
                cmap=self.w_cmap.value,
                alpha=0.8,
                edgecolors='white',
                linewidth=0.3,
            )
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('log₁₀(Intensity) [arb. u.]')
        
        # FBZ boundaries
        if self.w_show_fbz.value:
            k_bz = np.pi / a / 1e6  # rad/μm
            ax.axvline(-k_bz, color='red', linestyle='--', linewidth=2,
                       label=f'FBZ boundary (±{k_bz:.2f})')
            ax.axvline(k_bz, color='red', linestyle='--', linewidth=2)
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.legend(loc='upper right')
        
        # Labels
        ax.set_xlabel(r'Wave vector $k$ [rad/μm]', fontsize=12)
        ax.set_ylabel('Frequency [GHz]', fontsize=12)
        ax.set_title(
            f'BZ-Folded Dispersion | a = {a*1e9:.0f} nm | '
            f'{folded.n_modes} modes',
            fontsize=13
        )
        ax.grid(True, alpha=0.3)
        
        # Set k limits to FBZ
        k_bz = np.pi / a / 1e6
        ax.set_xlim(-k_bz * 1.1, k_bz * 1.1)
    
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
            alpha=0.3,
            interpolation='bilinear',
        )
    
    def plot_static(
        self,
        result: Optional["DispersionResult1D"] = None,
        lattice_constant: Optional[float] = None,
        n_periods: int = 3,
        peak_threshold: float = 0.01,
        f_min: float = 0,
        f_max: float = np.inf,
        figsize: Tuple[float, float] = (12, 7),
        show_heatmap: bool = True,
        show_origin: bool = True,
        show_fbz: bool = True,
        cmap: str = "viridis",
        ax: Optional[Axes] = None,
        **compute_kwargs,
    ) -> Tuple[Figure, Axes, "FoldedDispersionResult"]:
        """
        Create a static (non-interactive) folded dispersion plot.
        
        Parameters
        ----------
        result : DispersionResult1D, optional
            Pre-computed dispersion
        lattice_constant : float, optional
            Lattice constant [m], auto-detected if None
        n_periods : int
            Number of BZ periods
        peak_threshold : float
            Peak detection threshold
        f_min, f_max : float
            Frequency range [GHz]
        figsize : tuple
            Figure size
        show_heatmap : bool
            Show faint heatmap background
        show_origin : bool
            Color points by BZ origin
        show_fbz : bool
            Show FBZ boundary lines
        cmap : str
            Colormap name
        ax : Axes, optional
            Existing axes to plot on
            
        Returns
        -------
        fig, ax, folded : Figure, Axes, FoldedDispersionResult
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
        # Get result
        if result is None:
            result = self.interface.compute_1d(**compute_kwargs)
        self.result = result
        
        # Auto-detect lattice if needed
        if lattice_constant is None:
            lattice_constant = self.detector.detect_lattice_constant(result)
        
        # Fold
        folder = self._get_folder(lattice_constant, n_periods)
        folded = folder.fold_dispersion(result, peak_threshold)
        
        # Filter frequency
        folded = folded.filter_frequency(f_min * 1e9, f_max * 1e9)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Temporarily set widget values for helper methods
        class DummyWidget:
            def __init__(self, val):
                self.value = val
        
        self.w_show_heatmap = DummyWidget(show_heatmap)
        self.w_show_origin = DummyWidget(show_origin)
        self.w_show_fbz = DummyWidget(show_fbz)
        self.w_cmap = DummyWidget(cmap)
        self.w_fmin = DummyWidget(f_min)
        self.w_fmax = DummyWidget(f_max)
        
        # Plot
        self._plot_folded_dispersion(fig, ax, folded, lattice_constant)
        
        plt.tight_layout()
        
        return fig, ax, folded
    
    # =========================================================================
    # Mode extraction methods (Krok 4 & 5 z algorytmu Rychły et al. 2015)
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
        
        Uses the algorithm from Rychły et al. (2015):
        1. Create mask in Fourier space including all periodic copies
        2. Apply mask to S(k, f)
        3. Perform inverse FFT to get real-space profile
        
        Parameters
        ----------
        k : float
            Mode wave vector in FBZ [rad/m]
        f : float
            Mode frequency [Hz]
        lattice_constant : float, optional
            Lattice constant [m]. If None, auto-detects.
        n_periods : int
            Number of BZ periods for mask
        delta_k : float, optional
            k-space filter width [rad/m]. Default: 0.1 * G
        delta_f : float, optional
            Frequency filter width [Hz]. Default: 0.5 GHz
        result : DispersionResult1D, optional
            Pre-computed dispersion
            
        Returns
        -------
        y_axis : np.ndarray
            Position axis [m]
        mode_profile : np.ndarray
            Mode amplitude profile Re[IFFT{M̃_filtered}]
        info : dict
            Mask and extraction information
        """
        # Get dispersion result
        if result is None:
            if self.result is not None:
                result = self.result
            else:
                result = self.interface.compute_1d(**compute_kwargs)
        
        # Auto-detect lattice constant
        if lattice_constant is None:
            lattice_constant = self.detector.detect_lattice_constant(result)
        
        # Create folder and extract
        folder = self._get_folder(lattice_constant, n_periods)
        y_axis, profile, info = folder.extract_mode_profile(
            result=result,
            k_0=k,
            f_0=f,
            delta_k=delta_k,
            delta_f=delta_f,
        )
        
        return y_axis, profile, info
    
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
        """
        Extract full time-space evolution of a specific mode.
        
        Returns m_mode(t, y) via 2D inverse FFT.
        
        Parameters
        ----------
        k : float
            Mode wave vector [rad/m]
        f : float
            Mode frequency [Hz]
        lattice_constant : float, optional
            Lattice constant [m]
        n_periods : int
            Number of BZ periods
        delta_k, delta_f : float, optional
            Filter widths
        result : DispersionResult1D, optional
            Pre-computed dispersion
            
        Returns
        -------
        t_axis : np.ndarray
            Time axis [s]
        y_axis : np.ndarray
            Position axis [m]
        mode_evolution : np.ndarray
            2D array m(t, y) of shape (N_t, N_y)
        """
        # Get dispersion result
        if result is None:
            if self.result is not None:
                result = self.result
            else:
                result = self.interface.compute_1d(**compute_kwargs)
        
        # Auto-detect lattice constant
        if lattice_constant is None:
            lattice_constant = self.detector.detect_lattice_constant(result)
        
        # Create folder and extract
        folder = self._get_folder(lattice_constant, n_periods)
        t_axis, y_axis, evolution = folder.extract_mode_time_evolution(
            result=result,
            k_0=k,
            f_0=f,
            delta_k=delta_k,
            delta_f=delta_f,
        )
        
        return t_axis, y_axis, evolution
    
    def plot_mode_profile(
        self,
        k: float,
        f: float,
        lattice_constant: Optional[float] = None,
        figsize: Tuple[float, float] = (10, 4),
        ax: Optional["Axes"] = None,
        **extract_kwargs,
    ) -> Tuple["Figure", "Axes", np.ndarray]:
        """
        Plot the spatial profile of a specific mode.
        
        Parameters
        ----------
        k : float
            Mode wave vector [rad/m]
        f : float
            Mode frequency [Hz]
        lattice_constant : float, optional
            Lattice constant [m]
        figsize : tuple
            Figure size
        ax : Axes, optional
            Existing axes
            
        Returns
        -------
        fig, ax, profile
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
        y, profile, info = self.extract_mode_profile(
            k=k, f=f, lattice_constant=lattice_constant, **extract_kwargs
        )
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        ax.plot(y * 1e6, profile, 'b-', linewidth=1.5)
        ax.fill_between(y * 1e6, 0, profile, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Position y [μm]')
        ax.set_ylabel('Amplitude [arb. u.]')
        ax.set_title(
            f"Mode profile: k = {k/1e6:.3f} rad/μm, f = {f/1e9:.2f} GHz"
        )
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, ax, profile
    
    def __repr__(self) -> str:
        status = "result loaded" if self.result is not None else "no result"
        return f"InteractiveDispersionModes({status})"
