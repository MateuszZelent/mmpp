"""
Interactive Spectrum Module

Provides interactive spectrum visualization with mode panels.
Split layout: spectrum on left with peak selection, 3x3 mode grid on right.
"""

from typing import Any, Optional, Tuple, List, Union
import numpy as np
import logging

log = logging.getLogger("mmpp.fft.modes")

# Component labels
COMPONENT_LABELS = [r"$m_x$", r"$m_y$", r"$m_z$"]
COMPONENT_NAMES = ["x", "y", "z"]

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = Any
    Axes = Any


class InteractiveSpectrum:
    """Interactive spectrum with full mode visualization.
    
    Split layout:
    - Left panel: FFT power spectrum with clickable peaks
    - Right panel: 3x3 grid showing magnitude, phase, combined for each component
    
    Parameters
    ----------
    data_loader : ModeDataLoader
        Data loader with slice context
    dpi : int
        Figure resolution
    figsize : tuple
        Figure size (width, height)
    
    Examples
    --------
    >>> from .data_loader import ModeDataLoader, ModeDataContext
    >>> context = ModeDataContext(zarr_path="...", dataset_name="m", slice_info=(...,1))
    >>> loader = ModeDataLoader(context)
    >>> spectrum = InteractiveSpectrum(loader, dpi=150)
    >>> fig = spectrum.show()
    """
    
    def __init__(
        self,
        data_loader: Any = None,  # ModeDataLoader
        spectrum_result: Any = None,  # SpectrumResult from FFT.spectrum()
        component_label: str = None,
        dpi: int = 100,
        figsize: Tuple[float, float] = (16, 10),
    ):
        """Initialize InteractiveSpectrum.
        
        Parameters
        ----------
        data_loader : ModeDataLoader, optional
            Data loader for mode visualization
        spectrum_result : SpectrumResult, optional
            Pre-computed spectrum from FFT.spectrum() for consistency.
            If provided, uses frequencies/power from this result.
            If not provided, loads spectrum via data_loader.
        component_label : str, optional
            Label for single-component display
        dpi : int
            Figure resolution
        figsize : tuple
            Figure size
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for interactive spectrum")
        
        self.data_loader = data_loader
        self.spectrum_result = spectrum_result
        self._component_label = component_label
        self.dpi = dpi
        self.figsize = figsize
        
        # State
        self._fig = None
        self._ax_spectrum = None
        self._mode_axes = None  # 3x3 grid
        self._frequency_line = None
        self._current_frequency = None
        self._frequencies = None
        self._spectrum = None
        self._power = None
        self._peaks = []
        
    def show(
        self,
        components: Optional[List[Union[int, str]]] = None,
        z_layer: int = -1,
        log_scale: bool = False,
        normalize: bool = True,
        freq_unit: str = "GHz",
        show_peaks: bool = True,
        title: Optional[str] = None,
        initial_frequency: Optional[float] = None,
    ) -> Figure:
        """Create interactive spectrum with mode panels.
        
        Parameters
        ----------
        components : list, optional
            Components to show: ['x', 'y', 'z'] or [0, 1, 2]
        z_layer : int
            Z-layer for mode visualization
        log_scale : bool
            Use log scale for spectrum
        normalize : bool
            Normalize spectrum
        freq_unit : str
            Frequency unit
        show_peaks : bool
            Show detected peaks
        title : str, optional
            Custom title
        initial_frequency : float, optional
            Initial frequency to display modes for
        
        Returns
        -------
        None
            Figure is displayed via plt.show()
        """
        if components is None:
            components = ["x", "y", "z"]
        
        # Check backend
        try:
            import matplotlib
            backend = matplotlib.get_backend()
            if "inline" in backend:
                log.warning(f"Current matplotlib backend is '{backend}'. Interactivity features (clicking) will likely NOT work.")
                print(f"⚠️ Warning: Backend is '{backend}'. For interactivity, run `%matplotlib widget` (VS Code/JupyterLab) or `%matplotlib notebook`.")
        except Exception:
            pass
        
        # Normalize component names
        components = self._normalize_components(components)
        
        # Load spectrum data - prefer spectrum_result from FFT.spectrum()
        if self.spectrum_result is not None:
            # Use pre-computed spectrum from FFT (respects slice_context!)
            self._frequencies = self.spectrum_result.frequencies
            self._spectrum = self.spectrum_result.spectrum
            self._power = self.spectrum_result.power
            component_label = self._component_label or self.spectrum_result.component_label
            
            # Extract peaks from spectrum_result if available
            if self.spectrum_result.peaks_info:
                # peaks_info contains Peak objects with .freq and .amplitude attributes
                self._peaks = []
                for p in self.spectrum_result.peaks_info:
                    if hasattr(p, 'freq'):
                        # Peak dataclass object
                        self._peaks.append((p.freq, getattr(p, 'amplitude', getattr(p, 'power', 1.0))))
                    elif isinstance(p, dict):
                        # Dict format (fallback)
                        self._peaks.append((p.get('frequency', p.get('freq', 0)), p.get('amplitude', p.get('power', 1.0))))
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        # Tuple format
                        self._peaks.append((p[0], p[1]))
            log.debug(f"Using spectrum from FFT: {len(self._frequencies)} points, {len(self._peaks)} peaks")
        elif self.data_loader is not None:
            # Fallback: load spectrum via data_loader
            self._frequencies, self._spectrum, component_label = self.data_loader.load_spectrum()
            log.debug(f"Loaded spectrum via data_loader: {len(self._frequencies)} points")
        else:
            raise ValueError("Either spectrum_result or data_loader must be provided")
        
        # Create figure with GridSpec
        self._fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = GridSpec(3, 4, figure=self._fig, width_ratios=[1.5, 1, 1, 1])
        
        # Left panel: Spectrum (spans all 3 rows)
        self._ax_spectrum = self._fig.add_subplot(gs[:, 0])
        
        # Right panel: 3x3 mode grid
        # Rows: magnitude, phase, combined
        # Cols: mx, my, mz (or selected component)
        self._mode_axes = np.empty((3, 3), dtype=object)
        row_labels = ["Magnitude", "Phase", "Combined"]
        
        for row in range(3):
            for col in range(3):
                ax = self._fig.add_subplot(gs[row, col + 1])
                self._mode_axes[row, col] = ax
                
                # Set labels
                if row == 0:
                    ax.set_title(f"{COMPONENT_LABELS[col]}")
                if col == 0:
                    ax.set_ylabel(row_labels[row])
        
        # Plot spectrum
        self._plot_spectrum(log_scale, normalize, freq_unit, show_peaks, title, component_label)
        
        # Detect peaks
        if show_peaks:
            self._detect_peaks()
        
        # Set initial frequency
        if initial_frequency is not None:
            self._current_frequency = initial_frequency
        elif self._peaks:
            # Use highest peak as initial
            self._current_frequency = self._peaks[0][0]
        elif len(self._frequencies) > 0:
            # Use middle frequency
            self._current_frequency = self._frequencies[len(self._frequencies) // 2]
        
        # Draw initial frequency line and modes
        if self._current_frequency is not None:
            self._draw_frequency_line()
            self._update_mode_plots(components, z_layer)
        
        # Connect click event
        self._fig.canvas.mpl_connect('button_press_event', 
            lambda event: self._on_click(event, components, z_layer))
        
        # Add help text
        self._ax_spectrum.text(
            0.02, 0.02, 
            "Click: select freq | Right-click: snap to peak",
            transform=self._ax_spectrum.transAxes,
            fontsize=8, alpha=0.7, verticalalignment='bottom'
        )
        
        plt.tight_layout()
        plt.show()
        # Do not return figure to avoid double display in notebooks
        return None
    
    def _normalize_components(self, components: List[Union[int, str]]) -> List[str]:
        """Normalize component names to 'x', 'y', 'z'."""
        result = []
        for c in components:
            if isinstance(c, int):
                result.append(COMPONENT_NAMES[c])
            elif isinstance(c, str):
                c = c.lower().replace('m', '').replace('_', '')
                if c in COMPONENT_NAMES:
                    result.append(c)
                else:
                    result.append('z')  # fallback
            else:
                result.append('z')
        return result
    
    def _plot_spectrum(
        self, 
        log_scale: bool, 
        normalize: bool, 
        freq_unit: str, 
        show_peaks: bool,
        title: Optional[str],
        component_label: Optional[str],
    ):
        """Plot spectrum on left panel."""
        ax = self._ax_spectrum
        
        # Frequency scaling
        freq_scales = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
        freq_scale = freq_scales.get(freq_unit, 1e9)
        
        # Use frequencies as-is (already in GHz from loader)
        freqs = self._frequencies
        
        # Power spectrum
        if np.iscomplexobj(self._spectrum):
            power = np.abs(self._spectrum) ** 2
        else:
            power = self._spectrum ** 2 if self._spectrum.min() >= 0 else np.abs(self._spectrum) ** 2
        
        if normalize and power.max() > 0:
            power = power / power.max()
        
        # Store for peak detection
        self._power = power
        
        # Plot based on shape
        if power.ndim == 1:
            label = component_label or "Power"
            ax.plot(freqs, power, label=label, linewidth=1.5, color='steelblue')
            ax.legend(loc="upper right")
        elif power.ndim == 2 and power.shape[-1] == 3:
            colors = ['#e74c3c', '#2ecc71', '#3498db']  # red, green, blue
            for i in range(3):
                ax.plot(freqs, power[:, i], label=COMPONENT_LABELS[i], 
                       linewidth=1.5, color=colors[i])
            ax.legend(loc="upper right")
        else:
            ax.plot(freqs, power.flatten(), label="Power", linewidth=1.5)
        
        ax.set_xlabel(f"Frequency ({freq_unit})")
        ax.set_ylabel("Power" + (" (normalized)" if normalize else ""))
        ax.set_title(title or "FFT Power Spectrum")
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale("log")
    
    def _detect_peaks(self, min_prominence: float = 0.1):
        """Detect peaks in spectrum."""
        try:
            from scipy.signal import find_peaks as scipy_find_peaks
        except ImportError:
            log.warning("SciPy not available for peak detection")
            return
        
        # Use 1D power
        if self._power.ndim > 1:
            power_1d = np.mean(self._power, axis=-1)
        else:
            power_1d = self._power
        
        # Normalize
        if power_1d.max() > 0:
            norm_power = power_1d / power_1d.max()
        else:
            return
        
        try:
            peak_indices, props = scipy_find_peaks(
                norm_power,
                height=min_prominence,
                distance=5,
            )
        except Exception as e:
            log.debug(f"Peak detection failed: {e}")
            return
        
        if len(peak_indices) == 0:
            return
        
        # Store peaks sorted by power (highest first)
        peak_freqs = self._frequencies[peak_indices]
        peak_powers = power_1d[peak_indices]
        sorted_indices = np.argsort(peak_powers)[::-1]
        
        self._peaks = [(peak_freqs[i], peak_powers[i]) for i in sorted_indices]
        
        # Plot peaks on spectrum
        ax = self._ax_spectrum
        ax.scatter(peak_freqs, peak_powers, color='red', s=50, zorder=5, marker='v')
    
    def _draw_frequency_line(self):
        """Draw vertical line at current frequency."""
        if self._current_frequency is None:
            return
        
        ax = self._ax_spectrum
        
        # Remove old line
        if self._frequency_line is not None:
            self._frequency_line.remove()
        
        # Draw new line
        self._frequency_line = ax.axvline(
            x=self._current_frequency,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )
        
        # Update title with frequency
        ax.set_title(f"FFT Power Spectrum - f = {self._current_frequency:.3f} GHz")
    
    def _update_mode_plots(self, components: List[str], z_layer: int):
        """Update 3x3 mode grid for current frequency."""
        if self._current_frequency is None:
            return
        
        try:
            mode_data, actual_freq, metadata = self.data_loader.load_mode_at_frequency(
                self._current_frequency, z_layer
            )
        except Exception as e:
            print(f"⚠️ Error loading mode data: {e}")
            log.warning(f"Could not load mode at {self._current_frequency:.3f} GHz: {e}")
            # Clear mode plots
            for row in range(3):
                for col in range(3):
                    self._mode_axes[row, col].clear()
                    self._mode_axes[row, col].text(
                        0.5, 0.5, "No data",
                        ha='center', va='center',
                        transform=self._mode_axes[row, col].transAxes
                    )
            return
        
        # mode_data shape: (ny, nx, 3) or (ny, nx) if single component
        if mode_data.ndim == 2:
            # Single component - expand to 3D for consistency
            mode_data = mode_data[:, :, np.newaxis]
        
        row_labels = ["Magnitude", "Phase", "Combined"]
        
        for col_idx, comp in enumerate(components):
            comp_idx = COMPONENT_NAMES.index(comp) if comp in COMPONENT_NAMES else col_idx
            
            if comp_idx >= mode_data.shape[-1]:
                # Component not available
                for row in range(3):
                    self._mode_axes[row, col_idx].clear()
                    self._mode_axes[row, col_idx].text(
                        0.5, 0.5, f"No {COMPONENT_LABELS[comp_idx]}",
                        ha='center', va='center'
                    )
                continue
            
            # Get component data
            comp_data = mode_data[:, :, comp_idx]
            
            # Calculate magnitude and phase
            magnitude = np.abs(comp_data)
            phase = np.angle(comp_data)
            combined = magnitude * np.cos(phase)  # Real part visualization
            
            # Row 0: Magnitude
            ax_mag = self._mode_axes[0, col_idx]
            ax_mag.clear()
            im_mag = ax_mag.imshow(
                magnitude,
                aspect='equal',
                origin='lower',
                cmap='viridis',
                interpolation='bilinear'
            )
            ax_mag.set_title(f"{COMPONENT_LABELS[comp_idx]} @ {actual_freq:.3f} GHz")
            if col_idx == 0:
                ax_mag.set_ylabel("Magnitude")
            ax_mag.set_xticks([])
            ax_mag.set_yticks([])
            
            # Row 1: Phase
            ax_phase = self._mode_axes[1, col_idx]
            ax_phase.clear()
            im_phase = ax_phase.imshow(
                phase,
                aspect='equal',
                origin='lower',
                cmap='twilight',
                vmin=-np.pi,
                vmax=np.pi,
                interpolation='bilinear'
            )
            if col_idx == 0:
                ax_phase.set_ylabel("Phase")
            ax_phase.set_xticks([])
            ax_phase.set_yticks([])
            
            # Row 2: Combined (magnitude * cos(phase))
            ax_comb = self._mode_axes[2, col_idx]
            ax_comb.clear()
            vmax = np.abs(combined).max() or 1
            im_comb = ax_comb.imshow(
                combined,
                aspect='equal',
                origin='lower',
                cmap='RdBu_r',
                vmin=-vmax,
                vmax=vmax,
                interpolation='bilinear'
            )
            if col_idx == 0:
                ax_comb.set_ylabel("Combined")
            ax_comb.set_xticks([])
            ax_comb.set_yticks([])
        
        self._fig.canvas.draw()
    
    def _on_click(self, event, components: List[str], z_layer: int):
        """Handle click events on spectrum."""
        if event.inaxes != self._ax_spectrum:
            return
        
        if event.xdata is None:
            return
        
        if event.button == 3:  # Right click - snap to peak
            if self._peaks:
                peak_freqs = [p[0] for p in self._peaks]
                closest_idx = np.argmin(np.abs(np.array(peak_freqs) - event.xdata))
                self._current_frequency = peak_freqs[closest_idx]
            else:
                self._current_frequency = event.xdata
        else:  # Left click - exact frequency
            self._current_frequency = event.xdata
        
        # Update visualization
        self._draw_frequency_line()
        self._update_mode_plots(components, z_layer)
        self._fig.canvas.draw()


# Alias for backward compatibility
def plot(
    data_loader,
    log_scale: bool = False,
    normalize: bool = True,
    freq_unit: str = "GHz",
    show_peaks: bool = True,
    title: Optional[str] = None,
    dpi: int = 100,
    figsize: Tuple[float, float] = (12, 6),
) -> Figure:
    """Simple spectrum plot without mode panels."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required")
    
    frequencies, spectrum, component_label = data_loader.load_spectrum()
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Power
    if np.iscomplexobj(spectrum):
        power = np.abs(spectrum) ** 2
    else:
        power = spectrum ** 2 if spectrum.min() >= 0 else np.abs(spectrum) ** 2
    
    if normalize and power.max() > 0:
        power = power / power.max()
    
    # Plot
    if power.ndim == 1:
        ax.plot(frequencies, power, label=component_label or "Power", linewidth=1.5)
        ax.legend()
    elif power.ndim == 2 and power.shape[-1] == 3:
        for i in range(3):
            ax.plot(frequencies, power[:, i], label=COMPONENT_LABELS[i], linewidth=1.5)
        ax.legend()
    else:
        ax.plot(frequencies, power.flatten(), label="Power", linewidth=1.5)
    
    ax.set_xlabel(f"Frequency ({freq_unit})")
    ax.set_ylabel("Power" + (" (normalized)" if normalize else ""))
    ax.set_title(title or "FFT Power Spectrum")
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale("log")
    
    plt.tight_layout()
    return fig
