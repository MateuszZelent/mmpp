"""
Interactive spectrum visualization with mode plots.

This module provides the interactive spectrum plotting functionality that allows
users to click on the spectrum to visualize corresponding FMR modes.
"""

import logging
from typing import Any, Optional, Union, TYPE_CHECKING
from datetime import datetime

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from ..compat import MATPLOTLIB_AVAILABLE

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

from ..compat import (
    AXES_GRID_AVAILABLE,
)

from ..styling import STYLING_AVAILABLE, load_paper_style

from .animation import (
    toggle_mode_animation as _toggle_mode_animation,
    stop_mode_animation as _stop_mode_animation,
    save_animated_view as _save_animated_view,
    start_mode_animation as _start_mode_animation,
)

from .static_plots import update_single_mode_plot as _update_single_mode_plot
from ..utils.scalebar import calculate_optimal_length, format_scalebar_label

# Import inset colorbar for publication-ready mode plots
try:
    from ...transmission.plot import _make_inset_colorbar
    INSET_COLORBAR_AVAILABLE = True
except ImportError:
    INSET_COLORBAR_AVAILABLE = False

def add_scale_bar(analyzer, ax, extent):
    """Add scale bar to axis - imported from static_plots via analyzer."""
    from .static_plots import add_scale_bar as _add_scale_bar_impl
    return _add_scale_bar_impl(analyzer, ax, extent)

_add_scale_bar = add_scale_bar  # For local use



# Utility functions for peak width annotation
def normalize_peak_width_option(option):
    """
    Normalize the peak_width option to (show: bool, label: str).
    
    Returns:
        tuple: (show_peak_width: bool, peak_width_label: str)
    """
    if option is None or option is False:
        return (False, "FWHM")
    elif option is True:
        return (True, "FWHM")
    elif isinstance(option, str):
        return (True, option)
    else:
        return (False, "FWHM")


def compute_half_width_at_half_max(frequencies, spectrum):
    """
    Compute half-width at half-maximum (FWHM) for the dominant peak.
    
    Parameters:
        frequencies: array of frequencies
        spectrum: array of spectrum values
        
    Returns:
        WidthInfo object or None if computation fails
    """
    from dataclasses import dataclass
    
    @dataclass
    class WidthInfo:
        peak_frequency: float
        peak_value: float
        half_level: float
        left_frequency: float
        right_frequency: float
        width: float
    
    if len(frequencies) == 0 or len(spectrum) == 0:
        return None
        
    # Find peak
    peak_idx = np.argmax(spectrum)
    peak_freq = frequencies[peak_idx]
    peak_val = spectrum[peak_idx]
    half_val = peak_val / 2.0
    
    if peak_val <= 0:
        return None
        
    # Find left half-max point
    left_idx = peak_idx
    while left_idx > 0 and spectrum[left_idx] > half_val:
        left_idx -= 1
        
    # Find right half-max point
    right_idx = peak_idx
    while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_val:
        right_idx += 1
        
    if left_idx == 0 or right_idx == len(spectrum) - 1:
        return None
        
    # Interpolate for more accurate boundary
    if left_idx < len(frequencies) - 1:
        left_freq = np.interp(half_val, 
                             [spectrum[left_idx], spectrum[left_idx + 1]],
                             [frequencies[left_idx], frequencies[left_idx + 1]])
    else:
        left_freq = frequencies[left_idx]
        
    if right_idx > 0:
        right_freq = np.interp(half_val,
                              [spectrum[right_idx], spectrum[right_idx - 1]],
                              [frequencies[right_idx], frequencies[right_idx - 1]])
    else:
        right_freq = frequencies[right_idx]
        
    width = abs(right_freq - left_freq)
    
    return WidthInfo(
        peak_frequency=peak_freq,
        peak_value=peak_val,
        half_level=half_val,
        left_frequency=left_freq,
        right_frequency=right_freq,
        width=width
    )


def format_width_value(width_ghz):
    """Format width value with appropriate units."""
    if width_ghz < 0.001:
        return f"{width_ghz * 1000:.2f} MHz"
    else:
        return f"{width_ghz:.3f} GHz"


from ..utils.peak_detection import detect_peaks

log = logging.getLogger(__name__)


def interactive_spectrum(
    analyzer,
    components: Optional[list[Union[int, str]]] = None,
    z_layer: int = 0,
    method: int = 1,
    show: bool = True,
    force: bool = False,
    use_fft_spectrum: bool = True,
    saveanim: Union[bool, str, None] = None,
    auto_animate: bool = False,
    auto_save: bool = False,
    spectrum_result: Any = None,  # NEW: Inject spectrum from FFT.spectrum()
    **kwargs,
) -> "Figure":
    """
    Create interactive spectrum plot with mode visualization.

    Click on spectrum to select frequency and visualize corresponding mode.
    Right-click to snap to nearest peak.
    Double-click on mode plots to toggle animations.
    Press 'c' key to characterize the current mode.
    Press 's' key to save animated view (if saveanim enabled).
    Press 'h' key for help.

    **Interactive Requirements:**
    - Jupyter/IPython: Function automatically tries to enable `%matplotlib widget`
    - Standalone Python: Requires interactive backend (Qt5Agg, TkAgg, etc.)
    - If interactivity doesn't work, manually run: `%matplotlib widget` (Jupyter)
      or install ipympl: `pip install ipympl`

    Each mode panel now includes a publication-ready scale bar (auto-sized in nm)
    and shared colorbars for magnitude/phase/combined maps when the required
    matplotlib toolkit extensions are available.

    Parameters:
    -----------
    analyzer : FMRModeAnalyzer
        The mode analyzer instance
    components : list, optional
        List of components to plot (default: ['x', 'y', 'z'])
    z_layer : int, optional
        Z-layer index (default: 0)
    method : int, optional
        Visualization method (default: 1)
        1 = Standard interactive plot
        2 = Alternative layout (if implemented)
    show : bool, optional
        Whether to automatically display the figure (default: True)
    force : bool, optional
        Force reload of data from zarr file (default: False)
    use_fft_spectrum : bool, optional
        Use spectrum from standard FFT analysis instead of modes data (default: True)
        This ensures consistency with plot_spectrum results
    saveanim : bool, str, or None, optional
        Enable animation saving functionality (default: None)
        - None or False: No animation saving
        - True: Enable saving with default naming ('mode_animation_%Y%m%d_%H%M%S.mp4')
        - str: Custom path/filename for animation (e.g., 'my_animation.mp4')
        Supported formats: .mp4, .gif, .avi (depends on matplotlib writers)
        Press 's' key in interactive mode to save current animated view
    auto_animate : bool, optional
        Automatically start animations for all mode plots (default: False)
        When True, all mode visualizations are animated immediately after display
        This is useful when you primarily want to save animations without manual interaction
    auto_save : bool, optional
        Automatically save animation after auto_animate completes (default: False)
        Requires saveanim to be enabled (True or custom path)
        When True with auto_animate=True, saves animation without requiring 's' key press
    \\*\\*kwargs : dict
        Additional keyword arguments:
        - figsize : tuple, optional
            Figure size (width, height) in inches (default: from config)
        - dpi : int, optional
            Figure resolution in dots per inch (default: from config)
        - cmap : str, optional
            Colormap for all mode visualizations (overrides config colormaps)
            Examples: 'viridis', 'inferno', 'plasma', 'cividis', 'balance'
        - acmap : str, optional
            Colormap specifically for amplitude/magnitude plots (overrides cmap for magnitude)
            Examples: 'viridis', 'inferno', 'plasma', 'hot'
        - pcmap : str, optional
            Colormap specifically for phase plots (overrides cmap for phase)
            Examples: 'hsv', 'twilight', 'twilight_shifted', 'phase'
        - peak_width / fwhh / fwhm / hwfh : bool or str, optional
            Annotate the dominant peak's half-width at half-maximum on the
            spectrum panel. Strings control the label text.

    Returns:
    --------
    Figure
        Interactive matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for interactive plotting")

    # Force reload data if requested
    if force:
        log.info(
            f"Force reloading data for interactive spectrum (dataset: {analyzer.dataset_name})"
        )
        analyzer._load_data()

    peak_width_option = kwargs.pop("peak_width", None)
    for alias in ("fwhh", "fwhm", "hwfh"):
        alias_value = kwargs.pop(alias, None)
        if alias_value is not None:
            peak_width_option = alias_value

    show_peak_width, peak_width_label = normalize_peak_width_option(
        peak_width_option
    )

    # Setup animation saving
    analyzer._saveanim_enabled = saveanim is not None and saveanim is not False
    if analyzer._saveanim_enabled:
        if isinstance(saveanim, str):
            analyzer._saveanim_path = saveanim
        else:
            # Default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analyzer._saveanim_path = f"mode_animation_{timestamp}.mp4"
        log.info(
            f"Animation saving enabled. Press 's' to save to: {analyzer._saveanim_path}"
        )
    else:
        analyzer._saveanim_path = None

    # Clear previous FWHM artists if figure is being re-used
    for artist in getattr(analyzer, "_fwhm_artists", []):
        try:
            artist.remove()
        except ValueError:
            pass
    analyzer._fwhm_artists: list[Any] = []
    analyzer._last_fwhm = None

    # === PRIORITY: Use injected spectrum_result from FFT.spectrum() ===
    # This ensures consistency with job[0].m[:200,...,1].fft.spectrum() calls
    if spectrum_result is not None:
        log.info("Using injected spectrum_result from FFT (respects slice context)")
        # spectrum_result.frequencies are in Hz, convert to GHz for legacy compatibility
        frequencies_raw = spectrum_result.frequencies
        if np.max(frequencies_raw) > 1e6:  # Likely in Hz
            frequencies_to_use = frequencies_raw / 1e9  # Convert Hz to GHz
            log.debug(f"Converted frequencies from Hz to GHz: max={np.max(frequencies_to_use):.3f} GHz")
        else:
            frequencies_to_use = frequencies_raw  # Already in GHz
        spectrum_to_use = spectrum_result.power
        # Store component label for plot title
        component_label = getattr(spectrum_result, 'component_label', None)
        if component_label:
            log.debug(f"Component label from spectrum_result: {component_label}")
    else:
        # Legacy path: load from analyzer or zarr file
        component_label = None
        spectrum_to_use = analyzer.spectrum
        frequencies_to_use = analyzer.frequencies

        # Check if we have consistent data sizes
        if (
            analyzer.spectrum is not None
            and analyzer.frequencies is not None
            and len(analyzer.spectrum) != len(analyzer.frequencies)
        ):
            log.warning(
                f"Inconsistent data sizes: spectrum ({len(analyzer.spectrum)}) vs frequencies ({len(analyzer.frequencies)}). Using modes spectrum only."
            )
            use_fft_spectrum = False

        if use_fft_spectrum:
            try:
                # Try to load spectrum from standard FFT analysis
                fft_spectrum_path = (
                    f"fft/{analyzer.dataset_name}_z{z_layer}_m{method}/spectrum"
                )
                fft_freqs_path = (
                    f"fft/{analyzer.dataset_name}_z{z_layer}_m{method}/frequencies"
                )

                if (
                    fft_spectrum_path in analyzer.zarr_file
                    and fft_freqs_path in analyzer.zarr_file
                ):
                    spectrum_to_use = (
                        np.abs(np.array(analyzer.zarr_file[fft_spectrum_path])) ** 2
                    )
                    frequencies_to_use = (
                        np.array(analyzer.zarr_file[fft_freqs_path]) / 1e9
                    )  # Convert to GHz
                    log.info(
                        f"Using FFT spectrum from {fft_spectrum_path} for consistency with plot_spectrum"
                    )
                else:
                    log.warning(
                        f"FFT spectrum not found at {fft_spectrum_path}, using modes spectrum"
                    )
                    # Reset to modes data for consistency
                    spectrum_to_use = analyzer.spectrum
                    frequencies_to_use = analyzer.frequencies
            except Exception as e:
                log.warning(f"Failed to load FFT spectrum: {e}, using modes spectrum")
                # Reset to modes data for consistency
                spectrum_to_use = analyzer.spectrum
                frequencies_to_use = analyzer.frequencies

    # Debug: Log the final array sizes
    log.debug(
        f"Final arrays - spectrum: {spectrum_to_use.shape if spectrum_to_use is not None else None}, frequencies: {frequencies_to_use.shape if frequencies_to_use is not None else None}"
    )

    if spectrum_to_use is None:
        raise ValueError("No spectrum data available for interactive mode")

    # Apply paper style for consistent visualization
    if STYLING_AVAILABLE:
        try:
            load_paper_style(verbose=False)
            log.debug("Applied paper style to interactive spectrum")
        except Exception as e:
            log.warning(f"Could not apply paper style: {e}")

    # Handle method parameter
    if method not in [1, 2]:
        log.warning(f"Unknown method {method}, using default method 1")
        method = 1

    # Auto-detect number of components from data if not specified
    if components is None:
        # Try to detect from mode data shape
        try:
            if analyzer.modes_path and analyzer.modes_path in analyzer.zarr_file:
                mode_shape = analyzer.zarr_file[analyzer.modes_path].shape
                n_comp_available = mode_shape[-1]  # Last dimension is components
                log.debug(f"Detected {n_comp_available} components from mode data shape: {mode_shape}")
                
                # Map components based on available count
                if n_comp_available == 1:
                    components = ["z"]  # Single component, likely z
                    log.info("Auto-selected component: z (single component data)")
                elif n_comp_available == 2:
                    components = ["x", "y"]  # Two components (in-plane)
                    log.info("Auto-selected components: x, y (in-plane data)")
                elif n_comp_available >= 3:
                    components = ["x", "y", "z"]  # Full 3D data
                    log.info("Auto-selected components: x, y, z (full 3D data)")
                else:
                    components = ["x", "y", "z"]  # Fallback
                    log.warning(f"Unexpected component count {n_comp_available}, using default [x,y,z]")
            else:
                components = ["x", "y", "z"]  # Default fallback
                log.info("No mode data available yet, using default components [x,y,z]")
        except Exception as e:
            log.debug(f"Could not auto-detect components: {e}, using default [x,y,z]")
            components = ["x", "y", "z"]
    
    n_components = len(components)
    log.info(f"Using {n_components} component(s) for visualization: {components}")
    
    # Store components as instance attribute for callbacks
    analyzer._current_components = components

    # Extract parameters from kwargs
    figsize = kwargs.get("figsize", analyzer.config.figsize)
    dpi = kwargs.get("dpi", analyzer.config.dpi)
    cmap = kwargs.get("cmap", None)
    acmap = kwargs.get("acmap", None)  # Amplitude/magnitude colormap
    pcmap = kwargs.get("pcmap", None)  # Phase colormap

    # Update colormaps if provided
    if cmap or acmap or pcmap:
        # Create a temporary copy of config with updated colormaps
        import copy

        temp_config = copy.deepcopy(analyzer.config)

        # If cmap is provided, use it for all types unless specifically overridden
        if cmap:
            temp_config.colormap_magnitude = cmap
            temp_config.colormap_phase = cmap

        # Override with specific colormaps if provided
        if acmap:
            temp_config.colormap_magnitude = acmap
        if pcmap:
            temp_config.colormap_phase = pcmap

        analyzer.config = temp_config

    # Update figure settings from kwargs
    analyzer.config.figsize = figsize
    analyzer.config.dpi = dpi

    # Validate number of components for layout
    if n_components > 3:
        raise ValueError(
            f"Too many components ({n_components}). Maximum supported: 3 (x, y, z)"
        )

    # Create figure with custom layout
    # Automatically configure interactive backend for Jupyter
    try:
        import matplotlib

        current_backend = matplotlib.get_backend()

        # Check if we're in Jupyter/IPython environment
        try:
            from IPython import get_ipython

            ipython = get_ipython()
            in_jupyter = ipython is not None and hasattr(ipython, "kernel")
        except ImportError:
            in_jupyter = False

        # Auto-configure interactive backend for Jupyter
        if in_jupyter:
            # Always try to switch to widget backend for full interactivity
            try:
                # Force widget backend for interactive features
                ipython.run_line_magic("matplotlib", "widget")
                log.info(
                    "Switched to matplotlib widget backend for full interactivity"
                )
            except Exception as e:
                log.debug(f"Could not auto-switch to widget backend: {e}")
                # Try nbagg as fallback
                try:
                    ipython.run_line_magic("matplotlib", "nbagg")
                    log.info("Switched to nbagg backend as fallback")
                except Exception:
                    if (
                        "ipympl" not in current_backend.lower()
                        and "widget" not in current_backend.lower()
                        and "nbagg" not in current_backend.lower()
                    ):
                        log.warning(
                            f"Current backend '{current_backend}' may not support full interactivity. "
                            "Please run '%matplotlib widget' manually. Install ipympl: pip install ipympl"
                        )
                    else:
                        log.info(f"Using Jupyter-compatible backend: {current_backend}")
        else:
            # Not in Jupyter - check for standalone interactive backends
            interactive_backends = ["qt5agg", "tkagg", "gtk3agg", "wxagg", "macosx"]
            current_lower = current_backend.lower()
            if current_lower not in interactive_backends:
                log.info(
                    f"Current backend: {current_backend}. Interactive features may be limited. "
                    f"Consider switching to: Qt5Agg, TkAgg, GTK3Agg, wxAgg"
                )
            else:
                log.info(f"Using interactive backend: {current_backend}")

    except Exception as e:
        log.warning(f"Could not configure interactive backend: {e}")

    # Create figure WITHOUT constrained_layout to avoid colorbar conflicts
    analyzer._interactive_fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)

    # Create grid layout: spectrum on left, modes on right
    # Use dynamic number of rows (3 for all visualization types)
    n_vis_types = sum(
        [
            analyzer.config.show_magnitude,
            analyzer.config.show_phase,
            analyzer.config.show_combined,
        ]
    )
    if n_vis_types == 0:
        raise ValueError("At least one visualization type must be enabled")

    gs = gridspec.GridSpec(
        n_vis_types,
        n_components + 1,
        width_ratios=[analyzer.config.spectrum_width_ratio]
        + [analyzer.config.modes_width_ratio / n_components] * n_components,
        height_ratios=[1] * n_vis_types,
    )

    # Spectrum plot spans all rows in first column
    ax_spectrum = analyzer._interactive_fig.add_subplot(gs[:, 0])

    # Mode plots in remaining columns - dynamic based on enabled visualizations
    analyzer._mode_axes = np.array(
        [
            [
                analyzer._interactive_fig.add_subplot(gs[row, col + 1])
                for col in range(n_components)
            ]
            for row in range(n_vis_types)
        ]
    )

    # Plot spectrum
    # Debug: check spectrum shape
    log.debug(f"Spectrum shape before mask: {spectrum_to_use.shape if hasattr(spectrum_to_use, 'shape') else 'N/A'}")
    log.debug(f"Frequencies shape: {frequencies_to_use.shape if hasattr(frequencies_to_use, 'shape') else 'N/A'}")
    log.debug(f"f_min={analyzer.config.f_min}, f_max={analyzer.config.f_max}")
    
    # Handle multi-dimensional spectrum - plot each component separately
    has_multi_components = spectrum_to_use.ndim > 1 and spectrum_to_use.shape[-1] <= 3
    n_components = spectrum_to_use.shape[-1] if has_multi_components else 1
    
    # Generate pastel colors for components
    from mmpp.fft.core import generate_pastel_colors
    pastel_colors = generate_pastel_colors(n_components)
    component_labels = [r"$m_x$", r"$m_y$", r"$m_z$"][:n_components]
    
    # Apply frequency mask FIRST
    freq_mask = (frequencies_to_use >= analyzer.config.f_min) & (
        frequencies_to_use <= analyzer.config.f_max
    )
    
    # Ensure we have data to plot
    if not np.any(freq_mask):
        log.warning(f"No frequencies in range [{analyzer.config.f_min}, {analyzer.config.f_max}] GHz. Using all data.")
        freq_mask = np.ones(len(frequencies_to_use), dtype=bool)
    
    freqs_plot = frequencies_to_use[freq_mask]
    
    if has_multi_components:
        log.info(f"Multi-component spectrum detected (shape={spectrum_to_use.shape}), plotting {n_components} curves")
        # For each component
        spectrum_components = []
        for i in range(n_components):
            comp_spectrum = spectrum_to_use[:, i] if spectrum_to_use.ndim == 2 else spectrum_to_use[..., i].mean(axis=tuple(range(1, spectrum_to_use.ndim - 1)))
            spectrum_components.append(comp_spectrum[freq_mask])
    else:
        spectrum_1d = spectrum_to_use
        spectrum_components = [spectrum_1d[freq_mask]]
    
    # For peak detection later, store averaged version
    spectrum_segment = np.mean(np.array(spectrum_components), axis=0) if len(spectrum_components) > 1 else spectrum_components[0]
    
    log.debug(f"After mask: freqs_plot.shape={freqs_plot.shape}")

    # Determine scale factor for Y-axis label
    all_max = max(np.max(s) for s in spectrum_components) if spectrum_components else 1
    scale_factor = 1
    
    if analyzer.config.spectrum_log_scale:
        ax_spectrum.set_ylabel("log₁₀(Power)")
    else:
        # Professional Y-axis label with exponent
        if all_max > 0:
            exponent = int(np.floor(np.log10(all_max)))
            if abs(exponent) >= 2:
                scale_factor = 10 ** exponent
                ax_spectrum.set_ylabel(f"Power (×10$^{{{exponent}}}$ arb. u.)")
            else:
                ax_spectrum.set_ylabel("Power (arb. u.)")
        else:
            ax_spectrum.set_ylabel("Power (arb. u.)")

    # Plot each component with pastel colors
    for i, comp_spectrum in enumerate(spectrum_components):
        spectrum_plot = comp_spectrum.copy()
        
        if analyzer.config.spectrum_normalize and spectrum_plot.size:
            max_val = np.max(spectrum_plot)
            if max_val > 0:
                spectrum_plot = spectrum_plot / max_val
        
        if analyzer.config.spectrum_log_scale:
            spectrum_plot = np.log10(spectrum_plot + 1e-10)
        else:
            spectrum_plot = spectrum_plot / scale_factor
        
        label = component_labels[i] if has_multi_components else "Power"
        ax_spectrum.plot(freqs_plot, spectrum_plot, 
                        color=pastel_colors[i], linewidth=1.8, alpha=0.9,
                        label=label)
    
    # Add legend if multiple components
    if has_multi_components:
        ax_spectrum.legend(loc='upper right', frameon=True, fancybox=True,
                          framealpha=0.9, edgecolor='lightgray', fontsize=9)
    
    ax_spectrum.set_xlabel("Frequency (GHz)")
    
    # Add component label to title if available
    if spectrum_result is not None and hasattr(spectrum_result, 'component_label') and spectrum_result.component_label:
        ax_spectrum.set_title(f"FMR Spectrum {spectrum_result.component_label} (Click to select frequency)")
    else:
        ax_spectrum.set_title("FMR Spectrum (Click to select frequency)")
    ax_spectrum.grid(True, alpha=0.3, linestyle='--')
    
    # Clean up spines
    ax_spectrum.spines['top'].set_visible(False)
    ax_spectrum.spines['right'].set_visible(False)

    # Find and mark peaks using the same spectrum data
    peaks = analyzer.find_peaks(
        spectrum=spectrum_to_use, frequencies=frequencies_to_use
    )
    for peak in peaks:
        if analyzer.config.f_min <= peak.freq <= analyzer.config.f_max:
            y_val = spectrum_plot[np.argmin(np.abs(freqs_plot - peak.freq))]
            # Professional peak markers
            ax_spectrum.plot(peak.freq, y_val, 'o', 
                           color='#E74C3C', markersize=6, 
                           markeredgecolor='white', markeredgewidth=1.5, zorder=5)
            ax_spectrum.annotate(
                f"{peak.freq:.2f} GHz",
                xy=(peak.freq, y_val),
                xytext=(5, 8),
                textcoords='offset points',
                fontsize=8,
                color='#2C3E50',
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='white',
                    edgecolor='#E74C3C',
                    alpha=0.85,
                    linewidth=0.8
                ),
                zorder=10
            )

    if show_peak_width and freqs_plot.size and spectrum_segment.size:
        width_info = compute_half_width_at_half_max(freqs_plot, spectrum_segment)
        if width_info is None:
            log.debug("FWHM annotation skipped: could not determine half-width")
        else:
            scale_factor = (
                width_info.peak_value if analyzer.config.spectrum_normalize else 1.0
            )
            if scale_factor <= 0:
                log.debug("FWHM annotation skipped: invalid scale factor")
            else:
                half_level_plot = width_info.half_level / scale_factor
                if analyzer.config.spectrum_log_scale:
                    if half_level_plot > 0:
                        half_level_plot = np.log10(half_level_plot + 1e-10)
                        delta = 0.05
                        ymin = half_level_plot - delta
                        ymax = half_level_plot + delta
                    else:
                        half_level_plot = None
                else:
                    delta = max(abs(half_level_plot) * 0.05, 0.02)
                    ymin = half_level_plot - delta
                    ymax = half_level_plot + delta

                if half_level_plot is not None:
                    color = "tab:orange"
                    h_line = ax_spectrum.hlines(
                        half_level_plot,
                        width_info.left_frequency,
                        width_info.right_frequency,
                        colors=color,
                        linewidth=1.5,
                        linestyles="-",
                        alpha=0.9,
                        zorder=5,
                    )
                    analyzer._fwhm_artists.append(h_line)

                    v_lines = ax_spectrum.vlines(
                        [width_info.left_frequency, width_info.right_frequency],
                        ymin=ymin,
                        ymax=ymax,
                        colors=color,
                        linewidth=1.2,
                        alpha=0.9,
                        zorder=5,
                    )
                    analyzer._fwhm_artists.append(v_lines)

                    text = ax_spectrum.annotate(
                        f"{peak_width_label}: {format_width_value(width_info.width)}",
                        xy=(
                            (width_info.left_frequency + width_info.right_frequency)
                            / 2.0,
                            half_level_plot,
                        ),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color=color,
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "facecolor": "white",
                            "edgecolor": color,
                            "linewidth": 0.8,
                            "alpha": 0.85,
                        },
                        zorder=6,
                    )
                    analyzer._fwhm_artists.append(text)
                    analyzer._last_fwhm = width_info
                else:
                    log.debug(
                        "FWHM annotation skipped: non-positive half level for log axis"
                    )

    # Initial frequency line - select peak with highest amplitude
    if peaks:
        # Find peak with highest amplitude
        max_peak = max(peaks, key=lambda p: p.amplitude)
        init_freq = max_peak.freq
        log.info(f"Initialized at peak frequency {init_freq:.3f} GHz (amplitude={max_peak.amplitude:.2e})")
    else:
        # Fallback to middle frequency if no peaks found
        init_freq = freqs_plot[len(freqs_plot) // 2]
        log.warning(f"No peaks found, using middle frequency {init_freq:.3f} GHz")
    
    analyzer._frequency_line = ax_spectrum.axvline(
        init_freq, color="red", linestyle="--", linewidth=2, alpha=0.8
    )

    # Plot initial mode
    analyzer._current_frequency = init_freq
    update_mode_plots(analyzer, components, z_layer)

    # Set up click handler with proper cleanup
    def on_click(event):
        # Handle spectrum clicks (single click only)
        if event.inaxes == ax_spectrum and event.xdata is not None:
            if event.button == 3:  # Right click - snap to peak
                if peaks:
                    peak_freqs = [p.freq for p in peaks]
                    closest_peak_freq = peak_freqs[
                        np.argmin(np.abs(np.array(peak_freqs) - event.xdata))
                    ]
                    selected_freq = closest_peak_freq
                else:
                    selected_freq = event.xdata
            else:  # Left click - exact frequency
                selected_freq = event.xdata

            # Update frequency line and mode plots
            analyzer._frequency_line.set_xdata([selected_freq, selected_freq])
            analyzer._current_frequency = selected_freq
            update_mode_plots(analyzer, components, z_layer)
            analyzer._interactive_fig.canvas.draw()

        # Handle double-clicks on mode plots for animation
        elif event.dblclick and event.inaxes is not None:
            # Find which mode axis was double-clicked
            for row_idx, ax_row in enumerate(analyzer._mode_axes):
                for col_idx, ax in enumerate(ax_row):
                    if event.inaxes == ax:
                        # Use stored components list
                        if hasattr(analyzer, '_current_components') and col_idx < len(analyzer._current_components):
                            component = analyzer._current_components[col_idx]
                            log.debug(f"Double-click detected on axis ({row_idx}, {col_idx}) for component {component}")
                            _toggle_mode_animation(
                                analyzer, ax, row_idx, col_idx, component, z_layer
                            )
                        else:
                            log.warning(f"Could not determine component for axis ({row_idx}, {col_idx})")
                        return

    # Store event connection for cleanup
    analyzer._click_connection = analyzer._interactive_fig.canvas.mpl_connect(
        "button_press_event", on_click
    )
    log.debug(f"Click handler connected with ID: {analyzer._click_connection}")

    # Add keyboard handler for mode characterization
    def on_key_press(event):
        """Handle keyboard events for mode characterization"""
        if event is None or event.key is None:
            return

        log.debug(
            f"Key pressed: '{event.key}' at frequency {analyzer._current_frequency}"
        )

        if event.key == "c" and analyzer._current_frequency is not None:
            try:
                log.info(
                    f"Characterizing mode at {analyzer._current_frequency:.3f} GHz..."
                )
                characterization = analyzer.characterize_mode(
                    analyzer._current_frequency, z_layer, verbose=False
                )

                # Display characterization results
                char_info = (
                    f"Mode Classification at {analyzer._current_frequency:.3f} GHz:\n"
                    f"• Primary Class: {characterization.primary_class}\n"
                    f"• m-index: {characterization.m_index}\n"
                    f"• Rotation: {characterization.rotation_sense or 'N/A'}\n"
                    f"• Radial nodes: {characterization.radial_nodes}\n"
                    f"• Confidence: {characterization.confidence:.2f}\n"
                    f"• Labels: {', '.join(characterization.labels)}"
                )

                print("\n" + "=" * 60)
                print(char_info)
                print("=" * 60)

                # Update figure title with classification
                main_title = f"Interactive Mode Spectrum - {characterization.primary_class.upper()} mode"
                if characterization.m_index is not None:
                    main_title += f" (m={characterization.m_index})"
                if characterization.rotation_sense:
                    main_title += f" [{characterization.rotation_sense}]"

                analyzer._interactive_fig.suptitle(
                    main_title, fontsize=12, fontweight="bold"
                )
                analyzer._interactive_fig.canvas.draw()

                log.info(
                    f"Mode classified as: {characterization.primary_class} (confidence: {characterization.confidence:.2f})"
                )

            except Exception as e:
                log.error(f"Failed to characterize mode: {e}")
                print(f"Error characterizing mode: {e}")

        elif event.key == "v" and analyzer._current_frequency is not None:
            # Verbose mode characterization - show detailed calculations
            try:
                print(
                    f"\n🔍 VERBOSE MODE CHARACTERIZATION at {analyzer._current_frequency:.3f} GHz"
                )
                print("=" * 70)
                log.info(
                    f"Verbose characterizing mode at {analyzer._current_frequency:.3f} GHz..."
                )
                characterization = analyzer.characterize_mode(
                    analyzer._current_frequency, z_layer, verbose=True
                )

                # Update figure title with classification
                main_title = f"Interactive Mode Spectrum - {characterization.primary_class.upper()} mode (VERBOSE)"
                if characterization.m_index is not None:
                    main_title += f" (m={characterization.m_index})"
                if characterization.rotation_sense:
                    main_title += f" [{characterization.rotation_sense}]"

                analyzer._interactive_fig.suptitle(
                    main_title, fontsize=12, fontweight="bold"
                )
                analyzer._interactive_fig.canvas.draw()
                print("=" * 70)

            except Exception as e:
                log.error(f"Failed to verbose characterize mode: {e}")
                print(f"❌ Error in verbose mode characterization: {e}")
                import traceback

                traceback.print_exc()

        elif event.key == "s" and analyzer._saveanim_enabled:
            # Save animation of current animated modes
            try:
                if not analyzer._mode_animations:
                    print(
                        "❌ No active animations to save! Double-click mode plots first."
                    )
                    return

                log.info(f"Saving animation to: {analyzer._saveanim_path}")
                print(
                    f"💾 Saving animation with {len(analyzer._mode_animations)} animated modes..."
                )

                _save_animated_view(analyzer, analyzer._saveanim_path, z_layer)
                print(f"✅ Animation saved to: {analyzer._saveanim_path}")

            except Exception as e:
                log.error(f"Failed to save animation: {e}")
                print(f"❌ Error saving animation: {e}")

        elif event.key == "h":
            # Show help
            help_controls = [
                "• Click spectrum: Select frequency",
                "• Right-click spectrum: Snap to nearest peak",
                "• Double-click mode: Toggle animation",
                "• 'c' key: Characterize current mode",
                "• 'v' key: Verbose characterization (detailed calculations)",
            ]

            if analyzer._saveanim_enabled:
                help_controls.append("• 's' key: Save animated view")

            help_controls.append("• 'h' key: Show this help")

            help_text = f"""
Interactive Spectrum Controls:
============================
{chr(10).join(help_controls)}
            """
            print(help_text)
        else:
            # Debug: show unhandled keys
            if (
                event.key in ["v", "c", "s", "h"]
                and analyzer._current_frequency is None
            ):
                print(
                    f"⚠️  Key '{event.key}' pressed but no frequency selected. Click spectrum first."
                )
            elif event.key not in ["v", "c", "s", "h", None]:
                log.debug(f"Unhandled key: '{event.key}'")

    # Connect keyboard handler
    analyzer._key_connection = analyzer._interactive_fig.canvas.mpl_connect(
        "key_press_event", on_key_press
    )
    log.debug(f"Keyboard handler connected with ID: {analyzer._key_connection}")

    # Add cleanup method to figure
    def cleanup():
        # Stop all running animations first
        if hasattr(analyzer, "_mode_animations") and analyzer._mode_animations:
            log.debug(
                f"Stopping {len(analyzer._mode_animations)} running animations..."
            )
            for animation in analyzer._mode_animations.values():
                try:
                    animation.event_source.stop()
                except AttributeError:
                    pass  # Animation might not have been started yet
            analyzer._mode_animations.clear()
            if hasattr(analyzer, "_animated_axes"):
                analyzer._animated_axes.clear()

        # Disconnect event handlers
        if hasattr(analyzer, "_click_connection") and analyzer._click_connection:
            analyzer._interactive_fig.canvas.mpl_disconnect(analyzer._click_connection)
            analyzer._click_connection = None
            log.debug("Click handler disconnected")

        if hasattr(analyzer, "_key_connection") and analyzer._key_connection:
            analyzer._interactive_fig.canvas.mpl_disconnect(analyzer._key_connection)
            analyzer._key_connection = None
            log.debug("Keyboard handler disconnected")

        log.debug("Interactive plot event handlers cleaned up")

    # Store cleanup function for later use
    analyzer._interactive_fig._mmpp_cleanup = cleanup

    plt.tight_layout()

    # Auto-animate all mode plots if requested
    if auto_animate:
        log.info("Auto-animating all mode plots...")
        vis_types = []
        if analyzer.config.show_magnitude:
            vis_types.append("magnitude")
        if analyzer.config.show_phase:
            vis_types.append("phase")
        if analyzer.config.show_combined:
            vis_types.append("combined")

        for row_idx in range(len(vis_types)):
            for col_idx in range(len(components)):
                try:
                    ax = analyzer._mode_axes[row_idx][col_idx]
                    component = components[col_idx]
                    _toggle_mode_animation(
                        analyzer, ax, row_idx, col_idx, component, z_layer
                    )
                except Exception as e:
                    log.warning(
                        f"Failed to auto-animate mode plot at ({row_idx}, {col_idx}): {e}"
                    )

        # Auto-save animation if requested
        if auto_save and analyzer._saveanim_enabled:
            try:
                if analyzer._mode_animations:
                    log.info(f"Auto-saving animation to: {analyzer._saveanim_path}")
                    print(
                        f"🎬 Auto-saving animation with {len(analyzer._mode_animations)} animated modes..."
                    )

                    _save_animated_view(analyzer, analyzer._saveanim_path, z_layer)
                    log.info("✅ Animation auto-saved successfully!")
                    print(f"✅ Animation auto-saved to: {analyzer._saveanim_path}")
                else:
                    log.warning("No animations were started for auto-save")
                    print("⚠️ No animations to auto-save")
            except Exception as e:
                log.error(f"Auto-save failed: {e}")
                print(f"❌ Auto-save failed: {e}")
        elif auto_save and not analyzer._saveanim_enabled:
            log.warning("auto_save=True requires saveanim to be enabled")
            print(
                "⚠️ auto_save=True requires saveanim parameter (True or custom path)"
            )

    # Update log message based on animation saving capability
    log_message = (
        "Interactive spectrum plot created. Click to select frequency, right-click to snap to peaks. "
        "Double-click mode plots for animations. Press 'c' to characterize current mode"
    )
    if analyzer._saveanim_enabled:
        if not auto_save:
            log_message += ", 's' to save animated view"
    if auto_animate:
        log_message += " (all animations auto-started)"
    if auto_save:
        log_message += " (animation auto-saved)"
    log_message += ", 'h' for help."

    log.info(log_message)

    # Control figure display to avoid double showing
    if show:
        plt.show()
        return None  # Don't return figure to avoid Jupyter auto-display
    else:
        return analyzer._interactive_fig


def update_mode_plots(
    analyzer, components: list[Union[int, str]], z_layer: int
) -> None:
    """Update mode plots for current frequency."""
    if analyzer._mode_axes is None or analyzer._current_frequency is None:
        return

    # Check for active animations and restart them with new frequency data
    has_active_animations = (
        hasattr(analyzer, "_mode_animations") and analyzer._mode_animations
    )
    active_animation_keys = set()

    if has_active_animations:
        # Store which axes were animated
        active_animation_keys = set(analyzer._animated_axes)
        log.debug(
            f"Found {len(active_animation_keys)} active animations, will restart them"
        )

        # Stop all current animations
        for axis_key in list(analyzer._mode_animations.keys()):
            _stop_mode_animation(analyzer, axis_key)

    # Clear previous shared colorbars safely
    for cbar in getattr(analyzer, "_row_colorbars", []):
        try:
            if cbar is not None and hasattr(cbar, "ax") and cbar.ax is not None:
                cbar.remove()
        except (ValueError, AttributeError) as e:
            # Silently ignore already removed or invalid colorbars
            log.debug(f"Colorbar removal failed: {e}")
            pass
    analyzer._row_colorbars = []

    vis_types = []
    if analyzer.config.show_magnitude:
        vis_types.append("magnitude")
    if analyzer.config.show_phase:
        vis_types.append("phase")
    if analyzer.config.show_combined:
        vis_types.append("combined")

    images_for_colorbar: list[Optional[Any]] = [None] * len(vis_types)

    # Clear all axes (only non-animated ones, or all if we stopped animations)
    for ax_row in analyzer._mode_axes:
        for ax in ax_row:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

    # Get mode data
    try:
        mode_data = analyzer.get_mode(analyzer._current_frequency, z_layer)
    except Exception as e:
        log.error(f"Failed to get mode data: {e}")
        # Show error message on plots instead of leaving them empty
        for ax_row in analyzer._mode_axes:
            for ax in ax_row:
                ax.text(
                    0.5,
                    0.5,
                    f"Error loading mode data:\n{str(e)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                    wrap=True,
                )
        return

    # Plot each component
    for i, comp in enumerate(components):
        try:
            comp_data = mode_data.get_component(comp)
            magnitude = np.abs(comp_data)
            phase = np.angle(comp_data)

            row_idx = 0

            # Magnitude plot (if enabled)
            if analyzer.config.show_magnitude:
                img = analyzer._mode_axes[row_idx, i].imshow(
                    magnitude,
                    cmap=analyzer.config._resolve_colormap(
                        analyzer.config.colormap_magnitude
                    ),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=analyzer.config.interpolation,
                    origin="lower",
                )
                analyzer._mode_axes[row_idx, i].set_title(f"|m_{comp}|")
                if images_for_colorbar[row_idx] is None:
                    images_for_colorbar[row_idx] = img
                if i == 0:
                    _add_scale_bar(
                        analyzer, analyzer._mode_axes[row_idx, i], mode_data.extent
                    )
                row_idx += 1

            # Phase plot (if enabled)
            if analyzer.config.show_phase:
                img = analyzer._mode_axes[row_idx, i].imshow(
                    phase,
                    cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=analyzer.config.interpolation,
                    vmin=-np.pi,
                    vmax=np.pi,
                    origin="lower",
                )
                analyzer._mode_axes[row_idx, i].set_title(f"arg(m_{comp})")
                if images_for_colorbar[row_idx] is None:
                    images_for_colorbar[row_idx] = img
                if i == 0:
                    _add_scale_bar(
                        analyzer, analyzer._mode_axes[row_idx, i], mode_data.extent
                    )
                row_idx += 1

            # Combined plot (if enabled)
            if analyzer.config.show_combined:
                # Create combined visualization: magnitude * cos(phase) for real part
                # or magnitude * sin(phase) for imaginary part
                # This shows the actual complex amplitude with sign
                combined_data = magnitude * np.cos(phase)  # Real part
                # Alternative: combined_data = magnitude * np.sin(phase)  # Imaginary part

                img = analyzer._mode_axes[row_idx, i].imshow(
                    combined_data,
                    cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=analyzer.config.interpolation,
                    origin="lower",
                )
                analyzer._mode_axes[row_idx, i].set_title(f"m_{comp} (mag×cos(φ))")
                if images_for_colorbar[row_idx] is None:
                    images_for_colorbar[row_idx] = img
                if i == 0:
                    _add_scale_bar(
                        analyzer, analyzer._mode_axes[row_idx, i], mode_data.extent
                    )

        except Exception as e:
            log.error(f"Failed to plot component {comp}: {e}")
            continue

    # Add frequency info
    analyzer._interactive_fig.suptitle(
        f"FMR Modes at {analyzer._current_frequency:.3f} GHz",
        fontsize=14,
        fontweight="bold",
    )

    # Create shared colorbars per visualization type - use INSET colorbars for publication quality
    for row_idx, (vis_type, img) in enumerate(zip(vis_types, images_for_colorbar)):
        if img is None:
            continue
        try:
            # Get the rightmost axis in this row for colorbar placement
            rightmost_ax = analyzer._mode_axes[row_idx, -1]
            
            # Determine vmin/vmax from image
            try:
                vmin = img.get_clim()[0]
                vmax = img.get_clim()[1]
            except:
                vmin, vmax = 0, 1
            
            label = analyzer.config.colorbar_labels.get(vis_type, vis_type.title())
            
            # Use inset colorbar if available (publication-ready)
            if INSET_COLORBAR_AVAILABLE:
                _make_inset_colorbar(
                    ax=rightmost_ax,
                    image=img,
                    fig=analyzer._interactive_fig,
                    vmin=vmin,
                    vmax=vmax,
                    label=label,
                    width="40%",
                    height="5%",
                    position="upper right",
                    bg_alpha=0.6,
                    text_color="white",
                    fontsize=8,
                    title_fontsize=9,
                )
            elif AXES_GRID_AVAILABLE:
                from mpl_toolkits.axes_grid1 import make_axes_locatable

                # Use make_axes_locatable for proper positioning
                divider = make_axes_locatable(rightmost_ax)
                cax = divider.append_axes(
                    "right",
                    size=f"{analyzer.config.colorbar_fraction*100}%",
                    pad=analyzer.config.colorbar_pad,
                )
                cbar = analyzer._interactive_fig.colorbar(img, cax=cax)
                cbar.set_label(label, fontsize=analyzer.config.colorbar_label_size)
                cbar.ax.tick_params(labelsize=analyzer.config.colorbar_ticklabel_size)
                analyzer._row_colorbars.append(cbar)
            else:
                # Fallback to basic colorbar positioned at rightmost axis
                cbar = analyzer._interactive_fig.colorbar(
                    img,
                    ax=rightmost_ax,
                    fraction=analyzer.config.colorbar_fraction,
                    pad=analyzer.config.colorbar_pad,
                )
                cbar.set_label(label, fontsize=analyzer.config.colorbar_label_size)
                cbar.ax.tick_params(labelsize=analyzer.config.colorbar_ticklabel_size)
                analyzer._row_colorbars.append(cbar)
        except Exception as exc:
            log.debug(f"Skipping colorbar for {vis_type}: {exc}")

    # Restart animations that were active before the update
    if has_active_animations and active_animation_keys:
        log.debug(
            f"Restarting {len(active_animation_keys)} animations with new frequency data"
        )
        for row_idx, col_idx in active_animation_keys:
            try:
                if row_idx < len(analyzer._mode_axes) and col_idx < len(components):
                    ax = analyzer._mode_axes[row_idx, col_idx]
                    component = components[col_idx]
                    _start_mode_animation(
                        analyzer, ax, row_idx, col_idx, component, z_layer
                    )
                    log.debug(
                        f"Restarted animation for m_{component} at {analyzer._current_frequency:.3f} GHz"
                    )
            except Exception as e:
                log.warning(
                    f"Failed to restart animation for axis ({row_idx}, {col_idx}): {e}"
                )
