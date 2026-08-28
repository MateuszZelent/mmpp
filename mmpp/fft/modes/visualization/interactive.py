"""
Interactive spectrum visualization with mode plots.

This module provides the interactive spectrum plotting functionality that allows
users to click on the spectrum to visualize corresponding FMR modes.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from ..compat import MATPLOTLIB_AVAILABLE

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec


from ..style import STYLING_AVAILABLE, load_paper_style
from .animation import (
    save_animated_view as _save_animated_view,
)
from .animation import (
    start_mode_animation as _start_mode_animation,
)
from .animation import (
    stop_mode_animation as _stop_mode_animation,
)
from .animation import (
    toggle_mode_animation as _toggle_mode_animation,
)


def add_scale_bar(analyzer, ax, extent):
    """Add scale bar to axis - imported from static_plots via analyzer."""
    from .static_plots import add_scale_bar as _add_scale_bar_impl

    return _add_scale_bar_impl(analyzer, ax, extent)


_add_scale_bar = add_scale_bar  # For local use


def _spectrum_result_frequency_axis_ghz(spectrum_result: Any) -> np.ndarray:
    """Read the public SpectrumResult frequency contract without unit guessing."""
    if hasattr(spectrum_result, "frequencies_ghz"):
        frequencies = np.asarray(spectrum_result.frequencies_ghz, dtype=float)
    else:
        frequencies = (
            np.asarray(getattr(spectrum_result, "frequencies", []), dtype=float) * 1e-9
        )
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("SpectrumResult frequency axis must be a non-empty 1D array")
    if not np.isfinite(frequencies).all():
        raise ValueError("SpectrumResult frequency axis must be finite")
    return frequencies


# Utility functions for peak width annotation
@dataclass(frozen=True)
class WidthInfo:
    peak_frequency: float
    peak_value: float
    half_level: float
    left_frequency: float
    right_frequency: float
    width: float


def normalize_peak_width_option(option: Any) -> tuple[bool, str]:
    """
    Normalize the peak_width option to (show: bool, label: str).

    Returns:
        tuple: (show_peak_width: bool, peak_width_label: str)
    """
    if option is None or option is False:
        return (False, "FWHM")
    elif option is True:
        return (True, "FWHM")
    if isinstance(option, str):
        label = option.strip()
        if not label:
            raise ValueError("peak_width label must not be empty")
        return (True, label)
    raise TypeError("peak_width must be None, a boolean, or a non-empty label")


def compute_half_width_at_half_max(frequencies, spectrum) -> WidthInfo | None:
    """
    Compute full width at half height for the dominant peak.

    Parameters:
        frequencies: array of frequencies
        spectrum: array of spectrum values

    Returns:
        WidthInfo object, or ``None`` when the peak has no two half-height
        crossings inside the supplied frequency window.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if frequencies.ndim != 1 or spectrum.ndim != 1:
        raise ValueError("frequencies and spectrum must be 1D arrays")
    if frequencies.size != spectrum.size:
        raise ValueError("frequencies and spectrum must have equal lengths")
    if frequencies.size < 3:
        return None
    if not np.isfinite(frequencies).all() or not np.isfinite(spectrum).all():
        raise ValueError("frequencies and spectrum must contain only finite values")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies must be strictly increasing")
    if np.any(spectrum < 0):
        raise ValueError("spectrum must be non-negative for FWHM calculation")

    peak_idx = int(np.argmax(spectrum))
    peak_freq = float(frequencies[peak_idx])
    peak_val = float(spectrum[peak_idx])
    baseline = float(np.min(spectrum))
    peak_height = peak_val - baseline
    if peak_height <= 0:
        return None
    half_val = baseline + 0.5 * peak_height

    left_candidates = np.flatnonzero(spectrum[:peak_idx] <= half_val)
    right_candidates = np.flatnonzero(spectrum[peak_idx + 1 :] <= half_val)
    if left_candidates.size == 0 or right_candidates.size == 0:
        return None

    left_idx = int(left_candidates[-1])
    right_idx = int(peak_idx + 1 + right_candidates[0])

    def interpolate_crossing(i0: int, i1: int) -> float:
        y0, y1 = float(spectrum[i0]), float(spectrum[i1])
        x0, x1 = float(frequencies[i0]), float(frequencies[i1])
        if y1 == y0:
            return 0.5 * (x0 + x1)
        fraction = (half_val - y0) / (y1 - y0)
        return x0 + fraction * (x1 - x0)

    left_freq = interpolate_crossing(left_idx, left_idx + 1)
    right_freq = interpolate_crossing(right_idx - 1, right_idx)
    width = right_freq - left_freq

    return WidthInfo(
        peak_frequency=peak_freq,
        peak_value=peak_val,
        half_level=half_val,
        left_frequency=left_freq,
        right_frequency=right_freq,
        width=width,
    )


def format_width_value(width_ghz: float) -> str:
    """Format width value with appropriate units."""
    width_ghz = float(width_ghz)
    if not np.isfinite(width_ghz) or width_ghz < 0:
        raise ValueError("width_ghz must be finite and non-negative")
    if width_ghz < 0.001:
        return f"{width_ghz * 1000:.2f} MHz"
    else:
        return f"{width_ghz:.3f} GHz"


log = logging.getLogger(__name__)


def interactive_spectrum(
    analyzer,
    components: list[int | str] | None = None,
    z_layer: int = 0,
    method: int = 1,
    show: bool = True,
    force: bool = False,
    use_fft_spectrum: bool = True,
    saveanim: bool | str | None = None,
    auto_animate: bool = False,
    auto_save: bool = False,
    spectrum_result: Any = None,  # NEW: Inject spectrum from FFT.spectrum()
    use_holography: bool = False,  # NEW: Enable complex holography visualization
    **kwargs,
) -> Any:
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
    use_holography : bool, optional
        Enable complex holography (domain coloring) for phase visualization (default: False)
        When True, replaces standard phase plots with RGB images encoding both
        amplitude and phase simultaneously. Particularly useful for:
        - Gyrotropic vortex core modes (circular basis: m+, m-)
        - Azimuthal spin wave modes (cylindrical basis: m_rho, m_phi)
        - Topological singularities and phase defects
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

    show_peak_width, peak_width_label = normalize_peak_width_option(peak_width_option)

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
    analyzer._fwhm_artists = []
    analyzer._last_fwhm = None

    # === PRIORITY: Use injected spectrum_result from FFT.spectrum() ===
    # This ensures consistency with job[0].m[:200,...,1].fft.spectrum() calls
    if spectrum_result is not None:
        log.info("Using injected spectrum_result from FFT (respects slice context)")
        frequencies_to_use = _spectrum_result_frequency_axis_ghz(spectrum_result)
        spectrum_to_use = spectrum_result.power
        # Store component label for plot title
        component_label = getattr(spectrum_result, "component_label", None)
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
                log.debug(
                    f"Detected {n_comp_available} components from mode data shape: {mode_shape}"
                )

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
                    log.warning(
                        f"Unexpected component count {n_comp_available}, using default [x,y,z]"
                    )
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
    analyzer.config.dpi = dpi

    # Update holography setting if provided
    if use_holography:
        analyzer.config.use_holography = use_holography
        log.info("Complex holography enabled for phase visualization")

    # Validate number of components for layout
    if n_components > 3:
        raise ValueError(
            f"Too many components ({n_components}). Maximum supported: 3 (x, y, z)"
        )

    # ── Determine visualization types ────────────────────────────────────────
    n_vis_types = sum(
        [
            analyzer.config.show_magnitude,
            analyzer.config.show_phase,
            analyzer.config.show_combined,
        ]
    )
    if n_vis_types == 0:
        raise ValueError("At least one visualization type must be enabled")

    # ── Auto-size figure based on content ────────────────────────────────────
    _horizontal = n_components == 1  # single-component → horizontal layout
    analyzer._layout_horizontal = _horizontal

    if _horizontal:
        # [spectrum | vis0 | cbar0 | vis1 | cbar1 | … ]
        _fig_w = 5.5 + n_vis_types * 4.2
        _fig_h = 5.0
    else:
        # [spectrum | c0 | c1 | … | cbar]
        _fig_w = 6.0 + n_components * 3.8
        _fig_h = 3.2 * n_vis_types + 1.0

    # Allow user override
    if figsize != (16, 10) and figsize != analyzer.config.figsize:
        _fig_w, _fig_h = figsize
    analyzer.config.figsize = (_fig_w, _fig_h)

    # ── Interactive backend ──────────────────────────────────────────────────
    try:
        import matplotlib

        current_backend = matplotlib.get_backend()
        try:
            from IPython import get_ipython

            ipython = get_ipython()
            in_jupyter = ipython is not None and hasattr(ipython, "kernel")
        except ImportError:
            in_jupyter = False

        if in_jupyter:
            try:
                ipython.run_line_magic("matplotlib", "widget")
                log.info("Switched to widget backend")
            except Exception:
                try:
                    ipython.run_line_magic("matplotlib", "nbagg")
                except Exception:
                    if not any(
                        k in current_backend.lower()
                        for k in ("ipympl", "widget", "nbagg")
                    ):
                        log.warning(
                            f"Backend '{current_backend}' may lack interactivity – run %matplotlib widget"
                        )
        else:
            log.info(f"Backend: {current_backend}")
    except Exception as e:
        log.warning(f"Could not configure interactive backend: {e}")

    # ── Create figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(_fig_w, _fig_h), dpi=dpi, constrained_layout=False)
    analyzer._interactive_fig = fig

    # ── Build GridSpec (adaptive layout) ─────────────────────────────────────
    _SPEC_W = 1.5  # spectrum column weight
    _MODE_W = 1.0  # each mode-image column weight
    _CBAR_W = 0.06  # each colorbar column weight

    if _horizontal:
        # Single component – one row, vis types are columns
        # [spectrum | vis0 | cbar0 | vis1 | cbar1 | vis2 | cbar2]
        n_gcols = 1 + n_vis_types * 2
        w_ratios: list[float] = [_SPEC_W]
        for _ in range(n_vis_types):
            w_ratios.extend([_MODE_W, _CBAR_W])

        gs = gridspec.GridSpec(
            1,
            n_gcols,
            width_ratios=w_ratios,
            hspace=0.05,
            wspace=0.25,
            left=0.06,
            right=0.97,
            top=0.88,
            bottom=0.12,
        )
        ax_spectrum = fig.add_subplot(gs[0, 0])

        # _mode_axes shape: (n_vis_types, 1) – row=vis_type, col=0
        _axes, _cbars = [], []
        for v in range(n_vis_types):
            gc = 1 + v * 2  # GridSpec column for image
            _axes.append([fig.add_subplot(gs[0, gc])])
            _cbars.append(fig.add_subplot(gs[0, gc + 1]))
        analyzer._mode_axes = np.array(_axes)  # (n_vis_types, 1)
        analyzer._cbar_axes = _cbars

    else:
        # Multi-component – rows=vis_types, cols=[spectrum, c0..cN, cbar]
        n_gcols = n_components + 2
        w_ratios = [_SPEC_W] + [_MODE_W] * n_components + [_CBAR_W]

        gs = gridspec.GridSpec(
            n_vis_types,
            n_gcols,
            width_ratios=w_ratios,
            height_ratios=[1.0] * n_vis_types,
            hspace=0.35,
            wspace=0.20,
            left=0.06,
            right=0.97,
            top=0.90,
            bottom=0.08,
        )
        ax_spectrum = fig.add_subplot(gs[:, 0])

        analyzer._mode_axes = np.array(
            [
                [fig.add_subplot(gs[row, col + 1]) for col in range(n_components)]
                for row in range(n_vis_types)
            ]
        )
        analyzer._cbar_axes = [
            fig.add_subplot(gs[row, -1]) for row in range(n_vis_types)
        ]

    # Plot spectrum
    # Debug: check spectrum shape
    log.debug(
        f"Spectrum shape before mask: {spectrum_to_use.shape if hasattr(spectrum_to_use, 'shape') else 'N/A'}"
    )
    log.debug(
        f"Frequencies shape: {frequencies_to_use.shape if hasattr(frequencies_to_use, 'shape') else 'N/A'}"
    )
    log.debug(f"f_min={analyzer.config.f_min}, f_max={analyzer.config.f_max}")

    # Handle multi-dimensional spectrum - plot each component separately
    # NOTE: n_spec_curves is the number of *spectral* curves, NOT the number of
    #       mode components used in the GridSpec layout (which is len(components)).
    has_multi_components = spectrum_to_use.ndim > 1 and spectrum_to_use.shape[-1] <= 3
    n_spec_curves = spectrum_to_use.shape[-1] if has_multi_components else 1

    # Generate pastel colors for spectral curves
    try:
        from mmpp.fft.spectrum._plotting.static import (
            _generate_pastel_colors as _gen_colors,
        )
    except ImportError:
        from matplotlib.colors import to_rgba

        def _gen_colors(n: int) -> list:
            colors = plt.get_cmap("Accent")(np.linspace(0, 1, max(int(n), 3)))
            return [to_rgba(c) for c in colors[: int(n)]]

    pastel_colors = _gen_colors(n_spec_curves)
    component_labels = [r"$m_x$", r"$m_y$", r"$m_z$"][:n_spec_curves]

    # Apply frequency mask FIRST
    freq_mask = (frequencies_to_use >= analyzer.config.f_min) & (
        frequencies_to_use <= analyzer.config.f_max
    )

    # Ensure we have data to plot
    if not np.any(freq_mask):
        log.warning(
            f"No frequencies in range [{analyzer.config.f_min}, {analyzer.config.f_max}] GHz. Using all data."
        )
        freq_mask = np.ones(len(frequencies_to_use), dtype=bool)

    freqs_plot = frequencies_to_use[freq_mask]

    if has_multi_components:
        log.info(
            f"Multi-component spectrum detected (shape={spectrum_to_use.shape}), plotting {n_spec_curves} curves"
        )
        # For each component
        spectrum_components = []
        for i in range(n_spec_curves):
            comp_spectrum = (
                spectrum_to_use[:, i]
                if spectrum_to_use.ndim == 2
                else spectrum_to_use[..., i].mean(
                    axis=tuple(range(1, spectrum_to_use.ndim - 1))
                )
            )
            spectrum_components.append(comp_spectrum[freq_mask])
    else:
        spectrum_1d = spectrum_to_use
        spectrum_components = [spectrum_1d[freq_mask]]

    # For peak detection later, store averaged version
    spectrum_segment = (
        np.mean(np.array(spectrum_components), axis=0)
        if len(spectrum_components) > 1
        else spectrum_components[0]
    )

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
                scale_factor = 10**exponent
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
        ax_spectrum.plot(
            freqs_plot,
            spectrum_plot,
            color=pastel_colors[i],
            linewidth=1.8,
            alpha=0.9,
            label=label,
        )

    # Add legend if multiple components
    if has_multi_components:
        ax_spectrum.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor="lightgray",
            fontsize=9,
        )

    ax_spectrum.set_xlabel("Frequency (GHz)", fontsize=10)

    # ── Professional spectrum panel styling ──────────────────────────────────
    _title_comp = ""
    if (
        spectrum_result is not None
        and hasattr(spectrum_result, "component_label")
        and spectrum_result.component_label
    ):
        _title_comp = f" {spectrum_result.component_label}"
    ax_spectrum.set_title(
        f"FMR Spectrum{_title_comp}",
        fontsize=11,
        fontweight="semibold",
        pad=6,
    )
    ax_spectrum.grid(True, alpha=0.25, linestyle="-", linewidth=0.5, color="#cccccc")
    ax_spectrum.tick_params(axis="both", labelsize=9, direction="in", length=3)
    for spine in ("top", "right"):
        ax_spectrum.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax_spectrum.spines[spine].set_linewidth(0.6)

    # Find and mark peaks using the same spectrum data
    peaks = analyzer.find_peaks(
        spectrum=spectrum_to_use, frequencies=frequencies_to_use
    )
    for peak in peaks:
        if analyzer.config.f_min <= peak.freq <= analyzer.config.f_max:
            y_val = spectrum_plot[np.argmin(np.abs(freqs_plot - peak.freq))]
            # Professional peak markers
            ax_spectrum.plot(
                peak.freq,
                y_val,
                "o",
                color="#E74C3C",
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.5,
                zorder=5,
            )
            ax_spectrum.annotate(
                f"{peak.freq:.2f} GHz",
                xy=(peak.freq, y_val),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=8,
                color="#2C3E50",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "#E74C3C",
                    "alpha": 0.85,
                    "linewidth": 0.8,
                },
                zorder=10,
            )

    if show_peak_width and freqs_plot.size and spectrum_segment.size:
        width_info = compute_half_width_at_half_max(freqs_plot, spectrum_segment)
        if width_info is None:
            log.debug("FWHM annotation skipped: could not determine half-width")
        else:
            width_scale_factor = float(
                width_info.peak_value if analyzer.config.spectrum_normalize else 1.0
            )
            if width_scale_factor <= 0:
                log.debug("FWHM annotation skipped: invalid scale factor")
            else:
                half_level_plot: Any = width_info.half_level / width_scale_factor
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
        log.info(
            f"Initialized at peak frequency {init_freq:.3f} GHz (amplitude={max_peak.amplitude:.2e})"
        )
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
                        if hasattr(analyzer, "_current_components") and col_idx < len(
                            analyzer._current_components
                        ):
                            component = analyzer._current_components[col_idx]
                            log.debug(
                                f"Double-click detected on axis ({row_idx}, {col_idx}) for component {component}"
                            )
                            _toggle_mode_animation(
                                analyzer, ax, row_idx, col_idx, component, z_layer
                            )
                        else:
                            log.warning(
                                f"Could not determine component for axis ({row_idx}, {col_idx})"
                            )
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
    setattr(analyzer._interactive_fig, "_mmpp_cleanup", cleanup)  # noqa: B010

    # Margins are already set in the GridSpec; no tight_layout needed.
    analyzer._interactive_fig.canvas.draw_idle()

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
        if (
            auto_save
            and analyzer._saveanim_enabled
            and analyzer._saveanim_path is not None
        ):
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
            print("⚠️ auto_save=True requires saveanim parameter (True or custom path)")

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

    # Control figure display to avoid double showing in Jupyter
    if show:
        plt.show()
        return None  # Return None to prevent Jupyter auto-display after plt.show()

    # Return figure and axes for user customization when show=False
    # Returns: (fig, ax_spectrum, mode_axes)
    return analyzer._interactive_fig, ax_spectrum, analyzer._mode_axes


def update_mode_plots(analyzer, components: list[int | str], z_layer: int) -> None:
    """Update mode plots for current frequency."""
    if analyzer._mode_axes is None or analyzer._current_frequency is None:
        return

    # Check for active animations and restart them with new frequency data
    has_active_animations = (
        hasattr(analyzer, "_mode_animations") and analyzer._mode_animations
    )
    has_active_column_animations = (
        hasattr(analyzer, "_column_animations") and analyzer._column_animations
    )
    active_animation_keys = set()
    active_column_indices: list[int] = []

    if has_active_animations:
        # Store which axes were animated
        active_animation_keys = set(analyzer._animated_axes)
        log.debug(
            f"Found {len(active_animation_keys)} active animations, will restart them"
        )

        # Stop all current animations
        for axis_key in list(analyzer._mode_animations.keys()):
            _stop_mode_animation(analyzer, axis_key)

    if has_active_column_animations:
        from .animation import stop_column_animation as _stop_col_anim

        active_column_indices = list(range(len(components)))
        for col_idx in active_column_indices:
            _stop_col_anim(analyzer, col_idx)
        log.debug(
            f"Stopped {len(active_column_indices)} column animations for frequency update"
        )

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

    images_for_colorbar: list[Any | None] = [None] * len(vis_types)

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

    # ── Pre-compute shared rendering parameters (constant across all components) ──
    try:
        from ..vortex_optics import VortexOptics

        _have_vortex = True
    except ImportError:
        _have_vortex = False

    use_holo = getattr(analyzer.config, "use_holography", False)
    holo_gamma = getattr(analyzer.config, "holography_gamma", 0.5)
    holo_noise = getattr(analyzer.config, "holography_noise_threshold", 1e-4)

    # Plot each component
    for i, comp in enumerate(components):
        try:
            comp_data = mode_data.get_component(comp)
            magnitude = np.abs(comp_data)
            phase = np.angle(comp_data)

            (
                VortexOptics.get_component_label(str(comp), latex=False)
                if _have_vortex
                else str(comp)
            )

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
                if images_for_colorbar[row_idx] is None:
                    images_for_colorbar[row_idx] = img
                if i == 0:
                    _add_scale_bar(
                        analyzer, analyzer._mode_axes[row_idx, i], mode_data.extent
                    )
                row_idx += 1

            # Phase plot (if enabled)
            if analyzer.config.show_phase:
                ax_phase = analyzer._mode_axes[row_idx, i]
                if use_holo and _have_vortex:
                    # Holographic domain coloring (amplitude + phase → HSV→RGB)
                    holo_img = VortexOptics.complex_holography(
                        comp_data, gamma=holo_gamma, noise_threshold=holo_noise
                    )
                    img = ax_phase.imshow(
                        holo_img,
                        extent=mode_data.extent,
                        aspect="equal",
                        interpolation=analyzer.config.interpolation,
                        origin="lower",
                    )
                    # Title is set in the post-loop label block
                else:
                    img = ax_phase.imshow(
                        phase,
                        cmap=analyzer.config._resolve_colormap(
                            analyzer.config.colormap_phase
                        ),
                        extent=mode_data.extent,
                        aspect="equal",
                        interpolation=analyzer.config.interpolation,
                        vmin=-np.pi,
                        vmax=np.pi,
                        origin="lower",
                    )
                if images_for_colorbar[row_idx] is None and not use_holo:
                    images_for_colorbar[row_idx] = img
                if i == 0:
                    _add_scale_bar(analyzer, ax_phase, mode_data.extent)
                row_idx += 1

            # Combined plot (if enabled)
            if analyzer.config.show_combined:
                # Re[m·exp(-iφ)] with fixed colour scale → no frame-to-frame flicker
                vmax_combined = float(np.nanmax(magnitude)) or 1.0
                combined_data = magnitude * np.cos(phase)

                img = analyzer._mode_axes[row_idx, i].imshow(
                    combined_data,
                    cmap=analyzer.config._resolve_colormap(
                        analyzer.config.colormap_phase
                    ),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=analyzer.config.interpolation,
                    vmin=-vmax_combined,
                    vmax=vmax_combined,
                    origin="lower",
                )
                if images_for_colorbar[row_idx] is None:
                    images_for_colorbar[row_idx] = img
                if i == 0:
                    _add_scale_bar(
                        analyzer, analyzer._mode_axes[row_idx, i], mode_data.extent
                    )

        except Exception as e:
            log.error(f"Failed to plot component {comp}: {e}")
            continue

    # ── Labels (adaptive: horizontal vs vertical) ────────────────────────────
    _vis_label = {
        "magnitude": "Amplitude",
        "phase": "Hologram" if use_holo else "Phase",
        "combined": "Re[m(t)]",
    }

    _is_horiz = getattr(analyzer, "_layout_horizontal", False)

    if _is_horiz:
        # Horizontal (1 component): each vis type is a column → use title
        _comp_lbl = (
            VortexOptics.get_component_label(str(components[0]), latex=False)
            if _have_vortex
            else str(components[0])
        )
        for v_idx, vis_type in enumerate(vis_types):
            ax = analyzer._mode_axes[v_idx, 0]
            ax.set_title(
                f"{_vis_label.get(vis_type, vis_type)} |{_comp_lbl}|"
                if vis_type == "magnitude"
                else f"{_vis_label.get(vis_type, vis_type)} ({_comp_lbl})",
                fontsize=9,
                fontweight="semibold",
                pad=4,
            )
    else:
        # Vertical (multi-component): row labels on left, column headers on top
        for r_idx, vis_type in enumerate(vis_types):
            if r_idx < len(analyzer._mode_axes) and analyzer._mode_axes.shape[1] > 0:
                analyzer._mode_axes[r_idx, 0].set_ylabel(
                    _vis_label.get(vis_type, vis_type),
                    fontsize=9,
                    labelpad=6,
                    rotation=90,
                )
        for c_idx, comp in enumerate(components):
            if (
                analyzer._mode_axes.shape[0] > 0
                and c_idx < analyzer._mode_axes.shape[1]
            ):
                _cl = (
                    VortexOptics.get_component_label(str(comp), latex=False)
                    if _have_vortex
                    else str(comp)
                )
                analyzer._mode_axes[0, c_idx].set_title(
                    _cl,
                    fontsize=10,
                    fontweight="semibold",
                    pad=4,
                )

    # Suptitle with frequency
    analyzer._interactive_fig.suptitle(
        f"FMR Modes @ {analyzer._current_frequency:.3f} GHz",
        fontsize=13,
        fontweight="bold",
        y=0.98 if _is_horiz else 0.96,
    )

    # ── Per-row colorbars in dedicated axes ──────────────────────────────────
    _CBAR_LABELS = {
        "magnitude": "Amplitude",
        "phase": "Phase (rad)",
        "combined": "Re[m]",
    }
    _PHASE_TICKS = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
    _PHASE_TLBLS = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]

    _cbar_axes_list = getattr(analyzer, "_cbar_axes", [])
    for row_idx, (vis_type, img) in enumerate(
        zip(vis_types, images_for_colorbar, strict=False)
    ):
        if img is None or row_idx >= len(_cbar_axes_list):
            continue
        cax = _cbar_axes_list[row_idx]
        try:
            cax.clear()
            cbar = analyzer._interactive_fig.colorbar(
                img,
                cax=cax,
                orientation="vertical",
            )
            cbar.set_label(
                _CBAR_LABELS.get(vis_type, vis_type.capitalize()),
                fontsize=8,
                labelpad=3,
            )
            cbar.ax.tick_params(labelsize=7, length=2, pad=1, width=0.5)
            cbar.outline.set_linewidth(0.4)
            # Phase tick labels
            if vis_type == "phase" and not use_holo:
                try:
                    cbar.set_ticks(_PHASE_TICKS)
                    cbar.set_ticklabels(_PHASE_TLBLS)
                except Exception:
                    pass
            elif vis_type == "magnitude":
                cbar.ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.2g}")
                )
            analyzer._row_colorbars.append(cbar)
        except Exception as exc:
            log.debug(f"Colorbar for row {row_idx} ({vis_type}) failed: {exc}")

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

    # Restart column animations that were active before the update
    if active_column_indices:
        from .animation import start_column_animation as _start_col_anim

        frames = analyzer.config.animation_time_steps
        for col_idx in active_column_indices:
            if col_idx < len(components):
                try:
                    _start_col_anim(
                        analyzer, col_idx, components[col_idx], z_layer, frames=frames
                    )
                    log.debug(
                        f"Restarted column animation for col={col_idx} at {analyzer._current_frequency:.3f} GHz"
                    )
                except Exception as e:
                    log.warning(
                        f"Failed to restart column animation for col={col_idx}: {e}"
                    )
