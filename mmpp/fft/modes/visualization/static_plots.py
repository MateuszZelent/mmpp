"""
Static plot functions for FMR mode visualization.

Contains functions for creating static mode visualizations:
- plot_modes: Main plotting function for mode components
- _update_single_mode_plot: Update single plot in interactive view
- _add_scale_bar: Add publication-style scale bars
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

log = logging.getLogger("mmpp.fft.modes")

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import utilities
from ..style import setup_animation_styling
from ..utils.scalebar import calculate_optimal_length, format_scalebar_label

try:
    import matplotlib.font_manager as fm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    AXES_GRID_AVAILABLE = True
except ImportError:
    AXES_GRID_AVAILABLE = False


def _visible_mode_rows(config: Any) -> list[str]:
    """Return enabled static mode rows in their display order."""
    rows = []
    if config.show_magnitude:
        rows.append("magnitude")
    if config.show_phase:
        rows.append("phase")
    if config.show_combined:
        rows.append("combined")
    if not rows:
        raise ValueError("At least one mode visualization row must be enabled")
    return rows


def plot_modes(
    analyzer,
    frequency: float,
    z_layer: int = 0,
    components: list[int | str] | None = None,
    save_path: str | None = None,
) -> tuple["Figure", np.ndarray]:
    """
    Plot mode visualization for a specific frequency.

    Parameters:
    -----------
    analyzer : FMRModeAnalyzer
        The analyzer instance
    frequency : float
        Frequency in GHz
    z_layer : int, optional
        Z-layer index (default: 0)
    components : list, optional
        List of components to plot (default: ['x', 'y', 'z'])
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    Tuple[Figure, np.ndarray]
        Matplotlib figure and axes array
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for plotting")

    # Setup professional styling for mode plots
    setup_animation_styling(use_paper_style=True, use_custom_fonts=True)

    components = ["x", "y", "z"] if components is None else list(components)
    if not components:
        raise ValueError("components must contain at least one component")
    mode_data = analyzer.get_mode(frequency, z_layer)

    # Create figure with subplots
    n_components = len(components)
    visible_rows = _visible_mode_rows(analyzer.config)
    n_rows = len(visible_rows)
    actual_frequency = mode_data.frequency

    fig, axes = plt.subplots(
        n_rows,
        n_components,
        figsize=(4 * n_components, 3 * n_rows),
        dpi=analyzer.config.dpi,
    )

    if n_components == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each component
    for i, comp in enumerate(components):
        comp_data = mode_data.get_component(comp)
        magnitude = np.abs(comp_data)
        phase = np.angle(comp_data)

        row = 0

        # Magnitude plot
        if analyzer.config.show_magnitude:
            im1 = axes[row, i].imshow(
                magnitude,
                cmap=analyzer.config._resolve_colormap(
                    analyzer.config.colormap_magnitude
                ),
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                origin="lower",
            )
            axes[row, i].set_title(f"|m_{comp}| @ {actual_frequency:.3f} GHz")
            axes[row, i].set_xlabel("x (nm)")
            if i == 0:
                axes[row, i].set_ylabel("y (nm)")
            plt.colorbar(im1, ax=axes[row, i], shrink=0.8)
            row += 1

        # Phase plot
        if analyzer.config.show_phase:
            im2 = axes[row, i].imshow(
                phase,
                cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                vmin=-np.pi,
                vmax=np.pi,
                origin="lower",
            )
            axes[row, i].set_title(f"arg(m_{comp}) @ {actual_frequency:.3f} GHz")
            axes[row, i].set_xlabel("x (nm)")
            if i == 0:
                axes[row, i].set_ylabel("y (nm)")
            plt.colorbar(im2, ax=axes[row, i], shrink=0.8)
            row += 1

        # Combined plot (phase with magnitude as alpha)
        if analyzer.config.show_combined:
            # Create combined visualization: magnitude * cos(phase) for real part
            combined_data = magnitude * np.cos(phase)  # Real part
            vmax = max(float(np.max(np.abs(combined_data))), np.finfo(float).eps)

            im3 = axes[row, i].imshow(
                combined_data,
                cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                origin="lower",
                vmin=-vmax,
                vmax=vmax,
            )
            axes[row, i].set_title(
                f"m_{comp} combined (mag×cos(φ)) @ {actual_frequency:.3f} GHz"
            )
            axes[row, i].set_xlabel("x (nm)")
            if i == 0:
                axes[row, i].set_ylabel("y (nm)")
            plt.colorbar(im3, ax=axes[row, i], shrink=0.8)

    plt.tight_layout()

    # Save if requested
    if save_path:
        Path(save_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=analyzer.config.dpi, bbox_inches="tight")
        log.info(f"Saved mode plots to {save_path}")

    return fig, axes


def update_single_mode_plot(
    analyzer,
    ax: Any,
    row_idx: int,
    col_idx: int,
    component: str | int,
    z_layer: int,
) -> None:
    """Update single mode plot (used when stopping animation)."""
    mode_data = analyzer.get_mode(analyzer._current_frequency, z_layer)
    comp_data = mode_data.get_component(component)
    vis_types = _visible_mode_rows(analyzer.config)
    if row_idx < 0 or row_idx >= len(vis_types):
        raise IndexError(f"Invalid mode visualization row: {row_idx}")
    vis_type = vis_types[row_idx]

    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])

    from ..vortex_optics import VortexOptics

    comp_label = VortexOptics.get_component_label(str(component), latex=True)

    if vis_type == "magnitude":
        ax.imshow(
            np.abs(comp_data),
            cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_magnitude),
            extent=mode_data.extent,
            aspect="equal",
            interpolation=analyzer.config.interpolation,
            origin="lower",
        )
        ax.set_title(f"|{comp_label}|")

    elif vis_type == "phase":
        use_holo = getattr(analyzer.config, "use_holography", False)
        if use_holo:
            holo_gamma = getattr(analyzer.config, "holography_gamma", 0.6)
            holo_noise = getattr(analyzer.config, "holography_noise_threshold", 1e-4)
            holo_img = VortexOptics.complex_holography(
                comp_data, holo_gamma, holo_noise
            )
            ax.imshow(holo_img, extent=mode_data.extent, aspect="equal", origin="lower")
            ax.set_title(f"Hologram of {comp_label}")
        else:
            ax.imshow(
                np.angle(comp_data),
                cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                vmin=-np.pi,
                vmax=np.pi,
                origin="lower",
            )
            ax.set_title(f"arg({comp_label})")

    elif vis_type == "combined":
        combined_data = np.real(comp_data)
        vmax = max(float(np.max(np.abs(combined_data))), np.finfo(float).eps)
        ax.imshow(
            combined_data,
            cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
            extent=mode_data.extent,
            aspect="equal",
            interpolation=analyzer.config.interpolation,
            origin="lower",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"Re[{comp_label}]")


def add_scale_bar(analyzer, ax: Any, extent: tuple[float, float, float, float]) -> None:
    """Add a publication-style scale bar to the supplied axis."""
    if not (analyzer.config.show_scalebar and AXES_GRID_AVAILABLE):
        return

    x_min, x_max, y_min, y_max = extent
    width_nm = float(x_max - x_min)
    height_nm = float(y_max - y_min)
    if width_nm <= 0 or height_nm <= 0:
        return

    bar_length = (
        analyzer.config.scalebar_length_nm
        if analyzer.config.scalebar_length_nm is not None
        else calculate_optimal_length(width_nm)
    )
    if bar_length is None or bar_length <= 0:
        return

    size_vertical = height_nm * analyzer.config.scalebar_height_fraction
    label = format_scalebar_label(bar_length, units=analyzer.config.scale_units)

    # Larger font size for better visibility
    scalebar_fontsize = max(analyzer.config.scalebar_fontsize, 11)

    try:
        scalebar = AnchoredSizeBar(
            ax.transData,
            bar_length,
            label,
            analyzer.config.scalebar_location,
            pad=analyzer.config.scalebar_pad,
            color=analyzer.config.scalebar_color,
            frameon=True,  # Enable frame for background
            size_vertical=size_vertical,
            fontproperties=fm.FontProperties(size=scalebar_fontsize, weight="bold"),
        )
        # Add semi-transparent black background (30% alpha)
        scalebar.patch.set_facecolor((0, 0, 0, 0.3))
        scalebar.patch.set_edgecolor("none")
    except Exception as exc:
        log.debug(f"Could not create scale bar: {exc}")
        return

    ax.add_artist(scalebar)
