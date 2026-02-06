"""
Static plot functions for FMR mode visualization.

Contains functions for creating static mode visualizations:
- plot_modes: Main plotting function for mode components
- _update_single_mode_plot: Update single plot in interactive view
- _add_scale_bar: Add publication-style scale bars
"""

import numpy as np
import logging
from typing import Any, Optional, Union, TYPE_CHECKING

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

# Import for scale bar
try:
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    AXES_GRID_AVAILABLE = True
except ImportError:
    AXES_GRID_AVAILABLE = False


def plot_modes(
    analyzer,
    frequency: float,
    z_layer: int = 0,
    components: Optional[list[Union[int, str]]] = None,
    save_path: Optional[str] = None,
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

    components = components or ["x", "y", "z"]
    mode_data = analyzer.get_mode(frequency, z_layer)

    # Create figure with subplots
    n_components = len(components)
    n_rows = (
        3
        if analyzer.config.show_magnitude
        and analyzer.config.show_phase
        and analyzer.config.show_combined
        else 2
    )

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
                cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_magnitude),
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                origin="lower",
            )
            axes[row, i].set_title(f"|m_{comp}| @ {frequency:.3f} GHz")
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
            axes[row, i].set_title(f"arg(m_{comp}) @ {frequency:.3f} GHz")
            axes[row, i].set_xlabel("x (nm)")
            if i == 0:
                axes[row, i].set_ylabel("y (nm)")
            plt.colorbar(im2, ax=axes[row, i], shrink=0.8)
            row += 1

        # Combined plot (phase with magnitude as alpha)
        if analyzer.config.show_combined:
            # Create combined visualization: magnitude * cos(phase) for real part
            combined_data = magnitude * np.cos(phase)  # Real part

            im3 = axes[row, i].imshow(
                combined_data,
                cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                origin="lower",
            )
            axes[row, i].set_title(
                f"m_{comp} combined (mag×cos(φ)) @ {frequency:.3f} GHz"
            )
            axes[row, i].set_xlabel("x (nm)")
            if i == 0:
                axes[row, i].set_ylabel("y (nm)")
            plt.colorbar(im3, ax=axes[row, i], shrink=0.8)

    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=analyzer.config.dpi, bbox_inches="tight")
        log.info(f"Saved mode plots to {save_path}")

    return fig, axes


def update_single_mode_plot(
    analyzer,
    ax: Any,
    row_idx: int,
    col_idx: int,
    component: Union[str, int],
    z_layer: int,
) -> None:
    """Update single mode plot (used when stopping animation)."""
    try:
        # Get mode data
        mode_data = analyzer.get_mode(analyzer._current_frequency, z_layer)
        comp_data = mode_data.get_component(component)

        # Determine visualization type
        vis_types = []
        if analyzer.config.show_magnitude:
            vis_types.append("magnitude")
        if analyzer.config.show_phase:
            vis_types.append("phase")
        if analyzer.config.show_combined:
            vis_types.append("combined")

        vis_type = vis_types[row_idx]

        # Clear and redraw
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])

        if vis_type == "magnitude":
            magnitude = np.abs(comp_data)
            ax.imshow(
                magnitude,
                cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_magnitude),
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                origin="lower",
            )
            ax.set_title(f"|m_{component}|")

        elif vis_type == "phase":
            phase = np.angle(comp_data)
            ax.imshow(
                phase,
                cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                vmin=-np.pi,
                vmax=np.pi,
                origin="lower",
            )
            ax.set_title(f"arg(m_{component})")

        elif vis_type == "combined":
            magnitude = np.abs(comp_data)
            phase = np.angle(comp_data)
            combined_data = magnitude * np.cos(phase)
            ax.imshow(
                combined_data,
                cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                origin="lower",
            )
            ax.set_title(f"m_{component} (mag×cos(φ))")

    except Exception as e:
        log.error(f"Failed to update single mode plot: {e}")


def add_scale_bar(
    analyzer,
    ax: Any,
    extent: tuple[float, float, float, float]
) -> None:
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
            fontproperties=fm.FontProperties(size=scalebar_fontsize, weight='bold'),
        )
        # Add semi-transparent black background (30% alpha)
        scalebar.patch.set_facecolor((0, 0, 0, 0.3))
        scalebar.patch.set_edgecolor('none')
    except Exception as exc:
        log.debug(f"Could not create scale bar: {exc}")
        return

    ax.add_artist(scalebar)
