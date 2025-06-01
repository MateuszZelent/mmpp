"""
Optimized Colorbar Function for MMPP Mode Visualization

This module provides an enhanced colorbar function that integrates with the existing
MMPP infrastructure, supports dark themes, and includes system size indicators.
"""

from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Import MMPP logging if available
try:
    from mmpp.logging_config import get_mmpp_logger

    log = get_mmpp_logger("mmpp.colorbar")
except ImportError:
    import logging

    log = logging.getLogger("colorbar")

# Try to import cmocean for scientific colormaps
try:
    import cmocean

    CMOCEAN_AVAILABLE = True
    log.info("cmocean available - using scientific colormaps")
except ImportError:
    CMOCEAN_AVAILABLE = False
    log.warning("cmocean not available - falling back to matplotlib colormaps")


def create_optimized_colorbar(
    mappable,
    ax=None,
    cax=None,
    colormap: str = "balance",
    label: str = "Magnetization",
    units: str = "arb. units",
    system_size: Optional[Tuple[float, float]] = None,
    spatial_resolution: Optional[Tuple[float, float]] = None,
    n_ticks: int = 7,
    discrete_levels: Optional[int] = None,
    dark_theme: bool = False,
    show_scale_info: bool = True,
    fontsize: int = 10,
    label_fontsize: int = 12,
    tick_fontsize: int = 9,
    shrink: float = 0.8,
    aspect: int = 20,
    pad: float = 0.05,
    **kwargs,
) -> plt.colorbar:
    """
    Create an optimized colorbar with enhanced visibility and system information.

    Parameters:
    -----------
    mappable : matplotlib mappable
        The image, contour, etc. mappable object
    ax : matplotlib.axes.Axes, optional
        The axes to attach the colorbar to
    cax : matplotlib.axes.Axes, optional
        Specific axes for the colorbar
    colormap : str, default="balance"
        Colormap name (prefers cmocean scientific maps)
    label : str, default="Magnetization"
        Primary label for the colorbar
    units : str, default="arb. units"
        Units for the colorbar
    system_size : tuple, optional
        (width, height) of system in nm for scale indicator
    spatial_resolution : tuple, optional
        (dx, dy) spatial resolution in nm
    n_ticks : int, default=7
        Number of ticks on colorbar
    discrete_levels : int, optional
        Number of discrete levels (makes colorbar discrete)
    dark_theme : bool, default=False
        Optimize for dark themes
    show_scale_info : bool, default=True
        Show system size information
    fontsize : int, default=10
        General font size
    label_fontsize : int, default=12
        Label font size
    tick_fontsize : int, default=9
        Tick font size
    shrink : float, default=0.8
        Shrink factor for colorbar
    aspect : int, default=20
        Aspect ratio of colorbar
    pad : float, default=0.05
        Padding between axes and colorbar

    Returns:
    --------
    matplotlib.colorbar.Colorbar
        The created colorbar object
    """

    # Get the colormap
    cmap = _get_scientific_colormap(colormap)

    # Make discrete if requested
    if discrete_levels is not None:
        cmap = cmap.resampled(discrete_levels)

    # Apply colormap to mappable if it's not already set
    if hasattr(mappable, "set_cmap"):
        mappable.set_cmap(cmap)

    # Create colorbar
    cbar = plt.colorbar(
        mappable, ax=ax, cax=cax, shrink=shrink, aspect=aspect, pad=pad, **kwargs
    )

    # Optimize for dark theme
    if dark_theme:
        _apply_dark_theme_styling(cbar, fontsize, tick_fontsize)

    # Set up ticks for better readability
    _setup_colorbar_ticks(cbar, n_ticks, discrete_levels)

    # Create comprehensive label with system info
    full_label = _create_enhanced_label(
        label, units, system_size, spatial_resolution, show_scale_info
    )

    # Apply styling
    cbar.set_label(full_label, fontsize=label_fontsize, labelpad=15)
    cbar.ax.tick_params(labelsize=tick_fontsize, length=4, width=1)

    # Enhance visibility
    _enhance_colorbar_visibility(cbar, dark_theme)

    log.debug(f"Created optimized colorbar with {colormap} colormap")
    return cbar


def _get_scientific_colormap(colormap_name: str):
    """Get scientific colormap with fallback to matplotlib."""

    # Scientific colormap mappings (cmocean -> matplotlib fallback)
    scientific_maps = {
        "balance": ("balance", "RdBu_r"),  # Perfect for diverging data
        "diff": ("diff", "RdBu"),  # Another diverging option
        "curl": ("curl", "RdYlBu_r"),  # Good for complex/phase data
        "delta": ("delta", "PuOr_r"),  # Deviations from mean
        "tarn": ("tarn", "viridis"),  # Complex data
        "thermal": ("thermal", "inferno"),  # Heat-like data
        "haline": ("haline", "Blues"),  # Sequential blue
        "solar": ("solar", "plasma"),  # Sequential hot
        "ice": ("ice", "Blues_r"),  # Sequential cold
        "gray": ("gray", "gray"),  # Grayscale
        "oxy": ("oxy", "RdYlBu"),  # Oxygen-like
        "deep": ("deep", "Blues"),  # Deep blue
        "dense": ("dense", "viridis"),  # Dense sequential
        "algae": ("algae", "Greens"),  # Green sequential
        "matter": ("matter", "magma"),  # Matter-like
        "turbid": ("turbid", "YlOrBr"),  # Turbidity
        "speed": ("speed", "YlOrRd"),  # Speed visualization
        "amp": ("amp", "YlOrRd"),  # Amplitude
        "tempo": ("tempo", "YlGnBu"),  # Temporal
        "rain": ("rain", "Blues"),  # Precipitation
        "phase": ("phase", "hsv"),  # Phase data
    }

    if CMOCEAN_AVAILABLE and colormap_name in scientific_maps:
        cmocean_name = scientific_maps[colormap_name][0]
        try:
            return getattr(cmocean.cm, cmocean_name)
        except AttributeError:
            log.warning(f"cmocean colormap '{cmocean_name}' not found, using fallback")

    # Fallback to matplotlib
    fallback_name = scientific_maps.get(colormap_name, (None, colormap_name))[1]
    try:
        return plt.get_cmap(fallback_name)
    except ValueError:
        log.warning(f"Colormap '{fallback_name}' not found, using 'viridis'")
        return plt.get_cmap("viridis")


def _apply_dark_theme_styling(cbar, fontsize: int, tick_fontsize: int):
    """Apply dark theme optimizations to colorbar."""

    # Enhance text visibility
    cbar.ax.tick_params(
        colors="white", labelsize=tick_fontsize, length=5, width=1.5, direction="out"
    )

    # Add subtle outline for better definition
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(1.2)

    # Enhance colorbar background contrast
    cbar.ax.patch.set_facecolor("none")


def _setup_colorbar_ticks(cbar, n_ticks: int, discrete_levels: Optional[int]):
    """Set up colorbar ticks for optimal readability."""

    if discrete_levels is not None:
        # For discrete colorbars, show boundaries
        bounds = np.linspace(cbar.vmin, cbar.vmax, discrete_levels + 1)
        cbar.set_ticks(bounds)

        # Create centered tick labels for discrete levels
        centers = (bounds[:-1] + bounds[1:]) / 2
        cbar.set_ticklabels([f"{val:.2g}" for val in centers])
    else:
        # For continuous colorbars, use smart tick placement
        tick_locator = ticker.MaxNLocator(nbins=n_ticks, prune="both", min_n_ticks=3)
        cbar.locator = tick_locator
        cbar.update_ticks()

        # Format tick labels intelligently
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 3))
        cbar.formatter = formatter
        cbar.update_ticks()


def _create_enhanced_label(
    label: str,
    units: str,
    system_size: Optional[Tuple[float, float]],
    spatial_resolution: Optional[Tuple[float, float]],
    show_scale_info: bool,
) -> str:
    """Create enhanced label with system information."""

    # Base label
    full_label = f"{label} ({units})"

    if not show_scale_info:
        return full_label

    # Add system size information
    scale_info = []

    if system_size is not None:
        width, height = system_size
        if width == height:
            scale_info.append(f"System: {width:.0f}×{height:.0f} nm")
        else:
            scale_info.append(f"System: {width:.0f}×{height:.0f} nm")

    if spatial_resolution is not None:
        dx, dy = spatial_resolution
        if dx == dy:
            scale_info.append(f"Δx = {dx:.2f} nm")
        else:
            scale_info.append(f"Δx = {dx:.2f}, Δy = {dy:.2f} nm")

    if scale_info:
        full_label += f"\n{' | '.join(scale_info)}"

    return full_label


def _enhance_colorbar_visibility(cbar, dark_theme: bool):
    """Apply final enhancements for visibility."""

    # Add subtle grid for better value reading
    cbar.ax.grid(
        True,
        alpha=0.3 if dark_theme else 0.2,
        linewidth=0.5,
        linestyle="-",
        color="white" if dark_theme else "gray",
    )

    # Optimize spacing
    cbar.ax.tick_params(pad=4)

    # Ensure proper text color for labels
    label_color = "white" if dark_theme else "black"
    cbar.ax.yaxis.label.set_color(label_color)


def extract_system_size_from_zarr(zarr_result) -> Optional[Dict[str, Any]]:
    """
    Extract system size and spatial resolution from MMPP zarr result.

    Parameters:
    -----------
    zarr_result : ZarrJobResult
        MMPP zarr result object

    Returns:
    --------
    Optional[Dict[str, Any]]
        Dictionary with keys: total_width, total_height, dx, dy, unit
        Returns None if extraction fails
    """

    try:
        # Get spatial resolution
        dx = float(zarr_result.z.attrs.get("dx", 1e-9)) * 1e9  # Convert to nm
        dy = float(zarr_result.z.attrs.get("dy", 1e-9)) * 1e9  # Convert to nm

        # Calculate system size from first magnetization dataset
        for dset_name in ["m", "m_z11", "m_x11", "m_y11"]:
            if dset_name in zarr_result.z:
                dset = zarr_result.z[dset_name]
                if hasattr(dset, "shape") and len(dset.shape) >= 4:
                    # Shape is typically (time, z, y, x, components)
                    ny, nx = dset.shape[-3], dset.shape[-2]
                    total_width = nx * dx
                    total_height = ny * dy

                    return {
                        "total_width": total_width,
                        "total_height": total_height,
                        "dx": dx,
                        "dy": dy,
                        "unit": "nm",
                        "nx": nx,
                        "ny": ny,
                    }
                break

        # If no magnetization data found, return just resolution
        return {
            "total_width": None,
            "total_height": None,
            "dx": dx,
            "dy": dy,
            "unit": "nm",
            "nx": None,
            "ny": None,
        }

    except Exception as e:
        log.warning(f"Could not extract system size: {e}")
        return None


# Integration function for MMPP mode visualization
def create_mmpp_mode_colorbar(
    mappable,
    ax,
    data_type: str = "magnitude",
    system_size: Optional[Dict[str, Any]] = None,
    frequency: Optional[float] = None,
    colormap: str = "balance",
    discrete_levels: int = 10,
    dark_theme: bool = False,
    **kwargs,
) -> plt.colorbar:
    """
    Create optimized colorbar specifically for MMPP mode visualization.

    Parameters:
    -----------
    mappable : matplotlib mappable
        The mode visualization mappable
    ax : matplotlib.axes.Axes
        Axes for colorbar attachment
    data_type : str, default='magnitude'
        Type of data ('magnitude', 'phase', 'combined')
    system_size : Dict[str, Any], optional
        System size information dictionary
    frequency : float, optional
        Current frequency in GHz
    colormap : str, default="balance"
        Scientific colormap name
    discrete_levels : int, default=10
        Number of discrete levels for better readability
    dark_theme : bool, default=False
        Optimize for dark themes

    Returns:
    --------
    matplotlib.colorbar.Colorbar
        Optimized colorbar for mode visualization
    """

    # Generate appropriate label based on data type
    if data_type == "magnitude":
        label = "Mode Amplitude"
        units = "arb. units"
    elif data_type == "phase":
        label = "Phase"
        units = "rad"
    elif data_type == "combined":
        label = "Combined Mode"
        units = "arb. units"
    else:
        label = "Mode Amplitude"
        units = "arb. units"

    # Prepare system size for main function
    system_size_tuple = None
    spatial_resolution = None
    if system_size and system_size["total_width"] is not None:
        system_size_tuple = (system_size["total_width"], system_size["total_height"])
        spatial_resolution = (system_size["dx"], system_size["dy"])

    return create_optimized_colorbar(
        mappable=mappable,
        ax=ax,
        colormap=colormap,
        label=label,
        units=units,
        system_size=system_size_tuple,
        spatial_resolution=spatial_resolution,
        discrete_levels=discrete_levels,
        dark_theme=dark_theme,
        **kwargs,
    )
