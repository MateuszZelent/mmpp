"""
Integration patch for MMPP FMRModeAnalyzer to use optimized colorbars.

This file provides the necessary modifications to integrate the optimized
colorbar functionality into the existing MMPP mode visualization system.
"""

from typing import Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import the optimized colorbar functions
try:
    from .optimized_colorbar import (
        create_mmpp_mode_colorbar,
        extract_system_size_from_zarr,
    )

    OPTIMIZED_COLORBAR_AVAILABLE = True
except ImportError:
    OPTIMIZED_COLORBAR_AVAILABLE = False
    print("Warning: optimized_colorbar module not found. Using standard colorbars.")


def create_enhanced_colorbar_for_modes(
    mappable,
    ax,
    zarr_result=None,
    component: str = "z",
    plot_type: str = "magnitude",
    frequency: float = None,
    discrete_levels: int = 10,
    dark_theme: bool = False,
    shrink: float = 0.8,
):
    """
    Create enhanced colorbar for MMPP mode visualization.

    This function can be used as a drop-in replacement for plt.colorbar()
    in the existing MMPP mode visualization code.

    Parameters:
    -----------
    mappable : matplotlib mappable
        The image mappable object
    ax : matplotlib.axes.Axes
        The axes to attach colorbar to
    zarr_result : ZarrJobResult, optional
        MMPP zarr result for system size extraction
    component : str, default='z'
        Magnetization component ('x', 'y', 'z')
    plot_type : str, default='magnitude'
        Type of plot ('magnitude', 'phase', 'combined')
    frequency : float, optional
        Frequency in GHz for label
    discrete_levels : int, default=10
        Number of discrete colorbar levels
    dark_theme : bool, default=False
        Use dark theme optimization
    shrink : float, default=0.8
        Colorbar shrink factor

    Returns:
    --------
    matplotlib.colorbar.Colorbar
        Enhanced colorbar object
    """

    if not OPTIMIZED_COLORBAR_AVAILABLE:
        # Fallback to standard colorbar
        return plt.colorbar(mappable, ax=ax, shrink=shrink)

    # Determine appropriate colormap and label based on plot type
    if plot_type == "magnitude":
        colormap = "thermal"  # Good for magnitude data
        label = f"|m_{component}|"
        units = "arb. units"
    elif plot_type == "phase":
        colormap = "phase"  # HSV-like for phase data
        label = f"arg(m_{component})"
        units = "rad"
        discrete_levels = None  # Phase is continuous
    elif plot_type == "combined":
        colormap = "phase"  # Phase with magnitude as alpha
        label = f"m_{component} (phase+mag)"
        units = "rad"
        discrete_levels = None
    else:
        colormap = "balance"  # Default diverging colormap
        label = f"m_{component}"
        units = "arb. units"

    # Add frequency information to label if provided
    if frequency is not None:
        label += f" @ {frequency:.3f} GHz"

    # Create optimized colorbar
    return create_mmpp_mode_colorbar(
        mappable=mappable,
        zarr_result=zarr_result,
        ax=ax,
        colormap=colormap,
        label=label,
        discrete_levels=discrete_levels,
        dark_theme=dark_theme,
        shrink=shrink,
    )


def patch_modes_plot_method():
    """
    Example of how to patch the existing FMRModeAnalyzer.plot_modes method.

    This shows the minimal changes needed to integrate optimized colorbars.
    """

    patch_code = """
# In mmpp/fft/modes.py, in the FMRModeAnalyzer.plot_modes method:

# BEFORE (around line 765):
plt.colorbar(im1, ax=axes[row, i], shrink=0.8)

# AFTER:
from .colorbar_integration import create_enhanced_colorbar_for_modes
cbar1 = create_enhanced_colorbar_for_modes(
    mappable=im1,
    ax=axes[row, i],
    zarr_result=self.zarr_file,  # Pass zarr file for system info
    component=comp,
    plot_type='magnitude',
    frequency=frequency,
    dark_theme=False,  # Set based on your theme preference
    shrink=0.8
)

# Similarly for phase plots (around line 781):
# BEFORE:
plt.colorbar(im2, ax=axes[row, i], shrink=0.8)

# AFTER:
cbar2 = create_enhanced_colorbar_for_modes(
    mappable=im2,
    ax=axes[row, i],
    zarr_result=self.zarr_file,
    component=comp,
    plot_type='phase',
    frequency=frequency,
    dark_theme=False,
    shrink=0.8
)

# And for combined plots (around line 799):
# BEFORE:
plt.colorbar(im3, ax=axes[row, i], shrink=0.8)

# AFTER:
cbar3 = create_enhanced_colorbar_for_modes(
    mappable=im3,
    ax=axes[row, i],
    zarr_result=self.zarr_file,
    component=comp,
    plot_type='combined',
    frequency=frequency,
    dark_theme=False,
    shrink=0.8
)
"""

    return patch_code


def patch_animation_colorbars():
    """
    Example of how to patch the animation colorbars in save_modes_animation.
    """

    patch_code = """
# In the save_modes_animation method (around line 1307):

# BEFORE:
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(f'Magnetization (arb. units)')

# AFTER:
from .colorbar_integration import create_enhanced_colorbar_for_modes
cbar = create_enhanced_colorbar_for_modes(
    mappable=im,
    ax=ax,
    zarr_result=self.zarr_file,
    component=component,
    plot_type='magnitude' if animation_type == 'temporal' else 'phase',
    discrete_levels=12,  # Good for animations
    dark_theme=False,
    shrink=1.0  # Full size for animations
)
"""

    return patch_code


def demonstrate_system_size_extraction():
    """
    Demonstrate how system size is extracted from zarr metadata.
    """

    example_code = '''
# Example of system size extraction:

def get_system_info_example(zarr_result):
    """Example of extracting system information."""
    
    # Get spatial resolution
    dx = float(zarr_result.z.attrs.get("dx", 1e-9)) * 1e9  # nm
    dy = float(zarr_result.z.attrs.get("dy", 1e-9)) * 1e9  # nm
    
    # Get system dimensions from magnetization data
    if 'm' in zarr_result.z:
        dset = zarr_result.z['m']
        ny, nx = dset.shape[-3], dset.shape[-2]  # spatial dimensions
        width = nx * dx   # total width in nm
        height = ny * dy  # total height in nm
        
        print(f"System size: {width:.1f} × {height:.1f} nm")
        print(f"Resolution: {dx:.3f} × {dy:.3f} nm/pixel")
        print(f"Grid points: {nx} × {ny}")
        
        return (width, height), (dx, dy)
    
    return None, (dx, dy)

# This information is automatically included in the optimized colorbar labels
'''

    return example_code


def recommended_colormap_usage():
    """
    Recommended colormap usage for different types of mode data.
    """

    recommendations = {
        "Mode magnitude": {
            "colormap": "thermal",
            "description": "Sequential colormap perfect for magnitude data",
            "discrete_levels": 10,
            "example": "|m_z| amplitude visualization",
        },
        "Mode phase": {
            "colormap": "phase",
            "description": "Circular colormap (HSV-like) for phase data",
            "discrete_levels": None,
            "example": "arg(m_z) phase visualization",
        },
        "Diverging modes": {
            "colormap": "balance",
            "description": "Diverging colormap for data with positive/negative symmetry",
            "discrete_levels": 12,
            "example": "Real part of complex mode data",
        },
        "Frequency sweep": {
            "colormap": "solar",
            "description": "Sequential colormap for frequency-dependent intensity",
            "discrete_levels": 8,
            "example": "Mode amplitude vs frequency",
        },
        "Spatial derivatives": {
            "colormap": "diff",
            "description": "Diverging colormap for derivatives and differences",
            "discrete_levels": 10,
            "example": "Spatial gradients of mode patterns",
        },
    }

    return recommendations


if __name__ == "__main__":
    print("🔧 MMPP Colorbar Integration Guide")
    print("=" * 50)

    print("\n📋 Mode Plot Patches:")
    print(patch_modes_plot_method())

    print("\n🎬 Animation Patches:")
    print(patch_animation_colorbars())

    print("\n📏 System Size Extraction:")
    print(demonstrate_system_size_extraction())

    print("\n🎨 Recommended Colormap Usage:")
    recs = recommended_colormap_usage()
    for plot_type, info in recs.items():
        print(f"\n{plot_type}:")
        print(f"  Colormap: {info['colormap']}")
        print(f"  Description: {info['description']}")
        print(f"  Discrete levels: {info['discrete_levels']}")
        print(f"  Example: {info['example']}")

    print("\n✅ Integration guide complete!")
    print("\nNext steps:")
    print("1. Copy optimized_colorbar.py to your mmpp directory")
    print("2. Copy this file as colorbar_integration.py")
    print("3. Apply the patches to mmpp/fft/modes.py")
    print("4. Test with your data using colorbar_examples.py")
