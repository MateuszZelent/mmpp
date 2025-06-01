"""
Example: Integrating Optimized Colorbar with MMPP Mode Visualization

This example demonstrates how to enhance the existing MMPP mode visualization
with the optimized colorbar functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
from optimized_colorbar import (create_mmpp_mode_colorbar,
                                extract_system_size_from_zarr)

from mmpp import MMPP


def enhanced_mode_plot_example():
    """
    Example of enhanced mode visualization with optimized colorbar.
    """

    # Load MMPP data (adjust path as needed)
    # op = MMPP("/path/to/your/zarr/file")
    # result = op[0]

    # For demonstration, let's create synthetic mode data
    x = np.linspace(0, 100, 64)  # 100 nm system
    y = np.linspace(0, 100, 64)
    X, Y = np.meshgrid(x, y)

    # Create a synthetic mode pattern
    mode_data = (
        np.sin(2 * np.pi * X / 25)
        * np.cos(2 * np.pi * Y / 25)
        * np.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / 1000)
    )

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # === Standard colorbar (existing MMPP style) ===
    im1 = ax1.imshow(
        mode_data,
        cmap="RdBu_r",
        extent=[0, 100, 0, 100],
        aspect="equal",
        origin="lower",
    )

    # Standard colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Mode Amplitude (arb. units)")
    ax1.set_title("Standard Colorbar")
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("y (nm)")

    # === Enhanced optimized colorbar ===
    im2 = ax2.imshow(
        mode_data,
        cmap="RdBu_r",  # Will be upgraded to cmocean.balance if available
        extent=[0, 100, 0, 100],
        aspect="equal",
        origin="lower",
    )

    # Create optimized colorbar with system information
    cbar2 = create_mmpp_mode_colorbar(
        mappable=im2,
        ax=ax2,
        colormap="balance",  # Scientific colormap
        label="Mode Amplitude",
        discrete_levels=10,  # Discrete but readable
        dark_theme=False,  # Adjust based on your theme
        system_size=(100, 100),  # 100x100 nm system
        spatial_resolution=(1.56, 1.56),  # Example resolution
        n_ticks=7,
        shrink=0.8,
    )

    ax2.set_title("Optimized Colorbar with System Info")
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("y (nm)")

    plt.tight_layout()
    plt.show()

    return fig, (ax1, ax2)


def integrate_with_mmpp_modes():
    """
    Example of how to integrate with actual MMPP FMRModeAnalyzer.
    """

    # This is how you would modify the existing mode plotting in MMPP
    example_integration_code = """
    # In your FMRModeAnalyzer.plot_modes method, replace:
    
    # OLD CODE:
    # plt.colorbar(im1, ax=axes[row, i], shrink=0.8)
    
    # NEW CODE:
    from optimized_colorbar import create_mmpp_mode_colorbar
    
    cbar = create_mmpp_mode_colorbar(
        mappable=im1,
        zarr_result=self.zarr_result,  # Pass the zarr result for system info
        ax=axes[row, i],
        colormap='balance',  # Use scientific colormap
        label='Mode Amplitude',
        discrete_levels=10,  # More readable discrete levels
        dark_theme=False,    # Set True if using dark theme
        shrink=0.8
    )
    """

    print("Integration Example:")
    print(example_integration_code)


def colormap_comparison():
    """
    Compare different scientific colormaps for mode visualization.
    """

    # Create test data
    x = np.linspace(-50, 50, 64)
    y = np.linspace(-50, 50, 64)
    X, Y = np.meshgrid(x, y)

    # Diverging mode data (positive and negative values)
    mode_data = (
        np.sin(2 * np.pi * X / 25)
        * np.cos(2 * np.pi * Y / 25)
        * np.exp(-(X**2 + Y**2) / 1000)
    )

    # Different colormap options
    colormaps = ["balance", "diff", "curl", "delta", "RdBu_r", "seismic"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, cmap_name in enumerate(colormaps):
        ax = axes[i]

        im = ax.imshow(
            mode_data, extent=[-50, 50, -50, 50], aspect="equal", origin="lower"
        )

        # Use optimized colorbar
        cbar = create_mmpp_mode_colorbar(
            mappable=im,
            ax=ax,
            colormap=cmap_name,
            label="Mode Amplitude",
            discrete_levels=8,
            system_size=(100, 100),
            spatial_resolution=(1.56, 1.56),
        )

        ax.set_title(f"Colormap: {cmap_name}")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")

    plt.suptitle("Scientific Colormap Comparison for Mode Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig, axes


def dark_theme_example():
    """
    Example of dark theme optimization.
    """

    # Set dark theme
    plt.style.use("dark_background")

    # Create test data
    x = np.linspace(0, 100, 64)
    y = np.linspace(0, 100, 64)
    X, Y = np.meshgrid(x, y)
    mode_data = (
        np.sin(2 * np.pi * X / 25)
        * np.cos(2 * np.pi * Y / 25)
        * np.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / 1000)
    )

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="black")

    im = ax.imshow(mode_data, extent=[0, 100, 0, 100], aspect="equal", origin="lower")

    # Dark theme optimized colorbar
    cbar = create_mmpp_mode_colorbar(
        mappable=im,
        ax=ax,
        colormap="balance",
        label="Mode Amplitude",
        discrete_levels=12,
        dark_theme=True,  # Key parameter for dark optimization
        system_size=(100, 100),
        spatial_resolution=(1.56, 1.56),
        fontsize=11,
        label_fontsize=13,
        tick_fontsize=10,
    )

    ax.set_title("Dark Theme Optimized Mode Visualization", color="white", fontsize=14)
    ax.set_xlabel("x (nm)", color="white")
    ax.set_ylabel("y (nm)", color="white")

    plt.tight_layout()
    plt.show()

    # Reset style
    plt.style.use("default")

    return fig, ax


if __name__ == "__main__":
    print("🎨 MMPP Enhanced Colorbar Examples")
    print("=" * 50)

    print("\n1. Basic comparison:")
    enhanced_mode_plot_example()

    print("\n2. Integration guide:")
    integrate_with_mmpp_modes()

    print("\n3. Colormap comparison:")
    colormap_comparison()

    print("\n4. Dark theme example:")
    dark_theme_example()

    print("\n✅ All examples completed!")
    print("\nKey features of the optimized colorbar:")
    print("- Scientific cmocean colormaps with matplotlib fallbacks")
    print("- System size indicators from zarr metadata")
    print("- Discrete levels for better readability")
    print("- Dark theme optimization")
    print("- Enhanced tick formatting")
    print("- Automatic spatial resolution display")
