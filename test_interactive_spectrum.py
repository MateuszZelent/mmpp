#!/usr/bin/env python3
"""
Test script for interactive_spectrum function with modes
"""

import mmpp
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

print("Testing interactive_spectrum with modes...")
print("="*60)

# Load MMPP job
zarr_path = "/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr"
job = mmpp.MMPP(zarr_path)

print(f"Loaded zarr file: {zarr_path}")
print(f"Available job results: {len(job)} results")

# Test parameters from your command
dset = "m_dot"
force = True
z_layer = -1
use_fft_spectrum = False

print(f"Test parameters:")
print(f"  dset: {dset}")
print(f"  force: {force}")
print(f"  z_layer: {z_layer}")
print(f"  use_fft_spectrum: {use_fft_spectrum}")
print()

try:
    print("Testing interactive_spectrum...")
    
    # Test the function call exactly as you specified
    fig = job[0].fft.modes.interactive_spectrum(
        dset=dset,
        force=force, 
        tmax=100,
        z_layer=z_layer,
        use_fft_spectrum=use_fft_spectrum,
        show=False  # Don't show plot in test
    )
    
    print("✓ interactive_spectrum completed successfully!")
    print(f"  Returned figure type: {type(fig)}")
    
    if fig is not None:
        print(f"  Figure size: {fig.get_size_inches()}")
        print(f"  Number of axes: {len(fig.axes)}")
        
        # Check if figure has expected components
        axes_info = []
        for i, ax in enumerate(fig.axes):
            title = ax.get_title() if hasattr(ax, 'get_title') else "No title"
            axes_info.append(f"    Axis {i}: {title}")
        
        if axes_info:
            print("  Axes information:")
            for info in axes_info[:5]:  # Show first 5 axes
                print(info)
            if len(axes_info) > 5:
                print(f"    ... and {len(axes_info) - 5} more axes")
        
        # Save test figure
        test_output_path = "test_interactive_spectrum_output.png"
        fig.savefig(test_output_path, dpi=100, bbox_inches='tight')
        print(f"  Test figure saved as: {test_output_path}")
        
        # Cleanup
        plt.close(fig)
    
    print()
    print("Testing different parameter combinations...")
    
    # Test with use_fft_spectrum=True
    print("  Testing with use_fft_spectrum=True...")
    fig2 = job[0].fft.modes.interactive_spectrum(
        dset=dset,
        force=False,  # Don't force reload again
        z_layer=z_layer,
        tmax=100,
        use_fft_spectrum=True,  # Use FFT spectrum
        show=False
    )
    print("  ✓ use_fft_spectrum=True completed successfully!")
    if fig2 is not None:
        plt.close(fig2)
    
    # Test with different z_layer
    print("  Testing with z_layer=0...")
    fig3 = job[0].fft.modes.interactive_spectrum(
        dset=dset,
        tmax=100,
        force=False,
        z_layer=0,  # Specific layer
        use_fft_spectrum=False,
        show=False
    )
    print("  ✓ z_layer=0 completed successfully!")
    if fig3 is not None:
        plt.close(fig3)
    
    print()
    print("All tests passed successfully! ✓")
    
except Exception as e:
    print(f"✗ Error in interactive_spectrum test: {e}")
    import traceback
    traceback.print_exc()
    
    # Additional debugging info
    print("\nDebugging information:")
    try:
        print(f"job[0] type: {type(job[0])}")
        print(f"job[0].fft type: {type(job[0].fft)}")
        print(f"job[0].fft.modes type: {type(job[0].fft.modes)}")
        
        # Check if modes interface is available
        modes_interface = job[0].fft.modes
        print(f"modes_interface attributes: {[attr for attr in dir(modes_interface) if not attr.startswith('_')]}")
        
    except Exception as debug_e:
        print(f"Debug info error: {debug_e}")

print("="*60)
print("Test completed.")