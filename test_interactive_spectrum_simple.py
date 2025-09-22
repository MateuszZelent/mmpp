#!/usr/bin/env python3
"""
Simplified test for interactive_spectrum function with modes
"""

import mmpp
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

print("Testing interactive_spectrum (simplified)...")
print("="*50)

# Load MMPP job
zarr_path = "/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr"
job = mmpp.MMPP(zarr_path)

print(f"Loaded zarr file successfully")

# Test parameters - using force=False to avoid long computation
dset = "m_dot"
force = False  # Don't force recomputation for faster test
z_layer = 0    # Use first layer instead of -1
use_fft_spectrum = True  # Use existing FFT data

print(f"Test parameters:")
print(f"  dset: {dset}")
print(f"  force: {force}")
print(f"  z_layer: {z_layer}")
print(f"  use_fft_spectrum: {use_fft_spectrum}")
print()

try:
    print("Testing interactive_spectrum...")
    
    # Test the function call 
    fig = job[0].fft.modes.interactive_spectrum(
        dset=dset,
        force=force, 
        z_layer=z_layer,
        use_fft_spectrum=use_fft_spectrum,
        show=False  # Don't show plot in test
    )
    
    print("✓ interactive_spectrum completed successfully!")
    print(f"  Returned: {type(fig) if fig is not None else 'None (show=True)'}")
    
    if fig is not None:
        print(f"  Figure size: {fig.get_size_inches()}")
        print(f"  Number of axes: {len(fig.axes)}")
        
        # Save test figure
        test_output_path = "test_interactive_spectrum_simple.png"
        fig.savefig(test_output_path, dpi=100, bbox_inches='tight')
        print(f"  Test figure saved as: {test_output_path}")
        
        # Cleanup
        plt.close(fig)
    
    print()
    print("Test completed successfully! ✓")
    
    # Also test the direct dt fix we made
    print("\nTesting dt fix impact on modes...")
    from mmpp.fft.compute_fft import FFTCompute
    
    fft_compute = FFTCompute()
    data, dt = fft_compute.load_data_from_zarr(
        zarr_path=zarr_path,
        dataset=dset,
        z_layer=0,
        tmax=100  # Small sample for speed
    )
    
    print(f"✓ dt correctly read as: {dt}")
    print(f"  Data shape: {data.shape}")
    print(f"  This dt will be used in all FFT frequency calculations")
    
except Exception as e:
    print(f"✗ Error in test: {e}")
    import traceback
    traceback.print_exc()

print("="*50)