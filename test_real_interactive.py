#!/usr/bin/env python3
"""
Test interactive mode specifically
"""

import mmpp
import matplotlib
import matplotlib.pyplot as plt

print("Testing interactive mode...")
print("="*50)

# Set interactive backend
matplotlib.use('TkAgg')  # Force interactive backend
plt.ion()  # Turn on interactive mode

zarr_path = "/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr"
job = mmpp.MMPP(zarr_path)

print(f"Backend: {plt.get_backend()}")
print(f"Interactive mode: {plt.isinteractive()}")

try:
    fig = job[0].fft.modes.interactive_spectrum(
        dset="m_dot",
        force=False,
        z_layer=0,
        tmax=30,  # Limit for testing
        show=False  # We'll manually show
    )
    
    print("✓ Figure created successfully")
    
    # Check if ModeAnalyzer has the connection
    mode_analyzer = job[0].fft.modes
    if hasattr(mode_analyzer, '_click_connection'):
        conn = mode_analyzer._click_connection
        print(f"✓ Click handler found: {conn is not None}")
        if conn:
            print(f"  Connection ID: {conn}")
    else:
        print("✗ No click handler found")
    
    # Check available z layers properly
    try:
        zarr_group = job[0].zarr_group
        dset = zarr_group["m_dot"] 
        print(f"Data shape: {dset.shape}")
        print(f"Available z layers: 0 to {dset.shape[2]-1}")
    except Exception as e:
        print(f"Could not get data shape: {e}")
    
    print("\nFigure should be interactive now!")
    print("Close the window to continue...")
    
    # Show the figure
    plt.show()
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("Done.")