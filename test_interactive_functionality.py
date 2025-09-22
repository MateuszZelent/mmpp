#!/usr/bin/env python3
"""
Test interactive functionality (not with Agg backend)
"""

import mmpp
import matplotlib.pyplot as plt

print("Testing INTERACTIVE functionality...")
print("="*50)

# Load data
zarr_path = "/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr"
job = mmpp.MMPP(zarr_path)

print(f"Current matplotlib backend: {plt.get_backend()}")

# Test z_layer values
try:
    print("\nTesting z_layer=-1...")
    fig = job[0].fft.modes.interactive_spectrum(
        dset="m_dot",
        force=False,  # Don't force recompute for testing
        z_layer=-1,   # Last layer
        show=False    # Don't show for now
    )
    print("✓ z_layer=-1 works!")
    if fig:
        plt.close(fig)

except Exception as e:
    print(f"✗ z_layer=-1 error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\nTesting z_layer=0 (default)...")
    fig = job[0].fft.modes.interactive_spectrum(
        dset="m_dot", 
        force=False,
        z_layer=0,
        show=False
    )
    print("✓ z_layer=0 works!")
    
    if fig:
        # Check if click handler is connected
        if hasattr(job[0].fft.modes, '_click_connection'):
            conn = job[0].fft.modes._click_connection
            print(f"✓ Click handler connected: {conn is not None}")
        else:
            print("✗ No click handler attribute found")
            
        # Check backend for interactivity
        backend = plt.get_backend()
        interactive_backends = ['Qt5Agg', 'TkAgg', 'GTK3Agg', 'wxAgg']
        is_interactive = backend in interactive_backends
        print(f"Backend '{backend}' supports interaction: {is_interactive}")
        
        if not is_interactive:
            print("⚠️  Current backend doesn't support interaction!")
            print("   Use plt.ion() or change backend for full interactivity")
        
        plt.close(fig)

except Exception as e:
    print(f"✗ z_layer=0 error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Interactive test completed.")

# Check available z layers
try:
    zarr_file = job[0].fft.modes.zarr_file
    dset = zarr_file["m_dot"]
    z_size = dset.shape[2]  # z dimension
    print(f"\nAvailable z layers: 0 to {z_size-1} (total: {z_size})")
    print(f"z_layer=-1 maps to layer {z_size-1}")
except Exception as e:
    print(f"Could not determine z layers: {e}")