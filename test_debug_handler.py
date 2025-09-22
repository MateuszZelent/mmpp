#!/usr/bin/env python3
"""
Test with debug logging
"""

import mmpp
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

print("Testing with DEBUG logging...")
print("="*50)

zarr_path = "/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr"
job = mmpp.MMPP(zarr_path)

try:
    print("Creating interactive spectrum with debug...")
    
    fig = job[0].fft.modes.interactive_spectrum(
        dset="m_dot",
        force=False,
        z_layer=0,
        tmax=15,
        show=False
    )
    
    if fig:
        # Check immediately after creation
        fft_interface = job[0].fft.modes 
        actual_analyzer = fft_interface.mode_analyzer
        
        print(f"Checking _click_connection immediately after creation...")
        if hasattr(actual_analyzer, '_click_connection'):
            conn = actual_analyzer._click_connection
            print(f"  Found _click_connection: {conn}")
            print(f"  Type: {type(conn)}")
        else:
            print("  No _click_connection found!")
            # List ALL attributes to see what exists
            all_attrs = [attr for attr in dir(actual_analyzer) if not attr.startswith('__')]
            print(f"  Available attributes: {all_attrs[:10]}...")  # Show first 10
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    print("Test with debug completed.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()