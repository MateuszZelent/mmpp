#!/usr/bin/env python3
"""
Test click handler specifically
"""

import mmpp
import matplotlib.pyplot as plt

print("Testing click handler connectivity...")
print("="*50)

zarr_path = "/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr"
job = mmpp.MMPP(zarr_path)

try:
    print(f"Backend before: {plt.get_backend()}")
    
    # Create interactive spectrum
    fig = job[0].fft.modes.interactive_spectrum(
        dset="m_dot",
        force=False, 
        z_layer=0,
        tmax=20,
        show=False
    )
    
    print(f"Backend after: {plt.get_backend()}")
    
    if fig:
        # Check click connection more thoroughly
        mode_analyzer = job[0].fft.modes
        
        print(f"Mode analyzer type: {type(mode_analyzer)}")
        print(f"Has _click_connection: {hasattr(mode_analyzer, '_click_connection')}")
        
        if hasattr(mode_analyzer, '_click_connection'):
            conn = mode_analyzer._click_connection
            print(f"Click connection: {conn}")
            print(f"Connection type: {type(conn)}")
            print(f"Is None: {conn is None}")
            
            if conn is not None:
                print("✅ Click handler IS connected!")
            else:
                print("❌ Click handler is None")
        
        # Check figure canvas connections
        if hasattr(fig, 'canvas'):
            callbacks = getattr(fig.canvas.callbacks, 'callbacks', {})
            button_press_callbacks = callbacks.get('button_press_event', {})
            print(f"Button press callbacks count: {len(button_press_callbacks)}")
            
            if button_press_callbacks:
                print("✅ Button press callbacks are registered!")
                for cid, callback in button_press_callbacks.items():
                    print(f"  Callback {cid}: {callback}")
            else:
                print("❌ No button press callbacks found")
        
        plt.close(fig)
        
        print("\n🔍 DIAGNOSIS:")
        if hasattr(mode_analyzer, '_click_connection') and mode_analyzer._click_connection is not None:
            print("✅ Click handler should work correctly")
            print("   Try clicking on the spectrum when show=True")
        else:
            print("❌ Click handler is not properly connected")
            print("   There may be an issue with the connection setup")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("="*50)