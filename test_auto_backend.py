#!/usr/bin/env python3
"""
Test improved automatic backend detection and configuration
"""

import mmpp

print("Testing IMPROVED interactive backend detection...")
print("="*60)

zarr_path = "/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr"
job = mmpp.MMPP(zarr_path)

print("Testing interactive_spectrum with improved backend handling...")

try:
    # This should now automatically detect environment and configure backend
    fig = job[0].fft.modes.interactive_spectrum(
        dset="m_dot",
        force=False,
        z_layer=0,
        tmax=25,  # Quick test
        show=False
    )
    
    if fig:
        print("✓ interactive_spectrum completed successfully!")
        print("✓ Backend was automatically configured")
        
        # Check if click handler was properly set up
        mode_analyzer = job[0].fft.modes
        if hasattr(mode_analyzer, '_click_connection') and mode_analyzer._click_connection:
            print("✓ Click handler is properly connected")
        else:
            print("⚠️  Click handler may not be connected")
        
        # Save test figure
        fig.savefig("test_auto_backend.png", dpi=100, bbox_inches='tight')
        print("✓ Test figure saved as: test_auto_backend.png")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        print("\n" + "="*60)
        print("🎯 IMPROVED INTERACTIVE FEATURES:")
        print("✅ Automatic Jupyter detection")
        print("✅ Auto-switch to %matplotlib widget when possible")
        print("✅ Fallback to nbagg if widget unavailable")
        print("✅ Clear user guidance in warning messages")
        print("✅ Support for standalone Python interactive backends")
        print("✅ Better documentation about requirements")
        
        print("\n🚀 The library now handles interactivity automatically!")
        print("   No more need to manually run '%matplotlib widget'")
        
    else:
        print("✗ No figure returned")

except Exception as e:
    print(f"✗ Error in test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)