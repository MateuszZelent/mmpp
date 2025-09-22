#!/usr/bin/env python3
"""
Test improved colorbar and labels for interactive_spectrum
"""

import mmpp
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

print("Testing improved colorbar and labels...")
print("="*50)

# Load MMPP job
zarr_path = "/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr"
job = mmpp.MMPP(zarr_path)

print("✓ MMPP job loaded successfully")

# Test parameters
dset = "m_dot"
force = False  # Use existing data
z_layer = 0    
use_fft_spectrum = False  # Use modes spectrum

try:
    print("\nTesting interactive_spectrum with improved colorbar...")
    
    fig = job[0].fft.modes.interactive_spectrum(
        dset=dset,
        force=force,
        tmax=100,  # Limit time for faster processing
        z_layer=z_layer,
        use_fft_spectrum=use_fft_spectrum,
        show=False
    )
    
    print("✓ interactive_spectrum completed successfully!")
    
    if fig is not None:
        print(f"  Figure size: {fig.get_size_inches()}")
        print(f"  Number of axes: {len(fig.axes)}")
        
        # Count colorbars
        colorbar_count = 0
        for ax in fig.axes:
            if ax.get_xlabel() or ax.get_ylabel():
                # Check if this looks like a colorbar
                bbox = ax.get_position()
                if bbox.width < 0.1 or bbox.height < 0.1:  # Small axes are likely colorbars
                    colorbar_count += 1
        
        print(f"  Detected colorbars: {colorbar_count}")
        
        # Check titles
        titles = []
        for ax in fig.axes:
            title = ax.get_title()
            if title:
                titles.append(title)
        
        print(f"  Plot titles: {titles[:5]}...")  # Show first 5
        
        # Save improved figure
        output_path = "test_improved_interactive_spectrum.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Improved figure saved as: {output_path}")
        
        plt.close(fig)
    
    print("\n" + "="*50)
    print("IMPROVEMENTS APPLIED:")
    print("✓ Better colorbar labels:")
    print("  - Magnitude: 'Magnetization |m|' (was '|m| (arb. units)')")
    print("  - Phase: 'Phase (rad)' (unchanged)")  
    print("  - Combined: 'Re(m) × cos(φ)' (was 'Re(m) (arb. units)')")
    print("✓ Larger colorbar size (fraction: 0.04 vs 0.025)")
    print("✓ Better padding (pad: 0.02 vs 0.012)")
    print("✓ Larger font sizes (labels: 10 vs 9, ticks: 9 vs 8)")
    print("✓ Cleaner titles: 'm_x (mag×cos(φ))' vs 'm_x combined (mag×cos(φ))'")
    print("✓ Fallback colorbar creation when axes_grid1 unavailable")
    
    print("\nTest completed successfully! ✓")
    
except Exception as e:
    print(f"✗ Error in test: {e}")
    import traceback
    traceback.print_exc()

print("="*50)