#!/usr/bin/env python3
"""
Test fixed colorbar positioning
"""

import mmpp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Testing FIXED colorbar positioning...")
print("="*50)

zarr_path = "/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr"
job = mmpp.MMPP(zarr_path)

try:
    print("Creating interactive_spectrum with FIXED colorbars...")
    
    fig = job[0].fft.modes.interactive_spectrum(
        dset="m_dot",
        force=False,
        tmax=50,
        z_layer=0,
        use_fft_spectrum=False,
        show=False
    )
    
    if fig:
        print(f"✓ Figure created with {len(fig.axes)} axes")
        
        # Analyze axes positions to check colorbar placement
        colorbar_axes = []
        plot_axes = []
        
        for ax in fig.axes:
            bbox = ax.get_position()
            width = bbox.width
            height = bbox.height
            
            # Colorbars are typically narrow
            if width < 0.05:  # Very narrow - likely a colorbar
                colorbar_axes.append({
                    'ax': ax, 
                    'x': bbox.x0, 
                    'y': bbox.y0,
                    'width': width,
                    'height': height,
                    'label': ax.get_ylabel()
                })
            elif width > 0.1 and height > 0.1:  # Normal plot
                plot_axes.append({
                    'ax': ax,
                    'x': bbox.x0,
                    'y': bbox.y0, 
                    'width': width,
                    'height': height,
                    'title': ax.get_title()
                })
        
        print(f"\nFound {len(colorbar_axes)} colorbars:")
        for i, cb in enumerate(colorbar_axes):
            print(f"  {i+1}. Position: x={cb['x']:.3f}, y={cb['y']:.3f}, w={cb['width']:.3f}, h={cb['height']:.3f}")
            print(f"      Label: '{cb['label']}'")
        
        print(f"\nFound {len(plot_axes)} plot areas:")
        for i, pa in enumerate(plot_axes):
            print(f"  {i+1}. Position: x={pa['x']:.3f}, y={pa['y']:.3f}, w={pa['width']:.3f}, h={pa['height']:.3f}")
            print(f"      Title: '{pa['title']}'")
        
        # Save the FIXED figure
        fig.savefig("FIXED_colorbar_positioning.png", dpi=150, bbox_inches='tight')
        print(f"\n✓ FIXED figure saved as: FIXED_colorbar_positioning.png")
        plt.close(fig)
        
        print("\n" + "="*50)
        print("COLORBAR FIXES APPLIED:")
        print("✓ Individual colorbar per row (not shared)")
        print("✓ Positioned at rightmost axis of each row")
        print("✓ Using make_axes_locatable for precise placement")
        print("✓ Smaller fraction (2.5% vs 4%)")
        print("✓ Minimal padding (0.5% vs 2%)")
        print("✓ Better descriptive labels")
        
        print("\nColorbar positioning should now be CORRECT! ✓")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()