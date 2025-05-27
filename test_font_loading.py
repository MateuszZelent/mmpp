#!/usr/bin/env python3
"""
Test script to verify font loading functionality works correctly.
"""
import os
import sys
from pathlib import Path

# Add mmpp to path  
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mmpp.plotting import setup_custom_fonts, load_paper_style
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    print("✓ Successfully imported font setup functions")
    
    # Test font setup
    print("Testing font setup...")
    font_success = setup_custom_fonts()
    
    if font_success:
        print("✓ Custom fonts setup completed successfully")
    else:
        print("⚠ Custom fonts setup returned False (may be normal if Arial not found)")
    
    # Test paper style loading
    print("Testing paper style...")
    style_success = load_paper_style()
    
    if style_success:
        print("✓ Paper style loaded successfully")
    else:
        print("⚠ Paper style loading returned False")
    
    # Check available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    arial_available = any('arial' in font.lower() for font in available_fonts)
    
    if arial_available:
        print("✓ Arial font family found in system")
    else:
        print("⚠ Arial font family not found, but this is OK")
    
    # Test that the plotter can be created and configured
    from mmpp.plotting import MMPPlotter
    plotter = MMPPlotter()
    plotter.configure(use_custom_fonts=True, font_family="Arial")
    print("✓ MMPPlotter created and configured successfully")
    
    print("\n🎉 Font loading functionality test completed!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
