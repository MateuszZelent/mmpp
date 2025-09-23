#!/usr/bin/env python3
"""
Debug script to check FMRModeAnalyzer structure
"""

import sys
import importlib
import os

# Force import from local directory
sys.path.insert(0, '/home/kkingstoun/git/mmpp')

# Import and inspect
try:
    from mmpp.fft import modes
    importlib.reload(modes)
    
    print("🔍 FMRModeAnalyzer methods:")
    analyzer_methods = [method for method in dir(modes.FMRModeAnalyzer) if not method.startswith('__')]
    for method in sorted(analyzer_methods):
        print(f"  - {method}")
        
    print(f"\n🎯 compute_modes in class: {'compute_modes' in analyzer_methods}")
    
    # Check if we can instantiate the class
    print("\n🏗️  Testing class instantiation...")
    # We need a valid zarr path for this, so let's skip actual instantiation
    print("  Skipping instantiation (requires valid zarr file)")
    
    # Check the method directly from the class
    if hasattr(modes.FMRModeAnalyzer, 'compute_modes'):
        print("✅ compute_modes method found via hasattr")
        method = getattr(modes.FMRModeAnalyzer, 'compute_modes')
        print(f"  Method type: {type(method)}")
        print(f"  Method doc: {method.__doc__[:100] if method.__doc__ else 'No docstring'}...")
    else:
        print("❌ compute_modes method NOT found via hasattr")
        
    # Print module info
    print(f"\n📁 Module file: {modes.__file__}")
    print(f"📄 Module name: {modes.__name__}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()