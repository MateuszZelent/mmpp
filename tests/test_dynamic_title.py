#!/usr/bin/env python3
"""
Test script for dynamic title functionality
"""

# Add the project root to path
import sys
import os
sys.path.insert(0, '/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test/ammpp')

from mmpp.plotting import MMPPlotter
from mmpp.core import ZarrJobResult

def test_dynamic_title():
    """Test the new dynamic title functionality"""
    print("=== Testing Dynamic Title Functionality ===\n")
    
    # Create mock results with various parameters
    mock_results = [
        ZarrJobResult(path="/path1", attributes={
            'solver': 3,
            'amp_values': 0.0022,
            'amp': 0.0022,  # Alias for amp_values
            'f0': 1e9,
            'maxerr': 1e-6,
            'Nx': 64,
        }),
        ZarrJobResult(path="/path2", attributes={
            'solver': 5, 
            'amp_values': 0.0033,
            'amp': 0.0033,
            'f0': 2e9,
            'maxerr': 1e-7,
            'Nx': 128,
        }),
    ]
    
    plotter = MMPPlotter(mock_results)
    
    # Test 1: Single parameter in title
    print("1. Single parameter title=['amp']:")
    title = plotter._format_dynamic_title(['amp'], mock_results)
    print(f"   Generated title: '{title}'")
    
    # Test 2: Multiple parameters in title
    print("\n2. Multiple parameters title=['amp', 'f0']:")
    title = plotter._format_dynamic_title(['amp', 'f0'], mock_results)
    print(f"   Generated title: '{title}'")
    
    # Test 3: Scientific notation parameters
    print("\n3. Scientific notation title=['maxerr', 'f0']:")
    title = plotter._format_dynamic_title(['maxerr', 'f0'], mock_results)
    print(f"   Generated title: '{title}'")
    
    # Test 4: Integer parameters
    print("\n4. Integer parameters title=['solver', 'Nx']:")
    title = plotter._format_dynamic_title(['solver', 'Nx'], mock_results)
    print(f"   Generated title: '{title}'")
    
    # Test 5: Non-existent parameter
    print("\n5. Non-existent parameter title=['nonexistent']:")
    title = plotter._format_dynamic_title(['nonexistent'], mock_results)
    print(f"   Generated title: '{title}' (should be empty)")
    
    # Test 6: Mixed existing and non-existing parameters
    print("\n6. Mixed parameters title=['amp', 'nonexistent', 'solver']:")
    title = plotter._format_dynamic_title(['amp', 'nonexistent', 'solver'], mock_results)
    print(f"   Generated title: '{title}'")
    
    # Test 7: Empty title parameters
    print("\n7. Empty title parameters title=[]:")
    title = plotter._format_dynamic_title([], mock_results)
    print(f"   Generated title: '{title}' (should be empty)")
    
    print("\n=== Expected API Usage Examples ===")
    print("results.matplotlib.plot(x_series='t', y_series='m_z11', title=['amp'])")
    print("# Should generate title: 'Amp = 2.20e-03'")
    print()
    print("results.matplotlib.plot(x_series='t', y_series='m_z11', title=['amp', 'f0'])")
    print("# Should generate title: 'Amp = 2.20e-03, F0 = 1.00e+09'")
    print()
    print("results.matplotlib.plot(x_series='t', y_series='m_z11', title='Custom Title')")
    print("# Should use static title: 'Custom Title'")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_dynamic_title()
