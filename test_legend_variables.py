#!/usr/bin/env python3
"""
Test script for legend_variables functionality
"""

# Add the project root to path
import sys
import os
sys.path.insert(0, '/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test/ammpp')

from mmpp.plotting import MMPPlotter
from mmpp.core import ZarrJobResult

def test_legend_variables():
    """Test the new legend_variables parameter"""
    print("=== Testing legend_variables functionality ===\n")
    
    # Create mock results with various parameters
    mock_results = [
        ZarrJobResult(path="/path1", attributes={
            'solver': 3,
            'amp_values': 0.0022,
            'f0': 1e9,
            'maxerr': 1e-6,
            'Nx': 64,
            'end_time': '2024-01-01',  # Should be excluded
            'maxerr_path': '/tmp/path',  # Should be excluded
            'port': 8080,  # Should be excluded
            'Aex': '0x12345678',  # Should be excluded (memory address)
        }),
        ZarrJobResult(path="/path2", attributes={
            'solver': 3, 
            'amp_values': 0.0022,
            'f0': 2e9,  # different
            'maxerr': 1e-7,  # different
            'Nx': 128,  # different
            'end_time': '2024-01-02',  # Should be excluded
            'maxerr_path': '/tmp/path2',  # Should be excluded
            'port': 8081,  # Should be excluded
            'Aex': '0x87654321',  # Should be excluded (memory address)
        }),
        ZarrJobResult(path="/path3", attributes={
            'solver': 3,
            'amp_values': 0.0022, 
            'f0': 1e9,  # same as first
            'maxerr': 1e-8,  # different
            'Nx': 256,  # different
            'end_time': '2024-01-03',  # Should be excluded
            'maxerr_path': '/tmp/path3',  # Should be excluded
            'port': 8082,  # Should be excluded
            'Aex': '0xABCDEF00',  # Should be excluded (memory address)
        })
    ]
    
    plotter = MMPPlotter(mock_results)
    
    # Test 1: Auto-detection with filtering
    print("1. Auto-detection with filtering:")
    varying_params = plotter._get_varying_parameters(mock_results)
    print(f"   Detected varying parameters: {varying_params}")
    
    # Test 2: Manual specification with legend_variables
    print("\n2. Manual specification with legend_variables=['maxerr']:")
    for i, result in enumerate(mock_results):
        label = plotter._format_result_label(result, ['maxerr'])
        print(f"   Result {i+1}: '{label}'")
    
    # Test 3: Multiple variables in legend_variables
    print("\n3. Manual specification with legend_variables=['maxerr', 'f0']:")
    for i, result in enumerate(mock_results):
        label = plotter._format_result_label(result, ['maxerr', 'f0'])
        print(f"   Result {i+1}: '{label}'")
    
    # Test 4: Empty legend_variables (should show "Dataset")
    print("\n4. Empty legend_variables (should show 'Dataset'):")
    for i, result in enumerate(mock_results):
        label = plotter._format_result_label(result, [])
        print(f"   Result {i+1}: '{label}'")
    
    # Test 5: Non-existent parameter in legend_variables
    print("\n5. Non-existent parameter in legend_variables=['nonexistent']:")
    for i, result in enumerate(mock_results):
        label = plotter._format_result_label(result, ['nonexistent'])
        print(f"   Result {i+1}: '{label}'")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_legend_variables()
