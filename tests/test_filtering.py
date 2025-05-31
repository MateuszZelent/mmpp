#!/usr/bin/env python3
"""
Test script to verify parameter filtering for memory addresses
"""

# Add the project root to path
import sys
import os

sys.path.insert(
    0,
    "/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test/ammpp",
)

from mmpp.plotting import MMPPlotter
from mmpp.core import ZarrJobResult


def test_parameter_filtering():
    """Test that memory addresses and unwanted parameters are filtered out"""
    print("=== Testing Parameter Filtering ===\n")

    # Create mock results with memory addresses and other unwanted parameters
    mock_results = [
        ZarrJobResult(
            path="/path1",
            attributes={
                "solver": 3,  # Good parameter
                "f0": 1e9,  # Good parameter
                "maxerr": 1e-6,  # Good parameter
                "Aex": "0x7f8b4c001234",  # Memory address - should be excluded
                "Bext": "0x7f8b4c005678",  # Memory address - should be excluded
                "Ms": "0x7f8b4c009abc",  # Memory address - should be excluded
                "command_line": "/very/long/path/to/executable --with --many --flags --that --make --this --very --long",  # Long string - should be excluded
                "_internal_var": 123,  # Internal variable - should be excluded
                "timestamp_": 1234567890,  # Ends with underscore - should be excluded
            },
        ),
        ZarrJobResult(
            path="/path2",
            attributes={
                "solver": 5,  # Different - good parameter
                "f0": 2e9,  # Different - good parameter
                "maxerr": 1e-7,  # Different - good parameter
                "Aex": "0x7f8b4c001111",  # Different memory address - should be excluded
                "Bext": "0x7f8b4c005555",  # Different memory address - should be excluded
                "Ms": "0x7f8b4c009999",  # Different memory address - should be excluded
                "command_line": "/another/very/long/path/to/executable --with --different --flags --that --also --make --this --very --long",  # Long string - should be excluded
                "_internal_var": 456,  # Internal variable - should be excluded
                "timestamp_": 1234567999,  # Ends with underscore - should be excluded
            },
        ),
    ]

    print("📊 Created mock results with memory addresses and unwanted parameters:")
    for i, result in enumerate(mock_results):
        print(f"\nResult {i+1}:")
        print(f"  Path: {result.path}")
        print(f"  Attributes:")
        for key, val in result.attributes.items():
            print(f"    {key}: {val}")

    # Test the plotter
    plotter = MMPPlotter(mock_results)

    print(f"\n🔍 Testing _get_varying_parameters() with filtering:")
    varying_params = plotter._get_varying_parameters(mock_results)
    print(f"  Detected varying parameters: {varying_params}")

    # Check what should have been filtered out
    expected_good_params = ["solver", "f0", "maxerr"]
    expected_filtered = [
        "Aex",
        "Bext",
        "Ms",
        "command_line",
        "_internal_var",
        "timestamp_",
    ]

    print(f"\n✅ Expected good parameters: {expected_good_params}")
    print(f"❌ Expected filtered parameters: {expected_filtered}")

    # Verify filtering worked
    good_found = [p for p in expected_good_params if p in varying_params]
    bad_found = [p for p in expected_filtered if p in varying_params]

    print(f"\n📈 Good parameters found: {good_found}")
    print(f"🚫 Bad parameters that got through: {bad_found}")

    if len(bad_found) == 0:
        print("✅ SUCCESS: All unwanted parameters were filtered out!")
    else:
        print("❌ FAILURE: Some unwanted parameters got through!")

    print(f"\n🏷️  Testing _format_result_label() for each result:")
    for i, result in enumerate(mock_results):
        label = plotter._format_result_label(result, varying_params)
        print(f"  Result {i+1}: '{label}'")


if __name__ == "__main__":
    test_parameter_filtering()
