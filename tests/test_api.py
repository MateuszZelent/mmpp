#!/usr/bin/env python3
"""
Test script for the new legend_variables API
"""

# Add the project root to path
import sys

sys.path.insert(
    0,
    "/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test/ammpp",
)

from mmpp.core import ZarrJobResult
from mmpp.plotting import MMPPlotter


def test_api():
    """Test the complete API with legend_variables"""
    print("=== Testing Complete API ===\n")

    # Create mock results
    mock_results = [
        ZarrJobResult(
            path="/path1",
            attributes={
                "solver": 3,
                "f0": 1e9,
                "maxerr": 1e-6,
                "Nx": 64,
                "end_time": "2024-01-01",  # Should be excluded
                "Aex": "0x12345678",  # Should be excluded
            },
        ),
        ZarrJobResult(
            path="/path2",
            attributes={
                "solver": 3,
                "f0": 2e9,  # different
                "maxerr": 1e-7,  # different
                "Nx": 128,  # different
                "end_time": "2024-01-02",  # Should be excluded
                "Aex": "0x87654321",  # Should be excluded
            },
        ),
    ]

    # Set up mock references (normally done by MMPP)
    for result in mock_results:
        result._set_mmpp_ref(None)  # Normally would be MMPP instance

    print("1. Testing individual result .matplotlib property:")
    try:
        # This would work if we had a real MMPP instance
        # plotter = mock_results[0].matplotlib
        print("   Individual result .matplotlib: Would work with real MMPP instance")
    except Exception as e:
        print(f"   Expected error (no MMPP ref): {e}")

    print("\n2. Testing direct MMPPlotter with legend_variables:")
    plotter = MMPPlotter(mock_results)

    # Test auto-detection
    varying_params = plotter._get_varying_parameters(mock_results)
    print(f"   Auto-detected parameters: {varying_params}")

    # Test manual specification
    print("\n3. Testing label generation with different legend_variables:")

    test_cases = [
        None,  # Auto-detect
        ["maxerr"],  # Single parameter
        ["maxerr", "f0"],  # Multiple parameters
        ["nonexistent"],  # Non-existent parameter
        [],  # Empty list
    ]

    for legend_vars in test_cases:
        print(f"\n   legend_variables={legend_vars}:")

        if legend_vars is None:
            params_to_use = varying_params
        else:
            params_to_use = legend_vars

        for i, result in enumerate(mock_results):
            label = plotter._format_result_label(result, params_to_use)
            print(f"     Result {i + 1}: '{label}'")

    print("\n=== API Test Complete ===")
    print("\nCorrect usage:")
    print(
        "  results.plot(x_series='t', y_series='m_z11', average=(1,2,3), comp='z', paper_ready=True, legend_variables=['maxerr'])"
    )
    print(
        "  results.mpl.plot(x_series='t', y_series='m_z11', legend_variables=['maxerr', 'f0'])"
    )


if __name__ == "__main__":
    test_api()
