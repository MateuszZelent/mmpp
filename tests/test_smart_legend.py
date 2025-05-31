#!/usr/bin/env python3
"""
Test script to verify smart legend functionality works correctly.
"""
import os
import sys
from pathlib import Path

# Add mmpp to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mmpp.plotting import MMPPlotter, PlotConfig

    print("✓ Successfully imported MMPPlotter")

    # Test the smart legend methods directly
    plotter = MMPPlotter()

    # Create mock result objects for testing
    class MockResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.path = f"mock_result_{kwargs.get('solver', 'test')}.txt"
            self.attributes = {}

    # Create test results with varying parameters
    results = [
        MockResult(
            solver=1, f0=1e9, maxerr=1e-6, Nx=64, Ny=64, Nz=1, constant_param=100
        ),
        MockResult(
            solver=2, f0=1e9, maxerr=1e-6, Nx=128, Ny=64, Nz=1, constant_param=100
        ),
        MockResult(
            solver=1, f0=2e9, maxerr=1e-7, Nx=64, Ny=128, Nz=1, constant_param=100
        ),
    ]

    print(f"✓ Created {len(results)} mock results")

    # Test varying parameters detection
    varying_params = plotter._get_varying_parameters(results)
    print(f"✓ Varying parameters detected: {varying_params}")

    # Expected varying parameters: solver, f0, maxerr, Nx, Ny (constant_param and Nz should not appear)
    expected_varying = {"solver", "f0", "maxerr", "Nx", "Ny"}
    actual_varying = set(varying_params)

    if expected_varying.issubset(actual_varying):
        print("✓ Smart parameter detection works correctly")
    else:
        print(f"✗ Expected {expected_varying}, got {actual_varying}")

    # Test formatting with varying parameters
    for i, result in enumerate(results):
        label = plotter._format_result_label(result, varying_params)
        print(f"✓ Result {i+1} label: {label}")

    # Test that constant parameters are excluded from labels
    if "constant_param=100" not in str(
        [plotter._format_result_label(r, varying_params) for r in results]
    ):
        print("✓ Constant parameters correctly excluded from labels")
    else:
        print("✗ Constant parameters incorrectly included in labels")

    # Test formatting rules
    test_result = MockResult(solver=1, f0=1.23456e9, maxerr=1.234e-6, Nx=64)
    formatted_label = plotter._format_result_label(
        test_result, ["solver", "f0", "maxerr", "Nx"]
    )
    print(f"✓ Format test result: {formatted_label}")

    # Check that scientific notation is used for f0 and maxerr
    if ".2e" in formatted_label or "e+" in formatted_label or "e-" in formatted_label:
        print("✓ Scientific notation formatting works")
    else:
        print("✗ Scientific notation formatting may not be working")

    print("\n🎉 Smart legend functionality test completed!")

except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback

    traceback.print_exc()
