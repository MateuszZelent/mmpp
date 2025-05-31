#!/usr/bin/env python3
"""
Debug script to understand what's happening with the legend.
"""
import sys
from pathlib import Path

# Add mmpp to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mmpp.plotting import MMPPlotter
    from mmpp.core import ZarrJobResult

    print("🐛 Debugging Smart Legend Issue")
    print("=" * 50)

    # Create some mock ZarrJobResult objects like the real system would
    mock_results = [
        ZarrJobResult(
            path="/path1",
            attributes={
                "solver": 3,
                "amp_values": 0.0022,
                "f0": 1e9,
                "maxerr": 1e-6,
                "Nx": 64,
            },
        ),
        ZarrJobResult(
            path="/path2",
            attributes={
                "solver": 3,
                "amp_values": 0.0022,
                "f0": 2e9,  # different
                "maxerr": 1e-7,  # different
                "Nx": 128,  # different
            },
        ),
        ZarrJobResult(
            path="/path3",
            attributes={
                "solver": 3,
                "amp_values": 0.0022,
                "f0": 1e9,  # same as first
                "maxerr": 1e-6,  # same as first
                "Nx": 256,  # different
            },
        ),
    ]

    print("📊 Created mock results that mimic real ZarrJobResult structure:")
    for i, result in enumerate(mock_results):
        print(f"  Result {i+1}:")
        print(f"    Path: {result.path}")
        print(f"    Attributes: {result.attributes}")
        print(
            f"    Can access solver? {hasattr(result, 'solver')} -> {getattr(result, 'solver', 'N/A')}"
        )
        print(
            f"    Can access f0? {hasattr(result, 'f0')} -> {getattr(result, 'f0', 'N/A')}"
        )

    # Test the plotter
    plotter = MMPPlotter(mock_results)

    print(f"\n🔍 Testing _get_varying_parameters():")
    varying_params = plotter._get_varying_parameters(mock_results)
    print(f"  Detected varying parameters: {varying_params}")

    print(f"\n🏷️  Testing _format_result_label() for each result:")
    for i, result in enumerate(mock_results):
        label = plotter._format_result_label(result, varying_params)
        print(f"  Result {i+1}: '{label}'")

    # Test without varying params (old behavior)
    print(f"\n🏷️  Testing _format_result_label() without varying_params (old behavior):")
    for i, result in enumerate(mock_results):
        label = plotter._format_result_label(result, None)
        print(f"  Result {i+1}: '{label}'")

    # Check if the problem is in accessing attributes
    print(f"\n🔬 Direct attribute access test:")
    for i, result in enumerate(mock_results):
        print(f"  Result {i+1}:")
        print(
            f"    result.attributes['solver']: {result.attributes.get('solver', 'MISSING')}"
        )
        print(f"    result.solver: {getattr(result, 'solver', 'MISSING')}")
        print(f"    hasattr(result, 'solver'): {hasattr(result, 'solver')}")

        # Test the get_value function from _format_result_label
        def test_get_value(result, param):
            try:
                if hasattr(result, "attributes") and param in result.attributes:
                    return result.attributes[param]
                elif hasattr(result, param):
                    return getattr(result, param)
                else:
                    return None
            except Exception:
                return None

        print(
            f"    test_get_value(result, 'solver'): {test_get_value(result, 'solver')}"
        )
        print(f"    test_get_value(result, 'f0'): {test_get_value(result, 'f0')}")

    print(f"\n✅ Debug test completed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
