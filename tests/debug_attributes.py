#!/usr/bin/env python3
"""
Debug script to test smart legend functionality with real data
"""

# Add the project root to path
import sys

sys.path.insert(
    0,
    "/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test/ammpp",
)

import mmpp


def debug_smart_legend():
    """Debug the smart legend functionality"""
    print("=== Smart Legend Debug with Real Data ===\n")

    # Create MMPP instance
    base_path = (
        "/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test"
    )
    mp = mmpp.MMPP(base_path)

    print(f"Database loaded: {mp.dataframe is not None}")
    if mp.dataframe is not None:
        print(f"Total results: {len(mp.dataframe)}")

    # Find some results
    print("\n=== Finding Results ===")
    results = mp.find(solver="mgcg", limit=3)
    print(f"Found {len(results)} results")

    if len(results) > 0:
        print("\n=== Examining First Result ===")
        result = results[0]
        print(f"Result type: {type(result)}")
        print(f"Has attributes: {hasattr(result, 'attributes')}")

        if hasattr(result, "attributes"):
            print(f"Attributes keys count: {len(result.attributes)}")
            # Show first 10 keys
            keys = list(result.attributes.keys())[:10]
            print(f"First 10 keys: {keys}")

            # Show some sample attributes with values
            print("Sample attributes:")
            for key in keys[:5]:
                val = result.attributes[key]
                print(f"  {key}: {val} ({type(val)})")

        print("\n=== Testing _get_varying_parameters Debug ===")
        # Create plotter to test _get_varying_parameters
        plotter = results.mpl

        # Add some debug prints to see what's happening
        print("Manual analysis of parameter collection:")
        if len(results) >= 2:
            result1 = results[0]
            result2 = results[1]

            print(f"Result1 has {len(result1.attributes)} attributes")
            print(f"Result2 has {len(result2.attributes)} attributes")

            # Check what types of values we have
            simple_types_count = 0
            for _attr_name, value in result1.attributes.items():
                if isinstance(value, (int, float, str, bool)):
                    simple_types_count += 1

            print(f"Result1 has {simple_types_count} simple-type attributes")

            # Check for actual differences
            different_attrs = []
            common_keys = set(result1.attributes.keys()) & set(
                result2.attributes.keys()
            )
            for key in common_keys:
                val1 = result1.attributes[key]
                val2 = result2.attributes[key]
                if val1 != val2 and isinstance(val1, (int, float, str, bool)):
                    different_attrs.append(key)

            print(f"Found {len(different_attrs)} attributes that differ:")
            for attr in different_attrs[:5]:  # Show first 5
                val1 = result1.attributes[attr]
                val2 = result2.attributes[attr]
                print(f"  {attr}: {val1} vs {val2}")

        # Now test the function
        varying_params = plotter._get_varying_parameters(
            results[:2] if len(results) > 1 else results
        )
        print(f"_get_varying_parameters returned: {varying_params}")

        if len(varying_params) == 0:
            print("WARNING: No varying parameters found!")

        print("\n=== Testing Label Formatting ===")
        # Test label formatting
        label = plotter._format_result_label(result, varying_params)
        print(f"Generated label: '{label}'")


if __name__ == "__main__":
    debug_smart_legend()
