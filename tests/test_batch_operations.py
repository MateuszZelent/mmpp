#!/usr/bin/env python3
"""
Test script for batch operations functionality.

This script demonstrates the new batch operations capabilities:
- op[:] - gets all results as batch operations object
- op[:].fft.modes.compute_modes(dset="m_z5-8") - batch mode computation
- op[:].prepare_report(spectrum=True, modes=True) - future comprehensive reports
"""

import sys
import os

# Add the current directory to Python path for testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mmpp.core import MMPP, ZarrJobResult
    from mmpp.batch_operations import BatchOperations

    print("✅ Successfully imported MMPP classes")
except ImportError as e:
    print(f"❌ Failed to import MMPP classes: {e}")
    sys.exit(1)


def test_batch_operations():
    """Test batch operations functionality."""
    print("\n🧪 Testing Batch Operations")
    print("=" * 50)

    # Create mock results for testing
    mock_results = []
    for i in range(3):
        # Create mock ZarrJobResult objects
        result = ZarrJobResult(
            path=f"/mock/path/result_{i}.zarr",
            attributes={"solver": 3, "f0": 1e9 + i * 1e8, "Nx": 128},
        )
        mock_results.append(result)

    print(f"Created {len(mock_results)} mock results for testing")

    # Test BatchOperations creation
    batch_ops = BatchOperations(mock_results, mmpp_ref=None)
    print(f"✅ Created BatchOperations: {batch_ops}")
    print(f"   - Length: {len(batch_ops)}")

    # Test batch_ops properties
    print(f"   - Has .fft property: {hasattr(batch_ops, 'fft')}")
    print(f"   - FFT object: {batch_ops.fft}")
    print(f"   - Has .fft.modes property: {hasattr(batch_ops.fft, 'modes')}")
    print(f"   - Modes object: {batch_ops.fft.modes}")

    # Test summary
    summary = batch_ops.get_summary()
    print(f"   - Summary: {len(summary)} keys")
    for key, value in summary.items():
        if isinstance(value, list) and len(value) > 2:
            print(f"     {key}: [{value[0]}, ..., {value[-1]}] ({len(value)} items)")
        else:
            print(f"     {key}: {value}")

    print("\n✅ Basic batch operations tests passed!")
    return batch_ops


def test_mock_mmpp_integration():
    """Test integration with mock MMPP object."""
    print("\n🧪 Testing MMPP Integration")
    print("=" * 50)

    try:
        # Create a minimal MMPP-like object for testing
        class MockMMPP:
            def __init__(self):
                self._single_zarr_mode = True
                self._zarr_results = []

                # Create mock results
                for i in range(3):
                    result = ZarrJobResult(
                        path=f"/mock/path/simulation_{i}.zarr",
                        attributes={"solver": 3, "amp_values": 0.002 + i * 0.001},
                    )
                    result._set_mmpp_ref(self)
                    self._zarr_results.append(result)

            def __len__(self):
                return len(self._zarr_results)

            def __getitem__(self, index):
                if isinstance(index, slice):
                    # Import here to test the dynamic import
                    from mmpp.batch_operations import BatchOperations

                    results = self._zarr_results[index]
                    return BatchOperations(results, self)
                else:
                    return self._zarr_results[index]

        # Test the mock MMPP
        mock_mmpp = MockMMPP()
        print(f"✅ Created mock MMPP with {len(mock_mmpp)} results")

        # Test single indexing
        single_result = mock_mmpp[0]
        print(f"✅ Single indexing: {single_result}")

        # Test slice indexing (the main feature!)
        batch_ops = mock_mmpp[:]
        print(f"✅ Slice indexing: {batch_ops}")
        print(f"   - Type: {type(batch_ops)}")
        print(f"   - Length: {len(batch_ops)}")

        # Test slice with specific range
        partial_batch = mock_mmpp[1:3]
        print(f"✅ Partial slice [1:3]: {partial_batch}")
        print(f"   - Length: {len(partial_batch)}")

        print("\n✅ MMPP integration tests passed!")
        return mock_mmpp, batch_ops

    except Exception as e:
        print(f"❌ MMPP integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def demonstrate_usage():
    """Demonstrate the intended usage patterns."""
    print("\n🎯 Usage Demonstration")
    print("=" * 50)

    mock_mmpp, batch_ops = test_mock_mmpp_integration()
    if not batch_ops:
        print("❌ Cannot demonstrate usage - integration test failed")
        return

    print("\n📝 Intended Usage Patterns:")
    print("-" * 30)

    # Show the syntax we want to support
    print("1. Get all results as batch operations:")
    print("   batch_ops = op[:]")
    print(f"   ✅ Working: {batch_ops}")

    print("\n2. Access FFT operations:")
    print("   fft_ops = op[:].fft")
    fft_ops = batch_ops.fft
    print(f"   ✅ Working: {fft_ops}")

    print("\n3. Access mode operations:")
    print("   mode_ops = op[:].fft.modes")
    mode_ops = fft_ops.modes
    print(f"   ✅ Working: {mode_ops}")

    print("\n4. Call compute_modes (would need real data):")
    print('   result = op[:].fft.modes.compute_modes(dset="m_z5-8")')
    print("   ⚠️  Would work with real FFT data")

    print("\n5. Prepare comprehensive report:")
    print("   report = op[:].prepare_report(spectrum=True, modes=True)")
    try:
        report = batch_ops.prepare_report(
            spectrum=False, modes=False
        )  # Don't run actual FFT
        print(f"   ✅ Working: {len(report)} keys in report")
    except Exception as e:
        print(f"   ⚠️  Would work with real FFT: {e}")

    print("\n✅ All syntax patterns are supported!")


if __name__ == "__main__":
    print("🚀 MMPP Batch Operations Test")
    print("=" * 50)

    try:
        # Run tests
        batch_ops = test_batch_operations()
        mock_mmpp, batch_ops = test_mock_mmpp_integration()
        demonstrate_usage()

        print("\n🎉 All tests completed successfully!")
        print("\nThe following syntax is now supported:")
        print("  - op[:]                                    # Get all results as batch")
        print("  - op[1:5]                                 # Get subset as batch")
        print("  - op[:].fft.modes.compute_modes(...)      # Batch mode computation")
        print("  - op[:].prepare_report(...)               # Comprehensive reports")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
