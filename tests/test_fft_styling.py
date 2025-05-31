#!/usr/bin/env python3
"""
Test script for the updated FFT class __repr__ method with rich panel styling.
"""

import sys
import os

sys.path.insert(0, "/home/kkingstoun/git/mmpp")


# Mock the required components for testing
class MockJobResult:
    def __init__(self):
        self.path = "/path/to/test/simulation.zarr"


class MockPlotter:
    def power_spectrum(self, **kwargs):
        return "mock_figure", "mock_axes"


class MockMMPP:
    def __init__(self):
        self.debug = False
        self._interactive_mode = True


def test_fft_repr():
    """Test the FFT class __repr__ method."""
    print("Testing FFT class __repr__ method...")
    print("=" * 60)

    try:
        # Import the FFT class
        from mmpp.fft.core import FFT

        # Create a mock FFT instance
        fft = FFT(MockJobResult(), MockPlotter())
        fft.mmpp = MockMMPP()
        fft._cache = {"test": "data", "another": "entry"}  # Mock cache

        # Test rich display
        print("Testing rich __repr__ display:")
        print("-" * 40)
        repr_output = repr(fft)
        print(repr_output)
        print("-" * 40)

        # Test basic fallback
        print("\nTesting basic fallback display:")
        print("-" * 40)
        basic_output = fft._basic_fft_display()
        print(basic_output)
        print("-" * 40)

        # Test enhanced fallback
        print("\nTesting enhanced fallback display:")
        print("-" * 40)
        enhanced_output = fft._basic_fft_display_enhanced()
        print(enhanced_output)
        print("-" * 40)

        print("\n✅ All FFT __repr__ tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error testing FFT __repr__: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_import_compatibility():
    """Test that rich imports work properly."""
    print("\nTesting rich import compatibility...")

    try:
        from rich.console import Console
        from rich.text import Text
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.table import Table
        from rich.syntax import Syntax

        print("✅ Rich imports successful")
        return True
    except ImportError as e:
        print(f"❌ Rich import failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing updated FFT class __repr__ with rich panel styling")
    print("=" * 70)

    success = True

    # Test imports
    success &= test_import_compatibility()

    # Test FFT __repr__
    success &= test_fft_repr()

    if success:
        print("\n🎉 All tests passed! FFT class styling updated successfully.")
    else:
        print("\n❌ Some tests failed. Check output above.")
        sys.exit(1)
