#!/usr/bin/env python3
"""
Test script for FFT class __repr__ method.
"""

import os
import sys

# Add the current directory to Python path to import mmpp
sys.path.insert(0, "/home/MateuszZelent/git/mmpp")

try:
    # Mock a simple job result for testing
    class MockJobResult:
        def __init__(self, path):
            self.path = path

    # Import and test FFT class
    from mmpp.fft.core import FFT

    # Create mock job result
    mock_job = MockJobResult("/path/to/test/job")

    # Create FFT instance
    fft_instance = FFT(mock_job)

    # Test the rich repr
    print("Testing FFT __repr__ method:")
    print("=" * 80)
    print(repr(fft_instance))
    print("=" * 80)

except Exception as e:
    print(f"Error testing FFT repr: {e}")
    import traceback

    traceback.print_exc()
