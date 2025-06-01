#!/usr/bin/env python3
"""
Integration test for FMR mode visualization with MMPP
"""

import sys
from pathlib import Path

import numpy as np


def test_integration():
    """Test that FMR modes integrate properly with MMPP"""

    print("🔧 Testing MMPP FMR modes integration...")

    # Test 1: Import all components
    try:
        import mmpp
        from mmpp.fft import FFT
        from mmpp.fft.modes import FFTModeInterface, FMRModeAnalyzer

        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Test 2: Check that FFT has modes property
    try:
        # Create dummy job result for testing
        class DummyJobResult:
            def __init__(self, path="/tmp/test"):
                self.path = path

        # Create FFT instance
        job_result = DummyJobResult()
        fft = FFT(job_result)

        # Check modes property exists
        assert hasattr(fft, "modes"), "FFT should have modes property"
        assert hasattr(fft, "plot_modes"), "FFT should have plot_modes method"
        assert hasattr(fft, "interactive_spectrum"), (
            "FFT should have interactive_spectrum method"
        )

        print("✅ FFT has all required mode methods")
    except Exception as e:
        print(f"❌ FFT integration error: {e}")
        return False

    # Test 3: Check elegant syntax support
    try:
        # Test indexing interface
        indexed = fft[0]
        assert hasattr(indexed, "__getitem__"), "FFT should support indexing"

        print("✅ Elegant syntax interface available")
    except Exception as e:
        print(f"❌ Syntax interface error: {e}")
        return False

    print("🎉 All integration tests passed!")
    return True


if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
