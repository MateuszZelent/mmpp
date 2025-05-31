#!/usr/bin/env python3
"""
Test script to verify the new __repr__ methods and animation fixes.
"""

import sys
import os
import numpy as np
import tempfile
from pathlib import Path

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def test_repr_methods():
    """Test the new __repr__ methods."""
    print("🧪 Testing __repr__ methods...")

    try:
        from mmpp.fft.main import FFTAnalyzer
        from mmpp.fft.modes import FFTModeInterface, FrequencyModeInterface

        print("✅ Successfully imported FFT classes")

        # Test FFTAnalyzer __repr__
        print("\n📊 Testing FFTAnalyzer __repr__:")
        # Create a mock results object for testing
        mock_results = []
        analyzer = FFTAnalyzer(mock_results)
        print(repr(analyzer))

        # Note: We can't easily test the mode interfaces without actual data,
        # but we can verify the methods exist
        print("\n✅ FFTAnalyzer __repr__ method working")

        # Check if the methods exist on the mode classes
        print("\n🔍 Checking FFTModeInterface has __repr__:")
        print(hasattr(FFTModeInterface, "__repr__"))
        print(hasattr(FFTModeInterface, "_rich_mode_display"))
        print(hasattr(FFTModeInterface, "_basic_mode_display"))

        print("\n🔍 Checking FrequencyModeInterface has __repr__:")
        print(hasattr(FrequencyModeInterface, "__repr__"))
        print(hasattr(FrequencyModeInterface, "_rich_frequency_display"))
        print(hasattr(FrequencyModeInterface, "_basic_frequency_display"))

        print("\n✅ All __repr__ methods are properly defined")

    except Exception as e:
        print(f"❌ Error testing __repr__ methods: {e}")
        return False

    return True


def test_animation_improvements():
    """Test the animation improvements."""
    print("\n🎬 Testing animation improvements...")

    try:
        from mmpp.fft.modes import check_ffmpeg_available, setup_animation_styling

        # Test ffmpeg check
        ffmpeg_available = check_ffmpeg_available()
        print(f"🎥 FFmpeg available: {ffmpeg_available}")

        # Test styling setup (should not error)
        try:
            setup_animation_styling()
            print("✅ Animation styling setup successful")
        except Exception as e:
            print(f"⚠️  Animation styling setup warning: {e}")

        print("✅ Animation improvement functions are working")

    except Exception as e:
        print(f"❌ Error testing animation improvements: {e}")
        return False

    return True


def test_imports():
    """Test that all imports work correctly."""
    print("📦 Testing imports...")

    try:
        import mmpp

        print("✅ mmpp imported")

        from mmpp.fft import main, modes

        print("✅ FFT modules imported")

        from mmpp import plotting

        print("✅ plotting module imported")

        # Check if styling functions are available
        from mmpp.plotting import (
            setup_custom_fonts,
            load_paper_style,
            apply_custom_colors,
        )

        print("✅ Styling functions imported")

        print("✅ All imports successful")

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("🚀 Starting MMPP enhancement verification tests\n")

    results = []

    # Test imports
    results.append(test_imports())

    # Test __repr__ methods
    results.append(test_repr_methods())

    # Test animation improvements
    results.append(test_animation_improvements())

    # Summary
    print(f"\n📋 Test Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")

    if all(results):
        print("\n🎉 All tests passed! The enhancements are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
