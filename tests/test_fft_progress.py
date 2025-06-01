#!/usr/bin/env python3
"""
Test script for FFT progress bar functionality.
"""

import os
import sys
import time

import numpy as np

# Add the mmpp module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mmpp.fft.compute_fft import FFTCompute

    print("✅ Successfully imported FFTCompute")
except ImportError as e:
    print(f"❌ Failed to import FFTCompute: {e}")
    sys.exit(1)


def test_progress_bar():
    """Test FFT computation with progress bar."""
    print("\n🧪 Testing FFT Progress Bar Functionality")
    print("=" * 50)

    # Create test data
    print("📊 Creating test data...")
    time_steps = 2048
    nx, ny = 64, 64
    components = 3

    # Create synthetic magnetization data
    t = np.linspace(0, 1e-9, time_steps)  # 1 ns simulation
    data = np.zeros((time_steps, nx, ny, components))

    # Add some frequency components
    for i in range(nx):
        for j in range(ny):
            # Different frequencies for different spatial locations
            freq1 = 1e9 + (i * j) * 1e6  # GHz range
            freq2 = 2e9 + (i + j) * 5e5

            data[:, i, j, 0] = np.sin(2 * np.pi * freq1 * t) * np.exp(-t / 5e-10)
            data[:, i, j, 1] = np.cos(2 * np.pi * freq2 * t) * np.exp(-t / 3e-10)
            data[:, i, j, 2] = 0.1 * np.sin(2 * np.pi * freq1 * t + np.pi / 4)

    data_size_mb = data.nbytes / 1024**2
    print(f"   Data shape: {data.shape}")
    print(f"   Data size: {data_size_mb:.1f} MB")

    # Initialize FFT compute engine
    fft_compute = FFTCompute()

    # Test different scenarios
    test_cases = [
        {
            "show_progress": True,
            "description": "With progress bar (large data simulation)",
        },
        {"show_progress": False, "description": "Without progress bar"},
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test {i}: {test_case['description']}")
        print("-" * 40)

        start_time = time.time()

        try:
            # Test method 1
            print("Testing FFT Method 1...")
            result1 = fft_compute.calculate_fft_method1(
                data=data,
                dt=t[1] - t[0],
                window="hann",
                filter_type="remove_mean",
                engine="numpy",
                show_progress=test_case["show_progress"],
            )

            # Test method 2
            print("\nTesting FFT Method 2...")
            result2 = fft_compute.calculate_fft_method2(
                data=data,
                dt=t[1] - t[0],
                window="hann",
                filter_type="remove_mean",
                engine="numpy",
                show_progress=test_case["show_progress"],
            )

            elapsed = time.time() - start_time

            print(f"\n✅ Test {i} completed successfully!")
            print(f"   Execution time: {elapsed:.3f}s")
            print(f"   Method 1 spectrum shape: {result1.spectrum.shape}")
            print(f"   Method 2 spectrum shape: {result2.spectrum.shape}")
            print(
                f"   Frequency range: {result1.frequencies[0]:.2e} - {result1.frequencies[-1]:.2e} Hz"
            )

        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            import traceback

            traceback.print_exc()


def check_dependencies():
    """Check if optional dependencies are available."""
    print("\n🔍 Checking Dependencies")
    print("=" * 30)

    # Check tqdm
    try:
        import tqdm

        print("✅ tqdm: Available (progress bars enabled)")
    except ImportError:
        print("⚠️  tqdm: Not available (progress bars will show simple messages)")

    # Check psutil
    try:
        import psutil

        print("✅ psutil: Available (memory monitoring enabled)")
    except ImportError:
        print("⚠️  psutil: Not available (memory monitoring disabled)")

    # Check scipy
    try:
        import scipy

        print("✅ scipy: Available")
    except ImportError:
        print("⚠️  scipy: Not available")

    # Check pyfftw
    try:
        import pyfftw

        print("✅ pyfftw: Available (optimized FFT)")
    except ImportError:
        print("⚠️  pyfftw: Not available (using numpy/scipy FFT)")


if __name__ == "__main__":
    print("🚀 FFT Progress Bar Test Suite")
    print("=" * 40)

    # Check dependencies first
    check_dependencies()

    # Run tests
    test_progress_bar()

    print("\n🎉 All tests completed!")
    print("=" * 40)
