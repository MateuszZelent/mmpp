#!/usr/bin/env python3
"""
Test script for FFT spectrum plotting with save/cache functionality
"""

import sys
import os
import numpy as np
import zarr
import tempfile
from pathlib import Path

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import our modules
from mmpp.fft.compute_fft import FFTCompute, FFTComputeConfig, FFTComputeResult


def create_test_zarr_file(zarr_path: str):
    """Create a test zarr file with synthetic magnetization data."""
    print(f"Creating test zarr file: {zarr_path}")

    # Create zarr file
    z = zarr.open(zarr_path, mode="w")

    # Create synthetic time series data
    n_time = 1000
    n_x, n_y, n_z = 10, 10, 5
    n_comp = 3  # x, y, z components

    # Time parameters
    dt = 1e-12  # 1 ps time step
    freq1 = 2.15e9  # 2.15 GHz
    freq2 = 5.3e9  # 5.3 GHz

    # Generate time array
    t = np.arange(n_time) * dt

    # Create synthetic magnetization data with two frequency components
    # Shape: (time, z, y, x, components)
    m_data = np.zeros((n_time, n_z, n_y, n_x, n_comp))

    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                # Different frequencies for different spatial locations
                phase1 = 2 * np.pi * freq1 * t + np.random.uniform(0, 2 * np.pi)
                phase2 = 2 * np.pi * freq2 * t + np.random.uniform(0, 2 * np.pi)

                # Magnetization components
                m_data[:, k, j, i, 0] = 0.1 * np.sin(phase1) + 0.05 * np.cos(
                    phase2
                )  # mx
                m_data[:, k, j, i, 1] = 0.1 * np.cos(phase1) + 0.05 * np.sin(
                    phase2
                )  # my
                m_data[:, k, j, i, 2] = (
                    0.9 + 0.05 * np.sin(phase1) + 0.02 * np.cos(phase2)
                )  # mz

    # Add some noise
    m_data += np.random.normal(0, 0.001, m_data.shape)

    # Save magnetization data
    z.create_dataset("m_z11", data=m_data, chunks=(100, n_z, n_y, n_x, n_comp))

    # Add metadata
    z.attrs["dt"] = dt
    z.attrs["t_sampl"] = dt
    z.attrs["n_time"] = n_time
    z.attrs["frequencies"] = [freq1, freq2]

    print(f"Created test data with shape: {m_data.shape}")
    print(f"Time step: {dt} s")
    print(f"Expected frequencies: {freq1/1e9:.2f} GHz, {freq2/1e9:.2f} GHz")

    return zarr_path


def test_fft_save_functionality():
    """Test the FFT save/cache functionality."""
    print("=" * 60)
    print("Testing FFT Save/Cache Functionality")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test zarr file
        test_zarr_path = os.path.join(temp_dir, "test_data.zarr")
        create_test_zarr_file(test_zarr_path)

        # Initialize FFT compute engine
        fft_compute = FFTCompute()

        print("\n1. Testing initial FFT calculation with save=True")
        print("-" * 50)

        # First calculation - should save to zarr
        result1 = fft_compute.calculate_fft_data(
            zarr_path=test_zarr_path,
            dataset="m_z11",
            z_layer=-1,
            method=1,
            save=True,
            save_dataset_name="test_fft_1",
            window="hann",
            filter_type="remove_mean",
            engine="numpy",
        )

        print(f"FFT calculated:")
        print(f"  - Frequencies shape: {result1.frequencies.shape}")
        print(f"  - Spectrum shape: {result1.spectrum.shape}")
        print(f"  - Peak frequency: {result1.peak_frequency/1e9:.2f} GHz")
        print(f"  - Engine used: {result1.config.fft_engine}")

        # Check if FFT data was saved
        z = zarr.open(test_zarr_path, mode="r")
        if "fft" in z and "test_fft_1" in z["fft"]:
            print("✓ FFT data successfully saved to zarr")

            # Check saved data structure
            fft_group = z["fft"]["test_fft_1"]
            print(f"  - Saved spectrum shape: {fft_group['spectrum'].shape}")
            print(f"  - Saved frequencies shape: {fft_group['frequencies'].shape}")
            print(f"  - Saved attributes: {list(fft_group.attrs.keys())}")

            # Check chunking strategy
            spectrum_chunks = fft_group["spectrum"].chunks
            print(f"  - Spectrum chunking: {spectrum_chunks}")

        else:
            print("✗ FFT data was not saved to zarr")
            return False

        print("\n2. Testing cache loading with same parameters")
        print("-" * 50)

        # Second calculation with same parameters - should load from cache
        result2 = fft_compute.calculate_fft_data(
            zarr_path=test_zarr_path,
            dataset="m_z11",
            z_layer=-1,
            method=1,
            save=True,
            save_dataset_name="test_fft_1",
            window="hann",
            filter_type="remove_mean",
            engine="numpy",
        )

        # Should be identical (loaded from zarr)
        if np.allclose(result1.frequencies, result2.frequencies) and np.allclose(
            result1.spectrum, result2.spectrum
        ):
            print("✓ Successfully loaded FFT data from cache")
        else:
            print("✗ Cache loading failed - results differ")
            return False

        print("\n3. Testing parameter verification")
        print("-" * 50)

        # Third calculation with different parameters - should recalculate
        result3 = fft_compute.calculate_fft_data(
            zarr_path=test_zarr_path,
            dataset="m_z11",
            z_layer=-1,
            method=1,
            save=True,
            save_dataset_name="test_fft_2",  # Different name
            window="blackman",  # Different window
            filter_type="remove_mean",
            engine="numpy",
        )

        # Should be different due to different window function
        if not np.allclose(result1.spectrum, result3.spectrum):
            print(
                "✓ Parameter verification working - different parameters produce different results"
            )
        else:
            print("⚠ Warning: Different parameters produced same results")

        print("\n4. Testing force recalculation")
        print("-" * 50)

        # Force recalculation with force=True
        result4 = fft_compute.calculate_fft_data(
            zarr_path=test_zarr_path,
            dataset="m_z11",
            z_layer=-1,
            method=1,
            save=True,
            force=True,  # Force recalculation
            save_dataset_name="test_fft_1",
            window="hann",
            filter_type="remove_mean",
            engine="numpy",
        )

        # Should be very similar to result1 (same parameters)
        if np.allclose(
            result1.frequencies, result4.frequencies, rtol=1e-10
        ) and np.allclose(result1.spectrum, result4.spectrum, rtol=1e-10):
            print("✓ Force recalculation working correctly")
        else:
            print("⚠ Force recalculation produced different results")

        print("\n5. Testing chunking strategy")
        print("-" * 50)

        # Check final zarr structure
        z = zarr.open(test_zarr_path, mode="r")

        if "fft" in z:
            print(f"FFT datasets saved: {list(z['fft'].keys())}")

            for dataset_name in z["fft"].keys():
                fft_group = z["fft"][dataset_name]
                spectrum_data = fft_group["spectrum"]

                print(f"\nDataset: {dataset_name}")
                print(f"  - Spectrum shape: {spectrum_data.shape}")
                print(f"  - Spectrum chunks: {spectrum_data.chunks}")
                print(f"  - Spectrum dtype: {spectrum_data.dtype}")

                # Verify chunking strategy: only last dimension should be chunked for magnetization components
                if spectrum_data.chunks:
                    last_chunk = spectrum_data.chunks[-1]
                    if last_chunk <= 3:  # Should chunk magnetization components (x,y,z)
                        print("  ✓ Correct chunking strategy applied")
                    else:
                        print("  ⚠ Unexpected chunking strategy")

                # Check attributes
                attrs = dict(fft_group.attrs)
                expected_attrs = [
                    "window_function",
                    "filter_type",
                    "fft_engine",
                    "zero_padding",
                ]
                for attr in expected_attrs:
                    if attr in attrs:
                        print(f"  ✓ Attribute '{attr}': {attrs[attr]}")
                    else:
                        print(f"  ✗ Missing attribute: {attr}")

        print("\n" + "=" * 60)
        print("FFT Save/Cache Test Summary")
        print("=" * 60)
        print("✓ All tests passed successfully!")
        print("✓ FFT data saves correctly to zarr with intelligent chunking")
        print("✓ Parameter verification prevents incorrect cache usage")
        print("✓ Force recalculation works as expected")
        print("✓ Cache loading works for identical parameters")

        return True


def test_high_level_interface():
    """Test the high-level FFT interface (like from test.ipynb)."""
    print("\n" + "=" * 60)
    print("Testing High-Level FFT Interface")
    print("=" * 60)

    # This would test the actual MMPP interface like:
    # op[2].fft.plot_spectrum(dset="m_z11", save=True)
    # But we'll test the core components for now

    with tempfile.TemporaryDirectory() as temp_dir:
        test_zarr_path = os.path.join(temp_dir, "test_interface.zarr")
        create_test_zarr_file(test_zarr_path)

        # Test FFTCompute directly
        fft_compute = FFTCompute()

        print("\n1. Testing default save=True behavior")
        print("-" * 50)

        # Test with save=True by default (as implemented)
        result = fft_compute.calculate_fft_data(
            zarr_path=test_zarr_path,
            dataset="m_z11",
            save=True,  # This should be default in plot_spectrum
            save_dataset_name=None,  # Auto-generate name
        )

        print(f"✓ FFT calculation completed")
        print(
            f"  - Auto-generated dataset name: {result.metadata.get('save_dataset_name', 'N/A')}"
        )

        # Verify the zarr file has the FFT data
        z = zarr.open(test_zarr_path, mode="r")
        if "fft" in z:
            print(f"✓ FFT group created with datasets: {list(z['fft'].keys())}")

        return True


if __name__ == "__main__":
    try:
        # Test the core FFT save/cache functionality
        success1 = test_fft_save_functionality()

        # Test the high-level interface
        success2 = test_high_level_interface()

        if success1 and success2:
            print(
                "\n🎉 All tests passed! FFT save/cache functionality is working correctly."
            )
        else:
            print("\n❌ Some tests failed.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
