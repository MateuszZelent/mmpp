#!/usr/bin/env python3
"""
Test script to verify dt reading is fixed
"""

import mmpp
from mmpp.fft.compute_fft import FFTCompute

print("Testing dt reading fix...")
print("="*50)

# Load MMPP job
job = mmpp.MMPP("/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr")

# Get expected dt value
expected_dt = job[0].t_sampl
print(f"Expected dt (job[0].t_sampl): {expected_dt}")

# Create FFT compute instance
fft_compute = FFTCompute()

# Test the load_data_from_zarr method directly
try:
    data, actual_dt = fft_compute.load_data_from_zarr(
        zarr_path="/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr",
        dataset="m_dot",
        z_layer=0,
        tmax=100
    )
    print(f"Actual dt from load_data_from_zarr: {actual_dt}")
    print(f"Data shape: {data.shape}")
    
    # Check if they match
    if abs(expected_dt - actual_dt) < 1e-15:
        print("✓ SUCCESS: dt values match!")
        print(f"  Ratio: {actual_dt/expected_dt}")
    else:
        print("✗ FAILURE: dt values don't match!")
        print(f"  Difference: {abs(expected_dt - actual_dt)}")
        print(f"  Ratio: {actual_dt/expected_dt}")
        
except Exception as e:
    print(f"✗ Error in test: {e}")
    import traceback
    traceback.print_exc()

print("="*50)
print("Test completed.")