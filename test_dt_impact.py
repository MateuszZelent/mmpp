#!/usr/bin/env python3
"""
Test to demonstrate the impact of the dt fix
"""

import numpy as np
import mmpp
from pyzfn import Pyzfn

print("Demonstration of dt reading fix")
print("="*60)

zarr_path = "/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr"

# Load MMPP job to get the reference value
job = mmpp.MMPP(zarr_path)
correct_dt = job[0].t_sampl
print(f"Correct dt (from job[0].t_sampl): {correct_dt}")

# Show what old method would give
pyz_job = Pyzfn(zarr_path)
old_method_dt = getattr(pyz_job, "t_sampl", 1e-12)
new_method_dt_raw = pyz_job.attrs.get("t_sampl", 1e-12)
new_method_dt = float(new_method_dt_raw) if new_method_dt_raw is not None else 1e-12

print(f"Old method (getattr): {old_method_dt}")
print(f"New method (attrs.get): {new_method_dt}")
print()

# Show the impact on frequency calculation
print("Impact on frequency calculations:")
print("-"*40)

# Simulate with sample parameters
n_points = 1000
old_freq_max = 1 / (2 * old_method_dt)
new_freq_max = 1 / (2 * new_method_dt)

print(f"Max frequency (old method): {old_freq_max / 1e9:.2f} GHz")
print(f"Max frequency (new method): {new_freq_max / 1e9:.2f} GHz")
print(f"Frequency range ratio: {old_freq_max / new_freq_max:.1f}")

# Show frequency resolution impact
old_freq_res = 1 / (n_points * old_method_dt)
new_freq_res = 1 / (n_points * new_method_dt)

print(f"Frequency resolution (old): {old_freq_res / 1e6:.2f} MHz")
print(f"Frequency resolution (new): {new_freq_res / 1e6:.2f} MHz")
print(f"Resolution ratio: {old_freq_res / new_freq_res:.1f}")

print()
print("Summary:")
print(f"• dt correction factor: {new_method_dt / old_method_dt:.1f}x")
print(f"• This affects all FFT frequency calculations and scaling")
print(f"• Peak positions will shift by factor of {old_method_dt / new_method_dt:.1f}")
print("="*60)