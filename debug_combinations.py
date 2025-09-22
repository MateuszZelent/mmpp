#!/usr/bin/env python3
"""
Debug script to test parameter combinations in swap module.
"""

import itertools
import numpy as np

# Test data like user's config
test_params = {
    "aexCOFEB": list(np.linspace(0.9*1.6e-11, 1.1*1.6e-11, 3)),
    "aexYIG": list(np.linspace(0.9*4e-12, 1.1*4e-12, 3)),
    "msatCOFEB": list(np.linspace(1150e3*0.9, 1150e3*1.1, 3)), 
    "msatYIG": list(np.linspace(158e3*0.9, 158e3*1.1, 3))
}

print("Test parameters:")
for key, values in test_params.items():
    print(f"  {key}: {len(values)} values = {values}")

print("\nExpected combinations: 3×3×3×3 = 81")

# Test itertools.product (cartesian product - default)
combinations_product = list(itertools.product(*test_params.values()))
print(f"Cartesian product gives: {len(combinations_product)} combinations")

# Test zip (pairs mode)
combinations_zip = list(zip(*test_params.values()))
print(f"Zip (pairs mode) gives: {len(combinations_zip)} combinations")

print("\nFirst 5 combinations (cartesian product):")
param_names = list(test_params.keys())
for i, combo in enumerate(combinations_product[:5]):
    param_dict = dict(zip(param_names, combo))
    print(f"  {i}: {param_dict}")

print("\nFirst 3 combinations (zip/pairs mode):")
for i, combo in enumerate(combinations_zip):
    param_dict = dict(zip(param_names, combo))
    print(f"  {i}: {param_dict}")
