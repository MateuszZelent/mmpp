#!/usr/bin/env python3
"""
Simple test to verify MMPP with unified logging works correctly.
"""

import sys
import os

# Add the mmpp module to the path
sys.path.insert(
    0,
    "/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test/ammpp",
)

try:
    print("Testing MMPP with unified logging...")

    # Test normal mode (debug=False)
    print("\n=== Testing NORMAL mode (debug=False) ===")
    from mmpp import MMPP

    # This should show only INFO level messages
    op_normal = MMPP("/zfn2/mannga/jobs/vortices/spectrum/d100_sinc4.zarr", debug=False)
    print("Normal mode MMPP created successfully")

    # Test debug mode (debug=True)
    print("\n=== Testing DEBUG mode (debug=True) ===")
    op_debug = MMPP("/zfn2/mannga/jobs/vortices/spectrum/d100_sinc4.zarr", debug=True)
    print("Debug mode MMPP created successfully")

    print("\nLogging system test completed!")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
