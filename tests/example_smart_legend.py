#!/usr/bin/env python3
"""
Practical example showing the smart legend functionality for MMPP2 plotting.

This script demonstrates how the smart legend works with different datasets
where only varying parameters are shown in the legend.
"""
import sys
from pathlib import Path
import numpy as np

# Add mmpp to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mmpp.plotting import MMPPlotter
    import matplotlib.pyplot as plt

    print("🎯 Testing MMPP2 Smart Legend Functionality")
    print("=" * 50)

    # Create mock result objects that simulate real MMPP results
    class MockResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.path = f"simulation_{kwargs.get('solver', 'test')}_{kwargs.get('f0', 'freq')}.h5"
            self.attributes = {}

    # Create test datasets with some varying and some constant parameters
    # This simulates your example: results = jobs.find(solver=(3),amp_values=0.0022)
    results = [
        MockResult(
            solver=3,  # constant (from your filter)
            amp_values=0.0022,  # constant (from your filter)
            f0=1e9,  # varying
            maxerr=1e-6,  # varying
            Nx=64,  # varying
            Ny=64,  # constant
            Nz=1,  # constant
            PBCx=1,  # constant
            PBCy=1,  # constant
            PBCz=0,  # constant
            dt=1e-12,  # varying
        ),
        MockResult(
            solver=3,  # constant
            amp_values=0.0022,  # constant
            f0=2e9,  # varying (different value)
            maxerr=1e-6,  # varying (same value)
            Nx=128,  # varying (different value)
            Ny=64,  # constant
            Nz=1,  # constant
            PBCx=1,  # constant
            PBCy=1,  # constant
            PBCz=0,  # constant
            dt=5e-13,  # varying (different value)
        ),
        MockResult(
            solver=3,  # constant
            amp_values=0.0022,  # constant
            f0=1e9,  # varying (back to first value)
            maxerr=1e-7,  # varying (different value)
            Nx=64,  # varying (back to first value)
            Ny=64,  # constant
            Nz=1,  # constant
            PBCx=1,  # constant
            PBCy=1,  # constant
            PBCz=0,  # constant
            dt=2e-12,  # varying (different value)
        ),
    ]

    print(f"📊 Created {len(results)} mock datasets")
    print("\nDataset parameters:")
    for i, result in enumerate(results):
        print(
            f"  Dataset {i+1}: solver={result.solver}, f0={result.f0:.1e}, "
            f"maxerr={result.maxerr:.1e}, Nx={result.Nx}, dt={result.dt:.1e}"
        )

    # Test the smart legend functionality
    plotter = MMPPlotter(results)

    # Get varying parameters
    varying_params = plotter._get_varying_parameters(results)
    print(f"\n🔍 Detected varying parameters: {varying_params}")

    # Get constant parameters (for comparison)
    all_params = set()
    for result in results:
        for attr in dir(result):
            if (
                not attr.startswith("_")
                and attr not in ["path", "attributes"]
                and not callable(getattr(result, attr, None))
            ):
                try:
                    value = getattr(result, attr)
                    if isinstance(value, (int, float, str, bool)):
                        all_params.add(attr)
                except:
                    pass

    constant_params = all_params - set(varying_params)
    print(f"📌 Constant parameters (hidden from legend): {sorted(constant_params)}")

    # Test label formatting with smart legend
    print(f"\n🏷️  Smart legend labels (showing only varying parameters):")
    for i, result in enumerate(results):
        label = plotter._format_result_label(result, varying_params)
        print(f"  Dataset {i+1}: {label}")

    # Compare with old-style labels (showing all priority parameters)
    print(f"\n🏷️  Old-style labels (for comparison):")
    for i, result in enumerate(results):
        label = plotter._format_result_label(result, None)  # None = use old behavior
        print(f"  Dataset {i+1}: {label}")

    # Test configuration
    print(f"\n⚙️  Testing configuration options:")
    plotter.configure(max_legend_params=3, sort_results=True)
    print(f"   - max_legend_params set to 3")
    print(f"   - sort_results enabled")

    # Test with limited parameters
    print(f"\n🏷️  Limited smart legend labels (max 3 params):")
    for i, result in enumerate(results):
        label = plotter._format_result_label(result, varying_params[:3])
        print(f"  Dataset {i+1}: {label}")

    print(f"\n✅ Smart legend functionality test completed!")
    print(f"\n💡 Usage in your code:")
    print(f"   results = jobs.find(solver=(3), amp_values=0.0022)")
    print(
        f"   results.plot(x_series='t', y_series='m_z11', average=(1,2,3), comp='z')"
    )
    print(
        f"   # Legend will automatically show only: f0, maxerr, Nx, dt (parameters that vary)"
    )
    print(
        f"   # Hidden from legend: solver, amp_values, Ny, Nz, PBCx, PBCy, PBCz (constant parameters)"
    )

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure matplotlib and other dependencies are installed")
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback

    traceback.print_exc()
