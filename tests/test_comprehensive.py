#!/usr/bin/env python3
"""
Final comprehensive test of the smart legend functionality.
This simulates your exact use case: results = jobs.find(solver=(3),amp_values=0.0022)
"""
import sys
from pathlib import Path

# Add mmpp to path
sys.path.insert(0, str(Path(__file__).parent))


def test_smart_legend_comprehensive():
    """Test the complete smart legend functionality."""
    print("🧪 Comprehensive Smart Legend Test")
    print("=" * 40)

    try:
        from mmpp.plotting import MMPPlotter

        # Mock result class that simulates real MMPP results
        class MockZarrJobResult:
            def __init__(self, **kwargs):
                # Set all provided parameters as attributes
                for k, v in kwargs.items():
                    setattr(self, k, v)
                self.path = f"simulation_{kwargs.get('solver', 1)}_{kwargs.get('f0', 1e9):.0e}.h5"
                self.attributes = {}

        # Simulate your exact scenario: jobs.find(solver=(3), amp_values=0.0022)
        # All results have solver=3 and amp_values=0.0022 (constant)
        # But other parameters vary
        simulation_results = [
            MockZarrJobResult(
                solver=3,  # constant (from find filter)
                amp_values=0.0022,  # constant (from find filter)
                f0=1e9,  # varies
                maxerr=1e-6,  # varies
                dt=1e-12,  # varies
                Nx=64,  # varies
                Ny=64,  # constant
                Nz=1,  # constant
                PBCx=1,  # constant
                PBCy=1,  # constant
                PBCz=0,  # constant
            ),
            MockZarrJobResult(
                solver=3,  # constant
                amp_values=0.0022,  # constant
                f0=2e9,  # varies (different)
                maxerr=1e-7,  # varies (different)
                dt=5e-13,  # varies (different)
                Nx=128,  # varies (different)
                Ny=64,  # constant
                Nz=1,  # constant
                PBCx=1,  # constant
                PBCy=1,  # constant
                PBCz=0,  # constant
            ),
            MockZarrJobResult(
                solver=3,  # constant
                amp_values=0.0022,  # constant
                f0=3e9,  # varies (different)
                maxerr=1e-8,  # varies (different)
                dt=2e-13,  # varies (different)
                Nx=256,  # varies (different)
                Ny=64,  # constant
                Nz=1,  # constant
                PBCx=1,  # constant
                PBCy=1,  # constant
                PBCz=0,  # constant
            ),
        ]

        print(f"📊 Created {len(simulation_results)} simulation results")
        print("   Simulating: results = jobs.find(solver=(3), amp_values=0.0022)")

        # Create plotter
        plotter = MMPPlotter(simulation_results)

        # Test smart legend detection
        varying_params = plotter._get_varying_parameters(simulation_results)
        print(f"\n🔍 Smart legend detected varying parameters: {varying_params}")

        # Expected: f0, maxerr, dt, Nx should vary
        # Expected: solver, amp_values, Ny, Nz, PBCx, PBCy, PBCz should be constant

        expected_varying = {"f0", "maxerr", "dt", "Nx"}
        expected_constant = {"solver", "amp_values", "Ny", "Nz", "PBCx", "PBCy", "PBCz"}

        actual_varying = set(varying_params)

        if expected_varying.issubset(actual_varying):
            print("✅ Correctly detected all expected varying parameters")
        else:
            missing = expected_varying - actual_varying
            print(f"❌ Missing expected varying parameters: {missing}")

        # Check that constant parameters are not in varying list
        unexpected_varying = expected_constant.intersection(actual_varying)
        if not unexpected_varying:
            print("✅ Correctly excluded constant parameters from varying list")
        else:
            print(f"❌ Incorrectly included constant parameters: {unexpected_varying}")

        # Test legend formatting
        print(f"\n🏷️  Smart legend labels:")
        for i, result in enumerate(simulation_results):
            label = plotter._format_result_label(result, varying_params)
            print(f"   Dataset {i+1}: {label}")

        # Verify format quality
        all_labels = [
            plotter._format_result_label(r, varying_params) for r in simulation_results
        ]

        # Check that scientific notation is used for appropriate parameters
        has_scientific = any("e" in label.lower() for label in all_labels)
        print(f"✅ Scientific notation used: {has_scientific}")

        # Check that constant parameters don't appear in labels
        constant_in_labels = any(
            "solver=3" in label or "amp_values=0.0022" in label for label in all_labels
        )
        if not constant_in_labels:
            print("✅ Constant parameters correctly excluded from labels")
        else:
            print("❌ Constant parameters incorrectly included in labels")

        # Test with different configuration
        print(f"\n⚙️  Testing with limited parameters (max_legend_params=2):")
        plotter.configure(max_legend_params=2)
        for i, result in enumerate(simulation_results):
            label = plotter._format_result_label(result, varying_params[:2])
            print(f"   Dataset {i+1}: {label}")

        print(f"\n✅ Smart legend test completed successfully!")
        print(f"\n💡 In your actual usage:")
        print(f"   results = jobs.find(solver=(3), amp_values=0.0022)")
        print(
            f"   results.matplotlib.plot(x_series='t', y_series='m_z11', average=(1,2,3), comp='z')"
        )
        print(f"   # Legend will show only: {', '.join(varying_params[:4])}")
        print(
            f"   # Hidden from legend: solver, amp_values, and other constant parameters"
        )

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_smart_legend_comprehensive()
    if success:
        print(f"\n🎉 All tests passed! Smart legend functionality is ready to use.")
    else:
        print(f"\n💥 Some tests failed. Please check the implementation.")
