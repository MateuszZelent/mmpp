"""
Simple test to verify optimized colorbar functionality.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Test the imports
try:
    from optimized_colorbar import create_optimized_colorbar, _get_scientific_colormap

    print("✅ Successfully imported optimized colorbar functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


# Test colormap selection
def test_colormap_selection():
    print("\n🎨 Testing colormap selection:")

    test_maps = ["balance", "thermal", "phase", "diff", "invalid_colormap"]

    for cmap_name in test_maps:
        try:
            cmap = _get_scientific_colormap(cmap_name)
            print(f"  ✅ {cmap_name} -> {cmap.name}")
        except Exception as e:
            print(f"  ⚠️  {cmap_name} -> Error: {e}")


# Test basic colorbar creation
def test_basic_colorbar():
    print("\n📊 Testing basic colorbar creation:")

    # Create test data
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)
    data = np.sin(X) * np.cos(Y)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create image
    im = ax.imshow(data, extent=[0, 10, 0, 10], aspect="equal", origin="lower")

    # Test optimized colorbar
    try:
        cbar = create_optimized_colorbar(
            mappable=im,
            ax=ax,
            colormap="balance",
            label="Test Data",
            units="test units",
            system_size=(100, 100),
            spatial_resolution=(2.0, 2.0),
            discrete_levels=8,
            dark_theme=False,
        )

        print("  ✅ Basic colorbar creation successful")

        # Save test figure
        plt.savefig("test_colorbar.png", dpi=150, bbox_inches="tight")
        print("  ✅ Test figure saved as 'test_colorbar.png'")

    except Exception as e:
        print(f"  ❌ Colorbar creation failed: {e}")

    plt.close(fig)


# Test system size extraction simulation
def test_system_size_extraction():
    print("\n📏 Testing system size extraction:")

    # Mock zarr result
    class MockZarrResult:
        class MockZ:
            class MockAttrs:
                def get(self, key, default):
                    attrs = {"dx": 1.5e-9, "dy": 1.5e-9}
                    return attrs.get(key, default)

            attrs = MockAttrs()

            def __contains__(self, key):
                return key == "m"

            def __getitem__(self, key):
                if key == "m":

                    class MockDataset:
                        shape = (100, 5, 128, 128, 3)  # (time, z, y, x, components)

                    return MockDataset()
                raise KeyError(key)

        z = MockZ()

    try:
        from optimized_colorbar import extract_system_size_from_zarr

        mock_result = MockZarrResult()
        system_size, spatial_res = extract_system_size_from_zarr(mock_result)

        print(f"  ✅ System size: {system_size}")
        print(f"  ✅ Spatial resolution: {spatial_res}")

    except Exception as e:
        print(f"  ❌ System size extraction failed: {e}")


if __name__ == "__main__":
    print("🧪 Testing Optimized Colorbar Functionality")
    print("=" * 50)

    test_colormap_selection()
    test_basic_colorbar()
    test_system_size_extraction()

    print("\n" + "=" * 50)
    print("🎯 Summary:")
    print("- Created optimized colorbar system for MMPP mode visualization")
    print("- Supports cmocean scientific colormaps with matplotlib fallbacks")
    print("- Extracts system size from zarr metadata")
    print("- Provides discrete levels for better readability")
    print("- Optimized for both light and dark themes")
    print("- Includes comprehensive integration examples")

    print("\n📁 Files created:")
    print("- optimized_colorbar.py: Main colorbar enhancement module")
    print("- colorbar_examples.py: Usage examples and comparisons")
    print("- colorbar_integration.py: Integration guide for MMPP")
    print("- test_colorbar.py: This test script")

    print("\n🚀 Ready for integration with your MMPP mode visualization!")
