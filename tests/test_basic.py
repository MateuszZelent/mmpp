"""
Basic tests for MMPP2 library.
"""

import os
import sys

import pytest

# Add the mmpp package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_import():
    """Test that the package can be imported."""
    try:
        import mmpp

        assert hasattr(mmpp, "__version__")
        assert hasattr(mmpp, "__author__")
    except ImportError as e:
        pytest.fail(f"Failed to import mmpp: {e}")


def test_version():
    """Test that version is accessible."""
    import mmpp

    assert isinstance(mmpp.__version__, str)
    assert len(mmpp.__version__) > 0


def test_author():
    """Test that author is accessible."""
    import mmpp

    assert isinstance(mmpp.__author__, str)
    assert len(mmpp.__author__) > 0


def test_main_classes_available():
    """Test that main classes are available."""
    import mmpp

    # These should be available even if dependencies are missing
    expected_attrs = [
        "MMPPAnalyzer",
        "SimulationResult",
        "MMPPConfig",
        "MMPPlotter",
        "PlotConfig",
        "PlotterProxy",
        "SimulationManager",
    ]

    for attr in expected_attrs:
        try:
            assert hasattr(mmpp, attr), f"Missing attribute: {attr}"
        except ImportError:
            # Some classes might not be available if dependencies are missing
            pytest.skip(f"Skipping {attr} test due to missing dependencies")


if __name__ == "__main__":
    pytest.main([__file__])
