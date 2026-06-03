"""
Basic tests for mmpp library.
"""

import ast
import os
import sys
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from setuptools.config.pyprojecttoml import read_configuration

# Add the mmpp package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = Path(__file__).resolve().parents[1]


def _zarr_requirements(requirements: list[str]) -> list[Requirement]:
    return [
        requirement
        for requirement in (Requirement(value) for value in requirements)
        if requirement.name == "zarr"
    ]


def _applies_to_python(requirement: Requirement, python_version: str) -> bool:
    if requirement.marker is None:
        return True

    environment = default_environment()
    environment["python_version"] = python_version
    return requirement.marker.evaluate(environment)


def _assert_zarr_python_markers(requirements: list[str]) -> None:
    zarr_requirements = _zarr_requirements(requirements)

    python310 = [
        requirement
        for requirement in zarr_requirements
        if _applies_to_python(requirement, "3.10")
    ]
    python311 = [
        requirement
        for requirement in zarr_requirements
        if _applies_to_python(requirement, "3.11")
    ]

    assert python310
    assert any(
        requirement.specifier.contains("2.18.3", prereleases=True)
        for requirement in python310
    )
    assert not any(
        requirement.specifier.contains("3.0.0", prereleases=True)
        for requirement in python310
    )

    assert python311
    assert any(
        requirement.specifier.contains("3.0.0", prereleases=True)
        for requirement in python311
    )


def _setup_install_requires() -> list[str]:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "install_requires":
                return ast.literal_eval(keyword.value)

    raise AssertionError("setup.py does not declare install_requires")


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


def test_pyproject_zarr_dependency_matches_supported_python_versions():
    configuration = read_configuration(str(ROOT / "pyproject.toml"))

    _assert_zarr_python_markers(configuration["project"]["dependencies"])


def test_setup_py_zarr_dependency_matches_supported_python_versions():
    _assert_zarr_python_markers(_setup_install_requires())


if __name__ == "__main__":
    pytest.main([__file__])
