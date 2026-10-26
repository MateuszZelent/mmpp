"""
Basic tests for mmpp library.
"""

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


def _numcodecs_requirements(requirements: list[str]) -> list[Requirement]:
    return [
        requirement
        for requirement in (Requirement(value) for value in requirements)
        if requirement.name == "numcodecs"
    ]


def _applies_to_python(requirement: Requirement, python_version: str) -> bool:
    if requirement.marker is None:
        return True

    environment = default_environment()
    environment["python_version"] = python_version
    return requirement.marker.evaluate(environment)


def _assert_zarr_python_markers(requirements: list[str]) -> None:
    zarr_requirements = _zarr_requirements(requirements)
    project = read_configuration(str(ROOT / "pyproject.toml"))["project"]
    prefix = "Programming Language :: Python :: "
    versions = [
        value.removeprefix(prefix)
        for value in project["classifiers"]
        if value.startswith(prefix) and value.removeprefix(prefix).startswith("3.")
    ]
    assert versions
    for python_version in versions:
        applicable = [
            requirement
            for requirement in zarr_requirements
            if _applies_to_python(requirement, python_version)
        ]
        assert applicable
        assert any(r.specifier.contains("2.18.7", prereleases=True) for r in applicable)
        assert not any(
            r.specifier.contains("2.18.6", prereleases=True) for r in applicable
        )
        assert not any(
            r.specifier.contains("3.0.0", prereleases=True) for r in applicable
        )

    numcodecs_requirements = _numcodecs_requirements(requirements)
    assert numcodecs_requirements
    assert all(
        requirement.specifier.contains("0.15.2", prereleases=True)
        and not requirement.specifier.contains("0.16.0", prereleases=True)
        for requirement in numcodecs_requirements
    )


def _setup_install_requires() -> list[str]:
    configuration = read_configuration(str(ROOT / "pyproject.toml"))
    return configuration["project"]["dependencies"]


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
