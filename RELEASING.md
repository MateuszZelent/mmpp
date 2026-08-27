# MMPP release process

MMPP uses `pyproject.toml` as the single source of package metadata. The
version is read from `mmpp.__version__`; update that value and the matching
documentation version when preparing a release.

## Before tagging

Run the complete local gate from a clean environment:

```bash
uv sync --extra dev
uv run ruff format --check mmpp/ tests/ scripts/
uv run ruff check mmpp/ tests/ scripts/
uv run mypy mmpp/
uv run pytest tests/ -q
uv run python -m build --sdist --wheel
uv run twine check dist/*
```

The equivalent pip workflow is:

```bash
python -m pip install -e ".[dev]"
python -m build --sdist --wheel
python -m twine check dist/*
```

Verify that the wheel and source distribution contain the same version, then
install each artifact in a fresh environment and run the import smoke test.

## Publishing

Create an annotated tag matching the package version, for example:

```bash
git tag -a v0.5.5 -m "Release v0.5.5"
git push origin v0.5.5
```

The tag workflow rebuilds and verifies the artifacts, runs the release smoke
checks, and publishes through PyPI Trusted Publishing (OIDC). TestPyPI can be
selected from the workflow dispatch UI. Configure the `pypi` and `testpypi`
GitHub environments and their Trusted Publisher records before the first
publication.
