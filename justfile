# Justfile for mmpp library automation

# Default recipe
default:
    @just --list

# Install development dependencies
install-dev:
    pip install -e ".[dev]"

# Install the locked development environment with uv
install-dev-uv:
    uv sync --extra dev

# Build the package
build:
    @echo "🔨 Building package..."
    python -m build --sdist --wheel
    @echo "✅ Build complete!"

# Build with uv's PEP 517 frontend
build-uv:
    @echo "🔨 Building package with uv..."
    uv build
    @echo "✅ Build complete!"

# Clean build artifacts
clean:
    @echo "🧹 Cleaning build artifacts..."
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    @echo "✅ Clean complete!"

# Run tests
test:
    @echo "🧪 Running tests..."
    python -m pytest tests/ -v --cov=mmpp --cov-report=term-missing

# Format code
format:
    @echo "🎨 Formatting code..."
    python -m ruff format mmpp/ tests/ scripts/
    @echo "✅ Formatting complete!"

# Format and fix linting issues
format-fix:
    @echo "🎨 Formatting and fixing code..."
    python -m ruff format mmpp/ tests/ scripts/
    python -m ruff check --fix --unsafe-fixes mmpp/ tests/ scripts/ || true
    @echo "✅ Format and fix complete!"

# Lint code
lint:
    @echo "🔍 Linting code..."
    python -m ruff check mmpp/ tests/ scripts/
    python -m mypy mmpp/ --ignore-missing-imports
    @echo "✅ Linting complete!"

# Check package before release
check:
    @echo "🔍 Checking package..."
    python -m twine check dist/*
    @echo "✅ Package check complete!"

# Install package locally for testing
install-local: clean build
    @echo "📦 Installing package locally..."
    pip install dist/*.whl --force-reinstall
    @echo "✅ Local installation complete!"

# Prepare for release (format, lint, test, build, check)
prepare-release: format lint test clean build check
    @echo "🚀 Package ready for release!"

# Upload to PyPI (requires TWINE_USERNAME and TWINE_PASSWORD env vars)
release: prepare-release
    @echo "🚀 Uploading to PyPI..."
    python -m twine upload dist/*
    @echo "✅ Release complete!"

# Upload to TestPyPI for testing
release-test: prepare-release
    @echo "🧪 Uploading to TestPyPI..."
    python -m twine upload --repository testpypi dist/*
    @echo "✅ TestPyPI upload complete!"

# Create and setup virtual environment
setup-env:
    @echo "🐍 Setting up virtual environment..."
    python -m venv .venv
    @echo "Activate with: source .venv/bin/activate"
    @echo "Then run: just install-dev"

# Show package info
info:
    @echo "📊 Package Information:"
    @echo "Name: mmpp"
    @python -c "import mmpp; print(f\"Version: {mmpp.__version__}\")"
    @echo "Author: Mateusz Zelent"
    @python -c "import sys; print(f'Python: {sys.version}')"

# Show package size
size:
    @echo "📏 Package size:"
    @find mmpp/ -name "*.py" -exec wc -l {} + | tail -1

# Generate documentation (if you add docs later)
docs:
    @echo "📚 Generating documentation..."
    cd docs && sphinx-build -b html . _build --keep-going
    @echo "✅ Documentation generated in docs/_build"

# Quick development setup
dev-setup: setup-env install-dev
    @echo "🎉 Development environment ready!"
    @echo "Don't forget to activate your virtual environment!"

# Bump version (patch)
bump-patch:
    @echo "⬆️ Bumping patch version..."
    @python3 scripts/bump_version.py patch

# Bump version (minor)
bump-minor:
    @echo "⬆️ Bumping minor version..."
    @python3 scripts/bump_version.py minor

# Bump version (major)
bump-major:
    @echo "⬆️ Bumping major version..."
    @python3 scripts/bump_version.py major
