# MMPP Library Release v0.1.0

## 🎉 MMPP is now a PIP-compliant Python package!

### What's New

This release transforms the MMPP code from a collection of scripts into a professional, installable Python library with full automation for development and releases.

### 📦 Package Structure
- **Package Name**: `mmpp` (Micro Magnetic Post Processing)
- **Version**: 0.1.0
- **Author**: Mateusz Zelent (mateusz.zelent@amu.edu.pl)
- **License**: MIT

### 🚀 Installation
```bash
# Install from PyPI (once published)
pip install mmpp

# Install from source
git clone <repository>
cd mmpp
just install-local

# Or install manually
pip install -e .
```

### 🎯 Usage
```python
# Import the library
import mmpp

# Use individual components
from mmpp import MMPPAnalyzer, MMPPlotter, SimulationManager

# CLI usage
mmpp --help
mmpp info
```

### 🛠️ Development Automation

The project now includes comprehensive automation through `justfile`:

```bash
# Build and test
just build          # Build the package
just test           # Run tests
just install-local  # Install locally

# Code quality
just format         # Format code with black
just lint           # Lint with flake8
just check          # Run all checks

# Version management
just bump-patch     # Bump patch version (0.1.0 -> 0.1.1)
just bump-minor     # Bump minor version (0.1.0 -> 0.2.0)
just bump-major     # Bump major version (0.1.0 -> 1.0.0)

# Release
just prepare-release  # Prepare for release
just release-test     # Test release to TestPyPI
just release          # Release to PyPI
```

### 🔄 CI/CD Pipeline

- **GitHub Actions** for automated testing on multiple Python versions (3.8-3.11)
- **Cross-platform testing** (Ubuntu, Windows, macOS)
- **Automated PyPI releases** when git tags are pushed
- **Dependency updates** via Dependabot

### 📁 File Structure
```
mmpp/
├── mmpp/                    # Main package
│   ├── __init__.py         # Package exports
│   ├── core.py            # Core analysis functions (was main.py)
│   ├── plotting.py        # Plotting utilities
│   ├── simulation.py      # Simulation management (was swapper.py)
│   ├── cli.py             # Command line interface
│   ├── paper.mplstyle     # Matplotlib style
│   └── fonts/             # Font assets
├── tests/                  # Test suite
├── scripts/               # Utility scripts
├── .github/               # GitHub Actions workflows
├── pyproject.toml         # Modern Python packaging config
├── setup.py              # Legacy setup for compatibility
├── justfile              # Automation commands
├── README.md             # User documentation
├── DEVELOPMENT.md        # Developer guide
└── LICENSE               # MIT license
```

### 🔧 Technical Improvements

1. **Modern Python Packaging**
   - Uses `pyproject.toml` with SPDX license format
   - Proper package discovery with setuptools
   - Includes all assets (fonts, styles) as package data

2. **Graceful Dependency Handling**
   - Optional dependencies with fallback behavior
   - Clear error messages for missing packages

3. **CLI Interface**
   - Professional command-line interface
   - Version reporting and help system

4. **Automation**
   - One-command building, testing, and release
   - Automated version bumping
   - Cross-platform compatibility

### 🚀 Next Steps

1. **Set up PyPI credentials** for automated releases:
   ```bash
   # Add to GitHub Secrets:
   PYPI_API_TOKEN=your_pypi_token
   TEST_PYPI_API_TOKEN=your_test_pypi_token
   ```

2. **Create first release**:
   ```bash
   just prepare-release
   git tag v0.1.0
   git push origin v0.1.0  # Triggers automated PyPI release
   ```

3. **Test the package**:
   ```bash
   pip install mmpp
   python -c "import mmpp; print(mmpp.__version__)"
   ```

### 📝 Release Commands Summary

| Command | Description |
|---------|-------------|
| `just build` | Build wheel and source distribution |
| `just test` | Run test suite |
| `just release-test` | Release to TestPyPI |
| `just release` | Release to PyPI |
| `just bump-patch` | Version 0.1.0 → 0.1.1 |
| `just install-local` | Install package locally |

The MMPP library is now ready for professional distribution and use! 🎉
