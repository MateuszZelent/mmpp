# MMPP2 Development Guide

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git
- Just (command runner) - install with: `cargo install just` or `pip install just-install`

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mmpp2.git
   cd mmpp2
   ```

2. **Setup development environment**
   ```bash
   just dev-setup
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Test the setup**
   ```bash
   python test_setup.py
   ```

## Development Workflow

### Available Just Commands

- `just build` - Build the package
- `just install-local` - Install package locally for testing
- `just test` - Run tests
- `just lint` - Check code quality
- `just format` - Format code with black
- `just clean` - Clean build artifacts
- `just prepare-release` - Prepare package for release
- `just release` - Release to PyPI
- `just release-test` - Release to TestPyPI for testing

### Building and Testing Locally

```bash
# Format code
just format

# Run linting
just lint

# Run tests
just test

# Build package
just build

# Install locally
just install-local

# Test installation
python -c "import mmpp; print('Success!')"
```

### Release Process

1. **Prepare for release**
   ```bash
   just prepare-release
   ```

2. **Test on TestPyPI first**
   ```bash
   just release-test
   ```

3. **Create a git tag and push**
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

4. **GitHub Actions will automatically release to PyPI**

## GitHub Actions

### Automatic Releases

1. **Set up PyPI tokens**:
   - Go to PyPI.org and create an API token
   - Add it to GitHub repository secrets as `PYPI_API_TOKEN`
   - For TestPyPI, add `TEST_PYPI_API_TOKEN`

2. **Create a release**:
   - Push a tag starting with 'v' (e.g., v0.1.0)
   - GitHub Actions will automatically build and release

3. **Manual release**:
   - Go to GitHub Actions tab
   - Run "Release to PyPI" workflow manually
   - Choose TestPyPI option for testing

### CI/CD Pipeline

- **Continuous Integration**: Runs on every push and PR
- **Multi-platform testing**: Ubuntu, Windows, macOS
- **Multi-Python testing**: Python 3.8, 3.9, 3.10, 3.11
- **Code quality checks**: flake8, black, mypy
- **Test coverage**: pytest with coverage reporting

## Package Structure

```
mmpp2/
├── mmpp/                   # Main package
│   ├── __init__.py        # Package initialization
│   ├── core.py            # Core functionality (from main.py)
│   ├── plotting.py        # Plotting utilities
│   ├── simulation.py      # Simulation management (from swapper.py)
│   ├── cli.py             # Command line interface
│   ├── paper.mplstyle     # Matplotlib style
│   └── fonts/             # Custom fonts
├── tests/                 # Test suite
├── .github/               # GitHub workflows
├── pyproject.toml         # Modern Python package configuration
├── setup.py              # Legacy setup (for compatibility)
├── justfile              # Task automation
├── README.md             # Package documentation
├── LICENSE               # MIT license
└── MANIFEST.in           # Package data inclusion
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
   ```bash
   pip install -e ".[dev]"
   ```

2. **Font issues**: The package includes custom fonts, but system fonts will be used as fallback

3. **Missing dependencies**: Use extras for optional features:
   ```bash
   pip install mmpp2[plotting,interactive]
   ```

### Version Management

```bash
# Bump patch version (0.1.0 -> 0.1.1)
just bump-patch

# Bump minor version (0.1.0 -> 0.2.0)
just bump-minor
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

All contributions are welcome!
