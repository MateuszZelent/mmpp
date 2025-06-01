# Contributing to MMPP

Thank you for your interest in contributing to MMPP (Micro Magnetic Post Processing)! We welcome contributions from the community.

## 🚀 Quick Start

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/mmpp.git
   cd mmpp
   ```

2. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run Tests**
   ```bash
   pytest tests/
   ```

## 📋 Contribution Guidelines

### Code Style

We use several tools to maintain code quality:

- **Ruff** for code formatting and linting
- **MyPy** for type checking

Run these tools before submitting:
```bash
ruff format mmpp/
mypy mmpp/
```

### Testing

- Write tests for new features
- Ensure all tests pass: `pytest tests/`
- Maintain or improve test coverage

### Documentation

- Update docstrings for new functions/classes
- Add examples to the documentation
- Update the README if needed

## 🐛 Reporting Issues

When reporting issues, please include:

- Python version
- MMPP version
- Operating system
- Clear description of the problem
- Minimal code example to reproduce the issue

## 💡 Feature Requests

Before requesting a feature:

1. Check if it already exists in the [issues](https://github.com/MateuszZelent/mmpp/issues)
2. Consider if it fits the project scope
3. Provide a clear use case

## 📝 Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   pytest tests/
   ruff check mmpp/
   ruff format --check mmpp/
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   git push origin feature/your-feature-name
   ```

5. **Submit Pull Request**
   - Use a clear title and description
   - Reference any related issues
   - Wait for review and address feedback

## 🏗️ Development Workflow

### Using Just (Recommended)

If you have [just](https://github.com/casey/just) installed:

```bash
# Setup development environment
just dev-setup

# Run tests
just test

# Format code
just format

# Build documentation
just docs
```

### Manual Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Format code
ruff format mmpp/ tests/ scripts/

# Lint code
ruff check mmpp/ tests/ scripts/

# Type checking
mypy mmpp/

# Build documentation
cd docs && sphinx-build -b html . _build
```

## 🔬 Testing Guidelines

### Test Categories

- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **Performance Tests**: Verify performance requirements

### Writing Tests

```python
import pytest
from mmpp import MMPP

def test_feature():
    """Test description."""
    # Arrange
    data = setup_test_data()
    
    # Act
    result = feature_under_test(data)
    
    # Assert
    assert result.is_valid()
```

## 📚 Documentation

### Building Documentation

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build
```

### Documentation Structure

- **tutorials/**: Step-by-step guides
- **api/**: Auto-generated API documentation
- **development/**: Development documentation

## 🎯 Areas for Contribution

We especially welcome contributions in these areas:

- **Performance Optimization**: Improving computational efficiency
- **Visualization**: New plotting features and improvements
- **Testing**: Expanding test coverage
- **Documentation**: Examples, tutorials, and API docs
- **Bug Fixes**: Resolving issues and edge cases

## 🤝 Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers get started
- Follow the code of conduct

## 📞 Getting Help

- **Issues**: [GitHub Issues](https://github.com/MateuszZelent/mmpp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MateuszZelent/mmpp/discussions)
- **Email**: mateusz.zelent@amu.edu.pl

## 🏷️ Release Process

For maintainers:

1. Update version in `pyproject.toml`
2. Update `RELEASE_NOTES.md`
3. Create release tag
4. Build and publish to PyPI

---

Thank you for contributing to MMPP! 🧲
