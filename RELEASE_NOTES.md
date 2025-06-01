# MMPP Library Release v0.5.0

## 🚀 Major Feature Release: Comprehensive Batch Processing & Auto-Selection

### 🎯 What's New in v0.5.0

This release introduces powerful new features for efficient batch processing, automatic dataset selection, and enhanced memory management for large-scale micromagnetic simulations.

### 🔥 Key Features

#### 🤖 Automatic Dataset Selection (NEW!)
- **Smart auto-selection**: MMPP now automatically selects the optimal dataset for analysis
- **Intelligent detection**: Finds the largest `m_z` dataset for best analysis quality
- **Zero configuration**: Works out-of-the-box without manual dataset specification

```python
# Auto-selection in action
fft_analyzer = op[0].fft
spectrum = fft_analyzer.spectrum()  # Automatically selects best dataset
modes = fft_analyzer.modes.compute_modes()  # No dataset needed!

# Batch operations with auto-selection
batch = op[:]
modes = batch.fft.modes.compute_modes(parallel=True)  # Auto-selects optimal dataset
```

#### ⚡ Enhanced Batch Processing
- **Complete processing pipeline**: New `process()` method for comprehensive analysis
- **Memory management**: Efficient chunking for large datasets
- **Progress tracking**: Real-time progress bars and detailed logging
- **Error handling**: Robust error reporting and recovery

```python
# Process large datasets efficiently
for i in range(0, len(op), batch_size):
    chunk = op[i:i+batch_size]
    results = chunk.process(parallel=True, max_workers=4)
    print(f"Processed {results['successful']}/{results['total']} files")
```

#### 🔧 Implementation Improvements
- **Fixed parameter names**: Corrected `n_workers` → `max_workers` in all examples
- **Complete API**: Implemented missing `chunks()` and `process()` methods
- **Type annotations**: Enhanced type safety and IDE support
- **Documentation**: Comprehensive README with examples and troubleshooting

### � Full Changelog

#### ✨ New Features
- **Auto-Selection Engine**: Automatically selects optimal datasets for analysis
- **BatchOperations.process()**: Complete batch processing with FFT + mode analysis
- **MMPP.chunks()**: Memory-efficient chunking for large datasets
- **Enhanced Progress Tracking**: Detailed progress bars and timing information
- **Comprehensive Error Handling**: Better error reporting and recovery

#### 🔧 Improvements
- **Parameter Consistency**: Fixed `n_workers` → `max_workers` throughout codebase
- **Memory Optimization**: Improved memory usage for large batch operations  
- **Documentation**: Enhanced README with FAQ, troubleshooting, and performance tips
- **Type Safety**: Added comprehensive type annotations
- **Logging**: Enhanced logging with structured output

#### 🐛 Bug Fixes
- Fixed missing implementation for `chunk.process()` method referenced in README
- Corrected parameter naming inconsistencies in batch operations
- Fixed examples in documentation to match actual API

### 🏗️ Technical Details

#### Package Information
- **Version**: 0.5.0
- **Python Compatibility**: 3.8+ 
- **License**: MIT
- **Author**: Mateusz Zelent (mateusz.zelent@amu.edu.pl)

### 🚀 Installation & Upgrade
```bash
# Install from PyPI (once published)
pip install mmpp

```bash
# Upgrade existing installation
pip install --upgrade mmpp

# Install from source
git clone https://github.com/MateuszZelent/mmpp
cd mmpp
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### 🎯 Usage Examples

#### Basic Usage
```python
import mmpp

# Load simulation data
op = mmpp.MMPP('simulation_results/')

# Auto-selection magic ✨
result = op[0]
fft_analyzer = result.fft
spectrum = fft_analyzer.spectrum()  # Automatically selects best dataset!
modes = fft_analyzer.modes.compute_modes()
```

#### Batch Processing
```python
# Process all files with auto-selection
batch = op[:]
results = batch.process(parallel=True, max_workers=4)
print(f"Successfully processed {results['successful']}/{results['total']} files")

# Memory-efficient chunking for large datasets
batch_size = 50
for i in range(0, len(op), batch_size):
    chunk = op[i:i+batch_size]
    results = chunk.process(parallel=True)
```

### � Links & Resources

- **📖 Documentation**: [GitHub Pages](https://MateuszZelent.github.io/mmpp/)
- **🚀 Getting Started**: [Quick Start Guide](https://MateuszZelent.github.io/mmpp/tutorials/getting_started/)
- **🔬 API Reference**: [Complete API Docs](https://MateuszZelent.github.io/mmpp/api/)
- **🎓 Tutorials**: [Step-by-step Guides](https://MateuszZelent.github.io/mmpp/tutorials/)

### 🙏 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Full Diff**: [v0.1.0...v0.5.0](https://github.com/MateuszZelent/mmpp/compare/v0.1.0...v0.5.0)
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
