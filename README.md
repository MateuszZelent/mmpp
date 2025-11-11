# 🧲 MMPP - Micro Magnetic Post Processing

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-blue?style=flat-square)](https://MateuszZelent.github.io/mmpp/)
[![GitHub Issues](https://img.shields.io/github/issues/MateuszZelent/mmpp?style=flat-square)](https://github.com/MateuszZelent/mmpp/issues)
[![GitHub Stars](https://img.shields.io/github/stars/MateuszZelent/mmpp?style=flat-square)](https://github.com/MateuszZelent/mmpp/stargazers)

**A powerful Python library for micromagnetic simulation analysis and visualization**

[📖 Documentation](https://MateuszZelent.github.io/mmpp/) • [🚀 Getting Started](#-quick-start) • [🎯 Features](#-features) • [💡 Examples](#-examples)

</div>

---

## 🎯 Features

<table>
<tr>
<td width="50%">

### 🔬 **Advanced Analysis**
- 🌊 Fast Fourier Transform (FFT) computation
- 📊 Frequency spectrum analysis  
- 🎭 FMR mode identification
- 📈 Statistical data processing

</td>
<td width="50%">

### ⚡ **High Performance**
- 🚀 Parallel batch processing
- 💾 Efficient data handling with Zarr
- 🔄 Concurrent operations
- 📦 Memory-optimized workflows

</td>
</tr>
<tr>
<td width="50%">

### 🎨 **Rich Visualization**
- 📊 Publication-ready plots
- 🎬 Interactive animations
- 🎨 Custom styling themes
- 🖼️ Multiple export formats

</td>
<td width="50%">

### 🛠️ **Developer Friendly**
- 🐍 Pythonic API design
- 📚 Comprehensive documentation
- 🧪 Well-tested codebase
- 🔌 Extensible architecture

</td>
</tr>
<tr>
<td colspan="2">

### 🤖 **Smart Auto-Selection** ✨ NEW!
- 🎯 Automatic dataset detection and selection
- 📊 Intelligently chooses the largest magnetization dataset
- 🚀 Simplified API - no need to specify dataset names
- 🔄 Backwards compatible with manual dataset selection

</td>
</tr>
</table>

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install mmpp

# Or install latest development version
pip install git+https://github.com/MateuszZelent/mmpp.git
```

### Basic Usage

```python
import mmpp

# 📂 Load simulation data
op = mmpp.MMPP('path/to/simulation.zarr')

# 🔍 Single file analysis with auto-selection
result = op[0]
fft_analyzer = result.fft

# 🤖 Auto-dataset selection (NEW!) - automatically chooses largest m_z dataset
spectrum = fft_analyzer.spectrum()  # Uses auto-selection
power_spectrum = fft_analyzer.power()  # Uses auto-selection

# 🎯 Or specify dataset explicitly
spectrum = fft_analyzer.spectrum(dset='m_z5-8')

# ⚡ Batch processing
batch = op[:]  # Get all results
modes = batch.fft.modes.compute_modes(parallel=True)  # Auto-selection in batch too
```

## 🤖 Smart Auto-Selection Feature

MMPP now includes intelligent dataset auto-selection that automatically chooses the best magnetization dataset for analysis:

```python
# ✨ NEW: Auto-selection API (recommended)
result = op[0]
fft_analyzer = result.fft

# No need to specify dataset - MMPP chooses the largest m_z dataset automatically
spectrum = fft_analyzer.spectrum()
power_spectrum = fft_analyzer.power()
modes = fft_analyzer.modes.compute_modes()

# 🔍 Check which dataset was auto-selected
selected_dataset = result.get_largest_m_dataset()
print(f"Auto-selected dataset: {selected_dataset}")  # e.g., "m_z5-8"

# 🔄 Traditional API still works for manual control
spectrum = fft_analyzer.spectrum(dset='m_z5-8')
```

**Benefits:**
- 🎯 **Simplified API**: No need to remember dataset names
- 🚀 **Intelligent Selection**: Automatically finds the best dataset
- 🔄 **Backward Compatible**: Existing code continues to work
- 📊 **Consistent Results**: Always uses the dataset with most data points

## 💡 Examples

### 🔄 Batch Processing
Process multiple simulation files efficiently:

```python
# 📁 Process all files in a directory
op = mmpp.MMPP('simulation_results/')
batch = op[:]

# ⚡ Parallel FFT analysis with auto-selection (NEW!)
modes = batch.fft.modes.compute_modes(parallel=True)  # Auto-selects best dataset

# � Or specify dataset explicitly for batch operations
modes = batch.fft.modes.compute_modes(dset='m_z5-8', parallel=True)

# 🚀 Complete analysis in one call (NEW!)
results = batch.process(parallel=True, max_workers=4)  # FFT + mode analysis
print(f"Processed {results['successful']}/{results['total']} files successfully")
```

### 🌊 Advanced FFT Analysis
Comprehensive frequency domain analysis:

```python
# 🤖 Auto-selection (NEW!) - Let MMPP choose the best dataset
spectrum = fft_analyzer.spectrum()  # Automatically selects largest m_z dataset
power_spectrum = fft_analyzer.power()
frequencies = fft_analyzer.frequencies()
modes = fft_analyzer.modes.compute_modes()

# 🎯 Manual dataset selection (traditional approach)
spectrum = fft_analyzer.spectrum(dset='m_z5-8')
power_spectrum = fft_analyzer.power(dset='m_z5-8')
frequencies = fft_analyzer.frequencies(dset='m_z5-8')
modes = fft_analyzer.modes.compute_modes(dset='m_z5-8')

# 🎬 Plot mode visualizations at specific frequency
plot_result = fft_analyzer.plot_modes(frequency=10.5)  # Auto-selection
plot_result = fft_analyzer.plot_modes(frequency=10.5, dset='m_z5-8')  # Manual
```

### 🎨 Publication-Ready Visualizations
Create stunning plots with built-in themes:

```python
# 📈 Custom styled plots
import mmpp.plotting as mplt
mplt.plot_spectrum(spectrum, style='publication')

# 🎨 Interactive visualizations
mplt.interactive_plot(data, colormap='viridis')

# 💾 Export in multiple formats
mplt.save_figure('spectrum.png', dpi=300, format='png')
```

## ⚡ Performance Tips

### 🚀 Optimize Your Workflow

#### Use Parallel Processing
```python
# Enable parallel processing for batch operations
modes = batch.fft.modes.compute_modes(parallel=True)

# Control number of workers
modes = batch.fft.modes.compute_modes(parallel=True, max_workers=4)
```

#### Leverage Auto-Selection
```python
# Let MMPP choose the optimal dataset automatically
spectrum = fft_analyzer.spectrum()  # Faster than manual selection
```

#### Memory Management
```python
# Process large datasets in chunks to manage memory usage
op = mmpp.MMPP('large_simulation_directory/')
batch_size = 50  # Process 50 results at a time

print(f"Total files: {len(op)}")
for i in range(0, len(op), batch_size):
    chunk = op[i:i+batch_size]
    results = chunk.process(parallel=True, max_workers=4)
    
    chunk_num = i//batch_size + 1
    total_chunks = (len(op) + batch_size - 1) // batch_size
    print(f"Chunk {chunk_num}/{total_chunks}: {results['successful']}/{results['total']} successful "
          f"({results['computation_time']:.1f}s)")
    
    # Optional: Clear memory or save intermediate results
    if results['failed'] > 0:
        print(f"⚠️  {results['failed']} files failed in chunk {chunk_num}")
```

#### Efficient Data Loading
```python
# Load only what you need
result = op[0]  # Single result
specific_results = op.find(solver=3, amp_values=0.0022)  # Filtered results
```

### 📊 Benchmarks

Typical performance on a modern system (16GB RAM, 8-core CPU):

| Operation | Single File | Batch (10 files) | Parallel Batch |
|-----------|-------------|------------------|-----------------|
| Load Data | ~0.1s | ~1.0s | ~0.3s |
| FFT Analysis | ~2.0s | ~20s | ~5s |
| Mode Computation | ~5.0s | ~50s | ~12s |

> Performance varies significantly based on dataset size and system specifications.

## 📚 Documentation & Resources

<div align="center">

| Resource | Description | Link |
|----------|-------------|------|
| 📖 **Documentation** | Complete API reference and tutorials | [GitHub Pages](https://MateuszZelent.github.io/mmpp/) |
| 🎓 **Tutorials** | Step-by-step guides and examples | [Tutorials](https://MateuszZelent.github.io/mmpp/tutorials/) |
| 🔬 **API Reference** | Detailed function documentation | [API Docs](https://MateuszZelent.github.io/mmpp/api/) |
| 🚀 **Getting Started** | Quick start guide | [Getting Started](https://MateuszZelent.github.io/mmpp/tutorials/getting_started/) |
| 🗂️ **PyZFN Library** | ZFN file format handling (dependency) | [PyZFN by Mathieu Moalic](https://github.com/MathieuMoalic/pyzfn) |

</div>

### 🏗️ Build Documentation Locally

```bash
# Quick build and serve
./build_docs.sh --serve

# Manual build
cd docs
pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints
sphinx-build -b html . _build
```

## 🔧 Installation Options

### 📦 Standard Installation
```bash
pip install mmpp
```

### 🛠️ Development Installation
```bash
git clone https://github.com/MateuszZelent/mmpp.git
cd mmpp
pip install -e ".[dev]"
```

### 🎯 Optional Features
```bash
# Interactive Jupyter support
pip install mmpp[interactive]

# Enhanced plotting capabilities
pip install mmpp[plotting]

# Full development environment
pip install mmpp[dev]
```

## 📋 Requirements

### Core Dependencies
- 🐍 **Python** ≥3.9
- 🔢 **NumPy** ≥1.20.0
- 🐼 **Pandas** ≥1.3.0
- 📊 **Matplotlib** ≥3.5.0
- 🗂️ **PyZFN** - ZFN file format handling ([Mathieu Moalic](https://github.com/MathieuMoalic/pyzfn))
- ⚡ **Zarr** - High-performance data storage
- 🎨 **Rich** - Beautiful terminal output
- 📈 **TQDM** - Progress bars

### Optional Dependencies
- 🪐 **Jupyter Ecosystem** (`itables`, `IPython`, `jupyter`)
- 🌊 **Enhanced Plotting** (`cmocean`, `seaborn`)
- 🧪 **Development Tools** (`pytest`, `ruff`, `mypy`)

## 💻 System Requirements

### Supported Platforms
- 🐧 **Linux** (Ubuntu 18.04+, CentOS 7+, etc.)
- 🍎 **macOS** (10.14+)
- 🪟 **Windows** (10+)

### Hardware Recommendations
- **RAM**: 8GB minimum, 16GB+ recommended for large datasets
- **Storage**: SSD recommended for better I/O performance
- **CPU**: Multi-core processor recommended for parallel operations

### Python Compatibility
- ✅ **Python 3.9** - Minimum supported version
- ✅ **Python 3.10** - Fully supported  
- ✅ **Python 3.11** - Fully supported
- ⚠️ **Python 3.12** - Beta support (some dependencies may vary)

## 📚 Additional Documentation

For developers and advanced users, additional documentation is available:

### 🔬 FFT Analysis Documentation
- **[Complete FFT API Analysis](docs/analysis/KOMPLETNA_ANALIZA_FFT_API.md)** - Detailed technical analysis of FFT functionality
- **[FFT API Verification](docs/analysis/WERYFIKACJA_POPRAWNOSCI_FFT.md)** - Verification of all FFT examples and methods
- **[Detailed FFT Analysis](docs/analysis/FFT_API_ANALIZA_SZCZEGOLOWA.md)** - In-depth FFT implementation details

### 🛠️ Development Documentation
- **[Performance Optimization](docs/development/PERFORMANCE_OPTIMIZATION_SUMMARY.md)** - Performance enhancement strategies
- **[Smart Legend Documentation](docs/development/SMART_LEGEND_DOCS.md)** - Advanced plotting features
- **[GitHub Pages Setup](docs/development/GITHUB_PAGES_SETUP.md)** - Documentation deployment guide
- **[Workflow Fixes](docs/development/WORKFLOW_FIXES.md)** - Development workflow improvements

## 🤝 Contributing

We welcome contributions! Here's how you can help:

<div align="center">

| Type | Description | Action |
|------|-------------|--------|
| 🐛 **Bug Reports** | Found an issue? | [Open Issue](https://github.com/MateuszZelent/mmpp/issues/new) |
| 💡 **Feature Requests** | Have an idea? | [Discussion](https://github.com/MateuszZelent/mmpp/discussions) |
| 🔧 **Pull Requests** | Want to contribute code? | [Contributing Guide](CONTRIBUTING.md) |
| 📖 **Documentation** | Improve the docs | [Edit on GitHub](https://github.com/MateuszZelent/mmpp/tree/main/docs) |

</div>

### 🚀 Quick Contribution Setup
```bash
# Fork and clone the repository
git clone https://github.com/MateuszZelent/mmpp.git
cd mmpp

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Check code style
ruff check mmpp/
ruff format --check mmpp/
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ by [MateuszZelent](https://github.com/MateuszZelent)
- Powered by the amazing Python scientific computing ecosystem
- **PyZFN integration**: Utilizes components from [PyZFN](https://github.com/MathieuMoalic/pyzfn) by [Mathieu Moalic](https://github.com/MathieuMoalic) for efficient ZFN file handling
- Special thanks to all contributors and users

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/MateuszZelent/mmpp/issues) • [Request Feature](https://github.com/MateuszZelent/mmpp/discussions) • [Documentation](https://MateuszZelent.github.io/mmpp/)

</div>

## ❓ Frequently Asked Questions

### 🔍 **Q: How does auto-selection work?**
A: MMPP automatically identifies and selects the largest magnetization dataset (m_z*) in your simulation files. This ensures you're always working with the most comprehensive data available.

### 📊 **Q: Can I still use manual dataset selection?**
A: Yes! The auto-selection feature is backward compatible. You can still specify datasets manually using the `dset` parameter in any method.

### ⚡ **Q: How do I speed up batch processing?**
A: Use the `parallel=True` parameter in batch operations:
```python
batch.fft.modes.compute_modes(parallel=True)
```

### 🐛 **Q: I'm getting import errors. What should I do?**
A: Make sure you have all dependencies installed:
```bash
pip install mmpp[dev]  # For full functionality
```

### 📁 **Q: What file formats does MMPP support?**
A: MMPP primarily works with Zarr archives (.zarr) from micromagnetic simulations. The library is optimized for this format's high-performance capabilities.

### 🔌 **Q: Do I need to install pyzfn separately?**
A: No! MMPP now includes an embedded version of pyzfn. This ensures stability and compatibility without requiring a separate pyzfn installation.

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```python
# Problem: ModuleNotFoundError
# Solution: Install missing dependencies
pip install mmpp[dev]
```

#### Memory Issues with Large Datasets
```python
# Problem: Out of memory errors
# Solution: Process data in chunks or use batch operations
batch_size = 10
for chunk in op.chunks(batch_size):
    results = chunk.fft.modes.compute_modes()
```

#### Performance Issues
```python
# Problem: Slow FFT computation
# Solution: Use parallel processing
modes = batch.fft.modes.compute_modes(parallel=True, max_workers=4)
```

### Getting Help

If you encounter issues:

1. **Check the Documentation**: [GitHub Pages](https://MateuszZelent.github.io/mmpp/)
2. **Search Issues**: [GitHub Issues](https://github.com/MateuszZelent/mmpp/issues)
3. **Ask Questions**: [GitHub Discussions](https://github.com/MateuszZelent/mmpp/discussions)
4. **Contact**: mateusz.zelent@amu.edu.pl
