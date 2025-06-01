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

# 🔍 Single file analysis
result = op[0]
fft_analyzer = result.fft
spectrum = fft_analyzer.spectrum(dset='m_z5-8')
power_spectrum = fft_analyzer.power(dset='m_z5-8')

# ⚡ Batch processing - NEW!
batch = op[:]  # Get all results
# Note: Batch operations use different API (see batch section)
modes = batch.fft.modes.compute_modes('m_z5-8', parallel=True)
```

## 💡 Examples

### 🔄 Batch Processing
Process multiple simulation files efficiently:

```python
# 📁 Process all files in a directory
op = mmpp.MMPP('simulation_results/')
batch = op[:]

# ⚡ Parallel FFT analysis with progress tracking
# Note: Batch operations may use different methods
# spectra = batch.fft.compute_all('m_z5-8', parallel=True, progress=True)

# 🎭 Batch mode computation
modes = batch.fft.modes.compute_modes('m_z5-8', parallel=True)
```

### 🌊 Advanced FFT Analysis
Comprehensive frequency domain analysis:

```python
# 📊 Compute frequency spectrum (complex)
spectrum = fft_analyzer.spectrum(dset='m_z5-8')

# ⚡ Compute power spectrum  
power_spectrum = fft_analyzer.power(dset='m_z5-8')

# 📈 Get frequency array
frequencies = fft_analyzer.frequencies(dset='m_z5-8')

# 🎯 Identify FMR modes
modes = fft_analyzer.modes.compute_modes('m_z5-8')

# 🎬 Plot mode visualizations at specific frequency
plot_result = fft_analyzer.plot_modes(frequency=10.5, dset='m_z5-8')
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

## 📚 Documentation & Resources

<div align="center">

| Resource | Description | Link |
|----------|-------------|------|
| 📖 **Documentation** | Complete API reference and tutorials | [GitHub Pages](https://MateuszZelent.github.io/mmpp/) |
| 🎓 **Tutorials** | Step-by-step guides and examples | [Tutorials](https://MateuszZelent.github.io/mmpp/tutorials/) |
| 🔬 **API Reference** | Detailed function documentation | [API Docs](https://MateuszZelent.github.io/mmpp/api/) |
| 🚀 **Getting Started** | Quick start guide | [Getting Started](https://MateuszZelent.github.io/mmpp/tutorials/getting_started/) |

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
- 🐍 **Python** ≥3.8
- 🔢 **NumPy** ≥1.20.0
- 🐼 **Pandas** ≥1.3.0
- 📊 **Matplotlib** ≥3.5.0
- ⚡ **Zarr** - High-performance data storage
- 🎨 **Rich** - Beautiful terminal output
- 📈 **TQDM** - Progress bars

### Optional Dependencies
- 🪐 **Jupyter Ecosystem** (`itables`, `IPython`, `jupyter`)
- 🌊 **Enhanced Plotting** (`cmocean`, `seaborn`)
- 🧪 **Development Tools** (`pytest`, `black`, `flake8`)

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
black --check mmpp/
flake8 mmpp/
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ by [MateuszZelent](https://github.com/MateuszZelent)
- Powered by the amazing Python scientific computing ecosystem
- Special thanks to all contributors and users

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/MateuszZelent/mmpp/issues) • [Request Feature](https://github.com/MateuszZelent/mmpp/discussions) • [Documentation](https://MateuszZelent.github.io/mmpp/)

</div>
