# MMPP2 - Micro Magnetic Post Processing Library

A Python library for simulation and analysis of micromagnetic simulations with advanced post-processing capabilities.

## Features

- Micromagnetic simulation post-processing
- Advanced data analysis and visualization
- Rich plotting capabilities with custom styling
- Interactive visualization support
- Concurrent processing support
- Export/import functionality

## Installation

```bash
pip install mmpp
```

### Development Installation

```bash
git clone https://github.com/yourusername/mmpp2.git
cd mmpp2
pip install -e ".[dev]"
```

## Quick Start

```python
import mmpp

# Load simulation data
op = mmpp.MMPP('path/to/simulation.zarr')

# Single file analysis
result = op[0]
fft_analyzer = result.fft
spectrum = fft_analyzer.compute_spectrum(dset='m_z5-8')

# Batch processing (NEW!)
batch = op[:]  # Get all results
batch_results = batch.fft.compute_spectrum('m_z5-8', parallel=True)
modes = batch.fft.modes.compute_modes('m_z5-8', parallel=True)
```

## Key Features

### Batch Operations
Process multiple simulation files at once:
```python
# Process all files in a directory
op = mmpp.MMPP('simulation_results/')
batch = op[:]

# Batch FFT analysis
spectra = batch.fft.compute_spectrum('m_z5-8', parallel=True, progress=True)

# Batch mode computation
modes = batch.fft.modes.compute_modes('m_z5-8', parallel=True)
```

### Advanced FFT Analysis
- Fast Fourier Transform computation
- FMR mode identification
- Frequency spectrum analysis
- Mode visualization and animation

### Rich Plotting Capabilities
- Custom styling with matplotlib
- Interactive visualizations
- Export in multiple formats
- Publication-ready figures

## Documentation

📖 **Full documentation is available at: [https://yourusername.github.io/mmpp/](https://yourusername.github.io/mmpp/)**

### Local Documentation

Build documentation locally:
```bash
./build_docs.sh --serve
```

Or manually:
```bash
cd docs
pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints
sphinx-build -b html . _build
```

## Dependencies

- Python >=3.8
- numpy >=1.20.0
- pandas >=1.3.0
- matplotlib >=3.5.0
- pyzfn
- zarr
- rich
- tqdm

### Optional Dependencies

- `interactive`: For Jupyter notebook support (`itables`, `IPython`, `jupyter`)
- `plotting`: For enhanced plotting (`cmocean`)
- `dev`: For development tools (`pytest`, `black`, `flake8`, etc.)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
