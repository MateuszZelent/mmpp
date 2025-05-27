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
from mmpp import MMPPAnalyzer

# Create analyzer instance
analyzer = MMPPAnalyzer()

# Your code here...
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
