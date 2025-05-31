# Getting Started with MMPP

## Installation

MMPP can be installed using pip:

```bash
pip install mmpp
```

Or for development:

```bash
git clone https://github.com/kkingstoun/mmpp.git
cd mmpp
pip install -e .
```

## Quick Start

### Loading Simulation Data

```python
import mmpp

# Load a single simulation result
result = mmpp.MMPP('path/to/simulation.zarr')

# Access the underlying data
zarr_data = result[0]  # Get first (and only) result
print(f"Available datasets: {list(zarr_data.root.keys())}")
```

### Basic FFT Analysis

```python
# Perform FFT analysis
fft_analyzer = result[0].fft
spectrum = fft_analyzer.spectrum(dset='m_z5-8')

print(f"Spectrum shape: {spectrum.shape}")

# Plot the spectrum
fft_analyzer.plot_spectrum()
```

### FMR Mode Analysis

```python
# Analyze FMR modes
mode_analyzer = fft_analyzer.modes
modes = mode_analyzer.compute_modes(dset='m_z5-8')

print(f"Found {len(modes)} modes")

# Plot modes
mode_analyzer.plot_modes()
```

## Working with Multiple Files

```python
# Load multiple simulation results
op = mmpp.MMPP('path/to/results_directory/')
print(f"Loaded {len(op)} simulation results")

# Process all files at once
batch = op[:]  # Get batch operations interface
results = batch.fft.compute_all('m_z5-8')

print(f"Processed {len(results)} files")
```

## Next Steps

- Learn about [Batch Operations](batch_operations.md) for processing multiple files
- Explore the [API Reference](../api/index.md) for detailed documentation
- Check out [Examples](examples.md) for more complex workflows
