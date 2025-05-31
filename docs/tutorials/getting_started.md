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
import numpy as np

# Perform FFT analysis
fft = result[0].fft

# Get complex spectrum
spectrum = fft.spectrum(dset='m_z11')
print(f"Spectrum shape: {spectrum.shape}")

# Get frequency array
frequencies = fft.frequencies()
print(f"Frequency range: {frequencies[0]/1e9:.2f} - {frequencies[-1]/1e9:.2f} GHz")

# Get power spectrum  
power = fft.power(dset='m_z11')
print(f"Peak power: {np.max(power):.2e}")

# Plot the spectrum
fig, ax = fft.plot_spectrum(dset='m_z11', log_scale=True)
```

### FMR Mode Analysis

```python
# Analyze FMR modes
modes = fft.modes

# Find peaks in spectrum
peaks = modes.find_peaks(threshold=0.1)
print(f"Found {len(peaks)} peaks")

# Interactive spectrum visualization
fig = modes.interactive_spectrum(components=['x', 'y', 'z'])

# Plot modes at specific frequency
if peaks:
    peak_freq = peaks[0].freq  # First peak frequency
    fig, axes = modes.plot_modes(frequency=peak_freq)
    print(f"Plotted modes at {peak_freq:.3f} GHz")

# Compute spatial modes
modes.compute_modes(save=True)
```

## Working with Multiple Files

```python
# Load multiple simulation results from database
db = mmpp.open('path/to/results_directory/')
results = db.find(solver=3, limit=10)
print(f"Found {len(results)} simulation results")

# Analyze each result
for i, result in enumerate(results):
    fft = result.fft
    power = fft.power(dset='m_z11')
    frequencies = fft.frequencies()
    
    # Find peak frequency
    peak_idx = np.argmax(power)
    peak_freq = frequencies[peak_idx] / 1e9  # Convert to GHz
    
    print(f"Result {i}: Peak at {peak_freq:.3f} GHz")

# Use batch operations for parallel processing
from mmpp.batch_operations import BatchOperations
batch = BatchOperations(results)
batch_results = batch.fft.compute_all(dset='m_z11')
print(f"Processed {len(batch_results)} files in batch")
```

## Next Steps

- Learn about [Batch Operations](batch_operations.md) for processing multiple files
- Explore the [API Reference](../api/index.md) for detailed documentation
- Check out [Examples](examples.md) for more complex workflows
