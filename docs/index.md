# MMPP Documentation

Welcome to the **MMPP** (Micro Magnetic Post Processing) library documentation!

MMPP is a comprehensive Python library for post-processing and analysis of micromagnetic simulation results. It provides powerful tools for:

- **Data Loading**: Efficient loading and management of simulation results from Zarr archives
- **FFT Analysis**: Fast Fourier Transform analysis for frequency domain studies  
- **Mode Analysis**: FMR (Ferromagnetic Resonance) mode computation and visualization
- **Batch Operations**: Serial computation across entire directories of simulation results
- **Interactive Plotting**: Rich visualization capabilities with matplotlib integration
- **Smart Data Management**: Automatic database creation and intelligent parameter scanning

## Key Features

### 🚀 Batch Operations
```python
import mmpp as mp

# Load entire directory of simulations
op = mp.open("/path/to/simulations")

# Batch FFT mode computation for all results (auto-selects optimal dataset)
op[:].fft.modes.compute_modes()

# Comprehensive reports
op[:].prepare_report(spectrum=True, modes=True)
```

### 📊 Interactive Analysis
```python
# Find specific simulations
results = op.find(solver=3, amp_values=0.0022)

# Plot comparative results
results.plot("time", "my")

# Interactive table view
op.show()
```

### 🔍 Advanced FFT Analysis
```python
# Single result analysis
result = op[0]
fft_analyzer = result.fft

# Compute modes
modes = fft_analyzer.modes.compute_modes()

# Generate animations
modes.save_modes_animation(
    frequency_range=(1.0, 3.0),
    animation_type="temporal"
)
```

## Quick Start

```{note}
Make sure you have Python 3.8+ installed with numpy, pandas, and matplotlib.
```

### Installation

```bash
pip install mmpp
```

### Basic Usage

```python
import mmpp as mp

# Open simulation directory
op = mp.open("/path/to/your/simulations")

# Scan for available results
df = op.scan()

# Access individual results
first_result = op[0]

# Batch operations on all results
batch_ops = op[:]
```

## Documentation Structure

```{toctree}
:maxdepth: 2
:caption: Contents:

api/index
tutorials/index
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
