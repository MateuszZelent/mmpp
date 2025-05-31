# FFT Analysis Module

The FFT module provides comprehensive frequency domain analysis capabilities for micromagnetic simulations.

```{toctree}
:maxdepth: 2

core
modes
main
compute_fft
electromagnetic_analysis
plot
```

## Module Overview

### Main Classes
- {class}`mmpp.fft.FFT` - Main FFT interface
- {class}`mmpp.fft.FMRModeAnalyzer` - FMR mode analysis  
- {class}`mmpp.fft.FFTAnalyzer` - Core FFT computation

### Key Features
- Fast Fourier Transform computation
- FMR mode identification and analysis
- Frequency spectrum analysis
- Mode visualization and animation
- Electromagnetic field analysis

## Quick Start

```python
import mmpp as mp

# Load simulation result
op = mp.open("/path/to/simulation")
result = op[0]

# Access FFT functionality
fft = result.fft

# Compute and analyze modes
modes = fft.modes.compute_modes()
peaks = fft.modes.analyze_all()

# Generate visualizations
fft.modes.save_modes_animation(
    frequency_range=(1.0, 3.0),
    animation_type="temporal"
)
```
