# API Reference

This section contains the complete API documentation for MMPP.

## Core Modules

```{toctree}
:maxdepth: 2

core
batch_operations
plotting
fft/index
simulation
logging_config
```

## Module Overview

### Core Classes
- {class}`mmpp.MMPP` - Main interface class for simulation management
- {class}`mmpp.ZarrJobResult` - Individual simulation result wrapper
- {class}`mmpp.BatchOperations` - Batch operations for multiple results

### FFT Analysis
- {class}`mmpp.fft.FFT` - FFT analysis interface
- {class}`mmpp.fft.FMRModeAnalyzer` - FMR mode analysis
- {class}`mmpp.fft.FFTAnalyzer` - Core FFT computation engine

### Plotting & Visualization  
- {class}`mmpp.plotting.MMPPlotter` - Main plotting interface
- {class}`mmpp.plotting.PlotterProxy` - Proxy for result collections

### Simulation Management
- {class}`mmpp.simulation.SimulationManager` - Simulation execution and management
