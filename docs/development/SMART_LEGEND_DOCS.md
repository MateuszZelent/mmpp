# Smart Legend Feature for mmpp Plotting

## Overview
The mmpp plotting functionality now includes **smart legend** capability that automatically detects which parameters vary between datasets and shows only those parameters in the legend, making plots much cleaner and more informative.

## How It Works

### Before (Old Behavior)
```python
results = jobs.find(solver=(3), amp_values=0.0022)
results.plot(x_series='t', y_series="m_z11", average=(1,2,3), comp='z')
```
**Old legend would show:** `solver=3    , f0=1.00e+09, maxerr=1.00e-06, Nx=64`
- Includes constant parameters like `solver=3` (same for all datasets)
- Cluttered with information that doesn't help distinguish between curves

### After (New Smart Legend)
```python
results = jobs.find(solver=(3), amp_values=0.0022)
results.plot(x_series='t', y_series="m_z11", average=(1,2,3), comp='z')
```
**Smart legend shows only:** `f0=1.00e+09, maxerr=1.00e-06, Nx=64`
- Automatically excludes constant parameters (`solver`, `amp_values`, etc.)
- Shows only parameters that actually differ between datasets
- Much cleaner and more informative

## Key Features

### 1. Automatic Parameter Detection
- Analyzes all datasets to determine which parameters vary
- Excludes parameters that are constant across all datasets
- Handles floating-point precision issues automatically

### 2. Smart Formatting
- Scientific notation for small/large values (`f0=1.23e+09`, `maxerr=1.23e-06`)
- Integer formatting for grid sizes (`Nx=64`, `solver=3`)
- Customizable precision per parameter type

### 3. Priority-Based Ordering
Parameters shown in order of importance:
1. `solver` (if varying)
2. `f0` (frequency)
3. `maxerr` (error tolerance)
4. `dt` (time step)
5. `Nx`, `Ny`, `Nz` (grid sizes)
6. `PBCx`, `PBCy`, `PBCz` (boundary conditions)
7. `amp_values` (amplitude)
8. Other parameters alphabetically

### 4. Configurable Limits
```python
# Limit number of parameters in legend
plotter.configure(max_legend_params=3)

# Disable smart legend (use old behavior)
plotter.configure(sort_results=False)
```

## Configuration Options

```python
results.configure(
    max_legend_params=4,    # Maximum parameters to show in legend
    sort_results=True,      # Enable smart legend and sorting
    style='paper',          # Plotting style
    use_custom_fonts=True   # Use Arial fonts
)
```

## Technical Implementation

### Core Methods
- `_get_varying_parameters()`: Analyzes datasets to find varying parameters
- `_format_result_label()`: Creates formatted labels with proper precision
- `_sort_results_by_parameters()`: Sorts datasets by parameter values

### Type Safety
- Full mypy compliance with proper type annotations
- Handles edge cases (single dataset, missing parameters, etc.)

## Examples

### Scenario 1: Mixed Parameters
```python
# Some parameters vary, some don't
results = jobs.find(solver=(1,2,3), amp_values=0.001)
# Legend shows: solver=1, f0=1.00e+09, maxerr=1.00e-06
# Hidden: amp_values=0.001 (constant)
```

### Scenario 2: Grid Size Studies
```python
# Grid convergence study
results = jobs.find(f0=1e9, Nx=(32,64,128))
# Legend shows: Nx=32, Nx=64, Nx=128
# Hidden: f0=1.00e+09 (constant)
```

### Scenario 3: Parameter Sweeps
```python
# Frequency sweep with fixed other parameters
results = jobs.find(solver=3, f0=(1e9, 2e9, 3e9))
# Legend shows: f0=1.00e+09, f0=2.00e+09, f0=3.00e+09  
# Hidden: solver=3 (constant)
```

## Backward Compatibility

The plotting is available directly on the results object:
```python
# Correct way to plot
results.plot('t', 'm_z11')
```

## Testing

Test files are available to verify functionality:
- `test_smart_legend.py` - Basic functionality test
- `test_comprehensive.py` - Complete scenario testing
- `example_smart_legend.py` - Usage examples

Run tests with:
```bash
python test_comprehensive.py
```

This smart legend feature makes mmpp plots significantly more readable and professional, especially when comparing multiple simulation results with different parameters.
