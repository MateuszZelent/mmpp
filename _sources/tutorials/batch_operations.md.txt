# Batch Operations Tutorial

## Overview

MMPP provides powerful batch processing capabilities that allow you to perform operations on entire directories of simulation results. This is particularly useful when you have many simulation outputs and want to process them all at once.

## Basic Usage

### Getting All Results

To work with all results in a directory, use slice notation:

```python
import mmpp

# Load all results from a directory
op = mmpp.MMPP('path/to/results')
all_results = op[:]  # Returns BatchOperations instance
```

### FFT Analysis on Multiple Results

```python
# Perform FFT analysis on all results
batch_fft = all_results.fft
fft_results = batch_fft.compute_spectrum(
    dset='m_z5-8',
    parallel=True,
    progress=True
)

print(f"Processed {len(fft_results)} results")
for result in fft_results:
    if result['success']:
        print(f"✓ {result['file']}: {result['result'].shape}")
    else:
        print(f"✗ {result['file']}: {result['error']}")
```

### Mode Analysis

```python
# Compute FMR modes for all results
mode_results = all_results.fft.modes.compute_modes(
    dset='m_z5-8',
    parallel=True,
    max_workers=4
)

# Filter successful results
successful_modes = [r for r in mode_results if r['success']]
print(f"Successfully computed modes for {len(successful_modes)} files")
```

## Advanced Features

### Custom Processing

You can define custom processing functions:

```python
def custom_analysis(zarr_result):
    """Custom analysis function"""
    # Your custom processing logic here
    return {"custom_metric": some_value}

# Apply custom function to all results
custom_results = all_results.apply(custom_analysis, parallel=True)
```

### Error Handling

The batch operations provide comprehensive error handling:

```python
results = all_results.fft.compute_spectrum('m_z5-8')

# Check for errors
errors = [r for r in results if not r['success']]
if errors:
    print(f"Found {len(errors)} errors:")
    for error in errors:
        print(f"  {error['file']}: {error['error']}")
```

### Progress Tracking

All batch operations support progress tracking:

```python
# Enable progress bars
results = all_results.fft.modes.compute_modes(
    dset='m_z5-8',
    progress=True,  # Shows progress bar
    parallel=True,
    max_workers=8
)
```

## Best Practices

1. **Use parallel processing** for large datasets with `parallel=True`
2. **Enable progress tracking** with `progress=True` for long operations  
3. **Handle errors gracefully** by checking the `success` field in results
4. **Limit workers** with `max_workers` to avoid overwhelming your system
5. **Use appropriate datasets** - make sure the dataset name exists in your files

## Example: Complete Workflow

```python
import mmpp

# Load data
op = mmpp.MMPP('simulation_results/')
batch = op[:]

print(f"Found {len(batch.zarr_results)} simulation files")

# Compute FFT spectra
print("Computing FFT spectra...")
spectra = batch.fft.compute_spectrum(
    dset='m_z5-8',
    parallel=True,
    progress=True,
    max_workers=6
)

# Compute modes
print("Computing FMR modes...")  
modes = batch.fft.modes.compute_modes(
    dset='m_z5-8', 
    parallel=True,
    progress=True,
    max_workers=6
)

# Generate summary
successful_spectra = sum(1 for r in spectra if r['success'])
successful_modes = sum(1 for r in modes if r['success'])

print(f"\\nResults Summary:")
print(f"  Spectra computed: {successful_spectra}/{len(spectra)}")
print(f"  Modes computed: {successful_modes}/{len(modes)}")

# Save results for later analysis
import pickle
with open('batch_results.pkl', 'wb') as f:
    pickle.dump({
        'spectra': spectra,
        'modes': modes
    }, f)
```

This workflow demonstrates the power of batch operations for processing large simulation datasets efficiently.
