# Batch Operations Module

```{eval-rst}
.. automodule:: mmpp.batch_operations
   :members:
   :undoc-members:
   :show-inheritance:
```

## Batch Operations Classes

### BatchOperations

```{eval-rst}
.. autoclass:: mmpp.batch_operations.BatchOperations
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __len__, __iter__, __repr__
```

### BatchFFT

```{eval-rst}
.. autoclass:: mmpp.batch_operations.BatchFFT
   :members:
   :undoc-members:
   :show-inheritance:
```

### BatchModeAnalyzer

```{eval-rst}
.. autoclass:: mmpp.batch_operations.BatchModeAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

### Basic Batch Operations

```python
import mmpp as mp

# Load simulations
op = mp.open("/path/to/simulations")

# Get batch operations for all results
batch = op[:]

# Compute modes for all results
summary = batch.fft.modes.compute_modes(dset="m_z5-8")

print(f"Computed modes for {summary['successful']} results")
```

### Parallel Processing

```python
# Use parallel processing for faster computation
summary = batch.fft.modes.compute_modes(
    dset="m_z5-8",
    parallel=True,
    max_workers=4
)

print(f"Total time: {summary['total_time']:.2f}s")
print(f"Average per result: {summary['average_time_per_result']:.2f}s")
```

### Comprehensive Reports

```python
# Generate comprehensive analysis report
report = batch.prepare_report(
    spectrum=True,
    modes=True,
    parallel=True
)

print(f"Analyzed {report['total_results']} results")
```
