# Batch Operations API

Batch interfaces returned by `MMPP.__getitem__(slice)` (for example `job[:]`).

```{eval-rst}
.. automodule:: mmpp.batch_operations
   :members:
   :undoc-members:
   :show-inheritance:
```

## Main Classes

```{eval-rst}
.. autoclass:: mmpp.batch_operations.BatchOperations
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: mmpp.batch_operations.BatchFFT
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: mmpp.batch_operations.BatchModeAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
```

## Typical Usage

```python
batch = job[:]

# modes
summary = batch.fft.modes.compute_modes(parallel=True, max_workers=4)

# spectrum (batch)
spec_batch = batch.fft.spectrum.compute_all(extract_parameters=["B0", "d"])

# transmission (batch)
trans_batch = batch.fft.transmission.compute_all(extract_parameters=["B0", "d"])
```
