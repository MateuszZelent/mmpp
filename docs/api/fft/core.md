# FFT Core API

Core user-facing FFT interface (`result.fft`) and spectrum result objects.

```{eval-rst}
.. automodule:: mmpp.fft.core
   :members:
   :undoc-members:
   :show-inheritance:
```

## Key Classes

```{eval-rst}
.. autoclass:: mmpp.fft.core.FFT
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: mmpp.fft.core.SpectrumResult
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: mmpp.fft.core.MultiSpectrumResult
   :members:
   :undoc-members:
   :show-inheritance:
```

## Minimal Example

```python
result = job[0]
spec = result.fft.spectrum(dset=result.get_largest_m_dataset())
fig, ax, peaks = spec.plot_spectrum(freq_unit="GHz")
```
