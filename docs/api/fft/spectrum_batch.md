# Batch Spectrum API

Batch FFT spectrum computation and visualization.

```{eval-rst}
.. automodule:: mmpp.fft.spectrum_batch
   :members:
   :undoc-members:
   :show-inheritance:
```

## Main Classes

```{eval-rst}
.. autoclass:: mmpp.fft.spectrum_batch.BatchSpectrum
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: mmpp.fft.spectrum_batch.BatchSpectrumResult
   :members:
   :undoc-members:
```

## Typical Calls

```python
res = job[:].m_layer13[:800, ..., 0:1].fft.spectrum(
    extract_parameters=["B0", "d"],
    fmin=1e9,
    fmax=25e9,
)

fig, ax = res.plot_heatmap(parameter="B0", freq_unit="GHz")
```
