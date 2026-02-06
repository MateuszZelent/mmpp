# Transmission API

Transmission interfaces for single-result and batch workflows.

```{eval-rst}
.. automodule:: mmpp.fft.transmission.interface
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: mmpp.fft.transmission.batch
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: mmpp.fft.transmission.compute
   :members:
   :undoc-members:
   :show-inheritance:
```

## Typical Calls

```python
# single result
trans = result.fft.transmission(spatial_window=120)
fig, ax, image = trans.plot_transmission()

# batch
batch_trans = job[:].m_layer13[:800, ..., 0:1].fft.transmission(
    spatial_window=120,
    extract_parameters=["B0", "d"],
)
fig, ax = batch_trans.plot_transmission_crosssection_heatmap(
    swapping_parameter="B0",
    x=120,
)
```
