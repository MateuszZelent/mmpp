# Dispersion API

Spin-wave dispersion interfaces, result models, and utilities.

```{eval-rst}
.. automodule:: mmpp.fft.dispersion
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: mmpp.fft.dispersion.interface
   :members:
   :undoc-members:
   :show-inheritance:
```

## Key Models

```{eval-rst}
.. autoclass:: mmpp.fft.dispersion.models.DispersionConfig
   :members:
```

```{eval-rst}
.. autoclass:: mmpp.fft.dispersion.models.DispersionResult1D
   :members:
```

```{eval-rst}
.. autoclass:: mmpp.fft.dispersion.models.DispersionResult2D
   :members:
```

```{eval-rst}
.. autoclass:: mmpp.fft.dispersion.models.DispersionBranch
   :members:
```

## Typical Calls

```python
disp = result.fft.dispersion
res = disp.compute_1d(axis="x", avg_over_orthogonal=False)
fig, ax = disp.plot_dispersion(axis="x", f_units="GHz")
modes = disp.dispersion_modes(result=res, lattice_constant_nm=470)
```
