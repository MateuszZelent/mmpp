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

`DispersionResult2D` and `compute_2d()` are experimental. They expose
`S(kx, ky, f)` and `slice_1d(...)`, but do not yet provide the complete 1D
raw/display/cache/scaling contract.

```{eval-rst}
.. autoclass:: mmpp.fft.dispersion.models.DispersionBranch
   :members:
```

## Typical Calls

```python
disp = result.fft.dispersion

# Lightweight compute-and-view workflow. This does not store S_complex unless
# explicitly requested, so it is the preferred notebook preview path.
viewer = disp.plot.interactive(
    axis="x",
    component="perp",
    fmax=25,
    show=False,
)

# Compute once and reuse the result for static plots, filters, and headless UI.
res = disp.compute_1d(
    axis="x",
    component="perp",
    store_complex=False,
    scaling="amplitude_squared",
)
viewer = res.plot.interactive(show=False, fmax=25)
fig, ax = res.plot.heatmap(f_units="GHz", fmax=25)
```

`DispersionInteractiveViewer.state` and `export_selection()` include
`result_notes`, so sampling warnings and raw/display semantics travel with
headless notebook presets and exported selections. They also normalize common
NumPy scalars and arrays to JSON-safe Python values.

## Mode Reconstruction

Mode reconstruction needs complex spectra and orthogonal spatial context. Use
`store_complex=True` and `avg_over_orthogonal=False` when you want to extract
spatial mode profiles.

```python
res = disp.compute_1d(
    axis="x",
    component="perp",
    avg_over_orthogonal=False,
    store_complex=True,
    scaling="amplitude_squared",
)
modes = res.modes.interactive(show=False, lattice_constant_nm=470)
mode = res.modes.at(k_rad_um=2.3, f_ghz=1.1)
mode_viewer = mode.plot.interactive(show=False, mode_type="abs")
animation = res.modes.plot.animation(peaks=[0], show=False)
```

`DispersionResult1D.S_raw` is the analysis source. `S_display` is the current
display view after live/post filters, and `result.S` remains a
backward-compatible alias for the active display array. Use
`analysis_source="display"` only when that is intentional. When
`avg_over_orthogonal=False` stores local spectra, `S_local_raw` and
`S_local_display` carry the same raw/display contract for each orthogonal slice;
`S_local` remains the backward-compatible active-display alias.

## FFT Backends

Dispersion FFTs can use NumPy, SciPy, or pyFFTW when installed. For reproducible
CI and notebooks, set the backend and worker count explicitly:

```bash
MMPP_FFT_BACKEND=numpy MMPP_FFT_WORKERS=1 python -m pytest tests/test_dispersion_mode_extraction.py -q -k backend
```

```python
disp = result.fft.dispersion(backend="scipy", workers=1)
res = disp.compute_1d(axis="x", component="perp")
```

`MMPP_FFT_BACKEND` accepts `numpy`, `scipy`, or `pyfftw`. `pyfftw` is optional;
tests skip it when the package is missing. `MMPP_FFT_WORKERS=1` is recommended
for deterministic CI gates; `-1` means all available cores.

## Legacy Folded-Mode Workflow

The older folded-mode entry point remains available for compatibility, but new
notebooks should prefer `disp.plot.interactive(...)` and
`res.modes.interactive(...)`.

```python
modes = disp.dispersion_modes(result=res, lattice_constant_nm=470)
modes.plot_interactive()
```
