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
disp = result.m.fft.dispersion

# Main compute-and-view workflow. Without tmax or a dataset time slice, all
# available timesteps are used. Dataset slices are preserved, so [:100, ...]
# limits the time axis before FFT. modes=True requests S_complex for mode work.
viewer = disp.plot.interactive(
    axis="x",
    component="perp",
    fmax=25,
    analytical="DE",
    model="kalinikos",
    modes=True,
    lattice_constant_nm=470,
    show=False,
)

# For a bounded compute window, use explicit sample indices:
preview = result.m[:100, ...].fft.dispersion.plot.interactive(show=False)
window = disp.plot.interactive(tmin=100, tmax=300, show=False)

# Long-running notebook cells report progress by default before the heavy FFT
# starts. Use progress=False for quiet batch runs or progress_callback=events.append
# to collect structured events.
events = []
viewer = disp.plot.interactive(axis="x", show=False, progress_callback=events.append)

# Compute once and reuse the result for static plots, filters, and headless UI.
res = disp.compute_1d(
    axis="x",
    component="perp",
    avg_over_orthogonal=False,
    store_complex=True,
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
viewer = disp.plot.interactive(
    axis="x",
    component="perp",
    avg_over_orthogonal=False,
    store_complex=True,
    modes=True,
    lattice_constant_nm=470,
    show=False,
)
mode = viewer.mode_at_selection(k_rad_um=2.3, f_ghz=1.1, component="z")
mode_viewer = mode.plot.interactive(show=False, mode_type="abs")
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
notebooks should prefer `disp.plot.interactive(modes=True, ...)`.

```python
modes = disp.dispersion_modes(result=res, lattice_constant_nm=470)
modes.plot_interactive()
```

Migration:

```python
# Old
modes = disp.dispersion_modes(result=res, lattice_constant_nm=470)
modes.plot_interactive()

# New
viewer = disp.plot.interactive(modes=True, lattice_constant_nm=470)
mode = viewer.mode_at_selection(k_rad_um=2.3, f_ghz=1.1)
```
