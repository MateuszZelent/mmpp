# Dispersion Analysis

`result.fft.dispersion` provides spin-wave dispersion workflows: `compute_1d`,
`plot.interactive`, static heatmaps, filtering, branch tracking, and folded-mode
extraction.

## Basic Plot-First Workflow

```python
disp = result.fft.dispersion

fig, ax = disp.plot_dispersion(
    axis="x",
    component="perp",
    kscale="rad_um",
    f_units="GHz",
    fmax=25,
)
```

## Compute and Reuse `DispersionResult1D`

```python
res1d = disp.compute_1d(
    axis="x",
    component="perp",
    avg_over_orthogonal=False,
    scaling="amplitude_squared",
    save=True,
    disk_cache=True,
)

print(res1d.shape)
print(res1d.k_range, res1d.f_range)
```

## Interactive Preview

Use `disp.plot.interactive(...)` for compute-and-view notebooks, or
`res1d.plot.interactive(...)` when you already have a result. `show=False`
returns a headless controller that can be tested, saved as a preset, or displayed
later with `.show()`.

```python
viewer = disp.plot.interactive(
    axis="x",
    component="perp",
    fmax=25,
    show=False,
)

viewer = res1d.plot.interactive(show=False, fmax=25)
viewer.save_preset("/tmp/mmpp-dispersion-viewer.json")
```

The viewer state includes `result_notes`; use it to surface sampling warnings
or raw/display notes in notebook exports without recomputing the FFT.

## Raw and Display Spectra

`DispersionResult1D.S_raw` is the analysis source used by sampling and branch
tracking. `S_display` is the current display view after filters, and `result.S`
remains a backward-compatible alias for the active display array. Use
`analysis_source="display"` only when you intentionally want analysis on a
filtered visual view. When `avg_over_orthogonal=False` stores local spectra,
use `S_local_raw` for analysis-grade local slices and `S_local_display` for
filtered visual local slices; `S_local` remains the active-display alias.

## Configure Once

```python
disp.configure(
    dx=5e-9,
    dt=1e-12,
    component="perp",
    tmax=800,
)
```

For reproducible runs, pin the FFT backend and workers either in code or through
environment variables:

```python
disp = result.fft.dispersion(backend="numpy", workers=1)
```

```bash
MMPP_FFT_BACKEND=numpy MMPP_FFT_WORKERS=1 python notebook_smoke.py
```

## Filtering Pipeline

```python
disp.filters(
    remove_static=True,
    window="both",
    post={
        "snr_filter": {"enabled": True, "threshold_snr": 3.0},
    },
    live={
        "gaussian_morph": {"enabled": True, "sigma_f": 1.0, "sigma_k": 1.0},
    },
)

fig, ax = disp.plot_dispersion(axis="x", live_filters={"live": {"gaussian_morph": {"enabled": True}}})
```

## Branch Tracking

```python
import numpy as np

k_path = np.linspace(res1d.k_axis.min(), res1d.k_axis.max(), 300)
branch = disp.track_branch(res1d, k_path=k_path, f_seed=8e9)

fig, (ax1, ax2) = disp.plot_branch(branch, f_units="GHz")
```

## Mode Reconstruction

Spatial mode reconstruction needs the complex spectrum and orthogonal spatial
context. Recompute with `store_complex=True` and `avg_over_orthogonal=False`
before using `.modes.at(...)` or mode-ready interactive panels.

```python
mode_ready = disp.compute_1d(
    axis="x",
    component="perp",
    avg_over_orthogonal=False,
    store_complex=True,
    scaling="amplitude_squared",
)

modes = mode_ready.modes.interactive(
    show=False,
    lattice_constant_nm=470,
)

mode = mode_ready.modes.at(k_rad_um=2.3, f_ghz=1.1)
mode.plot.imshow(mode_type="abs")
mode_viewer = mode.plot.interactive(show=False, mode_type="phase")
animation = mode_ready.modes.plot.animation(peaks=[0], show=False)
```

## Legacy Folded-Mode Workflow

`dispersion_modes(...).plot_interactive()` remains available for older
notebooks, but new code should prefer `disp.plot.interactive(...)` for spectrum
exploration and `res1d.modes.interactive(...)` for mode workflows.

```python
modes = disp.dispersion_modes(result=res1d, lattice_constant_nm=470)
modes.plot_interactive()

mode = modes.mode(k=2.3, f=1.1)
mode.plot(mode_type="abs")
```

## 2D Dispersion

`compute_2d()` is experimental. Use it when you need full `S(kx, ky, f)`, but
do not assume the full 1D raw/display/cache contract yet.

```python
res2d = disp.compute_2d(component="perp")
print(res2d.shape)
slice_x = res2d.slice_1d("kx", k_value=0.0)
```

Use `compute_2d()` when you need full `S(kx, ky, f)` instead of one propagation axis.
