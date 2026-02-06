# MMPP

Micro Magnetic Post Processing (`mmpp`) is a Python library for scanning, filtering, and analyzing micromagnetic simulation results stored in `.zarr` containers.

It provides:

- fast simulation discovery and metadata indexing,
- FFT / FMR spectrum analysis,
- mode visualization,
- spin-wave dispersion analysis,
- transmission analysis,
- batch processing with caching.

Live documentation:

- https://mateuszzelent.github.io/mmpp/

## Installation

```bash
pip install mmpp
```

For development (tests, docs, linting):

```bash
pip install -e .[dev]
```

## Quick Start

```python
import mmpp as mp

# Open directory with many *.zarr results
job = mp.open("/path/to/simulations")

print(len(job))         # number of discovered results
print(job.columns[:10]) # available metadata columns for filtering

# Filter by metadata (numeric fields use nearest match)
subset = job.find(B0=0.12, d=150e-9)
result = subset[0]

# Inspect available datasets inside one zarr result
print(result.datasets)
print(result.get_largest_m_dataset())
```

## FFT / FMR Spectrum

```python
result = job[0]

# Auto-select best magnetization dataset
dataset = result.get_largest_m_dataset()

# SpectrumResult: supports tuple unpacking and fluent plotting
spec = result.fft.spectrum(
    dset=dataset,
    tmin=0,
    tmax=800,
    find_peaks={"min_prominence": 0.02},
    fmin=1e9,
    fmax=30e9,
)

freqs, spectrum = spec
power = spec.power

fig, ax, peaks = spec.plot_spectrum(
    freq_unit="GHz",
    log_scale=True,
    show_peaks=True,
)
```

### FMR Modes

```python
# Interactive spectrum + mode panels
result.fft.modes.interactive_spectrum(dpi=140)

# Static mode visualization at a selected frequency [GHz]
fig = result.fft.modes.plot_modes(
    frequency=9.6,
    component="mz",
    z_layer=-1,
)
```

## Dispersion (S(k, f))

```python
disp = result.fft.dispersion

# Optional global config for this interface instance
disp.configure(
    dx=5e-9,
    dt=1e-12,
    component="perp",
    tmax=800,
)

# Compute explicit result for reuse
res1d = disp.compute_1d(
    axis="x",
    avg_over_orthogonal=False,
    save=True,
)

# Plot using the same cached result path
fig, ax = disp.plot_dispersion(
    axis="x",
    kscale="rad_um",
    f_units="GHz",
    fmax=25,
)

# Brillouin-zone folding + interactive mode extraction
modes = disp.dispersion_modes(result=res1d, lattice_constant_nm=470)
modes.plot_interactive()

mode = modes.mode(k=2.3, f=1.1)
mode.plot(mode_type="abs")
```

## Transmission

```python
# Single-result transmission map
trans = result.fft.transmission(
    spatial_window=120,
    spatial_step=2,
    normalize="reference",
    save=True,
)

fig, ax, image = trans.plot_transmission()
```

## Batch Processing

```python
batch = job[:]

# 1) Batch modes
mode_summary = batch.fft.modes.compute_modes(
    dset="m",
    parallel=True,
    max_workers=4,
)

# 2) Batch spectrum with dataset/slice context and parameter extraction
spec_batch = batch.m_layer13[:800, ..., 0:1].fft.spectrum(
    extract_parameters=["B0", "d", "p"],
    fmin=1e9,
    fmax=25e9,
    parallel=True,
    save=True,
)

spec_batch.show_parameters()
fig, ax = spec_batch.plot_heatmap(parameter="B0", freq_unit="GHz", fmax=25)

# 3) Batch transmission
trans_batch = batch.m_layer13[:800, ..., 0:1].fft.transmission(
    spatial_window=120,
    extract_parameters=["B0", "d", "p"],
    parallel=True,
    save_batch=True,
)

fig, ax = trans_batch.plot_transmission_crosssection_heatmap(
    swapping_parameter="B0",
    x=120,
    freq_unit="GHz",
)
```

## Caching Notes

- `result.fft.*` uses in-memory cache during a session.
- `save=True` stores per-result outputs to zarr cache groups.
- `batch.fft.spectrum(..., save_batch=True)` and `batch.fft.transmission(..., save_batch=True)` store hash-keyed batch cache files.
- `force=True` recomputes and overwrites matching cache entries.

## Main API Surface

- opening and scanning:
  - `mmpp.open(...)`
  - `MMPP.scan()`, `MMPP.force_rescan()`, `MMPP.find(...)`
- per-result access:
  - `ZarrJobResult.datasets`, `ZarrJobResult.get_largest_m_dataset()`
  - `ZarrJobResult.fft`, `ZarrJobResult.mpl`
- FFT:
  - `FFT.spectrum`, `FFT.frequencies`, `FFT.power`, `FFT.phase`, `FFT.magnitude`
  - `FFT.modes`, `FFT.dispersion`, `FFT.transmission`
- batch:
  - `BatchOperations.fft.modes.compute_modes(...)`
  - `BatchOperations.fft.spectrum.compute_all(...)` / `BatchOperations.fft.spectrum(...)`
  - `BatchOperations.fft.transmission.compute_all(...)` / `BatchOperations.fft.transmission(...)`

## Documentation

Full documentation is in `docs/` and includes:

- getting started and architecture,
- FMR spectrum and mode workflows,
- dispersion workflows,
- batch workflows,
- API reference per module.
