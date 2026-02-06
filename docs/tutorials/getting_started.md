# Getting Started

## Installation

```bash
pip install mmpp
```

For development/docs:

```bash
pip install -e .[dev]
```

## 1. Open Simulations

```python
import mmpp as mp

job = mp.open("/path/to/simulations")
print(len(job))
print(job.columns)
```

`mmpp.open(...)` scans `.zarr` directories and builds/loads a metadata index.

## 2. Filter by Metadata

```python
subset = job.find(B0=0.12, d=150e-9)
print(len(subset))

result = subset[0]
print(result.path)
```

For numeric columns, `find()` uses nearest value matching.

## 3. Inspect Datasets in One Result

```python
print(result.datasets)
print(result.get_largest_m_dataset())

# quick tree view
result.pp
```

## 4. Run a First FFT Spectrum

```python
dset = result.get_largest_m_dataset()
spec = result.fft.spectrum(dset=dset, tmax=800)

freqs, complex_spec = spec
power = spec.power

fig, ax, peaks = spec.plot_spectrum(freq_unit="GHz", log_scale=True)
```

## 5. Access Modes, Dispersion, Transmission

```python
# FMR mode workflow
result.fft.modes.interactive_spectrum(dpi=140)

# Dispersion S(k,f)
fig, ax = result.fft.dispersion.plot_dispersion(axis="x", f_units="GHz")

# Transmission
trans = result.fft.transmission(spatial_window=120)
fig, ax, image = trans.plot_transmission()
```

## 6. Batch Operations

```python
batch = job[:]

summary = batch.fft.modes.compute_modes(parallel=True, max_workers=4)
print(summary["successful"], summary["failed"])
```

`job[:]` is the main entry point for processing many simulations together.
