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
# Dataset-first notebook workflow. Slices on the dataset, e.g. [:100, ...],
# are preserved by the FFT/dispersion pipeline.
disp = result.m[:100, ...].fft.dispersion

# Optional global config for this interface instance
disp.configure(
    dx=5e-9,
    dt=1e-12,
    component="perp",
    tmax=800,
)

# Main interactive explorer: dispersion heatmap, analytical overlay, selected
# point export, and mode extraction in one window.
viewer = disp.plot.interactive(
    axis="x",
    component="perp",
    fmax=25,
    analytical="DE",
    model="kalinikos",
    modes=True,
    lattice_constant_nm=470,
    cache="/tmp/mmpp-dispersion",
    show=False,
)

# Long-running notebook cells show stage progress by default. Disable with
# progress=False or collect events with progress_callback=events.append.

# Compute explicit result for reuse when you want static plots or scripts.
res1d = disp.compute_1d(
    axis="x",
    avg_over_orthogonal=False,
    store_complex=True,
    scaling="amplitude_squared",  # or "raw_power" for legacy |FFT|^2, "psd" for density
    save=True,
)
fig, ax = res1d.plot.heatmap(kscale="rad_um", f_units="GHz", fmax=25)

# Old notebooks using dispersion_modes(...).plot_interactive() still work, but
# new notebooks should prefer disp.plot.interactive(modes=True, ...).
mode = viewer.mode_at_selection(k_rad_um=2.3, f_ghz=1.1, component="z")
mode.plot.imshow(mode_type="abs")
```

Benchmark the same compute path with synthetic data:

```bash
python scripts/analysis/benchmark_fft_dispersion.py \
  --profile small-ci \
  --backend scipy \
  --workers 1 \
  --output /tmp/mmpp-dispersion-benchmark.json
```

Available profiles are `small-ci`, `medium-dev`, and `research-reference`.
The JSON report includes elapsed time, peak memory, result array sizes, and a
preflight memory breakdown for the FFT pipeline.

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

## FFT Refactor Status

Current refactor status:

- Phase 1 complete: shared FFT filtering infrastructure.
- Phase 2 started: refactored `SpectrumResult` moved to `mmpp/fft/spectrum/`.
- Phase 3 started: `SpectrumHelper` moved to `mmpp/fft/spectrum/helpers.py` and `core.py` reduced.
- Phase 4 started: added `mmpp/fft/spectrum/batch/` shim namespace for batch decomposition.
- Phase 4 progress: `SpectrumEntry` + `BatchSpectrumResult` moved to `mmpp/fft/spectrum/batch/result.py`.
- Phase 4 progress: `BatchSpectrum` moved to `mmpp/fft/spectrum/batch/compute.py`.
- Phase 4 progress: plotting methods extracted to `mmpp/fft/spectrum/batch/plotting.py`.
- Phase 4 progress: `FFTCompute` internals split into helper modules:
  - `mmpp/fft/_compute_engines.py` (engine selection + FFT backends),
  - `mmpp/fft/_compute_methods.py` (method1/method2 execution + metadata),
  - `mmpp/fft/_compute_loading.py` (zarr loading + z-layer normalization + profiled loading metrics),
  - `mmpp/fft/_compute_cache.py` (cache load + parameter matching).

Phase 1 introduced:

- `mmpp/fft/filters/config.py`
- `mmpp/fft/filters/pipeline.py`
- `mmpp/fft/filters/preprocess.py`
- `mmpp/fft/filters/postprocess.py`
- `mmpp/fft/filters/windows.py`

Integrated modules now delegating to shared filters:

- `mmpp/fft/compute_fft.py` (window + pre-FFT filters),
- `mmpp/fft/modes/filter_utils.py` (compatibility shim),
- `mmpp/fft/modes/interactive.py` (post-filter wrappers + delegated toolbar helpers),
- `mmpp/fft/modes/_interactive/presets.py`, `mmpp/fft/modes/_interactive/controls.py`, `mmpp/fft/modes/_interactive/data.py`, `mmpp/fft/modes/_interactive/callbacks.py`, `mmpp/fft/modes/_interactive/widgets.py`, `mmpp/fft/modes/_interactive/rendering.py`, `mmpp/fft/modes/_interactive/interactions.py`, `mmpp/fft/modes/_interactive/mode_plots.py`, `mmpp/fft/modes/_interactive/mode_layout.py`, `mmpp/fft/modes/_interactive/status.py`, and `mmpp/fft/modes/_interactive/compat.py`,
- `mmpp/fft/spectrum/modes/bridge.py` and `mmpp/fft/spectrum/modes/accessor.py` (SpectrumResult -> modes bridge split),
- `mmpp/fft/dispersion/modes/_interactive/state.py`, `mmpp/fft/dispersion/modes/_interactive/presets.py`, `mmpp/fft/dispersion/modes/_interactive/callbacks.py`, `mmpp/fft/dispersion/modes/_interactive/layout.py`, and `mmpp/fft/dispersion/modes/_interactive/filters.py` (runtime state, presets, animation callbacks, layout, and filter-config builders extracted from `interactive.py`),
- `mmpp/fft/dispersion/_interface/k0_filtering.py` (shared smoothing path).

Planned next steps:

- move remaining `FFT` helper internals from `core.py` into `spectrum/*` modules,
- clean/trim legacy imports and typing debt in refactored batch modules.

## Documentation

Full documentation is in `docs/` and includes:

- getting started and architecture,
- FMR spectrum and mode workflows,
- dispersion workflows,
- batch workflows,
- API reference per module.
