# MMPP

**MMPP (Micro Magnetic Post Processing)** is a Python library for post-processing
micromagnetic simulations stored in `.zarr` and HDF containers. It covers
metadata discovery, batch handling, lazy numerical loading, FFT/frequency-domain
analysis, mode extraction, dispersion workflows, transmission analysis, and hysteresis
post-processing.

This README is designed as a practical onboarding and reference document. It
contains both beginner and advanced examples and replaces outdated helper usage
with current, consistent APIs.

## Table of contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Opening and scanning results](#opening-and-scanning-results)
4. [Working with metadata tables](#working-with-metadata-tables)
5. [Filtering and selecting results](#filtering-and-selecting-results)
6. [Single result API (`ZarrJobResult`)](#single-result-api-zarrjobresult)
7. [Dataset wrappers (`DatasetAwareWrapper`)](#dataset-wrappers-datasetawarewrapper)
8. [Selecting data: `frame`, `sel`, slicing, downsampling](#selecting-data-frame-sel-slicing-downsampling)
9. [Converting data to NumPy](#converting-data-to-numpy)
10. [Batch workflows](#batch-workflows)
11. [Table access (`TableAwareWrapper`)](#table-access-tableawarewrapper)
12. [Plotting and analysis accessors](#plotting-and-analysis-accessors)
13. [FFT facade](#fft-facade)
14. [Spectrum](#spectrum)
15. [Modes](#modes)
16. [Dispersion](#dispersion)
17. [Transmission](#transmission)
18. [Hysteresis](#hysteresis)
19. [CLI](#cli)
20. [Environment helpers and optional dependencies](#environment-helpers-and-optional-dependencies)
21. [Migration notes: outdated helper cleanup](#migration-notes-outdated-helper-cleanup)
22. [Common pitfalls](#common-pitfalls)
23. [Performance and safety notes](#performance-and-safety-notes)

## Installation

```bash
pip install mmpp
```

The same project metadata works with `uv`:

```bash
uv add mmpp
uv add "mmpp[fft]"
```

For development from a checkout, use `uv sync --extra dev` or the equivalent
`python -m pip install -e ".[dev]"`. The supported Python range is 3.9–3.12.

Optional extras are available for optional workflows:

```bash
pip install mmpp[fft]         # FFT and spectrum support
pip install mmpp[plotting]    # plotting helpers
pip install mmpp[interactive] # notebook interactive tools
pip install mmpp[tui]         # terminal UI helpers
pip install mmpp[wavelets]    # wavelet tools
pip install mmpp[image]       # image helpers
pip install mmpp[ml]          # machine-learning helpers (selected)
pip install mmpp[dev]         # development/test/lint/type tools
```

CLI entrypoint is installed with the package:

```bash
mmpp --help
```

## Quick start

```python
import mmpp as mp

jobs = mp.open("/path/to/results")
print("jobs:", len(jobs))
print("available columns:", jobs.columns[:10])
print("dataframe:", type(jobs.df), jobs.df.shape)
print("first result:", jobs[0].name)
```

## Opening and scanning results

`mp.open(...)` creates an `MMPP` object and scans the provided location. The input
may be:

- a directory containing many simulation outputs
- a single `.zarr` result path
- a path that already points to one finished result dataset

```python
jobs = mp.open("/path/to/results")
jobs = mp.open("/path/to/results/sim_001.zarr")
jobs = mp.open("/path/to/results", force=True)  # force full rescan immediately
```

Scanning helpers:

```python
jobs.scan()             # scans only if not yet scanned
jobs.scan(force=True)   # explicit full scan now
jobs.force_rescan()     # alias for scan(force=True)

jobs.get_parsing_examples("/path/to/example_file.zarr")  # inspect parser heuristics
```

## Working with metadata tables

`jobs` behaves like a table-backed collection with metadata columns and result records.

```python
jobs.df                  # pandas.DataFrame metadata
jobs.dataframe           # same as jobs.df
jobs.columns             # metadata column names
jobs.base_path           # path used to construct this object
jobs.jobs                # raw list of result objects
jobs[0]                  # first result (`ZarrJobResult`)
jobs[:]                   # all results as BatchOperations
jobs[1:4]                # slice as batch
```

A quick check for the number of selected jobs:

```python
subset = jobs.find(Nx=256, Ny=256)
print(len(subset))
```

## Filtering and selecting results

`find(...)` and `find_paths(...)` apply **AND** logic across all criteria.

```python
sweep = jobs.find(Nx=256, Ny=256)
xy_pbc = jobs.find(PBCx=1, PBCy=1)
by_field = jobs.find(Bext=0.05)
by_many = jobs.find(solver=3, Nx=128, Ny=128)

paths = by_many.find_paths()
print(paths[:3])
```

Numeric columns use nearest-match behavior, so `Bext=0.0500001` selects nearest value
from metadata when exact match is not available.

## Single result API (`ZarrJobResult`)

```python
res = jobs.find(PBCx=1)[0]

res.path          # full dataset path
res.name          # short result name
res.attrs         # result attributes (zarr attrs mapping)
res.datasets      # top-level dataset names
res.keys()        # dataset/group keys available in root
res.list_datasets()  # recursive dataset listing
res.has_dataset("m")
res.has_attr("dx")
res.is_finished()
res.is_running()

res.mock_data     # small helper fixture-like view (when available)
res.script        # source mx3 or script metadata helper (if exposed)
```

You can also access datasets directly through attributes:

```python
res["m"]
res["table"]
res.m              # lazy wrapper for magnetization (if present)
res.table          # table wrapper, if table group exists
```

Raw zarr accessors and typed getters:

```python
res.get_raw("m")              # raw zarr.Array-like object
res.get_raw_data("m")         # eager numpy with source dtype
res.get_raw_f32("m")
res.get_raw_c64("modes/arr")

res.get_f32("m", (0, 10, slice(None), slice(None), slice(None)))
res.get_np1d("t", (slice(None),))
res.get_np2d("m", (slice(None), slice(None)))
res.get_np3d("m", (slice(None), slice(None), slice(None)))
res.get_np4d("m", (slice(None), slice(None), slice(None), slice(None)))
res.get_np5d("m", (slice(None), slice(None), slice(None), slice(None), slice(None)))
res.get_np4dc("modes/arr", (slice(None), slice(None), slice(None), slice(None)))
```

There is no generic `get_np(...)` helper; use `get_np1d`..`get_np5d` and suffixed
complex variants where available.

## Dataset wrappers (`DatasetAwareWrapper`)

Access dataset data lazily via attributes (`res.m`, `res.mx`, `res.my`, etc.
if present). Wrappers are cheap until materialized.

```python
m = res.m

print("analysis_shape:", m.analysis_shape)
print("numpy_shape:", m.numpy_shape)
print("shape:", m.shape)
print("is_lazy:", m.is_lazy)
print("is_materialized:", m.is_materialized)
print("estimated_nbytes:", m.estimated_nbytes)
print("keys:", m.keys())
```

Useful aliases:

```python
m.array
m.values
m.np           # immediate numpy through wrapper
m.np[...]      # chained indexing into materialized array
```

Materialization:

```python
x1 = m.to_numpy()                    # immediate ndarray
x2 = m.numpy()                       # immediate ndarray
x3 = m.to_numpy(dtype="float32", copy=False)
x4 = m.numpy(dtype="float32", keepdims=True)
```

## Selecting data: `frame`, `sel`, slicing, downsampling

### Slicing and indexing

```python
# positional and full-slice style
m_t0 = m[0]              # first time slice
m_tz = m[0:50, 0, ...]  # first 50 times + z layer
```

### `frame(...)`

`frame` selects by axis values in order where supported by dimensions.

```python
roi = m.frame(t=0, z=0, y=(0, 128), x=(0, 256))
```

### `sel(...)`

`sel` selects using physical coordinates when those axes carry coordinate metadata.

```python
roi_physical = m.sel(
    x=(0.0, 25e-9),
    y=(5e-9, 10e-9),
)
``` 

### Downsampling

```python
coarse = m.downsample(":", ":", 128, 128, ":")
coarse_strict = m.downsample(":", 300, 128, 64, ":", strict=True)
```

- default `strict=False` allows trimming on incompatible dimensions
- `strict=True` raises if downsample factors are incompatible with shape

Dataset wrappers also support direct materialized views from `jobs[:].get` for batching.

## Converting data to NumPy

There are two conceptually different paths:

- **lazy pipeline**: chaining and filtering stays lazy until materialization (`DatasetAwareWrapper`).
- **immediate path**: `res.get[...]` returns immediately materialized data.

```python
# wrapper path
a = res.m.to_numpy()
res.m[:10, ...].to_numpy(dtype="float32")
res.m.numpy(copy=False)

# immediate getter path
full_np = res.get.m[:]                    # always ndarray
slice_np = res.get.m[0:100, ..., 0]       # direct ndarray
layer_np = res.get["m_layer13"][:]

# conversion from lazy chain
view_np = (res.m[0:100, ..., 0]).to_numpy(dtype="float32")
```

## Batch workflows

`jobs[:]` and any filtered result set return a batch-style API.

```python
batch = jobs[:]                 # all results
pc_batch = jobs.find(PBCx=1, PBCy=1)
first_three = pc_batch[0:3]

print(len(pc_batch))
print(type(pc_batch))
```

Batch getter and FFT helpers:

```python
# eager materialization over all results
stack = pc_batch.get.m[:]         # shape: [n_jobs, ...]
stack_small = pc_batch.get.m[0:100, :, :, :, 0]

pc_batch.fft.compute_all()        # compute FFT pipeline for all results
specs = pc_batch.fft.spectrum.compute_all(dset="m", fmin=5e9, fmax=25e9)
specs.plot_heatmap(parameter="Bext")

pc_batch.fft.modes.compute_modes()
pc_batch.fft.modes.analyze_all()

pc_batch.fft.transmission.compute_all()
```

Legacy-compatible mixed forms are still supported in many places:

```python
pc_batch.m_layer13[:10].fft.transmission()
pc_batch["m_layer13"][:10].fft.spectrum()
```

Batching for HDF5/zarr table-like metadata is also available through wrapper methods
on table-oriented batches when present.

## Table access (`TableAwareWrapper`)

If a result exposes a `table` group, use the table wrapper for metadata columns,
quick summaries, and plotting.

```python
if hasattr(res, "table"):
    t = res.table
    print(t.columns)
    print(t.n_rows)
    print(t.shape)

    preview = t.preview(n=10, columns=["t", "mx", "my"])
    df = t.to_dataframe(columns=["t", "mx", "my"], max_rows=1000)
    fig = t.plot(x="t", y=["mx", "my"], kind="line")
    # optional interactive chart if optional plotting backend is available
    t.interactive(show=True)
```

## Plotting and analysis accessors

Result and dataset wrappers expose convenience sub-accessors:

```python
m = res.m

m.plot
m.analyze
m.fft
m.solitons
m.vortex

# table example already above
```

Use these as entry points for rich methods from their respective namespaces.

## FFT facade

FFT on a single result:

```python
spectrum = res.fft.spectrum()
print(spectrum.frequencies.shape)
print(spectrum.spectrum.shape)
print(spectrum.power[:3, :3])
```

You can also call direct spectrum helpers:

```python
freqs = res.fft.frequencies()
power = res.fft.power()
mag = res.fft.magnitude()
phase = res.fft.phase()
```

Plot quick helpers:

```python
res.fft.plot_spectrum(log_scale=True)
spectrum.plot.spectrum(log_scale=True)
spectrum.plot.power(log_scale=False)
```

## Spectrum

`SpectrumResult` exposes explicit fields and aliases.

```python
spec = res.fft.spectrum()

print(spec.frequencies)
print(spec.frequency)
print(spec.freqs)
print(spec.spectrum)
print(spec.data)
print(spec.power)
print(spec.magnitude)
print(spec.amplitude)
print(spec.phase)
print(spec.spectral_quantity)
print(spec.power_quantity)
print(spec.spectral_quantity_label)
```

Plot helpers on spectra:

```python
fig = spec.plot.spectrum()
fig = spec.plot.power()
fig = spec.plot.magnitude()
fig = spec.plot.phase()
fig = spec.plot.modes(freq=9.5)
```

## Modes

Modes are accessed from FFT namespace on the result:

```python
modes_iface = res.fft.modes
modes_res = modes_iface.compute_modes()
fig = modes_iface.plot_modes(frequency=9.5)
viewer = modes_iface.interactive_spectrum(dpi=140)
```

Legacy-compatible mode entry points are still available where present:

```python
res_modes = res.fft.modes.compute_modes()
res_modes = res.fft.modes.analyze()
```

## Dispersion

Canonical user-facing path is dataset-first and interactive:

```python
viewer = res.m.fft.dispersion.plot.interactive()
```

Programmatic compute paths are available too:

```python
result_1d = res.fft.dispersion.configure(
    component="perp",
    time_window="hann",
    filter_type="cosine"
).compute_1d(axis="x")

result_2d = res.fft.dispersion.compute_2d(component="mz")

result_1d.plot.heatmap()
res.fft.dispersion.plot_dispersion(axis="x", fmax=30)
```

If batch compute is needed:

```python
pc_batch.fft.dispersion.compute_all(axis="x")
```

## Transmission

```python
tx = res.fft.transmission(save=True)
fig, ax = tx.plot_transmission()

# explicit cache control
_tx2 = res.fft.transmission(
    save=True,
    cache_path="/tmp/fft_cache",
    force=True,
)

pc_batch.fft.transmission.compute_all(save=True)
```

## Hysteresis

Access hysteresis analysis through `analyze.hysteresis`.

```python
ha = res.analyze.hysteresis

ha_from_table = ha.from_table(field="B_extx", magnetization="mx")
ha_from_magnetization = ha.from_magnetization(dset="m", component="y", z_layer=0)
ha_from_arrays = ha.from_arrays(field=[1, 2, 3], magnetization=[0.1, 0.2, 0.3])
ha_from_keys = ha.from_zarr_keys(key_prefix="B", component="x")

ha_from_table.plot.loop(field="B_extx", magnetization="mx")
ha_from_table.plot.interactive(show_hc=True)
ha_from_table.plot.animation()
```

Result object fields often include:

```python
metrics = ha_from_table.metrics
comp = ha_from_table.compare(other_hysteresis)
ha_from_table.export("/tmp/loop.csv")
```

## CLI

`mmpp` ships with a command suite useful for running/inspecting jobs.

```bash
mmpp --version
mmpp --help
mmpp info

# authentication and server handling
mmpp auth login
mmpp auth status
mmpp auth logout

# job discovery and run helpers
mmpp jobs list
mmpp jobs list --server <alias-or-url>

mmpp run my_job.mx3
mmpp run "test*.mx3" --detach --time 10h --cpus 16 --memory 64 --gpus 1
mmpp run status
mmpp run check

# swap utility group
mmpp swap init
mmpp swap info
mmpp swap validate
mmpp swap run
```

## Environment helpers and optional dependencies

Runtime diagnostics for optional modules:

```python
status = mp.check_dependencies()
print(status)
print("core:", status["core"]["available"])
print("fft:", status["fft"]["available"])

mp.install_ffmpeg(verbose=True)
```

Use this to check what is available in headless or CI environments.

## Migration notes: outdated helper cleanup

Use these as the **canonical, supported examples**:

- `mp.open(...)` is the canonical entry point.
- `mp.mmpp(...)` does **not** exist.
- Prefer batch-aware paths (`jobs[:]`, `jobs.find(...)`, `jobs[:].get`) over ad-hoc manual loops.
- Prefer dataset wrappers (`res.m`, `res.get[...]`, `res.fft`, `res.analyze`) over repeatedly touching raw `zarr` internals.
- For downsampling and selection, use wrapper APIs (`frame`, `sel`, `downsample`, slicing) to retain axis metadata.

Outdated examples to avoid:

```python
# Do not use
mp.mmpp(path)
res.get_np("m")
```

Supported replacements:

```python
mp.open(path)
res.get_np1d("t", (slice(None),))
res.get_np2d("m", ...)
# etc.
```

Also note:

- `jobs.fft` on multiple results may warn and use the first result in legacy context.
  Use `jobs[:].fft` for explicit batch execution.
- If you need deterministic result selection, chain `find` filters and work on the exact
  returned batch/result object.

## Common pitfalls

- `jobs.find` numeric filters are nearest-match, not strict binary-match equality.
- `res.get[...]` and `batch.get[...]` materialize immediately.
- `DatasetAwareWrapper` methods remain lazy by default.
- `downsample(..., strict=False)` can trim edges; strict mode protects shape assumptions.
- `res.m.as_zarr()` works only for non-materialized, unsliced wrappers.
- `jobs.find_paths(...)` returns filesystem paths, not result objects.
- `show=False` on interactive views should be used for CI/headless workflows.

## Performance and safety notes

- Prefer narrow selection first (`find(...)`) then operations:
  smaller batches and fewer computations.
- Use `cache`, `save`, `force` arguments where provided to control recomputation.
- For very large results, avoid eager calls (`to_numpy`) on whole arrays unless needed.
- Keep plotting/interactive workflows in notebook-aware code paths and use headless mode for
  scripted checks.

## End-to-end example

This script demonstrates a realistic flow: scan, filter, inspect, downsample, FFT,
and visualization-oriented steps.

```python
import mmpp as mp
import numpy as np

# 1) discover results
jobs = mp.open("/path/to/results")
print(jobs.columns)

# 2) filter by metadata
subset = jobs.find(Nx=256, Ny=256, PBCx=1, PBCy=1)
print("selected:", len(subset))

# 3) pick first result
res = subset[0]
print("result:", res.name, res.path)
print("attrs keys:", list(res.attrs)[:5])

# 4) inspect mesh and simulation grid metadata
print("shape:", res.m.shape)
print("keys:", res.keys()[:10])

# 5) select region
roi = res.m.frame(t=(0, 64), z=0, y=(0, 128), x=(0, 128))
roi_down = roi.downsample(":", 4, 4, 4, ":")
arr = roi_down.to_numpy(dtype="float32")

# 6) spectral analysis
auto = res.fft.spectrum()                    # quick single-result spectrum
heat = auto.plot

# frequency slice + direct numeric access
freqs = auto.frequencies
s = auto.spectrum
print("frequency bins:", freqs.shape, "spectrum shape:", s.shape)

# 7) mode and dispersion entry points
modes = res.fft.modes
print(modes)

disp = res.m.fft.dispersion.plot.interactive(show=False)
print("dispersion viewer created in headless mode:", disp)

# 8) transmission
tx = res.fft.transmission(save=True)
fig, _ = tx.plot_transmission()

# 9) hysteresis helpers
ha = res.analyze.hysteresis.from_table(field="B_extx", magnetization="mx")
print("hysteresis points:", len(ha.data) if hasattr(ha, "data") else "n/a")

# 10) batch equivalent - compute FFT spectrum for first three matching jobs
batch = subset[:3].fft.spectrum.compute_all(dset="m", fmax=25e9)
print("batch specs:", len(batch))
```

## Contributing and maintainability notes

The library supports practical workflows across Python 3.9+ and documents interfaces in
`mmpp/api` and `docs/`. If you contribute examples, prefer:

- one concise path per snippet,
- explicit parameter names,
- a small amount of defensive checks (e.g., shape and state assertions),
- compatibility with optional dependencies where needed.
