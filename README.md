# MMPP

**MMPP (Micro Magnetic Post Processing)** is a Python library for post-processing results of micromagnetic simulations (`.zarr` and HDF-backed data).
It provides a practical layer for:

- scanning result folders,
- filtering metadata and running batch workflows,
- working with lazily loaded datasets,
- safe conversion to `NumPy`,
- frequency-domain analysis (`FFT`, spectrum, modes, dispersion, transmission),
- hysteresis analysis,
- CLI operations for submitting and monitoring jobs.

This document is a practical guide to the public API.

## Table of contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Core API: opening and finding results](#core-api-open-and-find-results)
4. [Scanning and refreshing metadata](#scanning-and-refreshing-metadata)
5. [Filtering (`find`, `find_paths`)](#filtering-find-find_paths)
6. [Working with a single result](#working-with-a-single-result)
7. [Data and dataset wrappers](#data-and-dataset-wrappers)
8. [Selecting and sampling data (`frame`, `sel`, `downsample`)](#selecting-and-sampling-data-frame-sel-downsample)
9. [Converting to `NumPy`](#converting-to-numpy)
10. [Batch workflow (`BatchOperations`)](#batch-workflow-batchoperations)
11. [Result tables (`TableAwareWrapper`)](#result-tables-tableawarewrapper)
12. [FFT / spectrum / modes / dispersion](#fft--spectrum--modes--dispersion)
13. [Transmission](#transmission)
14. [Hysteresis](#hysteresis)
15. [CLI](#cli)
16. [Environment helper tools](#environment-helper-tools)
17. [Common pitfalls](#common-pitfalls)

## Installation

```bash
pip install mmpp
```

Optional extras:

```bash
pip install mmpp[fft]         # spectral analysis
pip install mmpp[plotting]    # advanced plotting
pip install mmpp[interactive] # notebook widgets
pip install mmpp[tui]         # terminal UI
pip install mmpp[wavelets]    # wavelet workflows
pip install mmpp[image]       # image helpers
pip install mmpp[ml]          # ML helpers (selected use cases)
pip install mmpp[full]        # all extras above + optional extras
pip install mmpp[dev]         # development dependencies (mypy, pytest, docs, etc.)
```

After installation, the CLI is available:

```bash
mmpp --help
```

## Quick start

```python
import mmpp as mp

# Main entry point
jobs = mp.open("/path/to/results", max_workers=16)

# Opening a single .zarr file (same as a single-job MMPP object)
single = mp.open("/path/to/results/sim_001.zarr")

# Force a full scan at initialization time
jobs = mp.open("/path/to/results", force=True)

print(len(jobs))
print(jobs.columns[:5])
print(jobs.base_path)
```

## Core API: opening and finding results

`mp.open(...)` returns an `MMPP` instance.
`mp.mmpp(...)` is **not** an alias and does not exist.

```python
jobs = mp.open("/path/to/results")
jobs.base_path       # base directory / file
jobs.dataframe       # alias to jobs.df
jobs.df             # pandas.DataFrame with metadata
jobs.columns        # available metadata columns
len(jobs)           # number of discovered results
jobs[0]             # single result (ZarrJobResult)
jobs[:]             # BatchOperations for all results
jobs[1:10]          # BatchOperations for a slice
jobs[:].jobs         # in batch context, convenience alias to records
```

Note: when there are multiple results, `jobs.fft` uses the first one and emits a warning.
For batch analysis use `jobs[:].fft`, and for single-result analysis use `jobs[i].fft`.

## Scanning and refreshing metadata

Scanning runs automatically during `jobs` creation, but you can trigger it manually:

```python
jobs.scan()             # scans only when DF is empty
jobs.scan(force=True)   # full scan
jobs.force_rescan()     # direct alias for full scan

jobs.get_parsing_examples("/path/to/example_file.zarr")
jobs.scan()             # idempotent on empty DF
```

## Filtering (`find`, `find_paths`)

```python
subset = jobs.find(Nx=256, Ny=256, solver=3)
subset = jobs.find(PBCx=1, PBCy=1)
subset = jobs.find(Bext=0.0500001)      # nearest-match behavior for numeric columns
subset = jobs.find(alpha=0.02, PBCx=1)

paths = subset.find_paths()
paths = jobs.find(Nx=256).find_paths(PBCx=1)
```

`find()` and `find_paths()` apply logical AND across all provided keys.

## Working with a single result

```python
res = jobs.find(PBCx=1, PBCy=1)[0]

res.path                  # path to .zarr
res.name                  # result name
res.attrs                 # zarr attributes as mapping
res.datasets              # top-level datasets
res.list_datasets()       # full dataset list (recursive)
res.has_dataset("m")      # dataset exists
res.has_attr("dx")        # attribute exists
res.is_finished()         # whether simulation is finished
res.is_running()          # whether simulation is still running
res.get_largest_m_dataset()# heuristic pick of largest m dataset
res[res.keys()[0]]        # direct raw access
```

Most convenient access goes through dataset attributes:

```python
raw_m = res.m               # alias: DatasetAwareWrapper (lazy)
raw_tbl = res["table"]      # dataset/group "table" as a zarr object
```

Raw-value methods:

```python
res.get_raw("m")                        # raw zarr.Array (or array when slicing)
res.get_raw_data("m")                   # np.ndarray with source dtype
res.get_raw_f32("m")                    # np.float32
res.get_raw_c64("modes/arr")            # np.complex64

res.get_f32("m", (0, 10, slice(None), slice(None), slice(None)))
res.get_c64("modes/arr", slice(0, 10))

res.get_np1d("t", (slice(None),))
res.get_np2d("m", (slice(None), slice(None)))
res.get_np3d("m", (slice(None), slice(None), slice(None)))
res.get_np4d("m", (slice(None), slice(None), slice(None), slice(None)))
res.get_np5d("m", (slice(None), slice(None), slice(None), slice(None), slice(None)))
res.get_np4dc("modes/arr", (slice(None), slice(None), slice(None), slice(None)))
```

There is no `get_np(...)` method without a suffix.

## Data and dataset wrappers

`res.m` is a `DatasetAwareWrapper`:

```python
arr = res.m
arr.analysis_shape
arr.numpy_shape
arr.shape
arr.is_lazy
arr.is_materialized
arr.keys()
arr.array                 # np.ndarray alias
arr.values                # np.ndarray alias
arr.np                    # immediate np.ndarray getter for chained indexing
```

`numpy()` and `to_numpy()` are style variants that both materialize data:

```python
arr = res.m
arr2 = arr.to_numpy(copy=False)
arr3 = arr.numpy(dtype="float32", keepdims=True)
```

'to_numpy(copy=False)' exposes data immediately in memory; by default, analysis dimensions are cast in an approximate way as expected by existing behavior.

## Selecting and sampling data (`frame`, `sel`, `downsample`)

```python
view = res.m.frame(t=0, z=0, y=(0, 128), x=(0, 256))
roi = res.m.sel(x=(0.0, 25e-9), y=(5e-9, 10e-9))

coarse = res.m.downsample(":", ":", 128, 128, ":")
coarse_strict = res.m.downsample(":", 300, 128, 64, ":", strict=True)
```

## Converting to `NumPy`

```python
# Materializing wrapper (returns ndarray)
arr1 = res.m.to_numpy()
arr2 = res.m.numpy()
arr3 = res.m.to_numpy(dtype="float32")

# Direct access via get (returns immediate np.ndarray)
arr4 = res.get.m[:]                    # full dataset
arr5 = res.get.m[0:100, ..., 0]        # slicing via get
arr6 = res.get["m_layer13"][:]         # key-based get access
```

`res.get[...]` builds arrays immediately and always returns standard `np.ndarray`.

## Batch workflow (`BatchOperations`)

```python
all_batch = jobs[:]
pc_batch = jobs.find(PBCx=1, PBCy=1)
first_three = pc_batch[0:3]
len(pc_batch)
```

In batch mode, dataset operations are available through `pc_batch.<dataset>` and `pc_batch.get`:

```python
stack = pc_batch.get.m[:]  # shape: [n_jobs, ...]
stack = pc_batch.get.m[0:100, :, :, :, 0]

pc_batch.fft.compute_all()                                # FFT for all results
spectra = pc_batch.fft.spectrum.compute_all(dset="m", fmin=5e9, fmax=25e9)
spectra.plot_heatmap(parameter="Bext")                    # if parameter axis is variable

pc_batch.fft.modes.compute_modes()                        # batch mode analysis
pc_batch.fft.modes.analyze_all()                          # analyze all results

pc_batch.fft.transmission.compute_all()
pc_batch.m_layer13[:10].fft.transmission()               # dataset-aware usage
pc_batch["m_layer13"][:10].fft.spectrum()                # equivalent by key
```

## Result tables (`TableAwareWrapper`)

The table wrapper is available when a run contains a `table` group:

```python
if hasattr(res, "table"):
    t = res.table
    t.columns
    t.n_rows
    t.shape
    t_preview = t.preview(n=10, columns=["t", "mx", "my"])
    df = t.to_dataframe(columns=["t", "mx", "my"], max_rows=1000)
    fig = t.plot(x="t", y=["mx", "my"], kind="line")
    t.interactive(show=True)
```

## FFT / spectrum / modes / dispersion

### FFT + spectrum

```python
spec = res.fft.spectrum()                  # SpectrumResult
spec.frequencies
spec.spectrum
spec.power                               # |FFT|^2
spec.magnitude                           # |FFT|
spec.phase                               # arg(FFT)

res.fft.frequencies()
res.fft.power()
res.fft.magnitude()
res.fft.phase()
res.fft.plot_spectrum(log_scale=True)
spec.plot.spectrum(log_scale=True)
```

```python
res.fft.spectrum.plot.interactive()        # interactive browser if dependencies are available
res.fft.plot_spectrum()                    # quick plotting shortcut
```

### Modes

```python
res_modes = res.fft.modes.compute_modes()       # compute/recompute modes
fig_modes = res.fft.modes.plot_modes(frequency=9.5)
viewer = res.fft.modes.interactive_spectrum(dpi=140)  # legacy helper
```

### Dispersion

```python
res_disp1 = res.fft.dispersion.configure(
    component="perp",
    time_window="hann",
).compute_1d(axis="x")

res_disp2 = res.fft.dispersion.compute_2d(component="mz")

res_disp1.plot.heatmap()
res.fft.dispersion.plot_dispersion(axis="x", fmax=30)
```

## Transmission

```python
tx = res.fft.transmission(save=True)
fig, ax = tx.plot_transmission()

# Manual cache for repeated runs if needed
tx2 = res.fft.transmission(
    save=True,
    cache_path="/tmp/fft_cache",
    force=True,
)
```

Batch variant:

```python
pc_batch.fft.transmission.compute_all(save=True)
```

## Hysteresis

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

## CLI

```bash
mmpp --version
mmpp --help
mmpp info
```

Authentication:

```bash
mmpp auth login
mmpp auth status
mmpp auth logout
```

Jobs:

```bash
mmpp jobs list
mmpp jobs list --server <alias-or-url>
```

Running simulations:

```bash
mmpp run my_job.mx3
mmpp run "test*.mx3" --detach --time 10h --cpus 16 --memory 64 --gpus 1
mmpp run status
mmpp run check
```

Swap tool:

```bash
mmpp swap init
mmpp swap info
mmpp swap validate
mmpp swap run
```

## Environment helper tools

```python
status = mp.check_dependencies()
print(status["core"]["available"])
print(status["fft"]["available"])

mp.install_ffmpeg(verbose=True)
```

## Common pitfalls

- `mp.mmpp(...)` does not exist. Use `mp.open(...)`.
- `res.get_np(...)` does not exist. Use `get_np1d`, `get_np2d`, `get_np3d`, `get_np4d`, `get_np5d`, `get_np4dc`.
- For numeric columns, `jobs.find(...)` returns the closest value, not exact equality.
- `res.get(...)` and `jobs[:].get` return immediate `numpy` results; `DatasetAwareWrapper` remains lazy until materialized.
- `jobs.fft` with multiple results runs on the first record and warns; use `jobs[:].fft` for batch.
- `downsample(..., strict=False)` (default) may crop at boundaries when the division is not exact; set `strict=True` to error on mismatched dimensions.
- `jobs.find_paths(...)` returns a list of paths, not result objects.
