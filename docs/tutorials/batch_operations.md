# Batch Operations

Batch workflows start from `job[:]`, which returns `BatchOperations`.

```python
batch = job[:]
print(len(batch))
```

## 1. Batch Mode Computation

```python
summary = batch.fft.modes.compute_modes(
    dset="m",
    parallel=True,
    max_workers=4,
)

print(summary["successful"], summary["failed"])
print(summary["total_time"])
```

## 2. High-Level Batch Process

```python
report = batch.process(
    dset="m",
    parallel=True,
    max_workers=4,
)

print(report["successful"], report["failed"])
```

## 3. Batch Spectrum

### Explicit call

```python
spec_batch = batch.fft.spectrum.compute_all(
    dataset_name="m_layer13",
    slice_info=(slice(0, 800), Ellipsis, slice(0, 1)),
    extract_parameters=["B0", "d", "p"],
    fmin=1e9,
    fmax=25e9,
    parallel=True,
    save=True,
)
```

### Dataset-aware fluent call

```python
spec_batch = batch.m_layer13[:800, ..., 0:1].fft.spectrum(
    extract_parameters=["B0", "d", "p"],
    fmin=1e9,
    fmax=25e9,
)
```

### Plot and inspect

```python
spec_batch.show_parameters()

entry0 = spec_batch[0]
fig, ax = entry0.plot(freq_unit="GHz")

fig, ax = spec_batch.plot_heatmap(
    parameter="B0",
    freq_unit="GHz",
    fmax=25,
)
```

## 4. Batch Transmission

```python
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

## 5. Caching Strategy

- per-result cache:
  - `save=True`
  - `use_cache=True`
  - `force=True` to invalidate and recompute
- whole-batch cache:
  - `save_batch=True`
  - optional `batch_cache_dir="..."`

For repeated parameter sweeps, batch cache usually gives the biggest speedup.
