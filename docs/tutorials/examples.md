# Practical Recipes

## Overlay Spectra for Multiple Jobs

```python
batch = job[:20]

multi = batch.fft.spectrum.overlay(
    find_peaks={"min_prominence": 0.01}
)

fig, ax, peaks = multi.plot(freq_unit="GHz", log_scale=True)
```

## Save and Reload Batch Spectrum

```python
from mmpp.fft.spectrum_batch import BatchSpectrumResult

spec_batch = job[:].fft.spectrum.compute_all(
    extract_parameters=["B0", "d"],
    save_batch=True,
)

spec_batch.save("batch_spectrum.pkl")
loaded = BatchSpectrumResult.load("batch_spectrum.pkl")
```

## Batch Transmission Heatmap

```python
trans_batch = job[:].fft.transmission(
    spatial_window=120,
    extract_parameters=["B0", "d"],
    save_batch=True,
)

fig, ax = trans_batch.plot_transmission_crosssection_heatmap(
    swapping_parameter="B0",
    x=120,
    freq_unit="GHz",
)
```

## Dispersion with Advanced Color Normalization

```python
fig, ax = result.fft.dispersion.plot_dispersion(
    axis="x",
    f_units="GHz",
    colornorm="symlognorm",
    colornorm_kwargs={"linthresh": 1e-5},
    fmax=20,
)
```
