# FMR Spectrum and Modes

This guide covers the current FFT/FMR workflow around `result.fft`.

## SpectrumResult Basics

```python
result = job[0]
dset = result.get_largest_m_dataset()

spec = result.fft.spectrum(
    dset=dset,
    tmin=0,
    tmax=1000,
    fmin=1e9,
    fmax=30e9,
    find_peaks={"min_prominence": 0.02},
)

freqs, complex_spec = spec
power = spec.power
magnitude = spec.magnitude
```

`spec` is a `SpectrumResult` object with tuple-like behavior and plotting helpers.

## Plot Spectrum

```python
fig, ax, peaks = spec.plot_spectrum(
    freq_unit="GHz",
    log_scale=True,
    normalize=False,
    show_peaks=True,
    title="FMR spectrum",
)
```

## Direct FFT Helpers

```python
freqs = result.fft.frequencies(dset=dset)
power = result.fft.power(dset=dset)
phase = result.fft.phase(dset=dset)
mag = result.fft.magnitude(dset=dset)
```

## Dataset-Aware Slicing

When component/time slicing is important, use dataset wrappers:

```python
# Keep only first 800 timesteps and selected component
spec_my = result.m_layer13[:800, ..., 1].fft.spectrum()
fig, ax, peaks = spec_my.plot_spectrum(freq_unit="GHz")
```

## Interactive FMR Modes

```python
# Full interactive panel
result.fft.modes.interactive_spectrum(
    dpi=150,
    log_scale=True,
    show_peaks=True,
)
```

## Static Mode at Selected Frequency

```python
fig = result.fft.modes.plot_modes(
    frequency=9.6,  # GHz
    component="mz",
    z_layer=-1,
    dpi=120,
)
```

## Convenience Through FFT Root

```python
# aliases through result.fft
result.fft.plot_modes(frequency=9.6, dset=dset)
result.fft.interactive_spectrum(dset=dset)
```

Use `result.fft.modes` for most new workflows; it carries dataset/slice context more reliably.
