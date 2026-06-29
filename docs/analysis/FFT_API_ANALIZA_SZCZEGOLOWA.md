# Detailed FFT API Analysis

## Introduction

MMPP FFT functionality offers spectrum computation and FMR mode tooling through the `result.fft` facade and mode sub-accessor. This document summarizes the current implementation contract and usage patterns in English.

## Core API Surface

- `result.fft.spectrum(dset='m_z11', z_layer=-1, method=1, save=False, force=False, ...)`
- `result.fft.frequencies(dset='m_z11', ...)`
- `result.fft.power(dset='m_z11', ...)`
- `result.fft.magnitude(dset='m_z11', ...)`
- `result.fft.phase(dset='m_z11', ...)`
- `result.fft.plot_spectrum(dset='m_z11', log_scale=True, normalize=False, save=True, force=False, ...)`
- `result.fft.clear_cache()`

## Modes API Surface

- `result.fft.modes.find_peaks(threshold=...)`
- `result.fft.modes.interactive_spectrum(components=['x', 'y', 'z'], z_layer=0, ...)`
- `result.fft.modes.plot_modes(frequency=..., z_layer=0, ...)`
- `result.fft.modes.compute_modes(save=False, force=False)`

## Documentation Quality Notes

- Method descriptions are present and map to concrete implementations in source files.
- `help(...)` output and runtime inspection are consistent with expected parameters.
- Output of `print(result.fft)` is implementation-driven and serves as inline discovery for notebook workflows.

## Quick examples

```python
import mmpp

result = mmpp.open('/path/to/data.zarr')[0]
print(result.fft)

freqs = result.fft.frequencies('m_z11')
power = result.fft.power('m_z11')
peak_idx = power.argmax()
print('Peak [GHz]:', freqs[peak_idx] / 1e9)

fig, ax = result.fft.plot_spectrum(dset='m_z11', log_scale=True)
fig, ax = result.fft.plot_spectrum(save=True, force=True)

fig_modes, axes = result.fft.modes.plot_modes(frequency=freqs[peak_idx], z_layer=0)
```

## Validation Focus

- argument defaults and accepted ranges are mirrored between docs and code
- save and cache flags are used consistently
- mode flow handles dataset, z-layer, and threshold parameters predictably

## Conclusion

The FFT API documentation is valid at this stage of the tree and already suitable as user-facing reference for core users. Keep it updated when legacy helper signatures change.
