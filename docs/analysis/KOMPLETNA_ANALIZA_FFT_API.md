# Comprehensive FFT API Review (March 2025)

## Purpose

This document provides a compact English version of the full FFT API audit, covering behavior, consistency, and API quality.

## Scope

- Dataset-facing entry points in `mmpp/fft/core.py`
- Mode helpers under `mmpp/fft/modes.py`
- Plotting contract in `mmpp/fft/plot.py`
- Batch interface in `mmpp/batch_operations.py`

## Findings

1. The self-documenting FFT object (`print(result.fft)`) accurately reflects the supported method set.
2. Core FFT methods and plotting helpers are present and aligned with expected signatures.
3. `help(method)` output contains complete parameter and return documentation.
4. Batch wrappers expose the same semantics as single-result workflows.
5. Cache and save/force behavior is explicit in both docs and implementation.

## Reference Mapping

- Core functions live in `mmpp/fft/core.py`.
- Mode functionality lives in `mmpp/fft/modes.py`.
- Plot helper utilities live in `mmpp/fft/plot.py`.
- Batch operations are assembled through `mmpp/batch_operations.py`.

## Practical Contract

```python
import mmpp
result = mmpp.open('/path/to/data.zarr')[0]

# FFT
freqs, spectrum = result.fft.spectrum('m_z11', z_layer=-1, method=1, save=True)

# Mode workflow
peaks = result.fft.modes.find_peaks(threshold=0.1)
if peaks:
    fig, axes = result.fft.modes.plot_modes(frequency=peaks[0].freq)

# Plot
fig, ax = result.fft.plot_spectrum(dset='m_z11', log_scale=True)
```

## Recommendation

The FFT docs should remain synchronized with API evolution toward fluent object-based accessors and deprecation messaging for legacy pathways.
