# Mode Visualization API (Legacy Guide)

This page is kept for continuity after the FFT/dispersion refactor.

## Recommended Current Guides

- FMR spectrum and mode panels: `tutorials/fft_spectrum_analysis`
- Dispersion and folded mode extraction: `tutorials/dispersion_analysis`

## Current Primary Interfaces

- `result.fft.modes`
  - `.interactive_spectrum(...)`
  - `.plot_modes(...)`
- `result.fft.dispersion`
  - `.compute_1d(...)`
  - `.plot_dispersion(...)`
  - `.dispersion_modes(...)`

## Minimal Example

```python
# FMR modes
result.fft.modes.interactive_spectrum(dpi=140)
result.fft.modes.plot_modes(frequency=9.6, component="mz")

# Dispersion-mode extraction
res = result.fft.dispersion.compute_1d(axis="x", avg_over_orthogonal=False)
modes = result.fft.dispersion.dispersion_modes(result=res, lattice_constant_nm=470)
mode = modes.mode(k=2.3, f=1.1)
mode.plot(mode_type="abs")
```
