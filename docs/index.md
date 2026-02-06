# MMPP Documentation

`mmpp` is a micromagnetic post-processing library focused on real workflows:

- indexing large directories of `.zarr` simulation outputs,
- metadata-based filtering (`find`),
- FFT / FMR spectrum and mode analysis,
- spin-wave dispersion analysis,
- transmission analysis,
- scalable batch processing with cache-aware execution.

## Who This Is For

This documentation is intended for users who already generate micromagnetic simulations and want a reliable Python workflow for analysis and visualization.

## Documentation Map

```{toctree}
:maxdepth: 2
:caption: Contents

tutorials/index
api/index
```

## Fast Entry Points

- New users: start with `tutorials/getting_started`.
- FMR spectrum and modes: `tutorials/fft_spectrum_analysis`.
- Dispersion and folded-mode extraction: `tutorials/dispersion_analysis`.
- Batch processing across many jobs: `tutorials/batch_operations`.
- Complete API details: `api/index`.
