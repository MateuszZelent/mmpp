# API Reference

Reference for the currently supported public interfaces.

```{toctree}
:maxdepth: 2

core
batch_operations
plotting
solitons
vortex_thiele
fft/index
analyze
simulation
logging_config
```

## Main Entry Points

- `mmpp.open(...)` -> create `MMPP` scanner/indexer
- `MMPP.find(...)` -> metadata-driven selection
- `ZarrJobResult.fft` -> FFT/FMR/dispersion/transmission on one job
- `ZarrJobResult.solitons.skyrmion` -> skyrmion topology and size analysis
- `MMPP[:]` -> `BatchOperations` for multi-job execution
