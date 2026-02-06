# API Reference

Reference for the currently supported public interfaces.

```{toctree}
:maxdepth: 2

core
batch_operations
plotting
fft/index
simulation
logging_config
```

## Main Entry Points

- `mmpp.open(...)` -> create `MMPP` scanner/indexer
- `MMPP.find(...)` -> metadata-driven selection
- `ZarrJobResult.fft` -> FFT/FMR/dispersion/transmission on one job
- `MMPP[:]` -> `BatchOperations` for multi-job execution
