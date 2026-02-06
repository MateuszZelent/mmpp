# FFT API

FFT/FMR ecosystem in `mmpp`, including single-result and batch interfaces.

```{toctree}
:maxdepth: 2

core
compute_fft
modes
dispersion
spectrum_batch
transmission
plot
electromagnetic_analysis
main
```

## Main Objects

- `mmpp.fft.core.FFT`: single-result FFT entry point (`result.fft`)
- `mmpp.fft.core.SpectrumResult`: result object from `fft.spectrum(...)`
- `mmpp.fft.modes.interface.FFTModeInterfaceNew`: FMR mode interface
- `mmpp.fft.dispersion.interface.FFTDispersionInterface`: dispersion workflows
- `mmpp.fft.spectrum_batch.BatchSpectrum`: batch spectrum compute and plotting
- `mmpp.fft.transmission.interface.FFTTransmissionInterface`: transmission for one result
- `mmpp.fft.transmission.batch.BatchTransmission`: transmission for many results
