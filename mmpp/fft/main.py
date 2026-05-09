# This module has been removed.
#
# The legacy FFTAnalyzer, FFTConfig, and FFTResult classes that lived here
# were superseded by the modular FFT pipeline:
#
#   - FFTCompute / FFTComputeConfig / FFTComputeResult  (compute_fft.py)
#   - FFT._spectrum_impl / SpectrumResult               (core.py, spectrum/)
#
# If you were importing from this module, migrate to:
#
#   from mmpp.fft.compute_fft import FFTCompute, FFTComputeResult
#   # or simply:  job[i].fft.spectrum(...)
