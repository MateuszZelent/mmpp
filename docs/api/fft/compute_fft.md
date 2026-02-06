# FFT Compute Engine API

Low-level FFT compute backend used by higher-level interfaces.

```{eval-rst}
.. automodule:: mmpp.fft.compute_fft
   :members:
   :undoc-members:
   :show-inheritance:
```

## Main Data Structures

```{eval-rst}
.. autoclass:: mmpp.fft.compute_fft.FFTCompute
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: mmpp.fft.compute_fft.FFTComputeConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: mmpp.fft.compute_fft.FFTComputeResult
   :members:
   :undoc-members:
   :show-inheritance:
```

Use this module when you need backend-level tuning (`window_function`, `filter_type`, engine choices) beyond standard `result.fft.*` calls.
