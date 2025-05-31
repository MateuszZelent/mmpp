# FFT Computation Module

```{eval-rst}
.. automodule:: mmpp.fft.compute_fft
   :members:
   :undoc-members:
   :show-inheritance:
```

## Main Classes

### FFTCompute

```{eval-rst}
.. autoclass:: mmpp.fft.compute_fft.FFTCompute
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

### FFTComputeResult

```{eval-rst}
.. autoclass:: mmpp.fft.compute_fft.FFTComputeResult
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

### FFTConfig

```{eval-rst}
.. autoclass:: mmpp.fft.compute_fft.FFTConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Basic FFT Computation

```python
import mmpp

# Setup FFT computation
db = mmpp.open('/path/to/data')
result = db.find(solver=3)[0]

# Compute FFT with custom parameters
fft_result = result.fft.spectrum(
    dset='m_z11',
    z_layer=-1,
    method=1,
    save=True,
    force=False
)

# Access computed data
frequencies = result.fft.frequencies()
power_spectrum = result.fft.power()
magnitude = result.fft.magnitude()
phase = result.fft.phase()
```

### Advanced Configuration

```python
# Custom FFT configuration
from mmpp.fft.compute_fft import FFTConfig

config = FFTConfig(
    window_function='hann',
    fft_engine='scipy',
    zero_padding=True,
    nfft=1024
)

# Use configuration
fft_result = result.fft._compute_fft(
    dataset_name='m_z11',
    z_layer=-1,
    method=1,
    config=config
)
```
