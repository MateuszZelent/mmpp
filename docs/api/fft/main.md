# FFT Main Module

```{eval-rst}
.. automodule:: mmpp.fft.main
   :members:
   :undoc-members:
   :show-inheritance:
```

## Main Classes

### FFTAnalyzer

```{eval-rst}
.. autoclass:: mmpp.fft.main.FFTAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

### FFTResult

```{eval-rst}
.. autoclass:: mmpp.fft.main.FFTResult
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Basic FFT Analysis

```python
import mmpp

# Open dataset
db = mmpp.open('/path/to/data')
results = db.find(solver=3)

# Get first result and analyze FFT
analyzer = results[0].fft.main.FFTAnalyzer()
fft_result = analyzer.analyze_single(0, 'm_z11')

print(f"Frequency range: {fft_result.frequencies[0]:.2f} - {fft_result.frequencies[-1]:.2f} GHz")
print(f"Peak frequency: {fft_result.frequencies[fft_result.power.argmax()]:.2f} GHz")
```

### Batch FFT Analysis

```python
# Analyze multiple datasets
results = analyzer.analyze_all('m_z11')
for i, result in enumerate(results):
    peak_freq = result.frequencies[result.power.argmax()]
    print(f"Layer {i}: Peak at {peak_freq:.2f} GHz")
```
