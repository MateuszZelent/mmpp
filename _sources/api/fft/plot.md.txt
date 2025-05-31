# FFT Plotting Module

```{eval-rst}
.. automodule:: mmpp.fft.plot
   :members:
   :undoc-members:
   :show-inheritance:
```

## Main Classes

### FFTPlotter

```{eval-rst}
.. autoclass:: mmpp.fft.plot.FFTPlotter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Basic Spectrum Plotting

```python
import mmpp

# Load data and get FFT interface
db = mmpp.open('/path/to/data')
result = db.find(solver=3)[0]

# Plot power spectrum
fig, ax = result.fft.plot_spectrum(
    dset='m_z11',
    log_scale=True,
    normalize=False
)

# Customize plot
ax.set_title('FMR Power Spectrum')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Power (a.u.)')
```

### Advanced Plotting Options

```python
# Plot with custom styling
fig, ax = result.fft.plot_spectrum(
    dset='m_z11',
    method=1,
    z_layer=-1,
    log_scale=True,
    normalize=True,
    figsize=(12, 8),
    dpi=150,
    title='Normalized FMR Spectrum',
    xlabel='Frequency (GHz)',
    ylabel='Normalized Power'
)

# Add peak markers
frequencies = result.fft.frequencies()
power = result.fft.power()
peak_indices = np.where(power > 0.5 * power.max())[0]
ax.scatter(frequencies[peak_indices], power[peak_indices], 
          color='red', s=50, zorder=5, label='Peaks')
ax.legend()
```

### Multi-Dataset Comparison

```python
# Compare multiple datasets
datasets = ['m_x11', 'm_y11', 'm_z11']
fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4*len(datasets)))

for i, dset in enumerate(datasets):
    result.fft.plot_spectrum(
        dset=dset,
        ax=axes[i],
        title=f'Component: {dset}',
        log_scale=True
    )

plt.tight_layout()
```

### Batch Plotting

```python
# Plot spectra for multiple results
results = db.find(solver=3, limit=5)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, result in enumerate(results):
    if i < len(axes):
        result.fft.plot_spectrum(
            ax=axes[i],
            title=f'Result {i+1}',
            log_scale=True
        )

plt.tight_layout()
```
