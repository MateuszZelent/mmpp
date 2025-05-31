# Examples

## Basic FFT Analysis

```python
import mmpp
import matplotlib.pyplot as plt

# Load simulation data
result = mmpp.MMPP('simulation.zarr')[0]

# Compute FFT spectrum
fft = result.fft
spectrum = fft.spectrum(dset='m_z5-8')

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot spectrum
fft.plot_spectrum(ax=ax1)
ax1.set_title('FFT Spectrum')

# Plot time domain data
time_data = result.root['m_z5-8']
ax2.plot(time_data)
ax2.set_title('Time Domain Signal')
ax2.set_xlabel('Time')
ax2.set_ylabel('Magnetization')

plt.tight_layout()
plt.show()
```

## Batch Processing Workflow

```python
import mmpp
import numpy as np
from pathlib import Path

# Setup paths
data_dir = Path('simulation_results/')
output_dir = Path('analysis_output/')
output_dir.mkdir(exist_ok=True)

# Load all simulations
op = mmpp.MMPP(data_dir)
batch = op[:]

print(f"Processing {len(batch.zarr_results)} simulation files...")

# Batch FFT analysis - use compute_all for batch operations
fft_results = batch.fft.compute_all(
    dset='m_z5-8'
)

# Batch mode analysis
mode_results = batch.fft.modes.compute_modes(
    dset='m_z5-8',
    parallel=True,
    progress=True
)

# Extract successful results
successful_ffts = [r for r in fft_results if r['success']]
successful_modes = [r for r in mode_results if r['success']]

print(f"\\nResults:")
print(f"  Successful FFT computations: {len(successful_ffts)}")
print(f"  Successful mode computations: {len(successful_modes)}")

# Save summary
summary = {
    'total_files': len(batch.zarr_results),
    'successful_ffts': len(successful_ffts),
    'successful_modes': len(successful_modes),
    'fft_results': successful_ffts,
    'mode_results': successful_modes
}

import json
with open(output_dir / 'analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
```

## Custom Analysis Function

```python
import mmpp
import numpy as np

def analyze_peak_characteristics(zarr_result):
    """
    Custom analysis function to extract peak characteristics
    """
    try:
        # Get FFT analyzer
        fft = zarr_result.fft
        
        # Compute spectrum
        spectrum = fft.spectrum(dset='m_z5-8')
        frequencies = fft.frequencies()
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(
            np.abs(spectrum), 
            height=np.max(np.abs(spectrum)) * 0.1,
            distance=10
        )
        
        # Extract characteristics
        peak_freqs = frequencies[peaks]
        peak_heights = np.abs(spectrum[peaks])
        
        return {
            'peak_frequencies': peak_freqs.tolist(),
            'peak_heights': peak_heights.tolist(),
            'dominant_frequency': peak_freqs[np.argmax(peak_heights)],
            'num_peaks': len(peaks)
        }
        
    except Exception as e:
        return {'error': str(e)}

# Apply custom analysis to all files
op = mmpp.MMPP('simulation_results/')
batch = op[:]

# Note: Custom analysis functions would need to be implemented
# For demonstration purposes - this feature may not be available
# custom_results = batch.process_custom(
#     analyze_peak_characteristics,
#     parallel=True,
#     progress=True
# )

# Process results
for result in custom_results:
    if 'error' not in result['result']:
        analysis = result['result']
        print(f"{result['file']}: {analysis['num_peaks']} peaks, "
              f"dominant at {analysis['dominant_frequency']:.2f} GHz")
    else:
        print(f"{result['file']}: Error - {result['result']['error']}")
```

## Advanced Mode Analysis

```python
import mmpp
import matplotlib.pyplot as plt
import numpy as np

# Load data
result = mmpp.MMPP('complex_simulation.zarr')[0]

# Compute modes with different parameters
fft = result.fft
modes_analyzer = fft.modes

# Method 1: Basic mode computation
basic_modes = modes_analyzer.compute_modes(dset='m_z5-8')

# Method 2: Mode computation with custom parameters
advanced_modes = modes_analyzer.compute_modes(
    dset='m_z5-8',
    method='advanced',  # if supported
    threshold=0.05
)

# Analyze mode characteristics
print(f"Basic method found {len(basic_modes)} modes")
print(f"Advanced method found {len(advanced_modes)} modes")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot spectrum
spectrum = fft.spectrum(dset='m_z5-8')
frequencies = fft.frequencies()

axes[0,0].plot(frequencies, np.abs(spectrum))
axes[0,0].set_title('FFT Spectrum')
axes[0,0].set_xlabel('Frequency (GHz)')
axes[0,0].set_ylabel('Amplitude')

# Plot basic modes
if basic_modes:
    mode_freqs = [mode.frequency for mode in basic_modes]
    mode_amps = [mode.amplitude for mode in basic_modes]
    axes[0,1].scatter(mode_freqs, mode_amps, c='red', s=50)
    axes[0,1].set_title('Basic Mode Detection')
    axes[0,1].set_xlabel('Frequency (GHz)')
    axes[0,1].set_ylabel('Amplitude')

# Plot advanced modes
if advanced_modes:
    mode_freqs = [mode.frequency for mode in advanced_modes]
    mode_amps = [mode.amplitude for mode in advanced_modes]
    axes[1,0].scatter(mode_freqs, mode_amps, c='blue', s=50)
    axes[1,0].set_title('Advanced Mode Detection')
    axes[1,0].set_xlabel('Frequency (GHz)')
    axes[1,0].set_ylabel('Amplitude')

# Mode comparison
axes[1,1].plot(frequencies, np.abs(spectrum), alpha=0.7, label='Spectrum')
if basic_modes:
    basic_freqs = [mode.frequency for mode in basic_modes]
    axes[1,1].axvline(x=freq, color='red', alpha=0.7, linestyle='--' 
                     for freq in basic_freqs)
if advanced_modes:
    advanced_freqs = [mode.frequency for mode in advanced_modes]
    axes[1,1].axvline(x=freq, color='blue', alpha=0.7, linestyle=':' 
                     for freq in advanced_freqs)
axes[1,1].set_title('Mode Comparison')
axes[1,1].legend()

plt.tight_layout()
plt.show()
```

## Error Handling and Robustness

```python
import mmpp
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_batch_processing(data_directory):
    """
    Robust batch processing with comprehensive error handling
    """
    try:
        # Load data
        op = mmpp.MMPP(data_directory)
        batch = op[:]
        
        logger.info(f"Found {len(batch.zarr_results)} files to process")
        
        # Process with error handling
        results = {
            'fft_spectra': [],
            'modes': [],
            'errors': []
        }
        
        # FFT computation
        logger.info("Computing FFT spectra...")
        fft_results = batch.fft.compute_all(
            dset='m_z5-8'
        )
        
        # Separate successful and failed results
        for result in fft_results:
            if result['success']:
                results['fft_spectra'].append(result)
            else:
                results['errors'].append({
                    'file': result['file'],
                    'operation': 'fft_spectrum',
                    'error': result['error']
                })
        
        # Mode computation (only for successful FFT results)
        logger.info("Computing modes...")
        if results['fft_spectra']:
            mode_results = batch.fft.modes.compute_modes(
                dset='m_z5-8',
                parallel=True,
                progress=True
            )
            
            for result in mode_results:
                if result['success']:
                    results['modes'].append(result)
                else:
                    results['errors'].append({
                        'file': result['file'],
                        'operation': 'modes',
                        'error': result['error']
                    })
        
        # Summary
        logger.info(f"Processing complete:")
        logger.info(f"  Successful FFT: {len(results['fft_spectra'])}")
        logger.info(f"  Successful modes: {len(results['modes'])}")
        logger.info(f"  Errors: {len(results['errors'])}")
        
        # Log errors
        if results['errors']:
            logger.warning("Errors encountered:")
            for error in results['errors']:
                logger.warning(f"  {error['file']} ({error['operation']}): {error['error']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Critical error in batch processing: {e}")
        raise

# Usage
if __name__ == "__main__":
    data_dir = "simulation_data/"
    results = robust_batch_processing(data_dir)
    
    # Save results
    import pickle
    with open("batch_analysis_results.pkl", "wb") as f:
        pickle.dump(results, f)
```

These examples demonstrate the flexibility and power of MMPP for both single-file analysis and large-scale batch processing of simulation data.
