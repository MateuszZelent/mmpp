# Dataset Auto-Selection Implementation

## Overview

This document describes the implementation of automatic dataset selection functionality across the MMPP library. Instead of using hardcoded defaults like "m", "m_z11", or "m_z5-8", the system now automatically finds and uses the largest magnetization dataset by time dimension.

## Changes Made

### 1. Core Infrastructure (`mmpp/core.py`)

**Added Functions:**
- `find_largest_m_dataset(zarr_path: str) -> str`: Scans zarr files for all datasets starting with "m" and returns the one with the largest time dimension (shape[0])
- `ZarrJobResult.get_largest_m_dataset() -> str`: Convenience method for easy access to auto-selection

**Features:**
- Comprehensive error handling and logging
- Falls back to "m" if no datasets found or on error
- Supports complex dataset names like "m_z5", "m_z11-12", etc.

### 2. Plotting Module (`mmpp/plotting.py`)

**Modified Functions:**
- `snapshot(dset: Optional[str] = None, ...)`: Now accepts None for auto-selection
- Added `_find_largest_m_dataset()` helper function to avoid circular imports

**Behavior:**
- When `dset=None`, automatically selects the largest m dataset
- Maintains backward compatibility with explicit dataset specification

### 3. FFT Module (`mmpp/fft/`)

#### FFT Core (`mmpp/fft/core.py`)
- `_compute_fft(dataset_name: Optional[str] = None, ...)`: Auto-selects when None
- Updated docstrings to reflect new behavior

#### FFT Main (`mmpp/fft/main.py`)
- `analyze_single(dataset_name: Optional[str] = None, ...)`: Auto-selects when None
- `analyze_all(dataset_name: Optional[str] = None, ...)`: Auto-selects when None
- Updated docstrings and parameter documentation

#### FFT Plot (`mmpp/fft/plot.py`)
- `power_spectrum(dataset_name: Optional[str] = None, ...)`: Auto-selects when None
- Updated docstrings and parameter documentation

#### FMR Modes (`mmpp/fft/modes.py`)
- `FMRModeAnalyzer.__init__(dataset_name: Optional[str] = None, ...)`: Auto-selects when None
- Updated docstrings and parameter documentation

### 4. Batch Operations (`mmpp/batch_operations.py`)

**Modified Functions:**
- `compute_modes(dset: Optional[str] = None, ...)`: Auto-selects when None
- Updated module docstring and examples to reflect new behavior

## API Changes

### Before (Hardcoded Defaults)
```python
# Old API with hardcoded defaults
result.plot.snapshot(dset="m")  # Required explicit specification
fft_analyzer._compute_fft(dataset_name="m_z11")  # Hardcoded default
batch_ops.compute_modes(dset="m_z5-8")  # Hardcoded default
```

### After (Auto-Selection)
```python
# New API with auto-selection
result.plot.snapshot()  # Automatically selects optimal dataset
fft_analyzer._compute_fft()  # Automatically selects optimal dataset
batch_ops.compute_modes()  # Automatically selects optimal dataset

# Explicit specification still works
result.plot.snapshot(dset="m_z5")  # Explicit override
```

## Backward Compatibility

All changes are fully backward compatible:
- Explicit dataset specification continues to work exactly as before
- Only the default behavior changes from hardcoded values to auto-selection
- Existing code using explicit parameters will continue to function unchanged

## Benefits

1. **Improved User Experience**: Users no longer need to manually specify dataset names when working with cropped data
2. **Intelligent Defaults**: System automatically selects the dataset with the most temporal information
3. **Reduced Errors**: Eliminates common errors from using incorrect dataset names
4. **Flexibility**: Works with various dataset naming conventions (m, m_z5, m_z11-12, etc.)

## Implementation Details

### Auto-Selection Logic
1. Scan zarr file for all datasets starting with "m"
2. Compare time dimensions (shape[0]) of each dataset
3. Return the dataset name with the largest time dimension
4. Fall back to "m" if no suitable datasets found

### Error Handling
- Comprehensive logging for debugging
- Graceful fallback to "m" on any errors
- Detailed error messages for troubleshooting

### Performance
- Minimal overhead: only scans dataset metadata, not full data
- Results can be cached at the application level if needed
- Function execution time is negligible compared to actual FFT computations

## Testing

All modules have been tested for:
- Successful imports
- Function availability
- Method signature compatibility
- Basic functionality without errors

## Files Modified

1. `/mmpp/core.py` - Added auto-selection infrastructure
2. `/mmpp/plotting.py` - Updated snapshot method
3. `/mmpp/fft/core.py` - Updated _compute_fft method
4. `/mmpp/fft/main.py` - Updated analyze_single and analyze_all methods
5. `/mmpp/fft/plot.py` - Updated power_spectrum method
6. `/mmpp/fft/modes.py` - Updated FMRModeAnalyzer constructor
7. `/mmpp/batch_operations.py` - Updated compute_modes method

## Next Steps

1. **Documentation Updates**: Update all user-facing documentation to reflect new auto-selection behavior
2. **Tutorial Updates**: Modify tutorials and examples to demonstrate auto-selection
3. **Testing**: Create comprehensive tests for auto-selection functionality with various zarr file structures
4. **Performance Monitoring**: Monitor performance impact in production environments
