import zarr
import numpy as np
import inspect
from typing import TYPE_CHECKING, Any, Optional, Union

from .constants import ArraySlice, FFT_AVAILABLE, RICH_AVAILABLE

if TYPE_CHECKING:
    from .job import ZarrJobResult
    from .mmpp import MMPP

if FFT_AVAILABLE:
    from ..fft import FFT

if RICH_AVAILABLE:
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

class DatasetSpecificFFT:
    """FFT wrapper with pre-set dataset"""

    def __init__(self, job_result, dataset_name, mmpp_instance=None, slice_info=None):
        self.dataset_name = dataset_name
        self.slice_info = slice_info
        self._job_result = job_result  # Keep reference for path access
        # Create regular FFT instance
        if FFT_AVAILABLE:
            self._fft = FFT(job_result, mmpp_instance)
        else:
            self._fft = None
    
    def __repr__(self):
        """Concise repr to avoid printing zarr structure."""
        path = getattr(self._job_result, 'path', 'unknown')
        slice_str = f", slice={self.slice_info}" if self.slice_info else ""
        return f"<DatasetSpecificFFT(dataset='{self.dataset_name}'{slice_str}) @ {path}>"
    
    def __str__(self):
        return self.__repr__()

    def __getattr__(self, name):
        """Delegate to FFT, injecting dataset context when appropriate."""
        if self._fft is None:
             raise ImportError("FFT functionality not available. Install with: pip install mmpp[fft]")

        attr = getattr(self._fft, name)

        if name == "dispersion" and attr is not None:
            return attr.clone_for_dataset(self.dataset_name, slice_info=self.slice_info)

        if name == "transmission" and attr is not None:
            return attr.clone_for_dataset(self.dataset_name, slice_info=self.slice_info)

        # For modes, we need to inject dataset context into the mode analyzer
        if name == "modes" and attr is not None:
            # Set dataset context on the modes interface
            attr._dataset_context = self.dataset_name
            attr._slice_context = self.slice_info
            return attr

        # Special handling for spectrum property (returns SpectrumHelper)
        if name == "spectrum" and attr is not None:
            # Wrap SpectrumHelper to inject dataset and slice_info
            class SpectrumHelperWrapper:
                def __init__(self, spectrum_helper, dataset_name, slice_info):
                    self._spectrum_helper = spectrum_helper
                    self._dataset_name = dataset_name
                    self._slice_info = slice_info
                
                def __call__(self, *args, **kwargs):
                    # Inject dataset and slice_info into kwargs
                    if "dset" not in kwargs:
                        kwargs["dset"] = self._dataset_name
                    if self._slice_info is not None and "slice_info" not in kwargs:
                        kwargs["slice_info"] = self._slice_info
                    return self._spectrum_helper(*args, **kwargs)
                
                def __repr__(self):
                    return repr(self._spectrum_helper)
                
                def _repr_html_(self):
                    return getattr(self._spectrum_helper, '_repr_html_', lambda: None)()
            
            return SpectrumHelperWrapper(attr, self.dataset_name, self.slice_info)

        if callable(attr) and hasattr(attr, "__code__"):
            sig = inspect.signature(attr)
            params = sig.parameters
            
            # Check if method accepts dataset (via 'dset' or 'dataset_name')
            has_dataset_param = "dset" in params or "dataset_name" in params
            has_slice_param = "slice_info" in params

            if has_dataset_param or has_slice_param:
                def wrapper(*args, **kwargs):
                    # Inject dataset name
                    if "dset" in params and "dset" not in kwargs:
                        kwargs["dset"] = self.dataset_name
                    elif "dataset_name" in params and "dataset_name" not in kwargs:
                        kwargs["dataset_name"] = self.dataset_name
                    
                    # Inject slice_info
                    if (
                        self.slice_info is not None
                        and "slice_info" in params
                        and "slice_info" not in kwargs
                    ):
                        kwargs["slice_info"] = self.slice_info
                    
                    return attr(*args, **kwargs)

                return wrapper

        return attr

    def __repr__(self):
        """Rich documentation display for dataset-specific FFT interface."""
        try:
            return self._rich_dataset_fft_display()
        except Exception:
            return self._basic_dataset_fft_display()

    def _rich_dataset_fft_display(self) -> str:
        """Create rich documentation display for dataset-specific FFT."""
        try:
            import io
            
            if not RICH_AVAILABLE:
                 return self._basic_dataset_fft_display()

            console = Console(file=io.StringIO(), force_terminal=True, width=100)

            # Header
            header = Text()
            header.append("📊 FFT Analysis Interface\n", style="bold cyan")
            header.append(f"📁 Dataset: '{self.dataset_name}'\n", style="white")
            if self.slice_info:
                header.append(f"🔖 Slice: {self.slice_info}", style="yellow")
            console.print(Panel(header, border_style="cyan"))

            # Available Modules table
            modules = Text()
            modules.append("📦 Available Modules:\n\n", style="bold yellow")
            
            module_info = [
                ("spectrum", "Compute & plot FFT power spectrum", ".fft.spectrum()"),
                ("modes", "Interactive FMR mode visualization", ".fft.modes"),
                ("dispersion", "Dispersion relation analysis", ".fft.dispersion"),
                ("transmission", "Transmission/absorption analysis", ".fft.transmission"),
            ]
            
            for name, desc, usage in module_info:
                modules.append(f"  • ", style="dim")
                modules.append(f"{name:15}", style="bold green")
                modules.append(f" {desc}\n", style="white")
                modules.append(f"    └─ Usage: job[0].m[...]{usage}\n", style="dim cyan")
            
            console.print(modules)

            # Quick methods
            quick = Text()
            quick.append("\n⚡ Quick Methods:\n\n", style="bold magenta")
            quick_methods = [
                (".spectrum()", "→ SpectrumResult with .plot_spectrum(), .power, .frequencies"),
                (".frequencies()", "→ Frequency array (Hz)"),
                (".power()", "→ Power spectrum |FFT|²"),
            ]
            for method, result in quick_methods:
                quick.append(f"  job[0].m[...].fft{method} ", style="cyan")
                quick.append(f"{result}\n", style="dim")
            
            console.print(quick)

            # Examples
            example = '''# Spectrum with component selection:
job[0].m[:200,...,1].fft.spectrum().plot_spectrum(log_scale=True)

# Interactive modes:
job[0].m[:200,...,0].fft.modes.interactive_spectrum(dpi=150)

# Access modes helper:
job[0].fft.modes  # Shows mode analysis options'''
            
            from rich.syntax import Syntax
            syntax = Syntax(example, "python", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title="[bold green]Examples", border_style="green"))

            return console.file.getvalue()  # type: ignore
        except Exception:
            return self._basic_dataset_fft_display()

    def _basic_dataset_fft_display(self) -> str:
        """Basic text representation."""
        slice_str = f" [slice: {self.slice_info}]" if self.slice_info else ""
        return f"<DatasetSpecificFFT(dataset='{self.dataset_name}'{slice_str})>"


class DatasetAwareWrapper:
    """Wrapper that acts like zarr.Array but has .fft property"""

    def __init__(self, job_result, dataset_name, zarr_array, slice_info=None):
        self.job_result = job_result
        self.dataset_name = dataset_name
        self.zarr_array = zarr_array
        self.slice_info = slice_info  # Store slicing information
        self._fft = None

    def _resolve_source(self):
        """Return underlying data respecting the stored slice."""
        if self.slice_info is not None:
            return self.zarr_array[self.slice_info]
        return self.zarr_array

    def __getattr__(self, name):
        """Delegate to zarr_array for most attributes (but not our own properties)"""
        # Don't delegate properties that are defined on this class
        if name in ('dt', 'fft', 'shape', 'data'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if self.slice_info is not None:
            # If sliced, get attribute from sliced data
            sliced_data = self._resolve_source()
            return getattr(sliced_data, name)
        return getattr(self.zarr_array, name)

    @staticmethod
    def _normalize_slice_to_keep_dims(key, ndim: int):
        """
        Convert integer indices to slice(i, i+1) to preserve array dimensions.
        
        This ensures that indexing like arr[:,...,0] returns shape (N, M, ..., 1)
        instead of (N, M, ...) - preserving the number of axes.
        
        Parameters
        ----------
        key : tuple, int, slice, or other indexing object
            The indexing key from __getitem__
        ndim : int
            Number of dimensions in the source array
            
        Returns
        -------
        tuple
            Normalized indexing tuple with integers converted to single-item slices
        """
        # Handle single element (not tuple)
        if not isinstance(key, tuple):
            key = (key,)
        
        # Expand Ellipsis to fill missing dimensions
        # Count non-ellipsis elements to determine how many dims ellipsis should expand to
        n_ellipsis = sum(1 for k in key if k is Ellipsis)
        if n_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        
        if n_ellipsis == 1:
            # Find ellipsis position and expand it
            ellipsis_idx = key.index(Ellipsis)
            n_explicit = len(key) - 1  # excluding ellipsis
            n_expand = max(0, ndim - n_explicit)
            expanded = key[:ellipsis_idx] + (slice(None),) * n_expand + key[ellipsis_idx + 1:]
            key = expanded
        
        # Now convert integers to single-item slices
        result = []
        for k in key:
            if isinstance(k, (int, np.integer)):
                # Convert integer index to slice to keep dimension
                # Handle negative indices
                result.append(slice(k, k + 1 if k != -1 else None))
            else:
                result.append(k)
        
        return tuple(result)

    def __getitem__(self, key):
        """Return new DatasetAwareWrapper with slicing info preserved.
        
        IMPORTANT: Integer indices are automatically converted to single-item
        slices to preserve array dimensions. For example:
        
            arr[:, :, 0]  ->  arr[:, :, 0:1]
        
        This means the number of dimensions is always preserved after slicing.
        Use .squeeze() or .numpy(squeeze=True) to remove singleton dimensions.
        """
        # Get source shape to properly handle ellipsis expansion
        source_shape = self.zarr_array.shape
        ndim = len(source_shape)
        
        # Normalize the slice to keep dimensions
        normalized_key = self._normalize_slice_to_keep_dims(key, ndim)
        
        # Combine with existing slice if present
        if self.slice_info is not None:
            # For now, we don't support chained slicing - use the new slice directly
            # This could be enhanced in the future to properly compose slices
            combined_slice = normalized_key
        else:
            combined_slice = normalized_key

        return DatasetAwareWrapper(
            self.job_result,
            self.dataset_name,
            self.zarr_array,  # Keep original zarr reference
            slice_info=combined_slice,
        )

    @property
    def fft(self):
        """Return FFT with this dataset pre-selected"""
        if self._fft is None and FFT_AVAILABLE:
            # Create DatasetSpecificFFT with slicing info
            self._fft = DatasetSpecificFFT(
                self.job_result,
                self.dataset_name,
                getattr(self.job_result, "_mmpp_ref", None),
                slice_info=self.slice_info,
            )
        return self._fft

    @property
    def shape(self):
        """Shape accounting for slicing"""
        if self.slice_info is not None:
            sliced_data = self._resolve_source()
            return sliced_data.shape
        return self.zarr_array.shape

    @property
    def dt(self):
        """
        Get time step for this dataset.
        
        Algorithm:
        1. Check if 't_sampl' exists in job_result attrs (global)
        2. Check if 't' exists in THIS dataset's attrs and calculate dt
        3. Look for 't' array in various locations (root, table, etc.)
        4. Calculate dt = t[1] - t[0]
        
        Returns:
            float: Time step in seconds
        """
        # Method 1: Check for t_sampl in main attributes
        if hasattr(self.job_result, '_z') and self.job_result._z is not None:
            if 't_sampl' in self.job_result._z.attrs:
                return self.job_result._z.attrs['t_sampl']
        
        # Method 2: Check THIS dataset's attrs for 't' array (MOST SPECIFIC)
        if hasattr(self.job_result, '_z') and self.job_result._z is not None:
            try:
                dataset = self.job_result._z[self.dataset_name]
                if hasattr(dataset, 'attrs') and 't' in dataset.attrs:
                    t_attr = dataset.attrs['t']
                    # t_attr is a list or array in attrs
                    if hasattr(t_attr, '__len__') and len(t_attr) >= 2:
                        dt = float(t_attr[1] - t_attr[0])
                        return dt
            except (KeyError, NameError, AttributeError, IndexError, TypeError):
                pass
        
        # Method 3: Look for time array in various locations
        # Try common naming patterns and locations
        time_locations = [
            ('t',),  # Root level 't'
            ('table', 't'),  # Often in 'table' group
            ('time',),  # Alternative name
            (f't_{self.dataset_name}',),  # Dataset-specific time
        ]
        
        for location in time_locations:
            try:
                if hasattr(self.job_result, '_z'):
                    # Navigate through the location path
                    t_array = self.job_result._z
                    for key in location:
                        t_array = t_array[key]
                    
                    # Calculate dt from first two time points
                    if t_array.shape[0] >= 2:
                        dt = float(t_array[1] - t_array[0])
                        return dt
            except (KeyError, NameError, AttributeError, IndexError):
                continue
        
        # Method 4: Fallback - raise informative error
        raise AttributeError(
            f"Cannot determine time step for dataset '{self.dataset_name}'. "
            f"Neither 't_sampl' attribute nor time array 't' found in zarr file."
        )

    @property
    def data(self):
        """Return data as numpy array (loads into memory)."""
        return self.numpy(copy=False)

    def numpy(self, *, copy: bool = True, dtype=None, squeeze: bool = False):
        """Materialize the wrapped data as numpy array."""
        data = self._resolve_source()
        if isinstance(data, zarr.Array):
            data = data[:]
        array = np.array(data, copy=copy)
        if dtype is not None:
            array = array.astype(dtype, copy=copy)
        if squeeze:
            array = np.squeeze(array)
        return array

    def to_numpy(self, **kwargs):
        """Alias for numpy() to match common API naming."""
        return self.numpy(**kwargs)

    def as_zarr(self):
        """Return the underlying zarr.Array when no slicing is active."""
        if self.slice_info is not None:
            raise TypeError(
                "Sliced view has no standalone zarr representation; use numpy() instead"
            )
        return self.zarr_array

    def __array__(self, dtype=None):
        """Support implicit numpy conversions (e.g. np.asarray)."""
        array = self.numpy(copy=False)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def __iter__(self):
        return iter(self.numpy(copy=False))

    def __len__(self):
        return len(self.numpy(copy=False))

    def __repr__(self):
        slice_str = f"[{self.slice_info}]" if self.slice_info else ""
        return (
            f"DatasetAwareWrapper({self.dataset_name}{slice_str}, shape={self.shape})"
        )
