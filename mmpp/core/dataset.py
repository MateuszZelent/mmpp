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
        # Create regular FFT instance
        if FFT_AVAILABLE:
            self._fft = FFT(job_result, mmpp_instance)
        else:
            self._fft = None

    def __getattr__(self, name):
        """Delegate to FFT, injecting dataset context when appropriate."""
        if self._fft is None:
             raise ImportError("FFT functionality not available. Install with: pip install mmpp[fft]")

        attr = getattr(self._fft, name)

        if name == "dispersion" and attr is not None:
            return attr.clone_for_dataset(self.dataset_name, slice_info=self.slice_info)

        if name == "transmission" and attr is not None:
            return attr.clone_for_dataset(self.dataset_name, slice_info=self.slice_info)

        if callable(attr) and hasattr(attr, "__code__"):
            sig = inspect.signature(attr)
            if "dataset_name" in sig.parameters:

                def wrapper(*args, **kwargs):
                    if "dataset_name" not in kwargs:
                        kwargs["dataset_name"] = self.dataset_name
                    if (
                        self.slice_info is not None
                        and "slice_info" in sig.parameters
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

            console = Console(file=io.StringIO(), force_terminal=True)

            # Header
            header = Text(f"FFT Analysis for dataset: '{self.dataset_name}'", style="bold cyan")
            if self.slice_info:
                header.append(f" [slice: {self.slice_info}]", style="yellow")
            console.print(Panel(header, border_style="blue"))

            # Available methods
            methods = []
            
            if self._fft:
                # Get methods from FFT class
                for name in dir(self._fft):
                    if name.startswith("_"):
                        continue
                    
                    attr = getattr(self._fft, name)
                    if not callable(attr):
                        continue
                        
                    # Check if method accepts dataset_name
                    try:
                        sig = inspect.signature(attr)
                        if "dataset_name" in sig.parameters:
                            doc = attr.__doc__ or ""
                            first_line = doc.strip().split("\n")[0]
                            methods.append(
                                Panel(
                                    Text(first_line, style="white"),
                                    title=f".{name}()",
                                    title_align="left",
                                    border_style="green",
                                )
                            )
                    except Exception:
                        continue

            if methods:
                console.print(Columns(methods, equal=True, expand=True))
            else:
                console.print("No dataset-specific methods available.")

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

    def __getitem__(self, key):
        """Return new DatasetAwareWrapper with slicing info preserved"""
        # Instead of returning raw numpy array, return new wrapper with slice info
        if self.slice_info is not None:
            # Combine existing slice with new slice - simplified for now
            combined_slice = key
        else:
            combined_slice = key

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
