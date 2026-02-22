import zarr
import numpy as np
import inspect
from html import escape as _html_escape
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
                
                @property
                def plot(self):
                    """Quick-plot proxy with dataset context pre-injected."""
                    from ..fft.spectrum.helpers import _SpectrumQuickPlot
                    return _SpectrumQuickPlot(self)
                
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
            has_kwargs_param = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in params.values()
            )

            if has_dataset_param or has_slice_param or has_kwargs_param:
                def wrapper(*args, **kwargs):
                    # Inject dataset name
                    if "dset" in params and "dset" not in kwargs:
                        kwargs["dset"] = self.dataset_name
                    elif "dataset_name" in params and "dataset_name" not in kwargs:
                        kwargs["dataset_name"] = self.dataset_name
                    
                    # Inject slice_info
                    if (
                        self.slice_info is not None
                        and (has_slice_param or has_kwargs_param)
                        and "slice_info" not in kwargs
                    ):
                        kwargs["slice_info"] = self.slice_info
                    
                    return attr(*args, **kwargs)

                return wrapper

        return attr

    def filters(self, **filters):
        """Create fluent filter chain bound to current dataset and slice context.

        Examples
        --------
        >>> job[0].m_layer13[:1000, ..., 2].fft.filters(remove_static=True).spectrum()
        >>> job[0].m_layer13.fft.filters(post={"normalize": True, "log_transform": True}).spectrum()
        """
        if self._fft is None:
            raise ImportError("FFT functionality not available. Install with: pip install mmpp[fft]")

        from ..fft.spectrum import SpectrumFilterChain

        return SpectrumFilterChain(self.spectrum, filters)

    @property
    def helpers(self):
        """Helper namespace with dataset/slice-aware method wrappers."""
        from ..fft.core import FFTHelpAccessor

        owner = f"{self.dataset_name}.fft"
        return FFTHelpAccessor(self, owner=owner)

    @property
    def help(self):
        """Alias for :attr:`helpers`."""
        return self.helpers

    def __repr__(self):
        """Concise text representation."""
        dataset = self.dataset_name
        slice_label = self._format_slice_display() or "[full]"
        return f"<DatasetFFT: {dataset}{slice_label}>"

    def __str__(self):
        return self.__repr__()

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        try:
            return self._html_dataset_fft_display()
        except Exception:
            return ""

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Prefer HTML card in notebook frontends with plain-text fallback."""
        html = self._repr_html_()
        text = self.__repr__()
        if html:
            return {"text/html": html, "text/plain": text}
        return {"text/plain": text}

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

    def _format_slice_display(self) -> str:
        if self.slice_info is None:
            return ""

        def _fmt(item: Any) -> str:
            if item is Ellipsis:
                return "..."
            if isinstance(item, slice):
                start = "" if item.start is None else item.start
                stop = "" if item.stop is None else item.stop
                step = "" if item.step is None else item.step
                if step == "":
                    return f"{start}:{stop}"
                return f"{start}:{stop}:{step}"
            return str(item)

        if isinstance(self.slice_info, tuple):
            inner = ", ".join(_fmt(part) for part in self.slice_info)
        else:
            inner = _fmt(self.slice_info)
        return f"[{inner}]"

    def _html_dataset_fft_display(self) -> str:
        job_result = self._job_result
        job_name = getattr(job_result, "name", "unknown")
        job_path = getattr(job_result, "path", "")
        slice_label = self._format_slice_display()
        dataset_access = self.dataset_name if isinstance(self.dataset_name, str) else str(self.dataset_name)
        if isinstance(self.dataset_name, str) and self.dataset_name.isidentifier():
            prefix = f"job[0].{self.dataset_name}{slice_label}.fft"
        else:
            prefix = f"job[0][{self.dataset_name!r}]{slice_label}.fft"

        # ── method groups ───────────────────────────────────────
        section_style = (
            "padding:4px 8px; font-weight:600; color:#f1f5f9; "
            "background:rgba(51,65,85,0.8); text-align:left;"
        )
        row_html = ""

        groups: list[tuple[str, list[tuple[str, str]]]] = [
            ("Compute", [
                ("spectrum()", "FFT spectrum → SpectrumResult"),
                ("filters(**f).spectrum()", "Fluent filter chain → SpectrumResult"),
                ("power()", "Power spectrum |FFT|²"),
                ("frequencies()", "Frequency axis (Hz)"),
                ("magnitude()", "Magnitude spectrum |FFT|"),
                ("phase()", "Phase spectrum (radians)"),
            ]),
            ("Analysis", [
                ("dispersion", "Dispersion relation analysis"),
                ("modes", "FMR mode analysis interface"),
                ("transmission", "Transmission / absorption analysis"),
            ]),
            ("Plotting", [
                ("plot_spectrum()", "Quick-look power spectrum plot"),
                ("interactive_spectrum()", "Interactive mode spectrum viewer"),
            ]),
        ]

        for group_name, methods in groups:
            row_html += (
                f"<tr><td colspan='2' style='{section_style}'>"
                f"{_html_escape(group_name)}</td></tr>"
            )
            for name, desc in methods:
                row_html += (
                    "<tr>"
                    f"<td style='padding:5px 8px 5px 16px; font-family:monospace; "
                    f"color:#93c5fd; white-space:nowrap;'>{_html_escape(name)}</td>"
                    f"<td style='padding:5px 8px; color:#cbd5e1;'>{_html_escape(desc)}</td>"
                    "</tr>"
                )

        # ── context-aware examples ──────────────────────────────
        example_code = "\n".join([
            f"data = {prefix.rsplit('.fft', 1)[0]}",
            "",
            "# Compute spectrum (dataset & slice are pre-set)",
            "result = data.fft.spectrum()",
            "",
            "# Plot power spectrum",
            "result.plot.spectrum(log_scale=True, freq_unit='GHz')",
            "",
            "# Fluent filter chain",
            "data.fft.filters(remove_static=True).spectrum()",
            "",
            "# Frequency range",
            "result = data.fft.spectrum(fmin=1e9, fmax=20e9)",
            "",
            "# Peak detection",
            "result = data.fft.spectrum(find_peaks={'min_prominence': 0.1})",
            "",
            "# Analysis sub-interfaces",
            "data.fft.modes.interactive_spectrum(dpi=150)",
            "data.fft.dispersion.plot_dispersion(axis='x')",
        ])

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; border: 2px solid #334155; border-radius: 12px; padding: 16px; margin: 10px 0; background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); color: #e2e8f0; box-shadow: 0 10px 22px rgba(0,0,0,0.28);">
          <div style="margin-bottom: 12px;">
            <div style="font-size: 1.1em; font-weight: 600; color: #f1f5f9;">Dataset FFT Interface</div>
            <div style="color: #94a3b8; margin-top: 4px;">Job: {_html_escape(job_name)}</div>
            <div style="color: #94a3b8; margin-top: 2px;">Path: <code style="color:#cbd5e1;">{_html_escape(job_path)}</code></div>
          </div>

          <div style="background: rgba(15,23,42,0.6); padding: 10px; border-radius: 8px; margin-bottom: 12px; border: 1px solid rgba(148,163,184,0.2);">
            <div style="display:flex; flex-wrap:wrap; gap:12px; font-size:0.9em;">
              <div><span style="color:#94a3b8;">Dataset:</span> <code style="color:#93c5fd;">{_html_escape(dataset_access)}</code></div>
              <div><span style="color:#94a3b8;">Slice:</span> <code style="color:#93c5fd;">{_html_escape(slice_label or 'full')}</code></div>
            </div>
          </div>

          <div style="background: rgba(15,23,42,0.6); padding: 10px; border-radius: 8px; margin-bottom: 12px; border: 1px solid rgba(148,163,184,0.2);">
            <table style="width:100%; border-collapse: collapse; font-size:0.9em;">
              <thead>
                <tr style="text-align:left; background: rgba(51,65,85,0.6);">
                  <th style="padding:6px 8px; color:#e2e8f0;">Method</th>
                  <th style="padding:6px 8px; color:#e2e8f0;">Description</th>
                </tr>
              </thead>
              <tbody>
                {row_html}
              </tbody>
            </table>
          </div>

          <div style="background: rgba(15,23,42,0.6); padding: 10px; border-radius: 8px; border: 1px solid rgba(148,163,184,0.2);">
            <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 6px;">Examples</div>
            <pre style="margin:0; background: rgba(15,23,42,0.85); padding: 10px; border-radius: 6px; color:#e2e8f0; overflow-x:auto; font-size:0.85em;"><code>{_html_escape(example_code)}</code></pre>
          </div>
        </div>
        """
        return html


class DatasetAwareWrapper:
    """Wrapper that acts like zarr.Array but has .fft property"""

    def __init__(self, job_result, dataset_name, zarr_array, slice_info=None):
        self.job_result = job_result
        self.dataset_name = dataset_name
        self.zarr_array = zarr_array
        self.slice_info = slice_info  # Store slicing information
        self._fft = None
        self._solitons = None
        self._analyze = None

    def _resolve_source(self):
        """Return underlying data respecting the stored slice."""
        if self.slice_info is not None:
            return self.zarr_array[self.slice_info]
        return self.zarr_array

    def __getattr__(self, name):
        """Delegate to zarr_array for most attributes (but not our own properties)"""
        # Don't delegate properties that are defined on this class
        if name in ('dt', 'fft', 'analyze', 'shape', 'data'):
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
    def solitons(self):
        """Return soliton analysis interface with this dataset pre-selected."""
        if self._solitons is None:
            from ..solitons import DatasetSpecificSolitons

            self._solitons = DatasetSpecificSolitons(
                self.job_result,
                self.dataset_name,
                getattr(self.job_result, "_mmpp_ref", None),
                slice_info=self.slice_info,
            )
        return self._solitons

    @property
    def analyze(self):
        """Return analysis interface with this dataset pre-selected."""
        if self._analyze is None:
            from ..analyze import DatasetSpecificAnalyze

            self._analyze = DatasetSpecificAnalyze(
                self.job_result,
                self.dataset_name,
                getattr(self.job_result, "_mmpp_ref", None),
                slice_info=self.slice_info,
            )
        return self._analyze

    @property
    def vortex(self):
        """Shortcut alias for ``self.solitons.vortex``."""
        return self.solitons.vortex

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


class NumpyDatasetWrapper:
    """Wrapper that returns numpy array directly on slicing.
    
    Used by job[0].get.dataset_name[slice] to return numpy arrays directly.
    
    Example
    -------
    >>> arr = job[0].get.m[:]  # Returns numpy array directly
    >>> arr = job[0].get.m[0:100, ...]  # Sliced numpy array
    """
    
    def __init__(self, job_result, dataset_name: str, zarr_array):
        self._job_result = job_result
        self._dataset_name = dataset_name
        self._zarr_array = zarr_array
    
    def __getitem__(self, key) -> np.ndarray:
        """Return sliced data as numpy array."""
        return np.asarray(self._zarr_array[key])
    
    @property
    def shape(self):
        """Shape of the underlying dataset."""
        return self._zarr_array.shape
    
    @property
    def dtype(self):
        """Data type of the underlying dataset."""
        return self._zarr_array.dtype
    
    def __repr__(self):
        return f"NumpyDatasetWrapper({self._dataset_name}, shape={self.shape}, dtype={self.dtype})"


class NumpyGetter:
    """Helper providing direct numpy access via job[0].get.dataset_name[slice].
    
    This provides an explicit way to get numpy arrays directly from zarr datasets
    without returning a DatasetAwareWrapper.
    
    Example
    -------
    >>> # Single job - returns numpy array
    >>> arr = job[0].get.m[:]
    >>> arr = job[0].get.m[0:100, :, :, :, 0]
    >>> 
    >>> # Works with any dataset name
    >>> arr = job[0].get.m_layer13[:]
    >>> arr = job[0].get["m_layer13"][:]  # Alternative syntax for special names
    """
    
    def __init__(self, job_result):
        self._job_result = job_result
    
    def __getattr__(self, name: str) -> NumpyDatasetWrapper:
        """Get NumpyDatasetWrapper for dataset by attribute access."""
        self._job_result._ensure_zarr_loaded()
        try:
            member = self._job_result._get_zarr_member(name)
        except NameError:
            raise AttributeError(f"Dataset '{name}' not found in zarr file")
        
        if isinstance(member, zarr.Array):
            return NumpyDatasetWrapper(self._job_result, name, member)
        raise AttributeError(f"'{name}' is not a dataset (it's a group)")
    
    def __getitem__(self, key: str) -> NumpyDatasetWrapper:
        """Get NumpyDatasetWrapper for dataset by item access (for special names)."""
        return self.__getattr__(key)
    
    def __repr__(self):
        self._job_result._ensure_zarr_loaded()
        datasets = list(self._job_result._z.array_keys())
        return f"NumpyGetter(datasets={datasets[:5]}{'...' if len(datasets) > 5 else ''})"
