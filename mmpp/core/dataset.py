from __future__ import annotations

import inspect
import warnings
from html import escape as _html_escape
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

from .constants import FFT_AVAILABLE, RICH_AVAILABLE
from .dataset_geometry import (
    IndexPlan,
    compose_index_keys,
    has_only_simple_slices,
    make_index_plan,
    normalize_index_key,
    resolve_dataset_geometry,
    shape_after_index,
)

if TYPE_CHECKING:
    pass

if RICH_AVAILABLE:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

from .dataset_plotting import DatasetPlotAccessor


class DatasetSpecificFFT:
    """FFT wrapper with pre-set dataset"""

    def __init__(
        self,
        job_result,
        dataset_name,
        mmpp_instance=None,
        slice_info=None,
        materialized_data: np.ndarray | None = None,
        index_plan: IndexPlan | None = None,
        time_step_scale: float = 1.0,
        view_geometry=None,
    ):
        self.dataset_name = dataset_name
        self.slice_info = slice_info
        self._job_result = job_result  # Keep reference for path access
        self._materialized_data = materialized_data
        self._index_plan = index_plan
        self._time_step_scale = float(time_step_scale)
        self._view_geometry = view_geometry
        # Create regular FFT instance
        try:
            from ..fft import FFT

            self._fft = FFT(job_result, mmpp_instance)
        except ImportError:
            self._fft = None

    def __getattr__(self, name):
        """Delegate to FFT, injecting dataset context when appropriate."""
        if self._fft is None:
            raise ImportError(
                "FFT functionality not available. Install with: pip install mmpp[fft]"
            )

        attr = getattr(self._fft, name)

        if name == "dispersion" and attr is not None:
            return attr.clone_for_dataset(
                self.dataset_name,
                slice_info=self.slice_info,
                preloaded_data=self._materialized_data,
                time_step_scale=self._time_step_scale,
                view_geometry=self._view_geometry,
            )

        if name == "transmission" and attr is not None:
            return attr.clone_for_dataset(
                self.dataset_name,
                slice_info=self.slice_info,
                preloaded_data=self._materialized_data,
                time_step_scale=self._time_step_scale,
                view_geometry=self._view_geometry,
            )

        # For modes, we need to inject dataset context into the mode analyzer
        if name == "modes" and attr is not None:
            # Set dataset context on the modes interface
            attr._dataset_context = self.dataset_name
            attr._slice_context = self.slice_info
            attr._preloaded_context = self._materialized_data
            attr._time_step_scale_context = self._time_step_scale
            attr._geometry_context = self._view_geometry
            return attr

        # Special handling for spectrum property (returns SpectrumHelper)
        if name == "spectrum" and attr is not None:
            # Wrap SpectrumHelper to inject dataset and slice_info
            class SpectrumHelperWrapper:
                def __init__(
                    self,
                    spectrum_helper,
                    dataset_name,
                    slice_info,
                    materialized_data=None,
                    time_step_scale=1.0,
                    view_geometry=None,
                    index_plan=None,
                ):
                    self._spectrum_helper = spectrum_helper
                    self._dataset_name = dataset_name
                    self._slice_info = slice_info
                    self._materialized_data = materialized_data
                    self._time_step_scale = float(time_step_scale)
                    self._view_geometry = view_geometry
                    self._index_plan = index_plan

                def __call__(self, *args, **kwargs):
                    # Inject dataset and slice_info into kwargs
                    if "dset" not in kwargs:
                        kwargs["dset"] = self._dataset_name
                    if self._slice_info is not None and "slice_info" not in kwargs:
                        kwargs["slice_info"] = self._slice_info
                    # Inject pre-materialized data so FFT doesn't reload from storage
                    if (
                        self._materialized_data is not None
                        and "preloaded_data" not in kwargs
                    ):
                        kwargs["preloaded_data"] = self._materialized_data
                    if (
                        self._materialized_data is not None
                        and self._time_step_scale != 1.0
                        and "time_step_scale" not in kwargs
                    ):
                        kwargs["time_step_scale"] = self._time_step_scale
                    result = self._spectrum_helper(*args, **kwargs)
                    mode_context = getattr(result, "_mode_context", None)
                    if isinstance(mode_context, dict):
                        mode_context["preloaded_data"] = self._materialized_data
                        mode_context["time_step_scale"] = self._time_step_scale
                        mode_context["view_geometry"] = self._view_geometry
                        mode_context["index_plan"] = self._index_plan
                    return result

                @property
                def plot(self):
                    """Quick-plot proxy with dataset context pre-injected."""
                    from ..fft.spectrum.helpers import _SpectrumQuickPlot

                    return _SpectrumQuickPlot(self)

                def __repr__(self):
                    return repr(self._spectrum_helper)

                def _repr_html_(self):
                    return getattr(self._spectrum_helper, "_repr_html_", lambda: None)()

            return SpectrumHelperWrapper(
                attr,
                self.dataset_name,
                self.slice_info,
                self._materialized_data,
                self._time_step_scale,
                self._view_geometry,
                self._index_plan,
            )

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
            raise ImportError(
                "FFT functionality not available. Install with: pip install mmpp[fft]"
            )

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
                (
                    "transmission",
                    "Transmission/absorption analysis",
                    ".fft.transmission",
                ),
            ]

            for name, desc, usage in module_info:
                modules.append("  • ", style="dim")
                modules.append(f"{name:15}", style="bold green")
                modules.append(f" {desc}\n", style="white")
                modules.append(
                    f"    └─ Usage: job[0].m[...]{usage}\n", style="dim cyan"
                )

            console.print(modules)

            # Quick methods
            quick = Text()
            quick.append("\n⚡ Quick Methods:\n\n", style="bold magenta")
            quick_methods = [
                (
                    ".spectrum()",
                    "→ SpectrumResult with .plot_spectrum(), .power, .frequencies",
                ),
                (".frequencies()", "→ Frequency array (Hz)"),
                (".power()", "→ Power spectrum |FFT|²"),
            ]
            for method, result in quick_methods:
                quick.append(f"  job[0].m[...].fft{method} ", style="cyan")
                quick.append(f"{result}\n", style="dim")

            console.print(quick)

            # Examples
            example = """# Spectrum with component selection:
job[0].m[:200,...,1].fft.spectrum().plot_spectrum(log_scale=True)

# Interactive modes:
job[0].m[:200,...,0].fft.modes.interactive_spectrum(dpi=150)

# Access modes helper:
job[0].fft.modes  # Shows mode analysis options"""

            from rich.syntax import Syntax

            syntax = Syntax(example, "python", theme="monokai", line_numbers=False)
            console.print(
                Panel(syntax, title="[bold green]Examples", border_style="green")
            )

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
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            NODE_COLOR_PLOT,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        job_result = self._job_result
        job_name = getattr(job_result, "name", "unknown")
        slice_label = self._format_slice_display()
        dataset_access = (
            self.dataset_name
            if isinstance(self.dataset_name, str)
            else str(self.dataset_name)
        )
        if isinstance(self.dataset_name, str) and self.dataset_name.isidentifier():
            data_prefix = f"job[0].{self.dataset_name}{slice_label}"
        else:
            data_prefix = f"job[0][{self.dataset_name!r}]{slice_label}"
        fft_prefix = f"{data_prefix}.fft"
        uid = str(_uuid.uuid4())[:8]

        status = metrics_section_html(
            [
                ("job", job_name, None),
                ("dataset", dataset_access, "#93c5fd"),
                (
                    "slice",
                    slice_label if slice_label else "full",
                    "#fbbf24" if slice_label else None,
                ),
            ]
        )

        accessors = accessors_section_html(
            [
                (
                    "Compute:",
                    [
                        ("spectrum()", NODE_COLOR_COMPUTE),
                        ("filters(**f).spectrum()", NODE_COLOR_COMPUTE),
                        ("power()", NODE_COLOR_COMPUTE),
                        ("frequencies()", NODE_COLOR_COMPUTE),
                        ("magnitude()", NODE_COLOR_COMPUTE),
                        ("phase()", NODE_COLOR_COMPUTE),
                    ],
                ),
                (
                    "Analysis:",
                    [
                        ("dispersion", NODE_COLOR_ANALYSIS),
                        ("modes", NODE_COLOR_ANALYSIS),
                        ("transmission", NODE_COLOR_ANALYSIS),
                    ],
                ),
                (
                    "Plotting:",
                    [
                        ("plot_spectrum()", NODE_COLOR_PLOT),
                        ("interactive_spectrum()", NODE_COLOR_PLOT),
                    ],
                ),
            ]
        )

        examples = examples_section_html(
            "\n".join(
                [
                    f"data = {data_prefix}",
                    "",
                    "# Compute spectrum (dataset & slice pre-set)",
                    "result = data.fft.spectrum()",
                    "",
                    "# Plot power spectrum",
                    "result.plot.spectrum(log_scale=True, freq_unit='GHz')",
                    "",
                    "# Fluent filter chain",
                    "data.fft.filters(remove_static=True).spectrum()",
                    "",
                    "# Frequency range & peak detection",
                    "result = data.fft.spectrum(fmin=1e9, fmax=20e9,",
                    "                          find_peaks={'min_prominence': 0.1})",
                    "",
                    "# Analysis sub-interfaces",
                    "data.fft.modes.interactive_spectrum(dpi=150)",
                    "data.fft.dispersion.plot_dispersion(axis='x')",
                ]
            )
        )

        api = api_help_html(
            self,
            title="Dataset FFT API help",
            prefix=fft_prefix,
            subtitle="Dataset and slice context are pre-bound; all methods operate on the selected view.",
            properties=[
                ("spectrum", "Callable spectrum helper"),
                ("dispersion", "Dispersion relation namespace"),
                ("modes", "FMR mode analysis namespace"),
                ("transmission", "Transmission / absorption namespace"),
            ],
            methods=[
                "filters",
                "power",
                "frequencies",
                "magnitude",
                "phase",
                "plot_spectrum",
                "plot_modes",
                "interactive_spectrum",
            ],
            chrome=False,
        )

        slice_part = (
            f" · slice <code style='color:#fbbf24'>{_html_escape(slice_label)}</code>"
            if slice_label
            else ""
        )
        subtitle = (
            f"Pre-bound to <code style='color:#93c5fd'>{_html_escape(dataset_access)}</code>"
            + slice_part
        )

        return node_card_html(
            "Dataset FFT Interface",
            icon="📊",
            subtitle=subtitle,
            sections=[status, accessors, examples],
            api=api,
            uid=f"dsfft-{dataset_access}-{uid}",
        )


class DatasetAwareWrapper:
    """Wrapper that acts like zarr.Array but has .fft property"""

    def __init__(
        self,
        job_result,
        dataset_name,
        zarr_array,
        slice_info=None,
        materialized_data: np.ndarray | None = None,
        geometry_override=None,
        index_plan: IndexPlan | None = None,
        time_step_scale: float = 1.0,
    ):
        self.job_result = job_result
        self.dataset_name = dataset_name
        self.zarr_array = zarr_array
        self.slice_info = slice_info  # Store slicing information
        self._materialized_data = materialized_data
        self._geometry_override = geometry_override
        self._index_plan: IndexPlan | None = index_plan
        self._time_step_scale = float(time_step_scale)
        self._fft: Any | None = None
        self._solitons: Any | None = None
        self._analyze: Any | None = None
        self._plot: Any | None = None

    # ------------------------------------------------------------------ #
    # Shape helpers                                                        #
    # ------------------------------------------------------------------ #

    def _base_shape(self) -> tuple[int, ...]:
        """Shape of the immediate backing store before ``slice_info``."""
        if self._materialized_data is not None:
            return tuple(int(v) for v in np.asarray(self._materialized_data).shape)

        shape = getattr(self.zarr_array, "shape", None)
        if shape is not None:
            return tuple(int(v) for v in shape)

        resolved = self._resolve_source()
        shape = getattr(resolved, "shape", None)
        if shape is None:
            raise AttributeError("Cannot determine source shape for dataset wrapper")
        return tuple(int(v) for v in shape)

    @staticmethod
    def _materialized_time_step_scale(local_key: Any, fallback: float) -> float:
        """Update source-time spacing from a local materialized-view selection.

        ``nan`` marks reversed or irregular sampling. Such data remains usable
        as an ndarray, but FFT entrypoints reject it because one scalar ``dt``
        cannot describe its time axis.

        The selection must be local to the current materialized view. Using a
        composed :class:`IndexPlan` here loses the distinction between its
        cumulative source stride and ``fallback`` and breaks chained slicing
        after ``downsample()``.
        """
        if local_key is None:
            return float(fallback)
        key = local_key if isinstance(local_key, tuple) else (local_key,)
        if not key:
            return float(fallback)
        token = key[0]
        if isinstance(token, slice):
            step = 1 if token.step is None else int(token.step)
            return float(fallback) * step if step > 0 else float("nan")
        if isinstance(token, (list, tuple, np.ndarray)):
            indices = np.asarray(token)
            if indices.dtype == bool:
                indices = np.flatnonzero(indices)
            indices = indices.astype(np.int64, copy=False).reshape(-1)
            if indices.size < 2:
                return float(fallback)
            differences = np.diff(indices)
            if np.all(differences == differences[0]) and differences[0] > 0:
                return float(fallback) * float(differences[0])
            return float("nan")
        return float(fallback)

    def _current_shape(self) -> tuple[int, ...]:
        """Shape of the current view without forcing zarr materialization."""
        base_shape = self._base_shape()
        if self.slice_info is None:
            return base_shape
        try:
            return shape_after_index(base_shape, self.slice_info)
        except Exception:
            resolved = self._resolve_source()
            return tuple(int(v) for v in getattr(resolved, "shape", ()))

    def _resolve_source(self):
        """Return underlying data respecting the stored slice."""
        if self._materialized_data is not None:
            if self.slice_info is not None:
                return self._materialized_data[self.slice_info]
            return self._materialized_data
        if self.slice_info is not None:
            return self.zarr_array[self.slice_info]
        return self.zarr_array

    # ------------------------------------------------------------------ #
    # Public shape / data properties                                       #
    # ------------------------------------------------------------------ #

    @property
    def analysis_shape(self) -> tuple[int, ...]:
        """Shape with all dimensions preserved (used by .fft, .solitons, etc.)."""
        if self._index_plan is not None:
            return self._index_plan.analysis_shape
        return self._current_shape()

    @property
    def numpy_shape(self) -> tuple[int, ...]:
        """NumPy-like shape after dropping integer-indexed axes."""
        if self._index_plan is not None:
            return self._index_plan.numpy_shape
        return self._current_shape()

    @property
    def is_lazy(self) -> bool:
        """True when data has not been materialized to memory."""
        return self._materialized_data is None

    @property
    def is_materialized(self) -> bool:
        """True when data has been materialized (fancy index, downsample, etc.)."""
        return self._materialized_data is not None

    @property
    def estimated_nbytes(self) -> int:
        """Estimated byte size of the current view (lazy estimate)."""
        shape = self.analysis_shape
        n_elements = 1
        for s in shape:
            n_elements *= s
        dtype = getattr(self.zarr_array, "dtype", np.dtype("float32"))
        return n_elements * dtype.itemsize

    def numpy(
        self,
        *,
        copy: bool = True,
        dtype: Any | None = None,
        keepdims: bool = False,
        squeeze: bool = False,
    ) -> np.ndarray:
        """Materialize data as a NumPy array.

        Parameters
        ----------
        copy:
            Return a copy of the data (default True).
        dtype:
            Cast to this dtype if provided.
        keepdims:
            If True, preserve all dimensions (analysis shape); integer-indexed
            axes are *not* dropped.  Use this for analysis code that needs the
            full dimensional layout.
        squeeze:
            If True, remove *all* length-1 axes (standard ``np.squeeze``).
            Takes effect after ``keepdims`` processing.
        """
        # Warn on large materializations
        _LARGE_BYTES = 1 << 30  # 1 GiB
        if self.is_lazy and self.estimated_nbytes > _LARGE_BYTES:
            warnings.warn(
                f"Materializing ~{self.estimated_nbytes / (1 << 30):.1f} GiB. "
                "Consider using chunks, downsample, or .sel() first.",
                stacklevel=2,
            )

        source = self._resolve_source()
        # Zarr arrays and H5QuantityGroup-like objects (shape but not ndarray)
        # must be explicitly loaded via [:] because np.asarray would misinterpret
        # them (e.g., iterating over dict-like keys for H5 groups).
        if (
            not isinstance(source, np.ndarray)
            and hasattr(source, "shape")
            and hasattr(source, "__getitem__")
        ):
            source = source[:]
        arr = np.asarray(source)

        if not keepdims and self._index_plan is not None:
            dropped = self._index_plan.dropped_axes
            if dropped:
                # Remove only the axes selected by integer indexing
                for axis in sorted(dropped, reverse=True):
                    arr = np.squeeze(arr, axis=axis)

        if squeeze:
            arr = np.squeeze(arr)

        if dtype is not None:
            arr = arr.astype(dtype, copy=False)

        if copy and not arr.flags["OWNDATA"]:
            arr = arr.copy()

        return arr

    @property
    def values(self) -> np.ndarray:
        """Materialize and return data without copying (NumPy-like shape)."""
        return self.numpy(copy=False)

    @property
    def array(self) -> np.ndarray:
        """Alias for :attr:`values`."""
        return self.numpy(copy=False)

    @property
    def np(self) -> WrapperNumpyGetter:
        """Eager NumPy getter — returns plain ``np.ndarray`` on indexing.

        This is a shorthand for chained slicing that immediately materializes
        data with standard NumPy dimension-dropping semantics.

        Examples
        --------
        >>> arr = job[0].m.np[0, ..., 0]        # shape (nz, ny, nx)
        >>> arr = job[0].m[0, ...].np[..., 0]   # chained: shape (ny, nx)
        >>> arr.shape                             # plain np.ndarray
        """
        return WrapperNumpyGetter(self)

    def __array__(self, dtype=None, copy=None):
        arr = self.numpy(copy=False, dtype=dtype)
        if copy:
            arr = arr.copy()
        return arr

    def __getattr__(self, name):
        """Delegate to zarr_array for most attributes (but not our own properties)"""
        # Don't delegate properties that are defined on this class
        if name in (
            "dt",
            "fft",
            "analyze",
            "shape",
            "data",
            "plot",
            "geometry",
            "region",
            "cell",
            "analysis_shape",
            "numpy_shape",
            "is_lazy",
            "is_materialized",
            "values",
            "array",
            "np",
        ):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # Never forward numpy array protocol attributes — numpy uses these before
        # calling __array__, so forwarding them would bypass our squeezed shape.
        if name in (
            "__array_interface__",
            "__array_struct__",
            "__array_priority__",
            "__array_finalize__",
            "__array_wrap__",
            "__array_ufunc__",
            "__array_function__",
        ):
            raise AttributeError(
                f"'{self.__class__.__name__}' does not expose {name!r}"
            )

        if self._materialized_data is not None:
            source = self._resolve_source()
            return getattr(source, name)
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
        return normalize_index_key(key, ndim, keep_dims=True)

    def __getitem__(self, key):
        """Return new DatasetAwareWrapper with slicing info preserved.

        Integer indices are tracked via IndexPlan so ``.numpy()`` can drop
        those axes (NumPy semantics) while ``.fft`` keeps full dimensionality.
        """
        source_shape = self._current_shape()
        ndim = len(source_shape)

        # local_normalized_key is relative to the CURRENT view shape.
        # It is used for (a) materialized-data slicing and (b) compose_index_keys
        # with self.slice_info to build the new combined_slice.
        local_key_tuple = key if isinstance(key, tuple) else (key,)
        local_normalized_key = tuple(
            normalize_index_key(local_key_tuple, ndim, keep_dims=True)
        )

        # Build IndexPlan — when previous_plan exists its storage_key is already
        # fully composed relative to the ORIGINAL source, so do NOT use it in
        # compose_index_keys below (that would double-compose).
        new_plan = make_index_plan(key, source_shape, previous_plan=self._index_plan)

        if self._materialized_data is not None:
            sliced = np.asarray(self._materialized_data[local_normalized_key])
            geometry_override = None
            try:
                geometry_override = self.geometry.sliced(local_normalized_key)
            except Exception:
                geometry_override = self._geometry_override
            return DatasetAwareWrapper(
                self.job_result,
                self.dataset_name,
                self.zarr_array,
                slice_info=None,
                materialized_data=sliced,
                geometry_override=geometry_override,
                index_plan=new_plan,
                time_step_scale=self._materialized_time_step_scale(
                    local_normalized_key, self._time_step_scale
                ),
            )

        if not has_only_simple_slices(local_normalized_key, ndim):
            sliced = np.asarray(self._resolve_source()[local_normalized_key])
            geometry_override = None
            try:
                geometry_override = self.geometry.sliced(local_normalized_key)
            except Exception:
                geometry_override = self._geometry_override
            return DatasetAwareWrapper(
                self.job_result,
                self.dataset_name,
                self.zarr_array,
                slice_info=None,
                materialized_data=sliced,
                geometry_override=geometry_override,
                index_plan=new_plan,
                time_step_scale=self._materialized_time_step_scale(
                    local_normalized_key, self._time_step_scale
                ),
            )

        base_shape = self._base_shape()
        # Always compose local_normalized_key (relative to current view) with
        # self.slice_info (relative to original source) to get the new combined_slice.
        combined_slice = compose_index_keys(
            self.slice_info, local_normalized_key, base_shape
        )

        return DatasetAwareWrapper(
            self.job_result,
            self.dataset_name,
            self.zarr_array,  # Keep original zarr reference
            slice_info=combined_slice,
            materialized_data=None,
            geometry_override=self._geometry_override,
            index_plan=new_plan,
            time_step_scale=self._time_step_scale,
        )

    @property
    def fft(self):
        """Return FFT with this dataset pre-selected.

        When the wrapper holds materialized data (e.g. after .downsample()),
        the FFT interface will operate on that materialized view rather than
        re-loading the full dataset from storage.
        """
        if self._fft is None and FFT_AVAILABLE:
            # Pass materialized data through so FFT doesn't ignore it
            self._fft = DatasetSpecificFFT(
                self.job_result,
                self.dataset_name,
                getattr(self.job_result, "_mmpp_ref", None),
                slice_info=self.slice_info,
                materialized_data=self._materialized_data,
                index_plan=self._index_plan,
                time_step_scale=self._time_step_scale,
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
                dataset_view=self,
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
    def plot(self):
        """Plot accessor with dataset context pre-selected.

        Examples
        --------
        >>> job[0].geom[:].plot.snapshot()
        >>> job[0].m[:].plot.snapshot(z=0, t=-1)
        >>> job[0].regions[:].plot.snapshot(cmap='tab10')
        """
        if self._plot is None:
            self._plot = DatasetPlotAccessor(self)
        return self._plot

    @property
    def vortex(self):
        """Shortcut alias for ``self.solitons.vortex``."""
        return self.solitons.vortex

    @property
    def skyrmion(self):
        """Shortcut alias for ``self.solitons.skyrmion``."""
        return self.solitons.skyrmion

    @property
    def shape(self):
        """Shape accounting for slicing"""
        return self._current_shape()

    @property
    def geometry(self):
        """Physical geometry of the current dataset view."""
        return resolve_dataset_geometry(self, include_slice=True)

    @property
    def region(self):
        """Alias for ``geometry`` for field-style workflows."""
        return self.geometry

    @property
    def cell(self):
        """Spatial cell size of the current dataset view in meters."""
        geometry = self.geometry
        if not geometry.axes:
            return None
        return geometry.cell_xyz_m()

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
        if hasattr(self.job_result, "_z") and self.job_result._z is not None:
            if "t_sampl" in self.job_result._z.attrs:
                return self.job_result._z.attrs["t_sampl"]

        # Method 2: Check THIS dataset's attrs for 't' array (MOST SPECIFIC)
        if hasattr(self.job_result, "_z") and self.job_result._z is not None:
            try:
                dataset = self.job_result._z[self.dataset_name]
                if hasattr(dataset, "attrs") and "t" in dataset.attrs:
                    t_attr = dataset.attrs["t"]
                    # t_attr is a list or array in attrs
                    if hasattr(t_attr, "__len__") and len(t_attr) >= 2:
                        dt = float(t_attr[1] - t_attr[0])
                        return dt
            except (KeyError, NameError, AttributeError, IndexError, TypeError):
                pass

        # Method 3: Look for time array in various locations
        # Try common naming patterns and locations
        time_locations = [
            ("t",),  # Root level 't'
            ("table", "t"),  # Often in 'table' group
            ("time",),  # Alternative name
            (f"t_{self.dataset_name}",),  # Dataset-specific time
        ]

        for location in time_locations:
            try:
                if hasattr(self.job_result, "_z"):
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

    def to_numpy(self, **kwargs):
        """Alias for numpy() to match common API naming."""
        return self.numpy(**kwargs)

    @staticmethod
    def _normalize_frame_token(token: Any, *, axis: str):
        if token is None:
            return slice(None)
        if isinstance(token, slice):
            return token
        if isinstance(token, (tuple, list)):
            if len(token) == 2:
                return slice(token[0], token[1])
            if len(token) == 3:
                return slice(token[0], token[1], token[2])
            raise ValueError(
                f"frame selection for axis {axis!r} must have length 2 or 3, got {len(token)}"
            )
        if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
            return int(token)
        raise TypeError(
            f"Unsupported frame selection for axis {axis!r}: {token!r}. "
            "Use int, slice, None, or (start, stop[, step])."
        )

    def frame(
        self,
        *,
        t: Any = 0,
        z: Any = 0,
        y: Any = None,
        x: Any = None,
        c: Any = None,
    ) -> DatasetAwareWrapper:
        """Create an exact index-based analysis view without materialising the full dataset.

        This is intended for large simulation datasets such as ``m``, where plotting
        should start from an explicit ``t/z/y/x`` subset rather than from the whole field.

        Examples
        --------
        >>> view = job[0].m.frame(t=0, z=0, y=(0, 100), x=(0, 300))
        >>> view.plot.mpl.magnetization(multiplier=1e-9)
        """
        shape = self._current_shape()
        ndim = len(shape)
        y_key = self._normalize_frame_token(y, axis="y")
        x_key = self._normalize_frame_token(x, axis="x")
        c_key = self._normalize_frame_token(c, axis="c")

        if ndim == 5:
            return self[
                self._normalize_frame_token(t, axis="t"),
                self._normalize_frame_token(z, axis="z"),
                y_key,
                x_key,
                c_key,
            ]

        if ndim == 4:
            if int(shape[-1]) <= 4:
                if z not in (None, 0):
                    raise ValueError(
                        "frame(z=...) is not available on a 4D vector view shaped like (t, y, x, c); "
                        "slice z earlier or call frame(...) on the original 5D dataset."
                    )
                return self[
                    self._normalize_frame_token(t, axis="t"),
                    y_key,
                    x_key,
                    c_key,
                ]
            return self[
                self._normalize_frame_token(t, axis="t"),
                self._normalize_frame_token(z, axis="z"),
                y_key,
                x_key,
            ]

        if ndim == 3:
            if int(shape[-1]) <= 4:
                if t not in (None, 0) or z not in (None, 0):
                    raise ValueError(
                        "frame(t=..., z=...) is not available on a 3D vector plane shaped like (y, x, c)."
                    )
                return self[y_key, x_key, c_key]
            if t not in (None, 0):
                raise ValueError(
                    "frame(t=...) is not available on a 3D scalar volume shaped like (z, y, x)."
                )
            return self[
                self._normalize_frame_token(z, axis="z"),
                y_key,
                x_key,
            ]

        raise ValueError(
            f"frame(...) expects a 3D, 4D, or 5D dataset view, got shape {shape}"
        )

    @staticmethod
    def _normalize_downsample_spec(
        spec: tuple[Any, ...], ndim: int
    ) -> tuple[int | None, ...]:
        if len(spec) == 1 and isinstance(spec[0], tuple):
            tokens = list(spec[0])
        elif len(spec) == 1 and isinstance(spec[0], list):
            tokens = list(spec[0])
        else:
            tokens = list(spec)

        if not tokens:
            raise ValueError("downsample requires at least one axis specification")

        if tokens.count(Ellipsis) > 1:
            raise ValueError("downsample spec can contain at most one Ellipsis")

        if Ellipsis in tokens:
            idx = tokens.index(Ellipsis)
            missing = ndim - (len(tokens) - 1)
            if missing < 0:
                raise ValueError(
                    f"downsample spec has too many axes ({len(tokens)}) for ndim={ndim}"
                )
            tokens = tokens[:idx] + [slice(None)] * missing + tokens[idx + 1 :]

        if len(tokens) < ndim:
            tokens.extend([slice(None)] * (ndim - len(tokens)))

        if len(tokens) != ndim:
            raise ValueError(
                f"downsample spec must describe exactly {ndim} axes, got {len(tokens)}"
            )

        normalized: list[int | None] = []
        for token in tokens:
            if token is None:
                normalized.append(None)
                continue
            if isinstance(token, str):
                if token.strip() == ":":
                    normalized.append(None)
                    continue
                raise TypeError(
                    f"Invalid downsample token {token!r}; use ':'/None/slice(None) or int target size"
                )
            if isinstance(token, slice):
                if token.start is None and token.stop is None and token.step is None:
                    normalized.append(None)
                    continue
                raise TypeError(
                    f"Unsupported downsample slice {token!r}; only full slice ':' is supported"
                )
            if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
                normalized.append(int(token))
                continue
            raise TypeError(
                f"Invalid downsample token {token!r}; use ':'/None/slice(None) or int target size"
            )

        return tuple(normalized)

    @staticmethod
    def _block_mean_downsample_axis(
        array: Any,
        axis: int,
        target: int,
        *,
        strict: bool = False,
    ) -> Any:
        source = int(array.shape[axis])
        if target <= 0:
            raise ValueError(f"Target size must be > 0 for axis {axis}, got {target}")
        if target == source:
            return array
        if target > source:
            if strict:
                raise ValueError(
                    f"Cannot increase axis {axis} from {source} to {target} with block downsample"
                )
            warnings.warn(
                f"Skipping axis {axis}: target {target} > source {source}",
                RuntimeWarning,
                stacklevel=3,
            )
            return array

        scale = source // target
        if scale < 1:
            raise ValueError(
                f"Invalid downsample scale for axis {axis}: source={source}, target={target}"
            )

        trimmed = target * scale
        if trimmed != source:
            message = (
                f"Axis {axis}: source size {source} is not divisible by target {target}; "
                f"trimming to {trimmed} for block-mean downsampling"
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=3)
            indexer = [slice(None)] * array.ndim
            indexer[axis] = slice(0, trimmed)
            array = array[tuple(indexer)]

        new_shape = array.shape[:axis] + (target, scale) + array.shape[axis + 1 :]
        reduced = array.reshape(new_shape).mean(axis=axis + 1, dtype=np.float32)
        return np.asarray(reduced, dtype=np.float32)

    def downsample(self, *spec: Any, strict: bool = False) -> DatasetAwareWrapper:
        """Downsample current dataset view with block-mean aggregation.

        Parameters
        ----------
        *spec :
            Axis specification. For each axis:
            - ``":"`` / ``None`` / ``slice(None)`` keeps original size.
            - ``int`` sets target size for block-mean downsample.

            Examples:
            ``downsample(":", 1, 100, 100, ":")``
            ``downsample(np.s_[:, 1, 100, 100, :])``
        strict : bool, default False
            If True, raises errors when axis size is not divisible by target
            or when target > source. If False, trims trailing cells and skips
            invalid upsampling axes.
        """
        source = self.numpy(copy=False, squeeze=False)
        array = np.asarray(source, dtype=np.float32)
        targets = self._normalize_downsample_spec(spec, array.ndim)

        reduced = array
        for axis, target in enumerate(targets):
            if target is None:
                continue
            reduced = self._block_mean_downsample_axis(
                reduced,
                axis=axis,
                target=int(target),
                strict=bool(strict),
            )

        geometry_override = None
        try:
            geometry_override = self.geometry.resampled(tuple(reduced.shape))
        except Exception:
            geometry_override = None

        effective_time_scale = self._time_step_scale
        if self.slice_info is not None:
            key = (
                self.slice_info
                if isinstance(self.slice_info, tuple)
                else (self.slice_info,)
            )
            if key and isinstance(key[0], slice) and key[0].step is not None:
                effective_time_scale *= abs(int(key[0].step))
        if targets and targets[0] is not None and reduced.shape[0] > 0:
            effective_time_scale *= max(int(array.shape[0] // reduced.shape[0]), 1)

        return DatasetAwareWrapper(
            self.job_result,
            self.dataset_name,
            self.zarr_array,
            slice_info=None,
            materialized_data=np.asarray(reduced, dtype=np.float32),
            geometry_override=geometry_override,
            time_step_scale=effective_time_scale,
        )

    def sel(self, *axes: str, **coords: Any) -> DatasetAwareWrapper:
        """Select spatial region using physical coordinates, similar to discretisedfield.

        Examples
        --------
        >>> job[0].m.sel("z")
        >>> job[0].m.sel(x=25e-9, z=(0.0, 5e-9))
        """
        geometry = self.geometry
        if geometry.spatial_axes is None or not geometry.axes:
            raise TypeError(
                f"Dataset '{self.dataset_name}' has no resolvable spatial geometry for sel(...)"
            )

        selections: dict[str, Any] = {}
        for axis in axes:
            canonical = geometry.canonical_axis(axis)
            if canonical in selections:
                raise ValueError(f"Axis {axis!r} specified more than once")
            selections[canonical] = "__center__"

        for axis, value in coords.items():
            canonical = geometry.canonical_axis(axis)
            if canonical in selections:
                raise ValueError(f"Axis {axis!r} specified more than once")
            selections[canonical] = value

        if not selections:
            return self

        key = [slice(None)] * len(self.shape)
        for axis, value in selections.items():
            axis_geom = geometry.axes[axis]
            if axis_geom.index is None:
                # Missing spatial dimensions are represented as one virtual
                # cell. Selecting them must not slice the time/component axis.
                continue
            if value == "__center__":
                key[axis_geom.index] = axis_geom.center_slice()
                continue
            if isinstance(value, slice):
                key[axis_geom.index] = value
                continue
            if isinstance(value, (tuple, list)) and len(value) == 2:
                key[axis_geom.index] = axis_geom.select_range(
                    float(value[0]),
                    float(value[1]),
                )
                continue
            if isinstance(
                value, (int, float, np.integer, np.floating)
            ) and not isinstance(value, bool):
                key[axis_geom.index] = axis_geom.select_value(float(value))
                continue
            raise TypeError(
                f"Unsupported selection for axis {axis!r}: {value!r}. "
                "Use an axis name, scalar coordinate, or (min, max) tuple."
            )

        return self[tuple(key)]

    def as_zarr(self):
        """Return the underlying zarr.Array when no slicing is active."""
        if self.slice_info is not None or self._materialized_data is not None:
            raise TypeError(
                "Sliced view has no standalone zarr representation; use numpy() instead"
            )
        return self.zarr_array

    def __iter__(self):
        return iter(self.numpy(copy=False))

    def __len__(self):
        return len(self.numpy(copy=False))

    def _repr_html_(self) -> str:
        """Rich HTML card for Jupyter notebooks."""
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ADVANCED,
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            NODE_COLOR_PLOT,
            NODE_COLOR_UTIL,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        job_name = _html_escape(getattr(self.job_result, "name", "?"))
        ds_name_raw = str(self.dataset_name)
        ds_name = _html_escape(ds_name_raw)
        ds_expr = (
            f"job[0].{ds_name_raw}"
            if ds_name_raw.isidentifier()
            else f"job[0][{ds_name_raw!r}]"
        )
        shape_str = " &times; ".join(str(s) for s in self.analysis_shape)
        numpy_shape_str = " &times; ".join(str(s) for s in self.numpy_shape)
        state = "materialized" if self.is_materialized else "lazy"
        state_color = NODE_COLOR_ANALYSIS if self.is_materialized else "#6272a4"
        dtype = str(getattr(self.zarr_array, "dtype", "unknown"))

        # Chunks info (only for zarr arrays)
        try:
            chunks = getattr(self.zarr_array, "chunks", None)
            chunks_str = " &times; ".join(str(c) for c in chunks) if chunks else "n/a"
        except Exception:
            chunks_str = "n/a"

        metrics = [
            ("job", job_name, None),
            ("dataset", ds_name, NODE_COLOR_COMPUTE),
            ("analysis shape", shape_str, NODE_COLOR_UTIL),
            ("numpy shape", numpy_shape_str, NODE_COLOR_ANALYSIS),
            ("dtype", dtype, None),
            ("chunks", chunks_str, None),
            ("state", state, state_color),
            ("estimated size", f"{self.estimated_nbytes / (1024**2):.2f} MiB", None),
        ]
        if self.slice_info is not None:
            metrics.append(("slice", str(self.slice_info), NODE_COLOR_UTIL))

        accessors = accessors_section_html(
            [
                (
                    "Data:",
                    [
                        (".numpy()", NODE_COLOR_COMPUTE),
                        (".numpy(keepdims=True)", NODE_COLOR_COMPUTE),
                        (".values", NODE_COLOR_COMPUTE),
                        (".np[...]", NODE_COLOR_COMPUTE),
                    ],
                ),
                (
                    "Slicing:",
                    [
                        ("[t, z, y, x, c]", NODE_COLOR_PLOT),
                        (".sel(...)", NODE_COLOR_PLOT),
                        (".downsample(...)", NODE_COLOR_PLOT),
                        (".frame(...)", NODE_COLOR_PLOT),
                    ],
                ),
                (
                    "Analysis:",
                    [
                        (".fft", NODE_COLOR_ANALYSIS),
                        (".fft.spectrum()", NODE_COLOR_ANALYSIS),
                        (".solitons", NODE_COLOR_ANALYSIS),
                        (".analyze", NODE_COLOR_ANALYSIS),
                    ],
                ),
                (
                    "Geometry:",
                    [
                        (".analysis_shape", NODE_COLOR_UTIL),
                        (".numpy_shape", NODE_COLOR_UTIL),
                        (".geometry", NODE_COLOR_UTIL),
                        (".dt", NODE_COLOR_UTIL),
                    ],
                ),
                (
                    "Plot:",
                    [
                        (".plot.snapshot()", NODE_COLOR_ADVANCED),
                        (".plot.mpl.magnetization()", NODE_COLOR_ADVANCED),
                        (".plot.k3d.magnetization()", NODE_COLOR_ADVANCED),
                    ],
                ),
            ]
        )

        examples = examples_section_html(
            "\n".join(
                [
                    f"data = {ds_expr}",
                    "",
                    "# Lazy dataset view; slicing returns another wrapper",
                    "view = data[:, 0, :, :, 2]",
                    "",
                    "# Materialize intentionally",
                    "arr = view.numpy(copy=False)",
                    "",
                    "# Analysis and plotting stay bound to this dataset",
                    "spec = data.fft.spectrum()",
                    "data.plot.snapshot(t=-1, z=0)",
                ]
            )
        )

        api = api_help_html(
            self,
            title="Dataset API help",
            prefix=ds_expr,
            subtitle="Dataset wrapper with lazy slicing, NumPy materialization, geometry, plotting and analysis accessors.",
            properties=[
                ("fft", "FFT / spectrum analysis namespace"),
                ("solitons", "Soliton and vortex analysis namespace"),
                ("analyze", "General dataset-aware analysis namespace"),
                ("plot", "Plotting helpers"),
                ("np", "Eager NumPy getter"),
                ("values", "Materialized NumPy values"),
                ("geometry", "Physical geometry of the current view"),
                ("dt", "Time step inferred from metadata"),
            ],
            methods=["numpy", "to_numpy", "frame", "downsample", "sel", "as_zarr"],
            chrome=False,
        )

        uid = str(_uuid.uuid4())[:8]
        return node_card_html(
            "Dataset View",
            icon="🧲",
            subtitle=(
                f"<code style='color:{NODE_COLOR_COMPUTE}'>{ds_name}</code> "
                f"from <code>{job_name}</code>"
            ),
            badge=(state, state_color),
            sections=[metrics_section_html(metrics), accessors, examples],
            api=api,
            uid=f"dataset-{ds_name_raw}-{uid}",
        )

    def __repr__(self):
        slice_str = f"[{self.slice_info}]" if self.slice_info else ""
        return (
            f"DatasetAwareWrapper({self.dataset_name}{slice_str}, shape={self.shape})"
        )


class WrapperNumpyGetter:
    """Eager NumPy accessor for :class:`DatasetAwareWrapper`.

    Returned by the ``.np`` property.  Indexing immediately materializes the
    view as a plain ``np.ndarray`` with standard NumPy dimension-dropping
    semantics (integer indices remove axes).

    Examples
    --------
    >>> job[0].m.np[0, ..., 0]         # shape (nz, ny, nx)
    >>> job[0].m[0, ...].np[..., 0]   # shape (ny, nx)
    """

    def __init__(self, wrapper: DatasetAwareWrapper) -> None:
        self._wrapper = wrapper

    def __getitem__(self, key) -> np.ndarray:
        """Index the wrapper and return a plain ``np.ndarray``."""
        sliced = self._wrapper[key]
        return sliced.numpy(copy=False)

    @property
    def shape(self) -> tuple:
        """NumPy-like shape of the view (integer-indexed axes are dropped)."""
        return self._wrapper.numpy_shape

    @property
    def dtype(self):
        """Data type of the underlying dataset."""
        return self._wrapper.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions in the NumPy-like view."""
        return len(self._wrapper.numpy_shape)

    def __repr__(self) -> str:
        return (
            f"WrapperNumpyGetter({self._wrapper.dataset_name}, "
            f"shape={self._wrapper.analysis_shape})"
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
            raise AttributeError(f"Dataset '{name}' not found in zarr file") from None

        if isinstance(member, zarr.Array):
            return NumpyDatasetWrapper(self._job_result, name, member)
        raise AttributeError(f"'{name}' is not a dataset (it's a group)")

    def __getitem__(self, key: str) -> NumpyDatasetWrapper:
        """Get NumpyDatasetWrapper for dataset by item access (for special names)."""
        return self.__getattr__(key)

    def __repr__(self):
        self._job_result._ensure_zarr_loaded()
        datasets = list(self._job_result._z.array_keys())
        return (
            f"NumpyGetter(datasets={datasets[:5]}{'...' if len(datasets) > 5 else ''})"
        )
