import zarr
import numpy as np
import inspect
import warnings
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


class _DatasetMatplotlibPlotAccessor:
    """Matplotlib backend namespace for dataset-aware plotting."""

    def __init__(self, parent: "DatasetPlotAccessor"):
        self._parent = parent

    def __call__(self, **kwargs):
        return self._parent._mpl_auto_impl(**kwargs)

    def scalar(self, **kwargs):
        return self._parent._mpl_scalar_impl(**kwargs)

    def vector(self, **kwargs):
        return self._parent._mpl_vector_impl(**kwargs)

    def contour(self, **kwargs):
        return self._parent._mpl_contour_impl(**kwargs)

    def lightness(self, **kwargs):
        return self._parent._mpl_lightness_impl(**kwargs)

    def snapshot(self, **kwargs):
        return self._parent._snapshot_impl(**kwargs)

    def heatmap(self, **kwargs):
        return self._parent._heatmap_impl(**kwargs)

    def heamtp(self, **kwargs):
        """Compatibility alias for a common typo: ``heatmap``."""
        return self.heatmap(**kwargs)

    def __repr__(self):
        dset = self._parent._dataset.dataset_name
        return f"<DatasetMplPlotAccessor('{dset}')>"


class _DatasetK3DPlotAccessor:
    """K3D backend namespace for dataset-aware plotting."""

    def __init__(self, parent: "DatasetPlotAccessor"):
        self._parent = parent

    def __call__(self, **kwargs):
        return self.scalar(**kwargs)

    def scalar(self, **kwargs):
        return self._parent._k3d_scalar_impl(**kwargs)

    def vector(self, **kwargs):
        return self._parent._k3d_vector_impl(**kwargs)

    def nonzero(self, **kwargs):
        return self._parent._k3d_nonzero_impl(**kwargs)

    def heatmap(self, **kwargs):
        return self._parent._k3d_heatmap_impl(**kwargs)

    def __repr__(self):
        dset = self._parent._dataset.dataset_name
        return f"<DatasetK3DPlotAccessor('{dset}')>"


class DatasetPlotAccessor:
    """Plot accessor for :class:`DatasetAwareWrapper`.

    Examples
    --------
    >>> job[0].m[:].plot.snapshot()
    >>> job[0].m[:].plot.mpl.heatmap(component="mz")
    >>> job[0].m[:].plot.mpl.heamtp()  # alias
    """

    def __init__(self, dataset_wrapper: "DatasetAwareWrapper"):
        self._dataset = dataset_wrapper
        self._mpl = None
        self._k3d = None

    _SI_PREFIX_BY_EXP = {
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "u",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
    }

    @staticmethod
    def _normalize_index(index: int, size: int) -> int:
        idx = int(index)
        if idx < 0:
            idx = size + idx
        return int(np.clip(idx, 0, max(size - 1, 0)))

    def _resolve_dx_dy_nm(self) -> tuple[float, float]:
        attrs = getattr(self._dataset.job_result, "attrs", {})
        if hasattr(attrs, "get"):
            dx = float(attrs.get("dx", 1e-9))
            dy = float(attrs.get("dy", 1e-9))
        else:
            dx = 1e-9
            dy = 1e-9
        return dx * 1e9, dy * 1e9

    def _resolve_dx_dy_m(self) -> tuple[float, float]:
        attrs = getattr(self._dataset.job_result, "attrs", {})
        if hasattr(attrs, "get"):
            dx = float(attrs.get("dx", 1e-9))
            dy = float(attrs.get("dy", 1e-9))
        else:
            dx = 1e-9
            dy = 1e-9
        return dx, dy

    def _resolve_axis_names(self) -> tuple[str, str]:
        attrs = getattr(self._dataset.job_result, "attrs", {})
        if hasattr(attrs, "get"):
            x_name = str(attrs.get("x_name", "x"))
            y_name = str(attrs.get("y_name", "y"))
        else:
            x_name, y_name = "x", "y"
        return x_name, y_name

    @classmethod
    def _auto_si_multiplier(cls, lengths_m: tuple[float, ...]) -> float:
        finite = [
            abs(float(value))
            for value in lengths_m
            if np.isfinite(value) and abs(float(value)) > 0.0
        ]
        if not finite:
            return 1.0

        vmax = max(finite)
        exp3 = int(np.floor(np.log10(vmax) / 3.0) * 3)
        exp3 = int(np.clip(exp3, -15, 12))
        return float(10.0**exp3)

    @classmethod
    def _unit_label_from_multiplier(cls, multiplier: float) -> str:
        if multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")

        exp = np.log10(multiplier)
        exp_round = int(round(exp))
        if abs(exp - exp_round) < 1e-10 and exp_round in cls._SI_PREFIX_BY_EXP:
            return f"{cls._SI_PREFIX_BY_EXP[exp_round]}m"
        if np.isclose(multiplier, 1.0):
            return "m"
        return f"{multiplier:g} m"

    def _resolve_plot_geometry(
        self,
        shape_xy: tuple[int, int],
        *,
        multiplier: Optional[float] = None,
    ) -> tuple[float, float, tuple[float, float, float, float], float, str]:
        dx_m, dy_m = self._resolve_dx_dy_m()
        ny, nx = int(shape_xy[0]), int(shape_xy[1])
        size_x = float(nx) * dx_m
        size_y = float(ny) * dy_m

        if multiplier is None:
            m = self._auto_si_multiplier((size_x, size_y))
        else:
            m = float(multiplier)
            if m <= 0:
                raise ValueError(f"multiplier must be > 0, got {m}")

        dx_u = dx_m / m
        dy_u = dy_m / m
        extent = (0.0, float(nx) * dx_u, 0.0, float(ny) * dy_u)
        unit_label = self._unit_label_from_multiplier(m)
        return dx_u, dy_u, extent, m, unit_label

    def _set_axis_labels(self, ax, unit_label: str) -> None:
        x_name, y_name = self._resolve_axis_names()
        ax.set_xlabel(f"{x_name} ({unit_label})")
        ax.set_ylabel(f"{y_name} ({unit_label})")

    def _extract_frame(
        self,
        *,
        z: int = 0,
        t: int = -1,
        zero: Optional[int] = None,
    ) -> np.ndarray:
        data = self._dataset.numpy(copy=False, squeeze=False)
        arr = np.asarray(data, dtype=np.float32)
        ndim = arr.ndim

        if ndim == 5:
            t_idx = self._normalize_index(t, arr.shape[0])
            z_idx = self._normalize_index(z, arr.shape[1])
            frame = np.asarray(arr[t_idx, z_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                frame = frame - np.asarray(arr[zref_idx, z_idx], dtype=np.float32)
            return frame

        if ndim == 4:
            t_idx = self._normalize_index(t, arr.shape[0])
            frame = np.asarray(arr[t_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                frame = frame - np.asarray(arr[zref_idx], dtype=np.float32)
            return frame

        if ndim == 3:
            if arr.shape[-1] <= 3:
                return arr
            z_idx = self._normalize_index(z, arr.shape[0])
            return np.asarray(arr[z_idx], dtype=np.float32)

        if ndim == 2:
            return arr

        raise ValueError(
            f"Dataset '{self._dataset.dataset_name}' has unsupported shape {arr.shape} for plotting"
        )

    def _extract_sequence(
        self,
        *,
        z: int = 0,
        zero: Optional[int] = None,
    ) -> np.ndarray:
        data = self._dataset.numpy(copy=False, squeeze=False)
        arr = np.asarray(data, dtype=np.float32)

        if arr.ndim == 5:
            z_idx = self._normalize_index(z, arr.shape[1])
            seq = np.asarray(arr[:, z_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                seq = seq - np.asarray(arr[zref_idx, z_idx], dtype=np.float32)
            return seq

        if arr.ndim == 4:
            # Typical magnetization after slicing: (t, y, x, c)
            if arr.shape[-1] <= 4:
                seq = np.asarray(arr, dtype=np.float32)
                if zero is not None:
                    zref_idx = self._normalize_index(zero, arr.shape[0])
                    seq = seq - np.asarray(arr[zref_idx], dtype=np.float32)
                return seq

            # Scalar-with-z volume over time: (t, z, y, x)
            z_idx = self._normalize_index(z, arr.shape[1])
            seq = np.asarray(arr[:, z_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                seq = seq - np.asarray(arr[zref_idx, z_idx], dtype=np.float32)
            return seq

        # No explicit time axis: wrap as single-frame sequence.
        frame = self._extract_frame(z=z, t=-1, zero=None)
        if zero is not None:
            frame = frame - frame
        return frame[np.newaxis, ...]

    def _extract_volume(
        self,
        *,
        t: int = -1,
        zero: Optional[int] = None,
    ) -> np.ndarray:
        """Extract 3d (scalar) or 4d (vector) volume for volumetric plotting."""
        data = self._dataset.numpy(copy=False, squeeze=False)
        arr = np.asarray(data, dtype=np.float32)

        if arr.ndim == 5:
            t_idx = self._normalize_index(t, arr.shape[0])
            volume = np.asarray(arr[t_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                volume = volume - np.asarray(arr[zref_idx], dtype=np.float32)
            return volume

        if arr.ndim == 4:
            # Heuristic:
            # - last axis <= 4 -> vector volume (z, y, x, c)
            # - otherwise first axis is treated as time in scalar volume (t, z, y, x)
            if arr.shape[-1] <= 4:
                return np.asarray(arr, dtype=np.float32)
            t_idx = self._normalize_index(t, arr.shape[0])
            volume = np.asarray(arr[t_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                volume = volume - np.asarray(arr[zref_idx], dtype=np.float32)
            return volume

        if arr.ndim == 3:
            return np.asarray(arr, dtype=np.float32)

        raise ValueError(
            f"Dataset '{self._dataset.dataset_name}' has unsupported shape {arr.shape} for volumetric plotting"
        )

    @staticmethod
    def _component_volume(
        volume: np.ndarray,
        component: Optional[Union[int, str]],
        *,
        default: str = "norm",
    ) -> np.ndarray:
        """Select scalar component from a volumetric scalar/vector array."""
        arr = np.asarray(volume, dtype=np.float32)

        if arr.ndim == 3:
            return arr

        if arr.ndim != 4:
            raise ValueError(f"Volume must be 3D or 4D, got shape {arr.shape}")

        n_comp = int(arr.shape[-1])
        if n_comp < 1:
            raise ValueError("Vector volume has no components")

        comp = default if component is None else component

        if isinstance(comp, (int, np.integer)) and not isinstance(comp, bool):
            idx = int(comp)
            if idx < 0 or idx >= n_comp:
                raise IndexError(
                    f"component={idx} is out of range for volume with {n_comp} components"
                )
            return np.asarray(arr[..., idx], dtype=np.float32)

        if isinstance(comp, str):
            key = comp.strip().lower()
            mapping = {"x": 0, "mx": 0, "y": 1, "my": 1, "z": 2, "mz": 2}
            if key in mapping:
                idx = mapping[key]
                if idx >= n_comp:
                    raise IndexError(
                        f"component='{comp}' requires component index {idx}, "
                        f"but volume has only {n_comp} components"
                    )
                return np.asarray(arr[..., idx], dtype=np.float32)
            if key in {"norm", "magnitude", "|m|"}:
                return np.linalg.norm(arr[..., : min(3, n_comp)], axis=-1).astype(
                    np.float32,
                    copy=False,
                )

        raise ValueError(
            f"Unsupported volumetric component selector: {component!r}. "
            "Use int, x/y/z, mx/my/mz, norm/magnitude."
        )

    @staticmethod
    def _coerce_mask(mask_like: Any, target_shape: tuple[int, ...]) -> np.ndarray:
        """Convert mask-like input to boolean array broadcastable to target shape."""
        if mask_like is None:
            return np.ones(target_shape, dtype=bool)

        raw = mask_like.numpy(copy=False, squeeze=False) if hasattr(mask_like, "numpy") else mask_like
        mask = np.asarray(raw, dtype=np.float32)
        mask = np.squeeze(mask)

        # If vector mask is provided, use its norm.
        if mask.ndim == len(target_shape) + 1 and mask.shape[-1] <= 4:
            mask = np.linalg.norm(mask[..., : min(3, mask.shape[-1])], axis=-1)

        if mask.shape != target_shape:
            try:
                mask = np.broadcast_to(mask, target_shape)
            except ValueError as exc:
                raise ValueError(
                    f"Mask shape {mask.shape} is not broadcastable to target shape {target_shape}"
                ) from exc

        return np.asarray(mask != 0, dtype=bool)

    def _resolve_dxyz_nm(self) -> tuple[float, float, float]:
        attrs = getattr(self._dataset.job_result, "attrs", {})
        if hasattr(attrs, "get"):
            dx = float(attrs.get("dx", 1e-9))
            dy = float(attrs.get("dy", 1e-9))
            dz = float(attrs.get("dz", 1e-9))
        else:
            dx = dy = dz = 1e-9
        return dx * 1e9, dy * 1e9, dz * 1e9

    @staticmethod
    def _normalise_to_uint8(
        values: np.ndarray,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        visible_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if vmin is None:
            lo = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
        else:
            lo = float(vmin)
        if vmax is None:
            hi = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
        else:
            hi = float(vmax)

        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = 1.0
        if hi <= lo:
            hi = lo + 1e-12

        scaled = (arr - lo) / (hi - lo)
        scaled = np.clip(scaled, 0.0, 1.0)
        out = (scaled * 254.0 + 1.0).astype(np.uint8)
        out[~np.isfinite(arr)] = 0

        if visible_mask is not None:
            mask = np.asarray(visible_mask, dtype=bool)
            if mask.shape != out.shape:
                mask = np.broadcast_to(mask, out.shape)
            out[~mask] = 0

        return out

    @staticmethod
    def _k3d_colormap_int(cmap_name: str) -> list[int]:
        import matplotlib.colors as mpl_colors
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(cmap_name or "viridis")
        cmap_int: list[int] = []
        for i in range(int(getattr(cmap, "N", 256))):
            rgb = cmap(i)[:3]
            cmap_int.append(int(mpl_colors.rgb2hex(rgb)[1:], 16))
        return cmap_int

    @staticmethod
    def _component_image(
        frame: np.ndarray,
        component: Optional[Union[int, str]],
        *,
        default: str = "norm",
    ) -> np.ndarray:
        if frame.ndim == 2:
            return np.asarray(frame, dtype=np.float32)

        if frame.ndim < 2:
            raise ValueError(f"Frame must be at least 2D, got shape {frame.shape}")

        if frame.ndim > 3:
            image = np.asarray(frame, dtype=np.float32)
            while image.ndim > 2:
                image = image[..., 0]
            return image

        n_comp = int(frame.shape[-1])
        if n_comp < 1:
            raise ValueError("Vector frame has no components")

        comp = default if component is None else component

        if isinstance(comp, (int, np.integer)) and not isinstance(comp, bool):
            idx = int(comp)
            if idx < 0 or idx >= n_comp:
                raise IndexError(
                    f"component={idx} is out of range for frame with {n_comp} components"
                )
            return np.asarray(frame[..., idx], dtype=np.float32)

        if isinstance(comp, str):
            key = comp.strip().lower()
            mapping = {"x": 0, "mx": 0, "y": 1, "my": 1, "z": 2, "mz": 2}
            if key in mapping:
                idx = mapping[key]
                if idx >= n_comp:
                    raise IndexError(
                        f"component='{comp}' requires component index {idx}, "
                        f"but frame has only {n_comp} components"
                    )
                return np.asarray(frame[..., idx], dtype=np.float32)
            if key in {"norm", "magnitude", "|m|", "snapshot"}:
                return np.linalg.norm(frame[..., : min(3, n_comp)], axis=-1).astype(
                    np.float32,
                    copy=False,
                )

        raise ValueError(
            f"Unsupported component selector: {component!r}. "
            "Use int, x/y/z, mx/my/mz, norm/magnitude."
        )

    @staticmethod
    def _resolve_component_index(
        token: Optional[Union[int, str]],
        n_comp: int,
        *,
        allow_none: bool = True,
    ) -> Optional[int]:
        if token is None:
            if allow_none:
                return None
            raise ValueError("Component token cannot be None in this context")

        if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
            idx = int(token)
            if idx < 0 or idx >= n_comp:
                raise IndexError(
                    f"component index {idx} is out of range for {n_comp} components"
                )
            return idx

        if isinstance(token, str):
            key = token.strip().lower()
            mapping = {"x": 0, "mx": 0, "y": 1, "my": 1, "z": 2, "mz": 2}
            if key in mapping:
                idx = mapping[key]
                if idx >= n_comp:
                    raise IndexError(
                        f"component '{token}' requires index {idx}, "
                        f"but only {n_comp} components are available"
                    )
                return idx
            raise ValueError(
                f"Unsupported component label {token!r}. Use x/y/z or mx/my/mz or int."
            )

        raise TypeError(
            f"Unsupported component token type {type(token).__name__}; use int/str/None"
        )

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        value = str(mode).strip().lower()
        if value in {"snapshot", "vector", "quiver"}:
            return "snapshot"
        if value in {"heatmap", "scalar", "mpl_heatmap"}:
            return "heatmap"
        raise ValueError(f"Unsupported render mode: {mode!r}. Use 'snapshot' or 'heatmap'.")

    @staticmethod
    def _k3d_colormap(cmap_name: str) -> Optional[list[float]]:
        """Build K3D-compatible colormap from k3d built-ins or matplotlib."""
        name = str(cmap_name or "viridis").strip()
        if not name:
            name = "viridis"

        try:
            import k3d

            mpl_maps = getattr(getattr(k3d, "colormaps", None), "matplotlib_color_maps", None)
            if mpl_maps is not None:
                candidates = [
                    name,
                    name.capitalize(),
                    name.title().replace("_", ""),
                    name.replace("_", "").capitalize(),
                ]
                for candidate in candidates:
                    if hasattr(mpl_maps, candidate):
                        value = getattr(mpl_maps, candidate)
                        if isinstance(value, list) and value:
                            return [float(v) for v in value]
        except Exception:
            pass

        try:
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap(name)
            samples = 256
            data: list[float] = []
            denom = max(samples - 1, 1)
            for i in range(samples):
                x = float(i) / float(denom)
                r, g, b, _ = cmap(x)
                data.extend([x, float(r), float(g), float(b)])
            return data
        except Exception:
            return None

    @staticmethod
    def _k3d_color_range(image: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> list[float]:
        if vmin is not None and vmax is not None:
            lo = float(vmin)
            hi = float(vmax)
        else:
            lo = float(np.nanmin(image)) if vmin is None else float(vmin)
            hi = float(np.nanmax(image)) if vmax is None else float(vmax)
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = 1.0
        if hi <= lo:
            hi = lo + 1e-12
        return [lo, hi]

    @staticmethod
    def _k3d_clear_plot(plot, keep_names: tuple[str, ...] = ("total_region",)) -> None:
        try:
            objects = list(getattr(plot, "objects", []))
        except Exception:
            return

        for obj in objects:
            name = getattr(obj, "name", None)
            if name in keep_names:
                continue
            try:
                plot -= obj
            except Exception:
                continue

    def _mpl_auto_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: Optional[tuple[float, float]] = None,
        multiplier: Optional[float] = None,
        zero: Optional[int] = None,
        scalar_kw: Optional[dict[str, Any]] = None,
        vector_kw: Optional[dict[str, Any]] = None,
        filename: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt

        scalar_kw = {} if scalar_kw is None else dict(scalar_kw)
        vector_kw = {} if vector_kw is None else dict(vector_kw)

        frame = self._extract_frame(z=z, t=t, zero=zero)
        is_vector = frame.ndim == 3 and frame.shape[-1] >= 2

        if not is_vector:
            scalar_kw.setdefault("multiplier", multiplier)
            ax = self._mpl_scalar_impl(z=z, t=t, ax=ax, zero=zero, **scalar_kw)
        else:
            if ax is None:
                base = self._component_image(frame, "norm", default="norm")
                shape_ratio = base.shape[1] / max(base.shape[0], 1)
                fig_size = figsize if figsize is not None else (4 * shape_ratio, 4)
                _, ax = plt.subplots(1, 1, figsize=fig_size, dpi=110)

            scalar_kw.setdefault("component", "norm")
            scalar_kw.setdefault("colorbar", True)
            scalar_kw.setdefault("multiplier", multiplier)
            self._mpl_scalar_impl(z=z, t=t, ax=ax, zero=zero, **scalar_kw)

            vector_kw.setdefault("use_color", False)
            vector_kw.setdefault("colorbar", False)
            vector_kw.setdefault("title", None)
            vector_kw.setdefault("multiplier", multiplier)
            ax = self._mpl_vector_impl(z=z, t=t, ax=ax, zero=zero, **vector_kw)

        if filename is not None:
            plt.savefig(str(filename), bbox_inches="tight", pad_inches=0.02)
        return ax

    def _mpl_scalar_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: Optional[tuple[float, float]] = None,
        multiplier: Optional[float] = None,
        repeat: int = 1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        filter_field: Any = None,
        cmap: str = "viridis",
        colorbar: bool = True,
        colorbar_label: str = "",
        symmetric_clim: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
        filename: Optional[str] = None,
        **imshow_kwargs,
    ):
        import matplotlib.pyplot as plt

        frame = self._extract_frame(z=z, t=t, zero=zero)
        image = self._component_image(frame, component, default="norm")
        image = np.tile(image, (max(int(repeat), 1), max(int(repeat), 1)))
        image = np.asarray(image, dtype=np.float32)

        if filter_field is not None:
            mask = self._coerce_mask(filter_field, image.shape)
            image = np.where(mask, image, np.nan)

        if "clim" in imshow_kwargs and (vmin is None and vmax is None):
            clim = imshow_kwargs.pop("clim")
            if isinstance(clim, (tuple, list)) and len(clim) == 2:
                vmin, vmax = float(clim[0]), float(clim[1])

        if symmetric_clim and vmin is None and vmax is None:
            local_min = float(np.nanmin(image)) if np.isfinite(image).any() else 0.0
            local_max = float(np.nanmax(image)) if np.isfinite(image).any() else 0.0
            vmax_abs = max(abs(local_min), abs(local_max))
            vmin = -vmax_abs
            vmax = vmax_abs

        if ax is None:
            shape_ratio = image.shape[1] / max(image.shape[0], 1)
            fig_size = figsize if figsize is not None else (4 * shape_ratio, 4)
            _, ax = plt.subplots(1, 1, figsize=fig_size, dpi=110)

        _, _, extent, _, unit_label = self._resolve_plot_geometry(
            image.shape,
            multiplier=multiplier,
        )
        im = ax.imshow(
            image,
            origin="lower",
            interpolation=imshow_kwargs.pop("interpolation", "none"),
            aspect=imshow_kwargs.pop("aspect", "equal"),
            extent=extent,
            cmap=imshow_kwargs.pop("cmap", cmap),
            vmin=vmin,
            vmax=vmax,
            **imshow_kwargs,
        )

        if colorbar:
            cb = self._mpl_add_colorbar(ax, im, colorbar_label or None)
            if colorbar_label:
                cb.set_label(str(colorbar_label))

        if title is None:
            comp_label = "norm" if component is None else str(component)
            title = f"{self._dataset.dataset_name} [{comp_label}]"
        ax.set(title=title)
        ax.set_aspect("equal")
        self._set_axis_labels(ax, unit_label)

        if filename is not None:
            plt.savefig(str(filename), bbox_inches="tight", pad_inches=0.02)
        return ax

    def _mpl_vector_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: Optional[tuple[float, float]] = None,
        multiplier: Optional[float] = None,
        repeat: int = 1,
        zero: Optional[int] = None,
        filter_field: Any = None,
        vdims: Optional[tuple[Optional[Union[int, str]], Optional[Union[int, str]]]] = None,
        color_field: Optional[Union[int, str, np.ndarray]] = None,
        cmap: str = "viridis",
        use_color: bool = True,
        colorbar: bool = True,
        colorbar_label: str = "",
        quiver_density: int = 20,
        vector_scale: Optional[float] = None,
        pivot: str = "mid",
        title: Optional[str] = None,
        filename: Optional[str] = None,
        **quiver_kwargs,
    ):
        import matplotlib.pyplot as plt

        frame = self._extract_frame(z=z, t=t, zero=zero)
        if frame.ndim != 3 or frame.shape[-1] < 2:
            raise ValueError(
                f"Vector plotting expects frame shape (y, x, c>=2), got {frame.shape}"
            )

        src_n_comp = int(frame.shape[-1])
        vec = np.asarray(frame, dtype=np.float32)
        if vec.shape[-1] < 3:
            padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
            padded[..., : vec.shape[-1]] = vec
            vec = padded
        vec = np.tile(vec, (max(int(repeat), 1), max(int(repeat), 1), 1))
        n_comp = int(src_n_comp)

        if vdims is None:
            arrow_x = 0
            arrow_y = 1 if n_comp >= 2 else None
        else:
            if len(vdims) != 2:
                raise ValueError(f"{vdims=} must contain exactly 2 elements")
            arrow_x = self._resolve_component_index(vdims[0], n_comp, allow_none=True)
            arrow_y = self._resolve_component_index(vdims[1], n_comp, allow_none=True)
            if arrow_x is None and arrow_y is None:
                raise ValueError(f"At least one element in {vdims=} must not be None")

        u = (
            np.asarray(vec[:, :, arrow_x], dtype=np.float32)
            if arrow_x is not None
            else np.zeros(vec.shape[:2], dtype=np.float32)
        )
        v = (
            np.asarray(vec[:, :, arrow_y], dtype=np.float32)
            if arrow_y is not None
            else np.zeros(vec.shape[:2], dtype=np.float32)
        )
        w = np.asarray(vec[:, :, 2], dtype=np.float32) if n_comp >= 3 else np.zeros_like(u)

        if filter_field is not None:
            mask = self._coerce_mask(filter_field, u.shape)
        else:
            mask = np.ones_like(u, dtype=bool)

        dens = max(int(quiver_density), 1)
        stepx = max(int(u.shape[1] / dens), 1)
        stepy = max(int(u.shape[0] / dens), 1)

        u_ds = np.asarray(u[::stepy, ::stepx], dtype=np.float32)
        v_ds = np.asarray(v[::stepy, ::stepx], dtype=np.float32)
        mask_ds = np.asarray(mask[::stepy, ::stepx], dtype=bool)
        u_ds = np.where(mask_ds, u_ds, np.nan)
        v_ds = np.where(mask_ds, v_ds, np.nan)

        if ax is None:
            shape_ratio = u.shape[1] / max(u.shape[0], 1)
            fig_size = figsize if figsize is not None else (4 * shape_ratio, 4)
            _, ax = plt.subplots(1, 1, figsize=fig_size, dpi=110)

        dx_u, dy_u, _, _, unit_label = self._resolve_plot_geometry(
            u.shape,
            multiplier=multiplier,
        )
        x, y = np.meshgrid(
            np.arange(0, u.shape[1], stepx) * dx_u,
            np.arange(0, u.shape[0], stepy) * dy_u,
        )

        c_ds = None
        if use_color:
            if color_field is None:
                if n_comp == 3:
                    preferred = [0, 1, 2]
                    for used in (arrow_x, arrow_y):
                        if used in preferred:
                            preferred.remove(used)
                    color_idx = preferred[0] if preferred else 2
                    c_full = np.asarray(vec[:, :, color_idx], dtype=np.float32)
                else:
                    warnings.warn(
                        "Automatic coloring is only supported for 3-component vectors. "
                        f"Ignoring '{use_color=}'.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    use_color = False
            elif isinstance(color_field, (int, np.integer, str)):
                c_full = self._component_image(vec, color_field, default="norm")
            else:
                c_full = np.asarray(color_field, dtype=np.float32)
                c_full = np.squeeze(c_full)
                if c_full.shape != u.shape:
                    c_full = np.broadcast_to(c_full, u.shape)
            if use_color:
                c_ds = np.asarray(c_full[::stepy, ::stepx], dtype=np.float32)
                c_ds = np.where(mask_ds, c_ds, np.nan)

        quiver_kw = dict(quiver_kwargs)
        passed_scale = quiver_kw.pop("scale", None)
        if passed_scale is not None:
            scale_value = float(passed_scale)
        elif vector_scale is not None:
            scale_value = float(vector_scale)
        else:
            scale_value = 1.0 / max(stepx, stepy)
        quiver_kw.setdefault("angles", "xy")
        quiver_kw.setdefault("scale_units", "xy")
        quiver_kw.setdefault("pivot", pivot)
        quiver_kw["scale"] = float(scale_value)

        if c_ds is None:
            quiver = ax.quiver(
                x,
                y,
                u_ds,
                v_ds,
                **quiver_kw,
            )
        else:
            quiver_kw.setdefault("cmap", cmap)
            quiver = ax.quiver(
                x,
                y,
                u_ds,
                v_ds,
                c_ds,
                **quiver_kw,
            )

        if colorbar and c_ds is not None:
            cb = self._mpl_add_colorbar(ax, quiver, colorbar_label or None)
            if colorbar_label:
                cb.set_label(str(colorbar_label))

        if title is None:
            title = f"{self._dataset.dataset_name} [vector]"
        ax.set(title=title)
        ax.set_aspect("equal")
        self._set_axis_labels(ax, unit_label)

        if filename is not None:
            plt.savefig(str(filename), bbox_inches="tight", pad_inches=0.02)
        return ax

    def _mpl_contour_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: Optional[tuple[float, float]] = None,
        multiplier: Optional[float] = None,
        repeat: int = 1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        filter_field: Any = None,
        levels: int = 12,
        filled: bool = True,
        cmap: str = "viridis",
        colorbar: bool = True,
        colorbar_label: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
        filename: Optional[str] = None,
        **contour_kwargs,
    ):
        import matplotlib.pyplot as plt

        frame = self._extract_frame(z=z, t=t, zero=zero)
        image = self._component_image(frame, component, default="norm")
        image = np.tile(image, (max(int(repeat), 1), max(int(repeat), 1)))
        image = np.asarray(image, dtype=np.float32)

        if filter_field is not None:
            mask = self._coerce_mask(filter_field, image.shape)
            image = np.where(mask, image, np.nan)

        if ax is None:
            shape_ratio = image.shape[1] / max(image.shape[0], 1)
            fig_size = figsize if figsize is not None else (4 * shape_ratio, 4)
            _, ax = plt.subplots(1, 1, figsize=fig_size, dpi=110)

        dx_u, dy_u, _, _, unit_label = self._resolve_plot_geometry(
            image.shape,
            multiplier=multiplier,
        )
        x = np.arange(image.shape[1], dtype=np.float32) * dx_u
        y = np.arange(image.shape[0], dtype=np.float32) * dy_u

        if filled:
            cp = ax.contourf(
                x,
                y,
                image,
                levels=max(int(levels), 2),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **contour_kwargs,
            )
        else:
            cp = ax.contour(
                x,
                y,
                image,
                levels=max(int(levels), 2),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **contour_kwargs,
            )

        if colorbar:
            self._mpl_add_colorbar(ax, cp, colorbar_label)

        if title is None:
            comp_label = "norm" if component is None else str(component)
            title = f"{self._dataset.dataset_name} [contour:{comp_label}]"
        ax.set(title=title)
        ax.set_aspect("equal")
        self._set_axis_labels(ax, unit_label)

        if filename is not None:
            plt.savefig(str(filename), bbox_inches="tight", pad_inches=0.02)
        return ax

    @staticmethod
    def _mpl_add_colorwheel(
        ax,
        *,
        width=1.0,
        height=1.0,
        loc: str = "lower right",
        **kwargs,
    ):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        from ..plotting import hsl2rgb

        n = 200
        x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        theta = np.mod(np.arctan2(yy, xx) + 2.0 * np.pi, 2.0 * np.pi)
        radius = np.sqrt(xx**2 + yy**2)

        hsl = np.ones((n, n, 3), dtype=np.float32)
        hsl[:, :, 0] = theta / (2.0 * np.pi)
        hsl[:, :, 1] = 1.0
        hsl[:, :, 2] = np.clip(radius / np.sqrt(2.0), 0.0, 1.0)
        rgb = hsl2rgb(hsl)

        rgba = np.zeros((n, n, 4), dtype=np.float32)
        inside = radius <= 1.0
        rgba[inside, :3] = rgb[inside]
        rgba[inside, 3] = 1.0

        cw_ax = inset_axes(ax, width=width, height=height, loc=loc, **kwargs)
        cw_ax.imshow(rgba, origin="lower")
        cw_ax.axis("off")
        return cw_ax

    @staticmethod
    def _mpl_add_colorbar(
        ax,
        mappable,
        colorbar_label: Optional[str] = None,
        *,
        min_height_inches: float = 2.0,
        min_width_inches: float = 0.35,
        min_pad_inches: float = 0.1,
    ):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import Size, make_axes_locatable

        fig = ax.figure
        fig_w, fig_h = fig.get_size_inches()
        pos = ax.get_position()

        min_height_norm = min_height_inches / (fig_h * max(pos.y1 - pos.y0, 1e-12))
        min_width_norm = min_width_inches / max(fig_w, 1e-12)
        min_pad_norm = min_pad_inches / max(fig_w, 1e-12)

        if min_pad_norm > 0.05:
            pad_h = Size.Fixed(min_pad_inches)
        else:
            pad_h = Size.AxesX(ax, aspect=0.05)

        if min_width_norm > 0.05:
            width_h = Size.Fixed(min_width_inches)
        else:
            width_h = Size.AxesX(ax, aspect=0.05)

        v_aspect = min_height_norm if min_height_norm > 1 else 1
        existing_cbs = [a for a in fig.get_axes() if f"cb_{id(ax)}" in a.get_label()]
        divider = make_axes_locatable(ax)
        cax = fig.add_axes(
            divider.get_position(),
            label=f"cb_{id(ax)}_{len(existing_cbs)}",
        )

        if len(existing_cbs) == 0:
            divider.set_horizontal([Size.AxesX(ax), pad_h, width_h])
        else:
            divider.new_horizontal(pad_h, pack_start=False)
            divider.new_horizontal(width_h, pack_start=False)

        divider.set_vertical([Size.AxesY(ax, aspect=v_aspect)])
        ax.set_axes_locator(divider.new_locator(nx=0, ny=0))

        for i, cb in enumerate(existing_cbs, start=1):
            cb.set_axes_locator(divider.new_locator(nx=2 * i, ny=0))
        cax.set_axes_locator(divider.new_locator(nx=2 * (len(existing_cbs) + 1), ny=0))

        cbar = plt.colorbar(mappable, cax=cax)
        if colorbar_label is not None and colorbar_label != "":
            cbar.ax.set_ylabel(str(colorbar_label))
        return cbar

    def _mpl_lightness_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: Optional[tuple[float, float]] = None,
        multiplier: Optional[float] = None,
        repeat: int = 1,
        zero: Optional[int] = None,
        filter_field: Any = None,
        lightness_field: Optional[Union[int, str, np.ndarray]] = None,
        clim: Optional[tuple[float, float]] = None,
        colorwheel: bool = True,
        colorwheel_xlabel: Optional[str] = None,
        colorwheel_ylabel: Optional[str] = None,
        colorwheel_args: Optional[dict[str, Any]] = None,
        title: Optional[str] = None,
        filename: Optional[str] = None,
        **imshow_kwargs,
    ):
        import matplotlib.pyplot as plt
        from ..plotting import hsl2rgb

        frame = self._extract_frame(z=z, t=t, zero=zero)

        if frame.ndim == 2:
            hue = np.asarray(frame, dtype=np.float32)
            if lightness_field is None:
                lightness = np.ones_like(hue, dtype=np.float32)
            elif isinstance(lightness_field, (int, np.integer, str)):
                lightness = self._component_image(frame, lightness_field, default="norm")
            else:
                lightness = np.asarray(lightness_field, dtype=np.float32)
                lightness = np.squeeze(lightness)
                if lightness.shape != hue.shape:
                    lightness = np.broadcast_to(lightness, hue.shape)
        elif frame.ndim == 3 and frame.shape[-1] >= 2:
            vec = np.asarray(frame, dtype=np.float32)
            if vec.shape[-1] < 3:
                padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
                padded[..., : vec.shape[-1]] = vec
                vec = padded
            u = vec[:, :, 0]
            v = vec[:, :, 1]
            hue = np.mod(np.arctan2(v, u) + 2.0 * np.pi, 2.0 * np.pi)

            if lightness_field is None:
                lightness = np.asarray(vec[:, :, 2], dtype=np.float32)
            elif isinstance(lightness_field, (int, np.integer, str)):
                lightness = self._component_image(vec, lightness_field, default="norm")
            else:
                lightness = np.asarray(lightness_field, dtype=np.float32)
                lightness = np.squeeze(lightness)
                if lightness.shape != hue.shape:
                    lightness = np.broadcast_to(lightness, hue.shape)
        else:
            raise ValueError(
                "lightness plot expects 2d scalar or 2d vector frame, "
                f"got shape {frame.shape}"
            )

        hue = np.tile(hue, (max(int(repeat), 1), max(int(repeat), 1))).astype(
            np.float32, copy=False
        )
        lightness = np.tile(
            np.asarray(lightness, dtype=np.float32),
            (max(int(repeat), 1), max(int(repeat), 1)),
        )

        if filter_field is not None:
            mask = self._coerce_mask(filter_field, hue.shape)
        else:
            mask = np.ones_like(hue, dtype=bool)

        if clim is not None:
            lo, hi = float(clim[0]), float(clim[1])
        else:
            lo = float(np.nanmin(lightness)) if np.isfinite(lightness).any() else 0.0
            hi = float(np.nanmax(lightness)) if np.isfinite(lightness).any() else 1.0
        if hi <= lo:
            hi = lo + 1e-12

        hue_norm = np.mod(hue, 2.0 * np.pi) / (2.0 * np.pi)
        lightness_norm = np.clip((lightness - lo) / (hi - lo), 0.0, 1.0)

        hsl = np.ones(hue.shape + (3,), dtype=np.float32)
        hsl[:, :, 0] = hue_norm
        hsl[:, :, 1] = 1.0
        hsl[:, :, 2] = lightness_norm
        rgb = hsl2rgb(hsl)

        rgba = np.zeros(hue.shape + (4,), dtype=np.float32)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 1.0
        rgba[~mask] = 0.0

        if ax is None:
            shape_ratio = hue.shape[1] / max(hue.shape[0], 1)
            fig_size = figsize if figsize is not None else (4 * shape_ratio, 4)
            _, ax = plt.subplots(1, 1, figsize=fig_size, dpi=110)

        _, _, extent, _, unit_label = self._resolve_plot_geometry(
            hue.shape,
            multiplier=multiplier,
        )
        ax.imshow(
            rgba,
            origin="lower",
            interpolation=imshow_kwargs.pop("interpolation", "none"),
            aspect=imshow_kwargs.pop("aspect", "equal"),
            extent=extent,
            **imshow_kwargs,
        )

        if colorwheel:
            kw = {} if colorwheel_args is None else dict(colorwheel_args)
            cw_ax = self._mpl_add_colorwheel(ax, **kw)
            if colorwheel_xlabel is not None:
                cw_ax.arrow(100, 100, 60, 0, width=5, fc="w", ec="w")
                cw_ax.annotate(str(colorwheel_xlabel), (115, 140), c="w")
            if colorwheel_ylabel is not None:
                cw_ax.arrow(100, 100, 0, -60, width=5, fc="w", ec="w")
                cw_ax.annotate(str(colorwheel_ylabel), (40, 80), c="w")

        if title is None:
            title = f"{self._dataset.dataset_name} [lightness]"
        ax.set(title=title)
        ax.set_aspect("equal")
        self._set_axis_labels(ax, unit_label)

        if filename is not None:
            plt.savefig(str(filename), bbox_inches="tight", pad_inches=0.02)
        return ax

    def _k3d_scalar_impl(
        self,
        *,
        plot=None,
        t: int = -1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        filter_field: Any = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        interactive_field: Any = None,
        interactive: bool = False,
        **kwargs,
    ):
        try:
            import k3d
        except Exception as exc:
            raise ImportError(
                "k3d is required for plot.k3d.scalar(). Install with: pip install k3d"
            ) from exc

        volume = self._extract_volume(t=t, zero=zero)
        scalar = self._component_volume(volume, component, default="norm")
        if scalar.ndim != 3:
            raise ValueError(
                "k3d.scalar expects a 3d scalar volume. "
                f"Got shape {scalar.shape}."
            )

        visible = self._coerce_mask(filter_field, scalar.shape)
        voxels = self._normalise_to_uint8(
            scalar,
            vmin=vmin,
            vmax=vmax,
            visible_mask=visible,
        )

        plot_obj = plot if plot is not None else k3d.plot(name=f"{self._dataset.dataset_name} scalar")
        if interactive or interactive_field is not None:
            self._k3d_clear_plot(plot_obj)

        cmap_int = self._k3d_colormap_int(cmap)
        try:
            plot_obj += k3d.voxels(voxels, color_map=cmap_int, outlines=False, **kwargs)
        except Exception:
            plot_obj += k3d.voxels(voxels, color_map=cmap_int, **kwargs)

        return plot_obj

    def _k3d_nonzero_impl(
        self,
        *,
        plot=None,
        t: int = -1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        threshold: float = 0.0,
        color: int = 0x4C72B0,
        interactive_field: Any = None,
        interactive: bool = False,
        **kwargs,
    ):
        try:
            import k3d
        except Exception as exc:
            raise ImportError(
                "k3d is required for plot.k3d.nonzero(). Install with: pip install k3d"
            ) from exc

        volume = self._extract_volume(t=t, zero=zero)
        scalar = self._component_volume(volume, component, default="norm")
        if scalar.ndim != 3:
            raise ValueError(
                "k3d.nonzero expects a 3d scalar volume. "
                f"Got shape {scalar.shape}."
            )

        voxels = np.where(np.abs(scalar) > float(threshold), 1, 0).astype(np.uint8)

        plot_obj = plot if plot is not None else k3d.plot(
            name=f"{self._dataset.dataset_name} nonzero"
        )
        if interactive or interactive_field is not None:
            self._k3d_clear_plot(plot_obj)

        try:
            plot_obj += k3d.voxels(
                voxels,
                color_map=int(color),
                outlines=False,
                **kwargs,
            )
        except Exception:
            plot_obj += k3d.voxels(voxels, color_map=int(color), **kwargs)
        return plot_obj

    def _k3d_vector_impl(
        self,
        *,
        plot=None,
        t: int = -1,
        zero: Optional[int] = None,
        color_field: Any = None,
        cmap: str = "viridis",
        head_size: float = 1.0,
        points: bool = True,
        point_size: Optional[float] = None,
        vector_multiplier: Optional[float] = None,
        vector_scale: float = 1.0,
        quiver_density: int = 8,
        min_magnitude: float = 0.0,
        color: int = 0xDD8452,
        interactive_field: Any = None,
        interactive: bool = False,
        **kwargs,
    ):
        try:
            import k3d
        except Exception as exc:
            raise ImportError(
                "k3d is required for plot.k3d.vector(). Install with: pip install k3d"
            ) from exc

        volume = self._extract_volume(t=t, zero=zero)
        if volume.ndim != 4 or volume.shape[-1] < 2:
            raise ValueError(
                "k3d.vector expects a vector volume with shape (z, y, x, c>=2), "
                f"got {volume.shape}"
            )

        vec = np.asarray(volume, dtype=np.float32)
        if vec.shape[-1] < 3:
            padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
            padded[..., : vec.shape[-1]] = vec
            vec = padded

        nz, ny, nx, _ = vec.shape
        dens = max(int(quiver_density), 1)
        stepx = max(int(nx / dens), 1)
        stepy = max(int(ny / dens), 1)
        stepz = max(int(nz / dens), 1)

        u = np.asarray(vec[::stepz, ::stepy, ::stepx, 0], dtype=np.float32)
        v = np.asarray(vec[::stepz, ::stepy, ::stepx, 1], dtype=np.float32)
        w = np.asarray(vec[::stepz, ::stepy, ::stepx, 2], dtype=np.float32)

        magnitude = np.sqrt(u**2 + v**2 + w**2)
        visible = np.isfinite(magnitude) & (magnitude >= float(min_magnitude))

        z_idx = np.arange(0, nz, stepz, dtype=np.float32)
        y_idx = np.arange(0, ny, stepy, dtype=np.float32)
        x_idx = np.arange(0, nx, stepx, dtype=np.float32)
        zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing="ij")

        dx_nm, dy_nm, dz_nm = self._resolve_dxyz_nm()
        origins = np.stack(
            [xx * dx_nm, yy * dy_nm, zz * dz_nm],
            axis=-1,
        ).reshape(-1, 3)
        vectors = np.stack([u, v, w], axis=-1).reshape(-1, 3)
        visible_flat = visible.reshape(-1)

        if vector_multiplier is None:
            cell_min = min(abs(dx_nm), abs(dy_nm), abs(dz_nm), 1.0)
            vmax = float(np.nanmax(np.linalg.norm(vectors, axis=1)))
            vector_multiplier = vmax / max(cell_min, 1e-12)
            if not np.isfinite(vector_multiplier) or vector_multiplier <= 0:
                vector_multiplier = 1.0

        vectors = vectors / float(vector_multiplier)
        vectors = vectors * float(vector_scale)
        origins = np.asarray(origins[visible_flat], dtype=np.float32)
        vectors = np.asarray(vectors[visible_flat], dtype=np.float32)

        if origins.size == 0:
            raise ValueError("No vectors left after filtering/downsampling.")

        plot_obj = plot if plot is not None else k3d.plot(
            name=f"{self._dataset.dataset_name} vector"
        )
        if interactive or interactive_field is not None:
            self._k3d_clear_plot(plot_obj)

        vector_kwargs = dict(kwargs)
        colors = None
        if color_field is not None:
            if isinstance(color_field, (int, np.integer, str)):
                color_volume = self._component_volume(volume, color_field, default="norm")
            else:
                color_volume = np.asarray(color_field, dtype=np.float32)
                color_volume = np.squeeze(color_volume)
                if color_volume.shape != vec.shape[:-1]:
                    color_volume = np.broadcast_to(color_volume, vec.shape[:-1])

            c_sampled = np.asarray(
                color_volume[::stepz, ::stepy, ::stepx],
                dtype=np.float32,
            ).reshape(-1)
            c_sampled = c_sampled[visible_flat]
            c_uint8 = self._normalise_to_uint8(c_sampled, vmin=None, vmax=None)
            cmap_int = self._k3d_colormap_int(cmap)
            colors = []
            for value in c_uint8:
                idx = int(np.clip(value, 0, len(cmap_int) - 1))
                colors.append(2 * (cmap_int[idx],))
            vector_kwargs["colors"] = colors
        else:
            vector_kwargs["color"] = int(color)

        try:
            plot_obj += k3d.vectors(
                origins - 0.5 * vectors,
                vectors,
                head_size=float(head_size),
                **vector_kwargs,
            )
        except Exception:
            plot_obj += k3d.vectors(
                origins=origins - 0.5 * vectors,
                vectors=vectors,
                head_size=float(head_size),
                **vector_kwargs,
            )

        if points:
            if point_size is None:
                point_size = min(abs(dx_nm), abs(dy_nm), abs(dz_nm)) / 4.0
            try:
                plot_obj += k3d.points(
                    origins,
                    color=0x4C72B0,
                    point_size=float(point_size),
                )
            except Exception:
                pass

        return plot_obj

    def _render_frame(
        self,
        frame: np.ndarray,
        *,
        ax,
        mode: str,
        multiplier: Optional[float] = None,
        repeat: int = 1,
        cmap: Optional[str] = None,
        component: Optional[Union[int, str]] = None,
        quiver_density: int = 20,
        colorbar: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt
        from ..plotting import hsl2rgb

        draw_mode = self._normalize_mode(mode)
        ax.clear()
        repeat_value = max(int(repeat), 1)
        if frame.ndim >= 2:
            shape_xy = (int(frame.shape[0]) * repeat_value, int(frame.shape[1]) * repeat_value)
        else:
            shape_xy = (repeat_value, repeat_value)
        dx_u, dy_u, extent, _, unit_label = self._resolve_plot_geometry(
            shape_xy,
            multiplier=multiplier,
        )

        if draw_mode == "snapshot":
            is_vector = frame.ndim == 3 and frame.shape[-1] >= 2 and component is None
            if is_vector:
                vec = np.asarray(frame, dtype=np.float32)
                if vec.shape[-1] < 3:
                    padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
                    padded[..., : vec.shape[-1]] = vec
                    vec = padded

                vector = np.tile(vec, (repeat_value, repeat_value, 1))
                u = vector[:, :, 0]
                v = vector[:, :, 1]
                w = vector[:, :, 2]

                alphas = np.clip(-np.abs(w) + 1, 0.0, 1.0)
                hsl = np.ones((u.shape[0], u.shape[1], 3), dtype=np.float32)
                hsl[:, :, 0] = np.angle(u + 1j * v) / np.pi / 2
                hsl[:, :, 1] = np.clip(np.sqrt(u**2 + v**2 + w**2), 0.0, 1.0)
                hsl[:, :, 2] = (w + 1) / 2
                rgb = hsl2rgb(hsl)

                dens = max(int(quiver_density), 1)
                stepx = max(int(u.shape[1] / dens), 1)
                stepy = max(int(u.shape[0] / dens), 1)
                scale = 1 / max(stepx, stepy)
                x, y = np.meshgrid(
                    np.arange(0, u.shape[1], stepx) * dx_u,
                    np.arange(0, u.shape[0], stepy) * dy_u,
                )

                ax.quiver(
                    x,
                    y,
                    u[::stepy, ::stepx],
                    v[::stepy, ::stepx],
                    alpha=alphas[::stepy, ::stepx],
                    angles="xy",
                    scale_units="xy",
                    scale=scale,
                )
                ax.imshow(
                    rgb,
                    interpolation="none",
                    origin="lower",
                    aspect="equal",
                    extent=extent,
                )
            else:
                image = self._component_image(frame, component, default="norm")
                image = np.tile(image, (repeat_value, repeat_value))
                im = ax.imshow(
                    image,
                    interpolation="none",
                    origin="lower",
                    aspect="equal",
                    cmap=cmap or "viridis",
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                )
                if colorbar:
                    self._mpl_add_colorbar(ax, im)
        else:
            image = self._component_image(frame, component, default="norm")
            image = np.tile(image, (repeat_value, repeat_value))
            im = ax.imshow(
                image,
                interpolation="none",
                origin="lower",
                aspect="equal",
                cmap=cmap or "viridis",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            if colorbar:
                self._mpl_add_colorbar(ax, im)

        job_name = getattr(self._dataset.job_result, "name", "job")
        dset = self._dataset.dataset_name
        if title is None:
            if draw_mode == "heatmap":
                comp_label = "norm" if component is None else str(component)
                title = f"{job_name} — {dset} [{comp_label}]"
            else:
                title = f"{job_name} — {dset}"
        ax.set(title=title)
        self._set_axis_labels(ax, unit_label)
        return ax

    def _snapshot_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        multiplier: Optional[float] = None,
        repeat: int = 1,
        zero: Optional[int] = None,
        cmap: Optional[str] = None,
        component: Optional[Union[int, str]] = None,
        quiver_density: int = 20,
        colorbar: bool = True,
    ):
        import matplotlib.pyplot as plt

        frame = self._extract_frame(z=z, t=t, zero=zero)
        if ax is None:
            base = self._component_image(frame, component, default="norm")
            shape_ratio = base.shape[1] / max(base.shape[0], 1)
            _, ax = plt.subplots(1, 1, figsize=(4 * shape_ratio, 4), dpi=100)
        ax = self._render_frame(
            frame,
            ax=ax,
            mode="snapshot",
            multiplier=multiplier,
            repeat=repeat,
            cmap=cmap,
            component=component,
            quiver_density=quiver_density,
            colorbar=colorbar,
        )
        return ax

    def _heatmap_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        multiplier: Optional[float] = None,
        repeat: int = 1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        cmap: str = "viridis",
        colorbar: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        import matplotlib.pyplot as plt

        frame = self._extract_frame(z=z, t=t, zero=zero)
        if ax is None:
            image = self._component_image(frame, component, default="norm")
            shape_ratio = image.shape[1] / max(image.shape[0], 1)
            _, ax = plt.subplots(1, 1, figsize=(4 * shape_ratio, 4), dpi=100)
        ax = self._render_frame(
            frame,
            ax=ax,
            mode="heatmap",
            multiplier=multiplier,
            repeat=repeat,
            cmap=cmap,
            component=component,
            colorbar=colorbar,
            vmin=vmin,
            vmax=vmax,
        )
        return ax

    def _k3d_heatmap_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        repeat: int = 1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_vectors: bool = False,
        quiver_density: int = 20,
        vector_scale: float = 1.0,
        vector_color: int = 0x00D1FF,
        vector_head_size: float = 0.8,
        vector_line_width: float = 0.015,
        height_scale: float = 0.0,
    ):
        try:
            import k3d
        except Exception as exc:
            raise ImportError(
                "k3d is required for plot.k3d.heatmap(). Install with: pip install k3d"
            ) from exc

        frame = self._extract_frame(z=z, t=t, zero=zero)
        image = self._component_image(frame, component, default="norm")
        image = np.tile(image, (max(int(repeat), 1), max(int(repeat), 1))).astype(
            np.float32,
            copy=False,
        )
        colormap = self._k3d_colormap(cmap)
        color_range = self._k3d_color_range(image, vmin=vmin, vmax=vmax)

        if float(height_scale) != 0.0:
            surface = (image * float(height_scale)).astype(np.float32, copy=False)
        else:
            surface = np.zeros_like(image, dtype=np.float32)

        plot = k3d.plot(name=f"{self._dataset.dataset_name} heatmap")
        surface_kwargs = {"attribute": image, "color_range": color_range}
        if colormap is not None:
            surface_kwargs["color_map"] = colormap

        try:
            plot += k3d.surface(surface, **surface_kwargs)
        except Exception:
            # Fallback for older/newer k3d variants with limited kwargs.
            fallback_kwargs = {}
            if colormap is not None:
                fallback_kwargs["color_map"] = colormap
            try:
                plot += k3d.surface(surface, **fallback_kwargs)
            except Exception:
                plot += k3d.surface(image.astype(np.float32, copy=False))

        if show_vectors and frame.ndim == 3 and frame.shape[-1] >= 2:
            vec = np.asarray(frame, dtype=np.float32)
            if vec.shape[-1] < 3:
                padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
                padded[..., : vec.shape[-1]] = vec
                vec = padded
            vec = np.tile(vec, (max(int(repeat), 1), max(int(repeat), 1), 1))
            u = vec[:, :, 0]
            v = vec[:, :, 1]
            stepx = max(int(u.shape[1] / max(int(quiver_density), 1)), 1)
            stepy = max(int(u.shape[0] / max(int(quiver_density), 1)), 1)
            grid_x, grid_y = np.meshgrid(
                np.arange(0, u.shape[1], stepx, dtype=np.float32),
                np.arange(0, u.shape[0], stepy, dtype=np.float32),
            )
            origins = np.stack(
                [grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size, dtype=np.float32)],
                axis=1,
            ).astype(np.float32)
            vectors = np.stack(
                [
                    u[::stepy, ::stepx].ravel(),
                    v[::stepy, ::stepx].ravel(),
                    np.zeros(grid_x.size, dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32)
            vectors *= float(vector_scale)
            try:
                plot += k3d.vectors(
                    origins,
                    vectors,
                    color=int(vector_color),
                    head_size=float(vector_head_size),
                    line_width=float(vector_line_width),
                )
            except Exception:
                try:
                    plot += k3d.vectors(
                        origins=origins,
                        vectors=vectors,
                        color=int(vector_color),
                        head_size=float(vector_head_size),
                        line_width=float(vector_line_width),
                    )
                except Exception:
                    pass

        return plot

    def interactive(
        self,
        *,
        mode: str = "snapshot",
        component: Optional[Union[int, str]] = None,
        z: int = 0,
        multiplier: Optional[float] = None,
        repeat: int = 1,
        zero: Optional[int] = None,
        cmap: str = "viridis",
        quiver_density: int = 20,
        fps: int = 20,
        toolbar: bool = True,
    ):
        import matplotlib.pyplot as plt
        from matplotlib import animation as mpl_animation
        from matplotlib.widgets import Button, Slider

        sequence = self._extract_sequence(z=z, zero=zero)
        n_frames = int(sequence.shape[0])
        first = np.asarray(sequence[0], dtype=np.float32)
        base = self._component_image(first, component, default="norm")
        shape_ratio = base.shape[1] / max(base.shape[0], 1)

        fig, ax = plt.subplots(1, 1, figsize=(4 * shape_ratio, 4), dpi=110)
        if toolbar:
            plt.subplots_adjust(bottom=0.18)
            slider_ax = fig.add_axes([0.15, 0.07, 0.55, 0.04])
            button_ax = fig.add_axes([0.74, 0.065, 0.12, 0.05])
            frame_slider = Slider(
                slider_ax,
                "Frame",
                valmin=0,
                valmax=max(n_frames - 1, 0),
                valinit=0,
                valstep=1,
            )
            play_btn = Button(button_ax, "Play", color="#e5e7eb", hovercolor="#d1d5db")
        else:
            frame_slider = None
            play_btn = None

        state = {"index": 0, "playing": False}
        render_mode = self._normalize_mode(mode)

        def _draw(index: int) -> None:
            idx = int(np.clip(int(index), 0, max(n_frames - 1, 0)))
            state["index"] = idx
            frame = np.asarray(sequence[idx], dtype=np.float32)
            title = f"{self._dataset.dataset_name} [{idx + 1}/{n_frames}]"
            self._render_frame(
                frame,
                ax=ax,
                mode=render_mode,
                multiplier=multiplier,
                repeat=repeat,
                cmap=cmap,
                component=component,
                quiver_density=quiver_density,
                colorbar=False,
                title=title,
            )
            if frame_slider is not None and int(round(frame_slider.val)) != idx:
                frame_slider.eventson = False
                frame_slider.set_val(idx)
                frame_slider.eventson = True
            fig.canvas.draw_idle()

        def _on_slider(val):
            _draw(int(round(float(val))))

        def _on_toggle(_event):
            state["playing"] = not state["playing"]
            if play_btn is not None:
                play_btn.label.set_text("Pause" if state["playing"] else "Play")
                fig.canvas.draw_idle()

        def _tick(_frame):
            if not state["playing"]:
                return ()
            _draw((state["index"] + 1) % max(n_frames, 1))
            return ()

        if frame_slider is not None:
            frame_slider.on_changed(_on_slider)
        if play_btn is not None:
            play_btn.on_clicked(_on_toggle)

        anim = mpl_animation.FuncAnimation(
            fig,
            _tick,
            interval=1000.0 / max(int(fps), 1),
            blit=False,
            cache_frame_data=False,
        )
        fig._mmpp_interactive = {
            "slider": frame_slider,
            "play_button": play_btn,
            "animation": anim,
            "state": state,
        }
        _draw(0)
        return fig

    def animate(
        self,
        *,
        mode: str = "snapshot",
        component: Optional[Union[int, str]] = None,
        z: int = 0,
        multiplier: Optional[float] = None,
        repeat: int = 1,
        zero: Optional[int] = None,
        cmap: str = "viridis",
        quiver_density: int = 20,
        fps: int = 20,
        save_path: Optional[str] = None,
        dpi: int = 120,
    ):
        import matplotlib.pyplot as plt
        from matplotlib import animation as mpl_animation

        sequence = self._extract_sequence(z=z, zero=zero)
        n_frames = int(sequence.shape[0])
        first = np.asarray(sequence[0], dtype=np.float32)
        base = self._component_image(first, component, default="norm")
        shape_ratio = base.shape[1] / max(base.shape[0], 1)
        fig, ax = plt.subplots(1, 1, figsize=(4 * shape_ratio, 4), dpi=dpi)
        render_mode = self._normalize_mode(mode)

        def _update(frame_idx: int):
            idx = int(frame_idx) % max(n_frames, 1)
            frame = np.asarray(sequence[idx], dtype=np.float32)
            title = f"{self._dataset.dataset_name} [{idx + 1}/{n_frames}]"
            self._render_frame(
                frame,
                ax=ax,
                mode=render_mode,
                multiplier=multiplier,
                repeat=repeat,
                cmap=cmap,
                component=component,
                quiver_density=quiver_density,
                colorbar=False,
                title=title,
            )
            return []

        anim = mpl_animation.FuncAnimation(
            fig,
            _update,
            frames=max(n_frames, 1),
            interval=1000.0 / max(int(fps), 1),
            repeat=True,
            blit=False,
        )

        if save_path is None:
            return anim

        path = str(save_path)
        suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        if suffix == "mp4":
            writer = mpl_animation.FFMpegWriter(fps=max(int(fps), 1), bitrate=2000)
        elif suffix == "gif":
            writer = mpl_animation.PillowWriter(fps=max(int(fps), 1))
        else:
            raise ValueError("save_path extension must be .mp4 or .gif")

        anim.save(path, writer=writer, dpi=dpi)
        return path

    @property
    def mpl(self):
        if self._mpl is None:
            self._mpl = _DatasetMatplotlibPlotAccessor(self)
        return self._mpl

    @property
    def k3d(self):
        if self._k3d is None:
            self._k3d = _DatasetK3DPlotAccessor(self)
        return self._k3d

    def snapshot(self, **kwargs):
        """Convenience alias for ``plot.mpl.snapshot(...)``."""
        return self.mpl.snapshot(**kwargs)

    def scalar(self, **kwargs):
        """Convenience alias for ``plot.mpl.scalar(...)``."""
        return self.mpl.scalar(**kwargs)

    def vector(self, **kwargs):
        """Convenience alias for ``plot.mpl.vector(...)``."""
        return self.mpl.vector(**kwargs)

    def contour(self, **kwargs):
        """Convenience alias for ``plot.mpl.contour(...)``."""
        return self.mpl.contour(**kwargs)

    def lightness(self, **kwargs):
        """Convenience alias for ``plot.mpl.lightness(...)``."""
        return self.mpl.lightness(**kwargs)

    def heatmap(self, **kwargs):
        """Convenience alias for ``plot.mpl.heatmap(...)``."""
        return self.mpl.heatmap(**kwargs)

    def heamtp(self, **kwargs):
        """Compatibility alias for ``heatmap``."""
        return self.mpl.heatmap(**kwargs)

    def __repr__(self):
        dset = self._dataset.dataset_name
        return (
            f"<DatasetPlotAccessor('{dset}'): .snapshot(), .scalar(), .vector(), "
            ".contour(), .lightness(), .heatmap(), .interactive(), .animate(), .mpl, .k3d>"
        )


class DatasetAwareWrapper:
    """Wrapper that acts like zarr.Array but has .fft property"""

    def __init__(
        self,
        job_result,
        dataset_name,
        zarr_array,
        slice_info=None,
        materialized_data: Optional[np.ndarray] = None,
    ):
        self.job_result = job_result
        self.dataset_name = dataset_name
        self.zarr_array = zarr_array
        self.slice_info = slice_info  # Store slicing information
        self._materialized_data = materialized_data
        self._fft = None
        self._solitons = None
        self._analyze = None
        self._plot = None

    def _resolve_source(self):
        """Return underlying data respecting the stored slice."""
        if self._materialized_data is not None:
            if self.slice_info is not None:
                return self._materialized_data[self.slice_info]
            return self._materialized_data
        if self.slice_info is not None:
            return self.zarr_array[self.slice_info]
        return self.zarr_array

    def __getattr__(self, name):
        """Delegate to zarr_array for most attributes (but not our own properties)"""
        # Don't delegate properties that are defined on this class
        if name in ('dt', 'fft', 'analyze', 'shape', 'data', 'plot'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
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
        source = self._resolve_source()
        source_shape = source.shape
        ndim = len(source_shape)
        
        # Normalize the slice to keep dimensions
        normalized_key = self._normalize_slice_to_keep_dims(key, ndim)
        
        if self._materialized_data is not None:
            sliced = np.asarray(source[normalized_key])
            return DatasetAwareWrapper(
                self.job_result,
                self.dataset_name,
                self.zarr_array,
                slice_info=None,
                materialized_data=sliced,
            )

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
                materialized_data=None,
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
    def shape(self):
        """Shape accounting for slicing"""
        if self._materialized_data is not None:
            return self._resolve_source().shape
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

    @staticmethod
    def _normalize_downsample_spec(spec: tuple[Any, ...], ndim: int) -> tuple[Optional[int], ...]:
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

        normalized: list[Optional[int]] = []
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
        array: np.ndarray,
        axis: int,
        target: int,
        *,
        strict: bool = False,
    ) -> np.ndarray:
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

    def downsample(self, *spec: Any, strict: bool = False) -> "DatasetAwareWrapper":
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

        return DatasetAwareWrapper(
            self.job_result,
            self.dataset_name,
            self.zarr_array,
            slice_info=None,
            materialized_data=np.asarray(reduced, dtype=np.float32),
        )

    def as_zarr(self):
        """Return the underlying zarr.Array when no slicing is active."""
        if self.slice_info is not None or self._materialized_data is not None:
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
