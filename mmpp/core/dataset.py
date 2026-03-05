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
    def _normalize_mode(mode: str) -> str:
        value = str(mode).strip().lower()
        if value in {"snapshot", "vector", "quiver"}:
            return "snapshot"
        if value in {"heatmap", "scalar", "mpl_heatmap"}:
            return "heatmap"
        raise ValueError(f"Unsupported render mode: {mode!r}. Use 'snapshot' or 'heatmap'.")

    def _render_frame(
        self,
        frame: np.ndarray,
        *,
        ax,
        mode: str,
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
        dx_nm, dy_nm = self._resolve_dx_dy_nm()
        repeat_value = max(int(repeat), 1)

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
                    np.arange(0, u.shape[1], stepx) * dx_nm,
                    np.arange(0, u.shape[0], stepy) * dy_nm,
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
                    extent=(0, rgb.shape[1] * dx_nm, 0, rgb.shape[0] * dy_nm),
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
                    extent=(0, image.shape[1] * dx_nm, 0, image.shape[0] * dy_nm),
                )
                if colorbar:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
                extent=(0, image.shape[1] * dx_nm, 0, image.shape[0] * dy_nm),
            )
            if colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        job_name = getattr(self._dataset.job_result, "name", "job")
        dset = self._dataset.dataset_name
        if title is None:
            if draw_mode == "heatmap":
                comp_label = "norm" if component is None else str(component)
                title = f"{job_name} — {dset} [{comp_label}]"
            else:
                title = f"{job_name} — {dset}"
        ax.set(title=title, xlabel="x (nm)", ylabel="y (nm)")
        return ax

    def _snapshot_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
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
        show_vectors: bool = False,
        quiver_density: int = 20,
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

        if float(height_scale) != 0.0:
            surface = (image * float(height_scale)).astype(np.float32, copy=False)
        else:
            surface = np.zeros_like(image, dtype=np.float32)

        plot = k3d.plot(name=f"{self._dataset.dataset_name} heatmap")
        try:
            plot += k3d.surface(surface, attribute=image)
        except Exception:
            # Fallback for k3d versions without `attribute`.
            plot += k3d.surface(image.astype(np.float32, copy=False))

        if show_vectors and frame.ndim == 3 and frame.shape[-1] >= 2:
            vec = np.asarray(frame, dtype=np.float32)
            if vec.shape[-1] < 3:
                padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
                padded[..., : vec.shape[-1]] = vec
                vec = padded
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
            try:
                plot += k3d.vectors(origins, vectors)
            except Exception:
                pass

        return plot

    def interactive(
        self,
        *,
        mode: str = "snapshot",
        component: Optional[Union[int, str]] = None,
        z: int = 0,
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

    def heatmap(self, **kwargs):
        """Convenience alias for ``plot.mpl.heatmap(...)``."""
        return self.mpl.heatmap(**kwargs)

    def heamtp(self, **kwargs):
        """Compatibility alias for ``heatmap``."""
        return self.mpl.heatmap(**kwargs)

    def __repr__(self):
        dset = self._dataset.dataset_name
        return (
            f"<DatasetPlotAccessor('{dset}'): .snapshot(), .heatmap(), "
            ".interactive(), .animate(), .mpl, .k3d>"
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
