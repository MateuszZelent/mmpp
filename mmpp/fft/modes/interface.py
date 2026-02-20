"""
FFT Mode Interface Module

Provides FFTModeInterface for elegant job[0].fft.modes syntax.
Integrates with DatasetAwareWrapper for slice propagation.
"""

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Optional, Union
import logging
import numpy as np

from ..method_helpers import CallableMethodHelper

log = logging.getLogger("mmpp.fft.modes")

# Cache schema version - bump when cached results are no longer compatible
SPECTRUM_CACHE_SCHEMA_VERSION = 1

# Lazy imports to avoid circular dependencies
def _get_data_loader():
    from .data_loader import ModeDataLoader, ModeDataContext
    return ModeDataLoader, ModeDataContext

def _get_interactive():
    from .interactive import InteractiveSpectrum
    return InteractiveSpectrum


def _get_interactive_plot():
    from .interactive import plot as plot_spectrum
    return plot_spectrum


@dataclass
class ModeResult:
    """Fluent wrapper around FMR mode data with plotting helpers."""

    _modes: "FFTModeInterfaceNew"
    mode_data: Any
    requested_frequency: float
    z_layer: int

    @property
    def frequency(self) -> float:
        """Actual frequency (GHz) used for loaded mode."""
        return float(getattr(self.mode_data, "frequency", self.requested_frequency))

    @property
    def data(self) -> np.ndarray:
        """Raw complex mode array."""
        return np.asarray(getattr(self.mode_data, "mode_array"))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Spatial extent passed to imshow."""
        ext = getattr(self.mode_data, "extent", None)
        if ext is None:
            arr = self.data
            return (0.0, float(arr.shape[1]), 0.0, float(arr.shape[0]))
        return tuple(ext)

    @property
    def plot(self) -> "ModePlotAccessor":
        """Plot helper namespace."""
        return ModePlotAccessor(self)

    @property
    def plt(self) -> "ModePlotAccessor":
        """Deprecated alias for :attr:`plot`."""
        return self.plot

    def get_component(
        self,
        component: Union[str, int] = "z",
        value: str = "complex",
    ) -> np.ndarray:
        """Return selected mode component transformed to requested representation."""
        comp_data = self.mode_data.get_component(component)
        mode = str(value).lower()
        if mode in {"complex", "raw"}:
            return np.asarray(comp_data)
        if mode in {"magnitude", "abs"}:
            return np.abs(comp_data)
        if mode == "phase":
            return np.angle(comp_data)
        if mode == "real":
            return np.real(comp_data)
        if mode in {"imag", "imaginary"}:
            return np.imag(comp_data)
        if mode == "combined":
            magnitude = np.abs(comp_data)
            return magnitude * np.cos(np.angle(comp_data))
        raise ValueError(
            "Unknown value mode. Use: complex, magnitude, phase, real, imag, combined."
        )

    def __getattr__(self, item: str) -> Any:
        """Delegate unknown attributes to underlying FMRModeData."""
        return getattr(self.mode_data, item)

    def __array__(self):
        return np.asarray(self.data)

    def __repr__(self) -> str:
        shape = getattr(self.data, "shape", None)
        return (
            f"ModeResult(f={self.frequency:.3f} GHz, z={self.z_layer}, "
            f"shape={shape})"
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        freq = self.frequency
        z = self.z_layer
        try:
            shape = self.data.shape
            shape_str = " × ".join(str(s) for s in shape)
        except Exception:
            shape_str = "N/A"

        props = [
            (".frequency", f"{freq:.3f} GHz"),
            (".z_layer", str(z)),
            (".data", f"np.ndarray ({shape_str})"),
            (".extent", "Spatial extent (x_min, x_max, y_min, y_max)"),
        ]
        prop_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(v)}</td></tr>"
            for n, v in props
        )
        methods = [
            (".get_component(component, value)", "Extract component with transform"),
            (".plot.imshow(...)", "Render mode as 2D image"),
            (".plot.interactive(...)", "Interactive explorer at this frequency"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        imshow_params = [
            ("component", "'z'", "'x', 'y', 'z' or 0, 1, 2"),
            ("value", "'magnitude'", "'magnitude', 'phase', 'real', 'imag', 'combined'"),
            ("cmap", "auto", "Colormap (auto: viridis/twilight/RdBu_r)"),
            ("colorbar", "False", "Show colorbar"),
            ("ax", "None", "Existing matplotlib Axes"),
        ]
        param_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#a5b4fc;'>{_esc(d)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
            for n, d, desc in imshow_params
        )
        example = (
            f"# Plot magnitude of mz\n"
            f"mode.plot.imshow(component='z')\n"
            f"\n"
            f"# Plot phase with colorbar\n"
            f"mode.plot.imshow(component='z', value='phase', colorbar=True)\n"
            f"\n"
            f"# Plot combined (magnitude × cos(phase))\n"
            f"mode.plot.imshow(component='z', value='combined')\n"
            f"\n"
            f"# Get raw data as numpy array\n"
            f"arr = mode.get_component('z', value='magnitude')"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            f"ModeResult</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            f"FMR eigenmode at {freq:.3f} GHz · z-layer: {z} · shape: {shape_str}</div>"
            # Properties
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Properties</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{prop_rows}</table></div>"
            # Methods
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
            # imshow params
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>"
            "Parameters <span style='color:#94a3b8;font-weight:400;'>(.plot.imshow)</span></div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Arg</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{param_rows}</tbody></table></div>"
            # Examples
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )


class ModePlotAccessor:
    """Plot helper API for ModeResult."""

    def __init__(self, mode_result: ModeResult):
        self._mode = mode_result

    def imshow(
        self,
        component: Union[str, int] = "z",
        value: str = "magnitude",
        ax: Any = None,
        cmap: Optional[str] = None,
        origin: str = "lower",
        interpolation: str = "nearest",
        aspect: str = "equal",
        colorbar: bool = False,
        **kwargs,
    ):
        """Render selected mode representation as imshow and return AxesImage."""
        import matplotlib.pyplot as plt

        data = self._mode.get_component(component=component, value=value)

        if cmap is None:
            value_mode = str(value).lower()
            if value_mode == "phase":
                cmap = "twilight"
            elif value_mode == "combined":
                cmap = "RdBu_r"
            else:
                cmap = "viridis"

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        else:
            fig = ax.figure

        image = ax.imshow(
            data,
            origin=origin,
            extent=self._mode.extent,
            cmap=cmap,
            interpolation=interpolation,
            aspect=aspect,
            **kwargs,
        )
        ax.set_title(
            f"m_{component} ({value}) @ {self._mode.frequency:.3f} GHz, z={self._mode.z_layer}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if colorbar:
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        return image

    def interactive(
        self,
        toolbar: bool = True,
        show: bool = True,
        **kwargs,
    ):
        """Open interactive spectrum pre-positioned at this mode frequency."""
        kwargs.setdefault("initial_frequency", self._mode.frequency)
        kwargs.setdefault("z_layer", self._mode.z_layer)
        return self._mode._modes._interactive_spectrum_impl(
            toolbar=toolbar,
            show=show,
            **kwargs,
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        freq = self._mode.frequency
        z = self._mode.z_layer

        methods = [
            (".imshow(component, value, ...)", "Render mode as 2D matplotlib image"),
            (".interactive(toolbar=True)", "Open interactive explorer at this frequency"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        imshow_params = [
            ("component", "'z'", "'x', 'y', 'z' or 0, 1, 2"),
            ("value", "'magnitude'", "'magnitude', 'phase', 'real', 'imag', 'combined'"),
            ("cmap", "auto", "Colormap (auto-selected by value type)"),
            ("colorbar", "False", "Show colorbar alongside plot"),
            ("ax", "None", "Existing matplotlib Axes to draw on"),
            ("origin", "'lower'", "Image origin ('lower' or 'upper')"),
            ("interpolation", "'nearest'", "Pixel interpolation method"),
            ("aspect", "'equal'", "Axes aspect ratio"),
        ]
        param_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#a5b4fc;'>{_esc(d)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
            for n, d, desc in imshow_params
        )
        example = (
            f"# Plot mz magnitude (default)\n"
            f"mode.plot.imshow()\n"
            f"\n"
            f"# Plot phase with colorbar\n"
            f"mode.plot.imshow(component='z', value='phase', colorbar=True)\n"
            f"\n"
            f"# Plot all components\n"
            f"import matplotlib.pyplot as plt\n"
            f"fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            f"for ax, c in zip(axes, ['x', 'y', 'z']):\n"
            f"    mode.plot.imshow(component=c, ax=ax)\n"
            f"\n"
            f"# Open interactive explorer\n"
            f"mode.plot.interactive()"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Mode Plot Accessor</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            f"Plotting helpers for FMR mode at {freq:.3f} GHz · z-layer: {z}</div>"
            # Methods
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
            # imshow params
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>"
            "Parameters <span style='color:#94a3b8;font-weight:400;'>(.imshow)</span></div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Arg</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{param_rows}</tbody></table></div>"
            # Examples
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )


class InteractiveSpectrumHelper:
    """Callable helper that shows documentation when accessed as property.
    
    When accessed (job.fft.modes.interactive_spectrum), displays helpful usage.
    When called (job.fft.modes.interactive_spectrum(...)), runs the method.
    """
    
    def __init__(self, modes_interface):
        self._modes = modes_interface
        self._method = modes_interface._interactive_spectrum_impl
    
    def __call__(self, **kwargs):
        """Delegate to actual interactive_spectrum method."""
        return self._method(**kwargs)
    
    def __repr__(self):
        return self._rich_display()
    
    def _rich_display(self) -> str:
        """Generate rich help display for interactive_spectrum."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich.syntax import Syntax
            from io import StringIO
            
            capture = StringIO()
            console = Console(file=capture, force_terminal=True, width=100)
            
            # Title
            title = Text()
            title.append("🎯 Interactive Spectrum Visualization\n", style="bold cyan")
            title.append(f"📁 Dataset: {self._modes.dataset_name}\n", style="dim")
            if self._modes.component_label:
                title.append(f"📊 Component: {self._modes.component_label}", style="green")
            
            console.print(Panel(title, border_style="cyan"))
            
            # Features
            features = Text()
            features.append("✨ Interactive Features:\n\n", style="bold yellow")
            feature_list = [
                ("Toolbar", "Display/mode/filter tabs inspired by dispersion UI"),
                ("Click", "Select frequency in spectrum → update mode panels"),
                ("Right-click", "Snap selected frequency to nearest detected peak"),
                ("Sweep", "Play + frame slider for frequency sweep animation"),
                ("Save sweep", "Export GIF/MP4 sweep animation from toolbar"),
                ("Presets", "Save/load/delete toolbar configuration"),
            ]
            for key, desc in feature_list:
                features.append(f"  • ", style="dim")
                features.append(f"{key:15}", style="bold green")
                features.append(f" {desc}\n", style="white")
            
            console.print(features)
            
            # Parameters table
            params = Table(show_header=True, header_style="bold magenta")
            params.add_column("Parameter", style="yellow")
            params.add_column("Type", style="cyan")
            params.add_column("Default", style="green")
            params.add_column("Description", style="white")
            
            param_data = [
                ("components", "list", "auto", "['x','y','z'] or [0,1,2]"),
                ("z_layer", "int", "-1", "Z-layer for modes (top layer)"),
                ("dpi", "int", "100", "Figure resolution"),
                ("figsize", "tuple", "(16,10)", "Figure size (width, height)"),
                ("toolbar", "bool", "True", "Toolbar UI with live filtering"),
                ("show", "bool", "True", "Display figure/widget immediately"),
                ("log_scale", "bool", "False", "Logarithmic Y-scale"),
                ("normalize", "bool", "True", "Normalize power to max"),
                ("baseline_mode", "str", "'none'", "none/mean/median/linear baseline correction"),
                ("clip_percentile_low", "float", "0.0", "Low percentile clipping"),
                ("clip_percentile_high", "float", "100.0", "High percentile clipping"),
                ("soft_threshold_percentile", "float", "0.0", "Soft-threshold denoising"),
                ("show_peaks", "bool", "True", "Detect and mark peaks"),
                ("toolbar=False", "bool", "legacy", "Fallback to legacy keyboard/mouse mode"),
            ]
            for p, t, d, desc in param_data:
                params.add_row(p, t, d, desc)
            
            console.print(params)
            console.print("")
            
            # Examples
            example = '''# Basic usage with component selection:
job[0].m[:200,...,1].fft.modes.interactive_spectrum(dpi=150)

# Full control:
job[0].fft.modes.interactive_spectrum(
    components=['x', 'y'],
    z_layer=-1,
    dpi=200,
    log_scale=True,
    show_peaks=True,
)

# Advanced filtering in toolbar:
job[0].fft.modes.interactive_spectrum(
    baseline_mode="linear",
    clip_percentile_low=2.0,
    clip_percentile_high=99.0,
    soft_threshold_percentile=40.0,
)

# Legacy fallback (double-click animations, key bindings):
job[0].fft.modes.interactive_spectrum(toolbar=False, auto_animate=True)'''
            
            syntax = Syntax(example, "python", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title="[bold green]Examples", border_style="green"))
            
            return capture.getvalue()
        except ImportError:
            return "interactive_spectrum(...) - Interactive spectrum with mode visualization. Call with () to execute."


class FFTModesHelpAccessor:
    """Callable helper namespace for major mode-interface methods."""

    def __init__(self, modes: "FFTModeInterfaceNew", owner: str = "fft.modes"):
        self._modes = modes
        self._owner = owner

    def _method(
        self,
        name: str,
        description: str,
        examples: list[str] | None = None,
    ) -> CallableMethodHelper:
        target = getattr(self._modes, name)
        return CallableMethodHelper(
            owner=self._owner,
            name=name,
            target=target,
            description=description,
            examples=examples or [],
        )

    @property
    def configure(self) -> CallableMethodHelper:
        return self._method(
            "configure",
            "Configure tmax/cache/filter defaults for cloned interface.",
            ["job[0].fft.modes.help.configure(tmax=500)"],
        )

    @property
    def filters(self) -> CallableMethodHelper:
        return self._method(
            "filters",
            "Apply fluent interactive-spectrum filters on a cloned interface.",
            ["job[0].fft.modes.help.filters(freq_min=2.0, normalize=True)"],
        )

    @property
    def clear_filters(self) -> CallableMethodHelper:
        return self._method(
            "clear_filters",
            "Reset fluent filters while preserving dataset/config context.",
            ["job[0].fft.modes.help.clear_filters()"],
        )

    @property
    def mode(self) -> CallableMethodHelper:
        return self._method(
            "mode",
            "Get mode object at frequency (GHz) with .plot helpers.",
            ["job[0].fft.modes.help.mode(f=9.5)"],
        )

    @property
    def plot(self) -> CallableMethodHelper:
        return self._method(
            "plot",
            "Plot filtered FMR spectrum.",
            ["job[0].fft.modes.help.plot(show=False)"],
        )

    @property
    def plot_modes(self) -> CallableMethodHelper:
        return self._method(
            "plot_modes",
            "Legacy static mode-grid plotting.",
            ["job[0].fft.modes.help.plot_modes(frequency=9.5)"],
        )

    @property
    def interactive_spectrum(self):
        """Return existing interactive-spectrum helper."""
        return self._modes.interactive_spectrum

    @property
    def compute_modes(self) -> CallableMethodHelper:
        return self._method(
            "compute_modes",
            "Compute or recompute mode datasets in zarr.",
            ["job[0].fft.modes.help.compute_modes(force=True)"],
        )

    @property
    def characterize_mode(self) -> CallableMethodHelper:
        return self._method(
            "characterize_mode",
            "Characterize and classify mode at frequency.",
            ["job[0].fft.modes.help.characterize_mode(frequency=9.5)"],
        )

    @property
    def save_modes_animation(self) -> CallableMethodHelper:
        return self._method(
            "save_modes_animation",
            "Export temporal/frequency mode animation.",
            ["job[0].fft.modes.help.save_modes_animation(frequency=9.5, save_path='mode.gif')"],
        )

    def __repr__(self) -> str:
        return (
            "<FFTModesHelpAccessor: configure, filters, clear_filters, mode, plot, "
            "plot_modes, interactive_spectrum, compute_modes, characterize_mode, "
            "save_modes_animation>"
        )


class FFTModeInterfaceNew:
    """Enhanced FFT interface with mode visualization capabilities.
    
    Supports slice propagation from DatasetAwareWrapper.
    Provides elegant syntax like: job[0].m[:200,...,1].fft.modes.interactive_spectrum()
    
    Attributes
    ----------
    fft_result_index : int
        Index of FFT result
    parent_fft : FFT
        Parent FFT instance
    _dataset_context : str, optional
        Dataset name from DatasetSpecificFFT
    _slice_context : tuple, optional
        Slice info from DatasetAwareWrapper
    """
    
    def __init__(self, fft_result_index: int, parent_fft: Any):
        """Initialize mode interface.
        
        Parameters
        ----------
        fft_result_index : int
            Index into parent FFT results
        parent_fft : FFT
            Parent FFT instance
        """
        self.fft_result_index = fft_result_index
        self.parent_fft = parent_fft
        
        # Context from DatasetSpecificFFT (set externally)
        self._dataset_context: Optional[str] = None
        self._slice_context: Optional[tuple] = None
        
        # Lazy-loaded instances
        self._data_loader = None
        self._mode_analyzer = None
        self._interactive_filters: dict[str, Any] = {}
        self._auto_compute_checked = False
        
        # Configuration (like dispersion module)
        self._tmax: Optional[int] = None
        self._filters_config: Optional[dict[str, Any]] = None
        self._cache_dir: Optional[str] = None
        self._memory_cache: dict[str, Any] = {}
        self._last_result: Optional[Any] = None
    
    @property
    def zarr_path(self) -> str:
        """Get zarr path from parent FFT."""
        return self.parent_fft.job_result.path
    
    @property
    def dataset_name(self) -> str:
        """Get dataset name (from context or auto-detect)."""
        if self._dataset_context:
            return self._dataset_context
        # Auto-detect
        try:
            from ...plotting import _find_largest_m_dataset
            return _find_largest_m_dataset(self.zarr_path)
        except Exception:
            return "m"  # Fallback
    
    @property
    def component_index(self) -> Optional[int]:
        """Extract component index from slice_context."""
        if self._slice_context and isinstance(self._slice_context, tuple):
            last = self._slice_context[-1]
            if isinstance(last, (int, np.integer)):
                idx = int(last)
                if idx in (0, 1, 2):
                    return idx
                if idx == -1:
                    return 2
            if isinstance(last, slice):
                step = 1 if last.step is None else int(last.step)
                if step == 1 and isinstance(last.start, (int, np.integer)):
                    start = int(last.start)
                    if start in (0, 1, 2) and isinstance(last.stop, (int, np.integer)):
                        if int(last.stop) == start + 1:
                            return start
                    if start == -1 and last.stop is None:
                        return 2
        return None
    
    @property
    def component_label(self) -> Optional[str]:
        """Get label for selected component."""
        labels = [r"$m_x$", r"$m_y$", r"$m_z$"]
        idx = self.component_index
        if idx is not None:
            return labels[idx]
        return None
    
    @property
    def last_result(self) -> Optional[Any]:
        """Get the result from the most recent computation."""
        return self._last_result

    @property
    def helpers(self) -> FFTModesHelpAccessor:
        """Helper namespace for major modes-interface methods."""
        return FFTModesHelpAccessor(self, owner=f"{self.dataset_name}.fft.modes")

    @property
    def help(self) -> FFTModesHelpAccessor:
        """Alias for :attr:`helpers`."""
        return self.helpers
    
    def configure(
        self,
        *,
        tmax: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
    ) -> "FFTModeInterfaceNew":
        """
        Configure interface settings (fluent API).
        
        Returns a new interface instance with updated configuration.
        
        Parameters
        ----------
        tmax : int, optional
            Maximum number of timesteps to use for FFT. 
            None means use all available timesteps.
        filters : dict, optional
            Filter configuration dict with stage keys (pre/post/live)
        cache_dir : str, optional
            External cache directory for results
            
        Returns
        -------
        FFTModeInterfaceNew
            Configured interface (for method chaining)
            
        Examples
        --------
        >>> job[0].fft.modes.configure(tmax=500).interactive_spectrum()
        >>> job[0].fft.modes.configure(filters={"normalize": True}).spectrum()
        """
        # Clone to avoid mutating original
        clone = self._clone()
        
        if tmax is not None:
            clone._tmax = tmax
        if filters is not None:
            clone._filters_config = copy.deepcopy(filters)
        if cache_dir is not None:
            clone._cache_dir = cache_dir
            
        return clone
    
    def _clone(self) -> "FFTModeInterfaceNew":
        """Create a shallow clone preserving runtime/configuration state."""
        clone = FFTModeInterfaceNew(self.fft_result_index, self.parent_fft)
        clone._dataset_context = self._dataset_context
        clone._slice_context = self._slice_context
        clone._tmax = self._tmax
        clone._filters_config = copy.deepcopy(self._filters_config) if self._filters_config else None
        clone._cache_dir = self._cache_dir
        clone._memory_cache = self._memory_cache  # Share memory cache
        clone._last_result = self._last_result
        clone._data_loader = self._data_loader
        clone._mode_analyzer = self._mode_analyzer
        clone._interactive_filters = dict(self._interactive_filters)
        clone._auto_compute_checked = self._auto_compute_checked
        return clone
    
    def _determine_tmax(self, default: int = 100) -> Optional[int]:
        """
        Determine number of time steps to load (dispersion-style priority).
        
        Priority order:
        1. Explicit slice from user (e.g., [:1000,...,2]) - ALWAYS respected
        2. Configured tmax via .configure(tmax=X)
        3. Default tmax=100 (only if no slice and no config)
        
        Returns
        -------
        int or None
            Number of timesteps, or None to use ALL available timesteps
        """
        # Check if user provided explicit time slice
        slice_length = self._infer_time_length_from_slice()
        
        if slice_length is not None:
            log.debug("Using EXPLICIT time slice from user: %d timesteps", slice_length)
            return slice_length
        
        # slice_length is None - could be:
        # A) User used [:] (slice with no stop) → wants ALL timesteps → return None
        # B) No slice at all → wants default optimization → use tmax
        
        if self._slice_context is not None:
            # Case A: User DID provide a slice, but it's [:] (no stop)
            log.debug("User provided [:] slice - using ALL available timesteps")
            return None
        
        # Case B: No slice at all - use configured tmax or default
        if self._tmax is not None:
            log.debug("No user slice - using configured tmax: %d timesteps", self._tmax)
            return int(self._tmax)
        
        log.debug("No slice or config - using default tmax: %d timesteps", default)
        return default
    
    def _infer_time_length_from_slice(self) -> Optional[int]:
        """
        Infer desired time window length from dataset slice info.
        
        For 5D data (t,z,y,x,c): data[:1000,...,2] → returns 1000
        
        Returns
        -------
        Optional[int]
            - None if no slice info, or slice is [:] (meaning "all timesteps")
            - Positive int if explicit time range specified
        """
        if self._slice_context is None:
            return None

        candidate = self._slice_context
        if isinstance(candidate, tuple) and candidate:
            for item in candidate:
                if item is Ellipsis:
                    continue
                candidate = item
                break

        if isinstance(candidate, slice):
            start = 0 if candidate.start is None else candidate.start
            stop = candidate.stop
            
            # If stop is None → [:] or [start:] → user wants ALL timesteps
            if stop is None:
                return None
            
            step = 1 if candidate.step is None else candidate.step
            if step == 0:
                return None
            length = math.ceil((stop - start) / step)
            return max(0, length)

        return None
    
    @property
    def spectrum_result(self):
        """Get spectrum using parent FFT with propagated slice context.
        
        This ensures consistency with job[0].m[...].fft.spectrum() calls.
        The slice_context (time slicing, component selection) is passed to
        the FFT spectrum calculation.
        
        Returns
        -------
        SpectrumResult
            Spectrum result with frequencies, power, peaks_info, component_label
        """
        return self.parent_fft._spectrum_impl(
            dset=self.dataset_name,
            slice_info=self._slice_context,
        )
    
    @property
    def frequencies(self):
        """Get frequencies from spectrum result (in GHz)."""
        return self.spectrum_result.frequencies
    
    @property
    def power_spectrum(self):
        """Get power spectrum (2D or 1D depending on component selection)."""
        return self.spectrum_result.power
    
    @property
    def data_loader(self):
        """Get or create data loader (lazy init)."""
        if self._data_loader is None:
            ModeDataLoader, ModeDataContext = _get_data_loader()
            
            context = ModeDataContext(
                zarr_path=self.zarr_path,
                dataset_name=self.dataset_name,
                slice_info=self._slice_context,
                component_index=self.component_index,
            )
            self._data_loader = ModeDataLoader(context)
            log.debug(f"Created data loader with dataset={self.dataset_name}, component={self.component_index}")
        
        return self._data_loader

    def filters(
        self,
        *,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        smooth_filter: Optional[str] = None,
        smooth_window: Optional[int] = None,
        smooth_sigma: Optional[float] = None,
        baseline_mode: Optional[str] = None,
        clip_percentile_low: Optional[float] = None,
        clip_percentile_high: Optional[float] = None,
        clip_low: Optional[float] = None,
        clip_high: Optional[float] = None,
        soft_threshold_percentile: Optional[float] = None,
        soft_threshold: Optional[float] = None,
        normalize: Optional[bool] = None,
        log_scale: Optional[bool] = None,
        show_peaks: Optional[bool] = None,
        peak_prominence: Optional[float] = None,
        peak_distance: Optional[int] = None,
    ) -> "FFTModeInterfaceNew":
        """Return cloned interface with configured interactive spectrum filters.

        This mirrors the fluent style used in dispersion:
        ``job[0].fft.modes.filters(...).plot()``.
        """
        clone = self._clone()
        updates: dict[str, Any] = {}

        fmin_val = freq_min if freq_min is not None else fmin
        fmax_val = freq_max if freq_max is not None else fmax
        clip_low_val = clip_percentile_low if clip_percentile_low is not None else clip_low
        clip_high_val = clip_percentile_high if clip_percentile_high is not None else clip_high
        soft_thr_val = (
            soft_threshold_percentile
            if soft_threshold_percentile is not None
            else soft_threshold
        )

        if fmin_val is not None:
            updates["freq_min"] = float(fmin_val)
        if fmax_val is not None:
            updates["freq_max"] = float(fmax_val)
        if smooth_filter is not None:
            updates["smooth_filter"] = str(smooth_filter)
        if smooth_window is not None:
            updates["smooth_window"] = int(smooth_window)
        if smooth_sigma is not None:
            updates["smooth_sigma"] = float(smooth_sigma)
        if baseline_mode is not None:
            updates["baseline_mode"] = str(baseline_mode)
        if clip_low_val is not None:
            updates["clip_percentile_low"] = float(clip_low_val)
        if clip_high_val is not None:
            updates["clip_percentile_high"] = float(clip_high_val)
        if soft_thr_val is not None:
            updates["soft_threshold_percentile"] = float(soft_thr_val)
        if normalize is not None:
            updates["normalize"] = bool(normalize)
        if log_scale is not None:
            updates["log_scale"] = bool(log_scale)
        if show_peaks is not None:
            updates["show_peaks"] = bool(show_peaks)
        if peak_prominence is not None:
            updates["peak_prominence"] = float(peak_prominence)
        if peak_distance is not None:
            updates["peak_distance"] = int(peak_distance)

        clone._interactive_filters.update(updates)
        return clone

    def clear_filters(self) -> "FFTModeInterfaceNew":
        """Return cloned interface without preconfigured filters."""
        clone = self._clone()
        clone._interactive_filters = {}
        return clone
    
    def _interactive_spectrum_impl(
        self,
        components: list = None,
        z_layer: int = -1,
        dpi: int = 100,
        figsize: tuple = (16, 10),
        toolbar: bool = True,
        show: bool = True,
        log_scale: Optional[bool] = None,
        normalize: Optional[bool] = None,
        freq_unit: Optional[str] = None,
        show_peaks: Optional[bool] = None,
        title: Optional[str] = None,
        initial_frequency: Optional[float] = None,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        smooth_filter: Optional[str] = None,
        smooth_window: Optional[int] = None,
        smooth_sigma: Optional[float] = None,
        baseline_mode: Optional[str] = None,
        clip_percentile_low: Optional[float] = None,
        clip_percentile_high: Optional[float] = None,
        soft_threshold_percentile: Optional[float] = None,
        peak_prominence: Optional[float] = None,
        peak_distance: Optional[int] = None,
        use_holography: bool = False,
        **kwargs,
    ):
        """Create interactive spectrum with mode visualization panels.
        
        **Key feature:** Uses FFT.spectrum() for spectrum data, ensuring
        consistency with `job[0].m[:200,...,1].fft.spectrum()` calls.
        Slice context (time range, component) is automatically propagated.
        
        Split layout:
        - Left: FFT power spectrum with clickable peaks
        - Right: 3x3 mode grid (magnitude, phase, combined) for each component
        
        Parameters
        ----------
        components : list, optional
            Components to show: ['x', 'y', 'z'] or [0, 1, 2]
            If component was selected via slicing, defaults to that component.
        z_layer : int
            Z-layer for mode visualization (default: -1 = top)
        dpi : int
            Figure resolution (default: 100)
        figsize : tuple
            Figure size (width, height) in inches
        log_scale : bool
            Use logarithmic Y-scale
        normalize : bool
            Normalize power to maximum
        freq_unit : str
            Frequency unit: Hz, kHz, MHz, GHz, THz
        show_peaks : bool
            Detect and show peaks
        title : str, optional
            Custom plot title
        initial_frequency : float, optional
            Start with this frequency selected
        **kwargs
            Additional arguments (find_peaks params, etc.)
        
        Returns
        -------
        Figure
            Interactive matplotlib figure
        
        Examples
        --------
        >>> # Full interactive view with mode panels
        >>> job[0].fft.modes.interactive_spectrum()
        
        >>> # Single component (my) with slice propagation
        >>> job[0].m[:200,...,1].fft.modes.interactive_spectrum(dpi=150)
        
        >>> # Start at specific frequency
        >>> job[0].fft.modes.interactive_spectrum(initial_frequency=9.5)
        """
        # Reuse an already computed spectrum when provided (e.g. SpectrumResult.plot.interactive()).
        # Fallback to parent FFT computation to preserve legacy behavior.
        spectrum_result = kwargs.pop("spectrum_result", None)

        # Convenience aliases so both spellings work transparently
        if "animate" in kwargs and "auto_animate" not in kwargs:
            kwargs["auto_animate"] = kwargs.pop("animate")
        elif "animate" in kwargs:
            kwargs.pop("animate")  # auto_animate already provided explicitly

        if spectrum_result is None:
            find_peaks_params = kwargs.pop("find_peaks", {"min_prominence": 0.01})
            spectrum_result = self.parent_fft._spectrum_impl(
                dset=self.dataset_name,
                slice_info=self._slice_context,
                find_peaks=find_peaks_params,
            )
            log.info(
                f"interactive_spectrum: using FFT spectrum with "
                f"dataset={self.dataset_name}, slice={self._slice_context}, "
                f"component={self.component_index}"
            )
        else:
            log.info(
                "interactive_spectrum: reusing provided SpectrumResult "
                f"(dataset={self.dataset_name}, slice={self._slice_context}, "
                f"component={self.component_index})"
            )
        
        # Auto-select components based on slice context
        if components is None and self.component_index is not None:
            # User selected specific component via slicing
            component_names = ['x', 'y', 'z']
            components = [component_names[self.component_index]]
            log.info(f"Auto-selected component: {components[0]} (from slice context)")

        viewer_kwargs = dict(self._interactive_filters)
        viewer_kwargs.update(kwargs)

        explicit_overrides = {
            "log_scale": log_scale,
            "normalize": normalize,
            "freq_unit": freq_unit,
            "show_peaks": show_peaks,
            "title": title,
            "initial_frequency": initial_frequency,
            "freq_min": freq_min,
            "freq_max": freq_max,
            "smooth_filter": smooth_filter,
            "smooth_window": smooth_window,
            "smooth_sigma": smooth_sigma,
            "baseline_mode": baseline_mode,
            "clip_percentile_low": clip_percentile_low,
            "clip_percentile_high": clip_percentile_high,
            "soft_threshold_percentile": soft_threshold_percentile,
            "peak_prominence": peak_prominence,
            "peak_distance": peak_distance,
        }
        for key, value in explicit_overrides.items():
            if value is not None:
                viewer_kwargs[key] = value
        
        # Some arguments are implemented only by the legacy analyzer view.
        legacy_only_keys = {"saveanim", "auto_animate", "auto_save", "method", "force", "use_fft_spectrum"}
        if any(key in viewer_kwargs for key in legacy_only_keys):
            toolbar = False
            log.info("Legacy interactive mode selected due to legacy-only arguments")

        # Topological components (+, -, rho, phi) and holography are only
        # handled by the legacy FMRModeAnalyzer path (VortexOptics engine).
        # Force the legacy path so the correct interface is always used.
        _TOPOLOGICAL_COMPONENTS = {"+", "-", "rho", "phi"}
        _has_topological = components is not None and any(
            str(c) in _TOPOLOGICAL_COMPONENTS for c in components
        )
        if use_holography or _has_topological:
            toolbar = False
            log.info(
                "Legacy interactive mode selected: use_holography=%s, "
                "topological_components=%s",
                use_holography,
                [c for c in (components or []) if str(c) in _TOPOLOGICAL_COMPONENTS],
            )


            resolved_log_scale = bool(viewer_kwargs.pop("log_scale", False))
            resolved_normalize = bool(viewer_kwargs.pop("normalize", True))
            resolved_freq_unit = str(viewer_kwargs.pop("freq_unit", "GHz"))
            resolved_show_peaks = bool(viewer_kwargs.pop("show_peaks", True))
            resolved_title = viewer_kwargs.pop("title", title)
            resolved_initial_frequency = viewer_kwargs.pop(
                "initial_frequency",
                initial_frequency,
            )

            InteractiveSpectrum = _get_interactive()
            viewer = InteractiveSpectrum(
                data_loader=self.data_loader,
                spectrum_result=spectrum_result,
                component_label=self.component_label,
                analyzer=self._legacy_analyzer,
                dpi=dpi,
                figsize=figsize,
            )
            return viewer.show(
                components=components,
                z_layer=z_layer,
                log_scale=resolved_log_scale,
                normalize=resolved_normalize,
                freq_unit=resolved_freq_unit,
                show_peaks=resolved_show_peaks,
                title=resolved_title,
                initial_frequency=resolved_initial_frequency,
                toolbar=True,
                use_holography=use_holography,
                show=show,
                **viewer_kwargs,
            )

        # Legacy fallback with full keyboard/mouse animation controls.
        toolbar_only_keys = {
            "freq_min",
            "freq_max",
            "smooth_filter",
            "smooth_window",
            "smooth_sigma",
            "baseline_mode",
            "clip_percentile_low",
            "clip_percentile_high",
            "soft_threshold_percentile",
            "peak_prominence",
            "peak_distance",
        }
        for key in toolbar_only_keys:
            viewer_kwargs.pop(key, None)

        return self._legacy_analyzer.interactive_spectrum(
            components=components,
            z_layer=z_layer,
            spectrum_result=spectrum_result,  # Inject FFT spectrum!
            figsize=figsize,
            dpi=dpi,
            log_scale=bool(viewer_kwargs.pop("log_scale", False)),
            normalize=bool(viewer_kwargs.pop("normalize", True)),
            show=show,
            use_holography=use_holography,
            **viewer_kwargs,
        )
    
    @property
    def interactive_spectrum(self):
        """Interactive spectrum with mode visualization (access for help, call to run).
        
        Access without () to see documentation. Call with () to execute.
        
        Examples
        --------
        >>> job[0].fft.modes.interactive_spectrum  # Show help
        >>> job[0].fft.modes.interactive_spectrum(dpi=150)  # Run
        """
        return InteractiveSpectrumHelper(self)
    
    # Alias for backward compatibility
    interactive_spectrum_old = interactive_spectrum
    
    # =========================================================================
    # Methods delegating to legacy FMRModeAnalyzer for features not yet migrated
    # =========================================================================
    
    @property
    def _legacy_analyzer(self):
        """Get legacy FMRModeAnalyzer for features not yet migrated."""
        if self._mode_analyzer is None:
            from . import FMRModeAnalyzer
            
            dataset = self._dataset_context or self.dataset_name
            self._mode_analyzer = FMRModeAnalyzer(
                zarr_path=self.zarr_path,
                dataset_name=dataset,
            )
        self._ensure_modes_ready()
        return self._mode_analyzer

    def _ensure_modes_ready(self) -> None:
        """Auto-bootstrap mode computation when mode datasets are missing."""
        if self._mode_analyzer is None or self._auto_compute_checked:
            return

        self._auto_compute_checked = True
        analyzer = self._mode_analyzer
        if getattr(analyzer, "modes_available", False):
            return
        if not hasattr(analyzer, "compute_modes"):
            # Lightweight/dummy analyzers may provide direct get_mode only.
            return

        dataset = self._dataset_context or self.dataset_name
        t_slice = self._extract_time_slice_from_context()
        log.info(
            "No precomputed FMR modes for dataset '%s' at '%s'. Running compute_modes() automatically.",
            dataset,
            self.zarr_path,
        )
        if t_slice is not None:
            log.info("Auto mode computation will use time slice: %s", t_slice)
        try:
            compute_kwargs: dict[str, Any] = {"save": True, "force": False}
            if t_slice is not None:
                compute_kwargs["t_slice"] = t_slice
            analyzer.compute_modes(**compute_kwargs)
            # Ensure analyzer refreshes its internal pointers if compute modified zarr.
            if hasattr(analyzer, "_load_data"):
                analyzer._load_data()
            if not getattr(analyzer, "modes_available", False):
                raise RuntimeError("Mode computation finished but modes are still unavailable")
            log.info("Auto mode computation completed for dataset '%s'.", dataset)
        except Exception as exc:
            raise RuntimeError(
                f"Automatic mode computation failed for dataset '{dataset}'. "
                "Run `job[0].fft.modes.compute_modes()` manually and retry."
            ) from exc

    def _extract_time_slice_from_context(self) -> Optional[slice]:
        """Extract time slice from dataset wrapper context."""
        if not isinstance(self._slice_context, tuple) or not self._slice_context:
            return None

        first = self._slice_context[0]
        if isinstance(first, slice):
            return first
        # For scalar time selection there are not enough samples for FFT.
        return None

    def _default_mode_frequency(self) -> float:
        """Resolve default mode frequency from peaks or maximum spectrum power."""
        spectrum_result = self.spectrum_result
        peaks_info = getattr(spectrum_result, "peaks_info", None)

        if peaks_info:
            try:
                first = peaks_info[0]
                if hasattr(first, "freq"):
                    return float(first.freq)
                if isinstance(first, dict):
                    return float(first.get("frequency", first.get("freq")))
                if isinstance(first, (list, tuple)) and first:
                    return float(first[0])
            except Exception:
                pass

        frequencies = np.asarray(getattr(spectrum_result, "frequencies", []), dtype=float)
        power = np.asarray(getattr(spectrum_result, "power", []))
        if frequencies.size == 0:
            raise ValueError("Cannot determine default frequency: no spectrum frequencies")

        if power.ndim > 1:
            if power.shape[-1] <= 3:
                avg_power = np.mean(power, axis=-1)
            else:
                avg_power = np.mean(power, axis=tuple(range(1, power.ndim)))
        else:
            avg_power = power

        idx = int(np.argmax(np.asarray(avg_power, dtype=float)))
        idx = max(0, min(idx, frequencies.size - 1))
        return float(frequencies[idx])

    def mode(
        self,
        f: Optional[float] = None,
        *,
        frequency: Optional[float] = None,
        z_layer: int = -1,
    ) -> ModeResult:
        """Return direct mode data wrapper for fluent plotting.

        Examples
        --------
        >>> mode = job[0].fft.modes.mode(f=20.0)
        >>> mode.plot.imshow()
        >>> mode.plot.interactive()
        """
        target_freq = frequency if frequency is not None else f
        if target_freq is None:
            target_freq = self._default_mode_frequency()

        mode_data = self._legacy_analyzer.get_mode(float(target_freq), z_layer=z_layer)
        return ModeResult(
            _modes=self,
            mode_data=mode_data,
            requested_frequency=float(target_freq),
            z_layer=int(z_layer),
        )

    def plot(
        self,
        *,
        show: bool = True,
        **kwargs,
    ):
        """Plot filtered FMR spectrum (fluent companion to ``filters()``)."""
        import matplotlib.pyplot as plt

        plot_kwargs = dict(self._interactive_filters)
        plot_kwargs.update(kwargs)

        plot_spectrum = _get_interactive_plot()
        fig = plot_spectrum(self.data_loader, **plot_kwargs)
        if show:
            plt.show()
        return fig

    # Alias matching explicit naming style.
    plot_spectrum = plot
    
    def plot_modes(
        self,
        frequency: float,
        z_layer: int = -1,
        component: str = "mz",
        show_phase: bool = True,
        show_magnitude: bool = True,
        dpi: int = 100,
        **kwargs,
    ):
        """Plot mode visualization at specified frequency.
        
        Parameters
        ----------
        frequency : float
            Frequency in GHz
        z_layer : int
            Z-layer index (default: -1 for top)
        component : str
            Component to visualize: 'mx', 'my', 'mz' (default: 'mz')
        show_phase : bool
            Show phase plot
        show_magnitude : bool
            Show magnitude plot
        dpi : int
            Figure resolution
        **kwargs
            Additional arguments for plot
        
        Returns
        -------
        Figure
            Matplotlib figure with mode visualization
        """
        # Use component from context if available
        if self.component_index is not None:
            component = ["mx", "my", "mz"][self.component_index]
        
        return self._legacy_analyzer.plot_modes(
            frequency=frequency,
            z_layer=z_layer,
            component=component,
            show_phase=show_phase,
            show_magnitude=show_magnitude,
            **kwargs,
        )
    
    def characterize_mode(
        self,
        frequency: float,
        z_layer: int = 0,
        verbose: bool = False,
        **kwargs,
    ):
        """Characterize mode at frequency.
        
        Parameters
        ----------
        frequency : float
            Frequency to analyze [GHz]
        z_layer : int
            Layer index
        verbose : bool
            Show detailed output
        **kwargs
            Additional arguments
        
        Returns
        -------
        ModeCharacterizationResult
            Classification result with metrics
        """
        return self._legacy_analyzer.characterize_mode(
            frequency=frequency,
            z_layer=z_layer,
            verbose=verbose,
            **kwargs,
        )
    
    def save_modes_animation(
        self,
        frequency: float = None,
        frequency_range: tuple = None,
        animation_type: str = "temporal",
        save_path: str = None,
        dpi: int = 100,
        **kwargs,
    ):
        """Create and save mode animation.
        
        Parameters
        ----------
        frequency : float, optional
            Single frequency for temporal animation
        frequency_range : tuple, optional
            (f_min, f_max) for frequency sweep
        animation_type : str
            'temporal', 'frequency', or 'phase'
        save_path : str, optional
            Output file path (.mp4 or .gif)
        dpi : int
            Animation resolution
        **kwargs
            Additional arguments
        
        Returns
        -------
        Animation or path
            Animation object or saved file path
        """
        return self._legacy_analyzer.save_modes_animation(
            frequency=frequency,
            frequency_range=frequency_range,
            animation_type=animation_type,
            save_path=save_path,
            **kwargs,
        )
    
    def compute_modes(self, dset: str = None, **kwargs):
        """Compute/recompute modes for dataset.
        
        Parameters
        ----------
        dset : str, optional
            Dataset name (uses context if not specified)
        **kwargs
            Additional arguments for mode computation
        """
        dataset = dset or self._dataset_context or self.dataset_name
        return self._legacy_analyzer.compute_modes(dset=dataset, **kwargs)
    
    def __repr__(self) -> str:
        """Rich representation of modes interface."""
        try:
            return self._rich_display()
        except Exception:
            return self._basic_display()
    
    def _rich_display(self) -> str:
        """Generate rich help display."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich.syntax import Syntax
            from io import StringIO
            
            capture = StringIO()
            console = Console(file=capture, force_terminal=True, width=100)
            
            # Title
            title = Text()
            title.append("🎯 FFT Mode Analyzer\n", style="bold cyan")
            title.append(f"📁 Dataset: {self.dataset_name}\n", style="dim")
            if self.component_label:
                title.append(f"📊 Component: {self.component_label}\n", style="green")
            title.append(f"📂 Path: {self.zarr_path}", style="dim")
            
            console.print(Panel(title, border_style="cyan"))
            
            # Methods table
            methods = Table(show_header=True, header_style="bold yellow")
            methods.add_column("Method", style="cyan")
            methods.add_column("Description", style="white")
            
            methods.add_row("filters(...)", "Clone interface with FMR spectrum filters")
            methods.add_row("plot(...)", "Plot filtered FMR spectrum")
            methods.add_row("mode(f=...)", "Direct mode data access with .plot helpers")
            methods.add_row("interactive_spectrum(dpi=100)", "Interactive toolbar spectrum+mode view")
            methods.add_row("plot_modes(frequency)", "Visualize mode at frequency")
            methods.add_row("characterize_mode(frequency)", "Classify mode type")
            methods.add_row("save_modes_animation(...)", "Create mode animations")
            methods.add_row("compute_modes()", "Compute/recompute modes")
            
            console.print(methods)
            console.print("")
            
            # Examples
            example = '''# With component selection (my) and DPI:
job[0].m[:200,...,1].fft.modes.interactive_spectrum(dpi=150)

# All components:
job[0].fft.modes.interactive_spectrum(log_scale=True)

# Fluent filters + plot:
job[0].fft.modes.filters(
    freq_min=2.0,
    freq_max=25.0,
    smooth_filter="gaussian",
    smooth_sigma=1.2,
    baseline_mode="linear",
).plot()

# Direct mode access:
mode = job[0].fft.modes.mode(f=9.5)
mode.plot.imshow(component="z", value="phase")
mode.plot.interactive()

# Mode visualization:
job[0].fft.modes.plot_modes(frequency=9.5)

# Mode animation:
job[0].fft.modes.save_modes_animation(frequency=9.5, save_path="mode.mp4")'''
            
            syntax = Syntax(example, "python", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title="[bold green]Examples", border_style="green"))
            
            return capture.getvalue()
        except ImportError:
            return self._basic_display()
    
    def _basic_display(self) -> str:
        """Basic text display."""
        return (
            f"FFTModeInterface(dataset={self.dataset_name}, "
            f"component={self.component_label or 'all'})"
        )
