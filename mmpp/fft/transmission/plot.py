"""Plotting helpers for transmission analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import matplotlib as mpl

from ...cli.logging_config import get_mmpp_logger, setup_mmpp_logging

from .compute import TransmissionResult


log = get_mmpp_logger("mmpp.fft.transmission.plot")


try:  # pragma: no cover - optional dependency check
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.colors import LogNorm

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    Axes = Figure = None  # type: ignore


FrequencyUnit = Literal["Hz", "kHz", "MHz", "GHz"]
XUnit = Literal["index", "cell"]


FREQ_SCALE = {
    "Hz": 1.0,
    "kHz": 1e-3,
    "MHz": 1e-6,
    "GHz": 1e-9,
}


def tex_escape(s: str) -> str:
    """Escape special LaTeX characters when usetex=True.
    
    Args:
        s: String to escape
        
    Returns:
        Escaped string safe for LaTeX rendering
    """
    # Escapowanie potrzebne tylko gdy usetex=True
    if not mpl.rcParams.get('text.usetex', False):
        return s
    for a, b in {
        '\\': r'\textbackslash{}',
        '_': r'\_',
        '%': r'\%',
        '&': r'\&',
        '#': r'\#',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }.items():
        s = s.replace(a, b)
    return s


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([0.0, 1.0])
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)

    diffs = np.diff(values)
    start_edge = values[0] - diffs[0] / 2.0
    end_edge = values[-1] + diffs[-1] / 2.0
    interior = values[:-1] + diffs / 2.0
    edges = np.concatenate(([start_edge], interior, [end_edge]))
    return edges


@dataclass(slots=True)
class TransmissionPlotConfig:
    which: Literal["transmission", "power", "power_plus", "power_minus"] = "transmission"
    freq_unit: FrequencyUnit = "GHz"
    x_unit: XUnit = "index"
    cmap: str = "viridis"
    log_scale: bool = False
    show_colorbar: bool = True
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    title: Optional[str] = None
    trim_0f: Optional[int] = None  # Number of lowest frequency points to set to zero (suppress DC/low freq)
    fmax: Optional[float] = None  # Maximum frequency to display (in freq_unit units)


class TransmissionPlotter:
    """Create heatmaps for transmission results."""

    def __init__(self, result: TransmissionResult, *, debug: bool | None = None):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for transmission plotting")

        self.result = result
        if debug is None:
            debug = False
        setup_mmpp_logging(debug=debug, logger_name="mmpp.fft.transmission.plot")

    def _select_data(self, which: str) -> tuple[np.ndarray, str]:
        which = which.lower()
        if which == "transmission":
            data = self.result.transmission
            # Handle complex data from raw_fft_output mode
            if np.iscomplexobj(data):
                # Convert complex to magnitude (absolute value, NO squaring)
                data = np.abs(data)
                label = "$|FFT|$ (raw mode)"
            else:
                label = "Transmission $T(f,x)"
            return data, label
        if which == "power":
            return self.result.power_map, "Averaged Power"
        if which == "power_plus":
            if self.result.power_plus is None:
                raise ValueError("Circular component power_plus not computed. Enable enable_circular_components in config.")
            return self.result.power_plus, "$P_+(f,x)$"
        if which == "power_minus":
            if self.result.power_minus is None:
                raise ValueError("Circular component power_minus not computed. Enable enable_circular_components in config.")
            return self.result.power_minus, "$P_-(f,x)$"
        raise ValueError(f"Unsupported data selection: {which}")

    def plot(
        self,
        *,
        config: Optional[TransmissionPlotConfig] = None,
        ax: Optional[Axes] = None,
        debug: bool = False,
        **kwargs,
    ) -> tuple[Figure, Axes, Any]:
        if config is None:
            config = TransmissionPlotConfig()

        if debug:
            print(f"\n{'='*60}")
            print(f"🔍 DEBUG: TransmissionPlotter.plot()")
            print(f"{'='*60}")
            print(f"Config: {config}")
            if self.result.dx is not None:
                print(
                    f"result.dx = {self.result.dx:.3e} m ({self.result.dx * 1e9:.3f} nm)"
                )
            else:
                print("result.dx = None")
            print(f"result.x_positions shape: {self.result.x_positions.shape}")
            print(f"result.x_positions[0:5]: {self.result.x_positions[:5]}")
            print(f"result.frequencies shape: {self.result.frequencies.shape}")

        data, default_label = self._select_data(config.which)
        if data.size == 0:
            raise ValueError("Transmission result contains no data to plot")

        # 🔑 Handle raw_fft_output mode with multi-dimensional data
        # Raw mode gives (freq, z, x, comp) - need to reduce to (freq, x)
        if data.ndim > 2:
            if debug:
                print(f"\n⚠️  Multi-dimensional data detected: {data.shape}")
                print(f"   Reducing to 2D for plotting...")
            
            # Strategy: sum over components, take z=0 (or sum over z if average_mode != "none")
            # Expected shapes:
            # - (freq, z, x, comp) for raw_fft_output with sum_m/sum_fft
            # - (freq, z, y, x, comp) for raw_fft_output with y_integration_mode="none"
            
            if data.ndim == 5:  # (freq, z, y, x, comp)
                # Sum over y and components, take z=0
                data = data[:, 0, :, :, :].sum(axis=(1, 3))  # → (freq, x)
                if debug:
                    print(f"   5D → 2D: summed over (y, comp), extracted z=0 → {data.shape}")
            elif data.ndim == 4:  # (freq, z, x, comp)
                # Sum over components, take z=0
                data = data[:, 0, :, :].sum(axis=2)  # → (freq, x)
                if debug:
                    print(f"   4D → 2D: summed over comp, extracted z=0 → {data.shape}")
            elif data.ndim == 3:  # (freq, x, comp) or (freq, z, x)
                # Could be either - try to detect
                if data.shape[1] == 1:  # Likely (freq, 1, x) - z dimension
                    data = data[:, 0, :]  # → (freq, x)
                    if debug:
                        print(f"   3D → 2D: extracted z=0 → {data.shape}")
                else:
                    # Assume (freq, x, comp) - sum over components
                    data = data.sum(axis=2)  # → (freq, x)
                    if debug:
                        print(f"   3D → 2D: summed over comp → {data.shape}")
            
            if data.ndim != 2:
                raise ValueError(f"Could not reduce data to 2D for plotting. Final shape: {data.shape}")
            
            if debug:
                print(f"   Final 2D shape: {data.shape}")
                print(f"   min={data.min():.3e}, max={data.max():.3e}")

        freq_unit = config.freq_unit
        if freq_unit not in FREQ_SCALE:
            raise ValueError(f"Unsupported frequency unit: {freq_unit}")
        freq_scale = FREQ_SCALE[freq_unit]
        freqs = self.result.frequencies * freq_scale
        
        if debug:
            print(f"\nFrequency scaling:")
            print(f"  freq_unit = {freq_unit}")
            print(f"  freq_scale = {freq_scale}")
            print(f"  freqs[0:5] = {freqs[:5]}")

        # Apply trim_0f if specified (set lowest frequency points to zero)
        trim_idx = 0
        if config.trim_0f is not None and config.trim_0f > 0:
            trim_idx = min(config.trim_0f, len(freqs) - 1)
            if debug:
                print(f"\nApplying trim_0f:")
                print(f"  trim_0f = {config.trim_0f}")
                print(f"  trim_idx = {trim_idx}")
                print(f"  Setting data[0:{trim_idx}, :] to zero")
            data[:trim_idx, :] = 0  # Set to zero instead of removing
            log.debug(f"Set {trim_idx} lowest frequency points to zero (trim_0f={config.trim_0f})")
        elif debug:
            print(f"\nNo trim_0f applied (trim_0f={config.trim_0f})")

        # Apply fmax if specified (remove frequencies above maximum)
        if config.fmax is not None and config.fmax > 0:
            fmax_mask = freqs <= config.fmax
            n_above = (~fmax_mask).sum()
            if np.any(fmax_mask):
                freqs = freqs[fmax_mask]
                data = data[fmax_mask, :]
                log.debug(f"Trimmed {n_above} frequency points above fmax={config.fmax} {freq_unit}")
            else:
                log.warning(f"fmax={config.fmax} {freq_unit} is below all frequencies, ignoring")

        x_positions = self.result.x_positions
        x_edges = _centers_to_edges(x_positions)
        freq_edges = _centers_to_edges(freqs)
        
        if debug:
            print(f"\nX-axis setup:")
            if self.result.dx is not None:
                print(
                    f"  result.dx = {self.result.dx:.3e} m ({self.result.dx * 1e9:.3f} nm)"
                )
            else:
                print("  result.dx = None")
            print(f"  x_positions shape = {x_positions.shape}")
            print(f"  x_positions[0:5] = {x_positions[:5]}")
            print(f"  x_edges[0:5] = {x_edges[:5]}")
            print(f"  x_unit = {config.x_unit}")
            if self.result.dx is not None:
                print(f"  ✅ dx available → x should be in nm")
            else:
                print(f"  ⚠️  dx NOT available → x will be in indices")

        mesh_data = np.ma.masked_invalid(data)

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 6)), dpi=kwargs.pop("dpi", 100))
        else:
            fig = ax.figure

        vmin = config.vmin
        vmax = config.vmax

        norm = None
        if config.log_scale:
            positive = mesh_data > 0
            vmin = config.vmin or (mesh_data[positive].min() if np.any(positive) else 1e-12)
            vmax = config.vmax or mesh_data.max()
            if vmin <= 0:
                vmin = 1e-12
            if vmax <= 0:
                vmax = 1.0
            norm = LogNorm(vmin=vmin, vmax=vmax)

        extent = (
            float(x_edges[0]),
            float(x_edges[-1]),
            float(freq_edges[0]),
            float(freq_edges[-1]),
        )

        image = ax.imshow(
            mesh_data,
            cmap=config.cmap,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=None if norm else vmin,
            vmax=None if norm else vmax,
            norm=norm,
            interpolation="nearest",
        )

        ylabel = f"Frequency ({freq_unit})"
        # Auto-detect x label based on whether dx is available
        if config.x_unit == "index":
            xlabel = "x (cell index)"
        elif self.result.dx is not None:
            xlabel = "x (nm)"
        else:
            xlabel = "x (cell index)"
        
        if debug:
            print(f"\nAxis labels:")
            print(f"  xlabel = '{xlabel}'")
            print(f"  ylabel = '{ylabel}'")
            print(f"  config.x_unit = '{config.x_unit}'")
            print(f"  Logic: x_unit={config.x_unit}, dx={self.result.dx} → xlabel={xlabel}")
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # Include spatial_window and formula in the title if using auto title
        if config.title:
            title = config.title
        else:
            # Get spatial window size
            spatial_window = self.result.config.spatial_window
            # Equation showing: sum over y, then spatial window, then FFT
            # Use .format() to insert W value
            title = r"Transmission: $\left|\operatorname{{FFT}}_{{t}}\!\left(\sum_{{x' \in W(x)}} \sum_{{y}} m_{{z}}(t, x', y)\right)\right|$, W={}".format(spatial_window)
        ax.set_title(title)

        if config.show_colorbar:
            fig.colorbar(image, ax=ax, label=default_label)

        ax.set_ylim(freq_edges[0], freq_edges[-1])
        ax.set_xlim(x_edges[0], x_edges[-1])

        return fig, ax, image


__all__ = [
    "TransmissionPlotter",
    "TransmissionPlotConfig",
]
