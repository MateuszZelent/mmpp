"""Plotting helpers for transmission analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

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
    trim_0f: Optional[int] = None  # Number of lowest frequency points to remove


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
            return self.result.transmission, "Transmission $T(f,x)"
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

        data, default_label = self._select_data(config.which)
        if data.size == 0:
            raise ValueError("Transmission result contains no data to plot")

        freq_unit = config.freq_unit
        if freq_unit not in FREQ_SCALE:
            raise ValueError(f"Unsupported frequency unit: {freq_unit}")
        freq_scale = FREQ_SCALE[freq_unit]
        freqs = self.result.frequencies * freq_scale

        # Apply trim_0f if specified (remove lowest frequency points)
        trim_idx = 0
        if config.trim_0f is not None and config.trim_0f > 0:
            trim_idx = min(config.trim_0f, len(freqs) - 1)
            freqs = freqs[trim_idx:]
            data = data[trim_idx:, :]
            log.debug(f"Trimmed {trim_idx} lowest frequency points (trim_0f={config.trim_0f})")

        x_positions = self.result.x_positions
        x_edges = _centers_to_edges(x_positions)
        freq_edges = _centers_to_edges(freqs)

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

        quad = ax.pcolormesh(
            x_edges,
            freq_edges,
            mesh_data,
            cmap=config.cmap,
            shading="auto",
            vmin=None if norm else vmin,
            vmax=None if norm else vmax,
            norm=norm,
        )

        ylabel = f"Frequency ({freq_unit})"
        xlabel = "$x-axis$" if config.x_unit == "index" else "x"
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        title = config.title or default_label
        ax.set_title(title)

        if config.show_colorbar:
            fig.colorbar(quad, ax=ax, label=default_label)

        ax.set_ylim(freq_edges[0], freq_edges[-1])
        ax.set_xlim(x_edges[0], x_edges[-1])

        return fig, ax, quad


__all__ = [
    "TransmissionPlotter",
    "TransmissionPlotConfig",
]
