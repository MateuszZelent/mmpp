"""Result models for vortex energy-channel analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)
from mmpp._shared.repr_html import make_simple_card


@dataclass
class EnergyTimeSeriesResult:
    """Time-resolved energy channels extracted from table data."""

    time: np.ndarray
    channels: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available_channels(self) -> list[str]:
        """Sorted list of available energy channel names."""
        return sorted(self.channels.keys())

    @property
    def total_energy(self) -> np.ndarray:
        """Total energy channel (explicit or reconstructed sum)."""
        if "E_total" in self.channels:
            return np.asarray(self.channels["E_total"], dtype=float)
        if not self.channels:
            return np.array([], dtype=float)
        stacked = np.column_stack([np.asarray(v, dtype=float) for v in self.channels.values()])
        return np.asarray(np.sum(stacked, axis=1), dtype=float)

    @property
    def plt(self) -> EnergyPlotAccessor:
        """Plotting accessor."""
        return EnergyPlotAccessor(self)

    def _repr_html_(self) -> str:
        rows = [
            ("samples", str(int(np.asarray(self.time).size))),
            ("n_channels", str(len(self.channels))),
            ("channels", ", ".join(self.available_channels) if self.channels else "(none)"),
            (".plt.time_resolved()", "Plot selected channels vs time"),
        ]
        return make_simple_card(
            title="EnergyTimeSeriesResult",
            subtitle="Time-resolved energy channels extracted from table",
            rows=rows,
        )


@dataclass
class EffectivePotentialResult:
    """Effective radial potential reconstructed from trajectory statistics."""

    radius_m: np.ndarray
    potential_j: np.ndarray
    probability: np.ndarray
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def plt(self) -> EffectivePotentialPlotAccessor:
        """Plotting accessor."""
        return EffectivePotentialPlotAccessor(self)

    def _repr_html_(self) -> str:
        rows = [
            ("samples", str(int(np.asarray(self.radius_m).size))),
            ("method", str(self.method)),
            ("radius_max_nm", f"{(np.max(self.radius_m) * 1e9 if self.radius_m.size else float('nan')):.6g}"),
            (
                "potential_max_j",
                f"{(np.max(self.potential_j) if self.potential_j.size else float('nan')):.6g}",
            ),
            (".plt.potential()", "Plot W(r)"),
            (".plt.probability()", "Plot radial occupancy P(r)"),
        ]
        return make_simple_card(
            title="EffectivePotentialResult",
            subtitle="Effective radial potential reconstructed from trajectory",
            rows=rows,
        )


@dataclass
class PinningSite:
    """Local minimum in effective potential interpreted as a pinning site."""

    radius_m: float
    potential_j: float
    depth_j: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PinningResult:
    """Detected pinning sites over a reconstructed potential."""

    potential: EffectivePotentialResult
    sites: list[PinningSite]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def plt(self) -> PinningPlotAccessor:
        """Plotting accessor."""
        return PinningPlotAccessor(self)

    def _repr_html_(self) -> str:
        if self.sites:
            conf_mean = float(np.mean([site.confidence for site in self.sites]))
        else:
            conf_mean = float("nan")
        rows = [
            ("n_sites", str(len(self.sites))),
            ("confidence_mean", f"{conf_mean:.6g}"),
            ("potential_method", str(self.potential.method)),
            (".plt.potential_with_sites()", "Plot W(r) with detected minima"),
        ]
        return make_simple_card(
            title="PinningResult",
            subtitle="Detected pinning sites from effective potential",
            rows=rows,
        )


class EnergyPlotAccessor:
    """Plot helpers for :class:`EnergyTimeSeriesResult`."""

    def __init__(self, result: EnergyTimeSeriesResult):
        self._result = result

    def time_resolved(
        self,
        *,
        channels: list[str] | tuple[str, ...] | None = None,
        ax=None,
        **kwargs,
    ):
        """Plot selected energy channels versus time."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        selected = list(channels) if channels is not None else self._result.available_channels
        if not selected:
            ax.set_title("No energy channels available")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Energy [J]")
            apply_axes_style(ax, style_kwargs)
            return ax

        for name in selected:
            if name not in self._result.channels:
                continue
            ax.plot(self._result.time, self._result.channels[name], label=name, **plot_kwargs)

        if selected:
            ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Energy [J]")
        ax.set_title("Energy vs time")
        apply_axes_style(ax, style_kwargs)
        return ax


class EffectivePotentialPlotAccessor:
    """Plot helpers for :class:`EffectivePotentialResult`."""

    def __init__(self, result: EffectivePotentialResult):
        self._result = result

    def potential(self, *, ax=None, as_nev: bool = False, **kwargs):
        """Plot effective potential versus radial coordinate."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        radius_nm = np.asarray(self._result.radius_m, dtype=float) * 1e9
        values = np.asarray(self._result.potential_j, dtype=float)
        ylabel = "Potential [J]"
        if as_nev:
            values = values * 6.241509074e27
            ylabel = "Potential [neV]"

        ax.plot(radius_nm, values, **plot_kwargs)
        ax.set_xlabel("Radius [nm]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Effective potential ({self._result.method})")
        apply_axes_style(ax, style_kwargs)
        return ax

    def probability(self, *, ax=None, **kwargs):
        """Plot radial probability density used for potential reconstruction."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        radius_nm = np.asarray(self._result.radius_m, dtype=float) * 1e9
        prob = np.asarray(self._result.probability, dtype=float)
        ax.plot(radius_nm, prob, **plot_kwargs)
        ax.set_xlabel("Radius [nm]")
        ax.set_ylabel("Probability [a.u.]")
        ax.set_title("Radial occupancy")
        apply_axes_style(ax, style_kwargs)
        return ax


class PinningPlotAccessor:
    """Plot helpers for :class:`PinningResult`."""

    def __init__(self, result: PinningResult):
        self._result = result

    def potential_with_sites(self, *, ax=None, as_nev: bool = False, **kwargs):
        """Plot potential and mark detected pinning sites."""
        ax = self._result.potential.plt.potential(ax=ax, as_nev=as_nev, **kwargs)
        if not self._result.sites:
            return ax

        x = np.array([site.radius_m for site in self._result.sites], dtype=float) * 1e9
        y = np.array([site.potential_j for site in self._result.sites], dtype=float)
        if as_nev:
            y = y * 6.241509074e27
        ax.scatter(x, y, color="tab:red", s=35, zorder=5, label="pinning site")
        ax.legend()
        return ax


__all__ = [
    "EnergyTimeSeriesResult",
    "EffectivePotentialResult",
    "PinningSite",
    "PinningResult",
]
