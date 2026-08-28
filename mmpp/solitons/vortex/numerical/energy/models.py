"""Result models for vortex energy-channel analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...._method_helpers import InteractiveNodeMixin
from ..._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)


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
        stacked = np.column_stack(
            [np.asarray(v, dtype=float) for v in self.channels.values()]
        )
        return np.asarray(np.sum(stacked, axis=1), dtype=float)

    @property
    def plt(self) -> EnergyPlotAccessor:
        """Plotting accessor."""
        return EnergyPlotAccessor(self)

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import (
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "Energy Time Series Result",
            icon="📉",
            subtitle="Time-resolved energy channels extracted from the simulation table.",
            sections=[
                metrics_section_html(
                    [
                        ("samples", int(np.asarray(self.time).size), None),
                        ("n_channels", len(self.channels), None),
                        (
                            "channels",
                            ", ".join(self.available_channels)
                            if self.channels
                            else "(none)",
                            None,
                        ),
                    ]
                ),
                examples_section_html(
                    "etrace = jobs[-1].solitons.vortex.energy.time_resolved()\n"
                    "etrace.plt.time_resolved()",
                    title="Result Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Energy time-series API help",
                prefix="jobs[-1].solitons.vortex.energy.time_resolved()",
                properties=[
                    ("time", "Time axis"),
                    ("channels", "Energy channels dictionary"),
                    ("available_channels", "Sorted list of channels"),
                    ("total_energy", "Total energy channel"),
                    ("plt", "Plotting accessor"),
                ],
                subtitle="Live attributes of the energy time-series result.",
                chrome=False,
            ),
            uid=f"energy-time-series-result-{str(_uuid.uuid4())[:8]}",
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
        import uuid as _uuid

        from mmpp._repr_helpers import (
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "Effective Potential Result",
            icon="🧭",
            subtitle="Effective radial potential reconstructed from vortex trajectory statistics.",
            sections=[
                metrics_section_html(
                    [
                        ("samples", int(np.asarray(self.radius_m).size), None),
                        ("method", self.method, None),
                        (
                            "radius_max_nm",
                            f"{(np.max(self.radius_m) * 1e9 if self.radius_m.size else float('nan')):.6g}",
                            None,
                        ),
                        (
                            "potential_max_j",
                            f"{(np.max(self.potential_j) if self.potential_j.size else float('nan')):.6g}",
                            None,
                        ),
                    ]
                ),
                examples_section_html(
                    "pot = jobs[-1].solitons.vortex.energy.potential()\n"
                    "pot.plt.potential()\n"
                    "pot.plt.probability()",
                    title="Result Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Effective potential API help",
                prefix="jobs[-1].solitons.vortex.energy.potential()",
                properties=[
                    ("radius_m", "Radial coordinate"),
                    ("potential_j", "Potential profile"),
                    ("probability", "Radial occupancy"),
                    ("method", "Reconstruction method"),
                    ("plt", "Plotting accessor"),
                ],
                subtitle="Live attributes of the effective potential result.",
                chrome=False,
            ),
            uid=f"effective-potential-result-{str(_uuid.uuid4())[:8]}",
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
        import uuid as _uuid

        from mmpp._repr_helpers import (
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "Pinning Result",
            icon="📍",
            subtitle="Detected pinning sites from the reconstructed effective potential.",
            sections=[
                metrics_section_html(
                    [
                        ("n_sites", len(self.sites), None),
                        ("confidence_mean", f"{conf_mean:.6g}", None),
                        ("potential_method", self.potential.method, None),
                    ]
                ),
                examples_section_html(
                    "pin = jobs[-1].solitons.vortex.energy.pinning()\n"
                    "pin.plt.potential_with_sites()",
                    title="Result Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Pinning-result API help",
                prefix="jobs[-1].solitons.vortex.energy.pinning()",
                properties=[
                    ("potential", "Underlying effective potential"),
                    ("sites", "Detected pinning sites"),
                    ("plt", "Plotting accessor"),
                ],
                subtitle="Live attributes of the detected pinning-site result.",
                chrome=False,
            ),
            uid=f"pinning-result-{str(_uuid.uuid4())[:8]}",
        )


class EnergyPlotAccessor(InteractiveNodeMixin):
    """Plot helpers for :class:`EnergyTimeSeriesResult`."""

    _interactive_owner = "result.plt"
    _interactive_nodes = frozenset({"time_resolved"})

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

        selected = (
            list(channels) if channels is not None else self._result.available_channels
        )
        if not selected:
            ax.set_title("No energy channels available")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Energy [J]")
            apply_axes_style(ax, style_kwargs)
            return ax

        for name in selected:
            if name not in self._result.channels:
                continue
            ax.plot(
                self._result.time,
                self._result.channels[name],
                label=name,
                **plot_kwargs,
            )

        if selected:
            ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Energy [J]")
        ax.set_title("Energy vs time")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "EnergyPlotAccessor",
            [
                (
                    ".time_resolved(channels=['E_total','E_exch'])",
                    "Energy channels vs time",
                    "channels: list of channel names (None = all). Accepts matplotlib kwargs.",
                ),
            ],
        )


class EffectivePotentialPlotAccessor(InteractiveNodeMixin):
    """Plot helpers for :class:`EffectivePotentialResult`."""

    _interactive_owner = "result.plt"
    _interactive_nodes = frozenset({"potential", "probability"})

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

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "EffectivePotentialPlotAccessor",
            [
                (
                    ".potential(as_nev=False)",
                    "Effective potential U(r) vs radius",
                    "as_nev: convert energy to neV.",
                ),
                (
                    ".probability()",
                    "Radial probability density p(r)",
                    "Used for Boltzmann inversion.",
                ),
            ],
        )


class PinningPlotAccessor(InteractiveNodeMixin):
    """Plot helpers for :class:`PinningResult`."""

    _interactive_owner = "result.plt"
    _interactive_nodes = frozenset({"potential_with_sites"})

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

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "PinningPlotAccessor",
            [
                (
                    ".potential_with_sites(as_nev=False)",
                    "Effective potential with pinning site markers",
                    "as_nev: convert to neV. Marks local minima as pinning sites.",
                ),
            ],
        )


__all__ = [
    "EnergyTimeSeriesResult",
    "EffectivePotentialResult",
    "PinningSite",
    "PinningResult",
]
