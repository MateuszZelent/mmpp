"""High-level interface for vortex energy analysis."""

from __future__ import annotations

import warnings
from typing import Any

from ...._method_helpers import InteractiveNodeMixin
from ..._shared.models import TrajectoryResult
from ...config import VortexConfig
from .models import EffectivePotentialResult, EnergyTimeSeriesResult, PinningResult
from .pinning import detect_pinning_sites
from .potential import potential_from_boltzmann, potential_from_energy_channel
from .time_resolved import extract_energy_time_series


class EnergyInterface(InteractiveNodeMixin):
    """Energy namespace with table-driven time-resolved channels."""

    _interactive_owner = "job[0].vortex.energy"
    _interactive_nodes = frozenset({"time_resolved", "potential", "pinning"})
    _interactive_descriptions = {
        "time_resolved": "Load energy channels sampled over simulation time.",
        "potential": "Estimate the effective radial potential from trajectory statistics.",
        "pinning": "Detect pinning sites as local minima of the effective potential.",
    }
    _interactive_examples = {
        "time_resolved": [
            "energy = job[0].vortex.energy.time_resolved()",
            "energy.plt.time_resolved()",
        ],
        "potential": [
            "potential = job[0].vortex.energy.potential(method='auto')",
            "potential.plt.potential()",
        ],
        "pinning": [
            "pinning = job[0].vortex.energy.pinning()",
            "pinning.plt.potential_with_sites()",
        ],
    }

    def __init__(
        self,
        job_result,
        dataset_name: str | None,
        slice_info: Any | None,
        config: VortexConfig,
        core_interface=None,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config
        self._core = core_interface
        self._last_result: EnergyTimeSeriesResult | None = None
        self._last_potential: EffectivePotentialResult | None = None
        self._last_pinning: PinningResult | None = None

    def time_resolved(
        self,
        *,
        columns: list[str] | tuple[str, ...] | None = None,
        strict: bool | None = None,
        force: bool = False,
    ) -> EnergyTimeSeriesResult:
        """Load energy-vs-time channels from the simulation table."""
        if (
            not force
            and self._last_result is not None
            and columns is None
            and strict is None
        ):
            return self._last_result

        strict_mode = (
            bool(self._config.energy.strict_missing) if strict is None else bool(strict)
        )
        result = extract_energy_time_series(
            self._job,
            columns=columns,
            prefixes=tuple(self._config.energy.column_prefixes),
        )

        if strict_mode and (not result.channels):
            available = result.metadata.get("available_columns", [])
            raise ValueError(
                f"No energy channels found in table. Available columns: {available}"
            )

        if not result.channels:
            warnings.warn(
                "No energy channels were found in table (expected prefixes "
                f"{self._config.energy.column_prefixes}).",
                RuntimeWarning,
                stacklevel=2,
            )

        self._last_result = result
        return result

    def _resolve_trajectory(self, trajectory: TrajectoryResult | None):
        if trajectory is not None:
            return trajectory
        if self._core is None:
            return None
        return self._core.track(method="centroid")

    def potential(
        self,
        *,
        trajectory: TrajectoryResult | None = None,
        method: str = "auto",
        temperature_k: float = 300.0,
        bins: int = 64,
        force: bool = False,
    ) -> EffectivePotentialResult:
        """Estimate effective radial potential from trajectory statistics."""
        if (
            not force
            and self._last_potential is not None
            and trajectory is None
            and method == "auto"
            and abs(float(temperature_k) - 300.0) < 1e-15
            and int(bins) == 64
        ):
            return self._last_potential

        traj = self._resolve_trajectory(trajectory)
        if traj is None:
            raise ValueError(
                "No trajectory available for potential reconstruction. "
                "Pass trajectory=... or use this interface from job.m.vortex."
            )

        method_norm = str(method).lower()
        if method_norm not in {"auto", "boltzmann", "energy_bin"}:
            raise ValueError("method must be 'auto', 'boltzmann', or 'energy_bin'")

        table_energy = self.time_resolved(force=False)
        has_e_total = "E_total" in table_energy.channels
        can_energy_bin = has_e_total and table_energy.time.size == traj.time.size

        if method_norm == "energy_bin" and not can_energy_bin:
            raise ValueError(
                "method='energy_bin' requires E_total channel aligned with trajectory time samples."
            )

        if method_norm == "energy_bin" or (method_norm == "auto" and can_energy_bin):
            result = potential_from_energy_channel(
                traj,
                table_energy.channels["E_total"],
                bins=bins,
            )
        else:
            result = potential_from_boltzmann(
                traj,
                temperature_k=temperature_k,
                bins=bins,
            )

        self._last_potential = result
        return result

    def pinning(
        self,
        *,
        potential: EffectivePotentialResult | None = None,
        trajectory: TrajectoryResult | None = None,
        method: str = "auto",
        temperature_k: float = 300.0,
        bins: int = 64,
        min_depth_fraction: float = 0.05,
        force: bool = False,
    ) -> PinningResult:
        """Detect pinning sites as local minima of effective potential."""
        if (
            not force
            and self._last_pinning is not None
            and potential is None
            and trajectory is None
            and method == "auto"
            and abs(float(temperature_k) - 300.0) < 1e-15
            and int(bins) == 64
            and abs(float(min_depth_fraction) - 0.05) < 1e-15
        ):
            return self._last_pinning

        pot = (
            potential
            if potential is not None
            else self.potential(
                trajectory=trajectory,
                method=method,
                temperature_k=temperature_k,
                bins=bins,
            )
        )
        result = detect_pinning_sites(
            pot,
            min_depth_fraction=min_depth_fraction,
        )
        self._last_pinning = result
        return result

    @property
    def plt(self):
        """Convenience plotting namespace."""
        return EnergyPlotFacade(self)

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_COMPUTE,
            NODE_COLOR_PLOT,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        sections = [
            metrics_section_html(
                [
                    (
                        "dataset",
                        self._dataset_name or "auto-detect",
                        NODE_COLOR_COMPUTE,
                    ),
                    (
                        "slice",
                        "custom" if self._slice_info is not None else "full geometry",
                        None,
                    ),
                    ("strict missing", self._config.energy.strict_missing, None),
                    ("prefixes", ", ".join(self._config.energy.column_prefixes), None),
                ]
            ),
            accessors_section_html(
                [
                    (
                        "Energy:",
                        [
                            (".time_resolved(...)", NODE_COLOR_COMPUTE),
                            (".potential(method='auto')", NODE_COLOR_COMPUTE),
                            (".pinning(...)", NODE_COLOR_COMPUTE),
                        ],
                    ),
                    (
                        "Plotting:",
                        [
                            (".plt.time_resolved()", NODE_COLOR_PLOT),
                            (".plt.potential()", NODE_COLOR_PLOT),
                            (".plt.pinning()", NODE_COLOR_PLOT),
                        ],
                    ),
                ]
            ),
            examples_section_html(
                "etrace = jobs[-1].solitons.vortex.energy.time_resolved()\n"
                "pot = jobs[-1].solitons.vortex.energy.potential(method='auto')\n"
                "pin = jobs[-1].solitons.vortex.energy.pinning()\n"
                "jobs[-1].solitons.vortex.energy.plt.potential()",
                title="Energy Workflows",
            ),
        ]
        api = api_help_html(
            self,
            title="Vortex energy API help",
            prefix="jobs[-1].solitons.vortex.energy",
            properties=[("plt", "Convenience plotting namespace")],
            methods=["time_resolved", "potential", "pinning"],
            subtitle="Live public API for energy time series, effective potential, and pinning.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Energy Interface",
            icon="🪫",
            subtitle="Energy channels, effective radial potential, and pinning-site analysis.",
            sections=sections,
            api=api,
            uid=f"mmpp-vortex-energy-{str(_uuid.uuid4())[:8]}",
        )


class EnergyPlotFacade(InteractiveNodeMixin):
    """Plotting facade for :class:`EnergyInterface`."""

    _interactive_owner = "job[0].vortex.energy.plt"
    _interactive_nodes = frozenset({"time_resolved", "potential", "pinning"})

    def __init__(self, interface: EnergyInterface):
        self._interface = interface

    def time_resolved(self, **kwargs):
        """Compute and plot energy channels vs time."""
        result = self._interface.time_resolved()
        return result.plt.time_resolved(**kwargs)

    def potential(self, **kwargs):
        """Compute and plot effective potential."""
        result = self._interface.potential()
        return result.plt.potential(**kwargs)

    def pinning(self, **kwargs):
        """Compute and plot potential with detected pinning sites."""
        result = self._interface.pinning()
        return result.plt.potential_with_sites(**kwargs)

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, node_card_html, plot_accessor_html

        overview = plot_accessor_html(
            "EnergyPlotFacade",
            [
                (
                    ".time_resolved()",
                    "Compute + plot energy channels vs time",
                    "Delegates to EnergyTimeSeriesResult.plt.time_resolved().",
                ),
                (
                    ".potential()",
                    "Compute + plot effective potential",
                    "Delegates to EffectivePotentialResult.plt.potential().",
                ),
                (
                    ".pinning()",
                    "Compute + plot potential with pinning sites",
                    "Delegates to PinningResult.plt.potential_with_sites().",
                ),
            ],
        )
        api = api_help_html(
            self,
            title="Vortex energy plot API help",
            prefix="jobs[-1].solitons.vortex.energy.plt",
            methods=["time_resolved", "potential", "pinning"],
            subtitle="Plot helpers that compute the matching energy result when needed.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Energy Plot Accessor",
            icon="🎨",
            subtitle="Plot shortcuts for energy channels, effective potentials, and pinning maps.",
            sections=[overview],
            api=api,
            uid=f"mmpp-vortex-energy-plot-{str(_uuid.uuid4())[:8]}",
        )


__all__ = ["EnergyInterface"]
