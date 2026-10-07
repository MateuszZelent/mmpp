"""High-level interface for vortex energy analysis."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

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
        selected_times = None
        if self._core is not None:
            try:
                selected_times = self._core._resolve_time_axis()
            except (AttributeError, TypeError, ValueError, IndexError):
                selected_times = None
        dataset_view = getattr(self._core, "_dataset_view", None)
        index_plan = getattr(dataset_view, "_index_plan", None)
        source_time_size = (
            int(index_plan.source_shape[0]) if index_plan is not None else None
        )
        time_selection = (
            self._slice_info[0]
            if isinstance(self._slice_info, tuple) and self._slice_info
            else None
        )
        result = extract_energy_time_series(
            self._job,
            columns=columns,
            prefixes=tuple(self._config.energy.column_prefixes),
            selected_times=selected_times,
            time_selection=time_selection,
            source_time_size=source_time_size,
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
        temperature_k: float | None = None,
        bins: int = 64,
        assume_equilibrium: bool = False,
        force: bool = False,
    ) -> EffectivePotentialResult:
        """Estimate a radial energy profile or explicitly assumed equilibrium PMF."""
        if (
            not force
            and self._last_potential is not None
            and trajectory is None
            and method == "auto"
            and temperature_k is None
            and not assume_equilibrium
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
        if method_norm not in {"auto", "boltzmann", "radial_pmf", "energy_bin"}:
            raise ValueError(
                "method must be 'auto', 'energy_bin', 'radial_pmf', or 'boltzmann'"
            )

        table_energy = None
        try:
            table_energy = self.time_resolved(force=False)
        except ValueError as exc:
            if method_norm == "energy_bin":
                raise
            warnings.warn(
                "Energy channels could not be aligned to the selected trajectory "
                f"time axis ({exc}); falling back to trajectory statistics.",
                RuntimeWarning,
                stacklevel=2,
            )
        has_e_total = table_energy is not None and "E_total" in table_energy.channels
        can_energy_bin = bool(
            has_e_total
            and table_energy is not None
            and table_energy.time.size == traj.time.size
            and np.allclose(table_energy.time, traj.time, rtol=1e-9, atol=1e-15)
        )

        if method_norm == "energy_bin" and not can_energy_bin:
            raise ValueError(
                "method='energy_bin' requires E_total channel aligned with trajectory time samples."
            )

        if method_norm == "energy_bin" or (method_norm == "auto" and can_energy_bin):
            assert table_energy is not None
            result = potential_from_energy_channel(
                traj,
                table_energy.channels["E_total"],
                bins=bins,
            )
        elif method_norm in {"boltzmann", "radial_pmf"}:
            if temperature_k is None:
                raise ValueError(
                    "Radial Boltzmann inversion requires an explicit temperature_k"
                )
            result = potential_from_boltzmann(
                traj,
                temperature_k=temperature_k,
                bins=bins,
                assume_equilibrium=assume_equilibrium,
            )
        else:
            raise ValueError(
                "No energy channel is aligned with this trajectory. Automatic "
                "radial Boltzmann inversion is disabled; request "
                "method='radial_pmf', set assume_equilibrium=True, and provide "
                "the physical temperature explicitly."
            )

        self._last_potential = result
        return result

    def pinning(
        self,
        *,
        potential: EffectivePotentialResult | None = None,
        trajectory: TrajectoryResult | None = None,
        method: str = "auto",
        temperature_k: float | None = None,
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
            and temperature_k is None
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
