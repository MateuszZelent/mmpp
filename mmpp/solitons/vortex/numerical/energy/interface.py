"""High-level interface for vortex energy analysis."""

from __future__ import annotations

import warnings
from typing import Any

from ...config import VortexConfig
from ..._shared.models import TrajectoryResult
from mmpp._shared.repr_html import make_simple_card
from .models import EnergyTimeSeriesResult, EffectivePotentialResult, PinningResult
from .pinning import detect_pinning_sites
from .potential import potential_from_boltzmann, potential_from_energy_channel
from .time_resolved import extract_energy_time_series


class EnergyInterface:
    """Energy namespace with table-driven time-resolved channels."""

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
        if not force and self._last_result is not None and columns is None and strict is None:
            return self._last_result

        strict_mode = (
            bool(self._config.energy.strict_missing)
            if strict is None
            else bool(strict)
        )
        result = extract_energy_time_series(
            self._job,
            columns=columns,
            prefixes=tuple(self._config.energy.column_prefixes),
        )

        if strict_mode and (not result.channels):
            available = result.metadata.get("available_columns", [])
            raise ValueError(
                "No energy channels found in table. "
                f"Available columns: {available}"
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

        pot = potential if potential is not None else self.potential(
            trajectory=trajectory,
            method=method,
            temperature_k=temperature_k,
            bins=bins,
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
        methods = [
            (".time_resolved(...)", "Load table energy channels E(t)"),
            (".potential(method='auto')", "Reconstruct effective W(r)"),
            (".pinning(...)", "Detect pinning minima in W(r)"),
            (".plt.time_resolved()", "Plot E(t) channels"),
            (".plt.potential()", "Plot effective potential"),
            (".plt.pinning()", "Plot potential with pinning sites"),
        ]
        return make_simple_card(
            title="Vortex Energy Interface",
            subtitle="Energy channels, effective potential and pinning analysis",
            rows=methods,
        )


class EnergyPlotFacade:
    """Plotting facade for :class:`EnergyInterface`."""

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


__all__ = ["EnergyInterface"]
