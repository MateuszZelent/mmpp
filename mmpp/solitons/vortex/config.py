"""Configuration models for vortex analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from .._base import SolitonConfig
from ._utils import XYConvention


@dataclass
class TrackingConfig:
    """Configuration for vortex core tracking."""

    method: str = "auto"
    z_layer: int = -1
    core_threshold: float = 0.9
    gaussian_roi: int = 7
    convention: XYConvention = field(default_factory=XYConvention)
    polarity_threshold_up: float = 0.3
    polarity_threshold_down: float = -0.3
    polarity_roi_pixels: int = 1


@dataclass
class TopologyConfig:
    """Configuration for vortex topology detection."""

    method: str = "finite_diff"
    z_layer: int = -1
    polarity_threshold: float = 0.5
    chirality_ring_r: tuple[float, float] | None = None
    convention: XYConvention = field(default_factory=XYConvention)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory post-processing."""

    filter_method: str = "savgol"
    filter_window: int = 11
    steady_state_threshold: float = 0.05
    steady_state_window: int = 31


@dataclass
class SpectrumConfig:
    """Configuration for vortex spectrum analysis."""

    method: str = "welch"
    nperseg: int | None = None
    noverlap: int | None = None


@dataclass
class ModesConfig:
    """Configuration for vortex mode classification heuristics."""

    max_modes: int = 6
    min_prominence: float = 0.05
    tracking_method: str = "auto"


@dataclass
class NonlinearConfig:
    """Configuration for nonlinear STNO parameter extraction."""

    phase_method: str = "complex"
    spectrum_method: str = "welch"
    steady_state_fraction: float = 0.4
    reference_radius: float | None = None


@dataclass
class SignalsConfig:
    """Configuration for synthetic electrical-signal extraction."""

    tracking_method: str = "centroid"
    resistance_parallel_ohm: float = 100.0
    delta_resistance_ohm: float = 40.0
    spectrum_method: str = "welch"


@dataclass
class EnergyConfig:
    """Configuration for energy time-series extraction from table data."""

    column_prefixes: tuple[str, ...] = ("E_", "energy", "W_")
    strict_missing: bool = False


@dataclass
class VortexConfig(SolitonConfig):
    """Top-level mutable configuration for vortex analysis."""

    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    modes: ModesConfig = field(default_factory=ModesConfig)
    nonlinear: NonlinearConfig = field(default_factory=NonlinearConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
