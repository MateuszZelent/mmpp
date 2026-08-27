# ruff: noqa: UP007
"""Configuration models for skyrmion topology and size analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .._coordinates import XYConvention


@dataclass
class SkyrmionTopologyConfig:
    """Numerical settings used to estimate topological charge and its centre."""

    method: str = "berg_luscher"
    min_abs_q: float = 0.5
    min_contrast: float = 0.15
    q_threshold_fraction: float = 0.05
    convention: XYConvention = field(default_factory=XYConvention)

    def __post_init__(self) -> None:
        method = str(self.method).lower()
        if method not in {"berg_luscher", "finite_diff"}:
            raise ValueError("method must be 'berg_luscher' or 'finite_diff'")
        self.method = method
        if not 0.0 <= float(self.min_abs_q) <= 1.5:
            raise ValueError("min_abs_q must be between 0 and 1.5")
        if float(self.min_contrast) <= 0.0:
            raise ValueError("min_contrast must be positive")
        if not 0.0 <= float(self.q_threshold_fraction) <= 1.0:
            raise ValueError("q_threshold_fraction must be between 0 and 1")


@dataclass
class SizeFitConfig:
    """Settings for radial profiles and size fits."""

    method: str = "auto"
    radial_bin_m: Optional[float] = None
    min_angular_coverage: float = 0.7
    min_profile_bins: int = 8
    edge_fraction: float = 0.9
    min_contrast: float = 0.15
    max_normalized_rmse: float = 0.2

    def __post_init__(self) -> None:
        method = str(self.method).lower()
        allowed = {"auto", "domain_wall", "ansatz", "gaussian", "threshold"}
        if method not in allowed:
            raise ValueError(
                "method must be 'auto', 'domain_wall', 'ansatz', "
                "'gaussian', or 'threshold'"
            )
        self.method = method
        if self.radial_bin_m is not None and float(self.radial_bin_m) <= 0.0:
            raise ValueError("radial_bin_m must be positive when provided")
        if int(self.min_profile_bins) < 4:
            raise ValueError("min_profile_bins must be at least 4")
        if not 0.0 < float(self.min_angular_coverage) <= 1.0:
            raise ValueError("min_angular_coverage must be in (0, 1]")
        if not 0.5 <= float(self.edge_fraction) <= 1.0:
            raise ValueError("edge_fraction must be between 0.5 and 1")
        if float(self.min_contrast) <= 0.0:
            raise ValueError("min_contrast must be positive")
        if float(self.max_normalized_rmse) <= 0.0:
            raise ValueError("max_normalized_rmse must be positive")


@dataclass
class SkyrmionConfig:
    """Top-level mutable configuration for skyrmion analysis."""

    topology: SkyrmionTopologyConfig = field(default_factory=SkyrmionTopologyConfig)
    size: SizeFitConfig = field(default_factory=SizeFitConfig)
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = ["SkyrmionConfig", "SkyrmionTopologyConfig", "SizeFitConfig"]
