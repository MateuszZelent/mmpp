"""Typed filter configuration models for FFT pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for time-domain filters applied before FFT."""

    remove_static: bool = False
    remove_mean: bool = True
    detrend: Literal["none", "linear", "quadratic"] = "none"
    window: str = "hann"

    # Advanced options
    high_pass_cutoff: float | None = None
    band_pass: tuple[float, float] | None = None
    savgol_window: int = 0
    savgol_polyorder: int = 3


@dataclass(frozen=True)
class PostprocessConfig:
    """Configuration for spectral-domain filters applied after FFT."""

    normalize: bool = False
    log_scale: bool = False
    gamma: float = 1.0
    percentile_clip: tuple[float, float] = (0.0, 100.0)
    soft_threshold: float = 0.0

    smooth: Literal["none", "gaussian", "savgol", "moving_average"] = "none"
    smooth_sigma: float = 1.0
    smooth_window: int = 7

    baseline: Literal["none", "mean", "median", "linear"] = "none"


@dataclass(frozen=True)
class FilterConfig:
    """Unified configuration for preprocessing and postprocessing filters."""

    pre: PreprocessConfig = field(default_factory=PreprocessConfig)
    post: PostprocessConfig = field(default_factory=PostprocessConfig)

    def with_pre(self, **kwargs) -> FilterConfig:
        """Return a new config with updated preprocessing settings."""
        return FilterConfig(pre=replace(self.pre, **kwargs), post=self.post)

    def with_post(self, **kwargs) -> FilterConfig:
        """Return a new config with updated postprocessing settings."""
        return FilterConfig(pre=self.pre, post=replace(self.post, **kwargs))
