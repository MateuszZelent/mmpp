"""Configuration objects for hysteresis analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HysteresisConfig:
    """Central configuration for hysteresis workflows."""

    # Layout / visualization
    figsize: tuple[float, float] = (14.0, 6.0)
    dpi: int = 120
    snapshot_component: str = "snapshot"
    z_layer: int | str = 0

    # Metrics / interpolation
    saturation_threshold: float = 0.02
    saturation_window: int = 5
    interpolation_order: int = 1

    # Filtering
    filter_method: str | None = None
    savgol_window: int = 11
    savgol_order: int = 3
    auto_filter: bool = True

    # Loop styling
    colormap_magnitude: str = "viridis"
    branch_colors: tuple[str, str] = ("#2196F3", "#F44336")
    marker_size: float = 10.0
    arrow_length: float = 0.08
    show_hc: bool = True
    show_mr: bool = True
    show_ms: bool = False
    show_arrow: bool = True
    show_branch_colors: bool = True

    # Interactive animation state
    animation_fps: int = 20
    trail_length: int = 40
    trail_alpha_decay: float = 0.85

    # Defaults for source extraction
    default_m_dataset: str = "m"
    default_component: str = "x"

    # Metadata / export placeholders (kept for forward compatibility)
    default_export_format: str = "csv"
    figure_export_format: str = "svg"

    # Future uncertainty knobs (not used in v1 calculations)
    bootstrap_n_samples: int = 1000
    bootstrap_ci: float = 0.95
