"""
Configuration classes and validation for mode visualization.

This module contains configuration settings and validation logic
for FMR mode analysis and visualization.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import matplotlib.pyplot as plt

from .compatibility import CMCRAMERI_AVAILABLE, CMOCEAN_AVAILABLE
from ...cli.logging_config import get_mmpp_logger

log = get_mmpp_logger(__name__)

if CMCRAMERI_AVAILABLE:
    import cmcrameri.cm as cmc

if CMOCEAN_AVAILABLE:
    import cmocean


@dataclass
class ModeVisualizationConfig:
    """Configuration for mode visualization."""

    # Figure settings
    figsize: tuple[float, float] = (16, 10)
    dpi: int = 100

    # Spectrum settings
    spectrum_log_scale: bool = False
    spectrum_normalize: bool = True
    peak_threshold: float = 0.1
    peak_min_distance: int = 5

    # Mode visualization settings
    show_magnitude: bool = True
    show_phase: bool = True
    show_combined: bool = True
    colormap_magnitude: str = "cmc.berlin"  # cmcrameri berlin for amplitude data
    colormap_phase: str = "cmc.romaO"  # cmcrameri romaO for phase data
    colormap_animation: str = (
        "balance"  # cmocean.cm.balance for animations, RdBu_r fallback
    )
    interpolation: str = "nearest"
    use_midpoint_norm: bool = False  # Use MidpointNormalize for diverging data
    animation_time_steps: int = 60  # Number of time steps for one full phase cycle

    # Publication-style annotations
    show_scalebar: bool = True
    scalebar_length_nm: Optional[float] = None  # Auto-computed when None
    scalebar_location: str = "lower right"
    scalebar_pad: float = 0.3
    scalebar_color: str = "white"
    scalebar_fontsize: int = 9
    scalebar_frame: bool = False
    scalebar_height_fraction: float = 0.01
    scale_units: str = "nm"

    colorbar_fraction: float = 0.04   # Proper colorbar width
    colorbar_pad: float = 0.01       # Small padding for close positioning
    colorbar_ticklabel_size: int = 9  # Larger tick labels
    colorbar_label_size: int = 10     # Larger labels
    colorbar_labels: dict[str, str] = field(
        default_factory=lambda: {
            "magnitude": "Magnetization |m|",
            "phase": "Phase (rad)", 
            "combined": "Re(m) × cos(φ)",
        }
    )

    # Frequency range for analysis
    f_min: float = 0.0
    f_max: float = 40.0

    # Layout settings
    spectrum_width_ratio: float = 0.4
    modes_width_ratio: float = 0.6

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.f_min >= self.f_max:
            raise ValueError(
                f"f_min ({self.f_min}) must be less than f_max ({self.f_max})"
            )

        if self.peak_threshold < 0 or self.peak_threshold > 1:
            raise ValueError(
                f"peak_threshold must be between 0 and 1, got {self.peak_threshold}"
            )

        if self.peak_min_distance < 1:
            raise ValueError(
                f"peak_min_distance must be >= 1, got {self.peak_min_distance}"
            )

        if self.spectrum_width_ratio <= 0 or self.modes_width_ratio <= 0:
            raise ValueError("Width ratios must be positive")

        if self.dpi < 50 or self.dpi > 500:
            log.warning(f"Unusual DPI value: {self.dpi}")

        # Validate colormaps
        try:
            self._resolve_colormap(self.colormap_magnitude)
            self._resolve_colormap(self.colormap_phase)
        except Exception as e:
            log.warning(f"Colormap validation failed: {e}")

        if self.scalebar_length_nm is not None and self.scalebar_length_nm <= 0:
            raise ValueError("scalebar_length_nm must be positive when provided")

        if self.scalebar_height_fraction <= 0 or self.scalebar_height_fraction > 0.1:
            raise ValueError(
                "scalebar_height_fraction should be within (0, 0.1] for sensible display"
            )

        if self.colorbar_fraction <= 0 or self.colorbar_pad < 0:
            raise ValueError("colorbar_fraction must be > 0 and colorbar_pad >= 0")

    def _resolve_colormap(self, cmap_name: str):
        """
        Resolve colormap from various sources (cmcrameri, cmocean, matplotlib).

        Parameters:
        -----------
        cmap_name : str
            Name of the colormap

        Returns:
        --------
        matplotlib colormap object
        """
        # Try cmcrameri first (scientific colormaps)
        if CMCRAMERI_AVAILABLE:
            try:
                return getattr(cmc, cmap_name)
            except AttributeError:
                pass

        # Try cmocean (oceanographic colormaps)
        if CMOCEAN_AVAILABLE:
            try:
                return getattr(cmocean.cm, cmap_name)
            except AttributeError:
                pass

        # Fallback to matplotlib
        return plt.get_cmap(cmap_name)

    def get_colormap(self, cmap_type: str):
        """
        Get resolved colormap for a specific type.
        
        Parameters:
        -----------
        cmap_type : str
            Type of colormap ('magnitude', 'phase', 'animation')
            
        Returns:
        --------
        matplotlib colormap object
        """
        if cmap_type == 'magnitude':
            return self._resolve_colormap(self.colormap_magnitude)
        elif cmap_type == 'phase':
            return self._resolve_colormap(self.colormap_phase)
        elif cmap_type == 'animation':
            return self._resolve_colormap(self.colormap_animation)
        else:
            raise ValueError(f"Unknown colormap type: {cmap_type}")

    def get_colorbar_label(self, view_type: str) -> str:
        """
        Get colorbar label for a specific view type.
        
        Parameters:
        -----------
        view_type : str
            Type of view ('magnitude', 'phase', 'combined')
            
        Returns:
        --------
        str
            Colorbar label
        """
        return self.colorbar_labels.get(view_type, view_type)

    def validate_frequency_range(self, frequencies: list[float]) -> bool:
        """
        Validate that frequencies are within configured range.
        
        Parameters:
        -----------
        frequencies : list[float]
            List of frequencies to validate
            
        Returns:
        --------
        bool
            True if all frequencies are within range
        """
        return all(self.f_min <= f <= self.f_max for f in frequencies)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'figsize': self.figsize,
            'dpi': self.dpi,
            'spectrum_log_scale': self.spectrum_log_scale,
            'spectrum_normalize': self.spectrum_normalize,
            'peak_threshold': self.peak_threshold,
            'peak_min_distance': self.peak_min_distance,
            'show_magnitude': self.show_magnitude,
            'show_phase': self.show_phase,
            'show_combined': self.show_combined,
            'colormap_magnitude': self.colormap_magnitude,
            'colormap_phase': self.colormap_phase,
            'colormap_animation': self.colormap_animation,
            'interpolation': self.interpolation,
            'use_midpoint_norm': self.use_midpoint_norm,
            'animation_time_steps': self.animation_time_steps,
            'show_scalebar': self.show_scalebar,
            'scalebar_length_nm': self.scalebar_length_nm,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'spectrum_width_ratio': self.spectrum_width_ratio,
            'modes_width_ratio': self.modes_width_ratio
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> 'ModeVisualizationConfig':
        """Create configuration from dictionary."""
        # Filter only known parameters
        known_params = {
            key: value for key, value in config_dict.items()
            if hasattr(cls, key)
        }
        return cls(**known_params)


def create_default_config() -> ModeVisualizationConfig:
    """Create default visualization configuration."""
    return ModeVisualizationConfig()


def create_high_dpi_config() -> ModeVisualizationConfig:
    """Create high-DPI configuration for publication figures."""
    config = ModeVisualizationConfig()
    config.dpi = 300
    config.figsize = (12, 8)
    config.colorbar_ticklabel_size = 12
    config.colorbar_label_size = 14
    config.scalebar_fontsize = 12
    return config


def create_animation_config() -> ModeVisualizationConfig:
    """Create configuration optimized for animations."""
    config = ModeVisualizationConfig()
    config.dpi = 150  # Balance between quality and performance
    config.show_scalebar = False  # Reduce clutter in animations
    config.animation_time_steps = 120  # Smoother animations
    return config