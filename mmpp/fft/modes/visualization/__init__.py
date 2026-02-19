"""
FMR Mode Visualization

Static plots, interactive spectrum, and animations.
All visualization code organized here for maintainability.
"""

from .animation import (
    save_modes_animation,
    toggle_mode_animation,
    stop_mode_animation,
    save_animated_view,
    start_mode_animation,
    start_column_animation,
    stop_column_animation,
)

from .static_plots import (
    plot_modes,
    update_single_mode_plot,
    add_scale_bar,
)
from .interactive import interactive_spectrum

__all__ = [
    # Animation functions
    'save_modes_animation',
    'toggle_mode_animation',
    'stop_mode_animation',
    'save_animated_view',
    'start_mode_animation',
    'start_column_animation',
    'stop_column_animation',
    # Static plot functions
    'plot_modes',
    'update_single_mode_plot',
    'add_scale_bar',
    # Interactive
    'interactive_spectrum',
]

