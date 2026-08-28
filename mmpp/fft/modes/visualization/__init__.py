"""
FMR Mode Visualization

Static plots, interactive spectrum, and animations.
All visualization code organized here for maintainability.
"""

from .animation import (
    save_animated_view,
    save_modes_animation,
    start_column_animation,
    start_mode_animation,
    stop_column_animation,
    stop_mode_animation,
    toggle_mode_animation,
)
from .interactive import interactive_spectrum
from .static_plots import (
    add_scale_bar,
    plot_modes,
    update_single_mode_plot,
)

__all__ = [
    # Animation functions
    "save_modes_animation",
    "toggle_mode_animation",
    "stop_mode_animation",
    "save_animated_view",
    "start_mode_animation",
    "start_column_animation",
    "stop_column_animation",
    # Static plot functions
    "plot_modes",
    "update_single_mode_plot",
    "add_scale_bar",
    # Interactive
    "interactive_spectrum",
]
