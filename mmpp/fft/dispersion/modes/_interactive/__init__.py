"""
Internal submodules for InteractiveDispersionModes.

This package contains separated concerns for better code organization:
- widgets: ipywidgets creation & layout
- plotting: Matplotlib plot updates
- mode_extraction: Spatial mode extraction algorithms
"""

from .widgets import WidgetBuilder
from .plotting import InteractivePlotter  
from .mode_extraction import ModeExtractor

__all__ = [
    "WidgetBuilder",
    "InteractivePlotter",
    "ModeExtractor",
]
