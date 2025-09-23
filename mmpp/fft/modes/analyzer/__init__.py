"""
Analyzer sub-package for FMR mode analysis.

This package contains the core analysis components split into
focused modules for better maintainability.
"""

# Import all analyzer components
from .data_access import DataAccessMixin
from .analysis import AnalysisMixin
from .visualization import VisualizationMixin
from .interactive import InteractiveMixin
from .animation import AnimationMixin
from .compute import ComputeMixin

# Main analyzer class combining all mixins
from .analyzer import FMRModeAnalyzer

__all__ = [
    'DataAccessMixin',
    'AnalysisMixin', 
    'VisualizationMixin',
    'InteractiveMixin',
    'AnimationMixin',
    'ComputeMixin',
    'FMRModeAnalyzer'
]