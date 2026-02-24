"""Shared plotting accessors for vortex trajectory results."""

from .interactive import trajectory_interactive
from .static import TrajectoryPlotAccessor

__all__ = ["TrajectoryPlotAccessor", "trajectory_interactive"]
