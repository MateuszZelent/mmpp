"""Interactive dispersion widget internals.

Public callers should use ``result.plot.interactive()`` or
``m.fft.dispersion.plot.interactive()``. This package keeps notebook widget
construction separate from the lightweight public controller.
"""

from .widget import DispersionHeatmapWidget

__all__ = ["DispersionHeatmapWidget"]
