"""Coordinate conventions shared by soliton analysis modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class XYConvention:
    """Mapping from array rows/cols to physical XY coordinates.

    ``y_axis='up'`` means row 0 corresponds to maximal physical y.
    ``y_axis='down'`` means row 0 corresponds to minimal physical y (image style).
    """

    y_axis: Literal["up", "down"] = "up"


def grid_xy(
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    *,
    convention: XYConvention = XYConvention(),
) -> tuple[np.ndarray, np.ndarray]:
    """Create physical coordinate meshgrids from array dimensions."""
    cx = (int(Nx) - 1) / 2.0
    cy = (int(Ny) - 1) / 2.0

    j = np.arange(int(Nx), dtype=float)
    i = np.arange(int(Ny), dtype=float)

    x_1d = (j - cx) * float(dx)
    if convention.y_axis == "up":
        y_1d = (int(Ny) - 1 - i - cy) * float(dy)
    else:
        y_1d = (i - cy) * float(dy)

    x_grid, y_grid = np.meshgrid(x_1d, y_1d)
    return np.asarray(x_grid, dtype=float), np.asarray(y_grid, dtype=float)


__all__ = ["XYConvention", "grid_xy"]
