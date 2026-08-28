from __future__ import annotations

import numpy as np

from mmpp.solitons.vortex import XYConvention, grid_xy


def test_grid_xy_shapes_and_y_axis_sign():
    x_up, y_up = grid_xy(7, 5, 1.5e-9, 2.0e-9, convention=XYConvention(y_axis="up"))
    x_down, y_down = grid_xy(
        7, 5, 1.5e-9, 2.0e-9, convention=XYConvention(y_axis="down")
    )

    assert x_up.shape == (5, 7)
    assert y_up.shape == (5, 7)
    assert x_down.shape == (5, 7)
    assert y_down.shape == (5, 7)

    assert np.allclose(x_up, x_down)
    assert np.allclose(y_up, -y_down)
