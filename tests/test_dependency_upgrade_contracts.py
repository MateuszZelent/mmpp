"""Regression checks for the updated optional dependency stack."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import numpy as np
import pytest

from mmpp.fft.dispersion.core import SpinWaveAnalyzer
from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D


def test_branch_tracking_without_frequency_window():
    config = DispersionConfig(dk_max=0.1, df_max=None)
    analyzer = SpinWaveAnalyzer.__new__(SpinWaveAnalyzer)
    analyzer.config = config
    result = DispersionResult1D(
        S=np.array([[0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 7]]),
        k_axis=np.array([0.0, 1.0, 2.0]),
        f_axis=np.array([0.0, 1.0, 2.0, 3.0]),
        axis="x",
        component="mz",
        config=config,
    )

    branch = analyzer.track_branch(result, np.array([0.0, 1.0, 2.0]))

    np.testing.assert_array_equal(branch.f_values, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(branch.amplitudes, [5, 6, 7])


def test_numba_fallback_decorators_remain_usable():
    code = """
import sys
sys.modules['numba'] = None
from mmpp.solitons import _topology
from mmpp.solitons.vortex.autofit import _numba_kernels
from mmpp.analytical.nonlinear_stno import engine
for module in (_topology, _numba_kernels, engine):
    assert not module.HAS_NUMBA
    assert module.njit(lambda x: x + 1)(2) == 3
    assert module.njit(fastmath=True)(lambda x: x + 1)(2) == 3
assert list(engine.prange(3)) == [0, 1, 2]
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_textual_quit_action_still_exits():
    pytest.importorskip("textual")
    from mmpp.tui import MMPPApp

    async def exercise():
        app = MMPPApp()
        async with app.run_test() as pilot:
            await pilot.press("ctrl+q")
            await pilot.pause()
        assert not app.is_running

    asyncio.run(exercise())
