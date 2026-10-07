"""Focused regressions for autofit numerical contracts."""

from __future__ import annotations

import numpy as np
import pytest

from mmpp.solitons.vortex._shared.models import TrajectoryResult
from mmpp.solitons.vortex.autofit.config import AutofitConfig, ParameterSpec
from mmpp.solitons.vortex.autofit.optimizers import run_optimization
from mmpp.solitons.vortex.trajectory.operations import _trajectory_dt


def _trajectory(time: list[float]) -> TrajectoryResult:
    values = np.asarray(time, dtype=float)
    return TrajectoryResult(
        time=values,
        x=np.zeros_like(values),
        y=np.zeros_like(values),
        polarity=np.ones(values.size, dtype=int),
        method="test",
        confidence=np.ones_like(values),
    )


def test_sampling_interval_requires_real_valid_timestamps():
    with pytest.raises(ValueError, match="At least two trajectory timestamps"):
        _trajectory_dt(_trajectory([0.0]))

    with pytest.raises(ValueError, match="strictly increasing"):
        _trajectory_dt(_trajectory([0.0, 1.0, 0.5]))


def test_scaled_optimizer_reports_active_bounds_in_physical_units():
    config = AutofitConfig(
        global_search=False,
        local_maxiter=20,
        max_eval=40,
        verbose=False,
        fit_params=("omega0",),
        param_specs={
            "omega0": ParameterSpec(
                lower=0.0,
                upper=2e9,
                initial=1e9,
                scale=1e9,
            )
        },
    )

    best, diagnostics = run_optimization(
        lambda values: (float(values["omega0"] - 3e9) ** 2, {}),
        param_names=["omega0"],
        param_specs=config.get_param_specs(),
        initial_values={"omega0": 1e9},
        config=config,
    )

    assert best["omega0"] == pytest.approx(2e9, rel=1e-6)
    assert diagnostics.active_bounds["omega0"] == "upper"
