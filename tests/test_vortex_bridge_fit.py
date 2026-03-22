from __future__ import annotations

import numpy as np
import pytest

from mmpp.solitons.vortex._shared.models import TrajectoryResult
from mmpp.solitons.vortex.bridge import BridgeInterface


def test_bridge_fit_thiele_from_trajectory_returns_simulated_result():
    t = np.linspace(0.0, 40e-9, 2000)
    omega = 2.0 * np.pi * 0.85e9
    radius = 5.0e-9
    x = radius * np.cos(omega * t)
    y = radius * np.sin(omega * t)

    traj = TrajectoryResult(
        time=t,
        x=x,
        y=y,
        polarity=np.ones_like(t, dtype=int),
        method="synthetic",
        confidence=np.ones_like(t, dtype=float),
    )

    bridge = BridgeInterface()
    fit = bridge.fit.thiele_from_trajectory(traj, damping=0.05)

    assert np.isfinite(fit.omega0_rad_s)
    assert fit.radius_m > 0.0
    assert fit.simulated_trajectory.time.shape == traj.time.shape
    assert fit.simulated_trajectory.method == "thiele_fit_proxy"
    assert "ThieleTrajectoryFitResult" in fit._repr_html_()
    assert "Vortex Bridge Interface" in bridge._repr_html_()

    cmp = traj.compare.with_(fit.simulated_trajectory, label=("num", "fit"))
    assert np.isfinite(cmp.metrics.delta_f_mean)


def test_bridge_fit_thiele_from_damped_trajectory_uses_initial_amplitude():
    t = np.linspace(0.0, 40e-9, 2000)
    omega = 2.0 * np.pi * 0.85e9
    radius0 = 8.0e-9
    radius1 = 2.0e-9
    tau = t[-1] / max(np.log(radius0 / radius1), 1e-12)
    radius = radius0 * np.exp(-t / tau)
    x = radius * np.cos(omega * t)
    y = radius * np.sin(omega * t)

    traj = TrajectoryResult(
        time=t,
        x=x,
        y=y,
        polarity=np.ones_like(t, dtype=int),
        method="synthetic_damped",
        confidence=np.ones_like(t, dtype=float),
    )

    bridge = BridgeInterface()
    fit = bridge.fit.thiele_from_trajectory(traj, damping=1.0)

    sim_x = np.asarray(fit.simulated_trajectory.x, dtype=float)
    sim_y = np.asarray(fit.simulated_trajectory.y, dtype=float)
    sim_r0 = float(np.hypot(sim_x[0] - np.mean(sim_x), sim_y[0] - np.mean(sim_y)))

    assert fit.radius_m == pytest.approx(radius0, rel=0.15)
    assert sim_r0 == pytest.approx(radius0, rel=0.15)
    assert fit.metadata["radius_initial_m"] == pytest.approx(radius0, rel=1e-3)
