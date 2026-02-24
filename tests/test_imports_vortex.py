from __future__ import annotations

import importlib

import numpy as np

from mmpp.solitons.vortex._shared.models import TrajectoryResult
from mmpp.solitons.vortex.model.thiele import cpp


def test_vortex_stage1_namespaces_import_cleanly():
    modules = [
        "mmpp.solitons.vortex._shared",
        "mmpp.solitons.vortex._shared.analysis",
        "mmpp.solitons.vortex.numerical",
        "mmpp.solitons.vortex.numerical.core",
        "mmpp.solitons.vortex.numerical.topology",
        "mmpp.solitons.vortex.numerical.snapshot",
        "mmpp.solitons.vortex.numerical.modes",
        "mmpp.solitons.vortex.numerical.events",
        "mmpp.solitons.vortex.numerical.nonlinear",
        "mmpp.solitons.vortex.numerical.signals",
        "mmpp.solitons.vortex.numerical.energy",
        "mmpp.solitons.vortex.model",
        "mmpp.solitons.vortex.model.thiele",
        "mmpp.solitons.vortex.bridge",
    ]
    for name in modules:
        module = importlib.import_module(name)
        assert module is not None


def test_trajectory_contract_has_analysis_compare_and_plot_accessors():
    t = np.linspace(0.0, 10e-9, 128)
    x = 3e-9 * np.cos(2.0 * np.pi * 1.2e9 * t)
    y = 2e-9 * np.sin(2.0 * np.pi * 1.2e9 * t)

    traj = TrajectoryResult(
        time=t,
        x=x,
        y=y,
        polarity=np.ones_like(t, dtype=int),
        method="synthetic",
        confidence=np.ones_like(t, dtype=float),
    )

    orbit = traj.analysis.orbit.fit(model="ellipse")
    freq = traj.analysis.phase.frequency(unit="hz")
    spec = traj.analysis.spectrum.directional(method="periodogram")
    cmp = traj.compare.with_(traj, label=("a", "b"))

    assert orbit.radius > 0.0
    assert freq.shape == t.shape
    assert spec.frequencies.ndim == 1
    assert cmp.metrics.delta_f_mean == 0.0
    assert hasattr(traj.plt, "orbit_2d")


def test_dataset_free_cpp_adapter_is_constructible():
    adapter = cpp()
    assert hasattr(adapter, "simulate")
