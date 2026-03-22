from __future__ import annotations

import importlib
from dataclasses import dataclass

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


@dataclass
class _MaterialLikeWithCppExtras:
    Ms: float = 8.0e5
    alpha: float = 0.01
    P: float = 0.5
    A: float = 1.3e-11
    Lambda: float = 1.4
    epsilonprime: float = 0.2
    FixedLayer: tuple[float, float, float] = (0.0, 0.0, -1.0)
    FixedLayerPosition: str = "bottom"
    L_stt: float = 9e-9


def test_cpp_adapter_reads_slonczewski_extras_from_attribute_object():
    adapter = cpp(
        material=_MaterialLikeWithCppExtras(),
        geom={"R": 45e-9, "L": 20e-9},
        polarity=1,
    )

    assert adapter._metadata["Lambda"] == 1.4
    assert adapter._metadata["epsilonprime"] == 0.2
    assert adapter._metadata["p_z"] == -1.0
    assert adapter._metadata["fixed_layer_position"] == "bottom"
    assert adapter.model.material.Ms == 8.0e5
