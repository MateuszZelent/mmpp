from __future__ import annotations

import numpy as np

from mmpp.solitons.vortex._shared.models import TrajectoryResult
from mmpp.solitons.vortex.bridge import BridgeInterface
from mmpp.solitons.vortex.config import VortexConfig
from mmpp.solitons.vortex.events.models import (
    CoreExpulsionEvent,
    DwellTimeResult,
    PolaritySwitchEvent,
    StateSwitchEvent,
)
from mmpp.solitons.vortex.events.interface import EventsInterface
from mmpp.solitons.vortex.model.interface import VortexModelInterface
from mmpp.solitons.vortex.modes.models import VortexModeResult
from mmpp.solitons.vortex.numerical.topology.interface import TopologyInterface
from mmpp.solitons.vortex.numerical.topology.models import TopologyResult


def test_stage2_result_html_helpers_smoke():
    t = np.linspace(0.0, 2e-9, 16)
    traj = TrajectoryResult(
        time=t,
        x=5e-9 * np.cos(2.0 * np.pi * 0.6e9 * t),
        y=5e-9 * np.sin(2.0 * np.pi * 0.6e9 * t),
        polarity=np.ones_like(t, dtype=int),
        method="synthetic",
        confidence=np.ones_like(t, dtype=float),
    )
    html_traj = traj._repr_html_()
    assert "TrajectoryResult" in html_traj
    assert "<div" in html_traj

    topo = TopologyResult(
        polarity=1,
        vorticity=1,
        chirality=1,
        Q=0.49,
        core_position=(2.5e-9, 3.0e-9),
        topological_density=np.zeros((8, 8), dtype=float),
        state="vortex",
        method="finite_diff",
        confidence=0.92,
        chirality_confidence=0.88,
        convention="down",
    )
    html_topo = topo._repr_html_()
    assert "TopologyResult" in html_topo
    assert "is_consistent" in html_topo

    mode = VortexModeResult(
        m_index=1,
        n_index=0,
        mode_type="gyrotropic",
        rotation_sense="CCW",
        confidence=0.95,
        frequency_hz=0.82e9,
        power=1.23,
    )
    html_mode = mode._repr_html_()
    assert "VortexModeResult" in html_mode
    assert "gyrotropic" in html_mode


def test_stage3_event_and_interface_html_helpers_smoke():
    switch = PolaritySwitchEvent(
        time=2.5e-9,
        index=25,
        from_p=1,
        to_p=-1,
        confidence=0.9,
    )
    assert "PolaritySwitchEvent" in switch._repr_html_()

    state = StateSwitchEvent(
        time=3.0e-9,
        index=30,
        from_state="G-state",
        to_state="C-state",
        confidence=0.85,
    )
    assert "StateSwitchEvent" in state._repr_html_()

    expulsion = CoreExpulsionEvent(
        time=4.0e-9,
        index=40,
        radius=42e-9,
        threshold=40e-9,
        confidence=0.8,
        duration=0.5e-9,
    )
    assert "CoreExpulsionEvent" in expulsion._repr_html_()

    dwell = DwellTimeResult(state="G-state", dwell_times=np.array([1e-9, 2e-9, 3e-9]))
    html_dwell = dwell._repr_html_()
    assert "DwellTimeResult" in html_dwell
    assert "dwell_histogram" in html_dwell

    topology_interface = TopologyInterface(
        job_result=object(),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
    )
    assert "Topology Interface" in topology_interface._repr_html_()

    model_interface = VortexModelInterface(job_result=object(), dataset_name="m")
    model_html = model_interface._repr_html_()
    assert "Vortex Model Interface" in model_html
    assert "Vortex model API help" in model_html
    assert ">Overview</button>" in model_html
    assert ">API</button>" in model_html

    bridge_interface = BridgeInterface()
    bridge_html = bridge_interface._repr_html_()
    assert "Vortex Bridge Interface" in bridge_html
    assert "Vortex bridge API help" in bridge_html
    assert ">Overview</button>" in bridge_html
    assert ">API</button>" in bridge_html


def test_events_interface_and_plot_repr_use_tabs():
    class _Core:
        def track(self):
            t = np.linspace(0.0, 1e-9, 8)
            return TrajectoryResult(
                time=t,
                x=np.zeros_like(t),
                y=np.zeros_like(t),
                polarity=np.ones_like(t),
                method="synthetic",
                confidence=np.ones_like(t),
            )

    interface = EventsInterface(
        job_result=object(),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
        core_interface=_Core(),
        trajectory_interface=object(),
    )
    html = interface._repr_html_()
    plot_html = interface.plt._repr_html_()

    assert "Vortex events API help" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "Vortex events plot API help" in plot_html
    assert ">Overview</button>" in plot_html
    assert ">API</button>" in plot_html
