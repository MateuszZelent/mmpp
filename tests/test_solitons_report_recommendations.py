from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np

from mmpp.batch_operations import BatchOperations
from mmpp.solitons.batch import (
    BatchSolitonsInterface,
    BatchVortexInterface,
    BatchVortexPhaseDiagramResult,
    BatchVortexSpectrumMapResult,
)
from mmpp.solitons.vortex.config import SpectrumConfig, VortexConfig
from mmpp.solitons.vortex.core.models import TrajectoryResult
from mmpp.solitons.vortex.nonlinear.models import (
    AmplitudeEquationResult,
    STBatchResult,
    STParametersResult,
    ThieleForceBalanceResult,
)
from mmpp.solitons.vortex.spectrum.interface import VortexSpectrumInterface
from mmpp.solitons.vortex.spectrum.models import VortexSpectrumResult
from mmpp.solitons.vortex.topology.models import TopologyResult
from mmpp.solitons.vortex.trajectory.models import OrbitFitResult


def _trajectory(freq_hz: float = 2.0e9) -> TrajectoryResult:
    time = np.linspace(0.0, 20e-9, 256)
    phase = 2.0 * np.pi * freq_hz * time
    return TrajectoryResult(
        time=time,
        x=np.cos(phase) * 5e-9,
        y=np.sin(phase) * 5e-9,
        polarity=np.ones_like(time),
        confidence=np.ones_like(time),
        method="synthetic",
    )


class _FakeSpectrumNamespace:
    def __init__(self, result: VortexSpectrumResult):
        self._result = result
        self.calls = 0

    def gyration(self, **kwargs) -> VortexSpectrumResult:
        self.calls += 1
        return self._result

    def breathing(self, **kwargs) -> VortexSpectrumResult:
        self.calls += 1
        return self._result


class _FakeTrajectoryNamespace:
    def __init__(self, trajectory: TrajectoryResult):
        self.raw = trajectory

    def steady_state(self) -> TrajectoryResult:
        return self.raw


class _FakeEventsNamespace:
    def polarity_switches(self, **kwargs) -> list:
        return []

    def state_switches(self, **kwargs) -> list:
        return []

    def core_expulsions(self, **kwargs) -> list:
        return []


class _FakeVortexNamespace:
    def __init__(self, trajectory: TrajectoryResult, spectrum: VortexSpectrumResult):
        self.trajectory = _FakeTrajectoryNamespace(trajectory)
        self.spectrum = _FakeSpectrumNamespace(spectrum)
        self.events = _FakeEventsNamespace()


class _FakeSolitonsNamespace:
    def __init__(self, vortex: _FakeVortexNamespace):
        self.vortex = vortex


class _FakeBatchResult:
    def __init__(self, index: int):
        self.path = f"run-{index}.zarr"
        self.attrs = {"i_pillar_ma": float(index), "D": 200e-9}
        trajectory = _trajectory(freq_hz=(index + 1) * 1.0e9)
        spectrum = VortexSpectrumResult(
            frequencies=np.array([0.0, 1.0e9, 2.0e9]),
            power=np.array([0.0, float(index + 1), 0.0]),
            method="periodogram",
        )
        self.solitons = _FakeSolitonsNamespace(
            _FakeVortexNamespace(trajectory, spectrum)
        )


class _FakeTableArray:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)
        self.shape = self._values.shape

    def __getitem__(self, key):
        return self._values[key]


class _FakeTableGroup:
    def __init__(self, columns: dict[str, np.ndarray]):
        self._columns = {
            name: _FakeTableArray(values) for name, values in columns.items()
        }

    def keys(self):
        return self._columns.keys()

    def __getitem__(self, key):
        return self._columns[key]


class _FakeBatchResultWithTable(_FakeBatchResult):
    def __init__(self, index: int, columns: dict[str, np.ndarray]):
        super().__init__(index)
        self._table = _FakeTableGroup(columns)

    def __contains__(self, key):
        return key == "table"

    def __getitem__(self, key):
        if key == "table":
            return self._table
        raise KeyError(key)


class _FakeRawArray:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data, dtype=float)
        self.shape = self._data.shape

    def __getitem__(self, key):
        return self._data[key]


class _FakeBatchResultWithMagnetization(_FakeBatchResult):
    def __init__(self, index: int, data: np.ndarray, *, dt: float):
        super().__init__(index)
        self.attrs["t_sampl"] = dt
        self._raw = _FakeRawArray(data)

    def get_largest_m_dataset(self):
        return "m"

    def get_raw(self, dataset_name):
        if dataset_name != "m":
            raise NameError(dataset_name)
        return self._raw


class _CountingCore:
    def __init__(self) -> None:
        self.calls = 0

    def track(self) -> TrajectoryResult:
        self.calls += 1
        return _trajectory(freq_hz=(self.calls + 1) * 1.0e9)


def test_shared_spectral_utility_is_used_by_gyration(monkeypatch) -> None:
    import mmpp.solitons.vortex.spectrum.gyration as gyration

    calls: list[str] = []

    def fake_compute_psd(signal, time=None, **kwargs):
        calls.append(str(kwargs.get("method")))
        return np.array([0.0, 1.0]), np.array([0.0, 2.0]), "shared_psd", {"dt": 1.0}

    monkeypatch.setattr(gyration, "compute_psd", fake_compute_psd)

    result = gyration.compute_gyration_spectrum(_trajectory(), method="welch")

    assert calls == ["welch", "welch"]
    assert result.method == "shared_psd"
    np.testing.assert_allclose(result.power, [0.0, 4.0])


def test_shared_spectral_fallback_uses_central_fft_backend(monkeypatch) -> None:
    import mmpp._shared.spectral as spectral

    calls: list[str] = []

    class FakeBackend:
        @staticmethod
        def rfft(signal, n=None, axis=-1):
            calls.append("rfft")
            return np.fft.rfft(signal, n=n, axis=axis)

        @staticmethod
        def rfftfreq(n, d=1.0):
            calls.append("rfftfreq")
            return np.fft.rfftfreq(n, d=d)

        @staticmethod
        def fft(signal, n=None, axis=-1):
            calls.append("fft")
            return np.fft.fft(signal, n=n, axis=axis)

        @staticmethod
        def fftfreq(n, d=1.0):
            calls.append("fftfreq")
            return np.fft.fftfreq(n, d=d)

        @staticmethod
        def get_info():
            return {"backend": "fake"}

    monkeypatch.setattr(spectral, "_central_fft_backend", FakeBackend)

    _, _, _, metadata = spectral.compute_psd(
        np.arange(16.0),
        dt=1.0,
        method="periodogram",
    )

    assert "rfft" in calls
    assert "rfftfreq" in calls
    assert metadata["backend"] == "fake"
    assert metadata["central_backend"] is True


def test_autofit_features_use_shared_spectral_helper(monkeypatch) -> None:
    import mmpp.solitons.vortex.autofit.features as features

    calls: list[str] = []

    def fake_compute_psd(signal, time=None, **kwargs):
        calls.append(str(kwargs.get("method")))
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, float(np.sum(signal**2))]),
            "fake",
            {},
        )

    monkeypatch.setattr(features, "compute_psd", fake_compute_psd)

    freqs, power = features._compute_psd(
        np.linspace(0.0, 1.0, 8),
        np.ones(8),
        np.ones(8) * 2.0,
    )

    assert calls == ["periodogram", "periodogram"]
    np.testing.assert_allclose(freqs, [1.0])
    np.testing.assert_allclose(power, [40.0])


def test_vortex_spectrum_and_trajectory_helpers_use_unified_template() -> None:
    core = _CountingCore()
    spectrum = VortexSpectrumInterface(
        job_result=SimpleNamespace(path="run.zarr"),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(spectrum=SpectrumConfig()),
        core_interface=core,
    )
    spectrum_html = spectrum._repr_html_()
    assert "Overview" in spectrum_html
    assert "API" in spectrum_html
    assert "jobs[-1].solitons.vortex.spectrum.gyration" in spectrum_html
    assert "Important Arguments" in spectrum_html
    assert "<h3" not in spectrum_html

    trajectory = __import__(
        "mmpp.solitons.vortex.trajectory.interface",
        fromlist=["TrajectoryInterface"],
    ).TrajectoryInterface(
        job_result=SimpleNamespace(path="run.zarr"),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
        core_interface=core,
    )
    trajectory_html = trajectory._repr_html_()
    assert "Overview" in trajectory_html
    assert "API" in trajectory_html
    assert "jobs[-1].solitons.vortex.trajectory.raw" in trajectory_html
    assert "Namespace Catalog" in trajectory_html
    assert "<h3" not in trajectory_html


def test_vortex_result_helpers_use_unified_template() -> None:
    spectrum_result = VortexSpectrumResult(
        frequencies=np.array([0.0, 1.0e9, 2.0e9]),
        power=np.array([0.0, 4.0, 1.0]),
        method="welch",
    )
    spectrum_html = spectrum_result._repr_html_()
    assert "Vortex Spectrum Result" in spectrum_html
    assert "jobs[-1].solitons.vortex.spectrum.gyration()" in spectrum_html
    assert "Result Usage" in spectrum_html

    orbit_html = OrbitFitResult(
        center=(0.0, 0.0),
        semi_major=8.0,
        semi_minor=6.0,
        eccentricity=0.2,
        tilt_angle=0.1,
        residual=0.01,
    )._repr_html_()
    assert "Orbit Fit Result" in orbit_html
    assert "jobs[-1].solitons.vortex.trajectory.orbit.fit()" in orbit_html
    assert "Fit Usage" in orbit_html


def test_vortex_topology_model_and_bridge_helpers_use_unified_template() -> None:
    topology_interface = __import__(
        "mmpp.solitons.vortex.topology.interface",
        fromlist=["TopologyInterface"],
    ).TopologyInterface(
        job_result=SimpleNamespace(
            attrs={},
            get_largest_m_dataset=lambda: "m",
            _ensure_zarr_loaded=lambda: None,
            _z={"m": object()},
        ),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
    )
    topology_html = topology_interface._repr_html_()
    assert "jobs[-1].solitons.vortex.topology.detect" in topology_html
    assert "Topology Workflows" in topology_html
    assert "<h3" not in topology_html

    topology_result_html = TopologyResult(
        polarity=1,
        vorticity=1,
        chirality=1,
        Q=0.5,
        core_position=(0.0, 0.0),
        topological_density=np.zeros((4, 4)),
        state="vortex",
        method="finite_diff",
        confidence=0.95,
    )._repr_html_()
    assert "Topology Result" in topology_result_html
    assert "jobs[-1].solitons.vortex.topology.detect()" in topology_result_html

    model_interface = __import__(
        "mmpp.solitons.vortex.model.interface",
        fromlist=["VortexModelInterface"],
    ).VortexModelInterface(
        job_result=SimpleNamespace(path="run.zarr"),
        dataset_name="m",
        slice_info=None,
    )
    model_html = model_interface._repr_html_()
    assert "jobs[-1].solitons.vortex.model" in model_html
    assert "Model Workflows" in model_html

    bridge_interface = __import__(
        "mmpp.solitons.vortex.bridge.interface",
        fromlist=["BridgeInterface"],
    ).BridgeInterface(
        job_result=SimpleNamespace(path="run.zarr"),
        dataset_name="m",
        slice_info=None,
    )
    bridge_html = bridge_interface._repr_html_()
    assert "jobs[-1].solitons.vortex.bridge.fit.thiele_from_trajectory" in bridge_html
    assert "Bridge Workflows" in bridge_html


def test_vortex_events_helpers_use_unified_template() -> None:
    from mmpp.solitons.vortex.events.interface import EventsInterface
    from mmpp.solitons.vortex.events.models import (
        CoreExpulsionEvent,
        DwellTimeResult,
        PolaritySwitchEvent,
        StateSwitchEvent,
    )
    from mmpp.solitons.vortex.trajectory.interface import TrajectoryInterface

    core = _CountingCore()
    trajectory = TrajectoryInterface(
        job_result=SimpleNamespace(path="run.zarr"),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
        core_interface=core,
    )
    interface = EventsInterface(
        job_result=SimpleNamespace(path="run.zarr", attrs={}),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
        core_interface=core,
        trajectory_interface=trajectory,
    )
    html = interface._repr_html_()
    assert "jobs[-1].solitons.vortex.events.polarity_switches" in html
    assert "Event Workflows" in html
    assert "<h3" not in html

    plot_html = interface.plt._repr_html_()
    assert "jobs[-1].solitons.vortex.events.plt" in plot_html

    polarity_html = PolaritySwitchEvent(1e-9, 5, 1, -1, 0.9)._repr_html_()
    assert "Polarity Switch Event" in polarity_html
    assert "jobs[-1].solitons.vortex.events.polarity_switches()[0]" in polarity_html

    state_html = StateSwitchEvent(2e-9, 8, "G-state", "C-state", 0.8)._repr_html_()
    assert "State Switch Event" in state_html

    expulsion_html = CoreExpulsionEvent(3e-9, 10, 40e-9, 45e-9, 0.7, 1e-9)._repr_html_()
    assert "Core Expulsion Event" in expulsion_html

    dwell_html = DwellTimeResult(
        state="G-state",
        dwell_times=np.array([1e-9, 2e-9, 1.5e-9]),
    )._repr_html_()
    assert "Dwell Time Result" in dwell_html
    assert "jobs[-1].solitons.vortex.events.dwell_times()" in dwell_html


def test_vortex_modes_helpers_use_unified_template() -> None:
    from mmpp.solitons.vortex.modes.interface import VortexModesInterface
    from mmpp.solitons.vortex.modes.models import VortexModeResult

    interface = VortexModesInterface(
        job_result=SimpleNamespace(path="run.zarr"),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
        core_interface=_CountingCore(),
        spectrum_interface=SimpleNamespace(),
    )
    html = interface._repr_html_()
    assert "jobs[-1].solitons.vortex.modes.classify" in html
    assert "Modes Workflows" in html
    assert "<h3" not in html

    plot_html = interface.plt._repr_html_()
    assert "jobs[-1].solitons.vortex.modes.plt" in plot_html

    result_html = VortexModeResult(
        m_index=1,
        n_index=0,
        mode_type="gyration",
        rotation_sense="ccw",
        confidence=0.9,
        frequency_hz=1.2e9,
        power=3.5,
    )._repr_html_()
    assert "Vortex Mode Result" in result_html
    assert "jobs[-1].solitons.vortex.modes.classify()" in result_html


def test_vortex_nonlinear_helpers_use_unified_template() -> None:
    from mmpp.solitons.vortex.nonlinear.interface import NonlinearInterface
    from mmpp.solitons.vortex.nonlinear.models import (
        AmplitudeEquationResult,
        STBatchResult,
        STParametersResult,
        ThieleForceBalanceResult,
    )

    interface = NonlinearInterface(
        job_result=SimpleNamespace(path="run.zarr"),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
        core_interface=_CountingCore(),
        trajectory_interface=SimpleNamespace(),
        spectrum_interface=SimpleNamespace(),
    )
    html = interface._repr_html_()
    assert "jobs[-1].solitons.vortex.nonlinear.slavin_tiberkevich" in html
    assert "Nonlinear Examples" in html
    assert "<h3" not in html

    plot_html = interface.plt._repr_html_()
    assert "jobs[-1].solitons.vortex.nonlinear.plt" in plot_html

    amp_html = AmplitudeEquationResult(
        time=np.array([0.0, 1.0]),
        complex_amplitude=np.array([1 + 0j, 0.5 + 0.5j]),
        power=np.array([1.0, 0.5]),
        phase=np.array([0.0, 0.1]),
        omega=np.array([1.0, 1.1]),
        method="hilbert",
        reference_radius=5e-9,
    )._repr_html_()
    assert "Amplitude Equation Result" in amp_html

    st_html = STParametersResult(
        omega_0=1.0,
        f_0_ghz=0.5,
        N=2.0,
        Gamma_G=0.1,
        Q=0.2,
        sigma=1.0,
        I_threshold=1e-3,
        generation_power=0.3,
        linewidth_hz=1e6,
        quality_factor=100.0,
        linewidth_resolution_limited=False,
    )._repr_html_()
    assert "ST Parameters Result" in st_html

    batch_html = STBatchResult(
        currents=np.array([1e-3, 2e-3]),
        powers=np.array([0.2, 0.3]),
        linewidths=np.array([1e6, 0.8e6]),
        frequencies_hz=np.array([1e9, 1.1e9]),
        N=2.0,
    )._repr_html_()
    assert "ST Batch Result" in batch_html

    fb_html = ThieleForceBalanceResult(
        time=np.array([0.0, 1.0]),
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 1.0]),
        vx=np.array([0.0, 1.0]),
        vy=np.array([0.0, 1.0]),
        gyro_force=np.ones((2, 2)),
        conservative_force=np.ones((2, 2)),
        dissipative_force=np.ones((2, 2)),
        stt_force=np.ones((2, 2)),
        oersted_force=np.ones((2, 2)),
        residual_force=np.ones((2, 2)),
        G=1.0,
        D=1.0,
        kappa=1.0,
        polarity=1,
        vorticity=1,
    )._repr_html_()
    assert "Thiele Force Balance Result" in fb_html


def test_vortex_signals_and_energy_helpers_use_unified_template() -> None:
    from mmpp.solitons.vortex.numerical.energy.interface import EnergyInterface
    from mmpp.solitons.vortex.numerical.energy.models import (
        EffectivePotentialResult,
        EnergyTimeSeriesResult,
        PinningResult,
        PinningSite,
    )
    from mmpp.solitons.vortex.numerical.signals.interface import SignalsInterface
    from mmpp.solitons.vortex.numerical.signals.models import (
        MagnetoresistanceResult,
        SignalSpectrumResult,
        VoltageResult,
    )

    signals = SignalsInterface(
        job_result=SimpleNamespace(path="run.zarr"),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
        core_interface=_CountingCore(),
    )
    signals_html = signals._repr_html_()
    assert "jobs[-1].solitons.vortex.signals.magnetoresistance" in signals_html
    assert "Signals Workflows" in signals_html
    assert "<h3" not in signals_html
    assert "jobs[-1].solitons.vortex.signals.plt" in signals.plt._repr_html_()

    mr_html = MagnetoresistanceResult(
        time=np.array([0.0, 1.0]),
        resistance_ohm=np.array([100.0, 101.0]),
        projection=np.array([0.1, 0.2]),
        method="proxy",
    )._repr_html_()
    assert "Magnetoresistance Result" in mr_html

    v_html = VoltageResult(
        time=np.array([0.0, 1.0]),
        voltage_v=np.array([1e-3, 2e-3]),
        current_a=np.array([1e-3, 1e-3]),
        resistance_ohm=np.array([1.0, 2.0]),
    )._repr_html_()
    assert "Voltage Result" in v_html

    sig_html = SignalSpectrumResult(
        frequencies_hz=np.array([0.0, 1e9]),
        power=np.array([0.0, 1.0]),
        quantity="voltage",
        metadata={"method": "welch"},
    )._repr_html_()
    assert "Signal Spectrum Result" in sig_html

    energy = EnergyInterface(
        job_result=SimpleNamespace(path="run.zarr"),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
        core_interface=_CountingCore(),
    )
    energy_html = energy._repr_html_()
    assert "jobs[-1].solitons.vortex.energy.time_resolved" in energy_html
    assert "Energy Workflows" in energy_html
    assert "jobs[-1].solitons.vortex.energy.plt" in energy.plt._repr_html_()

    etrace_html = EnergyTimeSeriesResult(
        time=np.array([0.0, 1.0]),
        channels={"E_total": np.array([1.0, 2.0])},
    )._repr_html_()
    assert "Energy Time Series Result" in etrace_html

    pot = EffectivePotentialResult(
        radius_m=np.array([0.0, 1e-9]),
        potential_j=np.array([0.0, 1.0]),
        probability=np.array([0.5, 0.5]),
        method="boltzmann",
    )
    assert "Effective Potential Result" in pot._repr_html_()

    pin_html = PinningResult(
        potential=pot,
        sites=[
            PinningSite(radius_m=1e-9, potential_j=0.0, depth_j=1.0, confidence=0.9)
        ],
    )._repr_html_()
    assert "Pinning Result" in pin_html


def test_vortex_spectrogram_uses_shared_spectral_helper(monkeypatch) -> None:
    import mmpp.solitons.vortex.spectrum.spectrogram as spectrogram

    calls: list[int | None] = []

    def fake_compute_spectrogram_psd(signal, time=None, **kwargs):
        calls.append(kwargs.get("nperseg"))
        return (
            np.array([0.5]),
            np.array([1.0]),
            np.array([[2.0]]),
            "shared_stft",
            {"dt": 1.0, "nperseg": kwargs.get("nperseg")},
        )

    monkeypatch.setattr(
        spectrogram,
        "compute_spectrogram_psd",
        fake_compute_spectrogram_psd,
    )

    result = spectrogram.compute_spectrogram(
        _trajectory(),
        component="x",
        nperseg=32,
    )

    assert calls == [32]
    assert result.method == "shared_stft"
    np.testing.assert_allclose(result.power, [[2.0]])


def test_signal_power_spectrum_uses_shared_spectral_helper(monkeypatch) -> None:
    import mmpp.solitons.vortex.numerical.signals.power_spectrum as power_spectrum

    calls: list[str] = []

    def fake_compute_psd(signal, dt=None, **kwargs):
        calls.append(str(kwargs.get("method")))
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, 3.0]),
            "shared_psd",
            {"dt": dt},
        )

    monkeypatch.setattr(power_spectrum, "compute_psd", fake_compute_psd)

    result = power_spectrum.compute_signal_power_spectrum(
        np.arange(8.0),
        np.ones(8),
        quantity="voltage",
        method="periodogram",
    )

    assert calls == ["periodogram"]
    assert result.metadata["method"] == "shared_psd"
    np.testing.assert_allclose(result.power, [0.0, 3.0])


def test_directional_spectrum_uses_shared_spectral_helper(monkeypatch) -> None:
    import mmpp.solitons.vortex._shared.analysis as analysis

    calls: list[str] = []

    def fake_compute_psd(signal, dt=None, **kwargs):
        calls.append(str(kwargs.get("method")))
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, float(np.sum(np.abs(signal)))]),
            "shared_psd",
            {"dt": dt},
        )

    monkeypatch.setattr(analysis, "compute_psd", fake_compute_psd)

    result = _trajectory().analysis.spectrum.directional(method="periodogram")

    assert calls == ["periodogram", "periodogram"]
    assert result.method == "shared_psd"
    np.testing.assert_allclose(result.frequencies, [0.0, 1.0])


def test_interactive_proxy_psd_uses_shared_spectral_helper(monkeypatch) -> None:
    import mmpp.solitons.vortex.nonlinear.interactive as interactive

    calls: list[str] = []

    def fake_compute_psd(signal, dt=None, **kwargs):
        calls.append(str(kwargs.get("method")))
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, 4.0]),
            "shared_psd",
            {"dt": dt},
        )

    monkeypatch.setattr(interactive, "compute_psd", fake_compute_psd)

    frequencies, power = interactive.proxy_psd(
        np.ones(16),
        dt=1.0,
        method="fft",
    )

    assert calls == ["periodogram"]
    np.testing.assert_allclose(frequencies, [0.0, 1.0])
    np.testing.assert_allclose(power, [0.0, 4.0])


def test_vortex_spectrum_interface_reuses_cache_until_force() -> None:
    core = _CountingCore()
    interface = VortexSpectrumInterface(
        job_result=None,
        dataset_name=None,
        slice_info=None,
        config=VortexConfig(),
        core_interface=core,
    )

    first = interface.gyration()
    second = interface.gyration()
    forced = interface.gyration(force=True)

    assert first is second
    assert forced is not first
    assert core.calls == 2


def test_vortex_spectrum_interface_and_plot_repr_use_tabs() -> None:
    interface = VortexSpectrumInterface(
        job_result=None,
        dataset_name=None,
        slice_info=None,
        config=VortexConfig(),
        core_interface=_CountingCore(),
    )

    html = interface._repr_html_()
    plot_html = interface.plt._repr_html_()

    assert "Vortex spectrum API help" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "Vortex spectrum plot API help" in plot_html
    assert ">Overview</button>" in plot_html
    assert ">API</button>" in plot_html


def test_solitons_plot_accessors_accept_save(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    output = tmp_path / "spectrum.png"
    result = VortexSpectrumResult(
        frequencies=np.array([0.0, 1.0e9]),
        power=np.array([0.0, 1.0]),
        method="periodogram",
    )

    ax = result.plt.power_spectrum(save=output)

    assert ax is not None
    assert output.exists()


def test_missing_result_html_cards_are_present() -> None:
    time = np.array([0.0, 1.0])
    force = np.zeros((2, 2), dtype=float)
    cases = [
        OrbitFitResult(
            center=(0.0, 0.0),
            semi_major=2.0,
            semi_minor=1.0,
            eccentricity=0.5,
            tilt_angle=0.0,
            residual=0.01,
        ),
        AmplitudeEquationResult(
            time=time,
            complex_amplitude=np.array([1.0 + 0.0j, 0.0 + 1.0j]),
            power=np.array([1.0, 1.0]),
            phase=np.array([0.0, np.pi / 2.0]),
            omega=np.array([1.0, 1.0]),
            method="complex",
            reference_radius=1.0,
        ),
        STParametersResult(
            omega_0=1.0,
            f_0_ghz=1.0,
            N=2.0,
            Gamma_G=3.0,
            Q=0.1,
            sigma=0.2,
            I_threshold=0.0,
            generation_power=0.3,
            linewidth_hz=4.0,
            quality_factor=5.0,
            linewidth_resolution_limited=False,
        ),
        STBatchResult(
            currents=np.array([1.0]),
            powers=np.array([0.5]),
            linewidths=np.array([1.0]),
            frequencies_hz=np.array([2.0e9]),
            N=2.0,
        ),
        ThieleForceBalanceResult(
            time=time,
            x=time,
            y=time,
            vx=time,
            vy=time,
            gyro_force=force,
            conservative_force=force,
            dissipative_force=force,
            stt_force=force,
            oersted_force=force,
            residual_force=force,
            G=1.0,
            D=2.0,
            kappa=3.0,
            polarity=1,
            vorticity=1,
        ),
        VortexSpectrumResult(
            frequencies=np.array([1.0e9]),
            power=np.array([1.0]),
            method="periodogram",
        ),
    ]

    for result in cases:
        html = result._repr_html_()
        assert result.__class__.__name__ in html
        assert "<div" in html


def test_missing_batch_html_cards_are_present() -> None:
    batch = BatchSolitonsInterface([])
    vortex = BatchVortexInterface([])
    spectrum_map = BatchVortexSpectrumMapResult(
        coordinate=np.array([1.0]),
        frequencies=np.array([2.0e9]),
        power=np.array([[3.0]]),
        component="gyration",
    )

    cases = [
        batch,
        vortex,
        vortex.spectrum,
        vortex.plt,
        spectrum_map,
        spectrum_map.plt,
    ]

    for result in cases:
        html = result._repr_html_()
        assert result.__class__.__name__ in html
        assert "<div" in html


def test_batch_solitons_helpers_use_unified_node_card_template() -> None:
    batch = BatchSolitonsInterface([])
    vortex = BatchVortexInterface([])
    spectrum_map = BatchVortexSpectrumMapResult(
        coordinate=np.array([1.0]),
        frequencies=np.array([2.0e9]),
        power=np.array([[3.0]]),
        component="gyration",
    )

    cases = [
        (batch._repr_html_(), "Batch solitons API help"),
        (vortex._repr_html_(), "Batch vortex API help"),
        (vortex.spectrum._repr_html_(), "Batch vortex spectrum API help"),
        (vortex.plt._repr_html_(), "Batch vortex plot API help"),
        (spectrum_map._repr_html_(), "Batch vortex spectrum-map API help"),
        (spectrum_map.plt._repr_html_(), "Batch vortex spectrum-map plot API help"),
    ]

    for html, api_title in cases:
        assert api_title in html
        assert ">Overview</button>" in html
        assert ">API</button>" in html
        assert "<h3" not in html

    vortex_html = vortex._repr_html_()
    spectrum_html = vortex.spectrum._repr_html_()
    plot_html = vortex.plt._repr_html_()
    assert "jobs.vortex.spectrum_map" in vortex_html
    assert "jobs.vortex.current_spectrum_map" in vortex_html
    assert "spec = jobs.vortex.spectrum" in spectrum_html
    assert "plot = jobs.vortex.plt" in plot_html


def test_batch_vortex_interactive_displays_single_selected_result() -> None:
    calls: list[tuple[str, tuple[float, float], int, str, str]] = []

    class _InteractiveVortex:
        def __init__(self, label: str):
            self.label = label

        def interactive(
            self,
            *,
            figsize=(10, 7),
            dpi=100,
            trajectory_source="magnetization",
            center_mode="auto",
        ):
            calls.append((self.label, figsize, dpi, trajectory_source, center_mode))
            return f"dashboard-{self.label}"

    class _InteractiveSolitons:
        def __init__(self, label: str):
            self.vortex = _InteractiveVortex(label)

    class _InteractiveResult:
        def __init__(self, label: str, current_ma: float):
            self.path = f"{label}.zarr"
            self.attrs = {"i_pillar_ma": current_ma}
            self.solitons = _InteractiveSolitons(label)

    interface = BatchVortexInterface(
        [
            _InteractiveResult("high", 2.0),
            _InteractiveResult("low", 1.0),
        ]
    )

    dashboard = interface.interactive(
        index=0,
        figsize=(8, 5),
        dpi=150,
        trajectory_source="compare",
        center_mode="orbit",
    )

    assert dashboard == "dashboard-low"
    assert calls == [("low", (8, 5), 150, "compare", "orbit")]


def test_vortex_dashboard_show_reuses_display_handle(monkeypatch) -> None:
    from mmpp.solitons.vortex.ui import interactive_dashboard as dashboard_module

    updates: list[object] = []

    class _Handle:
        def update(self, value):
            updates.append(value)

    handle = _Handle()
    display_calls: list[tuple[object, bool]] = []

    def fake_display(value, display_id=False):
        display_calls.append((value, bool(display_id)))
        return handle

    dashboard = dashboard_module.VortexInteractiveDashboard.__new__(
        dashboard_module.VortexInteractiveDashboard
    )
    dashboard._build = lambda: None
    dashboard._root = object()
    dashboard._display_handle = None
    dashboard._css_displayed = False

    monkeypatch.setattr(dashboard_module, "display", fake_display)

    assert dashboard.show() is dashboard
    assert dashboard.show() is dashboard

    assert len(display_calls) == 2
    assert display_calls[1] == (dashboard._root, True)
    assert updates == [dashboard._root]


def test_vortex_dashboard_show_figure_uses_image_widget_not_output_or_global_display(
    monkeypatch,
) -> None:
    import matplotlib.pyplot as plt

    from mmpp.solitons.vortex.ui import interactive_dashboard as dashboard_module

    class _Output:
        def __init__(self):
            self.clear_calls = 0
            self.append_calls = 0
            self.displayed: list[object] = []
            self.layout = type("_Layout", (), {"display": ""})()
            self.outputs = ("old",)

        def clear_output(self, wait=False):
            self.clear_calls += 1

        def append_display_data(self, value):
            self.append_calls += 1
            self.displayed.append(value)

    class _Image:
        def __init__(self):
            self.value = b""
            self.layout = type("_Layout", (), {"display": "none"})()

    class _Placeholder:
        def __init__(self):
            self.layout = type("_Layout", (), {"display": ""})()

    global_display_calls: list[object] = []
    monkeypatch.setattr(
        dashboard_module,
        "display",
        lambda value, **kwargs: global_display_calls.append(value),
    )

    dashboard = dashboard_module.VortexInteractiveDashboard.__new__(
        dashboard_module.VortexInteractiveDashboard
    )
    output = _Output()
    dashboard._output = output
    dashboard._plot_image = _Image()
    dashboard._plot_placeholder = _Placeholder()
    dashboard._fig = None

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    dashboard._show_figure(fig)

    assert output.clear_calls == 0
    assert output.append_calls == 0
    assert output.outputs == ()
    assert output.layout.display == "none"
    assert dashboard._plot_image.value.startswith(b"\x89PNG")
    assert dashboard._plot_image.layout.display == ""
    assert dashboard._plot_placeholder.layout.display == "none"
    assert global_display_calls == []


def test_vortex_dashboard_table_tab_defaults_to_time_and_mx_columns() -> None:
    from mmpp.solitons.vortex.ui import interactive_dashboard as dashboard_module

    dashboard = dashboard_module.VortexInteractiveDashboard.__new__(
        dashboard_module.VortexInteractiveDashboard
    )
    dashboard._get_table_column_names = lambda: ["my", "t", "mx", "mz"]
    dashboard._module_run_buttons = {}
    dashboard._controls = {}

    dashboard._build_tab_table()

    controls = dashboard._controls["table"]
    assert controls["x_col"].value == "t"
    assert controls["y_col"].value == "mx"


def test_vortex_dashboard_core_tracking_uses_controls_and_avoids_empty_range() -> None:
    from mmpp.solitons.vortex.ui import interactive_dashboard as dashboard_module

    class _Widget:
        def __init__(self, value):
            self.value = value
            self.max = 5000

    class _Core:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def track(self, **kwargs):
            self.calls.append(kwargs)
            return _trajectory()

    class _Vortex:
        def __init__(self):
            self.core = _Core()

    dashboard = dashboard_module.VortexInteractiveDashboard.__new__(
        dashboard_module.VortexInteractiveDashboard
    )
    dashboard._vx = _Vortex()
    dashboard._state = dashboard_module._DashboardState()
    dashboard.figsize = (5, 3)
    dashboard.dpi = 80
    dashboard._status = type("_Status", (), {"value": ""})()
    dashboard._get_health = lambda: None
    shown = []
    dashboard._show_figure = lambda fig, health=None: shown.append(fig)

    controls = {
        "method": _Widget("centroid"),
        "threshold": _Widget(0.42),
        "t_start": _Widget(9999),
        "t_end": _Widget(12000),
        "cmap": _Widget("viridis"),
        "show_orbit": _Widget(True),
        "show_geom": _Widget(False),
        "smooth": _Widget(False),
        "smooth_window": _Widget(5),
    }

    dashboard._run_core(controls)

    assert dashboard._vx.core.calls == [{"method": "centroid", "core_threshold": 0.42}]
    assert controls["t_start"].max == 255
    assert controls["t_end"].max == 256
    assert "Selected time range was empty" in dashboard._status.value

    ax_xy = shown[0].axes[0]
    assert ax_xy.collections[0].get_offsets().shape[0] == 256


def test_vortex_dashboard_centers_grid_coordinates_for_disk_display() -> None:
    from mmpp.solitons.vortex.ui import interactive_dashboard as dashboard_module

    class _Raw:
        shape = (1001, 1, 320, 320, 3)

    class _Job:
        attrs = {"dx": 1e-9, "dy": 1e-9, "R": 154e-9}

        def get_raw(self, dataset_name):
            assert dataset_name == "m"
            return _Raw()

    class _Vortex:
        dataset_name = "m"
        _job = _Job()

    dashboard = dashboard_module.VortexInteractiveDashboard.__new__(
        dashboard_module.VortexInteractiveDashboard
    )
    dashboard._vx = _Vortex()

    phase = np.linspace(0.0, 2.0 * np.pi, 128)
    x_raw = 159.5e-9 + 25e-9 * np.cos(phase)
    y_raw = 159.5e-9 + 20e-9 * np.sin(phase)

    x_nm, y_nm, center_nm = dashboard._display_xy_nm(x_raw, y_raw)
    r_nm = np.hypot(x_nm, y_nm)

    assert np.allclose(center_nm, (159.5, 159.5))
    assert abs(float(np.mean(x_nm))) < 1.0
    assert abs(float(np.mean(y_nm))) < 1.0
    assert float(np.max(r_nm)) < 154.0


def test_vortex_dashboard_trajectory_tab_exposes_source_and_center_controls() -> None:
    from mmpp.solitons.vortex.ui import interactive_dashboard as dashboard_module

    dashboard = dashboard_module.VortexInteractiveDashboard.__new__(
        dashboard_module.VortexInteractiveDashboard
    )
    dashboard._module_run_buttons = {}
    dashboard._controls = {}

    dashboard._build_tab_trajectory()

    controls = dashboard._controls["trajectory"]
    assert controls["source"].value == "magnetization"
    assert controls["center_mode"].value == "auto"
    assert controls["show_centers"].value is True


def test_vortex_dashboard_interactive_defaults_seed_core_and_trajectory_controls() -> (
    None
):
    from mmpp.solitons.vortex.ui import interactive_dashboard as dashboard_module

    dashboard = dashboard_module.VortexInteractiveDashboard.__new__(
        dashboard_module.VortexInteractiveDashboard
    )
    dashboard._module_run_buttons = {}
    dashboard._controls = {}
    dashboard._default_center_mode = "orbit"
    dashboard._default_trajectory_source = "compare"
    dashboard._get_tracking_frame_count = lambda: 0

    dashboard._build_tab_core()
    dashboard._build_tab_trajectory()

    assert dashboard._controls["core"]["center_mode"].value == "orbit"
    assert dashboard._controls["trajectory"]["center_mode"].value == "orbit"
    assert dashboard._controls["trajectory"]["source"].value == "compare"


def test_vortex_interface_interactive_passes_dashboard_flags(monkeypatch) -> None:
    from mmpp.solitons.vortex import ui as ui_module
    from mmpp.solitons.vortex.interface import VortexInterface

    calls: list[dict[str, object]] = []

    class _Dashboard:
        def __init__(self, vortex, **kwargs):
            calls.append(kwargs)

        def show(self):
            calls.append({"show": True})

    monkeypatch.setattr(ui_module, "VortexInteractiveDashboard", _Dashboard)

    vortex = VortexInterface.__new__(VortexInterface)
    result = vortex.interactive(
        figsize=(7, 4),
        dpi=130,
        trajectory_source="compare",
        center_mode="orbit",
    )

    assert isinstance(result, _Dashboard)
    assert calls == [
        {
            "figsize": (7, 4),
            "dpi": 130,
            "trajectory_source": "compare",
            "center_mode": "orbit",
        },
        {"show": True},
    ]


def test_vortex_dashboard_header_contains_status_log_not_footer() -> None:
    from mmpp.solitons.vortex.ui import interactive_dashboard as dashboard_module

    class _Layout:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.display = kwargs.get("display", "")

    class _Widget:
        def __init__(self, *children, **kwargs):
            self.children = tuple(children[0]) if children else ()
            self.layout = kwargs.get("layout")
            self.value = kwargs.get("value", "")

        def observe(self, *args, **kwargs):
            return None

    class _Widgets:
        Layout = _Layout
        HTML = _Widget
        Image = _Widget
        Output = _Widget
        VBox = _Widget
        HBox = _Widget
        Select = _Widget

    dashboard = dashboard_module.VortexInteractiveDashboard.__new__(
        dashboard_module.VortexInteractiveDashboard
    )
    dashboard._built = False
    dashboard._vx = type("_Vortex", (), {"dataset_name": "m"})()
    dashboard._display_handle = None
    dashboard._css_displayed = False
    dashboard._module_panels = {}
    dashboard._module_run_buttons = {}
    dashboard._controls = {}
    dashboard._build_job_info = lambda: _Widget()
    dashboard._build_preset_row = lambda: _Widget()
    dashboard._build_tab_core = lambda: _Widget()
    dashboard._build_tab_topology = lambda: _Widget()
    dashboard._build_tab_trajectory = lambda: _Widget()
    dashboard._build_tab_spectrum = lambda: _Widget()
    dashboard._build_tab_spectrogram = lambda: _Widget()
    dashboard._build_tab_modes = lambda: _Widget()
    dashboard._build_tab_events = lambda: _Widget()
    dashboard._build_tab_signals = lambda: _Widget()
    dashboard._build_tab_thiele = lambda: _Widget()
    dashboard._build_tab_table = lambda: _Widget()

    old_widgets = dashboard_module.widgets
    try:
        dashboard_module.widgets = _Widgets()
        dashboard._build()
    finally:
        dashboard_module.widgets = old_widgets

    header = dashboard._root.children[0]
    assert header is dashboard._status
    assert "Vortex Dynamics" in header.value
    assert "Ready" in header.value
    assert "vdash-status" not in header.value
    assert len(dashboard._root.children) == 2

    dashboard._set_status("Computing core trajectory...", "info")
    assert "Vortex Dynamics" in dashboard._status.value
    assert "Computing core trajectory..." in dashboard._status.value


def test_vortex_dashboard_resolves_table_and_magnetization_timebases_separately() -> (
    None
):
    from mmpp.solitons.vortex.ui import interactive_dashboard as dashboard_module

    def make_traj(
        t0_ns: float, t1_ns: float, n: int, radius_nm: float
    ) -> TrajectoryResult:
        time = np.linspace(t0_ns, t1_ns, n) * 1e-9
        phase = np.linspace(0.0, 2.0 * np.pi, n)
        return TrajectoryResult(
            time=time,
            x=159.5e-9 + radius_nm * 1e-9 * np.cos(phase),
            y=159.5e-9 + radius_nm * 1e-9 * np.sin(phase),
            polarity=np.ones(n),
            confidence=np.ones(n),
            method="synthetic",
        )

    table_traj = make_traj(0.0, 40.0, 129, 20.0)
    mag_traj = make_traj(20.0, 40.0, 33, 18.0)

    class _Raw:
        shape = (1001, 1, 320, 320, 3)

    class _Job:
        attrs = {"dx": 1e-9, "dy": 1e-9, "R": 154e-9}

        def get_raw(self, dataset_name):
            return _Raw()

    class _Core:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def track(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("method") == "table":
                return table_traj
            return mag_traj

    class _Vortex:
        dataset_name = "m"
        _job = _Job()

        def __init__(self):
            self.core = _Core()

    dashboard = dashboard_module.VortexInteractiveDashboard.__new__(
        dashboard_module.VortexInteractiveDashboard
    )
    dashboard._vx = _Vortex()
    dashboard._state = dashboard_module._DashboardState()

    views = dashboard._resolve_trajectory_views(
        "compare",
        "orbit",
        tracking_method="centroid",
        core_threshold=0.42,
    )

    assert [v.source for v in views] == ["magnetization", "table"]
    assert [len(v.t_ns) for v in views] == [33, 129]
    assert views[0].stats["t_start_ns"] == 20.0
    assert views[1].stats["t_start_ns"] == 0.0
    assert dashboard._vx.core.calls == [
        {"method": "centroid", "core_threshold": 0.42},
        {"method": "table"},
    ]


def test_vortex_dashboard_auto_center_reports_orbit_radius_not_disk_offset() -> None:
    from mmpp.solitons.vortex.ui import interactive_dashboard as dashboard_module

    class _Raw:
        shape = (1001, 1, 320, 320, 3)

    class _Job:
        attrs = {"dx": 1e-9, "dy": 1e-9, "R": 154e-9}

        def get_raw(self, dataset_name):
            return _Raw()

    class _Vortex:
        dataset_name = "m"
        _job = _Job()

    dashboard = dashboard_module.VortexInteractiveDashboard.__new__(
        dashboard_module.VortexInteractiveDashboard
    )
    dashboard._vx = _Vortex()

    phase = np.linspace(0.0, 2.0 * np.pi, 256)
    traj = TrajectoryResult(
        time=np.linspace(0.0, 30e-9, 256),
        x=209.5e-9 + 24e-9 * np.cos(phase),
        y=159.5e-9 + 12e-9 * np.sin(phase),
        polarity=np.ones(256),
        confidence=np.ones(256),
        method="synthetic",
    )

    view = dashboard._resolve_trajectory_view(
        traj,
        label="magnetization",
        source="magnetization",
        center_mode="auto",
    )

    assert view.center_mode_used == "orbit"
    assert float(np.max(view.r_orbit_nm)) < 30.0
    assert 49.0 < view.stats["center_offset_nm"] < 51.0
    assert view.stats["r_max_over_disk_radius"] < 0.25
    assert (
        view.stats["geometry_r_max_over_disk_radius"]
        > view.stats["r_max_over_disk_radius"]
    )
    assert 0.7 < view.stats["eccentricity"] < 0.95


def test_api_help_html_includes_signatures_parameters_and_examples() -> None:
    from mmpp._repr_helpers import api_help_html

    class Demo:
        def compute(self, value: float, *, scale: float = 1.0) -> float:
            """Compute a scaled value."""
            return value * scale

    html = api_help_html(
        Demo(),
        title="Demo help",
        prefix="demo",
        properties=[("plt", "Plotting accessor")],
        methods=["compute"],
    )

    assert "Demo help" in html
    assert ".compute(" in html
    assert "value" in html
    assert "scale" in html
    assert "Compute a scaled value." in html
    assert "demo.compute(value=...)" in html
    assert "demo.plt" in html


def test_api_help_html_chrome_false_matches_overview_visual_language() -> None:
    from mmpp._repr_helpers import api_help_html

    class Demo:
        def compute(self, value: float, *, scale: float = 1.0) -> float:
            """Compute a scaled value."""
            return value * scale

    html = api_help_html(
        Demo(),
        title="Demo API help",
        prefix="demo",
        subtitle="Live signatures from the interface.",
        properties=[("plt", "Plotting accessor")],
        methods=["compute"],
        chrome=False,
    )

    assert "font-size:1.1em;font-weight:600;color:#f8f8f2" in html
    assert (
        "background:linear-gradient(135deg,#282a36 0%,#21222c 50%,#44475a 100%)" in html
    )
    assert (
        "background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%)"
        in html
    )
    assert "Namespaces / properties" in html
    assert "Methods" in html
    assert "Accessor" in html
    assert "Signature" in html
    assert "border:1px solid rgba(98,114,164,0.35)" in html


def test_node_card_html_uses_canonical_single_card_tabs_first_layout() -> None:
    from mmpp._repr_helpers import (
        NODE_COLOR_ANALYSIS,
        accessors_section_html,
        examples_section_html,
        metrics_section_html,
        node_card_html,
    )

    html = node_card_html(
        "Demo Node",
        icon="🧪",
        subtitle="Live signatures from the interface.",
        badge=("ready", "#22c55e"),
        sections=[
            metrics_section_html([("dataset", "m", "#93c5fd")]),
            accessors_section_html([("Analysis:", [(".fft", NODE_COLOR_ANALYSIS)])]),
            examples_section_html("demo.fft()"),
        ],
        api="<div>api body</div>",
        uid="demo-node",
    )

    assert "<h3" not in html
    assert "border: 2px solid #6272a4" in html
    assert "box-shadow: 0 10px 25px rgba(0,0,0,0.45)" in html
    assert html.find(">Overview</button>") < html.find("Demo Node")
    assert html.find("Demo Node") < html.find("dataset")
    assert "demo-node-panel-1" in html


def test_helper_card_html_provides_canonical_tabs_badge_and_action_grid() -> None:
    from mmpp._repr_helpers import helper_card_html, helper_table_html

    html = helper_card_html(
        "Demo Helper",
        subtitle="Canonical helper template",
        status=("ready", "#22c55e"),
        metrics=[("datasets", 2)],
        details=[
            (
                "Datasets",
                helper_table_html([("m", "magnetization"), ("table", "scalar data")]),
            )
        ],
        action_groups=[
            (
                "Analysis",
                [
                    (".fft", "FFT namespace", "#50fa7b"),
                    (".analyze", "analysis tools", "#50fa7b"),
                ],
            )
        ],
        tabs=[
            ("Overview", "<p>overview body</p>"),
            ("API", "<p>api body</p>"),
        ],
        uid="demo-helper",
    )

    assert "Demo Helper" in html
    assert "Canonical helper template" in html
    assert "ready" in html
    assert "border: 2px solid #6272a4" in html
    assert "box-shadow: 0 10px 25px rgba(0,0,0,0.45)" in html
    assert "font-family:'Courier New',monospace" in html
    assert "datasets" in html
    assert "ACCESSORS &amp; METHODS" in html
    assert ".fft" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "demo-helper-panel-1" in html
    assert "<h3" not in html
    assert html.find(">Overview</button>") < html.find("Demo Helper")


def test_mmpp_repr_keeps_api_help_inside_single_job_manager_card() -> None:
    from mmpp.core.mmpp import MMPP

    job = MMPP.__new__(MMPP)
    job.base_path = "/tmp/empty"
    job.zarr_results = []

    html = job._repr_html_()

    assert "MMPP Job Manager" in html
    assert "MMPP API help" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "style='display:none;'" in html
    assert html.count("MMPP Job Manager") == 1


def test_fft_repr_uses_tabs_for_overview_and_api() -> None:
    from mmpp.fft.core import FFT

    class _Job:
        name = "run"
        path = "/tmp/run.zarr"

    fft = FFT(_Job())
    html = fft._repr_html_()

    assert "FFT Analysis Interface" in html
    assert "FFT API help" in html
    assert "ready" in html
    assert "border: 2px solid #6272a4" in html
    assert "box-shadow: 0 10px 25px rgba(0,0,0,0.45)" in html
    assert "font-family:'Courier New',monospace" in html
    assert "cache entries" in html
    assert "ACCESSORS &amp; METHODS" in html
    assert ".spectrum()" in html
    assert ".modes" in html
    assert ".dispersion" in html
    assert ".transmission" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "<h3" not in html
    assert html.find(">Overview</button>") < html.find("FFT Analysis Interface")
    assert "fft-job-" in html
    assert "-panel-1" in html


def test_zarr_job_fft_property_uses_lazy_import_even_when_feature_flag_is_stale() -> (
    None
):
    from mmpp.core.job import ZarrJobResult

    job = ZarrJobResult("/tmp/run.zarr", {})
    fft = job.fft

    assert fft.job_result is job
    assert "FFT Analysis Interface" in fft._repr_html_()


def test_batch_repr_uses_tabs_for_overview_and_api() -> None:
    from mmpp.batch_operations import BatchOperations

    html = BatchOperations([], None)._repr_html_()

    assert "BatchOperations API help" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "style='display:none;'" in html


def test_dataset_wrapper_repr_uses_unified_node_card_template() -> None:
    from mmpp.core.dataset import DatasetAwareWrapper

    class _Job:
        name = "run"

    class _Array:
        shape = (4, 1, 8, 8, 3)
        dtype = np.dtype("float32")
        chunks = (1, 1, 4, 4, 3)

    dataset = DatasetAwareWrapper(_Job(), "m", _Array())
    html = dataset._repr_html_()

    assert "Dataset View" in html
    assert "Dataset API help" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "style='display:none;'" in html
    assert "<h3" not in html
    assert html.find(">Overview</button>") < html.find("Dataset View")
    assert "dataset-m-" in html
    assert "job[0].m" in html
    assert ".fft.spectrum()" in html
    assert ".plot.snapshot()" in html


def test_dataset_plot_backend_repr_uses_unified_node_card_template() -> None:
    from mmpp.core.dataset import DatasetAwareWrapper

    class _Job:
        name = "run"

    class _Array:
        shape = (4, 1, 8, 8, 3)
        dtype = np.dtype("float32")
        chunks = (1, 1, 4, 4, 3)

    dataset = DatasetAwareWrapper(_Job(), "m", _Array())
    html = dataset.plot.mpl._repr_html_()

    assert "Matplotlib Plot Backend" in html
    assert "Matplotlib Plot Backend API help" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "style='display:none;'" in html
    assert "<h3" not in html
    assert html.find(">Overview</button>") < html.find("Matplotlib Plot Backend")
    assert "mmpp-dataset-plot-" in html
    assert "job[0].m.plt.mpl" in html
    assert ".snapshot(z=0, t=-1, figsize=(8, 5), dpi=100)" in html


def test_spectrum_plot_accessor_repr_includes_live_api_help() -> None:
    from mmpp.fft.spectrum._plotting.accessor import SpectrumPlotAccessor

    class _Modes:
        def interactive(self, **kwargs):
            return kwargs

    class _Result:
        modes = _Modes()

    html = SpectrumPlotAccessor(_Result())._repr_html_()

    assert "Spectrum plot accessor API help" in html
    assert ".spectrum(**kwargs)" in html
    assert ".interactive(**kwargs)" in html
    assert "spec.plot.spectrum()" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_spectrum_quick_plot_repr_includes_forwarded_api_help() -> None:
    from mmpp.fft.spectrum.helpers import _SpectrumQuickPlot

    class _Helper:
        def __call__(self, **kwargs):
            return kwargs

    html = _SpectrumQuickPlot(_Helper())._repr_html_()

    assert "Spectrum quick-plot API help" in html
    assert ".spectrum(**compute_kw)" in html
    assert ".interactive(**compute_kw)" in html
    assert "job[0].fft.spectrum.plot.spectrum()" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_spectrum_modes_plot_repr_includes_live_api_help() -> None:
    from mmpp.fft.spectrum.modes.accessor import SpectrumModesPlotAccessor

    html = SpectrumModesPlotAccessor(object())._repr_html_()

    assert "Spectrum modes plot API help" in html
    assert ".imshow(" in html
    assert "component" in html
    assert ".animation(" in html
    assert "spec.modes.plot.imshow(f=...)" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_mode_plot_repr_includes_live_api_help() -> None:
    from mmpp.fft.modes.interface import ModePlotAccessor

    class _Mode:
        frequency = 5.2
        z_layer = -1

        def get_component(self, component="z", value="magnitude"):
            return np.zeros((2, 2))

    html = ModePlotAccessor(_Mode())._repr_html_()

    assert "Mode plot API help" in html
    assert ".imshow(" in html
    assert ".interactive(" in html
    assert "mode.plot.imshow()" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_dispersion_plot_accessor_repr_includes_live_api_help() -> None:
    from mmpp.fft.dispersion._plotting.accessor import DispersionPlotAccessor

    html = DispersionPlotAccessor(object())._repr_html_()

    assert "Dispersion plot API help" in html
    assert ".heatmap(" in html
    assert ".branch(" in html
    assert ".add_analytics(" in html
    assert "disp.plot.heatmap()" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_dispersion_filter_chain_repr_uses_unified_node_card_template() -> None:
    from types import SimpleNamespace

    from mmpp.fft.dispersion.filter_chain import DispersionFilterChain

    iface = SimpleNamespace(
        _filters_config={
            "remove_static": True,
            "live": {"gaussian_morph": {"enabled": True}},
        }
    )
    html = DispersionFilterChain(iface)._repr_html_()

    assert "Dispersion Filter Chain" in html
    assert "Dispersion filter-chain API help" in html
    assert ".compute_1d" in html
    assert ".compute_2d" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "<h3" not in html


def test_lowest_frequency_result_repr_uses_unified_node_card_template() -> None:
    from types import SimpleNamespace

    from mmpp.fft.dispersion.analyze import LowestFrequencyResult

    result = LowestFrequencyResult(
        f_min_hz=4.8e9,
        f_min_ghz=4.8,
        k_at_f_min=1.2e6,
        k_at_f_min_um=1.2,
        f_at_k0_hz=5.0e9,
        f_at_k0_ghz=5.0,
        group_velocity_at_min=1200.0,
        branch_f=np.array([4.8e9, 5.0e9]),
        branch_k=np.array([1.2e6, 0.0]),
        result=SimpleNamespace(),
        side="positive",
    )
    html = result._repr_html_()

    assert "LowestFrequencyResult" in html
    assert "Lowest frequency result API help" in html
    assert ".plot.heatmap(lognorm=True)" in html
    assert ".plot.branch" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "<h3" not in html


def test_dispersion_modes_bridge_repr_includes_live_api_help() -> None:
    from mmpp.fft.dispersion.modes.bridge import DispersionModesBridge

    html = DispersionModesBridge(object())._repr_html_()

    assert "Dispersion modes bridge API help" in html
    assert ".interactive(" in html
    assert ".at(" in html
    assert "disp_result.modes.interactive()" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_dispersion_mode_plot_repr_includes_live_api_help() -> None:
    from mmpp.fft.dispersion.modes.bridge import (
        DispersionModePlotAccessor,
        DispersionModeResult,
    )

    mode = DispersionModeResult(
        mode_data=np.zeros((2, 2)),
        k_rad_um=1.0,
        f_ghz=5.0,
        z_layer=0,
        component="z",
        result=object(),
    )
    html = DispersionModePlotAccessor(mode)._repr_html_()

    assert "Dispersion mode plot API help" in html
    assert ".imshow(" in html
    assert ".phase(" in html
    assert "mode.plot.imshow()" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_dispersion_modes_plot_repr_includes_live_api_help() -> None:
    from mmpp.fft.dispersion.modes.bridge import DispersionModesPlotAccessor

    html = DispersionModesPlotAccessor(object())._repr_html_()

    assert "Dispersion modes plot API help" in html
    assert ".animation(" in html
    assert "disp_result.modes.plot.animation()" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_dispersion_interface_repr_uses_tabs_for_overview_and_api() -> None:
    from types import SimpleNamespace

    from mmpp.fft.core import FFT
    from mmpp.fft.dispersion.interface import FFTDispersionInterface

    parent_fft = FFT(SimpleNamespace(name="run", path="/tmp/run.zarr"))
    html = FFTDispersionInterface(parent_fft)._repr_html_()

    assert "FFT dispersion API help" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_transmission_interface_repr_includes_live_api_help() -> None:
    from mmpp.fft.transmission.interface import FFTTransmissionInterface

    class _Job:
        def get_largest_m_dataset(self):
            return "m"

    interface = FFTTransmissionInterface(
        fft_instance=object(),
        fft_compute=object(),
        job_result=_Job(),
    )
    html = interface._repr_html_()

    assert "FFT transmission API help" in html
    assert ".compute(" in html
    assert ".plot_transmission(" in html
    assert ".visualize_mode(" in html
    assert "job[0].fft.transmission.compute()" in html


def test_transmission_result_repr_includes_live_api_help() -> None:
    from mmpp.fft.transmission.compute import TransmissionConfig, TransmissionResult

    result = TransmissionResult(
        frequencies=np.array([1.0, 2.0]),
        x_positions=np.array([0.0, 1.0]),
        transmission=np.ones((2, 2)),
        power_map=np.ones((2, 2)),
        reference_power=np.ones(2),
        config=TransmissionConfig(),
    )
    html = result._repr_html_()

    assert "TransmissionResult API help" in html
    assert ".plot_transmission(" in html
    assert ".plot_transmission_crosssection(" in html
    assert ".visualize_mode(" in html
    assert "transmission.plot_transmission()" in html


def test_soliton_and_vortex_interfaces_include_live_api_help() -> None:
    from mmpp.solitons.interface import SolitonInterface
    from mmpp.solitons.vortex.interface import VortexInterface

    class _Job:
        def get_largest_m_dataset(self):
            return "m"

    soliton_html = SolitonInterface(_Job(), dataset_name="m")._repr_html_()
    vortex_html = VortexInterface(_Job(), dataset_name="m")._repr_html_()

    assert "Soliton API help" in soliton_html
    assert "job[0].solitons.vortex" in soliton_html
    assert ">Overview</button>" in soliton_html
    assert ">API</button>" in soliton_html
    assert "Vortex API help" in vortex_html
    assert ".track(" in vortex_html
    assert ".detect(" in vortex_html
    assert "job[0].vortex.track()" in vortex_html
    assert ">Overview</button>" in vortex_html
    assert ">API</button>" in vortex_html


def test_analyze_and_hysteresis_interfaces_use_tabbed_api_help() -> None:
    from mmpp.analyze import AnalyzeInterface
    from mmpp.analyze.hysteresis import HysteresisInterface

    class _Job:
        name = "run"
        path = "/tmp/run.zarr"

    analyze_html = AnalyzeInterface(_Job())._repr_html_()
    hysteresis_html = HysteresisInterface(_Job())._repr_html_()
    quick_plot_html = HysteresisInterface(_Job()).plot._repr_html_()

    assert "Analyze API help" in analyze_html
    assert "job[0].analyze.hysteresis" in analyze_html
    assert ">Overview</button>" in analyze_html
    assert ">API</button>" in analyze_html

    assert "Hysteresis API help" in hysteresis_html
    assert ".from_table(" in hysteresis_html
    assert ".from_zarr_keys(" in hysteresis_html
    assert "job[0].analyze.hysteresis.from_table()" in hysteresis_html
    assert ">Overview</button>" in hysteresis_html
    assert ">API</button>" in hysteresis_html

    assert "Hysteresis quick plot API help" in quick_plot_html
    assert ".interactive(**kwargs)" in quick_plot_html
    assert "job[0].analyze.hysteresis.plot.loop()" in quick_plot_html
    assert ">Overview</button>" in quick_plot_html
    assert ">API</button>" in quick_plot_html


def test_hysteresis_result_plot_accessor_uses_tabbed_api_help() -> None:
    from mmpp.analyze.hysteresis.plot.accessor import HysteresisPlotAccessor

    html = HysteresisPlotAccessor(object())._repr_html_()

    assert "Hysteresis plot API help" in html
    assert ".loop(**kwargs)" in html
    assert ".animation(**kwargs)" in html
    assert "result.plot.interactive()" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_vortex_modes_and_topology_repr_use_tabbed_api_help() -> None:
    from mmpp.solitons.vortex.config import VortexConfig
    from mmpp.solitons.vortex.modes.interface import VortexModesInterface
    from mmpp.solitons.vortex.topology.interface import TopologyInterface

    class _Core:
        def track(self, **kwargs):
            return _trajectory()

    modes = VortexModesInterface(
        job_result=object(),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
        core_interface=_Core(),
        spectrum_interface=object(),
    )
    modes_html = modes._repr_html_()
    modes_plot_html = modes.plt._repr_html_()
    topology_html = TopologyInterface(
        job_result=object(),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
    )._repr_html_()

    assert "Vortex modes API help" in modes_html
    assert ".classify(" in modes_html
    assert "vortex.modes.classify()" in modes_html
    assert ">Overview</button>" in modes_html
    assert ">API</button>" in modes_html

    assert "Vortex modes plot API help" in modes_plot_html
    assert ".mode_map(" in modes_plot_html
    assert "vortex.modes.plt.mode_table()" in modes_plot_html
    assert ">Overview</button>" in modes_plot_html
    assert ">API</button>" in modes_plot_html

    assert "Topology API help" in topology_html
    assert ".detect(" in topology_html
    assert ".topological_charge(" in topology_html
    assert "vortex.topology.detect()" in topology_html
    assert ">Overview</button>" in topology_html
    assert ">API</button>" in topology_html


def test_vortex_nonlinear_repr_uses_tabbed_api_help() -> None:
    from mmpp.solitons.vortex.config import VortexConfig
    from mmpp.solitons.vortex.nonlinear.interface import NonlinearInterface

    class _Core:
        def track(self, **kwargs):
            return _trajectory()

    nonlinear = NonlinearInterface(
        job_result=object(),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
        core_interface=_Core(),
        trajectory_interface=object(),
        spectrum_interface=object(),
    )

    html = nonlinear._repr_html_()
    plot_html = nonlinear.plt._repr_html_()

    assert "Nonlinear dynamics API help" in html
    assert ".amplitude_equation(" in html
    assert ".slavin_tiberkevich_batch(" in html
    assert "vortex.nonlinear.force_balance()" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html

    assert "Nonlinear plot API help" in plot_html
    assert ".power_vs_current(" in plot_html
    assert ".linewidth_vs_current(" in plot_html
    assert "vortex.nonlinear.plt.force_balance()" in plot_html
    assert ">Overview</button>" in plot_html
    assert ">API</button>" in plot_html


def test_vortex_core_repr_uses_tabbed_api_help() -> None:
    from mmpp.solitons.vortex.config import VortexConfig
    from mmpp.solitons.vortex.numerical.core.interface import CoreInterface

    core = CoreInterface(
        job_result=object(),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
    )
    html = core._repr_html_()

    assert "Core tracking API help" in html
    assert ".track(" in html
    assert ".position(" in html
    assert "vortex.core.track()" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_vortex_autofit_repr_uses_tabbed_api_help() -> None:
    from mmpp.solitons.vortex.autofit.interface import AutofitInterface

    autofit = AutofitInterface(object())
    html = autofit._repr_html_()
    thiele_html = autofit.thiele._repr_html_()

    assert "Vortex Autofit Interface" in html
    assert "Vortex autofit API help" in html
    assert "vortex.autofit.thiele" in html
    assert "tracking_source" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "Vortex autofit API help" in thiele_html
    assert ">Overview</button>" in thiele_html
    assert ">API</button>" in thiele_html


def test_trajectory_interface_repr_includes_live_api_help() -> None:
    from mmpp.solitons.vortex.config import VortexConfig
    from mmpp.solitons.vortex.trajectory.interface import TrajectoryInterface

    class _Core:
        def track(self):
            return object()

    html = TrajectoryInterface(
        job_result=object(),
        dataset_name="m",
        slice_info=None,
        config=VortexConfig(),
        core_interface=_Core(),
    )._repr_html_()

    assert "Vortex trajectory API help" in html
    assert ".filtered(" in html
    assert ".steady_state(" in html
    assert "vortex.trajectory.filtered()" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html


def test_canonical_nonlinear_thiele_module_alias() -> None:
    from mmpp.solitons.vortex.nonlinear.nonlinear_thiele import ThieleAnalyzer
    from mmpp.solitons.vortex.nonlinear.nonliniearthiele import (
        ThieleAnalyzer as LegacyThieleAnalyzer,
    )

    assert ThieleAnalyzer is LegacyThieleAnalyzer


def test_canonical_vortex_comparison_module_alias() -> None:
    from mmpp.solitons.vortex.comparison import VortexAnalyticalComparison
    from mmpp.solitons.vortex.plotting import (
        VortexAnalyticalComparison as LegacyVortexAnalyticalComparison,
    )

    assert VortexAnalyticalComparison is LegacyVortexAnalyticalComparison


def test_vortex_spectrum_config_validates_segment_sizes() -> None:
    import pytest

    with pytest.raises(ValueError, match="nperseg"):
        SpectrumConfig(nperseg=-1)

    with pytest.raises(ValueError, match="noverlap"):
        SpectrumConfig(noverlap=-1)


def test_batch_operations_jobs_compatibility_index() -> None:
    class Result:
        def __init__(self, path: str):
            self.path = path
            self.attrs = {"path": path}

    class Parent:
        zarr_results = [Result("a.zarr"), Result("b.zarr")]

    batch = BatchOperations([Parent.zarr_results[1]], Parent())

    assert batch.jobs[0].index == 1
    assert batch.jobs[0].result is Parent.zarr_results[1]


def test_batch_summary_and_spectrum_map_parallel_match_sequential() -> None:
    results = [_FakeBatchResult(index) for index in range(4)]
    interface = BatchVortexInterface(results)

    sequential_summary = interface.summary(show_progress=False)
    parallel_summary = interface.summary(
        show_progress=False,
        parallel=True,
        max_workers=2,
    )

    assert sequential_summary.to_dict("records") == parallel_summary.to_dict("records")

    sequential_map = interface.spectrum_map(show_progress=False)
    parallel_map = interface.spectrum_map(
        show_progress=False,
        parallel=True,
        max_workers=2,
    )

    np.testing.assert_allclose(parallel_map.coordinate, sequential_map.coordinate)
    np.testing.assert_allclose(parallel_map.frequencies, sequential_map.frequencies)
    np.testing.assert_allclose(parallel_map.power, sequential_map.power)


def test_batch_current_spectrum_map_defaults_to_i_pillar_ma() -> None:
    results = [_FakeBatchResult(index) for index in range(3)]
    interface = BatchVortexInterface(results)

    spectrum_map = interface.current_spectrum_map(show_progress=False)

    assert spectrum_map.coordinate_name == "i_pillar_ma"
    np.testing.assert_allclose(spectrum_map.coordinate, [0.0, 1.0, 2.0])
    assert spectrum_map.power.shape[0] == 3
    assert spectrum_map.power.shape[1] == spectrum_map.frequencies.size


def test_batch_vortex_spectrum_map_can_use_table_magnetization_component() -> None:
    time = np.linspace(0.0, 20e-9, 256)
    results = [
        _FakeBatchResultWithTable(
            index,
            {
                "t": time,
                "mx": np.sin(2.0 * np.pi * (index + 1) * 1.0e9 * time),
                "my": np.zeros_like(time),
            },
        )
        for index in range(2)
    ]
    interface = BatchVortexInterface(results)

    spectrum_map = interface.spectrum_map(
        source="table",
        component="mx",
        spectrum_method="periodogram",
        show_progress=False,
        cache=False,
    )

    assert spectrum_map.component == "table:mx"
    assert spectrum_map.metadata["source"] == "table"
    assert spectrum_map.metadata["magnetization_component"] == "mx"
    assert results[0].solitons.vortex.spectrum.calls == 0
    assert spectrum_map.power.shape[0] == 2
    assert spectrum_map.frequencies.size > 0


def test_batch_vortex_table_spectrum_heatmap_returns_axis() -> None:
    from matplotlib.axes import Axes

    time = np.linspace(0.0, 20e-9, 256)
    results = [
        _FakeBatchResultWithTable(
            0,
            {"t": time, "mx": np.sin(2.0 * np.pi * 1.0e9 * time)},
        )
    ]
    interface = BatchVortexInterface(results)

    ax = interface.spectrum_map(
        sort_by="i_pillar_ma",
        source="table",
        component="mx",
        show_progress=False,
        cache=False,
    ).plt.heatmap()

    assert isinstance(ax, Axes)
    ax.set_ylim(0.0, 3.0)
    assert ax.get_ylim() == (0.0, 3.0)


def test_batch_vortex_spectrum_map_can_use_raw_magnetization_component() -> None:
    time = np.linspace(0.0, 20e-9, 256)
    data = np.zeros((time.size, 1, 2, 2, 3), dtype=float)
    data[..., 2] = np.sin(2.0 * np.pi * 2.0e9 * time)[:, None, None, None]
    results = [_FakeBatchResultWithMagnetization(0, data, dt=float(time[1] - time[0]))]
    interface = BatchVortexInterface(results)

    spectrum_map = interface.spectrum_map(
        source="magnetization",
        component="mz",
        spectrum_method="periodogram",
        show_progress=False,
        cache=False,
    )

    assert spectrum_map.component == "magnetization:mz"
    assert spectrum_map.metadata["source"] == "magnetization"
    assert spectrum_map.metadata["magnetization_component"] == "mz"
    assert results[0].solitons.vortex.spectrum.calls == 0
    assert spectrum_map.power.shape[0] == 1


def test_batch_vortex_spectrum_map_auto_component_splits_processed_and_magnetization() -> (
    None
):
    trajectory_result = BatchVortexInterface([_FakeBatchResult(0)]).spectrum_map(
        component="gyration",
        show_progress=False,
        cache=False,
    )

    time = np.linspace(0.0, 20e-9, 256)
    table_result = BatchVortexInterface(
        [
            _FakeBatchResultWithTable(
                0,
                {
                    "t": time,
                    "mx": np.sin(2.0 * np.pi * 1.0e9 * time),
                },
            )
        ]
    ).spectrum_map(
        component="mx",
        show_progress=False,
        cache=False,
    )

    assert trajectory_result.metadata["source"] == "processed"
    assert table_result.metadata["source"] == "table"


def test_batch_vortex_spectrum_map_persists_and_reuses_cache(tmp_path) -> None:
    results = [_FakeBatchResult(index) for index in range(2)]
    interface = BatchVortexInterface(results)

    first = interface.spectrum_map(
        show_progress=False,
        cache_dir=tmp_path,
    )
    second = interface.spectrum_map(
        show_progress=False,
        cache_dir=tmp_path,
    )

    assert first.metadata["cache"]["status"] == "stored"
    assert second.metadata["cache"]["status"] == "hit"
    assert results[0].solitons.vortex.spectrum.calls == 1
    np.testing.assert_allclose(second.coordinate, first.coordinate)
    np.testing.assert_allclose(second.frequencies, first.frequencies)
    np.testing.assert_allclose(second.power, first.power)
    assert first.metadata["cache"]["path"] == second.metadata["cache"]["path"]
    assert (tmp_path / "metadata.json").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["kind"] == "vortex.spectrum_map"
    assert metadata["config"]["sort_by"] == "i_pillar_ma"


def test_batch_vortex_spectrum_map_force_recomputes_cached_result(tmp_path) -> None:
    results = [_FakeBatchResult(index) for index in range(2)]
    interface = BatchVortexInterface(results)

    interface.spectrum_map(show_progress=False, cache_dir=tmp_path)
    forced = interface.spectrum_map(show_progress=False, cache_dir=tmp_path, force=True)

    assert forced.metadata["cache"]["status"] == "stored"
    assert forced.metadata["cache"]["force"] is True
    assert results[0].solitons.vortex.spectrum.calls == 2


def test_batch_vortex_spectrum_map_cache_can_be_disabled(tmp_path) -> None:
    results = [_FakeBatchResult(index) for index in range(2)]
    interface = BatchVortexInterface(results)

    result = interface.spectrum_map(
        show_progress=False,
        cache=False,
        cache_dir=tmp_path,
    )

    assert "cache" not in result.metadata
    assert not (tmp_path / "metadata.json").exists()


def test_batch_vortex_spectrum_map_can_use_process_pool(monkeypatch, tmp_path) -> None:
    import mmpp.solitons.batch as batch_module

    calls: list[int] = []

    class FakeProcessPool:
        def __init__(self, *, max_workers: int):
            calls.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, iterable):
            return [func(item) for item in iterable]

    monkeypatch.setattr(batch_module, "ProcessPoolExecutor", FakeProcessPool)

    results = [_FakeBatchResult(index) for index in range(2)]
    interface = BatchVortexInterface(results)

    result = interface.spectrum_map(
        show_progress=False,
        parallel="process",
        max_workers=3,
        cache_dir=tmp_path,
    )

    assert calls == [3]
    assert result.power.shape[0] == 2


def test_batch_progress_iter_uses_tqdm_auto(monkeypatch) -> None:
    import mmpp.solitons.batch as batch_module

    calls: list[dict[str, object]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iter(iterable)

    monkeypatch.setitem(
        sys.modules,
        "tqdm.auto",
        SimpleNamespace(tqdm=fake_tqdm),
    )

    values = list(
        batch_module._progress_iter(
            range(3),
            total=3,
            desc="Computing current map",
            enabled=True,
        )
    )

    assert values == [0, 1, 2]
    assert calls == [
        {
            "total": 3,
            "desc": "Computing current map",
            "unit": "result",
            "leave": True,
        }
    ]


def test_batch_vortex_plot_current_spectrum_map_uses_current_axis() -> None:
    results = [_FakeBatchResult(index) for index in range(3)]
    interface = BatchVortexInterface(results)

    ax = interface.plt.current_spectrum_map(show_progress=False)

    assert ax.get_xlabel() == "Current [mA]"
    assert "spectrum" in ax.get_title().lower()


def test_batch_vortex_spectrum_accessor_current_map_alias() -> None:
    results = [_FakeBatchResult(index) for index in range(2)]
    interface = BatchVortexInterface(results)

    spectrum_map = interface.spectrum.current_map(show_progress=False)

    assert spectrum_map.coordinate_name == "i_pillar_ma"
    assert spectrum_map.coordinate.shape == (2,)


def test_batch_vortex_analyze_phase_diagram_api_and_alias() -> None:
    results = [_FakeBatchResult(index) for index in range(3)]
    interface = BatchVortexInterface(results)

    result = interface.analyze.phase_diagram(x="i_pillar_ma", show_progress=False)
    alias = interface.phase_diagram(x="i_pillar_ma", show_progress=False)

    assert isinstance(result, BatchVortexPhaseDiagramResult)
    assert result.axes == ("i_pillar_ma", None, None)
    assert result.metric == "regime"
    assert result.frame.shape[0] == 3
    assert result.metadata["dimension"] == 1
    assert alias.axes == result.axes


def test_batch_vortex_phase_diagram_accepts_attrs_axes_and_aggregates_duplicates() -> (
    None
):
    results = [_FakeBatchResult(0), _FakeBatchResult(1), _FakeBatchResult(2)]
    results[0].attrs.update({"epsilonprime": 0.1, "i_pillar_ma": 1.0})
    results[1].attrs.update({"epsilonprime": 0.1, "i_pillar_ma": 1.0})
    results[2].attrs.update({"epsilonprime": 0.2, "i_pillar_ma": 2.0})
    interface = BatchVortexInterface(results)

    result = interface.analyze.phase_diagram(
        x="i_pillar_ma",
        y="epsilonprime",
        metric="peak_power_rel",
        aggregate="mean",
        show_progress=False,
    )

    assert result.axes == ("i_pillar_ma", "epsilonprime", None)
    assert result.metadata["dimension"] == 2
    assert result.metadata["aggregate"] == "mean"
    assert result.frame.shape[0] == 2
    first = result.frame.sort_values("i_pillar_ma").iloc[0]
    assert first["epsilonprime"] == 0.1
    assert np.isfinite(float(first["peak_power_rel"]))


def test_batch_vortex_phase_diagram_rejects_missing_axis() -> None:
    interface = BatchVortexInterface([_FakeBatchResult(0)])

    try:
        interface.analyze.phase_diagram(x="missing_axis", show_progress=False)
    except ValueError as exc:
        assert "missing_axis" in str(exc)
    else:
        raise AssertionError("missing phase-diagram axis should raise ValueError")


def test_batch_vortex_phase_diagram_plots_1d_2d_and_3d() -> None:
    results = [_FakeBatchResult(0), _FakeBatchResult(1), _FakeBatchResult(2)]
    for index, result in enumerate(results):
        result.attrs["epsilonprime"] = 0.1 + 0.1 * (index % 2)
        result.attrs["addfl"] = float(index)
    interface = BatchVortexInterface(results)

    one_d = interface.phase_diagram(x="i_pillar_ma", show_progress=False)
    ax_1d = one_d.plt.map()
    assert ax_1d.get_xlabel() == "Current [mA]"

    two_d = interface.phase_diagram(
        x="i_pillar_ma",
        y="epsilonprime",
        show_progress=False,
    )
    ax_2d = two_d.plt.map()
    assert ax_2d.get_xlabel() == "Current [mA]"
    assert ax_2d.collections or ax_2d.images

    three_d = interface.phase_diagram(
        x="i_pillar_ma",
        y="epsilonprime",
        z="addfl",
        show_progress=False,
    )
    ax_3d = three_d.plt.surface3d()
    assert hasattr(ax_3d, "get_zlabel")
    assert ax_3d.get_zlabel() == "addfl"


def test_batch_vortex_phase_diagram_repr_surfaces_new_namespace() -> None:
    interface = BatchVortexInterface([_FakeBatchResult(0)])
    result = interface.phase_diagram(show_progress=False)

    assert "Batch vortex analyze API help" in interface.analyze._repr_html_()
    assert "phase_diagram" in interface._repr_html_()
    assert "BatchVortexPhaseDiagramResult" in result._repr_html_()
    assert "phase diagram plot API help" in result.plt._repr_html_()


def test_batch_memory_profile_metadata_is_optional() -> None:
    results = [_FakeBatchResult(0)]
    interface = BatchVortexInterface(results)

    summary = interface.summary(show_progress=False, profile_memory=True)
    spectrum_map = interface.spectrum_map(show_progress=False, profile_memory=True)

    assert "memory_profile" in summary.attrs
    assert "memory_delta_mb" in summary.attrs["memory_profile"]
    assert "memory_profile" in spectrum_map.metadata
    assert "memory_delta_mb" in spectrum_map.metadata["memory_profile"]


def test_canonical_numerical_nonlinear_thiele_module_alias() -> None:
    from mmpp.solitons.vortex.numerical.nonlinear.nonlinear_thiele import (
        ThieleAnalyzer,
    )
    from mmpp.solitons.vortex.numerical.nonlinear.nonliniearthiele import (
        ThieleAnalyzer as LegacyThieleAnalyzer,
    )

    assert ThieleAnalyzer is LegacyThieleAnalyzer


def test_public_topology_is_canonical_and_numerical_path_is_compatibility() -> None:
    from mmpp.solitons.vortex.numerical.topology import (
        TopologyInterface as NumericalTopologyInterface,
    )
    from mmpp.solitons.vortex.topology import TopologyInterface
    from mmpp.solitons.vortex.topology.detection import detect_topology
    from mmpp.solitons.vortex.topology.invariants import winding_number

    assert TopologyInterface is NumericalTopologyInterface
    assert detect_topology.__module__ == "mmpp.solitons.vortex.topology.detection"
    assert winding_number.__module__ == "mmpp.solitons.vortex.topology.invariants"


def test_public_events_are_canonical_and_numerical_path_is_compatibility() -> None:
    from mmpp.solitons.vortex.events import EventsInterface
    from mmpp.solitons.vortex.numerical.events import (
        EventsInterface as NumericalEventsInterface,
    )

    assert EventsInterface is NumericalEventsInterface
    assert EventsInterface.__module__ == "mmpp.solitons.vortex.events.interface"


def test_autofit_guard_diagnostics_are_split_from_single_module() -> None:
    from mmpp.solitons.vortex.autofit import cpp_threshold_guard_penalty
    from mmpp.solitons.vortex.autofit.diagnostics import (
        cpp_threshold_guard_penalty as diagnostics_guard,
    )

    assert cpp_threshold_guard_penalty is diagnostics_guard
    assert diagnostics_guard.__module__ == "mmpp.solitons.vortex.autofit.diagnostics"


def test_autofit_seed_helpers_are_split_from_single_module() -> None:
    from mmpp.solitons.vortex.autofit import select_threshold_aware_seed
    from mmpp.solitons.vortex.autofit.seeds import (
        select_threshold_aware_seed as seeds_select,
    )

    assert select_threshold_aware_seed is seeds_select
    assert seeds_select.__module__ == "mmpp.solitons.vortex.autofit.seeds"


def test_autofit_simulation_context_is_split_from_single_module() -> None:
    from mmpp.solitons.vortex.autofit import SimulationContext
    from mmpp.solitons.vortex.autofit.simulation import (
        SimulationContext as module_context,
    )

    assert SimulationContext is module_context
    assert module_context.__module__ == "mmpp.solitons.vortex.autofit.simulation"


def test_single_autofit_orchestrator_uses_split_modules() -> None:
    import inspect

    import mmpp.solitons.vortex.autofit.single as single

    source = inspect.getsource(single)

    assert "class _SimulationContext" not in source
    assert "def _cpp_threshold_guard_penalty" not in source
    assert "def _select_threshold_aware_seed" not in source
    assert single._SimulationContext.__module__ == (
        "mmpp.solitons.vortex.autofit.simulation"
    )
    assert single._cpp_threshold_guard_penalty.__module__ == (
        "mmpp.solitons.vortex.autofit.diagnostics"
    )
    assert single._select_threshold_aware_seed.__module__ == (
        "mmpp.solitons.vortex.autofit.seeds"
    )


def test_batch_orbit_plots_expose_grid_alpha_disk_boundary_and_save(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from matplotlib.patches import Circle

    results = [_FakeBatchResult(0), _FakeBatchResult(1)]
    interface = BatchVortexInterface(results)

    radius_path = tmp_path / "orbit_radius.png"
    radius_ax = interface.plt.orbit_radius(show_progress=False, save=radius_path)
    assert radius_ax is not None
    assert radius_path.exists()

    dashboard_path = tmp_path / "dashboard.png"
    dashboard_fig, _, _ = interface.plt.dashboard(
        show_progress=False,
        save=dashboard_path,
    )
    assert dashboard_fig is not None
    assert dashboard_path.exists()

    orbits_path = tmp_path / "orbits.png"
    ax = interface.plt.orbits(
        show_progress=False,
        colorbar=False,
        grid_alpha=0.37,
        show_disk_boundary=True,
        save=orbits_path,
    )

    assert any(isinstance(patch, Circle) for patch in ax.patches)
    assert ax.xaxis.get_gridlines()[0].get_alpha() == 0.37
    assert orbits_path.exists()

    grid_path = tmp_path / "orbits_grid.png"
    fig, axes = interface.plt.orbits_grid(
        show_progress=False,
        colorbar=False,
        grid_alpha=0.41,
        show_disk_boundary=True,
        ncols=2,
        save=grid_path,
    )

    assert fig is not None
    assert any(isinstance(patch, Circle) for patch in axes[0, 0].patches)
    assert axes[0, 0].xaxis.get_gridlines()[0].get_alpha() == 0.41
    assert grid_path.exists()
