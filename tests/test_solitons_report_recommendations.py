from __future__ import annotations

import numpy as np

from mmpp.batch_operations import BatchOperations
from mmpp.solitons.batch import (
    BatchSolitonsInterface,
    BatchVortexInterface,
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

    def gyration(self, **kwargs) -> VortexSpectrumResult:
        return self._result

    def breathing(self, **kwargs) -> VortexSpectrumResult:
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


def test_batch_vortex_interactive_displays_single_selected_result() -> None:
    calls: list[tuple[str, tuple[float, float], int]] = []

    class _InteractiveVortex:
        def __init__(self, label: str):
            self.label = label

        def interactive(self, *, figsize=(10, 7), dpi=100):
            calls.append((self.label, figsize, dpi))
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

    dashboard = interface.interactive(index=0, figsize=(8, 5), dpi=150)

    assert dashboard == "dashboard-low"
    assert calls == [("low", (8, 5), 150)]


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
