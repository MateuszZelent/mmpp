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


def test_vortex_dashboard_show_figure_uses_output_widget_not_global_display(
    monkeypatch,
) -> None:
    import matplotlib.pyplot as plt

    from mmpp.solitons.vortex.ui import interactive_dashboard as dashboard_module

    class _Output:
        def __init__(self):
            self.clear_calls = 0
            self.displayed: list[object] = []

        def clear_output(self, wait=False):
            self.clear_calls += 1

        def append_display_data(self, value):
            self.displayed.append(value)

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
    dashboard._fig = None

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    dashboard._show_figure(fig)

    assert output.clear_calls == 1
    assert len(output.displayed) == 1
    assert global_display_calls == []


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
                    (".fft", "FFT namespace", "#34d399"),
                    (".analyze", "analysis tools", "#34d399"),
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
    assert (
        "background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)"
        in html
    )
    assert "box-shadow: 0 10px 25px rgba(0,0,0,0.3)" in html
    assert "font-family: 'Courier New', monospace" in html
    assert "datasets" in html
    assert "ACCESSORS &amp; METHODS" in html
    assert ".fft" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "demo-helper-panel-1" in html


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
    assert (
        "background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)"
        in html
    )
    assert "box-shadow: 0 10px 25px rgba(0,0,0,0.3)" in html
    assert "font-family: 'Courier New', monospace" in html
    assert "cache entries" in html
    assert "ACCESSORS &amp; METHODS" in html
    assert ".spectrum()" in html
    assert ".modes" in html
    assert ".dispersion" in html
    assert ".transmission" in html
    assert ">Overview</button>" in html
    assert ">API</button>" in html
    assert "fft-helper-panel-1" in html


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
