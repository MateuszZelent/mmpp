from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import zarr

import mmpp.fft.modes.interface as modes_interface_mod
from mmpp.core.job import ZarrJobResult
from mmpp.fft.modes.interface import FFTModeInterfaceNew
from mmpp.fft.spectrum.result import SpectrumResult
from mmpp.fft.transmission.compute import TransmissionConfig, TransmissionResult


def test_spectrum_modes_bridge_interactive_passes_existing_spectrum_result():
    class _FakeModesInterface:
        def __init__(self):
            self._dataset_context = None
            self._slice_context = None
            self.calls = []

        def interactive_spectrum(self, **kwargs):
            self.calls.append(kwargs)
            return "interactive-ok"

    fake_modes = _FakeModesInterface()
    fake_fft = SimpleNamespace(modes=fake_modes)

    spec = SpectrumResult(
        frequencies=np.array([1.0, 2.0]),
        spectrum=np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        source_fft=fake_fft,
        mode_context={"dset": "m_layer13", "slice_info": (slice(0, 100), Ellipsis, 2)},
    )

    out = spec.plot.interactive(show=False)

    assert out == "interactive-ok"
    assert len(fake_modes.calls) == 1
    assert fake_modes.calls[0]["spectrum_result"] is spec
    assert fake_modes.calls[0]["show"] is False
    assert fake_modes._dataset_context == "m_layer13"
    assert fake_modes._slice_context == (slice(0, 100), Ellipsis, 2)


def test_interactive_impl_uses_provided_spectrum_result_without_recomputing(monkeypatch):
    class _DummySpectrumResult:
        def __init__(self):
            self.frequencies = np.linspace(1.0, 30.0, 128)
            self.power = np.ones_like(self.frequencies)
            self.component_label = "$m_z$"

    class _CountingFFT:
        def __init__(self):
            self.job_result = SimpleNamespace(path="/tmp/dummy.zarr")
            self.calls = 0

        def _spectrum_impl(self, **_kwargs):
            self.calls += 1
            return _DummySpectrumResult()

    class _ViewerStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def show(self, **_kwargs):
            return self.kwargs["spectrum_result"]

    monkeypatch.setattr(modes_interface_mod, "_get_interactive", lambda: _ViewerStub)

    fft = _CountingFFT()
    interface = FFTModeInterfaceNew(0, fft)
    interface._mode_analyzer = SimpleNamespace(modes_available=True)
    provided = _DummySpectrumResult()

    out = interface._interactive_spectrum_impl(
        toolbar=True,
        show=False,
        spectrum_result=provided,
    )

    assert out is provided
    assert fft.calls == 0


def _create_fft_job(tmp_path):
    zarr_path = tmp_path / "fft_helpers_bridge.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    data = np.random.randn(40, 1, 4, 4, 3).astype(np.float32)
    z.create_dataset("m", data=data, chunks=data.shape)
    z.attrs["t_sampl"] = 1e-12
    return ZarrJobResult(str(zarr_path), {})


def test_dataset_fft_help_helpers_delegate_with_slice_context(tmp_path):
    job = _create_fft_job(tmp_path)
    data = job.m[:20, ..., 2]

    spec = data.fft.help.spectrum()
    assert spec.component_label == r"$m_z$"
    assert getattr(spec, "_single_component", False) is True

    filtered = data.fft.help.filters(remove_static=True).spectrum()
    assert filtered.component_label == r"$m_z$"
    assert getattr(filtered, "_single_component", False) is True


def test_modes_help_filters_helper_preserves_configuration():
    parent_fft = SimpleNamespace(job_result=SimpleNamespace(path="/tmp/dummy.zarr"))
    modes = FFTModeInterfaceNew(0, parent_fft).configure(
        tmax=321,
        filters={"post": {"normalize": True}},
        cache_dir="/tmp/cache",
    )

    clone = modes.help.filters(freq_min=1.5, normalize=True)
    assert clone._tmax == 321
    assert clone._filters_config == {"post": {"normalize": True}}
    assert clone._cache_dir == "/tmp/cache"
    assert clone._interactive_filters["freq_min"] == 1.5
    assert clone._interactive_filters["normalize"] is True


def test_transmission_and_dispersion_help_accessors_are_exposed(tmp_path):
    job = _create_fft_job(tmp_path)

    t_help = job.fft.transmission.help
    assert callable(t_help.compute)
    assert callable(t_help.plot_transmission)
    assert callable(t_help.visualize_mode)
    assert callable(t_help.visualize_modes)

    d_help = job.fft.dispersion.help
    assert callable(d_help.compute_1d)
    assert callable(d_help.compute_2d)
    assert hasattr(d_help, "filters")


def test_transmission_result_visualize_mode_reconstructs_frequency_bin():
    n_time = 16
    dt = 1e-12
    freqs = np.fft.rfftfreq(n_time, d=dt)

    raw_fft = np.zeros((freqs.size, 1, 2, 5, 3), dtype=np.complex128)
    raw_fft[3, 0, :, :, 2] = 1.0 + 0.5j

    result = TransmissionResult(
        frequencies=freqs,
        x_positions=np.arange(5, dtype=float),
        transmission=raw_fft,
        power_map=np.abs(raw_fft),
        reference_power=np.ones(freqs.size, dtype=float),
        config=TransmissionConfig(raw_fft_output=True, y_integration_mode="none"),
        dx=1e-9,
        metadata={"raw_fft_output": True, "n_time": n_time, "time_step": dt},
    )

    fig, ax, meta = result.visualize_mode(
        f=float(freqs[3] * 1e-9),
        freq_unit="GHz",
        component="z",
        copy_y=2,
        colorbar=False,
        x_lines=[1.0, 3.0],
        x_lines_in_index=True,
    )

    assert ax is not None
    assert meta["k"] == 3
    assert meta["xy"].shape == (4, 5)
    assert np.isfinite(meta["xy"]).all()
    assert ax.xaxis_inverted()
    plt.close(fig)


def test_transmission_help_visualize_mode_works_with_precomputed_result(tmp_path):
    job = _create_fft_job(tmp_path)
    result = job.fft.transmission(
        raw_fft_output=True,
        y_integration_mode="none",
        normalize="none",
        average_mode="none",
        spatial_window=1,
        spatial_step=1,
        use_cache=False,
    )

    target_ghz = float(result.frequencies[1] * 1e-9)
    fig, ax, meta = job.fft.transmission.help.visualize_mode(
        f=target_ghz,
        result=result,
        colorbar=False,
    )

    assert ax is not None
    assert isinstance(meta["k"], int)
    assert np.isfinite(meta["xy"]).all()
    assert ax.xaxis_inverted()
    plt.close(fig)


def test_transmission_raw_fft_forces_post_fft_when_pre_fft_requested(tmp_path):
    job = _create_fft_job(tmp_path)

    result = job.fft.transmission(
        raw_fft_output=True,
        spatial_window_mode="pre_fft",
        y_integration_mode="none",
        use_cache=False,
        force=True,
    )

    assert result.config.raw_fft_output is True
    assert result.config.spatial_window_mode == "post_fft"
    assert result.metadata.get("raw_fft_output") is True
    assert np.iscomplexobj(result.transmission)


def test_transmission_visualize_mode_single_component_accepts_z_alias():
    n_time = 12
    dt = 1e-12
    freqs = np.fft.rfftfreq(n_time, d=dt)
    raw_fft = np.zeros((freqs.size, 1, 3, 4, 1), dtype=np.complex128)
    raw_fft[2, 0, :, :, 0] = 1.0 + 0.0j

    result = TransmissionResult(
        frequencies=freqs,
        x_positions=np.arange(4, dtype=float),
        transmission=raw_fft,
        power_map=np.abs(raw_fft),
        reference_power=np.ones(freqs.size, dtype=float),
        config=TransmissionConfig(raw_fft_output=True, y_integration_mode="none"),
        dx=1e-9,
        metadata={"raw_fft_output": True, "n_time": n_time, "time_step": dt},
    )

    fig, ax, meta = result.visualize_mode(
        f=float(freqs[2] * 1e-9),
        freq_unit="GHz",
        component="z",
        colorbar=False,
    )
    assert ax is not None
    assert meta["component_index"] == 0
    assert meta["xy"].shape == (3, 4)
    assert ax.xaxis_inverted()
    plt.close(fig)


def test_transmission_calculate_modes_and_visualize_workflow():
    n_time = 20
    dt = 1e-12
    freqs = np.fft.rfftfreq(n_time, d=dt)
    raw_fft = np.zeros((freqs.size, 1, 3, 6, 1), dtype=np.complex128)
    raw_fft[2, 0, :, :, 0] = 1.0 + 0.2j
    raw_fft[4, 0, :, :, 0] = 0.6 + 0.3j
    raw_fft[5, 0, :, :, 0] = 0.2 + 0.8j

    result = TransmissionResult(
        frequencies=freqs,
        x_positions=np.arange(6, dtype=float),
        transmission=raw_fft,
        power_map=np.abs(raw_fft),
        reference_power=np.ones(freqs.size, dtype=float),
        config=TransmissionConfig(raw_fft_output=True, y_integration_mode="none"),
        dx=2e-9,
        metadata={"raw_fft_output": True, "n_time": n_time, "time_step": dt},
    )

    modes = result.calculate_modes(
        f=[float(freqs[2] * 1e-9), float(freqs[4] * 1e-9), float(freqs[5] * 1e-9)],
        component=0,
        t_show=0,
    )
    assert len(modes) == 3
    assert modes.modes[0]["xy_complex"].shape == (3, 6)

    fig, axes, metas = modes.visualize(
        mode="real",
        colorbar=False,
        y_lines=[1],
        y_spans=[(0.2, 1.8)],
    )
    assert len(axes) == 3
    assert len(metas) == 3
    assert metas[0]["xy"].shape == (3, 6)
    assert axes[0].xaxis_inverted()
    plt.close(fig)

    fig_single, ax_single, meta_single = modes.visualize(
        index=1,
        colorbar=False,
        flip_x=False,
    )
    assert ax_single is not None
    assert meta_single["xy"].shape == (3, 6)
    assert not ax_single.xaxis_inverted()
    plt.close(fig_single)
