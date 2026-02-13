from types import SimpleNamespace

import numpy as np
import zarr

import mmpp.fft.modes.interface as modes_interface_mod
from mmpp.core.job import ZarrJobResult
from mmpp.fft.modes.interface import FFTModeInterfaceNew
from mmpp.fft.spectrum.result import SpectrumResult


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

    d_help = job.fft.dispersion.help
    assert callable(d_help.compute_1d)
    assert callable(d_help.compute_2d)
    assert hasattr(d_help, "filters")
