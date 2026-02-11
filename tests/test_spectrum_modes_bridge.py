from types import SimpleNamespace

import numpy as np

import mmpp.fft.modes.interface as modes_interface_mod
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
