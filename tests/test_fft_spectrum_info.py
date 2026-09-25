from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import zarr

from mmpp.core.job import ZarrJobResult
from mmpp.fft.core import FFT
from mmpp.fft.spectrum.batch.result import BatchSpectrumResult
from mmpp.fft.spectrum.helpers import _SpectrumQuickPlot
from mmpp.fft.spectrum.modes.accessor import SpectrumModesPlotAccessor
from mmpp.fft.spectrum.multi import MultiSpectrumResult
from mmpp.fft.spectrum.result import SpectrumResult


def _spectrum_result(path: str = "/simulations/job.zarr") -> SpectrumResult:
    return SpectrumResult(
        frequencies=np.array([1e9, 2e9, 3e9]),
        spectrum=np.array([1.0 + 0j, 2.0 + 0j, 1.0 + 0j]),
        component_label=r"$m_z$",
        source_job=SimpleNamespace(path=path),
        mode_context={"dset": "m", "z_layer": 2, "slice_info": (Ellipsis, 2)},
        compute_metadata={
            "method": 2,
            "data_shape": (64, 4, 4, 3),
            "dt": 2e-12,
            "frequency_resolution": 1e9 / 128,
            "fft_length": 128,
            "window": "hann",
            "filter_type": "remove_mean",
            "engine_requested": "auto",
            "engine_selected": "numpy",
            "scaling": "raw",
            "zero_padding": True,
            "nfft_requested": None,
            "z_layer": 2,
        },
    )


def _caption(figure) -> str:
    matching = [
        item.get_text() for item in figure.texts if item.get_gid() == "mmpp-fft-info"
    ]
    assert len(matching) == 1
    return matching[0]


def test_spectrum_plot_full_info_uses_recorded_fft_metadata():
    result = _spectrum_result()
    figure, _axes, _peaks = result.plot.spectrum(info="full", show_peaks=False)
    try:
        caption = _caption(figure)
        assert "/simulations/job.zarr" in caption
        assert "dataset=m" in caption
        assert "component=$m_z$" in caption
        assert "z-layer=2" in caption
        assert "FFT method 2: FFT each cell" in caption
        assert "N=64" in caption
        assert "dt=2e-12 s" in caption
        assert "engine=numpy (requested auto)" in caption
        assert "window=hann" in caption
    finally:
        plt.close(figure)


def test_spectrum_plot_rejects_unknown_info_value():
    with pytest.raises(ValueError, match="info must be None or 'full'"):
        _spectrum_result().plot.spectrum(info="brief")


def test_info_shows_resolved_last_z_layer_index():
    result = _spectrum_result()
    result._mode_context["z_layer"] = -1
    result.compute_metadata["z_layer"] = -1
    result.compute_metadata["resolved_z_layer"] = 1
    figure, _axes, _peaks = result.plot.spectrum(info="full", show_peaks=False)
    try:
        assert "z-layer=1 (requested -1)" in _caption(figure)
    finally:
        plt.close(figure)


def test_fft_computation_retains_metadata_for_plot_info(tmp_path):
    path = tmp_path / "spectrum_info.zarr"
    root = zarr.open(str(path), mode="w")
    values = np.random.default_rng(4).normal(size=(32, 1, 3, 3, 3)).astype(np.float32)
    root.create_dataset("m", data=values, chunks=values.shape)
    root.attrs["t_sampl"] = 2e-12
    job = ZarrJobResult(str(path), {})

    result = FFT(job, None).spectrum(
        z_layer=0,
        method=2,
        window="none",
        filter_type="none",
        engine="numpy",
        save=False,
    )

    assert result.compute_metadata["method"] == 2
    assert result.compute_metadata["window"] == "none"
    assert result.compute_metadata["engine_selected"] == "numpy"
    assert result.compute_metadata["z_layer"] == 0
    assert result._mode_context["z_layer"] == 0

    figure, _axes, _peaks = FFT(job, None).spectrum.plot.spectrum(
        z_layer=0,
        method=2,
        window="none",
        filter_type="none",
        engine="numpy",
        save=False,
        info="full",
        show_peaks=False,
    )
    try:
        caption = _caption(figure)
        assert str(path) in caption
        assert "FFT method 2" in caption
        assert "N=32" in caption
    finally:
        plt.close(figure)


def test_quick_spectrum_plot_splits_info_from_compute_options():
    result = _spectrum_result()
    calls = {}

    class FakeHelper:
        def __call__(self, **kwargs):
            calls["compute"] = kwargs
            return result

    figure, _axes, _peaks = _SpectrumQuickPlot(FakeHelper()).spectrum(
        method=2, save=False, info="full", show_peaks=False
    )
    try:
        assert calls["compute"] == {"method": 2, "save": False}
        assert "FFT method 2" in _caption(figure)
    finally:
        plt.close(figure)


def test_multi_spectrum_overlay_can_add_full_info():
    first = _spectrum_result("/simulations/first.zarr")
    second = _spectrum_result("/simulations/second.zarr")
    figure, _axes, _peaks = MultiSpectrumResult([first, second]).plot(info="full")
    try:
        caption = _caption(figure)
        assert "Input files (2):" in caption
        assert "/simulations/first.zarr" in caption
        assert "/simulations/second.zarr" in caption
    finally:
        plt.close(figure)


def test_mode_image_can_show_the_source_spectrum_methodology():
    result = _spectrum_result()
    figure, axes = plt.subplots()
    mode = SimpleNamespace(
        plot=SimpleNamespace(imshow=lambda **kwargs: axes.imshow(np.ones((2, 2))))
    )
    modes = SimpleNamespace(at=lambda **kwargs: mode, _spectrum=result)
    accessor = SpectrumModesPlotAccessor(modes)
    try:
        image = accessor.imshow(f=2.0, component="z", info="full")
        assert image.axes is axes
        assert "FFT method 2" in _caption(figure)
    finally:
        plt.close(figure)


def test_legacy_fft_plotter_adds_info_from_computation_metadata(monkeypatch):
    pytest.importorskip("matplotlib")
    job = SimpleNamespace(path="/simulations/legacy.zarr")
    fft = FFT(job)
    plotter = fft.plotter
    metadata = {
        "method": 1,
        "data_shape": (32, 3),
        "dt": 1e-12,
        "frequency_resolution": 1e9 / 64,
        "fft_length": 64,
        "window": "blackman",
        "filter_type": "none",
        "engine_requested": "numpy",
        "engine_selected": "numpy",
        "scaling": "raw",
        "zero_padding": True,
    }
    monkeypatch.setattr(
        plotter.fft_compute,
        "calculate_fft_data",
        lambda *args, **kwargs: SimpleNamespace(
            frequencies=np.array([0.0, 1e9, 2e9]),
            spectrum=np.array([0.0 + 0j, 1.0 + 0j, 0.5 + 0j]),
            metadata=metadata,
        ),
    )
    figure, _axes = fft.plot_spectrum(
        dset="m",
        save=False,
        method=1,
        info="full",
        log_scale=False,
    )
    try:
        caption = _caption(figure)
        assert "/simulations/legacy.zarr" in caption
        assert "FFT method 1: spatially average magnetization" in caption
        assert "window=blackman" in caption
    finally:
        plt.close(figure)


def _batch_result() -> BatchSpectrumResult:
    return BatchSpectrumResult(
        frequencies=np.array([1e9, 2e9, 3e9]),
        spectra=[
            np.array([1.0 + 0j, 2.0 + 0j, 1.0 + 0j]),
            np.array([2.0 + 0j, 3.0 + 0j, 2.0 + 0j]),
        ],
        powers=[
            np.array([1.0, 4.0, 1.0]),
            np.array([4.0, 9.0, 4.0]),
        ],
        parameters={"theta": [0, 90]},
        job_paths=["/simulations/theta_0.zarr", "/simulations/theta_90.zarr"],
        config_dict={
            "method": 2,
            "window_function": "hann",
            "filter_type": "remove_mean",
            "engine": "auto",
            "resample_nonuniform": True,
            "component_weights": (1, 0, 0),
        },
        dataset_name="m",
        z_layer=-1,
    )


def test_batch_heatmap_and_sweeps_add_retained_full_info():
    batch = _batch_result()
    figure, _axes = batch.plot_heatmap(parameter="theta", log_scale=False, info="full")
    try:
        caption = _caption(figure)
        assert "Input files (2):" in caption
        assert "dataset=m" in caption
        assert "component=weighted components (weights=(1, 0, 0))" in caption
        assert "FFT method 2" in caption
        assert "engine=auto" in caption
        assert "Sampling metadata" in caption
    finally:
        plt.close(figure)

    with plt.rc_context({"figure.constrained_layout.use": True}):
        plots = batch.plot_sweeps(parameters="theta", info="full", log_scale=False)
    figure, _axes = plots["theta"]
    try:
        assert "FFT method 2" in _caption(figure)
    finally:
        plt.close(figure)


def test_batch_entry_plot_adds_full_info():
    figure, _axes = _batch_result()[0].plot(info="full", log_scale=False)
    try:
        caption = _caption(figure)
        assert "/simulations/theta_0.zarr" in caption
        assert "dataset=m" in caption
        assert "FFT method 2" in caption
    finally:
        plt.close(figure)
