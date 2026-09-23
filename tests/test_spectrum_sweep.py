"""Tests for automatic sweep-axis discovery and spectrum comparisons."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from mmpp.batch_operations import BatchFFT
from mmpp.core.job import ZarrJobResult
from mmpp.fft.core import FFT
from mmpp.fft.spectrum.batch.compute import BatchSpectrum
from mmpp.fft.spectrum.batch.result import (
    BatchSpectrumAnalysis,
    BatchSpectrumResult,
)
from mmpp.fft.spectrum.batch.sweep import discover_sweep_parameters


def test_batch_fft_helper_documents_auto_sweep_spectrum_workflow():
    html = BatchFFT([], None)._repr_html_()

    assert "analyze" in html
    assert "plot_sweeps" in html
    assert "plot_heatmap" in html
    assert "theta" in html and "t_sl" in html and "phi" in html
    assert "method=2" in html
    assert "average power" in html
    assert "ACCESSORS &amp; METHODS" in html
    assert ">Overview</button>" in html and ">API</button>" in html
    assert "linear-gradient(135deg, #282a36" in html
    assert "box-shadow: 0 10px" in html and "border: 2px solid #6272a4" in html
    assert html.index("<button") < html.index("Batch FFT Interface")
    assert "<h3" not in html


def test_analyze_binds_sweep_selection_and_forwards_compute_options(monkeypatch):
    frequencies = np.array([1e9, 2e9, 3e9])
    spectra = [np.full(3, index + 1, dtype=complex) for index in range(4)]
    powers = [np.full(3, index + 1, dtype=float) for index in range(4)]
    result = BatchSpectrumResult(
        frequencies=frequencies,
        spectra=spectra,
        powers=powers,
        parameters={"theta": [0, 0, 90, 90], "phi": [0, 45, 0, 45]},
        job_paths=[f"job_{index}.zarr" for index in range(4)],
    )
    analyzer = BatchSpectrum([], None)
    captured = {}

    def fake_compute_all(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(analyzer, "compute_all", fake_compute_all)
    analysis = analyzer.analyze("theta", fmin=5e9, parallel=False, method=2)

    assert isinstance(analysis, BatchSpectrumAnalysis)
    assert analysis.result is result
    assert analysis.parameters is result.parameters
    assert analysis.sweep_parameters == "theta"
    assert captured == {"fmin": 5e9, "parallel": False, "method": 2}

    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    sweep_plots = analysis.plot_sweeps(log_scale=False, colorbar=False)
    assert set(sweep_plots) == {"theta"}
    figure, axes = sweep_plots["theta"]
    assert axes.shape == (1, 2)
    plt.close(figure)


def test_discovers_declared_sweep_axes_and_computes_their_values(tmp_path, monkeypatch):
    (tmp_path / "run_metadata.json").write_text(
        json.dumps(
            {
                "parameter_ranges": {
                    "t_sl": {"values": [1e-9, 2e-9]},
                    "theta": {"values": [0, 90]},
                    "phi": {"values": [0, 45, 90]},
                }
            }
        ),
        encoding="utf-8",
    )
    results = [
        SimpleNamespace(
            path=str(tmp_path / f"job_{index}.zarr"),
            attributes={
                "t_sl": t_sl,
                "theta": theta,
                "phi": phi,
                "frequency_shift": 3e-7 if index == 1 else 0.0,
            },
        )
        for index, (t_sl, theta, phi) in enumerate(
            [(1e-9, 0, 0), (1e-9, 0, 45), (2e-9, 90, 90)]
        )
    ]
    mmpp_ref = SimpleNamespace(base_path=str(tmp_path))

    assert discover_sweep_parameters(results, mmpp_ref) == ["t_sl", "theta", "phi"]

    class FakeFFT:
        def __init__(self, result, _mmpp_ref):
            self.result = result

        def spectrum(self, **_kwargs):
            assert _kwargs["method"] == 2
            assert _kwargs["resample_nonuniform"] is True
            return SimpleNamespace(
                frequencies=np.array([0.0, 1e9, 2e9, 3e9])
                * (1.0 - self.result.attributes["frequency_shift"]),
                spectrum=np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j]),
                spectral_quantity=np.array([1.0, 4.0, 9.0, 16.0]),
            )

    import mmpp.fft.core as fft_core

    monkeypatch.setattr(fft_core, "FFT", FakeFFT)
    batch = BatchSpectrum(results, mmpp_ref).compute_all(
        method=2,
        parallel=False,
        save=False,
        save_batch=False,
        use_cache=False,
        extract_parameters="auto",
    )

    assert batch.parameters == {
        "t_sl": [1e-9, 1e-9, 2e-9],
        "theta": [0, 0, 90],
        "phi": [0, 45, 90],
    }
    assert batch.frequencies.tolist() == [0.0, 1e9, 2e9]
    assert all(power.shape == batch.frequencies.shape for power in batch.powers)


def test_generic_batch_compute_all_forwards_selected_fft_method(monkeypatch):
    calls = []

    class FakeFFT:
        def __init__(self, _result, _mmpp_ref):
            pass

        def _compute_fft(self, **kwargs):
            calls.append(kwargs)

    import mmpp.batch_operations as batch_module

    monkeypatch.setattr(batch_module, "_resolve_fft_class", lambda *_: FakeFFT)
    batch = BatchFFT([SimpleNamespace(path="job.zarr")], object())

    summary = batch.compute_all(method=2, save=False)

    assert summary["successful"] == 1
    assert summary["failed"] == 0
    assert calls == [{"method": 2, "save": False}]


def test_method2_batch_heatmap_reduces_component_axis_before_plotting(
    tmp_path, monkeypatch
):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    frequencies = np.arange(4, dtype=float) * 1e9
    results = [
        SimpleNamespace(
            path=str(tmp_path / f"theta_{theta}.zarr"),
            attributes={"theta": theta},
        )
        for theta in (0, 45, 90)
    ]

    class FakeFFT:
        def __init__(self, result, _mmpp_ref):
            self.result = result

        def spectrum(self, **kwargs):
            assert kwargs["method"] == 2
            scale = self.result.attributes["theta"] / 45 + 1
            spectrum = np.broadcast_to(
                np.array([1.0, 2.0, 3.0], dtype=complex) * scale,
                (frequencies.size, 3),
            )
            return SimpleNamespace(
                frequencies=frequencies,
                spectrum=spectrum,
                spectral_quantity=np.abs(spectrum) ** 2,
            )

    import mmpp.fft.core as fft_core

    monkeypatch.setattr(fft_core, "FFT", FakeFFT)
    batch = BatchSpectrum(
        results, SimpleNamespace(base_path=str(tmp_path))
    ).compute_all(
        method=2,
        parallel=False,
        save=False,
        save_batch=False,
        use_cache=False,
        extract_parameters=["theta"],
    )

    assert all(power.shape == frequencies.shape for power in batch.powers)
    np.testing.assert_allclose(batch.powers[0], np.full(4, 14 / 3))
    saved_path = tmp_path / "method2_batch.zarr"
    batch.save(saved_path)
    restored = BatchSpectrumResult.load(saved_path)
    assert all(power.shape == frequencies.shape for power in restored.powers)
    np.testing.assert_allclose(restored.powers[0], batch.powers[0])

    fig, ax = batch.plot_heatmap(parameter="theta", log_scale=False, colorbar=False)
    assert ax.images[0].get_array().shape == (4, 3)
    plt.close(fig)


def test_discovers_sweep_axes_from_nested_result_paths(tmp_path):
    results = [
        SimpleNamespace(
            path=str(tmp_path / f"t_sl_{t_sl}" / f"theta_{theta}" / f"phi_{phi}.zarr"),
            attributes={},
        )
        for t_sl, theta, phi in (
            ("5e-10", 0, 0),
            ("5e-10", 90, 0),
            ("1e-9", 0, 45),
            ("1e-9", 90, 45),
        )
    ]
    mmpp_ref = SimpleNamespace(base_path=str(tmp_path))

    assert discover_sweep_parameters(results, mmpp_ref) == ["t_sl", "theta", "phi"]


def test_plot_sweeps_facets_by_remaining_parameters():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    frequencies = np.array([1.0, 2.0, 3.0])
    spectra = []
    powers = []
    parameters = {"t_sl": [], "theta": []}
    job_paths = []
    index = 0
    for theta in (0, 90):
        for t_sl in (1e-9, 2e-9, 3e-9):
            parameters["t_sl"].append(t_sl)
            parameters["theta"].append(theta)
            spectra.append(np.full(3, index + 1, dtype=complex))
            powers.append(np.full(3, index + 1, dtype=float))
            job_paths.append(f"job_{index}.zarr")
            index += 1
    batch = BatchSpectrumResult(
        frequencies=frequencies,
        spectra=spectra,
        powers=powers,
        parameters=parameters,
        job_paths=job_paths,
    )

    figures = batch.plot_sweeps(log_scale=False, colorbar=False, max_columns=3)

    assert set(figures) == {"t_sl", "theta"}
    time_figure, time_axes = figures["t_sl"]
    assert time_axes.shape == (1, 2)
    assert all(axis.images[0].get_array().shape == (3, 3) for axis in time_axes.flat)
    assert all("theta=" in axis.get_title() for axis in time_axes.flat)

    angle_figure, angle_axes = figures["theta"]
    assert angle_axes.shape == (1, 3)
    assert all(axis.images[0].get_array().shape == (3, 2) for axis in angle_axes.flat)
    assert all("t_sl=" in axis.get_title() for axis in angle_axes.flat)
    plt.close(time_figure)
    plt.close(angle_figure)


def test_fft_resamples_nonuniform_time_axis_when_requested(tmp_path):
    nt = 128
    nominal_time = np.arange(nt, dtype=float) * 2e-11
    time_axis = nominal_time.copy()
    time_axis[1:-1] += (
        0.1
        * (nominal_time[1] - nominal_time[0])
        * np.sin(np.linspace(0.0, 6.0 * np.pi, nt - 2))
    )
    signal = np.cos(2.0 * np.pi * 1.0e9 * time_axis).astype(np.float32)
    data = np.broadcast_to(signal[:, None, None, None, None], (nt, 1, 3, 3, 3))
    path = tmp_path / "nonuniform.zarr"
    root = zarr.open(str(path), mode="w")
    magnetization = root.create_dataset("m", data=data, chunks=data.shape)
    magnetization.attrs["t"] = time_axis.tolist()
    job = ZarrJobResult(str(path), {})

    with pytest.raises(ValueError, match="uniformly sampled time axis"):
        FFT(job, None).spectrum(z_layer=0, window="none", filter_type="none")

    spectrum = FFT(job, None).spectrum(
        z_layer=0,
        window="none",
        filter_type="none",
        resample_nonuniform=True,
    )

    assert spectrum.frequencies.size == nt // 2 + 1
    assert spectrum.spectrum.shape[0] == nt // 2 + 1
    expected_dt = float(np.mean(np.diff(time_axis)))
    assert spectrum.frequencies[-1] == pytest.approx(1.0 / (2.0 * expected_dt))

    truncated = FFT(job, None).spectrum(
        z_layer=0,
        window="none",
        filter_type="none",
        tmax=64,
        resample_nonuniform=True,
    )
    truncated_dt = float(np.mean(np.diff(time_axis[:64])))
    assert truncated.frequencies.size == 64 // 2 + 1
    assert truncated.frequencies[-1] == pytest.approx(1.0 / (2.0 * truncated_dt))
