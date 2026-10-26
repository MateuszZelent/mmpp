"""Regression tests for committed Zarr analysis cache behavior."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np


def test_forced_spectrum_computation_refreshes_memory_cache(tmp_path):
    from mmpp.core.zarr_integrity import open_source
    from mmpp.fft.compute_fft import FFTComputeResult
    from mmpp.fft.spectrum.compute import compute_fft_cached
    from tests.test_committed_zarr_store import _make_committed_store

    class Engine:
        def __init__(self):
            self.calls = 0

        def calculate_fft_data(self, *args, **kwargs):
            self.calls += 1
            return FFTComputeResult(
                frequencies=np.array([0.0]),
                spectrum=np.array([float(self.calls)]),
                metadata={},
                config=SimpleNamespace(),
            )

    engine = Engine()
    cache = {}
    path, _, _ = _make_committed_store(tmp_path)
    job = SimpleNamespace(
        path=str(path), z=open_source(str(path)), get_largest_m_dataset=lambda: "m"
    )

    first = compute_fft_cached(compute_engine=engine, job_result=job, cache=cache)
    forced = compute_fft_cached(
        compute_engine=engine, job_result=job, cache=cache, force=True
    )
    cached = compute_fft_cached(compute_engine=engine, job_result=job, cache=cache)

    assert engine.calls == 2
    assert first.spectrum[0] == 1.0
    assert forced.spectrum[0] == 2.0
    assert cached is forced


def test_spectrum_without_source_revision_does_not_reuse_memory_cache():
    from mmpp.fft.spectrum.compute import compute_fft_cached

    calls = []
    engine = SimpleNamespace(
        calculate_fft_data=lambda *args, **kwargs: calls.append(kwargs) or object()
    )
    job = SimpleNamespace(path="unversioned.zarr", get_largest_m_dataset=lambda: "m")
    cache = {}

    compute_fft_cached(compute_engine=engine, job_result=job, cache=cache)
    compute_fft_cached(compute_engine=engine, job_result=job, cache=cache)

    assert len(calls) == 2
    assert cache == {}


def test_loaded_fft_cache_preserves_scaling_metadata(tmp_path):
    from mmpp.core.zarr_integrity import source_revision
    from mmpp.fft._compute_cache import load_existing_fft_result
    from mmpp.fft.compute_fft import FFTComputeConfig, FFTComputeResult
    from tests.test_committed_zarr_store import _make_committed_store

    path, _, _ = _make_committed_store(tmp_path)
    FFTComputeResult(
        frequencies=np.array([0.0, 1.0]),
        spectrum=np.array([1.0, 2.0]),
        metadata={
            "source_dataset": "m",
            "source_revision": source_revision(str(path), "m"),
        },
        config=FFTComputeConfig(scaling="psd"),
    ).save_to_zarr(str(path), "m")

    result = load_existing_fft_result(
        zarr_path=str(path),
        dataset_name="m",
        result_cls=FFTComputeResult,
        config_cls=FFTComputeConfig,
        psutil_module=None,
        logger=logging.getLogger("test"),
    )

    assert result is not None
    assert result.config.scaling == "psd"
    assert result.metadata["scaling"] == "psd"


def test_incomplete_legacy_fft_cache_is_not_loaded(tmp_path):
    import zarr

    from mmpp.fft._compute_cache import load_existing_fft_result
    from mmpp.fft.compute_fft import FFTComputeConfig, FFTComputeResult

    path = tmp_path / "legacy.zarr"
    root = zarr.open(str(path), mode="w")
    entry = root.create_group("fft").create_group("m")
    entry.create_dataset("spectrum", data=np.array([1.0, 2.0]))
    entry.create_dataset("frequencies", data=np.array([0.0, 1.0]))
    entry.attrs["scaling"] = "psd"

    assert (
        load_existing_fft_result(
            zarr_path=str(path),
            dataset_name="m",
            result_cls=FFTComputeResult,
            config_cls=FFTComputeConfig,
            psutil_module=None,
            logger=logging.getLogger("test"),
        )
        is None
    )


def test_transmission_cache_is_source_scoped_and_omits_callbacks(tmp_path):
    from mmpp.fft.transmission.cache import TransmissionCache
    from mmpp.fft.transmission.compute import TransmissionConfig, TransmissionResult
    from tests.test_committed_zarr_store import _make_committed_store

    cache_path = tmp_path / "shared-cache"
    source_a, _, _ = _make_committed_store(tmp_path)
    other_parent = tmp_path / "other"
    other_parent.mkdir()
    source_b, _, _ = _make_committed_store(other_parent)

    def callback(progress, stage):
        return None

    config = TransmissionConfig(dataset_name="m", progress_callback=callback)
    result = TransmissionResult(
        frequencies=np.array([0.0, 1.0]),
        x_positions=np.array([0.0]),
        transmission=np.ones((2, 1)),
        power_map=np.ones((2, 1)),
        reference_power=np.ones(2),
        config=config,
    )
    source_cache = TransmissionCache(
        SimpleNamespace(path=str(source_a)), dataset_name="m"
    )
    source_cache.save_result(result, cache_path=cache_path)

    loaded = source_cache.load_result(config, cache_path=cache_path)
    other_source = TransmissionCache(
        SimpleNamespace(path=str(source_b)), dataset_name="m"
    )

    assert loaded is not None
    assert loaded.config.progress_callback is None
    assert other_source.load_result(config, cache_path=cache_path) is None


def test_mode_cache_key_includes_window_and_failed_force_preserves_cache(tmp_path):
    import pytest
    import zarr

    from mmpp.fft.modes import FMRModeAnalyzer
    from tests.test_committed_zarr_store import _make_committed_store

    path, _, _ = _make_committed_store(tmp_path)
    root = zarr.open(str(path), mode="a")
    fft_group = root.create_group("fft").create_group("m")
    fft_group.create_dataset("spectrum", data=np.array([42.0]))

    analyzer = FMRModeAnalyzer(str(path), dataset_name="m")
    analyzer.compute_modes(window=False, save=True, force=True)
    unwindowed = np.asarray(analyzer.zarr_file["modes/m/arr"]).copy()

    analyzer.compute_modes(window=True, save=True)
    windowed = np.asarray(analyzer.zarr_file["modes/m/arr"]).copy()
    assert analyzer.zarr_file["modes/m"].attrs["window_applied"] is True
    assert not np.allclose(windowed, unwindowed)

    with pytest.raises(ValueError, match="at least two"):
        analyzer.compute_modes(
            window=False,
            t_slice=slice(0, 1),
            save=True,
            force=True,
        )

    assert np.array_equal(np.asarray(analyzer.zarr_file["modes/m/arr"]), windowed)
    assert np.array_equal(np.asarray(root["fft/m/spectrum"]), np.array([42.0]))
