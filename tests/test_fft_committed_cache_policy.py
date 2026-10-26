"""Regression tests for FFT cache policy on committed MuMax3 sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _patch_fft_engine(monkeypatch, calls: list[dict[str, object]]) -> None:
    """Keep the real FFT orchestration while making the numerical step tiny."""

    import mmpp.fft.compute_fft as compute_module

    monkeypatch.setattr(compute_module, "normalize_z_layer_index", lambda **_: 0)
    monkeypatch.setattr(compute_module, "log_input_load_metrics", lambda **_: None)

    def load_input(**kwargs):
        source_group = kwargs["source_group"]
        dataset = kwargs["dataset"]
        source_array = source_group[dataset]
        data = np.asarray(source_array[:])
        revision = source_group.store._field(dataset)["revision"]
        calls.append(
            {
                "group": source_group,
                "revision": revision,
                "steps": data.shape[0],
            }
        )
        return (
            data,
            1.0,
            SimpleNamespace(
                spatial_axes=(1, 2, 3),
                component_axis=4,
                source_revision=revision,
            ),
        )

    monkeypatch.setattr(compute_module, "load_fft_input_data_profiled", load_input)

    def calculate_method1(
        self,
        data,
        dt,
        window,
        filter_type,
        engine,
        scaling,
        zero_padding,
        nfft,
        **kwargs,
    ):
        return compute_module.FFTComputeResult(
            frequencies=np.array([0.0, 1.0]),
            spectrum=np.array([float(data.shape[0]), float(len(calls))]),
            metadata={},
            config=compute_module.FFTComputeConfig(
                window_function=window,
                filter_type=filter_type,
                fft_engine=engine,
                scaling=scaling,
                zero_padding=zero_padding,
                nfft=nfft,
            ),
        )

    monkeypatch.setattr(
        compute_module.FFTCompute,
        "calculate_fft_method1",
        calculate_method1,
    )


def _fft_request(*, engine, job, cache, **kwargs):
    from mmpp.fft.spectrum.compute import compute_fft_cached

    return compute_fft_cached(
        compute_engine=engine,
        job_result=job,
        cache=cache,
        dataset_name="m",
        z_layer=0,
        method=1,
        window="none",
        filter_type="none",
        engine="numpy",
        zero_padding=False,
        **kwargs,
    )


def _append_committed_step(path: Path, values: np.ndarray, timestamp: float) -> None:
    """Publish a third frame while leaving the ordinary projection unchanged."""

    from tests.test_committed_zarr_store import _json_bytes, _write_frame

    commit_path = path / "m" / "commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    previous = hashlib.sha256(
        (path / "m" / ".frames" / "00000000000000000001.json").read_bytes()
    ).hexdigest()

    _write_frame(path, 2, values, timestamp, "")
    frame_path = path / "m" / ".frames" / "00000000000000000002.json"
    frame = json.loads(frame_path.read_text(encoding="utf-8"))
    frame["previous_sha256"] = previous
    frame_bytes = _json_bytes(frame)
    frame_path.write_bytes(frame_bytes)

    commit["generation"] = 3
    commit["committed_steps"] = 3
    commit["metadata"]["shape"][0] = 3
    commit["last_record_sha256"] = hashlib.sha256(frame_bytes).hexdigest()
    commit_path.write_bytes(_json_bytes(commit))


def test_use_cache_false_skips_memory_and_disk_but_still_saves(
    monkeypatch, tmp_path: Path
):
    from mmpp.cache.transactions import open_published_cache
    from mmpp.core.job import ZarrJobResult
    from mmpp.fft.compute_fft import FFTCompute
    from tests.test_committed_zarr_store import _make_committed_store

    path, _, _ = _make_committed_store(tmp_path)
    job = ZarrJobResult(str(path), {})
    calls: list[dict[str, object]] = []
    _patch_fft_engine(monkeypatch, calls)
    engine = FFTCompute()
    monkeypatch.setattr(engine, "load_existing_fft_data", lambda *args, **kwargs: None)
    cache: dict[str, object] = {}

    first = _fft_request(engine=engine, job=job, cache=cache, use_cache=True)
    assert len(calls) == 1
    assert len(cache) == 1

    def unexpected_disk_read(*args, **kwargs):
        raise AssertionError("use_cache=False attempted a disk-cache read")

    monkeypatch.setattr(engine, "load_existing_fft_data", unexpected_disk_read)
    uncached = _fft_request(
        engine=engine,
        job=job,
        cache=cache,
        use_cache=False,
        save=False,
    )

    assert uncached is not first
    assert len(calls) == 2
    assert len(cache) == 1

    saved = _fft_request(
        engine=engine,
        job=job,
        cache=cache,
        use_cache=False,
        save=True,
        force=True,
        save_dataset_name="policy",
    )
    assert saved is not uncached
    assert len(calls) == 3
    assert len(cache) == 1

    revision = job.z.store._field("m")["revision"]
    persisted = open_published_cache(path / "fft", "policy", source_revision=revision)
    assert persisted is not None
    assert persisted.attrs["source_dataset"] == "m"
    np.testing.assert_array_equal(np.asarray(persisted["spectrum"]), saved.spectrum)


def test_fft_memory_cache_isolated_by_commit_and_pinned_job_snapshot(
    monkeypatch, tmp_path: Path
):
    from mmpp.core.job import ZarrJobResult
    from mmpp.fft.compute_fft import FFTCompute
    from tests.test_committed_zarr_store import _make_committed_store

    path, values, _ = _make_committed_store(tmp_path)
    old_job = ZarrJobResult(str(path), {})
    calls: list[dict[str, object]] = []
    _patch_fft_engine(monkeypatch, calls)
    engine = FFTCompute()
    monkeypatch.setattr(engine, "load_existing_fft_data", lambda *args, **kwargs: None)
    cache: dict[str, object] = {}

    old_result = _fft_request(engine=engine, job=old_job, cache=cache)
    old_revision = old_result.metadata["source_revision"]
    assert calls[0]["group"] is old_job.z
    assert calls[0]["steps"] == 2

    _append_committed_step(
        path,
        np.asarray([[[[5.5], [6.5]]]], dtype="<f4"),
        timestamp=0.625,
    )

    # The already-open job remains pinned to generation two and can reuse its
    # own memory entry even though the source directory now has generation three.
    assert _fft_request(engine=engine, job=old_job, cache=cache) is old_result
    assert len(calls) == 1

    new_job = ZarrJobResult(str(path), {})
    new_result = _fft_request(engine=engine, job=new_job, cache=cache)

    assert new_result is not old_result
    assert new_result.metadata["source_revision"] != old_revision
    assert calls[1]["group"] is new_job.z
    assert calls[1]["steps"] == values.shape[0] + 1
    assert len(cache) == 2
