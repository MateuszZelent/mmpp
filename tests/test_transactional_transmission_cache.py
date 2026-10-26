"""Transmission cache tests against committed MuMax source snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from mmpp.fft.transmission.cache import TransmissionCache
from mmpp.fft.transmission.compute import TransmissionConfig, TransmissionResult
from tests.test_committed_zarr_store import (
    _json_bytes,
    _make_committed_store,
    _write_frame,
)


def _result(config: TransmissionConfig, marker: float) -> TransmissionResult:
    values = np.asarray([marker, marker + 1.0], dtype=float)
    return TransmissionResult(
        frequencies=np.asarray([1.0, 2.0]),
        x_positions=np.asarray([0.0, 1.0]),
        transmission=values.copy(),
        power_map=values.copy(),
        reference_power=np.asarray([1.0, 1.0]),
        config=config,
        metadata={"marker": marker},
    )


def _cache_for(
    source: Path, cache_path: Path
) -> tuple[TransmissionCache, TransmissionConfig]:
    job = SimpleNamespace(path=str(source), name="committed-source")
    config = TransmissionConfig(dataset_name="m", z_layer=0, method="power_ratio")
    return TransmissionCache(job, dataset_name="m"), config


def _append_committed_source(source: Path) -> None:
    commit_path = source / "m" / "commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    previous = hashlib.sha256(
        (source / "m" / ".frames" / "00000000000000000001.json").read_bytes()
    ).hexdigest()
    last_record = _write_frame(
        source,
        2,
        np.asarray([[[[5.5], [6.5]]]], dtype="<f4"),
        0.625,
        previous,
    )
    commit["generation"] = 3
    commit["committed_steps"] = 3
    commit["metadata"]["shape"][0] = 3
    commit["last_record_sha256"] = last_record
    commit_path.write_bytes(_json_bytes(commit))


def test_transmission_cache_hits_for_same_committed_source(tmp_path: Path):
    source, _, _ = _make_committed_store(tmp_path)
    cache_path = tmp_path / "cache"
    cache, config = _cache_for(source, cache_path)
    result = _result(config, marker=10.0)

    cache.save_result(result, cache_path=cache_path)
    loaded = cache.load_result(config, cache_path=cache_path)

    assert loaded is not None
    np.testing.assert_array_equal(loaded.transmission, result.transmission)
    assert loaded.metadata["marker"] == 10.0


def test_transmission_cache_misses_after_source_commit_advances(tmp_path: Path):
    source, _, _ = _make_committed_store(tmp_path)
    cache_path = tmp_path / "cache"
    cache, config = _cache_for(source, cache_path)
    cache.save_result(_result(config, marker=10.0), cache_path=cache_path)
    assert cache.load_result(config, cache_path=cache_path) is not None

    _append_committed_source(source)

    assert cache.load_result(config, cache_path=cache_path) is None


def test_failed_transmission_overwrite_keeps_previous_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, _, _ = _make_committed_store(tmp_path)
    cache_path = tmp_path / "cache"
    cache, config = _cache_for(source, cache_path)
    original = _result(config, marker=10.0)
    replacement = _result(config, marker=20.0)
    cache.save_result(original, cache_path=cache_path)

    def fail_after_partial_write(entry, result, slice_info):
        cache._create_dataset(entry, "frequencies", result.frequencies)
        raise RuntimeError("simulated transmission cache write failure")

    monkeypatch.setattr(cache, "_populate_entry", fail_after_partial_write)
    with pytest.raises(
        RuntimeError, match="simulated transmission cache write failure"
    ):
        cache.save_result(replacement, cache_path=cache_path, overwrite=True)

    loaded = cache.load_result(config, cache_path=cache_path)
    assert loaded is not None
    assert loaded.metadata["marker"] == 10.0
    np.testing.assert_array_equal(loaded.transmission, original.transmission)


def test_legacy_source_without_commit_is_a_persistent_cache_miss(tmp_path: Path):
    source = tmp_path / "legacy.zarr"
    root = zarr.open(str(source), mode="w")
    root.create_dataset("m", data=np.zeros((1, 1, 1, 1, 1), dtype="<f4"))
    cache_path = tmp_path / "cache"
    cache, config = _cache_for(source, cache_path)

    cache.save_result(_result(config, marker=10.0), cache_path=cache_path)

    # A new object proves this is a durable-reader miss, not an in-memory
    # decision made by the writer instance.
    fresh_cache, fresh_config = _cache_for(source, cache_path)
    assert fresh_cache.load_result(fresh_config, cache_path=cache_path) is None
