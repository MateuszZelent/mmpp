"""Regression tests for decorator and batch cache source identity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mmpp.cache.batch import BatchCacheEntry
from mmpp.cache.decorators import cached_result
from mmpp.cache.key import CacheKey
from mmpp.core.zarr_integrity import source_revision
from tests.test_committed_zarr_store import _make_committed_store
from tests.test_transactional_transmission_cache import _append_committed_source


def _result(path: Path, marker: float) -> SimpleNamespace:
    revision = source_revision(str(path), "m")
    assert revision is not None
    return SimpleNamespace(
        frequencies=np.asarray([1.0, 2.0]),
        spectrum=np.asarray([marker, marker + 1.0]),
        metadata={"source_revision": revision, "marker": marker},
        config=SimpleNamespace(scaling="power"),
    )


def test_cached_result_hits_only_for_the_same_committed_source_revision(tmp_path: Path):
    source, _, _ = _make_committed_store(tmp_path)
    calls = 0

    @cached_result(
        "general_spectrum",
        key_params=("zarr_path", "dataset_name", "z_layer", "method"),
    )
    def compute(
        zarr_path,
        dataset_name="m",
        z_layer=0,
        method=1,
        config=None,
        force=False,
        use_cache=True,
        save=True,
    ):
        nonlocal calls
        calls += 1
        return _result(Path(zarr_path), float(calls))

    first = compute(str(source), config={"scaling": "power"})
    cached = compute(str(source), config={"scaling": "power"})
    assert calls == 1
    assert cached.metadata["marker"] == first.metadata["marker"]

    _append_committed_source(source)
    changed = compute(str(source), config={"scaling": "power"})
    assert calls == 2
    assert changed.metadata["marker"] != first.metadata["marker"]


def test_decorator_failed_generation_keeps_previous_published_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, _, _ = _make_committed_store(tmp_path)
    calls = 0

    @cached_result("interrupted_generation", key_params=("zarr_path", "dataset_name"))
    def compute(
        zarr_path,
        dataset_name="m",
        config=None,
        force=False,
        use_cache=True,
        save=True,
    ):
        nonlocal calls
        calls += 1
        return _result(Path(zarr_path), float(calls))

    baseline = compute(str(source), config={"scaling": "power"})

    import mmpp.cache.transactions as transactions

    original_publish = transactions.publish_cache

    def interrupted_publish(parent, name, populate, *, source_revision, overwrite):
        def fail_after_population(group):
            populate(group)
            raise RuntimeError("simulated interrupted cache generation")

        return original_publish(
            parent,
            name,
            fail_after_population,
            source_revision=source_revision,
            overwrite=overwrite,
        )

    monkeypatch.setattr(transactions, "publish_cache", interrupted_publish)
    replacement = compute(
        str(source),
        config={"scaling": "power"},
        force=True,
    )
    assert replacement.metadata["marker"] != baseline.metadata["marker"]

    # The failed generation was private to the transaction. A fresh cache
    # lookup still resolves the previous published generation.
    restored = compute(str(source), config={"scaling": "power"})
    assert calls == 2
    assert restored.metadata["marker"] == baseline.metadata["marker"]


def test_batch_cache_misses_after_one_source_revision_changes(tmp_path: Path):
    first_source, _, _ = _make_committed_store(tmp_path / "first")
    second_source, _, _ = _make_committed_store(tmp_path / "second")
    job_paths = [str(first_source), str(second_source)]
    revisions = [source_revision(path, "m") for path in job_paths]
    assert all(revision is not None for revision in revisions)

    results = [_result(first_source, 10.0), _result(second_source, 20.0)]
    key = CacheKey.for_batch(
        "general_batch",
        job_paths,
        "m",
        source_signatures=[str(revision) for revision in revisions],
    )
    entry = BatchCacheEntry(
        results=results,
        parameters={"field": ["first", "second"]},
        job_paths=job_paths,
        frequencies=np.asarray([1.0, 2.0]),
    )
    cache_root = tmp_path / "batch-cache"
    entry.save_to_zarr(cache_root, key)

    loaded = BatchCacheEntry.load_from_zarr(cache_root, key)
    assert loaded is not None
    assert loaded.job_paths == job_paths
    assert loaded.parameters["field"] == ["first", "second"]
    np.testing.assert_array_equal(loaded.frequencies, [1.0, 2.0])

    _append_committed_source(second_source)
    assert BatchCacheEntry.load_from_zarr(cache_root, key) is None
