"""Crash and concurrency tests for the published Zarr cache API."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import zarr

from mmpp.cache.transactions import open_published_cache, publish_cache


def _populate(values: list[int], label: str):
    def populate(group: zarr.Group) -> None:
        group.create_dataset(
            "values",
            data=np.asarray(values, dtype=np.int64),
            chunks=(len(values),),
        )
        group.attrs["label"] = label

    return populate


def _values(group: zarr.Group) -> list[int]:
    return np.asarray(group["values"][:], dtype=np.int64).tolist()


def _payload_key(group: zarr.Group) -> str:
    """Find a Zarr array payload through the public store mapping.

    The test deliberately does not know the cache generation, head, or lock
    filenames. It derives the payload prefix from the returned Zarr array.
    """

    array = group["values"]
    prefix = f"{array.path}/"
    metadata = {f"{array.path}/.zarray", f"{array.path}/.zattrs"}
    candidates = [
        key
        for key in group.store.keys()
        if key.startswith(prefix) and key not in metadata
    ]
    assert len(candidates) == 1, candidates
    return candidates[0]


def test_first_write_is_published_and_readable(tmp_path: Path):
    assert open_published_cache(tmp_path, "spectrum", source_revision="rev-1") is None

    assert publish_cache(
        tmp_path,
        "spectrum",
        _populate([1, 2], "first"),
        source_revision="rev-1",
        overwrite=False,
    )

    cache = open_published_cache(tmp_path, "spectrum", source_revision="rev-1")
    assert cache is not None
    assert _values(cache) == [1, 2]
    assert cache.attrs["label"] == "first"
    assert cache.attrs["complete"] is True


def test_overwrite_publishes_new_generation_without_mutating_old_reader(
    tmp_path: Path,
):
    assert publish_cache(
        tmp_path,
        "spectrum",
        _populate([1, 2], "first"),
        source_revision="rev-1",
        overwrite=False,
    )
    old = open_published_cache(tmp_path, "spectrum", source_revision="rev-1")
    assert old is not None

    assert publish_cache(
        tmp_path,
        "spectrum",
        _populate([3, 4, 5], "second"),
        source_revision="rev-1",
        overwrite=True,
    )
    current = open_published_cache(tmp_path, "spectrum", source_revision="rev-1")
    assert current is not None

    assert _values(old) == [1, 2]
    assert old.attrs["label"] == "first"
    assert _values(current) == [3, 4, 5]
    assert current.attrs["label"] == "second"


def test_population_failure_keeps_previous_generation_published(tmp_path: Path):
    assert publish_cache(
        tmp_path,
        "spectrum",
        _populate([1, 2], "first"),
        source_revision="rev-1",
        overwrite=False,
    )

    def failing_populate(group: zarr.Group) -> None:
        group.create_dataset("values", data=np.asarray([9, 9], dtype=np.int64))
        raise RuntimeError("simulated cache computation failure")

    with pytest.raises(RuntimeError, match="simulated cache computation failure"):
        publish_cache(
            tmp_path,
            "spectrum",
            failing_populate,
            source_revision="rev-1",
            overwrite=True,
        )

    cache = open_published_cache(tmp_path, "spectrum", source_revision="rev-1")
    assert cache is not None
    assert _values(cache) == [1, 2]
    assert cache.attrs["label"] == "first"


def test_source_revision_mismatch_is_a_cache_miss(tmp_path: Path):
    assert publish_cache(
        tmp_path,
        "spectrum",
        _populate([1], "source-a"),
        source_revision="source-a",
        overwrite=False,
    )

    assert (
        open_published_cache(tmp_path, "spectrum", source_revision="source-b") is None
    )
    cache = open_published_cache(tmp_path, "spectrum", source_revision="source-a")
    assert cache is not None
    assert _values(cache) == [1]

    assert publish_cache(
        tmp_path,
        "spectrum",
        _populate([2], "source-b"),
        source_revision="source-b",
        overwrite=True,
    )
    assert (
        open_published_cache(tmp_path, "spectrum", source_revision="source-a") is None
    )
    current = open_published_cache(tmp_path, "spectrum", source_revision="source-b")
    assert current is not None
    assert _values(current) == [2]


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_missing_or_corrupt_saved_chunk_is_not_a_cache_hit(
    tmp_path: Path, mutation: str
):
    assert publish_cache(
        tmp_path,
        "spectrum",
        _populate([1, 2, 3], "complete"),
        source_revision="rev-1",
        overwrite=False,
    )
    cache = open_published_cache(tmp_path, "spectrum", source_revision="rev-1")
    assert cache is not None
    payload_key = _payload_key(cache)
    if mutation == "missing":
        del cache.store[payload_key]
    else:
        cache.store[payload_key] = b"corrupted payload"

    assert open_published_cache(tmp_path, "spectrum", source_revision="rev-1") is None


def test_two_competing_first_writes_publish_exactly_one_generation(tmp_path: Path):
    def attempt(label: str) -> bool:
        # Keep both calls in flight long enough to exercise the inter-process
        # lock, while each callback writes only to its private generation.
        def populate(group: zarr.Group) -> None:
            time.sleep(0.05)
            group.create_dataset("values", data=np.asarray([label], dtype="U1"))

        return publish_cache(
            tmp_path,
            "spectrum",
            populate,
            source_revision="rev-1",
            overwrite=False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, ["a", "b"]))

    assert sorted(results) == [False, True]
    cache = open_published_cache(tmp_path, "spectrum", source_revision="rev-1")
    assert cache is not None
    assert np.asarray(cache["values"][:]).tolist() in (["a"], ["b"])
