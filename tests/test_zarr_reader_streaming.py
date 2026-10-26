"""Behavioural tests for streaming reads of committed MuMax3 Zarr fields."""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from numcodecs import Zstd

from mmpp.core.job import ZarrJobResult

_SHAPE = (0, 1, 2, 4, 1)
_CHUNKS = (1, 1, 1, 2, 1)
_CODEC = Zstd(level=1)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def _metadata(step_count: int) -> dict[str, object]:
    return {
        "chunks": list(_CHUNKS),
        "compressor": {"id": "zstd", "level": 1},
        "dtype": "<f4",
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": [step_count, *_SHAPE[1:]],
        "zarr_format": 2,
    }


def _write_projection(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")
    (path / ".zattrs").write_text("{}", encoding="utf-8")
    field = path / "m"
    field.mkdir()
    (field / ".zarray").write_text(
        json.dumps(_metadata(0), separators=(",", ":")), encoding="utf-8"
    )
    (field / ".zattrs").write_text(
        json.dumps({"t": [], "name": "m", "unit": "A/m"}), encoding="utf-8"
    )


def _write_frame(
    path: Path, step: int, values: np.ndarray, timestamp: float, previous: str
) -> str:
    """Write all four spatial chunks for one frame and its immutable record."""

    chunks = []
    for y_block in range(2):
        for x_block in range(2):
            y_start = y_block
            x_start = x_block * 2
            chunk_values = np.asarray(
                values[:, y_start : y_start + 1, x_start : x_start + 2, :],
                dtype="<f4",
            )
            compressed = bytes(_CODEC.encode(chunk_values.tobytes(order="C")))
            key = f"{step}.0.{y_block}.{x_block}.0"
            (path / "m" / key).write_bytes(compressed)
            chunks.append(
                {
                    "key": key,
                    "size": len(compressed),
                    "sha256": hashlib.sha256(compressed).hexdigest(),
                }
            )

    frame = {
        "step": step,
        "time": timestamp,
        "previous_sha256": previous,
        "chunks": chunks,
    }
    records = path / "m" / ".frames"
    records.mkdir(exist_ok=True)
    payload = _json_bytes(frame)
    record_path = records / f"{step:020d}.json"
    record_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _make_committed_store(
    tmp_path: Path, step_count: int = 5
) -> tuple[Path, np.ndarray, np.ndarray]:
    path = tmp_path / "streaming.zarr"
    _write_projection(path)
    values = np.arange(step_count * 1 * 2 * 4 * 1, dtype="<f4").reshape(
        step_count, 1, 2, 4, 1
    )
    times = np.arange(step_count, dtype=float) * 0.25 + 0.125

    previous = ""
    last_record = ""
    for step, (value, timestamp) in enumerate(zip(values, times, strict=True)):
        last_record = _write_frame(path, step, value, float(timestamp), previous)
        previous = last_record

    commit = {
        "schema_version": 1,
        "generation": step_count,
        "committed_steps": step_count,
        "metadata": _metadata(step_count),
        "name": "m",
        "unit": "A/m",
        "last_record_sha256": last_record,
    }
    (path / "m" / "commit.json").write_bytes(_json_bytes(commit))
    return path, values, times


def _append_committed_frame(path: Path, value: np.ndarray, timestamp: float) -> None:
    commit_path = path / "m" / "commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    step = int(commit["committed_steps"])
    previous = hashlib.sha256(
        (path / "m" / ".frames" / f"{step - 1:020d}.json").read_bytes()
    ).hexdigest()
    last_record = _write_frame(path, step, value, timestamp, previous)
    commit["generation"] = step + 1
    commit["committed_steps"] = step + 1
    commit["metadata"]["shape"][0] = step + 1
    commit["last_record_sha256"] = last_record
    commit_path.write_bytes(_json_bytes(commit))


def test_streaming_reader_reads_all_blocks_and_new_commit_generation(tmp_path: Path):
    path, values, times = _make_committed_store(tmp_path)
    old_result = ZarrJobResult(str(path), {})
    old_field = old_result.z["m"]
    np.testing.assert_array_equal(np.asarray(old_field[:]), values)

    appended = np.full((1, 2, 4, 1), 999.0, dtype="<f4")
    _append_committed_frame(path, appended, timestamp=float(times[-1] + 0.25))

    new_result = ZarrJobResult(str(path), {})
    new_field = new_result.z["m"]
    expected = np.concatenate([values, appended[np.newaxis, ...]], axis=0)

    assert tuple(new_field.shape) == expected.shape
    np.testing.assert_array_equal(
        np.asarray(new_field.attrs["t"], dtype=float),
        np.concatenate([times, [times[-1] + 0.25]]),
    )
    np.testing.assert_array_equal(np.asarray(new_field[:]), expected)

    # The first reader is pinned to the old commit and does not grow with the
    # newly published head.
    assert tuple(old_field.shape) == values.shape
    np.testing.assert_array_equal(np.asarray(old_field[:]), values)


def test_one_store_pins_one_generation_across_concurrent_first_reads(
    tmp_path: Path,
):
    path, values, times = _make_committed_store(tmp_path)
    root = ZarrJobResult(str(path), {}).z
    barrier = threading.Barrier(2)
    appended = np.full((1, 2, 4, 1), 777.0, dtype="<f4")

    def first_reader() -> np.ndarray:
        field = root["m"]
        barrier.wait(timeout=5)
        return np.asarray(field[:])

    def second_reader() -> np.ndarray:
        barrier.wait(timeout=5)
        _append_committed_frame(path, appended, timestamp=float(times[-1] + 0.25))
        return np.asarray(root["m"][:])

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first_reader)
        second_future = executor.submit(second_reader)
        first = first_future.result(timeout=10)
        second = second_future.result(timeout=10)

    np.testing.assert_array_equal(first, values)
    np.testing.assert_array_equal(second, values)

    fresh = ZarrJobResult(str(path), {}).z["m"]
    expected = np.concatenate([values, appended[np.newaxis, ...]], axis=0)
    np.testing.assert_array_equal(np.asarray(fresh[:]), expected)


def test_pickled_pinned_array_keeps_its_committed_snapshot(tmp_path: Path):
    path, values, _ = _make_committed_store(tmp_path)
    field = ZarrJobResult(str(path), {}).z["m"]
    assert tuple(field.shape) == values.shape
    serialized = pickle.dumps(field)

    appended = np.full((1, 2, 4, 1), 555.0, dtype="<f4")
    _append_committed_frame(path, appended, timestamp=1.375)

    restored = pickle.loads(serialized)
    assert tuple(restored.shape) == values.shape
    np.testing.assert_array_equal(np.asarray(restored[:]), values)


@pytest.mark.parametrize("mutation", ["changed", "removed"])
def test_pinned_reader_rejects_changed_or_removed_frame_record(
    tmp_path: Path, mutation: str
):
    path, _, _ = _make_committed_store(tmp_path)
    field = ZarrJobResult(str(path), {}).z["m"]

    # Resolve the committed metadata before mutating an already-pinned record.
    assert field.shape[0] == 5
    record_path = path / "m" / ".frames" / "00000000000000000002.json"
    if mutation == "removed":
        record_path.unlink()
    else:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["previous_sha256"] = "changed-after-pinning"
        record_path.write_bytes(_json_bytes(record))

    with pytest.raises(
        (ValueError, OSError, KeyError, RuntimeError),
        match=r"(?i)(frame|record|commit|integrity|hash|snapshot|missing)",
    ):
        np.asarray(field[2:3])


def test_pinned_reader_rejects_corrupt_committed_block(tmp_path: Path):
    path, _, _ = _make_committed_store(tmp_path)
    field = ZarrJobResult(str(path), {}).z["m"]
    assert field.shape == (5, 1, 2, 4, 1)

    # This is one of four spatial blocks in an already committed frame.
    (path / "m" / "3.0.1.1.0").write_bytes(b"corrupt compressed block")

    with pytest.raises(
        (ValueError, OSError, KeyError, RuntimeError),
        match=r"(?i)(chunk|checksum|integrity|corrupt|decompress|snapshot)",
    ):
        np.asarray(field[3:4])
