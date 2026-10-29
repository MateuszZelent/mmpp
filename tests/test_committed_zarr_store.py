"""Tests for the commit-aware MuMax3 Zarr reader.

The fixture below mirrors ``mumax3/zarr/commit.go``: every frame has a
content-addressed record in ``.frames`` and compressed chunks are described by
SHA-256 digests.  The ordinary ``.zarray``/``.zattrs`` files deliberately stay
at their initial zero-length projection so these tests verify that a reader
uses the pinned commit instead of the compatibility projection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Zstd

from mmpp.core.job import ZarrJobResult
from mmpp.core.mmpp import MMPP

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
        "shape": [step_count, 1, 1, 2, 1],
        "zarr_format": 2,
    }


def _write_projection(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")
    (path / ".zattrs").write_text("{}", encoding="utf-8")
    dataset = path / "m"
    dataset.mkdir()
    projection = _metadata(0)
    (dataset / ".zarray").write_text(json.dumps(projection, indent=2), encoding="utf-8")
    (dataset / ".zattrs").write_text(
        json.dumps({"t": [], "name": "m", "unit": "A/m"}), encoding="utf-8"
    )


def _write_frame(
    path: Path, step: int, values: np.ndarray, timestamp: float, previous: str
) -> str:
    """Write one compressed frame and return its record SHA-256."""

    # The Go writer stores each chunk as little-endian float32 in C order,
    # compressed with zstd level 1.  This fixture has one spatial chunk/frame.
    raw = np.asarray(values, dtype="<f4").reshape(1, 1, 2, 1).tobytes(order="C")
    compressed = bytes(_CODEC.encode(raw))
    key = f"{step}.0.0.0.0"
    (path / "m" / key).write_bytes(compressed)
    frame = {
        "step": step,
        "time": timestamp,
        "previous_sha256": previous,
        "chunks": [
            {
                "key": key,
                "size": len(compressed),
                "sha256": hashlib.sha256(compressed).hexdigest(),
            }
        ],
    }
    record = path / "m" / ".frames"
    record.mkdir(exist_ok=True)
    record_bytes = _json_bytes(frame)
    (record / f"{step:020d}.json").write_bytes(record_bytes)
    return hashlib.sha256(record_bytes).hexdigest()


def _write_commit(path: Path, values: list[np.ndarray], times: list[float]) -> None:
    previous = ""
    last_record = ""
    for step, (value, timestamp) in enumerate(zip(values, times, strict=True)):
        last_record = _write_frame(path, step, value, timestamp, previous)
        previous = last_record

    commit = {
        "schema_version": 1,
        "generation": len(values),
        "committed_steps": len(values),
        "metadata": _metadata(len(values)),
        "name": "m",
        "unit": "A/m",
        "last_record_sha256": last_record,
    }
    (path / "m" / "commit.json").write_bytes(_json_bytes(commit))


def _make_committed_store(
    tmp_path: Path, step_count: int = 2
) -> tuple[Path, np.ndarray, np.ndarray]:
    path = tmp_path / "committed.zarr"
    _write_projection(path)
    values = np.asarray([[[[[1.5], [2.5]]]], [[[[3.5], [4.5]]]]], dtype="<f4")
    times = np.asarray([0.125, 0.375], dtype=float)
    _write_commit(
        path, [values[i] for i in range(step_count)], times[:step_count].tolist()
    )
    return path, values, times


def _open_single_result(path: Path) -> ZarrJobResult:
    jobs = MMPP(str(path), debug=False)
    assert len(jobs) == 1
    return jobs[0]


def test_reader_uses_pinned_commit_for_shape_times_and_values(tmp_path: Path):
    path, values, times = _make_committed_store(tmp_path)

    result = _open_single_result(path)
    field = result.z["m"]

    assert tuple(field.shape) == (2, 1, 1, 2, 1)
    np.testing.assert_array_equal(np.asarray(field.attrs["t"], dtype=float), times)
    np.testing.assert_array_equal(np.asarray(field[:]), values)


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_reader_rejects_missing_or_changed_committed_chunk(
    tmp_path: Path, mutation: str
):
    path, _, _ = _make_committed_store(tmp_path)
    chunk = path / "m" / "1.0.0.0.0"
    if mutation == "missing":
        chunk.unlink()
    else:
        chunk.write_bytes(b"changed compressed payload")

    field = ZarrJobResult(str(path), {}).z["m"]
    with pytest.raises(
        (ValueError, OSError, KeyError, RuntimeError),
        match=r"(?i)(chunk|checksum|integrity|commit|decompress|not found|no such|record)",
    ):
        field[:]


def test_reader_rejects_same_length_valid_replacement_chunk(tmp_path: Path):
    from mmpp.core.zarr_integrity import ZarrIntegrityError

    path, _, _ = _make_committed_store(tmp_path)
    chunk = path / "m" / "0.0.0.0.0"
    original = chunk.read_bytes()
    replacement = bytes(_CODEC.encode(np.asarray([7.5, 8.5], dtype="<f4").tobytes()))
    assert len(replacement) == len(original)
    assert replacement != original
    chunk.write_bytes(replacement)

    with pytest.raises(ZarrIntegrityError, match="Corrupt committed Zarr chunk"):
        ZarrJobResult(str(path), {}).z["m"][0]


def test_reader_rejects_replaced_frame_record(tmp_path: Path):
    path, _, _ = _make_committed_store(tmp_path)
    (path / "m" / ".frames" / "00000000000000000001.json").write_bytes(
        b'{"step":1,"time":99.0,"previous_sha256":""}'
    )

    with pytest.raises(
        (ValueError, OSError, KeyError, RuntimeError),
        match=r"(?i)(frame|record|hash|commit|integrity|chunk)",
    ):
        ZarrJobResult(str(path), {}).z["m"]


def test_nonempty_projection_does_not_downgrade_when_commit_disappears(tmp_path: Path):
    path, _, _ = _make_committed_store(tmp_path)
    (path / "m" / "commit.json").unlink()
    (path / "m" / ".zarray").write_text(json.dumps(_metadata(2)), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing Zarr commit"):
        ZarrJobResult(str(path), {}).z["m"]


def test_new_commit_generation_does_not_change_pinned_reader(tmp_path: Path):
    path, values, times = _make_committed_store(tmp_path)
    old = ZarrJobResult(str(path), {})
    old_field = old.z["m"]
    old_values = np.asarray(old_field[:]).copy()
    old_times = np.asarray(old_field.attrs["t"], dtype=float).copy()

    extended_values = np.concatenate(
        [values, np.asarray([[[[[5.5], [6.5]]]]], dtype="<f4")], axis=0
    )
    extended_times = np.concatenate([times, np.asarray([0.625])])
    _write_frame(path, 2, extended_values[2], float(extended_times[2]), "")
    # Rebuild the chain and publish only the new commit head.  The ordinary
    # projection remains shape=0 to ensure the reader pins commit.json.
    previous = hashlib.sha256(
        (path / "m" / ".frames" / "00000000000000000001.json").read_bytes()
    ).hexdigest()
    frame2 = path / "m" / ".frames" / "00000000000000000002.json"
    frame2_data = json.loads(frame2.read_text(encoding="utf-8"))
    frame2_data["previous_sha256"] = previous
    frame2_bytes = _json_bytes(frame2_data)
    frame2.write_bytes(frame2_bytes)
    commit = json.loads((path / "m" / "commit.json").read_text(encoding="utf-8"))
    commit["generation"] = 3
    commit["committed_steps"] = 3
    commit["metadata"]["shape"][0] = 3
    commit["last_record_sha256"] = hashlib.sha256(frame2_bytes).hexdigest()
    (path / "m" / "commit.json").write_bytes(_json_bytes(commit))

    assert tuple(old_field.shape) == (2, 1, 1, 2, 1)
    np.testing.assert_array_equal(np.asarray(old_field[:]), old_values)
    np.testing.assert_array_equal(
        np.asarray(old_field.attrs["t"], dtype=float), old_times
    )

    new_field = ZarrJobResult(str(path), {}).z["m"]
    assert tuple(new_field.shape) == (3, 1, 1, 2, 1)
    np.testing.assert_array_equal(np.asarray(new_field[:]), extended_values)
    np.testing.assert_array_equal(
        np.asarray(new_field.attrs["t"], dtype=float), extended_times
    )


def test_reader_keeps_legacy_zarr_without_commit(tmp_path: Path):
    path = tmp_path / "legacy.zarr"
    root = zarr.open(str(path), mode="w")
    values = np.asarray([[[[[1.5], [2.5]]]], [[[[3.5], [4.5]]]]], dtype="<f4")
    array = root.create_dataset(
        "m", data=values, shape=values.shape, chunks=(1, 1, 1, 2, 1)
    )
    array.attrs["t"] = [0.125, 0.375]

    result = _open_single_result(path)
    field = result.z["m"]
    assert tuple(field.shape) == values.shape
    np.testing.assert_array_equal(np.asarray(field[:]), values)
