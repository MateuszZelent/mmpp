"""Zero-length projections can be read from a validated frame prefix."""

import hashlib
import json

import numpy as np
import pytest

from mmpp.core.job import ZarrJobResult
from mmpp.core.zarr_integrity import ZarrIntegrityError, job_source_revision
from tests.test_committed_zarr_store import _make_committed_store, _write_frame


def orphaned(tmp_path):
    path, values, times = _make_committed_store(tmp_path)
    (path / "m" / "commit.json").unlink()
    return path, values, times


def test_zero_projection_recovers_values_times_without_source_writes(tmp_path):
    path, values, times = orphaned(tmp_path)
    before = {
        p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }
    with pytest.warns(RuntimeWarning, match="Recovered.*2.*without commit.json"):
        job = ZarrJobResult(str(path), {})
        field = job.z["m"]
    assert job.m.shape == values.shape
    np.testing.assert_array_equal(field[:], values)
    np.testing.assert_array_equal(field.attrs["t"], times)
    assert field.attrs["mmpp_recovery"]["anchored_to_commit"] is False
    assert job_source_revision(job, "m")
    assert {
        p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()
    } == before


def test_recovery_stops_at_missing_chunk_and_pins_prefix(tmp_path):
    path, values, times = orphaned(tmp_path)
    (path / "m" / "1.0.0.0.0").unlink()
    with pytest.warns(RuntimeWarning, match="Recovered.*1"):
        old = ZarrJobResult(str(path), {})
        field = old.z["m"]
    np.testing.assert_array_equal(field[:], values[:1])
    revision = job_source_revision(old, "m")
    prev = hashlib.sha256(
        (path / "m" / ".frames" / f"{0:020d}.json").read_bytes()
    ).hexdigest()
    _write_frame(path, 1, values[1], times[1], prev)
    assert field.shape[0] == 1
    assert job_source_revision(old, "m") == revision
    with pytest.warns(RuntimeWarning):
        fresh = ZarrJobResult(str(path), {})
        assert fresh.m.shape[0] == 2
    assert job_source_revision(fresh, "m") != revision


@pytest.mark.parametrize("mutation", ["chain", "timestamp", "duplicate", "unsafe_key"])
def test_recovery_rejects_invalid_records(tmp_path, mutation):
    path, _, _ = orphaned(tmp_path)
    record = path / "m" / ".frames" / f"{1:020d}.json"
    data = json.loads(record.read_bytes())
    if mutation == "chain":
        data["previous_sha256"] = "0" * 64
    if mutation == "timestamp":
        data["time"] = 0.125
    if mutation == "duplicate":
        data["chunks"].append(data["chunks"][0])
    if mutation == "unsafe_key":
        data["chunks"][0]["key"] = "../escape"
    record.write_text(json.dumps(data))
    with pytest.raises(ZarrIntegrityError):
        ZarrJobResult(str(path), {}).z["m"]


def test_recovered_payload_hash_is_checked_on_read(tmp_path):
    path, _, _ = orphaned(tmp_path)
    chunk = path / "m" / "0.0.0.0.0"
    original = chunk.read_bytes()
    chunk.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.warns(RuntimeWarning):
        field = ZarrJobResult(str(path), {}).z["m"]
    with pytest.raises(ZarrIntegrityError, match="Corrupt.*chunk"):
        field[:]


def test_invalid_commit_never_falls_back(tmp_path):
    path, _, _ = _make_committed_store(tmp_path)
    (path / "m" / "commit.json").write_bytes(b"{")
    with pytest.raises(ZarrIntegrityError):
        ZarrJobResult(str(path), {}).z["m"]


def test_empty_projection_without_frames_remains_empty(tmp_path):
    from tests.test_committed_zarr_store import _write_projection

    path = tmp_path / "empty.zarr"
    _write_projection(path)
    assert ZarrJobResult(str(path), {}).m.shape[0] == 0
