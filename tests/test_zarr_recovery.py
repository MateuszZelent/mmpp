"""Diagnostics for committed and legacy Zarr datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from mmpp.core.zarr_recovery import diagnose_dataset
from tests.test_committed_zarr_store import _make_committed_store


def _files(path: Path) -> dict[str, bytes]:
    return {
        file.relative_to(path).as_posix(): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }


def _messages(report: dict) -> list[str]:
    messages: list[str] = []
    for key in ("issues", "warnings"):
        values = report.get(key, [])
        messages.extend(str(value) for value in values)
    return messages


def test_diagnose_healthy_committed_dataset(tmp_path: Path):
    path, _, _ = _make_committed_store(tmp_path)

    report = diagnose_dataset(path, "m")

    assert report["format"] == "committed-v1"
    assert report["committed_steps"] == 2
    assert report["verified_prefix"] == 2
    assert report["head_chain_valid"] is True


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_diagnose_reports_only_verified_prefix_without_mutating_dataset(
    tmp_path: Path, mutation: str
):
    path, _, _ = _make_committed_store(tmp_path)
    chunk = path / "m" / "1.0.0.0.0"
    if mutation == "missing":
        chunk.unlink()
    else:
        chunk.write_bytes(b"corrupted compressed payload")
    before = _files(path)

    report = diagnose_dataset(path, "m", verify_chunks=True)

    assert report["format"] == "committed-v1"
    assert report["committed_steps"] == 2
    assert report["verified_prefix"] == 1
    assert report["issues"]
    assert _files(path) == before


def test_diagnose_rejects_invalid_commit_head(tmp_path: Path):
    path, _, _ = _make_committed_store(tmp_path)
    commit_path = path / "m" / "commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["last_record_sha256"] = "0" * 64
    commit_path.write_text(json.dumps(commit), encoding="utf-8")

    report = diagnose_dataset(path, "m")

    assert report["format"] == "committed-v1"
    assert report["head_chain_valid"] is False
    assert _messages(report)
    assert (
        report["verified_prefix"] < report["committed_steps"]
        or report.get("success") is not True
    )


def test_diagnose_legacy_store_warns_without_calling_fill_chunks_corrupt(
    tmp_path: Path,
):
    path = tmp_path / "legacy.zarr"
    root = zarr.open(str(path), mode="w")
    array = root.create_dataset(
        "m",
        shape=(2, 1, 1, 2, 1),
        chunks=(1, 1, 1, 2, 1),
        dtype="<f4",
    )
    array[0] = np.asarray([[[[1.0], [2.0]]]], dtype="<f4")
    array.attrs["t"] = [0.125, 0.375]
    before = _files(path)

    report = diagnose_dataset(path, "m")

    assert report["format"] == "legacy"
    assert report["head_chain_valid"] is False
    messages = _messages(report)
    assert messages
    lowered = [message.lower() for message in messages]
    assert any("legacy" in message or "commit" in message for message in lowered)
    assert not any(
        ("missing" in message and "chunk" in message)
        or "corrupt" in message
        or "damaged" in message
        for message in lowered
    )
    assert _files(path) == before
