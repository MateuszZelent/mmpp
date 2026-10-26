"""Dispersion cache tests against committed MuMax source snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from mmpp.fft.dispersion.interface import FFTDispersionInterface
from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
from tests.test_committed_zarr_store import (
    _json_bytes,
    _make_committed_store,
    _write_frame,
)


def _result() -> DispersionResult1D:
    return DispersionResult1D(
        S=np.arange(6, dtype=np.float32).reshape(2, 3),
        k_axis=np.asarray([-1.0, 1.0]),
        f_axis=np.asarray([0.0, 1.0, 2.0]),
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=2.0, component="perp"),
        dt=1.0,
        dx=2.0,
        scaling="raw_power",
        scaling_factors={"scale": 1.0},
        notes=["transactional test"],
    )


def _interface(source: Path) -> FFTDispersionInterface:
    return FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(name="run", path=source)),
        dataset_name="m",
    )


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


def _save_and_load(
    interface: FFTDispersionInterface,
    group: zarr.Group,
    result: DispersionResult1D,
    *,
    overwrite: bool = True,
) -> DispersionResult1D | None:
    interface._save_dispersion_result(
        group,
        "dispersion1d_transactional",
        result,
        context_json='{"axis":"x","source":"committed"}',
        context_hash="transactional-context",
        overwrite=overwrite,
    )
    return interface._load_cached_dispersion_result(
        group,
        "dispersion1d_transactional",
        "transactional-context",
    )


def test_dispersion_cache_roundtrip_for_committed_source(tmp_path: Path):
    source, _, _ = _make_committed_store(tmp_path)
    interface = _interface(source)
    group = zarr.open(str(tmp_path / "cache.zarr"), mode="w")
    result = _result()

    loaded = _save_and_load(interface, group, result)

    assert loaded is not None
    np.testing.assert_array_equal(loaded.S, result.S)
    np.testing.assert_array_equal(loaded.k_axis, result.k_axis)
    np.testing.assert_array_equal(loaded.f_axis, result.f_axis)
    assert loaded.notes == result.notes


def test_failed_force_overwrite_keeps_previous_dispersion_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, _, _ = _make_committed_store(tmp_path)
    interface = _interface(source)
    group = zarr.open(str(tmp_path / "cache.zarr"), mode="w")
    result = _result()
    assert _save_and_load(interface, group, result) is not None

    original_create = interface._create_dataset

    def fail_after_partial_write(entry, name, data):
        original_create(entry, name, data)
        if name == "S":
            raise RuntimeError("simulated dispersion cache write failure")

    monkeypatch.setattr(interface, "_create_dataset", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated dispersion cache write failure"):
        interface._save_dispersion_result(
            group,
            "dispersion1d_transactional",
            result,
            context_json='{"axis":"x","source":"committed"}',
            context_hash="transactional-context",
            overwrite=True,
        )

    monkeypatch.undo()
    loaded = interface._load_cached_dispersion_result(
        group,
        "dispersion1d_transactional",
        "transactional-context",
    )
    assert loaded is not None
    np.testing.assert_array_equal(loaded.S, result.S)


def test_dispersion_cache_misses_after_source_commit_advances(tmp_path: Path):
    source, _, _ = _make_committed_store(tmp_path)
    interface = _interface(source)
    group = zarr.open(str(tmp_path / "cache.zarr"), mode="w")
    result = _result()
    assert _save_and_load(interface, group, result) is not None

    _append_committed_source(source)

    assert (
        interface._load_cached_dispersion_result(
            group,
            "dispersion1d_transactional",
            "transactional-context",
        )
        is None
    )


def test_legacy_source_without_commit_has_no_persistent_dispersion_hit(
    tmp_path: Path,
):
    source = tmp_path / "legacy.zarr"
    root = zarr.open(str(source), mode="w")
    root.create_dataset("m", data=np.zeros((1, 1, 1, 1, 1), dtype="<f4"))
    interface = _interface(source)
    group = zarr.open(str(tmp_path / "cache.zarr"), mode="w")

    interface._save_dispersion_result(
        group,
        "dispersion1d_transactional",
        _result(),
        context_json='{"axis":"x","source":"legacy"}',
        context_hash="legacy-context",
        overwrite=True,
    )

    fresh_interface = _interface(source)
    assert (
        fresh_interface._load_cached_dispersion_result(
            group,
            "dispersion1d_transactional",
            "legacy-context",
        )
        is None
    )
