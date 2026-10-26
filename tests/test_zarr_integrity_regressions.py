"""Regression tests for interrupted Zarr output and committed table prefixes.

These fixtures intentionally model the on-disk layout emitted by the current
MuMax3 tabular writer. A physical column can contain an uncommitted tail after
an interruption; ``manifest.json`` is the visibility boundary for readers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from mmpp.core.job import ZarrJobResult
from mmpp.core.mmpp import MMPP
from mmpp.core.zarr_integrity import ZarrIntegrityError


def test_single_zarr_with_invalid_root_attrs_reports_corruption(tmp_path: Path):
    """Opening one corrupt result must be explicit, never an empty result set."""

    result_path = tmp_path / "interrupted.zarr"
    root = zarr.open(str(result_path), mode="w")
    root.attrs["run_id"] = "run-1"

    # Simulate a process dying while replacing the sole root metadata file.
    (result_path / ".zattrs").write_bytes(b'{"run_id":')

    with pytest.raises(
        ValueError,
        match=r"(?i)(zarr|metadata|attrs).*(interrupted|corrupt|invalid|decode)",
    ):
        MMPP(str(result_path), debug=False)


def test_authoritative_run_status_overrides_projected_end_time(tmp_path: Path):
    path = tmp_path / "run.zarr"
    root = zarr.open(str(path), mode="w")
    root.attrs.update({"end_time": "projected", "run_status_file": "run.json"})
    run = path / "run.json"
    run.write_text('{"completion":"running"}', encoding="utf-8")
    job = ZarrJobResult(str(path), {})
    assert not job.is_finished()
    run.write_text('{"completion":"interrupted"}', encoding="utf-8")
    assert job.is_finished()
    run.write_text('{"completion":"unknown"}', encoding="utf-8")
    with pytest.raises(ZarrIntegrityError, match="Invalid completion"):
        job.is_finished()
    run.write_text('{"completion":[]}', encoding="utf-8")
    with pytest.raises(ZarrIntegrityError, match="Invalid completion"):
        job.is_finished()
    run.unlink()
    with pytest.raises(ZarrIntegrityError, match="Missing authoritative"):
        job.is_finished()


def _create_table_result(
    tmp_path: Path, *, committed_rows: int | None
) -> ZarrJobResult:
    result_path = tmp_path / (
        "committed_table.zarr" if committed_rows is not None else "legacy_table.zarr"
    )
    root = zarr.open(str(result_path), mode="w")
    table = root.create_group("table")
    table.create_dataset("t", data=np.arange(5, dtype=np.float64))
    table.create_dataset("mx", data=np.arange(10, 15, dtype=np.float64))

    if committed_rows is not None:
        manifest = {
            "version": 1,
            "format": "zarr",
            "columns": ["t", "mx"],
            "units": ["s", ""],
            "committed_rows": committed_rows,
            "chunk_rows": 1024,
        }
        (result_path / "table" / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    return ZarrJobResult(str(result_path), {})


def test_table_reader_uses_manifest_committed_rows(tmp_path: Path):
    """A physical tail beyond the manifest must stay invisible to MMPP."""

    job = _create_table_result(tmp_path, committed_rows=2)

    table = job.table
    assert table.n_rows == 2
    frame = table.to_dataframe()
    assert frame.shape == (2, 2)
    assert frame["t"].tolist() == [0.0, 1.0]
    assert frame["mx"].tolist() == [10.0, 11.0]


def test_table_reader_preserves_legacy_layout_without_manifest(tmp_path: Path):
    """Old Zarr tables without a manifest continue to use physical lengths."""

    job = _create_table_result(tmp_path, committed_rows=None)

    table = job.table
    assert table.n_rows == 5
    frame = table.to_dataframe()
    assert frame.shape == (5, 2)
    assert frame["t"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert frame["mx"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
