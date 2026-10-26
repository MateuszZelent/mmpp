"""Standalone Zarr exports retain a complete prior version on failure."""

import numpy as np
import pytest
import zarr

from mmpp.cache.exports import open_export
from mmpp.fft.spectrum.batch.result import BatchSpectrumResult


def _result(scale: float) -> BatchSpectrumResult:
    return BatchSpectrumResult(
        frequencies=np.array([1.0, 2.0]),
        spectra=[np.array([scale, 2 * scale], dtype=complex)],
        powers=[np.array([scale, 2 * scale], dtype=float)],
        parameters={},
        job_paths=["input.zarr"],
    )


def test_export_exchange_pins_an_existing_reader(tmp_path):
    path = tmp_path / "batch.zarr"
    _result(1).save(path)
    reader = open_export(path)
    _result(3).save(path)

    np.testing.assert_array_equal(reader["powers"][:], [[1, 2]])
    np.testing.assert_array_equal(BatchSpectrumResult.load(path).powers[0], [3, 6])
    assert list(tmp_path.glob(".mmpp-export-*"))


def test_export_failure_leaves_previous_data(tmp_path, monkeypatch):
    path = tmp_path / "batch.zarr"
    _result(1).save(path)

    def fail(*args, **kwargs):
        raise OSError("injected export write error")

    monkeypatch.setattr("mmpp.fft.spectrum.batch.result.write_zarr_array", fail)
    with pytest.raises(OSError, match="injected export"):
        _result(3).save(path)
    np.testing.assert_array_equal(BatchSpectrumResult.load(path).powers[0], [1, 2])


def test_export_rejects_missing_committed_chunk(tmp_path):
    path = tmp_path / "batch.zarr"
    _result(1).save(path)
    (path / "powers" / "0.0").unlink()
    with pytest.raises(ValueError, match="Invalid Zarr export"):
        BatchSpectrumResult.load(path)


def test_export_fails_closed_when_atomic_exchange_is_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "batch.zarr"
    _result(1).save(path)

    def fail(*args, **kwargs):
        raise OSError("atomic exchange unavailable")

    monkeypatch.setattr("mmpp.cache.exports._renameat2", fail)
    with pytest.raises(OSError, match="atomic exchange unavailable"):
        _result(3).save(path)
    np.testing.assert_array_equal(BatchSpectrumResult.load(path).powers[0], [1, 2])


def test_legacy_exports_remain_readable_without_proc(tmp_path, monkeypatch):
    path = tmp_path / "legacy.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("values", data=np.array([1.0, 2.0]))
    monkeypatch.setattr(
        "mmpp.cache.exports._supports_pinned_directories", lambda: False
    )

    np.testing.assert_array_equal(open_export(path)["values"][:], [1, 2])
    root.attrs["mmpp_export_integrity_version"] = 1
    with pytest.raises(RuntimeError, match="require /proc/self/fd"):
        open_export(path)
