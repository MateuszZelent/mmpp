"""Source fingerprints must ignore MuMax atomic-write staging files."""

from pathlib import Path

import pytest

from mmpp.fft.transmission.batch import _source_state_signature


def test_staging_files_do_not_change_source_signature(tmp_path):
    source = tmp_path / "source.zarr"
    dataset = source / "m"
    dataset.mkdir(parents=True)
    (source / ".zattrs").write_text("{}")
    chunk = dataset / "0.0.0.0.0"
    chunk.write_bytes(b"committed")
    original = _source_state_signature(source)

    (source / ".mumax-write-1").write_bytes(b"partial metadata")
    (dataset / ".mumax-write-2").write_bytes(b"partial chunk")
    assert _source_state_signature(source) == original

    chunk.write_bytes(b"new committed data")
    assert _source_state_signature(source) != original


def test_disappearing_staging_file_is_never_statted(tmp_path, monkeypatch):
    source = tmp_path / "source.zarr"
    source.mkdir()
    (source / ".zattrs").write_text("{}")
    original = _source_state_signature(source)
    staging = source / ".mumax-write-123"
    staging.write_bytes(b"staging")
    real_stat = Path.stat

    def stat(path, *args, **kwargs):
        if path == staging:
            raise FileNotFoundError(str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    assert _source_state_signature(source) == original


def test_missing_committed_file_is_not_silently_ignored(tmp_path, monkeypatch):
    source = tmp_path / "source.zarr"
    source.mkdir()
    chunk = source / "0.0.0.0.0"
    chunk.write_bytes(b"committed")
    real_stat = Path.stat
    calls = 0

    def stat(path, *args, **kwargs):
        nonlocal calls
        if path == chunk:
            calls += 1
            if calls == 2:
                raise FileNotFoundError(str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    with pytest.raises(FileNotFoundError):
        _source_state_signature(source)
