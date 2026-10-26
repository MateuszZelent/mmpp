"""Verified field snapshots are separate, conventional Zarr exports."""

import numpy as np
import pytest
import zarr

from mmpp.cache.exports import open_export
from mmpp.core.zarr_integrity import ZarrIntegrityError
from mmpp.core.zarr_snapshot import export_field_snapshot
from tests.test_committed_zarr_store import _make_committed_store


def test_export_snapshot_uses_committed_shape_and_times(tmp_path):
    source, values, times = _make_committed_store(tmp_path)
    destination = tmp_path / "snapshot.zarr"
    export_field_snapshot(source, "m", destination)

    raw = zarr.open_group(str(destination), mode="r")
    np.testing.assert_array_equal(raw["m"][:], values)
    np.testing.assert_array_equal(raw["m"].attrs["t"], times)
    assert raw.attrs["mmpp_snapshot"]["scope"] == "single_field"
    np.testing.assert_array_equal(open_export(destination)["m"][:], values)


def test_failed_snapshot_does_not_replace_previous_export(tmp_path):
    source, values, _ = _make_committed_store(tmp_path)
    destination = tmp_path / "snapshot.zarr"
    export_field_snapshot(source, "m", destination)
    (source / "m" / "1.0.0.0.0").unlink()
    with pytest.raises(ZarrIntegrityError):
        export_field_snapshot(source, "m", destination)
    np.testing.assert_array_equal(open_export(destination)["m"][:], values)


def test_snapshot_rejects_legacy_and_source_destination(tmp_path):
    source, _, _ = _make_committed_store(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        export_field_snapshot(source, "m", source / "nested.zarr")
    legacy = tmp_path / "legacy.zarr"
    zarr.open_group(str(legacy), mode="w").create_dataset(
        "m", data=np.ones((2, 1, 1, 1, 1))
    )
    with pytest.raises(ZarrIntegrityError, match="legacy"):
        export_field_snapshot(legacy, "m", tmp_path / "legacy-export.zarr")
