"""Mode-cache regression tests against committed MuMax source snapshots."""

from __future__ import annotations

import numpy as np
import pytest

from mmpp.fft.modes import FMRModeAnalyzer
from tests.test_committed_zarr_store import _make_committed_store
from tests.test_transactional_transmission_cache import _append_committed_source


def _mode_snapshot(analyzer: FMRModeAnalyzer) -> dict[str, np.ndarray]:
    """Read every persisted mode product through the public analyzer overlay."""

    group = analyzer.zarr_file[analyzer.mode_group]
    return {
        name: np.asarray(group[name][:]).copy()
        for name in ("arr", "freqs", "material_mask", "power_max", "power_sum")
    }


def test_modes_cache_roundtrip_from_committed_source(tmp_path):
    source, _, _ = _make_committed_store(tmp_path)

    writer = FMRModeAnalyzer(str(source), dataset_name="m")
    writer.compute_modes(window=False, save=True, force=True)
    expected = _mode_snapshot(writer)

    reader = FMRModeAnalyzer(str(source), dataset_name="m")
    assert reader.modes_available
    actual = _mode_snapshot(reader)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, err_msg=name)
    group = reader.zarr_file[reader.mode_group]
    assert group.attrs["window_applied"] is False
    assert group.attrs["power_definition"] == "abs_fft_squared"


def test_failed_force_save_keeps_previous_mode_generation(tmp_path, monkeypatch):
    source, _, _ = _make_committed_store(tmp_path)

    analyzer = FMRModeAnalyzer(str(source), dataset_name="m")
    analyzer.compute_modes(window=False, save=True, force=True)
    expected = _mode_snapshot(analyzer)

    def fail_after_partial_write(*args, **kwargs):
        raise RuntimeError("simulated mode cache write failure")

    monkeypatch.setattr("mmpp.fft.modes._write_zarr_array", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated mode cache write failure"):
        analyzer.compute_modes(window=True, save=True, force=True)

    reader = FMRModeAnalyzer(str(source), dataset_name="m")
    assert reader.modes_available
    actual = _mode_snapshot(reader)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, err_msg=name)
    assert reader.zarr_file[reader.mode_group].attrs["window_applied"] is False


def test_mode_cache_misses_after_source_commit_changes(tmp_path):
    source, _, _ = _make_committed_store(tmp_path)

    analyzer = FMRModeAnalyzer(str(source), dataset_name="m")
    analyzer.compute_modes(window=False, save=True, force=True)
    assert analyzer.modes_available

    _append_committed_source(source)

    reader = FMRModeAnalyzer(str(source), dataset_name="m")
    assert not reader.modes_available
    assert reader.modes_path is None

    reader.compute_modes(window=False, save=True, force=False)
    assert reader.modes_available
    assert np.asarray(reader.zarr_file[reader.mode_group]["arr"]).shape[0] == 2
