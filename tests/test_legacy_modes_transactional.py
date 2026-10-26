"""Transactional cache tests for the legacy FMR mode analyzer."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from mmpp.fft.old_modes import FMRModeAnalyzer
from tests.test_committed_zarr_store import _make_committed_store
from tests.test_transactional_transmission_cache import _append_committed_source


def _add_independent_fft_cache(source):
    root = zarr.open(str(source), mode="a")
    fft = root.require_group("fft").require_group("m")
    fft.create_dataset("spectrum", data=np.asarray([42.0]), overwrite=True)


def _mode_snapshot(analyzer: FMRModeAnalyzer) -> dict[str, np.ndarray]:
    group = analyzer.zarr_file[f"modes/{analyzer.dataset_name}"]
    return {
        name: np.asarray(group[name][:]).copy()
        for name in ("arr", "freqs", "power_max", "power_sum")
    }


def test_legacy_modes_roundtrip_and_force_preserves_fft_group(tmp_path):
    source, _, _ = _make_committed_store(tmp_path)
    _add_independent_fft_cache(source)

    analyzer = FMRModeAnalyzer(str(source), dataset_name="m")
    analyzer.compute_modes(window=False, save=True, force=True)
    expected = _mode_snapshot(analyzer)

    root = zarr.open(str(source), mode="r")
    np.testing.assert_array_equal(np.asarray(root["fft/m/spectrum"][:]), [42.0])

    reader = FMRModeAnalyzer(str(source), dataset_name="m")
    assert reader.modes_available
    actual = _mode_snapshot(reader)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, err_msg=name)
    np.testing.assert_array_equal(
        np.asarray(reader.zarr_file["fft/m/spectrum"][:]), [42.0]
    )


def test_legacy_modes_failed_force_save_keeps_previous_generation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    source, _, _ = _make_committed_store(tmp_path)
    _add_independent_fft_cache(source)
    analyzer = FMRModeAnalyzer(str(source), dataset_name="m")
    analyzer.compute_modes(window=False, save=True, force=True)
    expected = _mode_snapshot(analyzer)

    def fail_after_partial_write(*args, **kwargs):
        raise RuntimeError("simulated legacy mode cache write failure")

    monkeypatch.setattr(
        "mmpp.fft.old_modes._write_zarr_array", fail_after_partial_write
    )
    with pytest.raises(RuntimeError, match="simulated legacy mode cache write failure"):
        analyzer.compute_modes(window=True, save=True, force=True)

    reader = FMRModeAnalyzer(str(source), dataset_name="m")
    assert reader.modes_available
    actual = _mode_snapshot(reader)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, err_msg=name)
    np.testing.assert_array_equal(
        np.asarray(reader.zarr_file["fft/m/spectrum"][:]), [42.0]
    )


def test_legacy_modes_miss_after_committed_source_changes(tmp_path):
    source, _, _ = _make_committed_store(tmp_path)
    _add_independent_fft_cache(source)
    analyzer = FMRModeAnalyzer(str(source), dataset_name="m")
    analyzer.compute_modes(window=False, save=True, force=True)
    assert analyzer.modes_available

    _append_committed_source(source)

    reader = FMRModeAnalyzer(str(source), dataset_name="m")
    assert not reader.modes_available
    assert reader.modes_path is None
    np.testing.assert_array_equal(
        np.asarray(reader.zarr_file["fft/m/spectrum"][:]), [42.0]
    )
