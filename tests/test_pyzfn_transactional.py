"""Transactional contracts for the legacy :class:`Pyzfn` writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from mmpp.pyzfn.pyzfn import Pyzfn
from tests.test_committed_zarr_store import _make_committed_store
from tests.test_transactional_transmission_cache import _append_committed_source


def _make_legacy_source(tmp_path: Path) -> Path:
    path = tmp_path / "legacy.zarr"
    root = zarr.open(str(path), mode="w")
    values = np.asarray(
        [
            [[[[1.5], [2.5]]]],
            [[[[3.5], [4.5]]]],
        ],
        dtype="<f4",
    )
    field = root.create_dataset("m", data=values, chunks=(1, 1, 1, 2, 1))
    field.attrs["t"] = [0.125, 0.375]
    return path


def _mode_products(job: Pyzfn) -> dict[str, np.ndarray]:
    return {
        "modes_arr": np.asarray(job["modes/m/arr"][:]).copy(),
        "modes_freqs": np.asarray(job["modes/m/freqs"][:]).copy(),
        "fft_freqs": np.asarray(job["fft/m/freqs"][:]).copy(),
        "fft_spec": np.asarray(job["fft/m/spec"][:]).copy(),
        "fft_sum": np.asarray(job["fft/m/sum"][:]).copy(),
    }


def _fail_after_population(monkeypatch, message: str) -> None:
    import mmpp.cache.transactions as transactions

    real_publish = transactions.publish_cache

    def failing_publish(parent, name, populate, **kwargs):
        def populate_then_fail(group):
            populate(group)
            raise RuntimeError(message)

        return real_publish(parent, name, populate_then_fail, **kwargs)

    monkeypatch.setattr(transactions, "publish_cache", failing_publish)


def test_calc_modes_publishes_complete_modes_and_fft_generation(tmp_path: Path):
    source, _, _ = _make_committed_store(tmp_path)
    writer = Pyzfn(source)

    writer.calc_modes(window=False)
    products = _mode_products(writer)

    assert products["modes_arr"].shape == (2, 1, 1, 2, 1)
    assert products["modes_freqs"].shape == (2,)
    assert products["fft_freqs"].shape == (2,)
    assert products["fft_spec"].shape == (2, 1)
    assert products["fft_sum"].shape == (2, 1)
    assert writer["modes/m"].attrs["complete"] is True

    reader = Pyzfn(source)
    assert "modes/m/arr" in reader
    for name, expected in products.items():
        actual_name = {
            "modes_arr": "modes/m/arr",
            "modes_freqs": "modes/m/freqs",
            "fft_freqs": "fft/m/freqs",
            "fft_spec": "fft/m/spec",
            "fft_sum": "fft/m/sum",
        }[name]
        np.testing.assert_array_equal(np.asarray(reader[actual_name][:]), expected)


def test_calc_modes_failed_overwrite_keeps_previous_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, _, _ = _make_committed_store(tmp_path)
    writer = Pyzfn(source)
    writer.calc_modes(window=False)
    expected = _mode_products(writer)

    _fail_after_population(monkeypatch, "simulated Pyzfn mode write failure")
    with pytest.raises(RuntimeError, match="simulated Pyzfn mode write failure"):
        writer.calc_modes(window=True)

    for name, values in _mode_products(writer).items():
        np.testing.assert_array_equal(values, expected[name], err_msg=name)

    reader = Pyzfn(source)
    for name, expected_values in expected.items():
        actual_name = {
            "modes_arr": "modes/m/arr",
            "modes_freqs": "modes/m/freqs",
            "fft_freqs": "fft/m/freqs",
            "fft_spec": "fft/m/spec",
            "fft_sum": "fft/m/sum",
        }[name]
        np.testing.assert_array_equal(
            np.asarray(reader[actual_name][:]), expected_values, err_msg=name
        )


def test_committed_source_advance_is_a_modes_cache_miss(tmp_path: Path):
    source, _, _ = _make_committed_store(tmp_path)
    pinned = Pyzfn(source)
    pinned.calc_modes(window=False)
    assert "modes/m" in pinned

    _append_committed_source(source)

    reader = Pyzfn(source)
    assert "modes/m" not in reader
    with pytest.raises(KeyError):
        reader["modes/m"]


def test_legacy_modes_are_available_only_in_the_writing_instance(tmp_path: Path):
    source = _make_legacy_source(tmp_path)
    writer = Pyzfn(source)

    with pytest.warns(UserWarning, match="no committed revision"):
        writer.calc_modes(window=False)
    assert "modes/m" in writer
    assert np.asarray(writer["modes/m/arr"][:]).shape == (2, 1, 1, 2, 1)

    reader = Pyzfn(source)
    assert "modes/m" not in reader
    with pytest.raises(KeyError):
        reader["modes/m"]


def test_add_ndarray_failed_overwrite_keeps_previous_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = _make_legacy_source(tmp_path)
    writer = Pyzfn(source)
    original = np.asarray([1.0, 2.0, 3.0])
    replacement = np.asarray([9.0, 8.0, 7.0])

    writer.add_ndarray("scratch", original)
    np.testing.assert_array_equal(np.asarray(writer["scratch"][:]), original)

    _fail_after_population(monkeypatch, "simulated ndarray write failure")
    with pytest.raises(RuntimeError, match="simulated ndarray write failure"):
        writer.add_ndarray("scratch", replacement)

    np.testing.assert_array_equal(np.asarray(writer["scratch"][:]), original)
    reader = Pyzfn(source)
    np.testing.assert_array_equal(np.asarray(reader["scratch"][:]), original)
