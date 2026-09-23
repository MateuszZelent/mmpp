"""Checks for spatial averaging choices in FFT spectrum computation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_fft_compute_selects_average_signal_or_average_cell_power(monkeypatch):
    import mmpp.fft.compute_fft as compute_module

    sample_count = 32
    time = np.arange(sample_count, dtype=float)
    signal = np.sin(2 * np.pi * 5 * time / sample_count)
    data = np.zeros((sample_count, 1, 1, 2, 1), dtype=float)
    data[:, 0, 0, 0, 0] = signal
    data[:, 0, 0, 1, 0] = -signal

    monkeypatch.setattr(compute_module, "normalize_z_layer_index", lambda **_: 0)
    monkeypatch.setattr(
        compute_module,
        "load_fft_input_data_profiled",
        lambda **_: (
            data,
            1.0,
            SimpleNamespace(spatial_axes=(1, 2, 3), component_axis=4),
        ),
    )
    monkeypatch.setattr(compute_module, "log_input_load_metrics", lambda **_: None)

    engine = compute_module.FFTCompute()
    common = {
        "zarr_path": "/tmp/synthetic.zarr",
        "dataset": "m",
        "force": True,
        "preloaded_data": data,
        "window": "none",
        "filter_type": "remove_mean",
        "engine": "numpy",
        "zero_padding": False,
    }
    averaged_first = engine.calculate_fft_data(method=1, **common)
    per_cell_first = engine.calculate_fft_data(method=2, **common)

    assert averaged_first.metadata["method"] == 1
    assert per_cell_first.metadata["method"] == 2
    assert np.max(np.abs(averaged_first.spectrum)) == 0.0
    assert np.max(np.abs(per_cell_first.spectrum)) > 0.0
    assert np.argmax(np.abs(per_cell_first.spectrum[:, 0])) == 5


def test_fft_result_replaces_incomplete_zarr_cache_without_create_dataset(
    monkeypatch,
):
    import zarr

    from mmpp.fft.compute_fft import FFTComputeConfig, FFTComputeResult

    class ArrayOnlyGroup:
        """Minimal Zarr 3-style group that intentionally lacks create_dataset."""

        def __init__(self):
            self.members = {}
            self.attrs = {}

        def __contains__(self, name):
            return name in self.members

        def __getitem__(self, name):
            return self.members[name]

        def __delitem__(self, name):
            del self.members[name]

        def keys(self):
            return self.members.keys()

        def create_group(self, name):
            group = ArrayOnlyGroup()
            self.members[name] = group
            return group

        def create_array(self, name, *, data, overwrite=False, **_kwargs):
            if name in self.members and not overwrite:
                raise ValueError(f"{name} already exists")
            self.members[name] = np.asarray(data).copy()
            return self.members[name]

    root = ArrayOnlyGroup()
    fft_group = root.create_group("fft")
    fft_group.create_group("m_z0_m2")  # An interrupted earlier write.
    monkeypatch.setattr(zarr, "open", lambda *_args, **_kwargs: root)

    result = FFTComputeResult(
        frequencies=np.array([0.0, 1e9, 2e9]),
        spectrum=np.array([1 + 0j, 2 + 1j, 3 + 0j]),
        metadata={"method": 2},
        config=FFTComputeConfig(),
    )
    result.save_to_zarr("incomplete.zarr", "m_z0_m2")

    saved = root["fft"]["m_z0_m2"]
    np.testing.assert_array_equal(saved["frequencies"], result.frequencies)
    np.testing.assert_array_equal(saved["spectrum"], result.spectrum)
