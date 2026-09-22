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
