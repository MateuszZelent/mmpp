from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import zarr

from mmpp.core.job import ZarrJobResult


def _make_vortex_snapshot(
    nx: int, ny: int, cx: float, cy: float, core_radius_px: float = 3.5
) -> np.ndarray:
    x = np.arange(nx, dtype=float) - float(cx)
    y = np.arange(ny, dtype=float) - float(cy)
    xg, yg = np.meshgrid(x, y)

    radius = np.hypot(xg, yg)
    phi = np.arctan2(yg, xg)
    mz = np.exp(-((radius / core_radius_px) ** 2))
    m_perp = np.sqrt(np.clip(1.0 - mz**2, 0.0, 1.0))
    mx = -m_perp * np.sin(phi)
    my = m_perp * np.cos(phi)
    m = np.stack([mx, my, mz], axis=-1)
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norm > 1e-12, norm, 1.0)


def _create_job(tmp_path, *, nt: int = 80):
    zarr_path = tmp_path / "vortex_signals_test.zarr"
    z = zarr.open(str(zarr_path), mode="w")

    nx = ny = 48
    dx = dy = 1e-9
    dt = 5e-12
    t = np.arange(nt, dtype=float) * dt
    x0 = (nx - 1) * 0.5 * dx
    y0 = (ny - 1) * 0.5 * dy

    radius = 4.2e-9 + 0.8e-9 * np.sin(2.0 * np.pi * 1.6e9 * t)
    x = x0 + radius * np.cos(2.0 * np.pi * 0.9e9 * t)
    y = y0 + radius * np.sin(2.0 * np.pi * 0.9e9 * t)

    data = np.zeros((nt, 1, ny, nx, 3), dtype=float)
    for i in range(nt):
        data[i, 0] = _make_vortex_snapshot(nx, ny, x[i] / dx, y[i] / dy)
    z.create_dataset("m", data=data, chunks=data.shape)

    table = z.create_group("table")
    table.create_dataset("t", data=t, chunks=t.shape)
    table.create_dataset("mx", data=np.sin(2.0 * np.pi * 0.9e9 * t), chunks=t.shape)
    table.create_dataset("my", data=np.cos(2.0 * np.pi * 0.9e9 * t), chunks=t.shape)
    table.create_dataset("I_dc", data=np.full(nt, 2.0e-3), chunks=t.shape)

    z.attrs["dx"] = dx
    z.attrs["dy"] = dy
    z.attrs["t_sampl"] = dt

    return ZarrJobResult(str(zarr_path), {})


def test_signals_pipeline_from_tracked_trajectory(tmp_path):
    job = _create_job(tmp_path, nt=96)
    signals_html = job.m.vortex.signals._repr_html_()
    signals_plot_html = job.m.vortex.signals.plt._repr_html_()
    assert "Vortex Signals Interface" in signals_html
    assert "Vortex signals API help" in signals_html
    assert ".magnetoresistance(" in signals_html
    assert ">Overview</button>" in signals_html
    assert ">API</button>" in signals_html
    assert "Vortex signals plot API help" in signals_plot_html
    assert ".power_spectrum(" in signals_plot_html
    assert ">Overview</button>" in signals_plot_html
    assert ">API</button>" in signals_plot_html

    mr = job.m.vortex.signals.magnetoresistance(
        resistance_parallel_ohm=120.0,
        delta_resistance_ohm=30.0,
    )
    assert "MagnetoresistanceResult" in mr._repr_html_()
    assert mr.time.ndim == 1
    assert mr.resistance_ohm.shape == mr.time.shape
    assert np.isfinite(mr.mean_resistance_ohm)

    voltage = job.m.vortex.signals.voltage(current_a=2.0e-3, magnetoresistance=mr)
    assert "VoltageResult" in voltage._repr_html_()
    assert voltage.voltage_v.shape == mr.time.shape
    assert np.allclose(voltage.voltage_v, voltage.current_a * voltage.resistance_ohm)

    spec = job.m.vortex.signals.power_spectrum(signal="voltage", current_a=2.0e-3)
    assert "SignalSpectrumResult" in spec._repr_html_()
    assert spec.frequencies_hz.ndim == 1
    assert spec.power.shape == spec.frequencies_hz.shape
    ax = spec.plt.power_spectrum()
    assert hasattr(ax, "plot")


def test_signals_table_fallback_when_tracking_fails(tmp_path, monkeypatch):
    job = _create_job(tmp_path, nt=48)
    signals = job.m.vortex.signals

    def _raise_tracking(*_args, **_kwargs):
        raise RuntimeError("tracking unavailable")

    monkeypatch.setattr(signals._core, "track", _raise_tracking)
    mr = signals.magnetoresistance(force=True)

    assert mr.method == "table_projection"
    assert mr.time.size == 48
    assert "fallback_reason" in mr.metadata
