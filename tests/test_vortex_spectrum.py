from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import zarr

from mmpp.core.job import ZarrJobResult
from mmpp.solitons.vortex.spectrum.gyration import compute_gyration_spectrum


def _make_vortex_snapshot(
    nx: int,
    ny: int,
    *,
    center_x: float,
    center_y: float,
    core_radius_px: float = 3.5,
) -> np.ndarray:
    x = np.arange(nx, dtype=float) - center_x
    y = np.arange(ny, dtype=float) - center_y
    x_grid, y_grid = np.meshgrid(x, y)

    radius = np.hypot(x_grid, y_grid)
    phi = np.arctan2(y_grid, x_grid)

    mz = np.exp(-(radius / core_radius_px) ** 2)
    m_perp = np.sqrt(np.clip(1.0 - mz**2, 0.0, 1.0))

    mx = -m_perp * np.sin(phi)
    my = m_perp * np.cos(phi)

    m = np.stack([mx, my, mz], axis=-1)
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norm > 1e-12, norm, 1.0)


def _make_orbit_data(
    nt: int = 512,
    nx: int = 96,
    ny: int = 96,
    *,
    dx: float = 1e-9,
    dy: float = 1e-9,
    dt: float = 5e-12,
    radius_m: float = 4e-9,
    frequency_hz: float = 2.0e9,
):
    t = np.arange(nt, dtype=float) * dt
    x0_px = (nx - 1) / 2.0
    y0_px = (ny - 1) / 2.0

    x_expected = x0_px * dx + radius_m * np.cos(2.0 * np.pi * frequency_hz * t)
    y_expected = y0_px * dy + radius_m * np.sin(2.0 * np.pi * frequency_hz * t)

    data = np.zeros((nt, ny, nx, 3), dtype=float)
    for i in range(nt):
        data[i] = _make_vortex_snapshot(
            nx,
            ny,
            center_x=x_expected[i] / dx,
            center_y=y_expected[i] / dy,
        )

    return data, frequency_hz, dx, dy, dt


def _create_job(tmp_path, data: np.ndarray, *, dx: float, dy: float, dt: float):
    zarr_path = tmp_path / "vortex_spectrum_phase2.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    z.create_dataset("m", data=data, chunks=data.shape)
    z.attrs["dx"] = dx
    z.attrs["dy"] = dy
    z.attrs["t_sampl"] = dt
    return ZarrJobResult(str(zarr_path), {})


def test_gyration_spectrum_peak_and_plot(tmp_path):
    data, expected_freq, dx, dy, dt = _make_orbit_data()
    job = _create_job(tmp_path, data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)

    spec = job.m.solitons.vortex.spectrum.gyration(method="welch", nperseg=256)
    peak = spec.peak_frequency_hz

    assert spec.frequencies.size > 0
    assert spec.power.size > 0
    assert abs(peak - expected_freq) < 0.45e9

    ax = spec.plt.power_spectrum(as_ghz=True)
    assert hasattr(ax, "plot")


def test_interface_spectrogram_and_direct_compute(tmp_path):
    data, _, dx, dy, dt = _make_orbit_data(nt=160)
    job = _create_job(tmp_path, data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)

    traj = job.m.solitons.vortex.core.track(method="centroid")
    direct_spec = compute_gyration_spectrum(traj, method="periodogram")

    specgram = job.m.solitons.vortex.spectrum.spectrogram(component="x", nperseg=64)
    ax = specgram.plt.spectrogram(as_ghz=True)

    assert direct_spec.frequencies.size > 0
    assert specgram.power.ndim == 2
    assert specgram.power.shape[0] == specgram.frequencies.size
    assert specgram.power.shape[1] == specgram.times.size
    assert hasattr(ax, "pcolormesh")
