from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import zarr

from mmpp.core.job import ZarrJobResult


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


def _make_mode_rich_orbit_data(
    nt: int = 640,
    nx: int = 96,
    ny: int = 96,
    *,
    dx: float = 1e-9,
    dy: float = 1e-9,
    dt: float = 5e-12,
    gyration_hz: float = 1.8e9,
    breathing_hz: float = 3.6e9,
):
    t = np.arange(nt, dtype=float) * dt
    x0 = (nx - 1) / 2.0 * dx
    y0 = (ny - 1) / 2.0 * dy

    orbit_base = 4.0e-9
    breathing = 1.4e-9 * np.sin(2.0 * np.pi * breathing_hz * t)
    radius = orbit_base + breathing

    x = x0 + radius * np.cos(2.0 * np.pi * gyration_hz * t)
    y = y0 + radius * np.sin(2.0 * np.pi * gyration_hz * t)

    data = np.zeros((nt, ny, nx, 3), dtype=float)
    for i in range(nt):
        data[i] = _make_vortex_snapshot(
            nx,
            ny,
            center_x=x[i] / dx,
            center_y=y[i] / dy,
        )

    return data, gyration_hz, breathing_hz, dx, dy, dt


def _create_job(tmp_path, data: np.ndarray, *, dx: float, dy: float, dt: float):
    zarr_path = tmp_path / "vortex_modes_phase3.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    z.create_dataset("m", data=data, chunks=data.shape)
    z.attrs["dx"] = dx
    z.attrs["dy"] = dy
    z.attrs["t_sampl"] = dt
    return ZarrJobResult(str(zarr_path), {})


def _create_table_mode_job(
    tmp_path,
    *,
    nt: int = 640,
    dt: float = 5e-12,
    gyration_hz: float = 1.8e9,
    breathing_hz: float = 3.6e9,
):
    zarr_path = tmp_path / "vortex_modes_table_only.zarr"
    z = zarr.open(str(zarr_path), mode="w")

    t = np.arange(nt, dtype=float) * dt
    radius = 4.0e-9 + 1.4e-9 * np.sin(2.0 * np.pi * breathing_hz * t)
    x = radius * np.cos(2.0 * np.pi * gyration_hz * t)
    y = radius * np.sin(2.0 * np.pi * gyration_hz * t)
    core_signal = np.ones_like(t)

    table = z.create_group("table")
    table.create_dataset("t", data=t, chunks=t.shape)
    table.create_dataset("ext_coreposx", data=x, chunks=x.shape)
    table.create_dataset("ext_coreposy", data=y, chunks=y.shape)
    table.create_dataset("ext_coreposz", data=core_signal, chunks=core_signal.shape)

    end = _make_vortex_snapshot(96, 96, center_x=48.0, center_y=48.0)
    z.create_dataset("end", data=end[np.newaxis, ...], chunks=end[np.newaxis, ...].shape)

    z.attrs["dx"] = 1e-9
    z.attrs["dy"] = 1e-9
    z.attrs["t_sampl"] = dt
    return ZarrJobResult(str(zarr_path), {})


def test_modes_classify_all_and_single_frequency(tmp_path):
    data, gyr_f, breathing_f, dx, dy, dt = _make_mode_rich_orbit_data()
    job = _create_job(tmp_path, data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)

    modes = job.m.solitons.vortex.modes.classify_all(max_modes=8, min_prominence=0.03)

    assert len(modes) >= 1
    assert any(mode.mode_type in {"gyration", "breathing", "azimuthal"} for mode in modes)

    near_gyr = job.m.solitons.vortex.modes.classify(f=gyr_f * 1e-9, unit="ghz")
    near_breath = job.m.solitons.vortex.modes.classify(f=breathing_f, unit="hz")

    assert abs(near_gyr.frequency_hz - gyr_f) < 0.5e9
    assert abs(near_breath.frequency_hz - breathing_f) < 0.8e9


def test_modes_support_table_only_tracking(tmp_path):
    job = _create_table_mode_job(tmp_path)

    modes = job.solitons.vortex.modes.classify_all(max_modes=8, min_prominence=0.03)
    gyro = job.solitons.vortex.modes.gyration

    assert len(modes) >= 1
    assert any(mode.mode_type in {"gyration", "breathing", "azimuthal"} for mode in modes)
    assert gyro is None or gyro.mode_type == "gyration"


def test_modes_plot_accessor_and_mode_table(tmp_path):
    data, _, _, dx, dy, dt = _make_mode_rich_orbit_data(nt=512)
    job = _create_job(tmp_path, data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)

    ax = job.m.solitons.vortex.modes.plt.mode_map(
        figsize=(6, 3),
        dpi=100,
        title="Mode map",
        color="tab:blue",
        edgecolor="black",
        alpha=0.8,
    )
    rows = job.m.solitons.vortex.modes.plt.mode_table()

    assert hasattr(ax, "bar")
    assert ax.get_title() == "Mode map"
    assert isinstance(rows, list)
    if rows:
        assert "mode" in rows[0]
        assert "f_ghz" in rows[0]
