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
    polarity: int = 1,
    chirality: int = 1,
) -> np.ndarray:
    x = np.arange(nx, dtype=float) - center_x
    y = np.arange(ny, dtype=float) - center_y
    x_grid, y_grid = np.meshgrid(x, y)

    radius = np.hypot(x_grid, y_grid)
    phi = np.arctan2(y_grid, x_grid)

    mz = polarity * np.exp(-((radius / core_radius_px) ** 2))
    m_perp = np.sqrt(np.clip(1.0 - mz**2, 0.0, 1.0))

    mx = -chirality * m_perp * np.sin(phi)
    my = chirality * m_perp * np.cos(phi)

    m = np.stack([mx, my, mz], axis=-1)
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norm > 1e-12, norm, 1.0)


def _make_elliptic_orbit_data(
    nt: int = 256,
    nx: int = 96,
    ny: int = 96,
    *,
    dx: float = 1e-9,
    dy: float = 1e-9,
    dt: float = 5e-12,
    radius_x_m: float = 7e-9,
    radius_y_m: float = 3e-9,
    frequency_hz: float = 3.0e9,
):
    t = np.arange(nt, dtype=float) * dt

    x0_px = (nx - 1) / 2.0
    y0_px = (ny - 1) / 2.0

    x_expected = x0_px * dx + radius_x_m * np.cos(2.0 * np.pi * frequency_hz * t)
    y_expected = y0_px * dy + radius_y_m * np.sin(2.0 * np.pi * frequency_hz * t)

    data = np.zeros((nt, ny, nx, 3), dtype=float)
    for i in range(nt):
        data[i] = _make_vortex_snapshot(
            nx,
            ny,
            center_x=x_expected[i] / dx,
            center_y=y_expected[i] / dy,
        )

    return data, x_expected, y_expected, frequency_hz, dx, dy, dt


def _create_job(tmp_path, data: np.ndarray, *, dx: float, dy: float, dt: float):
    zarr_path = tmp_path / "vortex_trajectory_phase2.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    z.create_dataset("m", data=data, chunks=data.shape)
    z.attrs["dx"] = dx
    z.attrs["dy"] = dy
    z.attrs["t_sampl"] = dt
    return ZarrJobResult(str(zarr_path), {})


def _create_table_job(
    tmp_path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    polarity_signal: np.ndarray | None = None,
):
    zarr_path = tmp_path / "vortex_trajectory_table_only.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    table = z.create_group("table")
    table.create_dataset("ext_coreposx", data=np.asarray(x, dtype=float))
    table.create_dataset("ext_coreposy", data=np.asarray(y, dtype=float))
    table.create_dataset("t", data=np.asarray(t, dtype=float))
    if polarity_signal is not None:
        table.create_dataset(
            "ext_coreposz", data=np.asarray(polarity_signal, dtype=float)
        )
    dt = float(np.median(np.diff(t))) if np.asarray(t).size >= 2 else 1e-12
    z.attrs["dx"] = 1e-9
    z.attrs["dy"] = 1e-9
    z.attrs["t_sampl"] = dt
    return ZarrJobResult(str(zarr_path), {})


def test_orbit_fit_phase_and_plot_accessors(tmp_path):
    data, _, _, freq_hz, dx, dy, dt = _make_elliptic_orbit_data()
    data_5d = data[:, np.newaxis, ...]
    job = _create_job(tmp_path, data_5d, dx=dx, dy=dy, dt=dt)

    traj = job.m.solitons.vortex.core.track(method="gaussian")
    orbit = job.m.solitons.vortex.trajectory.orbit.fit()
    phase = job.m.solitons.vortex.trajectory.phase

    assert orbit.semi_major > orbit.semi_minor
    assert orbit.eccentricity > 0.2
    assert orbit.radius > 0.0

    freq_inst = phase.frequency(method="complex", unit="hz")
    freq_mean = float(np.mean(np.abs(freq_inst[10:])))
    assert abs(freq_mean - freq_hz) < 0.45e9

    ax_orbit = traj.plt.orbit_2d(
        figsize=(5, 5),
        dpi=90,
        xlim=(-20e-9, 20e-9),
        ylim=(-20e-9, 20e-9),
        title="Orbit 2D",
    )
    fig_overview = traj.plt.overview()
    ax_phase = phase.plt.frequency_vs_time(
        unit="ghz", figsize=(6, 3), dpi=100, grid=True
    )
    ax_portrait = phase.plt.phase_portrait(figsize=(4, 4), dpi=90, aspect="equal")

    assert hasattr(ax_orbit, "plot")
    assert hasattr(ax_phase, "plot")
    assert hasattr(ax_portrait, "plot")
    assert ax_orbit.get_title() == "Orbit 2D"
    assert hasattr(fig_overview, "savefig")


def test_filtered_and_steady_state_pipeline(tmp_path):
    data, _, _, _, dx, dy, dt = _make_elliptic_orbit_data(nt=96)
    job = _create_job(tmp_path, data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)

    traj_raw = job.m.solitons.vortex.trajectory.raw
    traj_filtered = job.m.solitons.vortex.trajectory.filtered(method="savgol", window=9)
    traj_ss = job.m.solitons.vortex.trajectory.steady_state(threshold=0.08)

    assert traj_filtered.time.shape == traj_raw.time.shape
    assert "filter_method" in traj_filtered.metadata

    assert traj_ss.time.size <= traj_raw.time.size
    assert "steady_state_start_index" in traj_ss.metadata


def test_table_only_trajectory_supports_orbit_pipeline(tmp_path):
    t = np.linspace(0.0, 60e-9, 241)
    x = 9e-9 * np.cos(2.0 * np.pi * 0.85e9 * t)
    y = 4e-9 * np.sin(2.0 * np.pi * 0.85e9 * t)
    polarity_signal = np.full_like(t, 0.95)

    job = _create_table_job(
        tmp_path,
        x=x,
        y=y,
        t=t,
        polarity_signal=polarity_signal,
    )

    traj = job.solitons.vortex.trajectory.raw
    orbit = job.solitons.vortex.trajectory.orbit.fit()
    phase = job.solitons.vortex.trajectory.phase

    assert traj.method == "table"
    assert orbit.semi_major > orbit.semi_minor
    assert orbit.radius > 0.0

    freq_inst = phase.frequency(method="complex", unit="hz")
    freq_mean = float(np.mean(np.abs(freq_inst[10:])))
    assert abs(freq_mean - 0.85e9) < 0.2e9
