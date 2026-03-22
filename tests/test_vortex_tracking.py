from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import zarr

from mmpp.core.job import ZarrJobResult
from mmpp.solitons.vortex.core import track_core
from mmpp.solitons.vortex.core import tracking as tracking_module
from mmpp.solitons.vortex.core.models import TrajectoryResult
from tests.fixtures.synthetic_vortex import (
    generate_vortex_mz_centered,
    generate_vortex_mz_near_edge,
)


def _make_vortex_snapshot(
    nx: int,
    ny: int,
    *,
    center_x: float,
    center_y: float,
    core_radius_px: float = 4.0,
    polarity: int = 1,
    chirality: int = 1,
) -> np.ndarray:
    x = np.arange(nx, dtype=float) - center_x
    y = np.arange(ny, dtype=float) - center_y
    x_grid, y_grid = np.meshgrid(x, y)

    radius = np.hypot(x_grid, y_grid)
    phi = np.arctan2(y_grid, x_grid)

    mz = polarity * np.exp(-(radius / core_radius_px) ** 2)
    m_perp = np.sqrt(np.clip(1.0 - mz**2, 0.0, 1.0))

    mx = -chirality * m_perp * np.sin(phi)
    my = chirality * m_perp * np.cos(phi)

    m = np.stack([mx, my, mz], axis=-1)
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norm > 1e-12, norm, 1.0)


def _make_orbit_data(
    nt: int = 48,
    nx: int = 96,
    ny: int = 96,
    *,
    dx: float = 1e-9,
    dy: float = 1e-9,
    dt: float = 5e-12,
    orbit_radius_m: float = 4e-9,
    frequency_hz: float = 0.8e9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    t = np.arange(nt, dtype=float) * dt
    center_x_px = (nx - 1) / 2.0
    center_y_px = (ny - 1) / 2.0

    x_expected = center_x_px * dx + orbit_radius_m * np.cos(2.0 * np.pi * frequency_hz * t)
    y_expected = center_y_px * dy + orbit_radius_m * np.sin(2.0 * np.pi * frequency_hz * t)

    data = np.zeros((nt, ny, nx, 3), dtype=float)
    for i in range(nt):
        cx = x_expected[i] / dx
        cy = y_expected[i] / dy
        data[i] = _make_vortex_snapshot(nx, ny, center_x=cx, center_y=cy)

    return data, x_expected, y_expected, dx, dy, dt


def _create_job(tmp_path, data: np.ndarray, *, dx: float, dy: float, dt: float):
    zarr_path = tmp_path / "vortex_tracking_test.zarr"
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
    zarr_path = tmp_path / "vortex_tracking_table_only.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    table = z.create_group("table")
    table.create_dataset("ext_coreposx", data=np.asarray(x, dtype=float))
    table.create_dataset("ext_coreposy", data=np.asarray(y, dtype=float))
    table.create_dataset("t", data=np.asarray(t, dtype=float))
    if polarity_signal is not None:
        table.create_dataset("ext_coreposz", data=np.asarray(polarity_signal, dtype=float))
    dt = float(np.median(np.diff(t))) if np.asarray(t).size >= 2 else 1e-12
    z.attrs["dx"] = 1e-9
    z.attrs["dy"] = 1e-9
    z.attrs["t_sampl"] = dt
    return ZarrJobResult(str(zarr_path), {})


def _attach_table_corepos(
    job: ZarrJobResult,
    *,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    polarity_signal: np.ndarray | None = None,
):
    z = zarr.open(job.path, mode="a")
    table = z.create_group("table")
    table.create_dataset("ext_coreposx", data=np.asarray(x, dtype=float))
    table.create_dataset("ext_coreposy", data=np.asarray(y, dtype=float))
    table.create_dataset("t", data=np.asarray(t, dtype=float))
    if polarity_signal is not None:
        table.create_dataset("ext_coreposz", data=np.asarray(polarity_signal, dtype=float))


@pytest.mark.parametrize(
    "method,max_rmse",
    [
        ("maximum", 2.6e-9),
        ("centroid", 1.6e-9),
        ("gaussian", 1.0e-9),
    ],
)
def test_track_core_methods_recover_synthetic_orbit(method: str, max_rmse: float):
    data, x_expected, y_expected, dx, dy, dt = _make_orbit_data()

    traj = track_core(data, dx, dy, dt, method=method)

    if traj.metadata.get("convention") == "up":
        ny = data.shape[1]
        y_expected_phys = (ny - 1) * dy - y_expected
    else:
        y_expected_phys = y_expected

    rmse = np.sqrt(
        np.mean((traj.x - x_expected) ** 2 + (traj.y - y_expected_phys) ** 2)
    )
    assert rmse < max_rmse


def test_trajectory_result_properties_are_available():
    data, _, _, dx, dy, dt = _make_orbit_data()

    traj = track_core(data, dx, dy, dt, method="centroid")

    assert traj.z.shape == traj.time.shape
    assert np.all(traj.r >= 0.0)
    assert traj.phi.shape == traj.time.shape
    assert traj.phi_unwrapped.shape == traj.time.shape

    vx, vy = traj.velocity
    assert vx.shape == traj.time.shape
    assert vy.shape == traj.time.shape

    omega = traj.instantaneous_frequency
    assert omega.shape == traj.time.shape
    assert traj.rotation_sense in {"CW", "CCW"}


def test_tracking_api_integration_and_plot_accessor(tmp_path):
    data, _, _, dx, dy, dt = _make_orbit_data()
    data_5d = data[:, np.newaxis, ...]
    job = _create_job(tmp_path, data_5d, dx=dx, dy=dy, dt=dt)

    traj = job.m.solitons.vortex.core.track(method="centroid")
    ax = traj.plt.xy(
        aspect="auto",
        figsize=(5, 3),
        dpi=90,
        title="Tracked XY",
        color="black",
    )
    ax_orbit = traj.plt.orbit_2d(
        show_center=True,
        figsize=(4, 4),
        dpi=90,
        xlim=(float(np.min(traj.x)), float(np.max(traj.x))),
        ylim=(float(np.min(traj.y)), float(np.max(traj.y))),
    )
    topo = job.solitons.vortex.detect()

    assert traj.time.size == data.shape[0]
    assert hasattr(ax, "plot")
    assert ax.get_aspect() == "auto"
    assert ax.get_title() == "Tracked XY"
    assert hasattr(ax_orbit, "plot")
    assert topo.state in {"vortex", "antivortex", "meron", "unknown", "skyrmion"}


def test_gaussian_tracking_falls_back_to_centroid_without_scipy(monkeypatch):
    data, _, _, dx, dy, dt = _make_orbit_data(nt=16)

    monkeypatch.setattr(tracking_module, "SCIPY_AVAILABLE", False)
    monkeypatch.setattr(tracking_module, "curve_fit", None)

    with pytest.warns(RuntimeWarning):
        traj = tracking_module.track_core(data, dx, dy, dt, method="gaussian")

    assert traj.method == "centroid"
    assert traj.metadata.get("fallback_from") == "gaussian"


def test_cwccw_from_complex_trajectory():
    t = np.linspace(0.0, 50e-9, 5000)
    r = 5e-9
    omega = 2.0 * np.pi * 1.0e9

    x_ccw = r * np.cos(omega * t)
    y_ccw = r * np.sin(omega * t)
    traj_ccw = TrajectoryResult(
        time=t,
        x=x_ccw,
        y=y_ccw,
        polarity=np.ones_like(t, dtype=int),
        method="synthetic",
        confidence=np.ones_like(t, dtype=float),
        metadata={"case": "ccw"},
    )
    assert traj_ccw.rotation_sense == "CCW"
    assert float(np.mean(traj_ccw.instantaneous_frequency)) > 0.0

    x_cw = r * np.cos(omega * t)
    y_cw = -r * np.sin(omega * t)
    traj_cw = TrajectoryResult(
        time=t,
        x=x_cw,
        y=y_cw,
        polarity=np.ones_like(t, dtype=int),
        method="synthetic",
        confidence=np.ones_like(t, dtype=float),
        metadata={"case": "cw"},
    )
    assert traj_cw.rotation_sense == "CW"
    assert float(np.mean(traj_cw.instantaneous_frequency)) < 0.0


def test_gaussian_fallback_at_edge():
    dx = dy = 1e-9
    dt = 1e-12

    m_edge = generate_vortex_mz_near_edge(Nx=64, Ny=64, core_pix=(62, 1))
    m_center = generate_vortex_mz_centered(Nx=64, Ny=64, core_pix=(32, 32))

    result_edge = track_core(
        m_edge[np.newaxis, ...],
        dx,
        dy,
        dt,
        method="gaussian",
        gaussian_roi=7,
    )
    result_center = track_core(
        m_center[np.newaxis, ...],
        dx,
        dy,
        dt,
        method="gaussian",
        gaussian_roi=7,
    )

    assert result_edge.x.size == 1
    assert result_edge.y.size == 1
    assert result_edge.metadata["gaussian_frame_fallbacks"] >= 1
    assert float(result_edge.confidence[0]) < 0.5
    assert "method_used" in result_edge.metadata
    assert result_edge.metadata["method_used"][0] == "centroid"

    assert result_center.metadata["gaussian_frame_fallbacks"] == 0
    assert float(result_center.confidence[0]) > float(result_edge.confidence[0])
    assert result_center.metadata["method_used"][0] == "gaussian"


def test_tracking_interface_accepts_roi_and_keeps_time_shape(tmp_path):
    data, _, _, dx, dy, dt = _make_orbit_data(nt=24, nx=80, ny=80)
    job = _create_job(tmp_path, data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)

    traj = job.m.vortex.track(method="gaussian", roi=(10, 70, 10, 70))

    assert traj.time.ndim == 1
    assert traj.x.shape == traj.time.shape
    assert traj.y.shape == traj.time.shape
    assert traj.metadata.get("roi") == (10, 70, 10, 70)


def test_polarity_series_contains_switch_metadata():
    nt, nx, ny = 24, 48, 48
    dx = dy = 1e-9
    dt = 1e-12

    base = _make_vortex_snapshot(nx, ny, center_x=(nx - 1) / 2.0, center_y=(ny - 1) / 2.0)
    data = np.repeat(base[np.newaxis, ...], nt, axis=0)
    data[: nt // 2, ..., 2] *= 1.0
    data[nt // 2 :, ..., 2] *= -1.0

    traj = track_core(
        data,
        dx,
        dy,
        dt,
        method="centroid",
        polarity_threshold_up=0.2,
        polarity_threshold_down=-0.2,
    )

    assert traj.metadata["p_switch_count"] >= 1
    assert len(traj.metadata["switch_times_s"]) >= 1
    assert int(traj.polarity[0]) == 1
    assert int(traj.polarity[-1]) == -1


def test_table_tracking_auto_supports_table_only_jobs(tmp_path):
    t = np.linspace(0.0, 40e-9, 129)
    x = 8e-9 * np.cos(2.0 * np.pi * 0.7e9 * t)
    y = 5e-9 * np.sin(2.0 * np.pi * 0.7e9 * t)
    polarity_signal = np.ones_like(t)
    polarity_signal[t >= 20e-9] = -1.0

    job = _create_table_job(
        tmp_path,
        x=x,
        y=y,
        t=t,
        polarity_signal=polarity_signal,
    )

    traj = job.solitons.vortex.track()

    assert traj.method == "table"
    assert np.allclose(traj.time, t)
    assert np.allclose(traj.x, x)
    assert np.allclose(traj.y, y)
    assert traj.metadata["source"] == "table"
    assert traj.metadata["requested_method"] == "auto"
    assert traj.metadata["x_column"] == "ext_coreposx"
    assert traj.metadata["y_column"] == "ext_coreposy"
    assert traj.metadata["polarity_column"] == "ext_coreposz"
    assert traj.metadata["p_switch_count"] >= 1
    assert int(traj.polarity[0]) == 1
    assert int(traj.polarity[-1]) == -1


def test_auto_prefers_time_resolved_dataset_when_available(tmp_path):
    data, x_expected, y_expected, dx, dy, dt = _make_orbit_data(nt=32)
    job = _create_job(tmp_path, data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)
    t = np.arange(x_expected.size, dtype=float) * dt
    _attach_table_corepos(
        job,
        x=x_expected + 50e-9,
        y=y_expected - 50e-9,
        t=t,
        polarity_signal=np.ones_like(t),
    )

    traj = job.solitons.vortex.track()

    assert traj.metadata["source"] == "dataset"
    assert traj.metadata["requested_method"] == "auto"
