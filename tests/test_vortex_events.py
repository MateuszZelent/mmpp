from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import zarr

from mmpp.core.job import ZarrJobResult
from mmpp.solitons.vortex.core.models import TrajectoryResult
from mmpp.solitons.vortex.health import check_core_health


def _make_vortex_snapshot(
    nx: int,
    ny: int,
    *,
    center_x: float,
    center_y: float,
    polarity: int,
    core_radius_px: float = 3.5,
) -> np.ndarray:
    x = np.arange(nx, dtype=float) - center_x
    y = np.arange(ny, dtype=float) - center_y
    x_grid, y_grid = np.meshgrid(x, y)

    radius = np.hypot(x_grid, y_grid)
    phi = np.arctan2(y_grid, x_grid)

    mz = float(polarity) * np.exp(-((radius / core_radius_px) ** 2))
    m_perp = np.sqrt(np.clip(1.0 - mz**2, 0.0, 1.0))

    mx = -m_perp * np.sin(phi)
    my = m_perp * np.cos(phi)
    m = np.stack([mx, my, mz], axis=-1)
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norm > 1e-12, norm, 1.0)


def _make_polarity_switch_data(
    *,
    nt: int = 120,
    nx: int = 64,
    ny: int = 64,
    dx: float = 1.0e-9,
    dy: float = 1.0e-9,
    dt: float = 8.0e-12,
):
    x0 = (nx - 1) / 2.0
    y0 = (ny - 1) / 2.0
    data = np.zeros((nt, ny, nx, 3), dtype=float)
    for idx in range(nt):
        p = 1 if idx < nt // 2 else -1
        data[idx] = _make_vortex_snapshot(nx, ny, center_x=x0, center_y=y0, polarity=p)
    return data, dx, dy, dt


def _create_job(
    tmp_path, name: str, data: np.ndarray, *, dx: float, dy: float, dt: float
):
    zarr_path = tmp_path / f"{name}.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    z.create_dataset("m", data=data, chunks=data.shape)
    z.attrs["dx"] = dx
    z.attrs["dy"] = dy
    z.attrs["t_sampl"] = dt
    return ZarrJobResult(str(zarr_path), {})


def _create_table_only_job(
    tmp_path, name: str, *, dt: float = 8.0e-12, diameter: float = 80.0e-9
):
    zarr_path = tmp_path / f"{name}.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    table = z.create_group("table")
    t = np.arange(16, dtype=float) * dt
    zeros = np.zeros_like(t)
    ones = np.ones_like(t)
    table.create_dataset("t", data=t, chunks=t.shape)
    table.create_dataset("ext_coreposx", data=zeros, chunks=zeros.shape)
    table.create_dataset("ext_coreposy", data=zeros, chunks=zeros.shape)
    table.create_dataset("ext_coreposz", data=ones, chunks=ones.shape)
    z.attrs["dx"] = 1.0e-9
    z.attrs["dy"] = 1.0e-9
    z.attrs["t_sampl"] = dt
    z.attrs["D"] = diameter
    return ZarrJobResult(str(zarr_path), {})


def test_events_polarity_switch_detection_and_timeline_plot(tmp_path):
    data, dx, dy, dt = _make_polarity_switch_data()
    job = _create_job(
        tmp_path, "vortex_events_switch", data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt
    )

    traj = job.m.solitons.vortex.core.track(method="centroid")
    switches = job.m.solitons.vortex.events.polarity_switches(
        trajectory=traj,
        threshold=0.5,
        refractory=0.0,
    )

    assert len(switches) >= 1
    assert switches[0].from_p in {-1, 1}
    assert switches[0].to_p in {-1, 1}
    assert switches[0].from_p != switches[0].to_p

    ax = job.m.solitons.vortex.events.plt.event_timeline(
        trajectory=traj,
        figsize=(7, 3),
        dpi=100,
        title="Event timeline",
        linewidth=1.0,
    )
    assert hasattr(ax, "plot")
    assert ax.get_title() == "Event timeline"


def test_events_state_switches_and_dwell_times(tmp_path):
    data, dx, dy, dt = _make_polarity_switch_data(nt=40)
    job = _create_job(
        tmp_path, "vortex_events_states", data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt
    )

    time = np.linspace(0.0, 20.0e-9, 400)
    omega = 2.0 * np.pi * 1.2e9
    radius = np.concatenate(
        [
            np.full(140, 2.0e-9),
            np.full(120, 8.5e-9),
            np.full(140, 2.2e-9),
        ]
    )
    x = radius * np.cos(omega * time)
    y = radius * np.sin(omega * time)
    traj = TrajectoryResult(
        time=time,
        x=x,
        y=y,
        polarity=np.ones_like(time, dtype=int),
        method="synthetic",
        confidence=np.ones_like(time, dtype=float),
        metadata={"source": "test"},
    )

    transitions = job.m.solitons.vortex.events.state_switches(
        trajectory=traj,
        radius_threshold=0.45,
        disk_radius=10e-9,
        min_dwell_periods=2,
        refractory=0.0,
    )
    assert len(transitions) >= 2
    assert transitions[0].from_state in {"G-state", "C-state"}
    assert transitions[0].to_state in {"G-state", "C-state"}
    assert transitions[0].from_state != transitions[0].to_state

    dwell = job.m.solitons.vortex.events.dwell_times(
        state="G-state",
        trajectory=traj,
        radius_threshold=0.45,
        disk_radius=10e-9,
        min_dwell_periods=2,
    )
    assert dwell.count >= 1
    assert np.isfinite(dwell.mean_dwell_time)
    assert dwell.mean_dwell_time > 0.0

    ax = dwell.plt.dwell_histogram(
        figsize=(5, 3),
        dpi=90,
        title="Dwell histogram",
        color="tab:green",
        alpha=0.7,
    )
    assert hasattr(ax, "hist")
    assert ax.get_title() == "Dwell histogram"


def test_events_core_expulsion_detection(tmp_path):
    data, dx, dy, dt = _make_polarity_switch_data(nt=30)
    job = _create_job(
        tmp_path,
        "vortex_events_expulsion",
        data[:, np.newaxis, ...],
        dx=dx,
        dy=dy,
        dt=dt,
    )

    time = np.linspace(0.0, 8.0e-9, 200)
    radius = np.linspace(1.0e-9, 25.0e-9, 200)
    traj = TrajectoryResult(
        time=time,
        x=radius,
        y=np.zeros_like(radius),
        polarity=np.ones_like(time, dtype=int),
        method="synthetic",
        confidence=np.ones_like(time, dtype=float),
        metadata={"source": "test"},
    )

    events = job.m.solitons.vortex.events.core_expulsions(
        trajectory=traj,
        disk_radius=20.0e-9,
        center=(0.0, 0.0),
        expulsion_ratio=0.9,
        refractory=0.0,
    )
    assert len(events) >= 1
    assert events[0].radius >= events[0].threshold
    assert events[0].time >= 0.0


def test_events_core_expulsion_infers_radius_from_diameter_attr(tmp_path):
    job = _create_table_only_job(
        tmp_path, "vortex_events_table_radius", diameter=80.0e-9
    )

    time = np.linspace(0.0, 8.0e-9, 200)
    radius = np.linspace(1.0e-9, 39.0e-9, 200)
    traj = TrajectoryResult(
        time=time,
        x=radius,
        y=np.zeros_like(radius),
        polarity=np.ones_like(time, dtype=int),
        method="synthetic",
        confidence=np.ones_like(time, dtype=float),
        metadata={"source": "test"},
    )

    events = job.solitons.vortex.events.core_expulsions(
        trajectory=traj,
        expulsion_ratio=0.95,
        refractory=0.0,
    )

    assert len(events) >= 1
    assert abs(events[0].threshold - 38.0e-9) < 1e-12


def test_core_health_samples_the_tracked_off_center_core():
    class Dataset:
        def __init__(self, values):
            self.values = values

        def numpy(self, *, copy=False):
            return self.values

    class Job:
        attrs = {"dx": 1.0e-9, "dy": 1.0e-9}

        def __init__(self, values):
            self.m = Dataset(values)

    values = np.zeros((2, 32, 32, 3), dtype=float)
    values[:, 9:12, 19:22, 2] = 1.0
    trajectory = TrajectoryResult(
        time=np.array([0.0, 1.0e-12]),
        x=np.array([20.0e-9, 20.0e-9]),
        y=np.array([10.0e-9, 10.0e-9]),
        polarity=np.ones(2, dtype=int),
        method="synthetic",
        confidence=np.ones(2),
        metadata={"y_axis": "down"},
    )

    health = check_core_health(
        Job(values),
        trajectory=trajectory,
        disk_radius=20.0e-9,
        disk_center=(15.5e-9, 15.5e-9),
    )

    assert health.annihilated is False
    assert health.mz_initial > 0.05
    assert health.mz_final > 0.05
    assert health.is_healthy is True


def test_core_health_does_not_invent_boundary_geometry():
    class Dataset:
        def numpy(self, *, copy=False):
            return np.zeros((2, 32, 32, 3), dtype=float)

    class Job:
        attrs = {}
        m = Dataset()

    trajectory = TrajectoryResult(
        time=np.array([0.0, 1.0e-12]),
        x=np.array([1.0e-9, 2.0e-9]),
        y=np.array([1.0e-9, 2.0e-9]),
        polarity=np.ones(2, dtype=int),
        method="synthetic",
        confidence=np.ones(2),
        metadata={"y_axis": "down"},
    )

    health = check_core_health(Job(), trajectory=trajectory, disk_radius=20.0e-9)

    assert health.min_wall_distance_frac is None
    assert health.annihilated is False
    assert health.is_healthy is None
    assert health.status == "unavailable"


def test_topological_charge_uses_one_physical_y_orientation_across_soliton_apis():
    from mmpp.solitons._coordinates import XYConvention
    from mmpp.solitons._topology import berg_luscher_Q, topological_density_fd
    from mmpp.solitons.skyrmion._core import detect_skyrmion
    from mmpp.solitons.skyrmion.config import SkyrmionTopologyConfig
    from mmpp.solitons.vortex.topology.detection import detect_topology

    n = 61
    spacing = 1e-9
    yy, xx = np.indices((n, n))
    x = (xx - (n - 1) / 2.0) * spacing
    y = ((n - 1) - yy - (n - 1) / 2.0) * spacing
    radius = np.hypot(x, y)
    phi = np.arctan2(y, x)
    theta = 2.0 * np.arctan(np.exp(-(radius - 12e-9) / (2e-9)))
    field = np.stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ),
        axis=-1,
    )
    convention = XYConvention(y_axis="up")

    q_shared_bl = berg_luscher_Q(field, convention=convention)
    q_shared_fd = topological_density_fd(
        field, spacing, spacing, convention=convention
    )[1]
    q_skyrmion_bl = detect_skyrmion(field, spacing, spacing, convention=convention).Q
    q_skyrmion_fd = detect_skyrmion(
        field,
        spacing,
        spacing,
        convention=convention,
        config=SkyrmionTopologyConfig(method="finite_diff"),
    ).Q
    q_vortex_bl = detect_topology(
        field, spacing, spacing, method="berg_luscher", convention=convention
    ).Q
    q_vortex_fd = detect_topology(
        field, spacing, spacing, method="finite_diff", convention=convention
    ).Q

    assert np.isclose(q_shared_bl, -1.0, atol=1e-5)
    assert np.isclose(q_shared_bl, q_skyrmion_bl)
    assert np.isclose(q_shared_bl, q_vortex_bl)
    assert q_shared_fd < -0.9
    assert np.isclose(q_shared_fd, q_skyrmion_fd, atol=1e-4)
    assert np.isclose(q_shared_fd, q_vortex_fd, atol=1e-4)

    reversed_q = berg_luscher_Q(field[::-1], convention=XYConvention(y_axis="down"))
    assert np.isclose(reversed_q, q_shared_bl)


def test_table_tracking_marks_missing_core_polarity_unknown():
    from mmpp.solitons.vortex.numerical.core.interface import _track_core_from_table

    class TableJob:
        attrs = {"t_sampl": 1e-12}

        def __init__(self):
            self.table = {
                "ext_coreposx": np.array([0.0, 1.0, 2.0]),
                "ext_coreposy": np.array([0.0, 0.0, 0.0]),
                "t": np.array([0.0, 1e-12, 2e-12]),
            }

        def __contains__(self, key):
            return key == "table"

        def __getitem__(self, key):
            return self.table if key == "table" else self.table[key]

    trajectory = _track_core_from_table(
        TableJob(),
        polarity_threshold_up=0.3,
        polarity_threshold_down=-0.3,
    )
    assert np.array_equal(trajectory.polarity, np.zeros(3, dtype=int))
    assert not trajectory.polarity_known.any()
    assert trajectory.metadata["polarity_status"] == "unavailable"
    assert np.array_equal(trajectory.metadata["polarity_confidence"], np.zeros(3))


def test_steady_state_fallback_tail_is_not_reported_as_detected():
    from mmpp.solitons.vortex._shared.models import TrajectoryResult
    from mmpp.solitons.vortex.trajectory.steady_state import extract_steady_state

    n = 120
    time = np.arange(n, dtype=float) * 1e-12
    unstable = TrajectoryResult(
        time=time,
        x=np.linspace(0.0, 1e-6, n) ** 2,
        y=np.zeros(n),
        polarity=np.ones(n, dtype=int),
        method="test",
        confidence=np.ones(n),
    )
    selected = extract_steady_state(unstable, threshold=0.01, window=9, min_samples=20)
    assert not selected.metadata["steady_state"]
    assert not selected.metadata["steady_state_detected"]
    assert selected.metadata["steady_state_status"] == "not_detected"
    assert selected.time.size == 20


def test_trajectory_frequency_exposes_hz_and_angular_frequency_separately():
    from mmpp.solitons.vortex._shared.models import TrajectoryResult

    frequency = 5e9
    time = np.linspace(0.0, 1e-9, 1001)
    trajectory = TrajectoryResult(
        time=time,
        x=np.cos(2.0 * np.pi * frequency * time),
        y=np.sin(2.0 * np.pi * frequency * time),
        polarity=np.ones(time.size, dtype=int),
        method="test",
        confidence=np.ones(time.size),
    )
    assert np.isclose(
        np.mean(trajectory.instantaneous_angular_frequency),
        2.0 * np.pi * frequency,
        rtol=1e-3,
    )
    assert np.isclose(
        np.mean(trajectory.instantaneous_frequency_hz), frequency, rtol=1e-3
    )
