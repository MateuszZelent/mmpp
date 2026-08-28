from __future__ import annotations

import numpy as np
import zarr

from mmpp.core.job import ZarrJobResult
from mmpp.solitons.vortex import XYConvention
from mmpp.solitons.vortex.topology import detect_topology
from tests.fixtures.synthetic_vortex import generate_synthetic_vortex


def _make_vortex_snapshot(
    nx: int = 96,
    ny: int = 96,
    *,
    center_x: float | None = None,
    center_y: float | None = None,
    core_radius_px: float = 4.0,
    polarity: int = 1,
    chirality: int = 1,
) -> np.ndarray:
    if center_x is None:
        center_x = (nx - 1) / 2.0
    if center_y is None:
        center_y = (ny - 1) / 2.0

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


def _create_job(tmp_path, data: np.ndarray, *, dx: float, dy: float, dt: float):
    zarr_path = tmp_path / "vortex_topology_test.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    z.create_dataset("m", data=data, chunks=data.shape)
    z.attrs["dx"] = dx
    z.attrs["dy"] = dy
    z.attrs["t_sampl"] = dt
    return ZarrJobResult(str(zarr_path), {})


def test_detect_topology_finite_diff_returns_expected_invariants():
    snapshot = _make_vortex_snapshot(polarity=1, chirality=1)
    dx = dy = 1e-9

    result = detect_topology(snapshot, dx, dy, method="finite_diff")

    assert result.polarity == 1
    assert result.vorticity == 1
    assert result.chirality == 1
    assert result.state == "vortex"
    assert 0.30 < result.Q < 0.70
    assert result.is_consistent
    assert result.topological_density.shape == snapshot.shape[:2]


def test_detect_topology_berg_luscher_returns_expected_invariants():
    snapshot = _make_vortex_snapshot(polarity=1, chirality=1)
    dx = dy = 1e-9

    result = detect_topology(snapshot, dx, dy, method="berg_luscher")

    assert result.polarity == 1
    assert result.vorticity == 1
    assert result.chirality == 1
    assert result.state == "vortex"
    assert 0.35 < result.Q < 0.65
    assert result.is_consistent
    assert 0.0 <= result.chirality_confidence <= 1.0
    assert result.convention in {"up", "down"}


def test_berg_luscher_q_sign_for_positive_and_negative_polarity():
    dx = dy = 1e-9
    m_pos = generate_synthetic_vortex(Nx=96, Ny=96, p=1, w=1)
    m_neg = generate_synthetic_vortex(Nx=96, Ny=96, p=-1, w=1)

    q_pos = detect_topology(m_pos, dx, dy, method="berg_luscher").Q
    q_neg = detect_topology(m_neg, dx, dy, method="berg_luscher").Q

    assert abs(q_pos - 0.5) < 0.02
    assert abs(q_neg + 0.5) < 0.02


def test_topology_convention_parameter_is_supported():
    dx = dy = 1e-9
    m = generate_synthetic_vortex(Nx=96, Ny=96, p=1, w=1)
    result_down = detect_topology(
        m,
        dx,
        dy,
        method="berg_luscher",
        convention=XYConvention(y_axis="down"),
    )
    result_up = detect_topology(
        m,
        dx,
        dy,
        method="berg_luscher",
        convention=XYConvention(y_axis="up"),
    )

    assert abs(result_down.Q - 0.5) < 0.02
    assert abs(result_up.Q - 0.5) < 0.02
    assert result_down.convention == "down"
    assert result_up.convention == "up"


def test_detect_topology_accepts_5d_input_with_default_frame_and_z_layer():
    frame0 = _make_vortex_snapshot(polarity=1)
    frame1 = _make_vortex_snapshot(polarity=-1)

    data = np.stack([frame0, frame1], axis=0)
    data_5d = np.stack([0.8 * data, data], axis=1)

    dx = dy = 1e-9
    result = detect_topology(data_5d, dx, dy, method="berg_luscher")

    assert result.polarity == 1
    assert result.state == "vortex"


def test_topology_interface_defaults_to_frame_zero(tmp_path):
    frame0 = _make_vortex_snapshot(polarity=1)
    frame1 = _make_vortex_snapshot(polarity=-1)
    data_5d = np.stack([np.stack([frame0], axis=0), np.stack([frame1], axis=0)], axis=0)

    job = _create_job(tmp_path, data_5d, dx=1e-9, dy=1e-9, dt=1e-12)

    result_default = job.m.solitons.vortex.topology.detect()
    result_frame1 = job.m.solitons.vortex.topology.detect(frame=1)

    assert result_default.polarity == 1
    assert result_frame1.polarity == -1


def test_job_level_solitons_interface_detects_topology(tmp_path):
    snapshot = _make_vortex_snapshot(polarity=1)
    data_4d = snapshot[np.newaxis, ...]
    job = _create_job(tmp_path, data_4d, dx=1e-9, dy=1e-9, dt=1e-12)

    result = job.solitons.vortex.detect()

    assert result.state == "vortex"
    assert result.polarity == 1


def test_dataset_vortex_alias_works(tmp_path):
    snapshot = _make_vortex_snapshot(polarity=1)
    data_4d = snapshot[np.newaxis, ...]
    job = _create_job(tmp_path, data_4d, dx=1e-9, dy=1e-9, dt=1e-12)

    result = job.m.vortex.detect()

    assert result.state == "vortex"


def test_topology_cache_manifest_written_to_zarr_attrs(tmp_path):
    snapshot = _make_vortex_snapshot(polarity=1)
    data_4d = snapshot[np.newaxis, ...]
    job = _create_job(tmp_path, data_4d, dx=1e-9, dy=1e-9, dt=1e-12)

    _ = job.m.vortex.topology.detect()

    manifest = job.z.attrs.get("_mmpp_solitons_cache_manifest", {})
    assert "topology" in manifest
    assert len(manifest["topology"]) >= 1
