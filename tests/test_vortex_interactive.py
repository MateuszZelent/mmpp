from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import zarr

from mmpp.core.job import ZarrJobResult


def _make_job(tmp_path):
    zarr_path = tmp_path / "vortex_interactive_test.zarr"
    z = zarr.open(str(zarr_path), mode="w")

    nt, ny, nx = 48, 40, 40
    dx = dy = 1e-9
    dt = 8e-12
    t = np.arange(nt, dtype=float) * dt

    data = np.zeros((nt, 1, ny, nx, 3), dtype=float)
    x0 = (nx - 1) * 0.5 + 3.5 * np.cos(2.0 * np.pi * 0.8e9 * t)
    y0 = (ny - 1) * 0.5 + 3.5 * np.sin(2.0 * np.pi * 0.8e9 * t)

    xx = np.arange(nx, dtype=float)[None, :]
    yy = np.arange(ny, dtype=float)[:, None]
    for i in range(nt):
        radius = np.hypot(xx - x0[i], yy - y0[i])
        phi = np.arctan2(yy - y0[i], xx - x0[i])
        mz = np.exp(-(radius / 3.0) ** 2)
        mperp = np.sqrt(np.clip(1.0 - mz**2, 0.0, 1.0))
        data[i, 0, :, :, 0] = -mperp * np.sin(phi)
        data[i, 0, :, :, 1] = mperp * np.cos(phi)
        data[i, 0, :, :, 2] = mz

    z.create_dataset("m", data=data, chunks=data.shape)
    z.attrs["dx"] = dx
    z.attrs["dy"] = dy
    z.attrs["t_sampl"] = dt
    return ZarrJobResult(str(zarr_path), {})


def test_trajectory_interactive_snapshot_fallback_builds_controls(tmp_path):
    job = _make_job(tmp_path)
    traj = job.m.vortex.track(method="centroid")

    fig = traj.plt.interactive(
        snapshot=True,
        toolbar=False,
        snapshot_component="snapshot",
        fps=20,
    )

    assert fig is not None
    assert hasattr(fig, "_mmpp_interactive")
    controls = fig._mmpp_interactive
    assert "slider" in controls
    assert "play_button" in controls
    assert len(fig.axes) >= 3  # orbit + snapshot + slider/button axes
