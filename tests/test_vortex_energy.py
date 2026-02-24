from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import zarr

from mmpp.core.job import ZarrJobResult
from mmpp.solitons.vortex._shared.models import TrajectoryResult


def _create_energy_job(tmp_path, *, with_energy_columns: bool):
    zarr_path = tmp_path / "vortex_energy_test.zarr"
    z = zarr.open(str(zarr_path), mode="w")

    nt, ny, nx = 64, 16, 16
    data = np.zeros((nt, 1, ny, nx, 3), dtype=float)
    data[..., 2] = 1.0
    z.create_dataset("m", data=data, chunks=data.shape)

    table = z.create_group("table")
    t = np.arange(nt, dtype=float) * 1e-12
    table.create_dataset("t", data=t, chunks=t.shape)
    table.create_dataset("mx", data=np.sin(2.0 * np.pi * 0.4e9 * t), chunks=t.shape)

    if with_energy_columns:
        e_ex = 1.0e-18 + 0.2e-18 * np.sin(2.0 * np.pi * 0.6e9 * t)
        e_demag = 2.5e-18 + 0.1e-18 * np.cos(2.0 * np.pi * 0.3e9 * t)
        e_zee = 0.8e-18 + 0.05e-18 * np.sin(2.0 * np.pi * 0.9e9 * t + 0.3)
        e_total = e_ex + e_demag + e_zee
        table.create_dataset("E_ex", data=e_ex, chunks=t.shape)
        table.create_dataset("E_demag", data=e_demag, chunks=t.shape)
        table.create_dataset("E_Zeeman", data=e_zee, chunks=t.shape)
        table.create_dataset("E_total", data=e_total, chunks=t.shape)

    z.attrs["dx"] = 1e-9
    z.attrs["dy"] = 1e-9
    z.attrs["t_sampl"] = 1e-12

    return ZarrJobResult(str(zarr_path), {})


def test_energy_time_resolved_reads_table_channels(tmp_path):
    job = _create_energy_job(tmp_path, with_energy_columns=True)
    result = job.m.vortex.energy.time_resolved()

    assert "Vortex Energy Interface" in job.m.vortex.energy._repr_html_()
    assert "EnergyTimeSeriesResult" in result._repr_html_()
    assert result.time.ndim == 1
    assert "E_ex" in result.channels
    assert "E_total" in result.channels
    assert result.total_energy.shape == result.time.shape

    ax = result.plt.time_resolved()
    assert hasattr(ax, "plot")


def test_energy_warns_when_channels_missing(tmp_path):
    job = _create_energy_job(tmp_path, with_energy_columns=False)
    with pytest.warns(RuntimeWarning):
        result = job.m.vortex.energy.time_resolved(force=True)

    assert result.channels == {}
    assert result.time.size == 0


def test_energy_strict_mode_raises_on_missing_channels(tmp_path):
    job = _create_energy_job(tmp_path, with_energy_columns=False)
    with pytest.raises(ValueError, match="No energy channels found"):
        job.m.vortex.energy.time_resolved(strict=True, force=True)


def test_energy_potential_and_pinning_from_boltzmann(tmp_path):
    job = _create_energy_job(tmp_path, with_energy_columns=True)

    t = np.linspace(0.0, 60e-9, 3000)
    omega = 2.0 * np.pi * 0.75e9
    r1 = 4.5e-9 + 0.4e-9 * np.sin(omega * t[:1500])
    r2 = 7.5e-9 + 0.5e-9 * np.sin(omega * t[1500:])
    radius = np.concatenate([r1, r2])
    phase = omega * t
    x = radius * np.cos(phase)
    y = radius * np.sin(phase)
    traj = TrajectoryResult(
        time=t,
        x=x,
        y=y,
        polarity=np.ones_like(t, dtype=int),
        method="synthetic",
        confidence=np.ones_like(t, dtype=float),
    )

    potential = job.m.vortex.energy.potential(
        trajectory=traj,
        method="boltzmann",
        bins=48,
        force=True,
    )
    assert "EffectivePotentialResult" in potential._repr_html_()
    assert potential.radius_m.size > 0
    assert potential.potential_j.shape == potential.radius_m.shape
    assert potential.method == "boltzmann"

    pinning = job.m.vortex.energy.pinning(
        potential=potential,
        min_depth_fraction=0.0,
        force=True,
    )
    assert "PinningResult" in pinning._repr_html_()
    assert len(pinning.sites) >= 1
    ax = pinning.plt.potential_with_sites()
    assert hasattr(ax, "plot")


def test_energy_potential_energy_bin_mode(tmp_path):
    job = _create_energy_job(tmp_path, with_energy_columns=True)

    nt = 64
    t = np.arange(nt, dtype=float) * 1e-12
    radius = 5.0e-9 + 0.7e-9 * np.sin(2.0 * np.pi * 0.35e9 * t)
    x = radius * np.cos(2.0 * np.pi * 0.35e9 * t)
    y = radius * np.sin(2.0 * np.pi * 0.35e9 * t)
    traj = TrajectoryResult(
        time=t,
        x=x,
        y=y,
        polarity=np.ones_like(t, dtype=int),
        method="synthetic",
        confidence=np.ones_like(t, dtype=float),
    )

    potential = job.m.vortex.energy.potential(
        trajectory=traj,
        method="energy_bin",
        bins=24,
        force=True,
    )
    assert potential.method == "energy_bin"
    assert potential.radius_m.size > 0
