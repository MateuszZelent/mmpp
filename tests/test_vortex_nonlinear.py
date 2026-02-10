from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import zarr

from mmpp.analytical import (
    CPPThieleModel,
    DiskGeometry,
    MaterialParams,
    ellipse_area,
    fit_omega0_N_to_fJ,
    slonczewski_mtj_efficiency,
)
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


def _make_nonlinear_orbit_data(
    nt: int = 220,
    nx: int = 72,
    ny: int = 72,
    *,
    dx: float = 1e-9,
    dy: float = 1e-9,
    dt: float = 6e-12,
    base_radius_m: float = 3.0e-9,
    mod_radius_m: float = 1.5e-9,
    f0_hz: float = 1.6e9,
    n_shift_hz: float = 0.9e9,
):
    t = np.arange(nt, dtype=float) * dt
    x0 = (nx - 1) / 2.0 * dx
    y0 = (ny - 1) / 2.0 * dy

    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.22e9 * t)
    radius = base_radius_m + mod_radius_m * envelope

    power_driver = (radius / np.max(radius)) ** 2
    omega = 2.0 * np.pi * (f0_hz + n_shift_hz * power_driver)
    phase = np.cumsum(omega) * dt

    x = x0 + radius * np.cos(phase)
    y = y0 + radius * np.sin(phase)

    data = np.zeros((nt, ny, nx, 3), dtype=float)
    for idx in range(nt):
        data[idx] = _make_vortex_snapshot(
            nx,
            ny,
            center_x=x[idx] / dx,
            center_y=y[idx] / dy,
        )

    return data, dx, dy, dt


def _create_job(tmp_path, name: str, data: np.ndarray, *, dx: float, dy: float, dt: float):
    zarr_path = tmp_path / f"{name}.zarr"
    z = zarr.open(str(zarr_path), mode="w")
    z.create_dataset("m", data=data, chunks=data.shape)
    z.attrs["dx"] = dx
    z.attrs["dy"] = dy
    z.attrs["t_sampl"] = dt
    return ZarrJobResult(str(zarr_path), {})


def test_amplitude_equation_outputs_and_plotting(tmp_path):
    data, dx, dy, dt = _make_nonlinear_orbit_data()
    job = _create_job(tmp_path, "vortex_nonlinear_amp", data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)

    traj = job.m.solitons.vortex.core.track(method="centroid")
    amp = job.m.solitons.vortex.nonlinear.amplitude_equation(trajectory=traj)

    assert amp.time.shape == traj.time.shape
    assert amp.complex_amplitude.shape == traj.time.shape
    assert np.all(np.isfinite(amp.power))
    assert np.all(amp.power >= 0.0)

    ax_power = amp.plt.power_vs_time()
    ax_complex = amp.plt.complex_plane()
    assert hasattr(ax_power, "plot")
    assert hasattr(ax_complex, "plot")


def test_slavin_tiberkevich_parameters_and_single_point_plot(tmp_path):
    data, dx, dy, dt = _make_nonlinear_orbit_data()
    job = _create_job(tmp_path, "vortex_nonlinear_st", data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)

    traj = job.m.solitons.vortex.core.track(method="centroid")
    st = job.m.solitons.vortex.nonlinear.slavin_tiberkevich(
        trajectory=traj,
        spectrum_method="periodogram",
        steady_state_fraction=0.5,
        current_a=5e-3,
    )

    assert np.isfinite(st.f_0_ghz)
    assert np.isfinite(st.N)
    assert np.isfinite(st.generation_power)
    assert np.isfinite(st.linewidth_hz)
    assert st.linewidth_hz > 0.0
    assert np.isfinite(st.quality_factor)
    assert st.quality_factor > 0.0
    assert isinstance(st.linewidth_resolution_limited, bool)

    ax = st.plt.power_vs_current()
    assert hasattr(ax, "plot")


def test_slavin_tiberkevich_batch_pipeline(tmp_path):
    params = [
        ("vortex_nonlinear_batch_a", 2.7e-9),
        ("vortex_nonlinear_batch_b", 3.1e-9),
        ("vortex_nonlinear_batch_c", 3.5e-9),
    ]

    jobs = []
    for name, base_radius in params:
        data, dx, dy, dt = _make_nonlinear_orbit_data(base_radius_m=base_radius)
        jobs.append(
            _create_job(tmp_path, name, data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)
        )

    currents = np.array([3.0e-3, 4.0e-3, 5.0e-3], dtype=float)
    batch = jobs[0].m.solitons.vortex.nonlinear.slavin_tiberkevich_batch(
        jobs,
        currents,
        spectrum_method="periodogram",
    )

    assert batch.currents.shape == currents.shape
    assert batch.powers.shape == currents.shape
    assert batch.linewidths.shape == currents.shape
    assert batch.frequencies_hz.shape == currents.shape
    assert np.all(np.isfinite(batch.powers))
    assert np.all(np.isfinite(batch.frequencies_hz))
    assert np.isfinite(batch.N)

    ax_batch = batch.plt.power_vs_current()
    ax_iface = jobs[0].m.solitons.vortex.nonlinear.plt.linewidth_vs_current()
    assert hasattr(ax_batch, "plot")
    assert hasattr(ax_iface, "plot")


def test_thiele_force_balance_and_plot_accessor(tmp_path):
    data, dx, dy, dt = _make_nonlinear_orbit_data()
    job = _create_job(tmp_path, "vortex_nonlinear_thiele_force", data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)

    traj = job.m.solitons.vortex.core.track(method="centroid")
    force = job.m.solitons.vortex.nonlinear.thiele.force_balance(
        trajectory=traj,
        Ms=8.0e5,
        thickness=20e-9,
        alpha=0.01,
        vorticity=1,
    )

    assert force.time.shape == traj.time.shape
    assert force.gyro_force.shape == (traj.time.size, 2)
    assert force.conservative_force.shape == (traj.time.size, 2)
    assert force.dissipative_force.shape == (traj.time.size, 2)
    assert np.isfinite(force.G)
    assert np.isfinite(force.D)
    assert np.isfinite(force.kappa)
    assert np.all(np.isfinite(force.residual_ratio))
    assert force.polarity in {-1, 1}

    ax_force = force.plt.force_balance()
    ax_iface = job.m.solitons.vortex.nonlinear.plt.force_balance(
        trajectory=traj,
        Ms=8.0e5,
        thickness=20e-9,
        alpha=0.01,
    )
    assert hasattr(ax_force, "plot")
    assert hasattr(ax_iface, "plot")


def test_thiele_cpp_and_cip_simulation_wrappers(tmp_path):
    pytest.importorskip("scipy")

    data, dx, dy, dt = _make_nonlinear_orbit_data(nt=120)
    job = _create_job(tmp_path, "vortex_nonlinear_thiele_sim", data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)
    thiele = job.m.solitons.vortex.nonlinear.thiele

    cpp = thiele.simulate_cpp(
        material={"Ms": 8.0e5, "alpha": 0.01, "P": 0.35},
        geometry={"R": 45e-9, "L": 20e-9},
        current_density=2.0e11,
        t_span=(0.0, 2.0e-9),
        dt=2.0e-11,
    )
    cip = thiele.simulate_cip(
        material={"Ms": 8.0e5, "alpha": 0.01, "P": 0.35},
        geometry={"R": 45e-9, "L": 20e-9},
        current_density=1.5e11,
        t_span=(0.0, 1.2e-9),
        dt=2.0e-11,
        r0=(2.0e-9, 0.0),
    )

    assert cpp.t.size > 8
    assert cip.t.size > 8
    assert np.all(np.isfinite(cpp.x))
    assert np.all(np.isfinite(cpp.y))
    assert np.all(np.isfinite(cip.x))
    assert np.all(np.isfinite(cip.y))
    assert cpp.metadata.get("source") == "mmpp.solitons.vortex.nonlinear.thiele"
    assert cip.metadata.get("source") == "mmpp.solitons.vortex.nonlinear.thiele"

    ax_cpp = cpp.plt.orbit()
    ax_cip = cip.plt.xy()
    assert hasattr(ax_cpp, "plot")
    assert hasattr(ax_cip, "plot")


def test_cpp_sde_helpers_and_threshold_prediction():
    mat = MaterialParams(Ms=8.0e5, alpha=0.01, P=0.35)
    geo = DiskGeometry(R=45e-9, L=20e-9)
    model = CPPThieleModel(material=mat, geom=geo, omega0=2.0 * np.pi * 0.75e9, N=0.22)

    area = ellipse_area(220e-9, 120e-9)
    assert np.isfinite(area)
    assert area > 0.0

    eff = slonczewski_mtj_efficiency(Pol=0.56, Lambda=1.2, cos_theta=0.5)
    assert np.isfinite(eff)
    assert eff > 0.0

    j_th = model.threshold_current_dc()
    assert np.isfinite(j_th)
    assert j_th > 0.0

    assert model.predict_frequency_dc(0.5 * j_th) is None
    f_above = model.predict_frequency_dc(1.4 * j_th, allow_edge=True)
    assert f_above is not None
    assert np.isfinite(f_above)
    assert f_above > 0.0

    sde = model.simulate_sde(
        t_span=(0.0, 1.2e-9),
        J_func=lambda _t: 1.6 * j_th,
        dt=2.0e-11,
        temperature_k=300.0,
        seed=7,
    )
    assert sde.t.size > 10
    assert np.all(np.isfinite(sde.x))
    assert np.all(np.isfinite(sde.y))
    assert sde.metadata.get("mode") == "CPP-SDE"


def test_cpp_sde_orbit_stays_near_steady_state_not_edge_clamped():
    peff = slonczewski_mtj_efficiency(Pol=0.56, Lambda=1.2, cos_theta=0.5)
    mat = MaterialParams(Ms=8.0e5, alpha=0.01, P=peff)
    geo = DiskGeometry(R=100e-9, L=20e-9)
    model = CPPThieleModel(material=mat, geom=geo, omega0=2.0 * np.pi * 0.9e9, N=0.25)

    j_th = model.threshold_current_dc()
    j_drive = 1.2 * j_th
    u_ss = model.steady_state_u(j_drive)
    assert u_ss is not None
    assert 0.0 < float(u_ss) < 1.0

    sde = model.simulate_sde(
        t_span=(0.0, 120.0e-9),
        s0=(0.0, 0.0),
        J_func=lambda _t: float(j_drive),
        dt=10.0e-12,
        temperature_k=300.0,
        noise_scale=1.0,
        seed=0,
    )
    u = np.sqrt(sde.sx**2 + sde.sy**2)
    tail = u[int(0.75 * u.size) :]
    assert np.all(np.isfinite(tail))
    assert float(np.mean(tail)) < 0.8
    assert abs(float(np.mean(tail)) - float(u_ss)) < 0.2
    assert float(np.quantile(u, 0.99)) < 0.9


def test_fit_omega0_n_to_fj_recovers_synthetic_params():
    mat = MaterialParams(Ms=8.0e5, alpha=0.01, P=0.35)
    geo = DiskGeometry(R=45e-9, L=20e-9)

    omega0_true = 2.0 * np.pi * 0.82e9
    n_true = 0.28
    model = CPPThieleModel(material=mat, geom=geo, omega0=omega0_true, N=n_true)
    j_th = model.threshold_current_dc()

    j_data = np.linspace(1.2 * j_th, 2.2 * j_th, 9)
    f_data = np.array(
        [model.predict_frequency_dc(j, allow_edge=True) for j in j_data],
        dtype=float,
    )

    fit = fit_omega0_N_to_fJ(
        j_data,
        f_data,
        material=mat,
        geom=geo,
        initial_omega0=0.7 * omega0_true,
        initial_N=0.1,
        allow_edge=True,
    )

    assert np.isfinite(fit.omega0)
    assert np.isfinite(fit.N)
    assert np.isfinite(fit.rmse_hz)
    assert fit.rmse_hz >= 0.0

    # Fallback grid (no SciPy) is less precise, so keep robust tolerances.
    assert abs(fit.omega0 - omega0_true) / omega0_true < 0.35
    assert abs(fit.N - n_true) < 0.35

    ax = fit.plt.frequency_vs_current()
    assert hasattr(ax, "plot")


def test_cpp_optimize_current_for_target_frequency(tmp_path):
    mat = MaterialParams(Ms=8.0e5, alpha=0.01, P=0.35)
    geo = DiskGeometry(R=45e-9, L=20e-9)
    omega0_true = 2.0 * np.pi * 0.9e9
    n_true = 0.22
    model = CPPThieleModel(material=mat, geom=geo, omega0=omega0_true, N=n_true)

    j_th = model.threshold_current_dc()
    j_ref = 1.7 * j_th
    f_ref = model.predict_frequency_dc(j_ref, allow_edge=True)
    assert f_ref is not None

    opt = model.optimize_current_for_target_frequency(
        float(f_ref),
        J_bounds=(1.05 * j_th, 2.5 * j_th),
        allow_edge=True,
    )
    assert np.isfinite(opt.current_density_a_per_m2)
    assert np.isfinite(opt.predicted_frequency_hz)
    assert abs(opt.predicted_frequency_hz - float(f_ref)) < 0.2e9

    data, dx, dy, dt = _make_nonlinear_orbit_data(nt=80)
    job = _create_job(tmp_path, "vortex_nonlinear_opt", data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)
    opt_iface = job.m.solitons.vortex.nonlinear.thiele.optimize_current_for_target_frequency(
        float(f_ref),
        material={"Ms": 8.0e5, "alpha": 0.01, "P": 0.35},
        geometry={"R": 45e-9, "L": 20e-9},
        omega0=omega0_true,
        N=n_true,
        J_bounds=(1.05 * j_th, 2.5 * j_th),
        allow_edge=True,
    )
    assert np.isfinite(opt_iface.current_density_a_per_m2)


def test_thiele_proxy_signal_psd_and_dashboard_entrypoint(tmp_path):
    data, dx, dy, dt = _make_nonlinear_orbit_data(nt=120)
    job = _create_job(tmp_path, "vortex_nonlinear_interactive", data[:, np.newaxis, ...], dx=dx, dy=dy, dt=dt)
    thiele = job.m.solitons.vortex.nonlinear.thiele

    traj = thiele.simulate_cpp(
        material={"Ms": 8.0e5, "alpha": 0.01, "P": 0.35},
        geometry={"R": 45e-9, "L": 20e-9},
        current_density=2.0e11,
        t_span=(0.0, 2.0e-9),
        dt=2.0e-11,
    )
    signal = thiele.proxy_signal(
        traj,
        disk_radius=45e-9,
        polarizer=(np.cos(np.deg2rad(20.0)), np.sin(np.deg2rad(20.0)), 0.0),
    )
    assert signal.shape == traj.t.shape
    assert np.all(np.isfinite(signal))

    freq, psd = thiele.proxy_psd(signal, dt=2.0e-11, method="welch")
    assert freq.ndim == 1
    assert psd.ndim == 1
    assert freq.size == psd.size
    assert freq.size > 4
    assert np.all(np.isfinite(psd))

    try:
        import ipywidgets  # noqa: F401

        widget = job.m.solitons.vortex.nonlinear.interactive_dashboard(
            fast_mode=True,
            figsize=(11.0, 3.8),
            dpi=90,
        )
        assert widget is not None
    except ImportError:
        with pytest.raises(ImportError):
            job.m.solitons.vortex.nonlinear.interactive_dashboard(
                fast_mode=True,
                figsize=(11.0, 3.8),
                dpi=90,
            )
