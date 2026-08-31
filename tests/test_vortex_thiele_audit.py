"""Physics and numerics regressions found by the 2026 Thiele-model audit."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from mmpp.analytical.constants import GAMMA_E, MU0
from mmpp.analytical.field_resolved_thiele import (
    FieldResolvedCalibration,
    FieldResolvedCPPThieleModel,
    SaturationCalibration,
)
from mmpp.analytical.nonlinear_stno import (
    SpectrumAnalyzer,
    STNOParameters,
    run_all_sweeps_parallel,
)
from mmpp.analytical.thiele import (
    CIPThieleModel,
    CPPThieleModel,
    DiskGeometry,
    MaterialParams,
    ThieleTrajectoryResult,
    current_ac,
    current_dc,
    fit_omega0_N_to_fJ,
    omega0_novosad,
    reduce_mumax_slonczewski_cpp,
)
from mmpp.solitons.vortex._shared.models import TrajectoryResult
from mmpp.solitons.vortex.autofit._numba_kernels import (
    integrate_cip_rk4,
    integrate_cpp_rk4,
)
from mmpp.solitons.vortex.autofit.diagnostics import (
    cpp_linear_threshold_metrics_from_params,
)
from mmpp.solitons.vortex.autofit.simulation import SimulationContext
from mmpp.solitons.vortex.model.thiele.cpp import cpp
from mmpp.solitons.vortex.model.thiele.field_resolved_cpp import field_resolved_cpp
from mmpp.solitons.vortex.model.thiele.fit import summarize_trajectory_kinematics
from mmpp.solitons.vortex.model.thiele.models import (
    infer_disk_geometry,
    resolve_current_waveform,
)
from mmpp.solitons.vortex.nonlinear.nonliniearthiele import ThieleAnalyzer

HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23


def _material(*, alpha: float = 0.013, P: float = 0.5) -> MaterialParams:
    return MaterialParams(Ms=8.0e5, alpha=alpha, P=P, A=1.0e-11)


def _geometry() -> DiskGeometry:
    return DiskGeometry(R=128e-9, L=9e-9)


def test_mumax_lambda_one_reduction_recovers_guslienko_polarization() -> None:
    """MuMax epsilon=P/2 must map back to the P used by the CPP model."""
    mat = _material(alpha=0.0, P=0.46)
    geo = _geometry()

    reduction = reduce_mumax_slonczewski_cpp(
        material=mat,
        torque_thickness=geo.L,
        polarizer=(0.0, 0.0, 1.0),
        fixed_layer_position="top",
        Lambda=1.0,
        epsilonprime=0.0,
    )

    assert reduction.epsilon == pytest.approx(mat.P / 2.0)
    assert reduction.pump_polarization == pytest.approx(mat.P)
    expected_chi_per_j = GAMMA_E * HBAR * mat.P / (4.0 * E_CHARGE * geo.L * mat.Ms)
    assert reduction.chi_prefactor_per_J == pytest.approx(expected_chi_per_j)


def test_mumax_efficiency_uses_explicit_mean_m_dot_p_not_polarizer_z() -> None:
    mat = _material(alpha=0.0, P=0.5)
    geo = _geometry()
    polarizer = (math.sqrt(3.0) / 2.0, 0.0, 0.5)

    centered = reduce_mumax_slonczewski_cpp(
        material=mat,
        torque_thickness=geo.L,
        polarizer=polarizer,
        Lambda=2.0,
    )
    aligned = reduce_mumax_slonczewski_cpp(
        material=mat,
        torque_thickness=geo.L,
        polarizer=polarizer,
        Lambda=2.0,
        mean_m_dot_p=0.5,
    )

    expected_centered = mat.P * 4.0 / 5.0
    assert centered.mean_m_dot_p == pytest.approx(0.0)
    assert centered.epsilon == pytest.approx(expected_centered)
    assert aligned.mean_m_dot_p == pytest.approx(0.5)
    assert aligned.epsilon != pytest.approx(centered.epsilon)


def test_cpp_threshold_includes_current_frequency_slope() -> None:
    mat = _material(P=0.4)
    geo = _geometry()
    base = CPPThieleModel(
        mat,
        geo,
        omega0=2.0 * math.pi * 0.55e9,
        polarity=-1,
    )
    pump_slope = base.chi(1.0)
    domega_dj = 0.35 * pump_slope / base._d0
    model = CPPThieleModel(
        mat,
        geo,
        omega0=base.omega0,
        polarity=-1,
        domega0_dJ=domega_dj,
    )

    expected = (
        model._d0 * model.omega0_eff(0.0) / (model.chi(1.0) - model._d0 * domega_dj)
    )
    assert model.J_threshold == pytest.approx(expected)
    growth = model.chi(model.J_threshold) - model._d0 * model.omega0_eff(
        model.J_threshold
    )
    assert growth == pytest.approx(0.0, abs=1e-7)


def test_default_frequency_optimization_bounds_support_negative_threshold() -> None:
    model = CPPThieleModel(
        _material(P=0.4),
        _geometry(),
        omega0=2.0 * math.pi * 0.55e9,
        polarity=1,
    )
    assert model.J_threshold < 0.0
    target = model.predict_frequency_dc(1.5 * model.J_threshold, allow_edge=True)
    assert target is not None

    result = model.optimize_current_for_target_frequency(
        target,
        allow_edge=True,
    )

    assert result.J_bounds[0] < result.J_bounds[1] < 0.0
    assert np.isfinite(result.current_density_a_per_m2)


def test_cpp_sde_uses_fluctuation_dissipation_diffusion() -> None:
    mat = _material(P=0.4)
    geo = _geometry()
    model = CPPThieleModel(
        mat,
        geo,
        omega0=2.0 * math.pi * 0.55e9,
        polarity=-1,
    )
    temperature = 325.0

    result = model.simulate_sde(
        (0.0, 2e-12),
        dt=1e-12,
        temperature_k=temperature,
        seed=1,
    )

    gyro = 2.0 * math.pi * mat.Ms * geo.L / mat.gamma
    expected = K_B * temperature * model._d0 / (gyro * (1.0 + model._d0**2) * geo.R**2)
    assert result.metadata["diffusion_model"] == "thiele_fdt"
    assert result.metadata["gyrocoefficient_kg_per_s"] == pytest.approx(gyro)
    assert result.metadata["diffusion"] == pytest.approx(expected)


def test_cpp_sde_uses_fractional_final_step() -> None:
    mat = _material(P=0.4)
    geo = _geometry()
    omega0 = 2.0 * math.pi * 0.2e9
    model = CPPThieleModel(mat, geo, omega0=omega0, N=0.0, polarity=-1)
    dt = 1e-12
    duration = 2.5 * dt

    result = model.simulate_sde(
        (0.0, duration),
        s0=(0.1, 0.0),
        J_func=current_dc(0.0),
        dt=dt,
        diffusion=0.0,
    )

    expected_radius = 0.1
    for step in (dt, dt, 0.5 * dt):
        expected_radius *= math.exp(-model.d(expected_radius) * omega0 * step)
    assert math.hypot(result.sx[-1], result.sy[-1]) == pytest.approx(
        expected_radius, rel=1e-12
    )


def test_field_potential_gradient_matches_force_with_gyro_calibration() -> None:
    mat = _material(P=0.4)
    geo = _geometry()
    calibration = FieldResolvedCalibration(
        G_u2=0.8,
        k_ip_iso_per_T2=3.0,
        k_ip_aniso_per_T2=7.0,
    )
    model = FieldResolvedCPPThieleModel(
        mat,
        geo,
        2.0 * math.pi * 0.55e9,
        N=0.3,
        calibration=calibration,
    )
    x = np.array([0.22 * geo.R, -0.13 * geo.R])
    field = (0.08, -0.04, 0.0)
    eps = 2e-13

    numerical = np.empty(2)
    for axis in range(2):
        delta = np.zeros(2)
        delta[axis] = eps
        numerical[axis] = (
            model.potential(x + delta, 2e10, field)
            - model.potential(x - delta, 2e10, field)
        ) / (2.0 * eps)

    np.testing.assert_allclose(
        model.grad_potential(x, 2e10, field),
        numerical,
        rtol=2e-5,
        atol=1e-18,
    )


def test_edge_potential_regularization_has_matching_gradient() -> None:
    mat = _material(P=0.4)
    geo = _geometry()
    calibration = FieldResolvedCalibration(
        saturation=SaturationCalibration(
            K_edge=2e-4,
            u_edge_max=0.9,
            edge_epsilon=0.1,
        )
    )
    model = FieldResolvedCPPThieleModel(
        mat,
        geo,
        2.0 * math.pi * 0.55e9,
        calibration=calibration,
    )
    x = np.array([0.95 * geo.R, 0.0])
    eps = 1e-13
    numerical_x = (
        model.edge_potential(x + np.array([eps, 0.0]))
        - model.edge_potential(x - np.array([eps, 0.0]))
    ) / (2.0 * eps)

    assert model.grad_edge_potential(x)[0] == pytest.approx(numerical_x, rel=2e-5)


def test_field_adapter_matches_reduced_cpp_pumping_for_tilted_polarizer() -> None:
    material = {"Ms": 8.0e5, "alpha": 0.0, "P": 0.5, "A": 1.0e-11}
    geometry = {"R": 128e-9, "L": 9e-9}
    polarizer = (math.sqrt(1.0 - 0.4**2), 0.0, 0.4)
    kwargs = {
        "material": material,
        "geom": geometry,
        "omega0": 2.0 * math.pi * 0.55e9,
        "polarity": 1,
        "polarizer": polarizer,
        "fixed_layer_position": "bottom",
        "Lambda": 1.0,
        "epsilonprime": 0.0,
        "torque_thickness": 6e-9,
    }
    reduced = cpp(**kwargs)
    resolved = field_resolved_cpp(**kwargs)
    current = 1e10

    assert resolved.model.radial_growth_rate_small_signal(current) == pytest.approx(
        reduced.model.chi(current), rel=1e-12
    )
    assert resolved._metadata["P_field_model"] == pytest.approx(
        -reduced._metadata["P_model"]
        * geometry["L"]
        / (kwargs["torque_thickness"] * polarizer[2])
    )


def test_field_adapter_propagates_reduced_field_like_phase_shift() -> None:
    adapter = field_resolved_cpp(
        material={"Ms": 8.0e5, "alpha": 0.013, "P": 0.5},
        geom={"R": 128e-9, "L": 9e-9},
        omega0=2.0 * math.pi * 0.55e9,
        polarizer=(0.0, 0.0, 1.0),
        fixed_layer_position="bottom",
        Lambda=1.0,
        epsilonprime=0.1,
    )

    assert adapter.model.cal.domega_dJ == pytest.approx(
        adapter._metadata["domega0_dJ_stt"]
    )


def test_cip_autofit_scipy_fallback_passes_current_direction_to_model() -> None:
    context = object.__new__(SimulationContext)
    context._field = None
    context._t0 = 0.0
    context._sim_t1 = 3e-11
    context._r0_x = 1e-9
    context._r0_y = 0.0
    context._current_density = 1e10
    context._J_const = 1e10
    context._dt = 1e-11

    result = context._simulate_cip_scipy(
        {
            "Ms": 8.0e5,
            "alpha": 0.01,
            "P": 0.4,
            "A": 1.0e-11,
            "R": 128e-9,
            "L": 9e-9,
            "omega0": 2.0 * math.pi * 0.55e9,
            "polarity": 1,
            "current_dir": (0.0, 1.0),
        }
    )

    assert result.t.size >= 2
    assert np.all(np.isfinite(result.x))
    assert result.params["current_dir"] == pytest.approx((0.0, 1.0))


def test_force_balance_uses_b_convention_gyromagnetic_ratio() -> None:
    time = np.linspace(0.0, 1e-9, 21)
    phase = 2.0 * math.pi * 0.5e9 * time
    trajectory = TrajectoryResult(
        time=time,
        x=5e-9 * np.cos(phase),
        y=5e-9 * np.sin(phase),
        polarity=np.ones(time.size, dtype=int),
        method="synthetic",
        confidence=np.ones(time.size),
        metadata={},
    )
    analyzer = ThieleAnalyzer(SimpleNamespace(attrs={}), None, None)
    ms = 8e5
    thickness = 9e-9

    direct = analyzer.force_balance(
        trajectory=trajectory,
        Ms=ms,
        thickness=thickness,
        gamma=GAMMA_E,
    )
    legacy_h_convention = analyzer.force_balance(
        trajectory=trajectory,
        Ms=ms,
        thickness=thickness,
        gamma0=GAMMA_E * MU0,
    )
    expected = 2.0 * math.pi * ms * thickness / GAMMA_E

    assert direct.G == pytest.approx(expected)
    assert legacy_h_convention.G == pytest.approx(expected)
    assert direct.metadata["gamma_rad_s_T"] == pytest.approx(GAMMA_E)


def test_invalid_core_model_inputs_fail_explicitly() -> None:
    mat = _material(P=0.4)
    geo = _geometry()

    with pytest.raises(ValueError, match="polarity"):
        CPPThieleModel(mat, geo, omega0=1e9, polarity=0)
    with pytest.raises(ValueError, match="current_dir"):
        from mmpp.analytical.thiele import CIPThieleModel

        CIPThieleModel(mat, geo, omega0=1e9, current_dir=(0.0, 0.0))
    with pytest.raises(ValueError, match="temperature_k"):
        CPPThieleModel(mat, geo, omega0=1e9).simulate_sde(
            (0.0, 1e-12), temperature_k=-1.0
        )


def test_fj_fit_rejects_underdetermined_parameter_set() -> None:
    with pytest.raises(ValueError, match="At least 5"):
        fit_omega0_N_to_fJ(
            np.array([1e10, 2e10, 3e10, 4e10]),
            np.array([0.5e9, 0.6e9, 0.7e9, 0.8e9]),
            material=_material(P=-0.4),
            geom=_geometry(),
            polarity=1,
            fit_domega0_dJ=True,
            fit_chi_scale=True,
        )


def test_novosad_helper_is_explicit_thin_disk_asymptote() -> None:
    mat = _material(P=0.4)
    geo = _geometry()
    expected = 5.0 / (9.0 * math.pi) * mat.gamma * MU0 * mat.Ms * geo.L / geo.R
    assert omega0_novosad(mat, geo) == pytest.approx(expected)


def test_cpp_fixed_step_kernel_matches_scipy_and_keeps_fractional_endpoint() -> None:
    mat = _material(P=0.42)
    geo = _geometry()
    model = CPPThieleModel(
        mat,
        geo,
        omega0=2.0 * math.pi * 0.55e9,
        N=0.31,
        polarity=-1,
    )
    current = 2.2e10
    dt = 20e-12
    t_end = 1.03e-9
    initial = (0.08, -0.03)

    reference = model.simulate(
        (0.0, t_end),
        s0=initial,
        J_func=current_dc(current),
        dt=dt,
        clamp_u=None,
        max_step=dt / 8.0,
        rtol=1e-11,
        atol=1e-14,
    )
    t, sx, sy = integrate_cpp_rk4(
        0.0,
        t_end,
        dt,
        initial[0],
        initial[1],
        model.chi(current),
        model.omega0_eff(current),
        model.N,
        model._d0,
        model._d1,
        float(model.polarity),
        0.0,
        0.0,
        8,
    )

    assert t[-1] == pytest.approx(t_end, abs=1e-24)
    np.testing.assert_allclose(t, reference.t, rtol=0.0, atol=1e-24)
    np.testing.assert_allclose(sx, reference.sx, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(sy, reference.sy, rtol=2e-8, atol=2e-10)


def test_cip_fixed_step_kernel_matches_scipy_and_keeps_fractional_endpoint() -> None:
    mat = MaterialParams(
        Ms=8.0e5,
        alpha=0.012,
        P=0.37,
        A=1.0e-11,
        beta_nonadiabatic=0.021,
    )
    geo = _geometry()
    model = CIPThieleModel(
        mat,
        geo,
        omega0=2.0 * math.pi * 0.55e9,
        polarity=-1,
        current_dir=(0.6, 0.8),
    )
    current = 3.0e10
    dt = 20e-12
    t_end = 0.53e-9
    initial = (8e-9, -3e-9)
    u0 = model._u0_prefactor * current

    reference = model.simulate(
        (0.0, t_end),
        r0=initial,
        J_func=current_dc(current),
        dt=dt,
        max_step=dt / 8.0,
        rtol=1e-11,
        atol=1e-16,
    )
    t, x, y = integrate_cip_rk4(
        0.0,
        t_end,
        dt,
        initial[0],
        initial[1],
        model.omega0,
        u0 * model.current_dir[0],
        u0 * model.current_dir[1],
        mat.alpha,
        mat.beta,
        model._dG,
        float(model.polarity),
        0.0,
        0.0,
        8,
    )

    assert t[-1] == pytest.approx(t_end, abs=1e-24)
    np.testing.assert_allclose(t, reference.t, rtol=0.0, atol=1e-24)
    np.testing.assert_allclose(x, reference.x, rtol=2e-8, atol=2e-16)
    np.testing.assert_allclose(y, reference.y, rtol=2e-8, atol=2e-16)


def test_cpp_threshold_diagnostics_honor_material_gamma_and_current_sign() -> None:
    params = {
        "Ms": 8.0e5,
        "alpha": 0.01,
        "P": 0.4,
        "A": 1.0e-11,
        "R": 128e-9,
        "L": 9e-9,
        "current_density": -2e10,
        "omega0": 2.0 * math.pi * 0.55e9,
        "polarity": 1,
        "gamma": 1.2 * GAMMA_E,
    }
    custom = cpp_linear_threshold_metrics_from_params(params)
    default = cpp_linear_threshold_metrics_from_params(
        {key: value for key, value in params.items() if key != "gamma"}
    )

    assert custom is not None and default is not None
    assert custom["gamma_rad_s_T"] == pytest.approx(1.2 * GAMMA_E)
    assert custom["chi"] == pytest.approx(1.2 * default["chi"])
    assert custom["chi_ratio"] > 0.0


def test_trajectory_spectrum_resamples_nonuniform_time_axis() -> None:
    uniform = np.linspace(0.0, 20e-9, 1001)
    time = uniform.copy()
    time[1:-1] += (
        0.15
        * (uniform[1] - uniform[0])
        * np.sin(np.linspace(0.0, 8.0 * math.pi, time.size - 2))
    )
    frequency = 0.8e9
    result = ThieleTrajectoryResult(
        t=time,
        x=5e-9 * np.cos(2.0 * math.pi * frequency * time),
        y=5e-9 * np.sin(2.0 * math.pi * frequency * time),
        sx=np.zeros_like(time),
        sy=np.zeros_like(time),
        disk_radius=100e-9,
    )

    assert result.dominant_frequency_ghz == pytest.approx(0.8, abs=0.06)


def test_time_dependent_cpp_current_produces_spectral_trajectory() -> None:
    model = CPPThieleModel(
        _material(alpha=0.005, P=0.0),
        _geometry(),
        omega0=2.0 * math.pi * 0.55e9,
        N=0.2,
        polarity=-1,
        domega0_dJ=0.01,
    )
    waveform = current_ac(J_amp=1.0e10, f_hz=0.2e9)

    result = model.simulate(
        (0.0, 5e-9),
        s0=(0.08, 0.0),
        J_func=waveform,
        dt=5e-12,
        clamp_u=None,
        rtol=1e-7,
        atol=1e-11,
    )

    assert result.t[-1] == pytest.approx(5e-9)
    assert result.power_spectrum.size > 10
    assert np.all(np.isfinite(result.power_spectrum))
    assert result.dominant_frequency_ghz == pytest.approx(0.55, abs=0.15)
    frequency_hz, power = result.compute_spectrum(transient_fraction=0.5)
    assert frequency_hz.size == power.size
    assert frequency_hz[np.argmax(power[1:]) + 1] == pytest.approx(0.55e9, abs=0.2e9)


def test_field_small_signal_diagnostics_use_potential_stiffness() -> None:
    mat = _material(alpha=0.0, P=0.0)
    geo = _geometry()
    cal = FieldResolvedCalibration(G_Bz=2.0)
    model = FieldResolvedCPPThieleModel(
        mat,
        geo,
        omega0=2.0 * math.pi * 0.55e9,
        polarity=1,
        calibration=cal,
    )
    field = (0.0, 0.0, 0.1)

    expected = (
        model.G0
        * model.omega0_eff(0.0, field)
        / model.G_mag(np.array([1e-12, 0.0]), 0.0, field)
    )
    assert model.small_signal_omega_exact(0.0, field) == pytest.approx(expected)


def test_field_calibration_rejects_or_exposes_nonphysical_coefficients() -> None:
    with pytest.raises(ValueError, match="edge_epsilon"):
        SaturationCalibration(edge_epsilon=0.0)

    model = FieldResolvedCPPThieleModel(
        _material(P=0.4),
        _geometry(),
        omega0=2.0 * math.pi * 0.55e9,
        calibration=FieldResolvedCalibration(G_Bz=-2.0),
    )
    with pytest.raises(ValueError, match="non-positive G"):
        model.G_mag(np.zeros(2), 0.0, (0.0, 0.0, 1.0))


def test_geometry_prefers_physical_metadata_and_auto_current_fails_closed() -> None:
    job = SimpleNamespace(attrs={"D": 240e-9, "L": 8e-9})
    geometry = infer_disk_geometry(None, job_result=job)
    assert geometry.R == pytest.approx(120e-9)
    assert geometry.L == pytest.approx(8e-9)

    with pytest.raises(ValueError, match="could not resolve"):
        resolve_current_waveform("auto_from_table", job_result=job)


def test_trajectory_proxy_is_explicitly_not_a_physical_parameter_fit() -> None:
    time = np.linspace(0.0, 2e-9, 201)
    phase = 2.0 * math.pi * 0.5e9 * time
    trajectory = TrajectoryResult(
        time=time,
        x=4e-9 * np.cos(phase),
        y=4e-9 * np.sin(phase),
        polarity=np.ones(time.size, dtype=int),
        method="synthetic",
        confidence=np.ones(time.size),
        metadata={},
    )

    summary = summarize_trajectory_kinematics(trajectory)
    assert summary.is_physical_parameter_fit is False
    assert summary.metadata["fit_kind"] == "kinematic_trajectory_proxy"
    assert summary.nonlinear_coeff_N == 0.0


def test_experimental_nonlinear_stno_is_labeled_validated_and_endpoint_aligned() -> (
    None
):
    with pytest.warns(UserWarning, match="experimental"):
        params = STNOParameters()
    kernel_params = params.get_numba_constants()
    assert kernel_params.shape == (14,)
    w0, nonlinear, d0, d1, wsw, chi = params.evaluate_field_arrays(np.array([0.0]))

    dt = 2e-12
    with pytest.warns(UserWarning, match="experimental"):
        signal = run_all_sweeps_parallel(
            np.array([1e10]),
            np.array([2e9]),
            np.array([0.5e9]),
            w0,
            nonlinear,
            d0,
            d1,
            wsw,
            chi,
            4.0 * dt,
            dt,
            2,
            kernel_params,
        )

    assert signal.shape == (1, 5)
    assert np.all(np.isfinite(signal))
    with pytest.raises(ValueError, match="Jac_arr"):
        run_all_sweeps_parallel(
            np.array([1e10]),
            np.array([1e9, 2e9]),
            np.array([0.5e9]),
            w0,
            nonlinear,
            d0,
            d1,
            wsw,
            chi,
            4.0 * dt,
            dt,
            2,
            kernel_params,
        )


def test_experimental_spectrum_analyzer_does_not_mutate_input() -> None:
    dt = 10e-12
    time = np.arange(512, dtype=float) * dt
    values = np.vstack(
        [
            np.sin(2.0 * math.pi * 0.8e9 * time),
            0.5 * np.sin(2.0 * math.pi * 1.2e9 * time),
        ]
    )
    original = values.copy()
    analyzer = SpectrumAnalyzer(dt_out=dt, cut_time=0.0)

    frequency, relative_power_db = analyzer.compute_psd(
        values, f_min_ghz=0.1, f_max_ghz=5.0
    )

    np.testing.assert_array_equal(values, original)
    assert frequency.ndim == 1
    assert relative_power_db.shape == (2, frequency.size)
