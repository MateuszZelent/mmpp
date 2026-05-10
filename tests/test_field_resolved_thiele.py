import math

import numpy as np

from mmpp.analytical.field_resolved_thiele import (
    J2,
    CurrentDrive,
    FieldResolvedCalibration,
    FieldResolvedCPPThieleModel,
    FieldResolvedTrajectoryResult,
    OerstedCalibration,
    SaturationCalibration,
    current_dc,
    field_dc,
)
from mmpp.analytical.thiele import (
    DiskGeometry,
    ExternalField,
    MaterialParams,
    reduce_mumax_slonczewski_cpp,
)


def make_model(**kwargs):
    mat = MaterialParams(Ms=800e3, alpha=0.01, P=0.4)
    geom = DiskGeometry(R=100e-9, L=10e-9)
    return FieldResolvedCPPThieleModel(mat, geom, 2.0 * math.pi * 200e6, **kwargs)


def test_pz_zero_has_no_perpendicular_pumping():
    model = make_model(polarizer=(1.0, 0.0, 0.0))
    growth_j0 = model.radial_growth_rate_small_signal(0.0)
    growth_j1 = model.radial_growth_rate_small_signal(1e11)
    assert np.isclose(growth_j1, growth_j0, rtol=0.0, atol=1e-9)


def test_pz_perpendicular_pumping_changes_growth():
    model = make_model(polarizer=(0.0, 0.0, 1.0))
    growth_j0 = model.radial_growth_rate_small_signal(0.0)
    growth_j1 = model.radial_growth_rate_small_signal(1e11)
    assert growth_j1 > growth_j0


def test_inplane_field_seq_maps_to_equilibrium_shift():
    cal = FieldResolvedCalibration(seq_per_T=2.5)
    model = make_model(chirality=1, calibration=cal)
    B = ExternalField(Bx_T=0.010, By_T=-0.020, Bz_T=0.0)
    Xeq = model.equilibrium_conservative(B=B, include_quartic=False)
    expected_s = 2.5 * (J2 @ np.array([B.Bx_T, B.By_T]))
    assert np.allclose(Xeq / model.geom.R, expected_s, rtol=1e-12, atol=1e-12)


def test_zero_field_small_signal_frequency_is_omega0_with_small_damping_correction():
    model = make_model(polarizer=(0.0, 0.0, 0.0 + 1.0))
    omega_exact = model.small_signal_omega_exact(0.0)
    expected = model.omega0 / (1.0 + model.d0 * model.d0)
    assert np.isclose(omega_exact, expected, rtol=1e-12, atol=0.0)


def test_reduce_mumax_slonczewski_cpp_respects_pz():
    mat = MaterialParams(Ms=800e3, alpha=0.01, P=0.4)
    red_z = reduce_mumax_slonczewski_cpp(
        material=mat, torque_thickness=10e-9, polarizer=(0, 0, 1)
    )
    red_x = reduce_mumax_slonczewski_cpp(
        material=mat, torque_thickness=10e-9, polarizer=(1, 0, 0)
    )
    assert abs(red_z.chi_prefactor_per_J) > 0.0
    assert red_x.chi_prefactor_per_J == 0.0


def test_rhs_returns_finite_velocity():
    model = make_model(calibration=FieldResolvedCalibration(seq_per_T=1.0))
    v = model.rhs(
        0.0,
        np.array([1e-9, 2e-9]),
        current_dc(1e10),
        field_dc((1e-3, 2e-3, 5e-3)),
    )
    assert v.shape == (2,)
    assert np.all(np.isfinite(v))


def test_current_drive_converts_current_to_density_for_disk_area():
    drive = CurrentDrive()
    geom = DiskGeometry(R=100e-9, L=10e-9)

    assert np.isclose(drive.J_from_I(1e-3, geom), 3.1830988618379066e10)


def test_saturation_damping_increases_near_configured_orbit_limit():
    cal = FieldResolvedCalibration(
        saturation=SaturationCalibration(d_edge=0.2, u_damp_max=0.85)
    )
    model = make_model(calibration=cal)

    low = model.D_coeff(np.array([0.20 * model.geom.R, 0.0]), 0.0)
    near_edge = model.D_coeff(np.array([0.84 * model.geom.R, 0.0]), 0.0)

    assert near_edge > 20.0 * low


def test_oersted_stiffness_changes_sign_with_chirality_and_current():
    cal = FieldResolvedCalibration(oersted=OerstedCalibration(K2_per_J=1e-22))
    model_cw = make_model(chirality=1, calibration=cal)
    model_ccw = make_model(chirality=-1, calibration=cal)
    x = np.zeros(2, dtype=float)

    base = model_cw.K2_tensor(x, 0.0)[0, 0]
    shift_cw = model_cw.K2_tensor(x, 1e11)[0, 0] - base
    shift_ccw = model_ccw.K2_tensor(x, 1e11)[0, 0] - base
    shift_reverse_current = model_cw.K2_tensor(x, -1e11)[0, 0] - base

    assert shift_cw > 0.0
    assert np.isclose(shift_ccw, -shift_cw, rtol=1e-12)
    assert np.isclose(shift_reverse_current, -shift_cw, rtol=1e-12)


def test_frequency_geometric_uses_shifted_orbit_center():
    freq_hz = 0.75e9
    t = np.linspace(0.0, 80e-9, 1001)
    cx = 40e-9
    cy = -20e-9
    radius = 4e-9
    x = cx + radius * np.cos(2.0 * math.pi * freq_hz * t)
    y = cy + radius * np.sin(2.0 * math.pi * freq_hz * t)
    result = FieldResolvedTrajectoryResult(
        t=t,
        x=x,
        y=y,
        sx=x / 100e-9,
        sy=y / 100e-9,
        disk_radius=100e-9,
    )
    model = make_model()

    centered = model.frequency_geometric(result, center="mean", t_min=10e-9)
    disk_center = model.frequency_geometric(result, center=(0.0, 0.0), t_min=10e-9)

    assert np.isclose(centered, freq_hz, rtol=1e-3)
    assert abs(disk_center - freq_hz) > 0.05 * freq_hz


def test_simulate_dc_sweep_returns_frequency_and_regime_columns():
    cal = FieldResolvedCalibration(
        saturation=SaturationCalibration(d_edge=0.1, u_damp_max=0.85),
    )
    model = make_model(calibration=cal)
    frame = model.simulate_dc_sweep(
        [0.0, 0.2e-3],
        B=(0.0, 0.0, 0.0),
        t_total=4e-9,
        dt=1e-11,
        transient_fraction=0.25,
    )

    expected = {
        "I_A",
        "I_mA",
        "J_Apm2",
        "frequency_geom_hz",
        "frequency_fft_hz",
        "u_mean",
        "u_max",
        "center_x_m",
        "center_y_m",
        "regime",
        "edge_limited",
    }
    assert expected.issubset(frame.columns)
    assert frame.loc[0, "regime"] == "damped"
