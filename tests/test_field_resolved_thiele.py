import math

import numpy as np

from mmpp.analytical.field_resolved_thiele import (
    J2,
    FieldResolvedCalibration,
    FieldResolvedCPPThieleModel,
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
