"""Regression tests for FFT and dispersion scientific contracts."""

from __future__ import annotations

import numpy as np
import pytest


def test_fftshifted_mirror_is_exact_for_even_and_odd_lengths():
    from mmpp.fft.dispersion.utils import mirror_fftshifted_indices

    for n in range(1, 12):
        permutation = mirror_fftshifted_indices(n)
        assert sorted(permutation.tolist()) == list(range(n))
        assert np.array_equal(permutation[permutation], np.arange(n))

        shifted = np.fft.fftshift(np.fft.fftfreq(n))
        # Compare frequencies modulo the sampling frequency; the even-length
        # Nyquist bin is self-negative modulo N.
        delta = np.mod(shifted[permutation] + shifted + 0.5, 1.0) - 0.5
        assert np.allclose(delta, 0.0)


def test_mode_reconstruction_undoes_public_k_flip_and_keeps_origin():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes.extraction import extract_mode_2d
    from mmpp.fft.dispersion.utils import mirror_fftshifted_indices

    n_k, n_f = 32, 12
    dx, dt, origin = 2e-9, 1e-10, 3.5e-6
    k_axis = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(n_k, dx))
    f_axis = np.fft.fftshift(np.fft.fftfreq(n_f, dt))
    x = np.arange(n_k)
    spatial_profile = np.exp(-(((x - 7.0) / 3.0) ** 2)) * np.exp(0.2j * x)
    shifted_fft = np.fft.fftshift(np.fft.fft(spatial_profile))
    public_fft = shifted_fft[mirror_fftshifted_indices(n_k)]
    f_idx = n_f // 2 + 2
    s_complex = np.zeros((n_k, n_f), dtype=np.complex128)
    s_complex[:, f_idx] = public_fft

    result = DispersionResult1D(
        S=np.abs(s_complex) ** 2,
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="mx",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        flipx=True,
        spatial_origin=origin,
        S_complex=s_complex,
    )
    x_axis, _, mode, _ = extract_mode_2d(
        result,
        k_0=0.0,
        f_0=float(f_axis[f_idx]),
        lattice_constant=1e-6,
        n_bz=0,
        delta_k=float(np.max(np.abs(k_axis)) + abs(k_axis[1] - k_axis[0])),
    )

    assert np.allclose(mode[0], spatial_profile)
    assert np.allclose(x_axis, origin + np.arange(n_k) * dx)


def test_folding_uses_frequency_continuity_and_correct_fallback_sign():
    from mmpp.fft.dispersion.modes.folding import BrillouinZoneFolding
    from mmpp.fft.dispersion.modes.models import DispersionMode

    folder = BrillouinZoneFolding(lattice_constant=1e-9, n_periods=0)
    k_folded, origin_bz, g_applied = folder.fold_k_to_fbz(7.3 * folder.k_bz)
    assert np.isclose(k_folded, 7.3 * folder.k_bz + g_applied)
    assert origin_bz == int(round(-g_applied / (2 * folder.k_bz)))

    modes = [
        DispersionMode(0.0, 1.0, 1, 0.0, 0, 1.0, 0.0),
        DispersionMode(0.0, 3.0, 0, 0.0, 0, 10.0, 0.0),
        DispersionMode(0.0, 1.2, 0, 0.0, 0, 10.0, 1.0),
        DispersionMode(0.0, 2.8, 1, 0.0, 0, 1.0, 1.0),
    ]
    branches = folder._group_into_branches(modes)
    tracked = {
        (mode.k_original, mode.branch_index): mode.omega
        for branch in branches.values()
        for mode in branch
    }
    assert tracked[(0.0, 0)] == 1.0
    assert tracked[(1.0, 0)] == 1.2
    assert tracked[(0.0, 1)] == 3.0
    assert tracked[(1.0, 1)] == 2.8


def test_relative_peak_threshold_is_invariant_to_spectrum_scale():
    from mmpp.fft.dispersion.modes.detection import BrillouinZoneDetector

    detector = BrillouinZoneDetector()
    spectrum = np.array([0.0, 0.2, 0.0, 1.0, 0.0, 0.5, 0.0])
    assert detector._find_peaks_simple(
        spectrum, threshold=0.3
    ) == detector._find_peaks_simple(100.0 * spectrum, threshold=0.3)


def test_period_detector_does_not_guess_and_labels_low_intensity_candidates():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes.detection import BrillouinZoneDetector

    k_axis = np.linspace(-1e7, 1e7, 32)
    f_axis = np.linspace(0, 10e9, 16)
    flat = DispersionResult1D(
        S=np.ones((k_axis.size, f_axis.size)),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="mx",
        config=DispersionConfig(),
    )
    detector = BrillouinZoneDetector()
    assert detector.detect_lattice_constant(flat) is None
    assert detector.last_detection["status"] == "not_detected"

    S = np.ones((k_axis.size, f_axis.size))
    S[:, 5:9] = 0.0
    weak_band = DispersionResult1D(
        S=S,
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="mx",
        config=DispersionConfig(),
    )
    candidates = detector.find_low_intensity_intervals(weak_band)
    assert len(candidates) == 1
    assert candidates[0].status == "low_observed_intensity_candidate"
    assert np.isclose(candidates[0].mean_relative_intensity, 0.0)


def test_mode_winding_is_invariant_to_global_complex_phase():
    from mmpp.fft.mode_characterization import ModeCharacterAnalyzer
    from mmpp.fft.vortex_classifier import AdvancedVortexClassifier

    ny = nx = 33
    yy, xx = np.indices((ny, nx))
    phi = np.arctan2(yy - 16.0, xx - 16.0)
    ring = np.hypot(xx - 16.0, yy - 16.0)
    mx = np.where((ring >= 7.0) & (ring <= 9.0), np.exp(1j * phi), 0.0j)
    my = 0.35j * mx
    phase = np.exp(1.27j)

    analyzer = ModeCharacterAnalyzer()
    amp = np.hypot(np.abs(mx), np.abs(my))
    winding_a = analyzer._estimate_winding_number(mx, my, (16.0, 16.0), 8.0, amp)
    winding_b = analyzer._estimate_winding_number(
        phase * mx, phase * my, (16.0, 16.0), 8.0, amp
    )
    assert winding_a[0] == winding_b[0] == 1
    assert np.isclose(winding_a[1], winding_b[1])

    legacy_a = AdvancedVortexClassifier._estimate_winding(
        mx, my, (16.0, 16.0), 8.0, 1.0
    )
    legacy_b = AdvancedVortexClassifier._estimate_winding(
        phase * mx, phase * my, (16.0, 16.0), 8.0, 1.0
    )
    assert legacy_a[0] == legacy_b[0]
    assert np.isclose(legacy_a[1], legacy_b[1])


def test_spatial_winding_is_not_reported_as_temporal_rotation_without_convention():
    from mmpp.fft.mode_characterization import ModeCharacterAnalyzer
    from mmpp.fft.modes.models import FMRModeData

    ny = nx = 41
    yy, xx = np.indices((ny, nx))
    phi = np.arctan2(yy - 20.0, xx - 20.0)
    radius = np.hypot(xx - 20.0, yy - 20.0)
    envelope = np.exp(-0.5 * ((radius - 10.0) / 1.0) ** 2)
    mode = np.zeros((ny, nx, 3), dtype=complex)
    mode[:, :, 0] = envelope * np.cos(phi)
    mode[:, :, 1] = envelope * np.sin(phi)
    result = ModeCharacterAnalyzer().analyze(
        FMRModeData(1.0, mode, metadata={"spatial_resolution": (1.0, 1.0)}),
        core_position=(20.0, 20.0),
        analysis_radius=10.0,
    )

    assert result.m_index == 1
    assert result.rotation_sense is None
    assert not any(label in {"CW", "CCW"} for label in result.labels)


def test_poynting_and_energy_density_use_real_or_peak_phasor_conventions():
    from mmpp.fft.electromagnetic_analysis import PoyntingVectorAnalysis

    analyzer = PoyntingVectorAnalysis()
    electric = np.array([[[1.0, 0.0, 0.0]]])
    magnetic = np.array([[[0.0, 1.0, 0.0]]])
    assert np.allclose(
        analyzer.compute_poynting_vector(electric, magnetic), [[[0.0, 0.0, 1.0]]]
    )

    electric_phasor = electric.astype(complex)
    magnetic_phasor = magnetic.astype(complex)
    assert np.allclose(
        analyzer.compute_poynting_vector(electric_phasor, magnetic_phasor),
        [[[0.0, 0.0, 0.5]]],
    )
    # A phase-shifted H component must be conjugated in the average.
    magnetic_phasor[0, 0, 1] = 1j
    assert np.allclose(
        analyzer.compute_poynting_vector(electric_phasor, magnetic_phasor),
        np.zeros((1, 1, 3)),
    )

    densities = analyzer.compute_energy_density(electric_phasor, magnetic_phasor)
    assert np.isclose(densities["electric"][0, 0], analyzer.config.epsilon_0 / 4)
    assert np.isclose(densities["magnetic"][0, 0], analyzer.config.mu_0 / 4)


def test_em_analysis_does_not_claim_success_for_magnetization_placeholders():
    from types import SimpleNamespace

    from mmpp.fft.electromagnetic_analysis import (
        ElectromagneticAnalysisConfig,
        analyze_electromagnetic_properties,
    )

    mode = SimpleNamespace(
        frequency=2.0,
        extent=(0.0, 30.0, 0.0, 20.0),
        mode_array=np.ones((2, 3, 3), dtype=complex),
    )
    unavailable = analyze_electromagnetic_properties(mode)
    assert unavailable["analysis_status"] == "unavailable"
    assert not unavailable["analysis_successful"]
    assert "poynting_vector" not in unavailable

    config = ElectromagneticAnalysisConfig(
        radiation_theta_points=5,
        radiation_phi_points=7,
    )
    complete = analyze_electromagnetic_properties(
        mode,
        config,
        electric_field_v_per_m=np.ones((2, 3, 3), dtype=complex),
        magnetic_field_a_per_m=np.ones((2, 3, 3), dtype=complex),
        current_density_a_per_m2=np.ones((2, 3, 3), dtype=complex),
    )
    assert complete["analysis_status"] == "complete"
    assert complete["analysis_successful"]
    assert complete["far_field"]["E_theta"].shape == (5, 7)


def test_far_field_spatial_grid_matches_rectangular_yx_input():
    from mmpp.fft.electromagnetic_analysis import (
        ElectromagneticAnalysisConfig,
        RadiationPatternAnalysis,
    )

    analyzer = RadiationPatternAnalysis(
        ElectromagneticAnalysisConfig(
            radiation_theta_points=4,
            radiation_phi_points=6,
        )
    )
    current = np.zeros((2, 3, 3), dtype=complex)
    current[1, 2, 0] = 1.0
    result = analyzer.compute_far_field(current, (0.0, 3e-6, 0.0, 2e-6), 1e9)
    assert result["E_theta"].shape == (4, 6)
    assert np.isfinite(result["E_theta"]).all()
    assert np.isfinite(result["E_phi"]).all()


def test_q_factor_interpolates_outer_half_maximum_crossings_and_can_be_unavailable():
    from mmpp.fft.electromagnetic_analysis import QFactorAnalysis

    analyzer = QFactorAnalysis()
    frequencies = np.arange(10.0, 17.0)
    power = np.array([0.0, 0.25, 0.75, 1.0, 0.75, 0.25, 0.0])
    q_factor = analyzer.compute_q_factor_spectral(frequencies, power, 13.0)
    assert np.isclose(q_factor, 13.0 / 3.0)
    assert (
        analyzer.compute_q_factor_spectral(
            np.arange(3.0, 10.0), np.array([0, 0, 1, 0.8, 0.7, 0.6, 0.55]), 5.0
        )
        is None
    )
    assert analyzer.compute_mode_lifetime(None, 1e9) is None


def test_filter_pipeline_respects_disabled_options_and_quadratic_config():
    from mmpp.fft.filters.config import FilterConfig, PreprocessConfig
    from mmpp.fft.filters.pipeline import FilterPipeline

    values = np.arange(12.0) ** 2 + np.sin(np.arange(12.0))
    pipeline = FilterPipeline()
    assert np.array_equal(
        pipeline.preprocess(values, filters={"pre": {"remove_mean": False}}), values
    )

    quadratic_pipeline = FilterPipeline(
        FilterConfig(
            pre=PreprocessConfig(
                remove_mean=False,
                detrend="quadratic",
                window="none",
            )
        )
    )
    detrended = quadratic_pipeline.preprocess(values)
    assert np.max(np.abs(np.polyfit(np.linspace(-1, 1, 12), detrended, 2))) < 0.2


def test_numpy_window_fallback_preserves_named_windows_and_single_sample_contract(
    monkeypatch,
):
    import mmpp.fft.filters.windows as windows

    monkeypatch.setattr(windows, "_SCIPY_AVAILABLE", False)
    for name in ("tukey", "gaussian", "flattop", "nuttall"):
        window = windows.get_window(name, 9)
        assert window.shape == (9,)
        assert not np.allclose(window, np.ones(9))
        assert np.array_equal(windows.get_window(name, 1), np.ones(1))


def test_log_filtered_spectrum_keeps_display_transform_separate_from_power():
    from mmpp.fft.spectrum.result import SpectrumResult

    result = SpectrumResult(
        frequencies=np.array([1.0, 2.0, 3.0]),
        spectrum=np.array([0.1, 1.0, 10.0], dtype=complex),
    )
    filtered = result.filtered(log_scale=True)

    assert np.allclose(filtered.power, [-2.0, 0.0, 2.0])
    assert np.all(filtered.spectral_quantity >= 0.0)
    assert np.allclose(filtered.amplitude**2, filtered.spectral_quantity)
    assert np.any(filtered.power < 0.0)
    assert filtered.display_quantity_label == filtered.spectral_quantity_label


def test_in_plane_pssw_keeps_both_zero_wavevector_stiffness_factors():
    from mmpp.analytical.constants import MU0
    from mmpp.analytical.dispersion import gamma, kalinikos_no_approx

    field_t = 0.1
    saturation_a_per_m = 8.0e5
    thickness_m = 10.0e-9
    exchange_j_per_m = 10.0e-12
    mode_index = 1
    exchange_field_t = (
        2.0
        * exchange_j_per_m
        * (mode_index * np.pi / thickness_m) ** 2
        / saturation_a_per_m
    )
    expected_hz = (
        gamma(2.0)
        * np.sqrt(
            (field_t + exchange_field_t)
            * (field_t + exchange_field_t + MU0 * saturation_a_per_m)
        )
        / (2.0 * np.pi)
    )

    result = kalinikos_no_approx(
        k=0.0,
        B=field_t,
        Ms=saturation_a_per_m,
        d=thickness_m,
        Aex=exchange_j_per_m,
        n=mode_index,
    )

    assert result.f[0] * 1e9 == pytest.approx(expected_hz)
    assert "diagonal" in result.model_name


def test_dispersion_anisotropy_axis_and_cubic_stability_are_explicit():
    from mmpp.analytical.dispersion import (
        _cubic_energy_deriv1,
        _cubic_energy_deriv2,
        _cubic_equilibrium_angle,
        kalinikos,
    )

    common = {
        "k": 0.0,
        "B": 0.5,
        "Ms": 8.0e5,
        "d": 10e-9,
        "Aex": 10e-12,
        "Ku": 8e4,
    }
    perpendicular = kalinikos(**common, ku_axis="perpendicular")
    in_plane = kalinikos(**common, ku_axis="in_plane")
    assert in_plane.f[0] > perpendicular.f[0]
    assert perpendicular.params["ku_axis"] == "perpendicular"
    assert in_plane.params["ku_axis"] == "in_plane"

    equilibrium = _cubic_equilibrium_angle(
        B=0.01,
        Ms=8.0e5,
        Kc1=-8.0e3,
        phi_H=0.0,
        phi_ani=0.0,
    )
    assert not np.isclose(equilibrium, 0.0)
    total_torque = 8.0e5 * 0.01 * np.sin(equilibrium) + _cubic_energy_deriv1(
        equilibrium, -8.0e3, 0.0
    )
    total_curvature = 8.0e5 * 0.01 * np.cos(equilibrium) + _cubic_energy_deriv2(
        equilibrium, -8.0e3, 0.0
    )
    assert abs(total_torque) < 1e-6
    assert total_curvature > 0.0
