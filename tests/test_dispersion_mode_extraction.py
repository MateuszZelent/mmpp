"""
Fast unit tests for dispersion mode extraction helpers.

These tests are synthetic and do not require zarr/job infrastructure.
"""

import os
import sys

import numpy as np

# Add the mmpp package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_axes(n_k: int, n_f: int, *, dx: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    k_axis = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(n_k, dx))
    f_axis = np.fft.fftshift(np.fft.fftfreq(n_f, dt))
    return k_axis, f_axis


def test_find_peaks_prominence_basic():
    from mmpp.fft.dispersion.utils import find_peaks_1d

    y = np.array([0.0, 1.0, 0.0, 0.5, 0.0, 2.0, 0.0])
    peaks = find_peaks_1d(y, min_prominence=1.5)
    assert peaks.tolist() == [5]


def test_canonicalize_s_complex_transpose_2d():
    from mmpp.fft.dispersion.modes.extraction import canonicalize_s_complex

    n_k, n_f = 16, 8
    k_axis, f_axis = _make_axes(n_k, n_f, dx=5e-9, dt=2e-9)
    arr = np.zeros((n_f, n_k), dtype=np.complex128)

    canon, has_orth = canonicalize_s_complex(arr, k_axis=k_axis, f_axis=f_axis)
    assert canon.shape == (n_k, n_f)
    assert has_orth is False


def test_extract_mode_2d_matches_ifft_for_single_bin():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes.extraction import extract_mode_2d

    n_k, n_f = 32, 16
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)

    idx_k = n_k // 2 + 3
    idx_f = n_f // 2 + 2
    k0 = float(k_axis[idx_k])
    f0 = float(f_axis[idx_f])

    S_complex = np.zeros((n_k, n_f), dtype=np.complex128)
    S_complex[idx_k, idx_f] = 1.0 + 0.0j

    result = DispersionResult1D(
        S=np.abs(S_complex) ** 2,
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=S_complex,
    )

    x_axis, y_axis, mode_2d, info = extract_mode_2d(
        result,
        k_0=k0,
        f_0=f0,
        lattice_constant=470e-9,
        n_bz=0,
        k_direction="both",
        k_margin_bins=0,
        f_margin_bins=0,
        neighbor_reduce="mean",
    )

    assert mode_2d.shape == (1, n_k)
    assert x_axis.shape == (n_k,)
    assert y_axis.shape == (1,)
    assert info["k_bins_selected"] == 1
    assert info["f_bins_selected"] == 1

    expected = np.fft.ifft(np.fft.ifftshift(S_complex[:, idx_f]))
    assert np.allclose(mode_2d[0], expected)
    assert np.allclose(x_axis, np.arange(n_k) * dx)


def test_extract_mode_2d_preserves_orthogonal_phase_factor():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes.extraction import extract_mode_2d

    n_k, n_f, n_orth = 32, 16, 3
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)

    idx_k = n_k // 2 + 1
    idx_f = n_f // 2 + 1
    k0 = float(k_axis[idx_k])
    f0 = float(f_axis[idx_f])

    base = np.zeros((n_k, n_f), dtype=np.complex128)
    base[idx_k, idx_f] = 1.0 + 0.0j

    phases = np.array([1.0 + 0.0j, 1.0j, -1.0 + 0.0j], dtype=np.complex128)
    S_complex = np.stack([phase * base for phase in phases], axis=0)  # (N_orth, Nk, Nf)

    result = DispersionResult1D(
        S=np.abs(base) ** 2,
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=S_complex,
        orth_axis=np.arange(n_orth) * (7e-9),
        orth_axis_label="y",
    )

    x_axis, y_axis, mode_2d, info = extract_mode_2d(
        result,
        k_0=k0,
        f_0=f0,
        lattice_constant=470e-9,
        n_bz=0,
        k_direction="both",
        k_margin_bins=0,
        f_margin_bins=0,
        neighbor_reduce="mean",
    )

    assert mode_2d.shape == (n_orth, n_k)
    assert y_axis.shape == (n_orth,)
    assert info["has_orth"] is True

    expected_base = np.fft.ifft(np.fft.ifftshift(base[:, idx_f]))
    for idx, phase in enumerate(phases):
        assert np.allclose(mode_2d[idx], phase * expected_base)


def test_folding_extract_mode_profile_supports_orthogonal_s_complex():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes.folding import BrillouinZoneFolding

    n_k, n_f, n_orth = 32, 16, 2
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)

    idx_k = n_k // 2 + 2
    idx_f = n_f // 2 + 2
    k0 = float(k_axis[idx_k])
    f0 = float(f_axis[idx_f])

    base = np.zeros((n_k, n_f), dtype=np.complex128)
    base[idx_k, idx_f] = 1.0 + 0.0j
    phases = np.array([1.0 + 0.0j, 1.0j], dtype=np.complex128)
    S_complex = np.stack([phase * base for phase in phases], axis=0)

    result = DispersionResult1D(
        S=np.abs(base) ** 2,
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=S_complex,
        orth_axis=np.arange(n_orth) * (7e-9),
        orth_axis_label="y",
    )

    folder = BrillouinZoneFolding(lattice_constant=470e-9, n_periods=0)
    dk = float(np.abs(k_axis[1] - k_axis[0])) * 0.1
    df = float(np.abs(f_axis[1] - f_axis[0])) * 0.1

    prop_axis, profile, info = folder.extract_mode_profile(
        result=result,
        k_0=k0,
        f_0=f0,
        delta_k=dk,
        delta_f=df,
        return_complex=True,
    )

    expected_base = np.fft.ifft(np.fft.ifftshift(base[:, idx_f]))
    expected = np.mean(phases) * expected_base

    assert prop_axis.shape == (n_k,)
    assert profile.shape == (n_k,)
    assert np.allclose(profile, expected)
    assert info["k_bins_selected"] >= 1

