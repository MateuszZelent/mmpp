"""
Fast unit tests for dispersion mode extraction helpers.

These tests are synthetic and do not require zarr/job infrastructure.
"""

import json
import logging
import os
import sys
import types

import numpy as np
import pytest
import zarr

# Add the mmpp package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_axes(
    n_k: int, n_f: int, *, dx: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
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


def test_folding_mode_mask_selects_bz_replicas_used_for_profile_reconstruction():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes.folding import BrillouinZoneFolding

    n_k, n_f = 9, 5
    dk, df = 1.0e6, 1.0e9
    k_axis = (np.arange(n_k) - 4) * dk
    f_axis = np.arange(n_f, dtype=float) * df
    reciprocal = 3 * dk
    period = 2 * np.pi / reciprocal
    k0 = dk
    f0 = 2 * df
    selected_k = np.array([k0 - reciprocal, k0, k0 + reciprocal])
    selected_idx = [int(np.argmin(np.abs(k_axis - value))) for value in selected_k]
    f_idx = int(np.argmin(np.abs(f_axis - f0)))
    spectrum = np.zeros((n_k, n_f), dtype=np.complex128)
    spectrum[selected_idx, f_idx] = np.array([1.0, 2.0, 4.0], dtype=np.complex128)
    result = DispersionResult1D(
        S=np.abs(spectrum) ** 2,
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=5e-9),
        dt=1.0,
        dx=5e-9,
        S_complex=spectrum,
    )
    folder = BrillouinZoneFolding(lattice_constant=period, n_periods=1)

    mask = folder.create_mode_mask(
        k_axis=k_axis,
        f_axis=f_axis,
        k_0=k0,
        f_0=f0,
        delta_k=0.25 * dk,
        delta_f=0.25 * df,
        include_all_copies=True,
    )
    prop_axis, profile, info = folder.extract_mode_profile(
        result=result,
        k_0=k0,
        f_0=f0,
        delta_k=0.25 * dk,
        delta_f=0.25 * df,
        return_complex=True,
    )
    expected_k_spectrum = np.zeros(n_k, dtype=np.complex128)
    expected_k_spectrum[selected_idx] = spectrum[selected_idx, f_idx]
    expected_profile = np.fft.ifft(np.fft.ifftshift(expected_k_spectrum))

    assert np.where(mask[:, f_idx])[0].tolist() == selected_idx
    assert int(mask.sum()) == 3
    assert info["k_bins_selected"] == 3
    assert info["f_bins_selected"] == 1
    assert info["n_points_masked"] == 3
    assert prop_axis.shape == (n_k,)
    assert np.allclose(profile, expected_profile)


def test_modes_at_uses_full_frequency_index_for_positive_frequency_selection():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    idx_k = n_k // 2 + 1
    idx_f_positive = n_f // 2 + 1
    idx_f_wrong_relative = 1
    S_complex = np.zeros((n_k, n_f), dtype=np.complex128)
    S_complex[idx_k, idx_f_wrong_relative] = 3.0 + 0.0j
    S_complex[idx_k, idx_f_positive] = 7.0 + 2.0j

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

    mode = result.modes.at(
        k_rad_um=float(k_axis[idx_k]) / 1e6,
        f_ghz=float(f_axis[idx_f_positive]) / 1e9,
    )

    assert mode.mode_data == 7.0 + 2.0j
    assert np.isclose(mode.f_ghz, float(f_axis[idx_f_positive]) / 1e9)


def test_single_mode_plot_interactive_show_false_returns_viewer():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    idx_k = n_k // 2 + 1
    idx_f = n_f // 2 + 1
    S_complex = np.zeros((n_k, n_f), dtype=np.complex128)
    S_complex[idx_k, idx_f] = 7.0 + 2.0j
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
    mode = result.modes.at(
        k_rad_um=float(k_axis[idx_k]) / 1e6,
        f_ghz=float(f_axis[idx_f]) / 1e9,
    )

    viewer = mode.plot.interactive(
        show=False,
        mode_type="phase",
        cmap="hsv",
        alpha=np.float32(0.5),
    )
    exported = viewer.export_selection(
        source="unit-test",
        frame=np.int64(2),
        marker=np.array([1.0, 2.0], dtype=np.float32),
    )
    html = viewer._repr_html_()

    assert viewer.mode is mode
    assert viewer.show_requested is False
    assert viewer.state["mode_type"] == "phase"
    assert viewer.state["component"] == "perp"
    assert viewer.state["options"] == {"cmap": "hsv", "alpha": 0.5}
    assert exported["selection"] == {
        "source": "unit-test",
        "frame": 2,
        "marker": [1.0, 2.0],
    }
    json.dumps(exported)
    assert "DispersionSingleModeInteractiveViewer" in html


def test_single_mode_interactive_show_and_close_update_display_state(monkeypatch):
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    class _DisplayHandle:
        def __init__(self):
            self.updates = []

        def update(self, value):
            self.updates.append(value)

    calls = []
    display_handle = _DisplayHandle()

    def fake_display(obj, display_id=False):
        calls.append({"obj": obj, "display_id": display_id})
        return display_handle

    ipython_module = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")
    display_module.display = fake_display
    monkeypatch.setitem(sys.modules, "IPython", ipython_module)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    S_complex = np.zeros((n_k, n_f), dtype=np.complex128)
    S_complex[n_k // 2, n_f // 2] = 1.0 + 0.0j
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
    mode = result.modes.at(k_rad_um=0.0, f_ghz=0.0)

    viewer = mode.plot.interactive(show=False)
    assert viewer.show() is viewer
    assert viewer.show_requested is True
    assert calls == [{"obj": viewer, "display_id": True}]

    viewer.close()
    assert viewer.show_requested is False
    assert viewer._display_handle is None
    assert display_handle.updates == [None]


def test_modes_plot_animation_show_false_returns_headless_viewer():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=np.ones((n_k, n_f), dtype=np.complex128),
    )

    viewer = result.modes.plot.animation(
        peaks=[np.int64(0), np.int64(2)],
        show=False,
        mode_type="abs",
        fps=np.int64(12),
    )
    exported = viewer.export_selection(
        source="unit-test",
        frame=np.int64(1),
        marker=np.array([3.0, 4.0], dtype=np.float32),
    )
    html = viewer._repr_html_()

    assert viewer.result is result
    assert viewer.show_requested is False
    assert viewer.state["peaks"] == [0, 2]
    assert viewer.state["has_complex"] is True
    assert viewer.state["options"] == {"mode_type": "abs", "fps": 12}
    assert exported["selection"] == {
        "source": "unit-test",
        "frame": 1,
        "marker": [3.0, 4.0],
    }
    json.dumps(exported)
    assert "DispersionModesAnimationViewer" in html


def test_dispersion_plot_interactive_modes_true_stores_complex_and_viewer_state():
    from mmpp.fft.dispersion.interface import _DispersionPlotAccessor
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    class FakeInterface:
        def __init__(self):
            self._tmax = None
            self._cache_dir = None
            self.calls = []

        def compute_1d(self, **kwargs):
            self.calls.append(kwargs)
            n_k, n_f = 8, 6
            dx, dt = 5e-9, 2e-9
            k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
            S_complex = (
                np.ones((n_k, n_f), dtype=np.complex128)
                if kwargs.get("store_complex")
                else None
            )
            return DispersionResult1D(
                S=np.ones((n_k, n_f), dtype=np.float32),
                k_axis=k_axis,
                f_axis=f_axis,
                axis=kwargs.get("axis", "x"),
                component=kwargs.get("component", "perp"),
                config=DispersionConfig(dt=dt, dx=dx),
                dt=dt,
                dx=dx,
                S_complex=S_complex,
            )

    fake = FakeInterface()

    viewer = _DispersionPlotAccessor(fake).interactive(
        show=False,
        modes=True,
        components=["z", "+", "-"],
        animate=True,
        axis="x",
    )

    assert fake.calls == [{"axis": "x", "store_complex": True}]
    assert viewer.can_reconstruct_modes is True
    assert viewer.state["modes"] is True
    assert viewer.state["mode_components"] == ["z", "+", "-"]
    assert viewer.state["spectrum_components"] == ["z", "+", "-"]
    assert viewer.state["options"]["auto_animate"] is True


def test_dispersion_interactive_kwargs_split_keeps_compute_and_viewer_roles():
    from mmpp.fft.dispersion._interactive_viewer import (
        split_dispersion_interactive_kwargs,
    )

    compute_kwargs, viewer_kwargs = split_dispersion_interactive_kwargs(
        {
            "axis": "x",
            "component": "perp",
            "save": True,
            "disk_cache": False,
            "components": ["z", "+"],
            "modes": True,
            "fmax": 4.0,
            "analytical": True,
            "model": "kalinikos",
            "B": 0.12,
            "linestyle": "--",
        }
    )

    assert compute_kwargs == {
        "axis": "x",
        "component": "perp",
        "save": True,
        "disk_cache": False,
    }
    assert viewer_kwargs == {
        "components": ["z", "+"],
        "modes": True,
        "fmax": 4.0,
        "analytical": True,
        "model": "kalinikos",
        "B": 0.12,
        "linestyle": "--",
    }


def test_dispersion_plot_interactive_preserves_analytical_overlay_state():
    from mmpp.fft.dispersion.interface import _DispersionPlotAccessor
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    class FakeInterface:
        _tmax = None
        _cache_dir = None

        def __init__(self):
            self.calls = []

        def compute_1d(self, **kwargs):
            self.calls.append(kwargs)
            n_k, n_f = 8, 6
            dx, dt = 5e-9, 2e-9
            k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
            return DispersionResult1D(
                S=np.ones((n_k, n_f), dtype=np.float32),
                k_axis=k_axis,
                f_axis=f_axis,
                axis=kwargs.get("axis", "x"),
                component=kwargs.get("component", "perp"),
                config=DispersionConfig(dt=dt, dx=dx),
                dt=dt,
                dx=dx,
            )

    viewer = _DispersionPlotAccessor(FakeInterface()).interactive(
        show=False,
        analytical=True,
        model="kalinikos",
        sw_config="DE",
        n_modes=3,
        B=0.12,
        Ms=800e3,
        Aex=13e-12,
        d=20e-9,
        linestyle="--",
        alpha=0.7,
    )

    assert viewer.state["analytical"] == {
        "enabled": True,
        "model": "kalinikos",
        "sw_config": "DE",
        "n_modes": 3,
        "B": 0.12,
        "Ms": 800e3,
        "Aex": 13e-12,
        "d": 20e-9,
        "linestyle": "--",
        "alpha": 0.7,
    }
    assert "model" not in viewer.state["options"]
    assert "B" not in viewer.state["options"]


def test_dispersion_viewer_passes_analytical_overlay_to_heatmap_widget(monkeypatch):
    from mmpp.fft.dispersion import _interactive as interactive_module
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    captured = {}

    class FakeHeatmapWidget:
        def __init__(self, result, options):
            captured["result"] = result
            captured["options"] = options
            self.figure = None
            self.axes = None
            self.controls = {}

        def build(self, display_func, toolbar="auto", **kwargs):
            captured["toolbar"] = toolbar
            return "fake-widget"

    monkeypatch.setattr(
        interactive_module, "DispersionHeatmapWidget", FakeHeatmapWidget
    )

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )

    viewer = result.plot.interactive(
        show=False,
        analytical=True,
        model="kalinikos",
        sw_config="BV",
        n_modes=2,
        k_points=800,
        B=0.08,
        color="white",
        linewidth=2.0,
    )

    assert viewer._build_widget(lambda *_args, **_kwargs: None) == "fake-widget"
    assert captured["result"] is result
    assert captured["options"]["analytical"] == "BV"
    assert captured["options"]["analytical_model"] == "kalinikos"
    assert captured["options"]["analytical_sw_config"] == "BV"
    assert captured["options"]["analytical_n_modes"] == 2
    assert captured["options"]["analytical_k_points"] == 800
    assert captured["options"]["B"] == 0.08
    assert captured["options"]["analytical_style"] == {
        "color": "white",
        "linewidth": 2.0,
    }


def test_modes_plot_animation_show_and_close_update_display_state(monkeypatch):
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    class _DisplayHandle:
        def __init__(self):
            self.updates = []

        def update(self, value):
            self.updates.append(value)

    calls = []
    display_handle = _DisplayHandle()

    def fake_display(obj, display_id=False):
        calls.append({"obj": obj, "display_id": display_id})
        return display_handle

    ipython_module = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")
    display_module.display = fake_display
    monkeypatch.setitem(sys.modules, "IPython", ipython_module)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )

    viewer = result.modes.plot.animation(show=False)
    assert viewer.show() is viewer
    assert viewer.show_requested is True
    assert calls == [{"obj": viewer, "display_id": True}]

    viewer.close()
    assert viewer.show_requested is False
    assert viewer._display_handle is None
    assert display_handle.updates == [None]


def test_dispersion_result_plot_interactive_show_false_returns_viewer():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )

    viewer = result.plot.interactive(
        show=False,
        fmax=4.0,
        components=["+", "-", "z"],
        animate=True,
    )

    assert viewer.result is result
    assert viewer.show_requested is False
    assert viewer.mode_components == ["+", "-", "z"]
    assert viewer.spectrum_components == ["+", "-", "z"]
    assert viewer.options["fmax"] == 4.0
    assert viewer.options["auto_animate"] is True
    assert viewer.options["positive_frequencies"] is True

    full_viewer = result.plot.interactive(show=False, positive_frequencies=False)
    assert full_viewer.options["positive_frequencies"] is False


def test_dispersion_interactive_viewer_can_save_load_preset_and_export_selection(
    tmp_path,
):
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    viewer = result.plot.interactive(
        show=False,
        components=["z", "+"],
        animate=True,
        alpha=np.float32(0.5),
        fmax=8.0,
    )

    exported = viewer.export_selection(
        k_rad_per_m=np.float64(1.25e6),
        f_hz=2.5e9,
        source="display",
        marker=np.array([1.0, 2.0], dtype=np.float32),
    )

    assert exported["selection"] == {
        "k_rad_per_m": 1.25e6,
        "f_hz": 2.5e9,
        "source": "display",
        "marker": [1.0, 2.0],
    }
    assert exported["viewer"]["mode_components"] == ["z", "+"]
    assert exported["viewer"]["options"]["auto_animate"] is True
    assert exported["viewer"]["options"]["alpha"] == 0.5
    json.dumps(exported)

    preset_path = tmp_path / "dispersion-viewer-preset.json"
    assert viewer.save_preset(preset_path) == preset_path

    reloaded = result.plot.interactive(show=False, components=["x"], fmax=1.0)
    assert reloaded.load_preset(preset_path) is reloaded

    assert reloaded.mode_components == ["z", "+"]
    assert reloaded.spectrum_components == ["z", "+"]
    assert reloaded.options["fmax"] == 8.0
    assert reloaded.options["auto_animate"] is True
    assert reloaded.options["positive_frequencies"] is True


def test_dispersion_interactive_viewer_exposes_result_notes_in_state_export_and_html():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        notes=[
            "Sampling warning: only 6 time samples",
            "Use <raw> spectrum for analysis",
        ],
    )

    viewer = result.plot.interactive(show=False)
    exported = viewer.export_selection(k_rad_per_m=0.0, f_hz=0.0)
    html = viewer._repr_html_()

    assert viewer.state["result_notes"] == result.notes
    assert exported["viewer"]["result_notes"] == result.notes
    assert "Sampling warning: only 6 time samples" in html
    assert "Use &lt;raw&gt; spectrum for analysis" in html
    assert "Lightweight status view" in html
    assert "call with store_complex=True" in html


def test_fft_dispersion_plot_accessor_has_notebook_helper_card():
    from mmpp.fft.dispersion.interface import FFTDispersionInterface

    parent_fft = types.SimpleNamespace(
        job_result=types.SimpleNamespace(
            path="/tmp/example.zarr",
            name="example",
        )
    )
    iface = FFTDispersionInterface(parent_fft, dataset_name="m")
    plot = iface.plot

    html = plot._repr_html_()

    assert "Dispersion Plot Helper" in html
    assert "plot.interactive(axis=&#x27;x&#x27;" in html
    assert "show=False" in html
    assert "toolbar=&#x27;auto&#x27;" in html
    assert "html_tabs" not in html
    assert "<h3" not in html
    bundle = plot._repr_mimebundle_()
    assert bundle["text/html"] == html
    assert "FFTDispersionInterface.plot" in bundle["text/plain"]


def test_dispersion_interactive_viewer_show_builds_heatmap_widget(monkeypatch):
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.arange(n_k * n_f, dtype=np.float32).reshape(n_k, n_f),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )

    displayed = []

    class FakeDisplayHandle:
        def __init__(self, payload):
            self.payload = payload
            self.updated = []

        def update(self, value):
            self.updated.append(value)

    fake_display_mod = types.ModuleType("IPython.display")
    fake_display_mod.display = lambda payload, display_id=True: (
        displayed.append(payload) or FakeDisplayHandle(payload)
    )
    fake_ipython = types.ModuleType("IPython")
    fake_ipython.display = fake_display_mod

    class FakeWidget:
        def __init__(self, *children, **kwargs):
            self.children = tuple(children)
            self.kwargs = kwargs
            self.value = kwargs.get("value")
            self.description = kwargs.get("description", "")
            self.options = kwargs.get("options", [])
            self.max = kwargs.get("max")
            self.min = kwargs.get("min")
            self.step = kwargs.get("step")
            self._observers = []

        def observe(self, callback, names=None):
            self._observers.append((callback, names))

        def close(self):
            self.closed = True

    fake_widgets = types.ModuleType("ipywidgets")
    fake_widgets.VBox = lambda children=(), **kwargs: FakeWidget(*children, **kwargs)
    fake_widgets.HBox = lambda children=(), **kwargs: FakeWidget(*children, **kwargs)
    fake_widgets.Output = lambda **kwargs: FakeWidget(**kwargs)
    fake_widgets.Checkbox = FakeWidget
    fake_widgets.Dropdown = FakeWidget
    fake_widgets.FloatSlider = FakeWidget
    fake_widgets.FloatText = FakeWidget
    fake_widgets.HTML = FakeWidget

    class FakeCanvas:
        def draw_idle(self):
            self.drawn = True

    class FakeFigure:
        def __init__(self):
            self.canvas = FakeCanvas()

    class FakeAxes:
        def __init__(self):
            self.images = []
            self.title = ""
            self.xlabel = ""
            self.ylabel = ""

        def imshow(self, data, **kwargs):
            image = types.SimpleNamespace(data=data, kwargs=kwargs)
            self.images.append(image)
            return image

        def clear(self):
            self.images.clear()

        def set_title(self, value):
            self.title = value

        def set_xlabel(self, value):
            self.xlabel = value

        def set_ylabel(self, value):
            self.ylabel = value

        def set_xlim(self, *args, **kwargs):
            pass

        def set_ylim(self, *args, **kwargs):
            pass

        def grid(self, *args, **kwargs):
            pass

    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_fig = FakeFigure()
    fake_ax = FakeAxes()
    fake_pyplot.subplots = lambda *args, **kwargs: (fake_fig, fake_ax)
    fake_pyplot.close = lambda fig=None: setattr(fig, "closed", True)
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_colors = types.ModuleType("matplotlib.colors")
    fake_colors.LogNorm = lambda **kwargs: ("LogNorm", kwargs)

    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display_mod)
    monkeypatch.setitem(sys.modules, "ipywidgets", fake_widgets)
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)
    monkeypatch.setitem(sys.modules, "matplotlib.colors", fake_colors)

    viewer = result.plot.interactive(show=False, fmax=1.0)
    assert viewer._widget is None

    assert viewer.show() is viewer

    assert viewer._widget is not None
    assert viewer._figure is fake_fig
    assert viewer._axes is fake_ax
    assert fake_ax.images
    assert viewer.state["widget_status"] == "ready"
    assert viewer.state["options"]["fmax"] == 1.0
    assert {"tabs", "status_log", "preset_select", "output"}.issubset(viewer._controls)
    assert viewer.diagnostics()["toolbar_enabled"] is True
    assert displayed and displayed[-1] is viewer._widget

    viewer.close()
    assert viewer._widget is None
    assert viewer._figure is None
    assert viewer._axes is None
    assert fake_fig.closed is True


def test_dispersion_notebook_repr_documents_public_accessors():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=np.ones((n_k, n_f), dtype=np.complex128),
        notes=["Sampling warning: synthetic smoke"],
    )

    result_html = result._repr_html_()
    assert "DispersionResult1D" in result_html
    assert ".plot" in result_html
    assert ".analyze" in result_html
    assert ".modes" in result_html
    assert ".filtered(...)" in result_html

    plot_html = result.plot._repr_html_()
    assert "Dispersion Plot Accessor" in plot_html
    assert ".interactive(show=False)" in plot_html
    assert ".heatmap(fmax=10, lognorm=True)" in plot_html
    assert ".branch(branch" in plot_html
    assert ".add_analytics(ax" in plot_html

    analyze_html = result.analyze._repr_html_()
    assert "DispersionAnalyzeAccessor" in analyze_html
    assert ".find_lowest_possible_frequency()" in analyze_html

    modes_html = result.modes._repr_html_()
    assert "DispersionModesBridge" in modes_html
    assert ".interactive(lattice_constant_nm=470)" in modes_html
    assert ".at(k_rad_um=2.3, f_ghz=5.0)" in modes_html

    viewer_html = result.plot.interactive(show=False)._repr_html_()
    assert "DispersionInteractiveViewer" in viewer_html
    assert "show=False, widget not shown" in viewer_html
    assert "Dispersion interactive viewer API help" not in viewer_html
    assert "Sampling warning: synthetic smoke" in viewer_html


def test_dispersion_interactive_viewer_show_and_close_update_display_state(monkeypatch):
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    class _DisplayHandle:
        def __init__(self):
            self.updates = []

        def update(self, value):
            self.updates.append(value)

    calls = []
    display_handle = _DisplayHandle()

    def fake_display(obj, display_id=False):
        calls.append({"obj": obj, "display_id": display_id})
        return display_handle

    ipython_module = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")
    display_module.display = fake_display
    monkeypatch.setitem(sys.modules, "IPython", ipython_module)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    viewer = result.plot.interactive(show=False)

    assert viewer.show_requested is False
    assert viewer.show() is viewer
    assert viewer.show_requested is True
    assert calls == [{"obj": viewer, "display_id": True}]
    assert viewer._display_handle is display_handle

    viewer.close()
    assert viewer.show_requested is False
    assert viewer._display_handle is None
    assert display_handle.updates == [None]


def test_dispersion_result_filtered_preserves_raw_spectrum_and_updates_display():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    raw = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 100.0],
        ],
        dtype=np.float32,
    )
    k_axis = np.array([-1.0, 1.0])
    f_axis = np.array([0.0, 1.0, 2.0])
    result = DispersionResult1D(
        S=raw.copy(),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0),
        dt=1.0,
        dx=1.0,
    )

    filtered = result.filtered(
        live={
            "percentile_autoscale": {
                "low_percentile": 0.0,
                "high_percentile": 50.0,
            }
        }
    )

    assert np.allclose(result.S_raw, raw)
    assert np.allclose(filtered.S_raw, raw)
    assert filtered.S_display is filtered.S
    assert not np.allclose(filtered.S_display, raw)
    assert float(filtered.S_display.max()) <= float(np.percentile(raw, 50.0))


def test_sample_at_k_defaults_to_raw_spectrum_not_display_view():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    raw = np.array([[10.0, 1.0]], dtype=np.float32)
    display = np.array([[1.0, 20.0]], dtype=np.float32)
    result = DispersionResult1D(
        S=display.copy(),
        S_raw=raw.copy(),
        S_display=display.copy(),
        k_axis=np.array([0.0]),
        f_axis=np.array([1.0, 2.0]),
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0, dk_max=1.0),
        dt=1.0,
        dx=1.0,
    )

    _, f_raw = result.sample_at_k(0.0)
    _, f_display = result.sample_at_k(0.0, analysis_source="display")

    assert f_raw == 1.0
    assert f_display == 2.0


def test_select_orthogonal_slice_preserves_local_raw_display_split():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    local_raw = np.array(
        [
            [[10.0, 1.0], [8.0, 1.0]],
            [[1.0, 12.0], [1.0, 9.0]],
        ],
        dtype=np.float32,
    )
    local_display = np.array(
        [
            [[1.0, 20.0], [1.0, 16.0]],
            [[14.0, 1.0], [11.0, 1.0]],
        ],
        dtype=np.float32,
    )
    result = DispersionResult1D(
        S=np.sum(local_display, axis=0),
        S_raw=np.sum(local_raw, axis=0),
        S_display=np.sum(local_display, axis=0),
        S_local=local_display.copy(),
        S_local_raw=local_raw.copy(),
        S_local_display=local_display.copy(),
        k_axis=np.array([-1.0, 1.0]),
        f_axis=np.array([1.0, 2.0]),
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0, dk_max=3.0),
        dt=1.0,
        dx=1.0,
    )

    sliced = result.select_orthogonal_slice(0)
    _, f_raw = sliced.sample_at_k(0.0)
    _, f_display = sliced.sample_at_k(0.0, analysis_source="display")

    assert sliced.S_local is None
    assert np.allclose(sliced.S_raw, local_raw[0])
    assert np.allclose(sliced.S_display, local_display[0])
    assert sliced.S is sliced.S_display
    assert f_raw == 1.0
    assert f_display == 2.0


def test_kmax_trim_preserves_local_raw_display_split():
    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    local_raw = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    local_display = local_raw + 100.0
    result = DispersionResult1D(
        S=np.sum(local_display, axis=0),
        S_raw=np.sum(local_raw, axis=0),
        S_display=np.sum(local_display, axis=0),
        S_local=local_display.copy(),
        S_local_raw=local_raw.copy(),
        S_local_display=local_display.copy(),
        k_axis=np.array([-3.0, -1.0, 1.0, 3.0]),
        f_axis=np.array([0.0, 1.0, 2.0]),
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0),
        dt=1.0,
        dx=1.0,
    )

    iface = FFTDispersionInterface.__new__(FFTDispersionInterface)
    trimmed = iface._trim_dispersion_kmax(result, 1.5)

    assert trimmed.k_axis.tolist() == [-1.0, 1.0]
    assert np.allclose(trimmed.S_local_raw, local_raw[:, 1:3, :])
    assert np.allclose(trimmed.S_local_display, local_display[:, 1:3, :])
    assert trimmed.S_local is trimmed.S_local_display


def test_dispersion_cache_roundtrip_preserves_local_raw_display_split(tmp_path):
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    local_raw = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    local_display = local_raw + 50.0
    result = DispersionResult1D(
        S=np.sum(local_display, axis=0),
        S_raw=np.sum(local_raw, axis=0),
        S_display=np.sum(local_display, axis=0),
        S_local=local_display.copy(),
        S_local_raw=local_raw.copy(),
        S_local_display=local_display.copy(),
        k_axis=np.array([-1.0, 0.0, 1.0]),
        f_axis=np.array([0.0, 1.0, 2.0, 3.0]),
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0, scaling="amplitude_squared"),
        dt=1.0,
        dx=1.0,
        scaling="amplitude_squared",
        scaling_factors={"scale": 2.0},
        notes=["cache roundtrip"],
    )

    iface = FFTDispersionInterface.__new__(FFTDispersionInterface)
    iface.dataset_name = "m"
    iface.slice_info = None
    iface.parent_fft = SimpleNamespace(
        job_result=SimpleNamespace(name="run", path=tmp_path / "run.zarr")
    )
    dataset_group = zarr.open(str(tmp_path / "cache.zarr"), mode="w")

    iface._save_dispersion_result(
        dataset_group,
        "dispersion1d_local",
        result,
        context_json='{"axis":"x"}',
        context_hash="localhash",
        overwrite=True,
    )
    loaded = iface._load_cached_dispersion_result(
        dataset_group,
        "dispersion1d_local",
        "localhash",
    )

    assert loaded is not None
    assert np.allclose(loaded.S_raw, result.S_raw)
    assert np.allclose(loaded.S_display, result.S_display)
    assert np.allclose(loaded.S_local_raw, local_raw)
    assert np.allclose(loaded.S_local_display, local_display)
    assert loaded.S_local is loaded.S_local_display
    assert loaded.scaling == "amplitude_squared"
    assert loaded.scaling_factors == {"scale": 2.0}
    assert loaded.notes == ["cache roundtrip"]


def test_dispersion_cache_context_hash_includes_array_filter_values(tmp_path):
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig

    iface = FFTDispersionInterface.__new__(FFTDispersionInterface)
    iface._config = DispersionConfig(dt=1.0, dx=1.0)
    iface.dataset_name = "m"
    iface.slice_info = (slice(None, None, 2), Ellipsis)
    iface.parent_fft = SimpleNamespace(
        job_result=SimpleNamespace(name="run", path=tmp_path / "run.zarr")
    )
    iface._tmax = 64

    base_payload = {
        "store_complex": False,
        "scaling": "amplitude_squared",
        "flipx": True,
    }
    context_a = iface._build_cache_context(
        mode="dispersion_1d",
        axis="x",
        component="mx",
        extra_kwargs={
            **base_payload,
            "filters": {"post": {"mask": np.array([1.0, 2.0])}},
        },
    )
    context_b = iface._build_cache_context(
        mode="dispersion_1d",
        axis="x",
        component="mx",
        extra_kwargs={
            **base_payload,
            "filters": {"post": {"mask": np.array([3.0, 4.0])}},
        },
    )

    json_a, hash_a = iface._context_signature(context_a)
    json_b, hash_b = iface._context_signature(context_b)

    assert hash_a != hash_b
    assert '"value_sha1"' in json_a
    assert '"value_sha1"' in json_b


def test_apply_live_filters_preserves_local_raw_and_updates_local_display():
    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    local_raw = np.array(
        [
            [[0.0, 1.0, 50.0], [3.0, 4.0, 100.0]],
            [[2.0, 3.0, 80.0], [4.0, 5.0, 120.0]],
        ],
        dtype=np.float32,
    )
    local_display = local_raw.copy()
    result = DispersionResult1D(
        S=np.sum(local_display, axis=0),
        S_raw=np.sum(local_raw, axis=0),
        S_display=np.sum(local_display, axis=0),
        S_local=local_display.copy(),
        S_local_raw=local_raw.copy(),
        S_local_display=local_display.copy(),
        k_axis=np.array([-1.0, 1.0]),
        f_axis=np.array([0.0, 1.0, 2.0]),
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0),
        dt=1.0,
        dx=1.0,
    )
    iface = FFTDispersionInterface.__new__(FFTDispersionInterface)
    filtered = iface.apply_live_filters(
        result,
        filters={
            "live": {
                "percentile_autoscale": {
                    "low_percentile": 0.0,
                    "high_percentile": 50.0,
                }
            }
        },
        include_configured=False,
        apply_to_local=True,
    )

    assert np.allclose(filtered.S_local_raw, local_raw)
    assert filtered.S_local is filtered.S_local_display
    assert not np.allclose(filtered.S_local_display, local_raw)
    assert float(filtered.S_local_display.max()) < float(local_raw.max())
    assert np.allclose(result.S_local_raw, local_raw)
    assert np.allclose(result.S_local_display, local_display)


def test_frequency_view_defaults_to_positive_frequencies_and_can_keep_full_axis():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    spectrum = np.arange(8, dtype=np.float32).reshape(2, 4)
    result = DispersionResult1D(
        S=spectrum.copy(),
        k_axis=np.array([-1.0, 1.0]),
        f_axis=np.array([-2.0, -1.0, 0.0, 1.0]),
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0),
        dt=1.0,
        dx=1.0,
    )

    S_pos, _, f_pos = result.frequency_view()
    S_full, _, f_full = result.frequency_view(positive_frequencies=False)

    assert f_pos.tolist() == [0.0, 1.0]
    assert S_pos.tolist() == [[2.0, 3.0], [6.0, 7.0]]
    assert f_full.tolist() == [-2.0, -1.0, 0.0, 1.0]
    assert np.allclose(S_full, spectrum)


def test_find_branches_defaults_to_raw_spectrum_not_display_view():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    k_axis = np.array([-1.0, 0.0, 1.0])
    f_axis = np.array([0.0, 1.0, 2.0, 3.0])
    raw = np.tile(np.array([[0.1, 10.0, 0.1, 0.1]], dtype=np.float32), (3, 1))
    display = np.tile(np.array([[0.1, 0.1, 20.0, 0.1]], dtype=np.float32), (3, 1))
    result = DispersionResult1D(
        S=display.copy(),
        S_raw=raw.copy(),
        S_display=display.copy(),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0),
        dt=1.0,
        dx=1.0,
    )

    raw_branches = result.analyze.find_branches(
        n_branches=1,
        min_branch_length=3,
        min_quality=0.0,
        noise_floor_percentile=0.0,
        min_prominence_log=0.0,
        max_df_ghz=1e-8,
        smooth_sigma=None,
        fmin_hz=0.0,
    )
    display_branches = result.analyze.find_branches(
        n_branches=1,
        min_branch_length=3,
        min_quality=0.0,
        noise_floor_percentile=0.0,
        min_prominence_log=0.0,
        max_df_ghz=1e-8,
        smooth_sigma=None,
        fmin_hz=0.0,
        analysis_source="display",
    )

    assert np.allclose(raw_branches.branches[0].f_hz, 1.0)
    assert np.allclose(display_branches.branches[0].f_hz, 2.0)
    assert raw_branches.branches[0].quality_metrics["coverage"] == 1.0
    assert (
        raw_branches.branches[0].quality_metrics["confidence"]
        == raw_branches.branches[0].quality
    )


def test_find_branches_can_search_full_frequency_axis_when_requested():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    k_axis = np.array([-1.0, 0.0, 1.0])
    f_axis = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    raw = np.tile(np.array([[0.1, 10.0, 0.1, 1.0, 0.1]], dtype=np.float32), (3, 1))
    result = DispersionResult1D(
        S=raw.copy(),
        S_raw=raw.copy(),
        S_display=raw.copy(),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0),
        dt=1.0,
        dx=1.0,
    )

    positive = result.analyze.find_branches(
        n_branches=1,
        min_branch_length=3,
        min_quality=0.0,
        noise_floor_percentile=0.0,
        min_prominence_log=0.0,
        max_df_ghz=1e-8,
        smooth_sigma=None,
        fmin_hz=0.0,
    )
    full = result.analyze.find_branches(
        n_branches=1,
        min_branch_length=3,
        min_quality=0.0,
        noise_floor_percentile=0.0,
        min_prominence_log=0.0,
        max_df_ghz=1e-8,
        smooth_sigma=None,
        fmin_hz=None,
        positive_frequencies=False,
    )

    assert np.allclose(positive.branches[0].f_hz, 1.0)
    assert np.allclose(full.branches[0].f_hz, -1.0)


def test_find_branches_reports_rejected_candidates():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    k_axis = np.array([-1.0, 0.0, 1.0])
    f_axis = np.array([0.0, 1.0, 2.0])
    raw = np.tile(np.array([[0.1, 10.0, 0.1]], dtype=np.float32), (3, 1))
    result = DispersionResult1D(
        S=raw.copy(),
        S_raw=raw.copy(),
        S_display=raw.copy(),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0),
        dt=1.0,
        dx=1.0,
    )

    branches = result.analyze.find_branches(
        n_branches=1,
        min_branch_length=4,
        min_quality=0.0,
        noise_floor_percentile=0.0,
        min_prominence_log=0.0,
        max_df_ghz=1e-8,
        smooth_sigma=None,
        fmin_hz=0.0,
    )

    assert len(branches.branches) == 0
    assert branches.rejected
    assert branches.rejected[0]["reason"] == "min_branch_length"


def test_find_branches_tracks_noisy_branch_and_reports_quality_and_rejections():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    rng = np.random.default_rng(123)
    n_k, n_f = 12, 10
    k_axis = np.linspace(-5e6, 5e6, n_k)
    f_axis = np.arange(n_f, dtype=float) * 1e9
    spectrum = (0.02 + 0.01 * rng.random((n_k, n_f))).astype(np.float32)
    main_f_idx = 2 + (np.arange(n_k) // 3)
    for idx_k, idx_f in enumerate(main_f_idx):
        spectrum[idx_k, idx_f] += 10.0
    for idx_k in range(1, 4):
        spectrum[idx_k, 8] += 7.0

    result = DispersionResult1D(
        S=spectrum.copy(),
        S_raw=spectrum.copy(),
        S_display=spectrum.copy(),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0),
        dt=1.0,
        dx=1.0,
    )

    branches = result.analyze.find_branches(
        n_branches=2,
        min_branch_length=6,
        min_quality=0.1,
        noise_floor_percentile=20.0,
        min_prominence_log=0.1,
        min_peak_distance=1,
        max_df_ghz=1.1,
        smooth_sigma=None,
        fmin_hz=0.0,
    )

    assert len(branches.branches) == 1
    branch = branches.branches[0]
    metrics = branch.quality_metrics

    assert branch.k.shape == (n_k,)
    assert np.allclose(branch.f_hz, f_axis[main_f_idx])
    assert branch.quality == metrics["confidence"]
    assert metrics["coverage"] == 1.0
    assert 0.0 < metrics["smoothness"] <= 1.0
    assert 0.0 < metrics["snr"] <= 1.0
    assert any(rej["reason"] == "min_branch_length" for rej in branches.rejected)


def _write_complex_wave_zarr(
    path,
    *,
    n_t: int,
    n_x: int,
    amplitude: float = 0.25,
    f_bin: int = 1,
    k_bin: int = 1,
    dt: float = 2e-12,
    dx: float = 5e-9,
):
    t = np.arange(n_t, dtype=float)[:, None]
    x = np.arange(n_x, dtype=float)[None, :]
    phase = 2.0 * np.pi * ((f_bin * t / n_t) + (k_bin * x / n_x))
    wave = amplitude * np.exp(1j * phase)
    data = np.zeros((n_t, 1, 1, n_x, 3), dtype=np.float32)
    data[:, 0, 0, :, 0] = wave.real
    data[:, 0, 0, :, 1] = wave.imag

    root = zarr.open(str(path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root.attrs["t_sampl"] = dt
    root.attrs["dx"] = dx
    root.attrs["dy"] = dx
    return path


def _write_complex_y_wave_zarr(
    path,
    *,
    n_t: int,
    n_y: int,
    n_x: int = 2,
    amplitude: float = 0.25,
    f_bin: int = 1,
    k_bin: int = 1,
    dt: float = 2e-12,
    dx: float = 5e-9,
    dy: float = 7e-9,
):
    t = np.arange(n_t, dtype=float)[:, None]
    y = np.arange(n_y, dtype=float)[None, :]
    phase = 2.0 * np.pi * ((f_bin * t / n_t) + (k_bin * y / n_y))
    wave = amplitude * np.exp(1j * phase)
    data = np.zeros((n_t, 1, n_y, n_x, 3), dtype=np.float32)
    data[:, 0, :, :, 0] = wave.real[..., None]
    data[:, 0, :, :, 1] = wave.imag[..., None]

    root = zarr.open(str(path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root.attrs["t_sampl"] = dt
    root.attrs["dx"] = dx
    root.attrs["dy"] = dy
    return path


def _write_complex_xy_wave_zarr(
    path,
    *,
    n_t: int,
    n_y: int,
    n_x: int,
    amplitude: float = 0.25,
    f_bin: int = 1,
    kx_bin: int = 1,
    ky_bin: int = 1,
    dt: float = 2e-12,
    dx: float = 5e-9,
    dy: float = 7e-9,
):
    t = np.arange(n_t, dtype=float)[:, None, None]
    y = np.arange(n_y, dtype=float)[None, :, None]
    x = np.arange(n_x, dtype=float)[None, None, :]
    phase = 2.0 * np.pi * ((f_bin * t / n_t) + (ky_bin * y / n_y) + (kx_bin * x / n_x))
    wave = amplitude * np.exp(1j * phase)
    data = np.zeros((n_t, 1, n_y, n_x, 3), dtype=np.float32)
    data[:, 0, :, :, 0] = wave.real
    data[:, 0, :, :, 1] = wave.imag

    root = zarr.open(str(path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root.attrs["t_sampl"] = dt
    root.attrs["dx"] = dx
    root.attrs["dy"] = dy
    return path


def _write_complex_x_wave_with_y_phases_zarr(
    path,
    *,
    n_t: int,
    n_y: int,
    n_x: int,
    phase_offsets: np.ndarray,
    amplitude: float = 0.25,
    f_bin: int = 1,
    k_bin: int = 1,
    dt: float = 2e-12,
    dx: float = 5e-9,
    dy: float = 7e-9,
):
    t = np.arange(n_t, dtype=float)[:, None, None]
    y_phase = phase_offsets[None, :, None]
    x = np.arange(n_x, dtype=float)[None, None, :]
    phase = 2.0 * np.pi * ((f_bin * t / n_t) + (k_bin * x / n_x)) + y_phase
    wave = amplitude * np.exp(1j * phase)
    data = np.zeros((n_t, 1, n_y, n_x, 3), dtype=np.float32)
    data[:, 0, :, :, 0] = wave.real
    data[:, 0, :, :, 1] = wave.imag

    root = zarr.open(str(path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root.attrs["t_sampl"] = dt
    root.attrs["dx"] = dx
    root.attrs["dy"] = dy
    return path


def _write_real_mz_wave_zarr(
    path,
    *,
    n_t: int,
    n_x: int,
    amplitude: float = 0.25,
    f_bin: int = 1,
    k_bin: int = 1,
    dt: float = 2e-12,
    dx: float = 5e-9,
):
    t = np.arange(n_t, dtype=float)[:, None]
    x = np.arange(n_x, dtype=float)[None, :]
    phase = 2.0 * np.pi * ((f_bin * t / n_t) + (k_bin * x / n_x))
    wave = amplitude * np.cos(phase)
    data = np.zeros((n_t, 1, 1, n_x, 3), dtype=np.float32)
    data[:, 0, 0, :, 2] = wave

    root = zarr.open(str(path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root.attrs["t_sampl"] = dt
    root.attrs["dx"] = dx
    root.attrs["dy"] = dx
    return path


def test_dispersion_amplitude_squared_scaling_corrects_fft_and_window_gain(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    amplitude = 0.25
    small_path = _write_complex_wave_zarr(
        tmp_path / "small_wave.zarr",
        n_t=8,
        n_x=8,
        amplitude=amplitude,
    )
    large_path = _write_complex_wave_zarr(
        tmp_path / "large_wave.zarr",
        n_t=16,
        n_x=8,
        amplitude=amplitude,
    )

    no_window_config = DispersionConfig(
        time_window=None, space_window=None, detrend=None
    )
    small = SpinWaveAnalyzer(small_path, config=no_window_config, tmax=None)
    large = SpinWaveAnalyzer(large_path, config=no_window_config, tmax=None)

    raw_small = small.compute_dispersion_1d(
        axis="x",
        component="perp",
        scaling="raw_power",
        store_complex=False,
    )
    raw_large = large.compute_dispersion_1d(
        axis="x",
        component="perp",
        scaling="raw_power",
        store_complex=False,
    )
    scaled_small = small.compute_dispersion_1d(
        axis="x",
        component="perp",
        scaling="amplitude_squared",
        store_complex=False,
    )
    scaled_large = large.compute_dispersion_1d(
        axis="x",
        component="perp",
        scaling="amplitude_squared",
        store_complex=False,
    )
    scaled_windowed = small.compute_dispersion_1d(
        axis="x",
        component="perp",
        time_window="hann",
        space_window="hann",
        detrend=None,
        scaling="amplitude_squared",
        store_complex=False,
    )
    psd_small = small.compute_dispersion_1d(
        axis="x",
        component="perp",
        scaling="psd",
        store_complex=False,
    )

    assert np.isclose(raw_large.S_raw.max() / raw_small.S_raw.max(), 4.0, rtol=1e-5)
    assert np.isclose(scaled_small.S_raw.max(), amplitude**2, rtol=1e-5)
    assert np.isclose(scaled_large.S_raw.max(), amplitude**2, rtol=1e-5)
    assert np.isclose(scaled_windowed.S_raw.max(), amplitude**2, rtol=1e-5)
    assert np.isclose(
        psd_small.S_raw.max(),
        raw_small.S_raw.max() * 2e-12 * 5e-9 / (8 * 8),
        rtol=1e-5,
    )
    assert scaled_small.scaling == "amplitude_squared"
    assert psd_small.scaling == "psd"
    assert scaled_small.scaling_factors["coherent_gain"] == 64.0


def test_compute_1d_complex_wave_extracts_spatial_mode_end_to_end(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig
    from mmpp.fft.dispersion.modes.extraction import extract_mode_2d

    n_t, n_x = 16, 16
    dt, dx = 2e-12, 5e-9
    zarr_path = _write_complex_wave_zarr(
        tmp_path / "mode_e2e_wave.zarr",
        n_t=n_t,
        n_x=n_x,
        amplitude=0.25,
        f_bin=2,
        k_bin=3,
        dt=dt,
        dx=dx,
    )
    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(time_window=None, space_window=None, detrend=None),
        tmax=None,
    )

    result = analyzer.compute_dispersion_1d(
        axis="x",
        component="perp",
        avg_over_orthogonal=False,
        orthogonal_avg_mode="fft_power",
        store_complex=True,
        scaling="amplitude_squared",
    )

    assert result.S_complex is not None
    assert result.S_complex.shape == (1, n_x, n_t)
    idx_k, idx_f = np.unravel_index(int(np.argmax(result.S_raw)), result.S_raw.shape)
    k0 = float(result.k_axis[idx_k])
    f0 = float(result.f_axis[idx_f])

    x_axis, y_axis, mode_2d, info = extract_mode_2d(
        result,
        k_0=k0,
        f_0=f0,
        lattice_constant=10.0,
        n_bz=0,
        k_margin_bins=0,
        f_margin_bins=0,
    )

    assert mode_2d.shape == (1, n_x)
    assert x_axis.shape == (n_x,)
    assert y_axis.shape == (1,)
    assert info["k_bins_selected"] == 1
    assert info["f_bins_selected"] == 1

    expected = np.exp(1j * k0 * x_axis)
    reconstructed = mode_2d[0] / np.max(np.abs(mode_2d[0]))
    phase = np.vdot(expected, reconstructed)
    phase /= abs(phase)

    assert np.allclose(reconstructed, phase * expected, atol=1e-5)


def test_compute_1d_local_spectra_preserve_orthogonal_phase_offsets(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    n_t, n_y, n_x = 16, 3, 16
    dt, dx, dy = 2e-12, 5e-9, 7e-9
    phase_offsets = np.array([0.0, 0.5 * np.pi, np.pi])
    zarr_path = _write_complex_x_wave_with_y_phases_zarr(
        tmp_path / "local_phase_offsets.zarr",
        n_t=n_t,
        n_y=n_y,
        n_x=n_x,
        phase_offsets=phase_offsets,
        amplitude=0.25,
        f_bin=2,
        k_bin=3,
        dt=dt,
        dx=dx,
        dy=dy,
    )
    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(time_window=None, space_window=None, detrend=None),
        tmax=None,
    )

    result = analyzer.compute_dispersion_1d(
        axis="x",
        component="perp",
        avg_over_orthogonal=False,
        orthogonal_avg_mode="fft_power",
        store_complex=True,
        scaling="amplitude_squared",
        flipx=False,
    )
    collapsed = analyzer.compute_dispersion_1d(
        axis="x",
        component="perp",
        avg_over_orthogonal=True,
        orthogonal_avg_mode="fft_power",
        store_complex=True,
        scaling="amplitude_squared",
        flipx=False,
    )

    assert result.S_local is not None
    assert result.S_complex is not None
    assert result.S_local.shape == (n_y, n_x, n_t)
    assert result.S_complex.shape == (n_y, n_x, n_t)
    assert np.allclose(result.orth_axis, np.arange(n_y) * dy)
    assert result.orth_axis_label == "y"
    assert collapsed.S_complex is None
    assert collapsed.S_local is None

    idx_k, idx_f = np.unravel_index(int(np.argmax(result.S_raw)), result.S_raw.shape)
    peak_values = result.S_complex[:, idx_k, idx_f]
    phase_ratios = peak_values / peak_values[0]
    expected_ratios = np.exp(1j * (phase_offsets - phase_offsets[0]))

    assert np.isclose(result.k_axis[idx_k], 3 * 2 * np.pi / (n_x * dx))
    assert np.isclose(result.f_axis[idx_f], 2 / (n_t * dt))
    assert np.allclose(phase_ratios, expected_ratios, atol=1e-6)


@pytest.mark.parametrize(
    ("axis", "n_space", "k_bin", "spacing"),
    [
        ("x", 16, 3, 5e-9),
        ("x", 15, -3, 5e-9),
        ("y", 16, 2, 7e-9),
        ("y", 15, -2, 7e-9),
    ],
)
def test_compute_1d_resolves_signed_wave_on_x_y_even_and_odd_grids(
    tmp_path,
    axis,
    n_space,
    k_bin,
    spacing,
):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    n_t, f_bin = 16, 2
    if axis == "x":
        zarr_path = _write_complex_wave_zarr(
            tmp_path / f"signed_{axis}_{n_space}_{k_bin}.zarr",
            n_t=n_t,
            n_x=n_space,
            amplitude=0.25,
            f_bin=f_bin,
            k_bin=k_bin,
            dx=spacing,
        )
    else:
        zarr_path = _write_complex_y_wave_zarr(
            tmp_path / f"signed_{axis}_{n_space}_{k_bin}.zarr",
            n_t=n_t,
            n_y=n_space,
            n_x=3,
            amplitude=0.25,
            f_bin=f_bin,
            k_bin=k_bin,
            dy=spacing,
        )
    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(time_window=None, space_window=None, detrend=None),
        tmax=None,
    )

    result = analyzer.compute_dispersion_1d(
        axis=axis,
        component="perp",
        avg_over_orthogonal=True,
        orthogonal_avg_mode="fft_power",
        store_complex=False,
        scaling="amplitude_squared",
        flipx=False,
    )

    idx_k, idx_f = np.unravel_index(int(np.argmax(result.S_raw)), result.S_raw.shape)

    assert result.axis == axis
    assert result.S_raw.shape == (n_space, n_t)
    assert np.isclose(result.dx, spacing)
    assert np.isclose(result.k_axis[idx_k], k_bin * 2 * np.pi / (n_space * spacing))
    assert np.isclose(result.f_axis[idx_f], f_bin / (n_t * 2e-12))


def test_compute_1d_flipx_mirrors_peak_without_changing_frequency_or_power(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    n_t, n_x = 16, 16
    dt, dx = 2e-12, 5e-9
    expected_k = 3 * 2 * np.pi / (n_x * dx)
    expected_f = 2 / (n_t * dt)
    zarr_path = _write_complex_wave_zarr(
        tmp_path / "flipx_mirror_wave.zarr",
        n_t=n_t,
        n_x=n_x,
        amplitude=0.25,
        f_bin=2,
        k_bin=3,
        dt=dt,
        dx=dx,
    )
    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(time_window=None, space_window=None, detrend=None),
        tmax=None,
    )

    raw_orientation = analyzer.compute_dispersion_1d(
        axis="x",
        component="perp",
        avg_over_orthogonal=True,
        orthogonal_avg_mode="fft_power",
        store_complex=False,
        scaling="amplitude_squared",
        flipx=False,
    )
    mirrored = analyzer.compute_dispersion_1d(
        axis="x",
        component="perp",
        avg_over_orthogonal=True,
        orthogonal_avg_mode="fft_power",
        store_complex=False,
        scaling="amplitude_squared",
        flipx=True,
    )

    idx_k_raw, idx_f_raw = np.unravel_index(
        int(np.argmax(raw_orientation.S_raw)), raw_orientation.S_raw.shape
    )
    idx_k_mirrored, idx_f_mirrored = np.unravel_index(
        int(np.argmax(mirrored.S_raw)), mirrored.S_raw.shape
    )

    assert raw_orientation.flipx is False
    assert mirrored.flipx is True
    assert np.isclose(raw_orientation.k_axis[idx_k_raw], expected_k)
    assert np.isclose(mirrored.k_axis[idx_k_mirrored], -expected_k)
    assert np.isclose(raw_orientation.f_axis[idx_f_raw], expected_f)
    assert np.isclose(mirrored.f_axis[idx_f_mirrored], expected_f)
    assert np.isclose(raw_orientation.S_raw[idx_k_raw, idx_f_raw], 0.25**2)
    assert np.isclose(mirrored.S_raw[idx_k_mirrored, idx_f_mirrored], 0.25**2)


def test_fold_spectrum_1d_aggregates_known_bz_replicas():
    from mmpp.fft.dispersion.utils import fold_spectrum_1d

    period = 10e-9
    reciprocal = 2 * np.pi / period
    k0 = 0.17 * reciprocal
    k_axis = np.array(
        [
            k0 - reciprocal,
            -0.31 * reciprocal,
            k0,
            0.32 * reciprocal,
            k0 + reciprocal,
        ],
        dtype=float,
    )
    spectrum = np.zeros((k_axis.size, 3), dtype=np.float32)
    spectrum[:, 1] = np.array([1.0, 10.0, 2.0, 20.0, 4.0], dtype=np.float32)

    k_folded_sum, folded_sum = fold_spectrum_1d(spectrum, k_axis, period, agg="sum")
    k_folded_max, folded_max = fold_spectrum_1d(spectrum, k_axis, period, agg="max")
    idx = int(np.argmin(np.abs(k_folded_sum - k0)))
    idx_max = int(np.argmin(np.abs(k_folded_max - k0)))

    assert k_folded_sum.shape == (3,)
    assert np.all(k_folded_sum >= -np.pi / period)
    assert np.all(k_folded_sum < np.pi / period)
    assert np.isclose(k_folded_sum[idx], k0)
    assert np.isclose(folded_sum[idx, 1], 1.0 + 2.0 + 4.0)
    assert np.isclose(folded_max[idx_max, 1], 4.0)


def test_compute_1d_y_axis_uses_effective_spacing_after_spatial_stride(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    n_t, n_y = 16, 16
    dt, dy = 2e-12, 7e-9
    zarr_path = _write_complex_y_wave_zarr(
        tmp_path / "y_stride_wave.zarr",
        n_t=n_t,
        n_y=n_y,
        n_x=3,
        amplitude=0.25,
        f_bin=2,
        k_bin=2,
        dt=dt,
        dy=dy,
    )
    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(time_window=None, space_window=None, detrend=None),
        tmax=None,
        slice_info=(
            slice(None),
            slice(None),
            slice(None, None, 2),
            slice(None),
            slice(None),
        ),
    )

    result = analyzer.compute_dispersion_1d(
        axis="y",
        component="perp",
        avg_over_orthogonal=True,
        orthogonal_avg_mode="fft_power",
        store_complex=False,
        scaling="amplitude_squared",
    )

    idx_k, idx_f = np.unravel_index(int(np.argmax(result.S_raw)), result.S_raw.shape)
    effective_n_y = n_y // 2
    effective_dy = 2 * dy
    expected_k_axis = np.fft.fftshift(
        2 * np.pi * np.fft.fftfreq(effective_n_y, effective_dy)
    )

    assert result.axis == "y"
    assert result.S_raw.shape == (effective_n_y, n_t)
    assert np.isclose(result.dx, effective_dy)
    assert np.allclose(result.k_axis, expected_k_axis)
    assert np.isclose(abs(result.f_axis[idx_f]), 2 / (n_t * dt))
    assert np.isclose(abs(result.k_axis[idx_k]), abs(expected_k_axis[2]))


def test_compute_1d_accepts_scalar_component_slice(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    n_t, n_x = 16, 16
    dt, dx = 2e-12, 5e-9
    zarr_path = _write_real_mz_wave_zarr(
        tmp_path / "scalar_component_slice.zarr",
        n_t=n_t,
        n_x=n_x,
        amplitude=0.25,
        f_bin=2,
        k_bin=3,
        dt=dt,
        dx=dx,
    )
    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(time_window=None, space_window=None, detrend=None),
        tmax=None,
        slice_info=(
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            2,
        ),
    )

    result = analyzer.compute_dispersion_1d(
        axis="x",
        component="auto",
        avg_over_orthogonal=True,
        orthogonal_avg_mode="fft_power",
        store_complex=False,
        scaling="amplitude_squared",
    )

    idx_k, idx_f = np.unravel_index(int(np.argmax(result.S_raw)), result.S_raw.shape)

    assert analyzer.M_data.shape == (n_t, 1, 1, n_x, 1)
    assert result.S_raw.shape == (n_x, n_t)
    assert np.isclose(result.dx, dx)
    assert np.isclose(abs(result.f_axis[idx_f]), 2 / (n_t * dt))
    assert np.isclose(abs(result.k_axis[idx_k]), 3 * 2 * np.pi / (n_x * dx))


def test_dispersion_interface_configured_tmax_controls_loaded_time_and_cache_key(
    tmp_path,
):
    from mmpp.fft.dispersion.interface import FFTDispersionInterface

    n_t, n_x = 16, 16
    dt, dx = 2e-12, 5e-9
    zarr_path = _write_complex_wave_zarr(
        tmp_path / "configured_tmax.zarr",
        n_t=n_t,
        n_x=n_x,
        amplitude=0.25,
        f_bin=4,
        k_bin=3,
        dt=dt,
        dx=dx,
    )
    parent_fft = types.SimpleNamespace(
        job_result=types.SimpleNamespace(path=zarr_path, name="run")
    )
    iface = FFTDispersionInterface(parent_fft)
    compute_kwargs = {
        "axis": "x",
        "component": "perp",
        "avg_over_orthogonal": True,
        "orthogonal_avg_mode": "fft_power",
        "store_complex": False,
        "scaling": "amplitude_squared",
        "disk_cache": False,
    }

    short = iface.configure(time_window=None, detrend=None, tmax=8).compute_1d(
        **compute_kwargs
    )
    assert iface.analyzer.M_data.shape[0] == 8
    idx_k_short, idx_f_short = np.unravel_index(
        int(np.argmax(short.S_raw)), short.S_raw.shape
    )

    full = iface.configure(time_window=None, detrend=None, tmax=n_t).compute_1d(
        **compute_kwargs
    )
    assert iface.analyzer.M_data.shape[0] == n_t
    idx_k_full, idx_f_full = np.unravel_index(
        int(np.argmax(full.S_raw)), full.S_raw.shape
    )

    assert short is not full
    assert short.S_raw.shape == (n_x, 8)
    assert full.S_raw.shape == (n_x, n_t)
    assert np.isclose(abs(short.f_axis[idx_f_short]), 2 / (8 * dt))
    assert np.isclose(abs(full.f_axis[idx_f_full]), 4 / (n_t * dt))
    assert np.isclose(abs(short.k_axis[idx_k_short]), 3 * 2 * np.pi / (n_x * dx))
    assert np.isclose(abs(full.k_axis[idx_k_full]), 3 * 2 * np.pi / (n_x * dx))

    windowed = iface.configure(
        time_window=None,
        detrend=None,
        tmin=4,
        tmax=12,
    ).compute_1d(**compute_kwargs)
    assert iface.analyzer.M_data.shape[0] == 8
    assert windowed.S_raw.shape == (n_x, 8)
    assert windowed is not short


def test_dispersion_numpy_and_scipy_backends_match_on_synthetic_wave(tmp_path):
    pytest.importorskip("scipy")

    from mmpp.fft.dispersion import _fft_backend
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    zarr_path = _write_complex_wave_zarr(
        tmp_path / "backend_parity_wave.zarr",
        n_t=16,
        n_x=16,
        amplitude=0.2,
        f_bin=2,
        k_bin=3,
    )
    config = DispersionConfig(time_window=None, space_window=None, detrend=None)
    original = _fft_backend.get_info()
    results = {}

    try:
        for backend in ("numpy", "scipy"):
            _fft_backend.set_backend(backend)
            _fft_backend.set_workers(1)
            analyzer = SpinWaveAnalyzer(zarr_path, config=config, tmax=None)
            results[backend] = analyzer.compute_dispersion_1d(
                axis="x",
                component="perp",
                avg_over_orthogonal=False,
                orthogonal_avg_mode="fft_power",
                store_complex=True,
                scaling="amplitude_squared",
            )
    finally:
        _fft_backend.set_backend(original["backend"])
        _fft_backend.set_workers(original["workers"])

    numpy_result = results["numpy"]
    scipy_result = results["scipy"]
    assert np.allclose(numpy_result.k_axis, scipy_result.k_axis)
    assert np.allclose(numpy_result.f_axis, scipy_result.f_axis)
    assert np.allclose(numpy_result.S_raw, scipy_result.S_raw, rtol=1e-6, atol=1e-9)
    assert np.allclose(
        numpy_result.S_display, scipy_result.S_display, rtol=1e-6, atol=1e-9
    )
    assert numpy_result.S_complex is not None
    assert scipy_result.S_complex is not None
    complex_scale = max(float(np.abs(numpy_result.S_complex).max()), 1.0)
    complex_delta = float(
        np.max(np.abs(numpy_result.S_complex - scipy_result.S_complex))
    )
    assert complex_delta <= complex_scale * 1e-6


def test_dispersion_pyfftw_backend_matches_numpy_when_available(tmp_path):
    pytest.importorskip("pyfftw")

    from mmpp.fft.dispersion import _fft_backend
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    zarr_path = _write_complex_wave_zarr(
        tmp_path / "pyfftw_backend_parity_wave.zarr",
        n_t=16,
        n_x=16,
        amplitude=0.2,
        f_bin=2,
        k_bin=3,
    )
    config = DispersionConfig(time_window=None, space_window=None, detrend=None)
    original = _fft_backend.get_info()
    results = {}

    try:
        for backend in ("numpy", "pyfftw"):
            _fft_backend.set_backend(backend)
            _fft_backend.set_workers(1)
            analyzer = SpinWaveAnalyzer(zarr_path, config=config, tmax=None)
            results[backend] = analyzer.compute_dispersion_1d(
                axis="x",
                component="perp",
                avg_over_orthogonal=False,
                orthogonal_avg_mode="fft_power",
                store_complex=True,
                scaling="amplitude_squared",
            )
    finally:
        _fft_backend.set_backend(original["backend"])
        _fft_backend.set_workers(original["workers"])

    numpy_result = results["numpy"]
    pyfftw_result = results["pyfftw"]
    assert np.allclose(numpy_result.k_axis, pyfftw_result.k_axis)
    assert np.allclose(numpy_result.f_axis, pyfftw_result.f_axis)
    assert np.allclose(numpy_result.S_raw, pyfftw_result.S_raw, rtol=1e-6, atol=1e-9)
    assert np.allclose(
        numpy_result.S_display,
        pyfftw_result.S_display,
        rtol=1e-6,
        atol=1e-9,
    )
    assert numpy_result.S_complex is not None
    assert pyfftw_result.S_complex is not None
    complex_scale = max(float(np.abs(numpy_result.S_complex).max()), 1.0)
    complex_delta = float(
        np.max(np.abs(numpy_result.S_complex - pyfftw_result.S_complex))
    )
    assert complex_delta <= complex_scale * 1e-6


def test_compute_2d_complex_wave_extracts_kx_ky_frequency_peak_and_slice(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    n_t, n_y, n_x = 16, 8, 8
    dt, dy, dx = 2e-12, 7e-9, 5e-9
    zarr_path = _write_complex_xy_wave_zarr(
        tmp_path / "dispersion_2d_wave.zarr",
        n_t=n_t,
        n_y=n_y,
        n_x=n_x,
        amplitude=0.25,
        f_bin=2,
        kx_bin=2,
        ky_bin=1,
        dt=dt,
        dx=dx,
        dy=dy,
    )
    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(time_window=None, detrend=None, dx=dx, dy=dy),
        tmax=None,
    )

    result = analyzer.compute_dispersion_2d(
        component="perp", time_window=None, detrend=None
    )
    idx_kx, idx_ky, idx_f = np.unravel_index(int(np.argmax(result.S)), result.S.shape)
    slice_kx = result.slice_1d("kx", k_value=float(result.ky_axis[idx_ky]), dk_max=0.0)

    assert result.shape == (n_x, n_y, n_t // 2 + 1)
    assert np.allclose(
        result.kx_axis, np.fft.fftshift(2 * np.pi * np.fft.fftfreq(n_x, dx))
    )
    assert np.allclose(
        result.ky_axis, np.fft.fftshift(2 * np.pi * np.fft.fftfreq(n_y, dy))
    )
    assert np.allclose(result.f_axis, np.abs(np.fft.fftfreq(n_t, dt)[: n_t // 2 + 1]))
    assert np.isclose(abs(result.kx_axis[idx_kx]), 2 * 2 * np.pi / (n_x * dx))
    assert np.isclose(abs(result.ky_axis[idx_ky]), 1 * 2 * np.pi / (n_y * dy))
    assert np.isclose(result.f_axis[idx_f], 2 / (n_t * dt))
    assert "experimental" in "\n".join(result.notes or []).lower()
    assert slice_kx.axis == "x"
    assert slice_kx.shape == (n_x, n_t // 2 + 1)
    assert np.allclose(slice_kx.k_axis, result.kx_axis)
    assert "experimental" in "\n".join(slice_kx.notes or []).lower()


def test_dispersion_scaling_rejects_unknown_mode(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    zarr_path = _write_complex_wave_zarr(tmp_path / "bad_scaling.zarr", n_t=8, n_x=8)
    analyzer = SpinWaveAnalyzer(zarr_path, config=DispersionConfig(), tmax=None)

    with pytest.raises(ValueError, match="Unknown dispersion scaling"):
        analyzer.compute_dispersion_1d(
            axis="x",
            component="perp",
            detrend=None,
            scaling="not-a-scaling",
            store_complex=False,
        )


def test_compute_1d_reports_sampling_quality_warnings(tmp_path, caplog):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    zarr_path = _write_complex_wave_zarr(
        tmp_path / "sampling_quality.zarr",
        n_t=6,
        n_x=6,
        f_bin=1,
        k_bin=1,
        dt=1e-12,
        dx=1.0,
    )
    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(
            time_window=None,
            space_window=None,
            detrend=None,
            dk_max=10.0,
        ),
        tmax=None,
    )

    with caplog.at_level(logging.WARNING, logger="mmpp.fft.dispersion.core"):
        result = analyzer.compute_dispersion_1d(
            axis="x",
            component="perp",
            avg_over_orthogonal=True,
            orthogonal_avg_mode="fft_power",
            store_complex=False,
            scaling="amplitude_squared",
        )

    notes = "\n".join(result.notes or [])

    assert "only 6 time samples" in notes
    assert "only 6 spatial samples" in notes
    assert "Nyquist limits" in notes
    assert "config.dk_max exceeds spatial Nyquist limit" in notes
    assert "only 6 time samples" in caplog.text


def test_compute_1d_cache_separates_scaling_modes():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    raw = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        scaling="raw_power",
    )
    scaled = DispersionResult1D(
        S=np.full((n_k, n_f), 2.0, dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        scaling="amplitude_squared",
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr", name="run"))
    )

    class FakeAnalyzer:
        def __init__(self):
            self.calls = []

        def compute_dispersion_1d(self, **kwargs):
            self.calls.append(dict(kwargs))
            return scaled if kwargs.get("scaling") == "amplitude_squared" else raw

    analyzer = FakeAnalyzer()
    iface._analyzer = analyzer

    first = iface.compute_1d(axis="x", disk_cache=False)
    second = iface.compute_1d(axis="x", scaling="raw_power", disk_cache=False)
    third = iface.compute_1d(axis="x", scaling="amplitude_squared", disk_cache=False)
    fourth = iface.compute_1d(axis="x", scaling="raw_power", disk_cache=False)

    assert first is raw
    assert second is raw
    assert third is scaled
    assert fourth is raw
    assert len(analyzer.calls) == 2


def test_compute_1d_cache_separates_axis_configuration():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result_x = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    result_y = DispersionResult1D(
        S=np.full((n_k, n_f), 2.0, dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="y",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr", name="run"))
    )

    class FakeAnalyzer:
        def __init__(self):
            self.calls = []

        def compute_dispersion_1d(self, **kwargs):
            self.calls.append(dict(kwargs))
            return result_y if kwargs.get("axis") == "y" else result_x

    analyzer = FakeAnalyzer()
    iface._analyzer = analyzer

    first = iface.compute_1d(axis="x", disk_cache=False)
    second = iface.compute_1d(axis="y", disk_cache=False)
    third = iface.compute_1d(axis="x", disk_cache=False)

    assert first is result_x
    assert second is result_y
    assert third is result_x
    assert [call["axis"] for call in analyzer.calls] == ["x", "y"]


def test_compute_1d_cache_separates_slice_context():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result_even = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    result_odd = DispersionResult1D(
        S=np.full((n_k, n_f), 2.0, dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    base = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr", name="run"))
    )
    even = base.clone_for_dataset(None, (slice(None, None, 2), Ellipsis))
    odd = base.clone_for_dataset(None, (slice(1, None, 2), Ellipsis))

    class FakeAnalyzer:
        def __init__(self):
            self.calls = []

        def compute_dispersion_1d(self, **kwargs):
            self.calls.append(dict(kwargs))
            return result_odd if len(self.calls) == 2 else result_even

    analyzer = FakeAnalyzer()
    even._analyzer = analyzer
    odd._analyzer = analyzer

    first = even.compute_1d(axis="x", disk_cache=False)
    second = odd.compute_1d(axis="x", disk_cache=False)
    third = even.compute_1d(axis="x", disk_cache=False)

    assert first is result_even
    assert second is result_odd
    assert third is result_even
    assert len(analyzer.calls) == 2


def test_compute_1d_cache_separates_filter_configuration():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    unfiltered = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    filtered = DispersionResult1D(
        S=np.full((n_k, n_f), 2.0, dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr", name="run"))
    )

    class FakeAnalyzer:
        def __init__(self):
            self.calls = []

        def compute_dispersion_1d(self, **kwargs):
            self.calls.append(dict(kwargs))
            return filtered if kwargs.get("filters") else unfiltered

    analyzer = FakeAnalyzer()
    iface._analyzer = analyzer
    filters = {"pre": {"remove_average": True}}

    first = iface.compute_1d(axis="x", disk_cache=False)
    second = iface.compute_1d(axis="x", filters=filters, disk_cache=False)
    third = iface.compute_1d(axis="x", disk_cache=False)
    fourth = iface.compute_1d(axis="x", filters=filters, disk_cache=False)

    assert first is unfiltered
    assert second is filtered
    assert third is unfiltered
    assert fourth is filtered
    assert len(analyzer.calls) == 2


def test_compute_1d_cache_separates_fft_worker_configuration():
    from types import SimpleNamespace

    from mmpp.fft.dispersion import _fft_backend
    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr", name="run"))
    )

    class FakeAnalyzer:
        def __init__(self):
            self.calls = []

        def compute_dispersion_1d(self, **kwargs):
            self.calls.append(
                {
                    "kwargs": dict(kwargs),
                    "backend_info": _fft_backend.get_info(),
                }
            )
            return result

    analyzer = FakeAnalyzer()
    iface._analyzer = analyzer
    original = _fft_backend.get_info()

    try:
        first = iface.compute_1d(axis="x", workers=1, disk_cache=False)
        second = iface.compute_1d(axis="x", workers=2, disk_cache=False)
        third = iface.compute_1d(axis="x", workers=1, disk_cache=False)
    finally:
        _fft_backend.set_backend(original["backend"])
        _fft_backend.set_workers(original["workers"])

    assert first is result
    assert second is result
    assert third is result
    assert [call["backend_info"]["workers"] for call in analyzer.calls] == [1, 2]


def test_dispersion_modes_interactive_without_interface_returns_headless_viewer():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )

    viewer = result.modes.interactive(show=False, mode_components=["z"])

    assert viewer.result is result
    assert viewer.can_reconstruct_modes is False
    assert "source FFT context" in viewer.mode_unavailable_reason
    assert viewer.mode_components == ["z"]


def test_dispersion_modes_interactive_show_false_with_interface_stays_headless():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=np.ones((n_k, n_f), dtype=np.complex128),
    )

    class FailIfUsedInterface:
        def dispersion_modes(self, **kwargs):
            raise AssertionError("show=False must not launch the modes widget")

    object.__setattr__(result, "_interface", FailIfUsedInterface())

    viewer = result.modes.interactive(show=False, components=["z"], fmax=4.0)

    assert viewer.result is result
    assert viewer.can_reconstruct_modes is True
    assert viewer.mode_components == ["z"]
    assert viewer.options["fmax"] == 4.0


def test_dispersion_modes_interactive_show_true_forwards_visual_options_to_widget():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=np.ones((n_k, n_f), dtype=np.complex128),
    )

    class FakeModes:
        def __init__(self):
            self.plot_calls = []

        def plot_interactive(self, **kwargs):
            self.plot_calls.append(dict(kwargs))

    class FakeInterface:
        def __init__(self):
            self.modes = FakeModes()
            self.dispersion_calls = []

        def dispersion_modes(self, **kwargs):
            self.dispersion_calls.append(dict(kwargs))
            return self.modes

    iface = FakeInterface()
    object.__setattr__(result, "_interface", iface)

    out = result.modes.interactive(
        show=True,
        lattice_constant_nm=515.0,
        figsize=(9.0, 4.0),
        fmax=7.5,
        f_units="GHz",
        lognorm=False,
        save=True,
        force=True,
    )

    assert out is iface.modes
    assert iface.dispersion_calls == [
        {
            "result": result,
            "lattice_constant_nm": 515.0,
            "save": True,
            "force": True,
        }
    ]
    assert iface.modes.plot_calls == [
        {
            "result": result,
            "figsize": (9.0, 4.0),
            "lattice_constant_nm": 515.0,
            "fmax": 7.5,
            "f_units": "GHz",
            "lognorm": False,
        }
    ]


def test_interactive_dispersion_modes_plot_interactive_applies_startup_options(
    monkeypatch,
):
    from types import SimpleNamespace

    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes import interactive as interactive_module
    from mmpp.fft.dispersion.modes.interactive import InteractiveDispersionModes

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=np.ones((n_k, n_f), dtype=np.complex128),
    )
    modes = InteractiveDispersionModes(SimpleNamespace(compute_1d=lambda **_: result))

    monkeypatch.setattr(interactive_module, "_HAS_WIDGETS", True)
    monkeypatch.setattr(interactive_module, "_HAS_MATPLOTLIB", True)
    monkeypatch.setattr(modes, "_create_widgets", lambda: None)
    monkeypatch.setattr(modes, "_create_layout", lambda: object())
    monkeypatch.setattr(modes, "_initialize_figure", lambda: None)
    monkeypatch.setattr(modes, "_update_dispersion_plot", lambda: None)
    monkeypatch.setattr(
        interactive_module,
        "display",
        lambda *_args, **_kwargs: SimpleNamespace(update=lambda *_: None),
        raising=False,
    )

    modes.plot_interactive(
        result=result,
        lattice_constant_nm=515.0,
        fmax=7.5,
        f_units="GHz",
        lognorm=True,
    )

    assert modes._default_params["lattice_nm"] == 515.0
    assert modes._default_params["f_max_ghz"] == 7.5
    assert modes._default_params["live_log_enabled"] is True
    assert modes._default_params["live_log_method"] == "log1p"


def test_interactive_dispersion_modes_plot_interactive_uses_shared_viewer_normalization(
    monkeypatch,
):
    from types import SimpleNamespace

    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes import interactive as interactive_module
    from mmpp.fft.dispersion.modes.interactive import InteractiveDispersionModes

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=np.ones((n_k, n_f), dtype=np.complex128),
    )
    calls = []

    def fake_compute_1d(**kwargs):
        calls.append(dict(kwargs))
        return result

    modes = InteractiveDispersionModes(SimpleNamespace(compute_1d=fake_compute_1d))

    monkeypatch.setattr(interactive_module, "_HAS_WIDGETS", True)
    monkeypatch.setattr(interactive_module, "_HAS_MATPLOTLIB", True)
    monkeypatch.setattr(modes, "_create_widgets", lambda: None)
    monkeypatch.setattr(modes, "_create_layout", lambda: object())
    monkeypatch.setattr(modes, "_initialize_figure", lambda: None)
    monkeypatch.setattr(modes, "_update_dispersion_plot", lambda: None)
    monkeypatch.setattr(
        interactive_module,
        "display",
        lambda *_args, **_kwargs: SimpleNamespace(update=lambda *_: None),
        raising=False,
    )

    modes.plot_interactive(
        axis="x",
        components=["z", "+"],
        mode_type="phase",
        n_bz=3,
        auto_animate=True,
        analytical=True,
        model="kalinikos",
    )

    assert calls == [{"axis": "x"}]
    assert modes._mode_components == ["z", "+"]
    assert modes._spectrum_components == ["z", "+"]
    assert modes._interactive_viewer_options["auto_animate"] is True
    assert modes._interactive_viewer_options["mode_type"] == "phase"
    assert modes._analytical_options["enabled"] is True
    assert modes._analytical_options["model"] == "kalinikos"
    assert modes._default_params["mode_type"] == "phase"
    assert modes._default_params["n_bz_mask"] == 3


def test_interactive_dispersion_modes_exports_and_applies_shared_selection():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes.interactive import InteractiveDispersionModes

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    selected_k = float(k_axis[n_k // 2 + 1])
    selected_f = float(f_axis[n_f // 2 + 1])
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=np.ones((n_k, n_f), dtype=np.complex128),
    )
    source = InteractiveDispersionModes(SimpleNamespace(compute_1d=lambda **_: result))
    target = InteractiveDispersionModes(SimpleNamespace(compute_1d=lambda **_: result))
    source.result = result
    target.result = result
    source._mode_components = ["z", "+"]
    source._spectrum_components = ["perp"]
    source._selected_k = selected_k
    source._selected_f = selected_f

    exported = source.export_selection(source="legacy-test")
    target.apply_selection(exported)
    mode = target.mode_at_selection()

    assert exported["viewer"]["mode_components"] == ["z", "+"]
    assert exported["viewer"]["spectrum_components"] == ["perp"]
    assert exported["selection"]["source"] == "legacy-test"
    assert exported["selection"]["k_rad_per_m"] == selected_k
    assert exported["selection"]["f_hz"] == selected_f
    assert exported["mode_request"]["available"] is True
    assert target._selected_k == selected_k
    assert target._selected_f == selected_f
    assert mode.k_rad_um == pytest.approx(selected_k / 1e6)
    assert mode.f_ghz == pytest.approx(selected_f / 1e9)


def test_interactive_dispersion_modes_collects_and_applies_shared_preset():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes.interactive import InteractiveDispersionModes

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    selected_k = float(k_axis[n_k // 2 + 1])
    selected_f = float(f_axis[n_f // 2 + 1])
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=np.ones((n_k, n_f), dtype=np.complex128),
    )
    source = InteractiveDispersionModes(SimpleNamespace(compute_1d=lambda **_: result))
    target = InteractiveDispersionModes(SimpleNamespace(compute_1d=lambda **_: result))
    source.result = result
    target.result = result
    source._mode_components = ["z", "+"]
    source._spectrum_components = ["perp"]
    source._interactive_viewer_options = {"mode_type": "phase", "auto_animate": True}
    source._analytical_options = {"enabled": True, "model": "kalinikos"}
    source._default_params["mode_type"] = "phase"
    source._default_params["n_bz_mask"] = 4
    source._selected_k = selected_k
    source._selected_f = selected_f

    preset = source.collect_preset()
    returned = target.apply_preset(preset)

    assert returned is target
    assert preset["schema_version"] == "dispersion-interactive-preset/v1"
    assert preset["viewer"]["mode_components"] == ["z", "+"]
    assert preset["legacy_modes"]["params"]["mode_type"] == "phase"
    assert preset["selection"]["k_rad_per_m"] == selected_k
    assert target._mode_components == ["z", "+"]
    assert target._spectrum_components == ["perp"]
    assert target._interactive_viewer_options["auto_animate"] is True
    assert target._analytical_options["model"] == "kalinikos"
    assert target._default_params["mode_type"] == "phase"
    assert target._default_params["n_bz_mask"] == 4
    assert target._selected_k == selected_k
    assert target._selected_f == selected_f


def test_interactive_dispersion_modes_close_cleans_display_animation_and_figure(
    monkeypatch,
):
    from types import SimpleNamespace

    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes import interactive as interactive_module
    from mmpp.fft.dispersion.modes.interactive import InteractiveDispersionModes

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    modes = InteractiveDispersionModes(SimpleNamespace(compute_1d=lambda **_: result))

    display_updates = []
    stopped = []
    closed = []
    fig = object()

    modes._display_handle = SimpleNamespace(
        update=lambda value: display_updates.append(value)
    )
    modes._animation = SimpleNamespace(
        event_source=SimpleNamespace(stop=lambda: stopped.append(True))
    )
    modes._is_animating = True
    modes._fig = fig
    modes._ax_disp = object()
    modes._ax_mode = object()
    modes._colorbar_disp = object()
    modes._colorbar_mode = object()
    modes._mask_markers = [object()]
    monkeypatch.setattr(
        interactive_module,
        "plt",
        SimpleNamespace(close=lambda value: closed.append(value)),
        raising=False,
    )

    modes.close()

    assert stopped == [True]
    assert display_updates == [None]
    assert closed == [fig]
    assert modes._display_handle is None
    assert modes._animation is None
    assert modes._is_animating is False
    assert modes._fig is None
    assert modes._ax_disp is None
    assert modes._ax_mode is None
    assert modes._colorbar_disp is None
    assert modes._colorbar_mode is None
    assert modes._mask_markers == []


def test_dispersion_interface_plot_interactive_computes_then_returns_viewer():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr"))
    )
    calls = []

    def fake_compute_1d(**kwargs):
        calls.append(kwargs)
        return result

    iface.compute_1d = fake_compute_1d

    viewer = iface.plot.interactive(
        show=False,
        axis="x",
        component="perp",
        fmax=4.0,
        components=["z"],
        progress=False,
    )

    assert calls == [{"axis": "x", "component": "perp", "store_complex": False}]
    assert viewer.result is result
    assert viewer.show_requested is False
    assert viewer.options["fmax"] == 4.0
    assert viewer.mode_components == ["z"]


def test_dispersion_interface_plot_interactive_handles_tmax_and_external_cache():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr"))
    )
    iface._tmin = 20
    iface._tmax = 100
    calls = []

    def fake_compute_1d(**kwargs):
        calls.append((dict(kwargs), iface._tmin, iface._tmax, iface._cache_dir))
        return result

    iface.compute_1d = fake_compute_1d

    viewer = iface.plot.interactive(
        show=False,
        axis="x",
        tmin=200,
        tmax=800,
        cache="/tmp/mmpp-dispersion",
        save=True,
        fmax=4.0,
        progress=False,
    )

    assert viewer.result is result
    assert calls == [
        (
            {"axis": "x", "save": True, "store_complex": False, "disk_cache": True},
            200,
            800,
            "/tmp/mmpp-dispersion",
        )
    ]
    assert iface._tmin == 20
    assert iface._tmax == 100
    assert iface._cache_dir is None
    assert viewer.result._interface is not iface
    assert viewer.result._interface._tmin == 200
    assert viewer.result._interface._tmax == 800


def test_dispersion_interface_plot_interactive_defaults_to_all_timesteps():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr"))
    )
    calls = []
    events = []

    def fake_compute_1d(**kwargs):
        calls.append((dict(kwargs), iface._tmax))
        return result

    iface.compute_1d = fake_compute_1d

    viewer = iface.plot.interactive(
        show=False,
        axis="x",
        progress=True,
        progress_callback=events.append,
    )

    assert viewer.result is result
    assert calls == [
        (
            {
                "axis": "x",
                "store_complex": False,
                "progress_callback": calls[0][0]["progress_callback"],
            },
            None,
        )
    ]
    assert callable(calls[0][0]["progress_callback"])
    assert iface._tmax is None
    assert any("time_steps=all" in event["message"] for event in events)
    assert events[0]["time_steps"] is None
    assert events[0]["requested_tmax"] is None


def test_dispersion_interface_plot_interactive_accepts_analitical_option():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr"))
    )
    calls = []

    def fake_compute_1d(**kwargs):
        calls.append(dict(kwargs))
        return result

    iface.compute_1d = fake_compute_1d

    viewer = iface.plot.interactive(
        show=False,
        axis="x",
        analitical="DE",
        progress=False,
    )

    assert viewer.result is result
    assert viewer.options["analitical"] == "DE"
    assert calls == [{"axis": "x", "store_complex": False}]


def test_draw_analytical_overlay_renders_scatter_from_layer_params(monkeypatch):
    from types import SimpleNamespace

    from mmpp.fft.dispersion._interactive import rendering
    from mmpp.fft.dispersion._plotting import _analytics_overlay

    captured = {}

    def fake_extract_material_params(_result):
        captured["extract_called"] = True
        return {
            "B": 0.12,
            "Ms": 8.0e5,
            "Aex": 13.0e-12,
            "d": 120.0e-9,
            "Ku": 0.0,
            "Kc1": 0.0,
            "Kc2": 0.0,
            "phi_ani": 0.0,
            "g": 2.0,
        }

    def fake_compute_analytical_dispersion(
        k_range,
        **kwargs,
    ):
        captured.update({"k_range": k_range, "compute_kwargs": dict(kwargs)})
        return [
            (np.array([k_range[0], k_range[1]]), np.array([2.0, 3.0]), "DE"),
            (np.array([k_range[0], k_range[1]]), np.array([1.0, 1.5]), "DE"),
        ]

    def fake_scatter(x, y, **kwargs):
        captured.setdefault("scatter_calls", []).append(
            {"x": np.asarray(x), "y": np.asarray(y), "kwargs": dict(kwargs)}
        )

    class FakeAxis:
        def get_xlim(self):
            return (0.0, 1.0)

        def scatter(self, x, y, **kwargs):
            fake_scatter(x, y, **kwargs)

        def legend(self, *args, **kwargs):
            captured["legend_called"] = (args, kwargs)

    class FakeExplorer:
        def __init__(self):
            self.options = {
                "analitical": "DE",
                "analytical_model": "kalinikos",
                "analytical_n_modes": 2,
            }
            self.result = SimpleNamespace(_interface=SimpleNamespace())

    monkeypatch.setattr(
        _analytics_overlay,
        "extract_material_params",
        fake_extract_material_params,
    )
    monkeypatch.setattr(
        _analytics_overlay,
        "compute_analytical_dispersion",
        fake_compute_analytical_dispersion,
    )

    explorer = FakeExplorer()
    rendering._draw_analytical_overlay(explorer, FakeAxis(), "rad_um")

    assert captured["extract_called"] is True
    assert captured["k_range"] == (0.0, 1.0e6)
    assert captured["compute_kwargs"]["sw_config"] == "DE"
    assert captured["compute_kwargs"]["model"] == "kalinikos"
    assert captured["compute_kwargs"]["n_modes"] == 2
    assert captured["compute_kwargs"]["B"] == 0.12
    assert captured["compute_kwargs"]["Ms"] == 8.0e5
    assert len(captured["scatter_calls"]) == 2
    assert captured["scatter_calls"][0]["kwargs"]["label"] == "DE"
    assert captured["scatter_calls"][1]["kwargs"]["label"] == "DE (n=1)"
    assert "legend_called" in captured


def test_extract_material_params_reads_b_ext_vector_from_mx3_sidecar(tmp_path):
    from types import SimpleNamespace

    from mmpp.fft.dispersion._plotting._analytics_overlay import (
        extract_material_params,
    )
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    zarr_path = tmp_path / "disp_2.zarr"
    root = zarr.open(str(zarr_path), mode="w")
    root.attrs.update(
        {
            "B_ext": "0xc000241a10",
            "Msat": "157600",
            "Aex": "3.7e-12",
            "Tz": 70e-9,
        }
    )
    (tmp_path / "disp_2.mx3").write_text(
        "\n".join(
            [
                "bex := 0.008",
                "B_ext = vector(0, bex, 0)",
                "B_ext.add(thermalNoise, 1e-4)",
            ]
        ),
        encoding="utf-8",
    )

    result = DispersionResult1D(
        S=np.ones((4, 4), dtype=np.float32),
        k_axis=np.arange(4, dtype=float),
        f_axis=np.arange(4, dtype=float),
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0),
        dt=1.0,
        dx=1.0,
    )
    object.__setattr__(
        result,
        "_interface",
        SimpleNamespace(
            parent_fft=SimpleNamespace(
                job_result=SimpleNamespace(path=zarr_path, name="disp_2")
            )
        ),
    )

    params = extract_material_params(result)

    assert params["B"] == pytest.approx(0.008)
    assert params["B_vector"] == (0.0, 0.008, 0.0)
    assert params["phi"] == pytest.approx(np.pi / 2)
    assert params["Ms"] == pytest.approx(157600)
    assert params["Aex"] == pytest.approx(3.7e-12)
    assert params["d"] == pytest.approx(70e-9)


def test_dispersion_interface_plot_interactive_defaults_to_lightweight_compute():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr"))
    )
    calls = []

    def fake_compute_1d(**kwargs):
        calls.append(dict(kwargs))
        return result

    iface.compute_1d = fake_compute_1d

    iface.plot.interactive(show=False, axis="x", progress=False)
    iface.plot.interactive(show=False, axis="x", store_complex=True, progress=False)

    assert calls == [
        {"axis": "x", "store_complex": False},
        {"axis": "x", "store_complex": True},
    ]


def test_dispersion_interactive_analysis_aliases_plot_interactive():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr"))
    )
    calls = []

    def fake_compute_1d(**kwargs):
        calls.append(dict(kwargs))
        return result

    iface.compute_1d = fake_compute_1d

    viewer = iface.interactive_analysis(
        show=False,
        axis="x",
        fmax=4.0,
        progress=False,
    )

    assert viewer.result is result
    assert viewer.show_requested is False
    assert viewer.options["fmax"] == 4.0
    assert calls == [{"axis": "x", "store_complex": False}]


def test_compute_1d_normalizes_default_store_complex_in_cache_key():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=np.ones((n_k, n_f), dtype=np.complex128),
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr", name="run"))
    )

    class FakeAnalyzer:
        def __init__(self):
            self.calls = []

        def compute_dispersion_1d(self, **kwargs):
            self.calls.append(dict(kwargs))
            return result

    analyzer = FakeAnalyzer()
    iface._analyzer = analyzer

    first = iface.compute_1d(axis="x", disk_cache=False)
    second = iface.compute_1d(axis="x", store_complex=True, disk_cache=False)

    assert first is second
    assert analyzer.calls == [
        {
            "axis": "x",
            "component": None,
            "filters": None,
            "flipx": True,
            "store_complex": True,
            "scaling": "raw_power",
        }
    ]


def test_spin_wave_analyzer_uses_effective_spacing_after_slice_stride(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    zarr_path = tmp_path / "dispersion_stride.zarr"
    data = np.zeros((12, 1, 9, 8, 3), dtype=np.float32)
    root = zarr.open(str(zarr_path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root.attrs["t_sampl"] = 1e-12
    root.attrs["dx"] = 1e-9
    root.attrs["dy"] = 2e-9

    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(),
        tmax=None,
        slice_info=(
            slice(None, None, 2),
            slice(None),
            slice(None, None, 3),
            slice(None, None, 4),
            slice(None),
        ),
    )

    assert analyzer.M_data.shape == (6, 1, 3, 2, 3)
    assert np.isclose(analyzer.dt, 2e-12)
    assert np.isclose(analyzer.grid_spacings["dx"], 4e-9)
    assert np.isclose(analyzer.grid_spacings["dy"], 6e-9)
    assert np.isclose(analyzer.config.dt, 2e-12)
    assert np.isclose(analyzer.config.dx, 4e-9)
    assert np.isclose(analyzer.config.dy, 6e-9)


def test_spin_wave_analyzer_infers_dt_from_uniform_time_axis(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    zarr_path = tmp_path / "dispersion_uniform_time_axis.zarr"
    data = np.zeros((4, 1, 1, 4, 3), dtype=np.float32)
    root = zarr.open(str(zarr_path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root.create_dataset("t", data=np.arange(4, dtype=float) * 2e-12)
    root.attrs["dx"] = 1e-9
    root.attrs["dy"] = 1e-9

    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(dt=1e-12),
        tmax=None,
    )

    assert np.isclose(analyzer.dt, 2e-12)
    assert np.allclose(analyzer.time_axis, np.arange(4, dtype=float) * 2e-12)


def test_spin_wave_analyzer_rejects_nonuniform_time_axis(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    zarr_path = tmp_path / "dispersion_nonuniform_time_axis.zarr"
    data = np.zeros((4, 1, 1, 4, 3), dtype=np.float32)
    root = zarr.open(str(zarr_path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root["m"].attrs["t"] = [0.0, 1.2e-12, 2.8e-12, 4.5e-12]
    root.attrs["t_sampl"] = 1e-12
    root.attrs["dx"] = 1e-9
    root.attrs["dy"] = 1e-9

    with pytest.raises(ValueError, match="Non-uniform time axis.*resample"):
        SpinWaveAnalyzer(
            zarr_path,
            config=DispersionConfig(dt=9e-12),
            tmax=None,
        )


def test_spin_wave_analyzer_infers_spacing_from_uniform_spatial_axes(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    zarr_path = tmp_path / "dispersion_uniform_spatial_axes.zarr"
    data = np.zeros((4, 1, 3, 4, 3), dtype=np.float32)
    root = zarr.open(str(zarr_path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root.create_dataset("x", data=np.arange(4, dtype=float) * 2e-9)
    root.create_dataset("y", data=np.arange(3, dtype=float) * 3e-9)
    root.attrs["t_sampl"] = 1e-12

    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(),
        tmax=None,
    )

    assert np.isclose(analyzer.grid_spacings["dx"], 2e-9)
    assert np.isclose(analyzer.grid_spacings["dy"], 3e-9)
    assert np.isclose(analyzer.config.dx, 2e-9)
    assert np.isclose(analyzer.config.dy, 3e-9)


def test_spin_wave_analyzer_rejects_nonmonotonic_spatial_axis(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    zarr_path = tmp_path / "dispersion_nonmonotonic_spatial_axis.zarr"
    data = np.zeros((4, 1, 1, 4, 3), dtype=np.float32)
    root = zarr.open(str(zarr_path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root.create_dataset("x", data=np.array([0.0, 2e-9, 1e-9, 3e-9]))
    root.attrs["t_sampl"] = 1e-12
    root.attrs["dy"] = 1e-9

    with pytest.raises(ValueError, match="Spatial axis 'x' must be strictly monotonic"):
        SpinWaveAnalyzer(
            zarr_path,
            config=DispersionConfig(),
            tmax=None,
        )


def test_branch_quality_coverage_uses_reference_k_axis():
    from mmpp.fft.dispersion._branch_linker import _branch_quality

    full_k = np.linspace(-10.0, 10.0, 101)
    full_f = np.linspace(1e9, 2e9, full_k.size)
    partial_k = full_k[45:56]
    partial_f = np.linspace(1.45e9, 1.55e9, partial_k.size)

    full_quality = _branch_quality(
        full_k,
        full_f,
        np.ones_like(full_k),
        reference_k_axis=full_k,
    )
    partial_quality = _branch_quality(
        partial_k,
        partial_f,
        np.ones_like(partial_k),
        reference_k_axis=full_k,
    )

    assert partial_quality < full_quality
    assert full_quality - partial_quality > 0.2


def test_dispersion_interactive_preset_roundtrips_analytical_overlay_state():
    import types

    from mmpp.fft.dispersion._interactive.presets import (
        apply_preset_state,
        collect_preset_state,
    )
    from mmpp.fft.dispersion._interactive.state import DispersionExplorerState

    analytical = {
        "enabled": True,
        "model": "kalinikos",
        "sw_config": "BV",
        "n_modes": 3,
        "k_points": 640,
    }
    explorer = types.SimpleNamespace(state=DispersionExplorerState())
    explorer.state.analytical = dict(analytical)

    payload = collect_preset_state(explorer)

    assert payload["analytical"] == analytical

    restored = types.SimpleNamespace(state=DispersionExplorerState())
    apply_preset_state(restored, payload)

    assert restored.state.analytical == analytical


def test_dispersion_interactive_display_change_syncs_analytical_overlay_options():
    import types

    from mmpp.fft.dispersion._interactive.callbacks import on_display_change
    from mmpp.fft.dispersion._interactive.state import DispersionExplorerState

    class Control:
        def __init__(self, value):
            self.value = value

    explorer = types.SimpleNamespace(
        state=DispersionExplorerState(fmin_ghz=0.0, fmax_ghz=25.0),
        options={},
        figure=None,
        axes=None,
        controls={
            "fmin": Control(1.0),
            "fmax": Control(18.0),
            "source": Control("display"),
            "kscale": Control("rad_um"),
            "cmap": Control("magma"),
            "positive": Control(True),
            "lognorm": Control(False),
            "grid": Control(True),
            "selection": Control(True),
            "notes": Control(False),
            "analytical_enabled": Control(True),
            "analytical_model": Control("kalinikos"),
            "analytical_sw_config": Control("DE"),
            "analytical_n_modes": Control(2),
            "analytical_k_points": Control(512),
            "analytical_B": Control(0.08),
            "analytical_Ms": Control(800e3),
            "analytical_Aex": Control(13e-12),
            "analytical_d": Control(30e-9),
            "analytical_phi": Control(0.15),
            "analytical_D": Control(1.2e-3),
        },
        draw=lambda: None,
        redraw=lambda: None,
    )

    on_display_change(explorer)

    assert explorer.state.analytical == {
        "enabled": True,
        "model": "kalinikos",
        "sw_config": "DE",
        "n_modes": 2,
        "k_points": 512,
        "B": 0.08,
        "Ms": 800e3,
        "Aex": 13e-12,
        "d": 30e-9,
        "phi": 0.15,
        "D": 1.2e-3,
    }
    assert explorer.options["analytical"] == "DE"
    assert explorer.options["analytical_model"] == "kalinikos"
    assert explorer.options["analytical_n_modes"] == 2
    assert explorer.options["analytical_k_points"] == 512
    assert explorer.options["B"] == 0.08
    assert explorer.options["Ms"] == 800e3
    assert explorer.options["Aex"] == 13e-12
    assert explorer.options["d"] == 30e-9
    assert explorer.options["analytical_phi"] == 0.15
    assert explorer.options["analytical_D"] == 1.2e-3


def test_dispersion_interactive_display_change_auto_renders_by_default(monkeypatch):
    """Auto-render is enabled by default so toolbar changes update the plot."""
    import types

    from mmpp.fft.dispersion._interactive import callbacks
    from mmpp.fft.dispersion._interactive.callbacks import on_display_change
    from mmpp.fft.dispersion._interactive.state import DispersionExplorerState

    class Control:
        def __init__(self, value):
            self.value = value

    calls = []
    monkeypatch.setattr(
        callbacks,
        "draw_dispersion_panel",
        lambda explorer: calls.append("draw"),
    )
    monkeypatch.setattr(
        callbacks,
        "refresh_output_widget",
        lambda explorer: calls.append("refresh"),
    )
    monkeypatch.setattr(callbacks, "set_status", lambda *args, **kwargs: None)

    explorer = types.SimpleNamespace(
        state=DispersionExplorerState(fmin_ghz=0.0, fmax_ghz=25.0),
        options={},
        controls={
            "fmin": Control(1.0),
            "fmax": Control(18.0),
            "source": Control("display"),
            "kscale": Control("rad_um"),
            "cmap": Control("magma"),
            "positive": Control(True),
            "lognorm": Control(False),
            "grid": Control(True),
            "selection": Control(True),
            "notes": Control(False),
        },
    )

    on_display_change(explorer)

    assert calls == ["draw", "refresh"]


def test_dispersion_interactive_display_change_can_auto_render(monkeypatch):
    import types

    from mmpp.fft.dispersion._interactive import callbacks
    from mmpp.fft.dispersion._interactive.callbacks import on_display_change
    from mmpp.fft.dispersion._interactive.state import DispersionExplorerState

    class Control:
        def __init__(self, value):
            self.value = value

    calls = []
    monkeypatch.setattr(
        callbacks,
        "draw_dispersion_panel",
        lambda explorer: calls.append("draw"),
    )
    monkeypatch.setattr(
        callbacks,
        "refresh_output_widget",
        lambda explorer: calls.append("refresh"),
    )
    monkeypatch.setattr(callbacks, "set_status", lambda *args, **kwargs: None)

    explorer = types.SimpleNamespace(
        state=DispersionExplorerState(fmin_ghz=0.0, fmax_ghz=25.0),
        options={"auto_render": True},
        controls={
            "fmin": Control(1.0),
            "fmax": Control(18.0),
            "source": Control("display"),
            "kscale": Control("rad_um"),
            "cmap": Control("magma"),
            "positive": Control(True),
            "lognorm": Control(False),
            "grid": Control(True),
            "selection": Control(True),
            "notes": Control(False),
        },
    )

    on_display_change(explorer)

    assert calls == ["draw", "refresh"]


def test_dispersion_interactive_frequency_window_defaults_to_ghz_limits():
    from mmpp.fft.dispersion._interactive.frequency import (
        normalize_frequency_window_ghz,
    )

    fmin, fmax = normalize_frequency_window_ghz(
        {"fmax": 25, "f_units": "GHz"},
        np.array([0.0, 25e9, 500e9]),
    )

    assert fmin == 0.0
    assert fmax == 25.0


def test_dispersion_interactive_frequency_window_accepts_hz_limits():
    from mmpp.fft.dispersion._interactive.frequency import (
        normalize_frequency_window_ghz,
    )

    fmin, fmax = normalize_frequency_window_ghz(
        {"fmin": 5e9, "fmax": 25e9, "f_units": "Hz"},
        np.array([0.0, 25e9, 500e9]),
    )

    assert fmin == 5.0
    assert fmax == 25.0


def test_dispersion_interactive_widget_state_keeps_requested_fmax():
    import types

    from mmpp.fft.dispersion._interactive.widget import DispersionHeatmapWidget

    explorer = DispersionHeatmapWidget(
        types.SimpleNamespace(f_axis=np.array([0.0, 25e9, 500e9]), notes=[]),
        {"fmax": 25, "f_units": "GHz"},
    )

    assert explorer.state.fmin_ghz == 0.0
    assert explorer.state.fmax_ghz == 25.0


def test_dispersion_interactive_deferred_toolbar_does_not_create_matplotlib(
    monkeypatch,
):
    import builtins
    import sys
    import types

    from mmpp.fft.dispersion._interactive.widget import DispersionHeatmapWidget

    class FakeWidget:
        def __init__(self, *children, **kwargs):
            self.children = tuple(children)
            self.kwargs = kwargs
            self.value = kwargs.get("value")
            self.description = kwargs.get("description", "")
            self.options = kwargs.get("options", [])
            self._observers = []
            self._clicks = []

        def observe(self, callback, names=None):
            self._observers.append((callback, names))

        def on_click(self, callback):
            self._clicks.append(callback)

    class FakeTab(FakeWidget):
        def __init__(self, *children, **kwargs):
            super().__init__(*children, **kwargs)
            self.titles = {}

        def set_title(self, index, title):
            self.titles[index] = title

    fake_widgets = types.SimpleNamespace(
        FloatText=FakeWidget,
        Dropdown=FakeWidget,
        Checkbox=FakeWidget,
        HTML=FakeWidget,
        Output=FakeWidget,
        Text=FakeWidget,
        Button=FakeWidget,
        VBox=lambda children=(), **kwargs: FakeWidget(*children, **kwargs),
        HBox=lambda children=(), **kwargs: FakeWidget(*children, **kwargs),
        Tab=FakeTab,
    )
    monkeypatch.setitem(sys.modules, "ipywidgets", fake_widgets)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise AssertionError(f"unexpected matplotlib import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    explorer = DispersionHeatmapWidget(
        types.SimpleNamespace(f_axis=np.array([0.0, 25e9, 500e9]), notes=[]),
        {"fmax": 25, "f_units": "GHz"},
    )

    widget = explorer.build(
        lambda _obj: None,
        toolbar=True,
        defer_initial_render=True,
    )

    assert widget is explorer.widget
    assert explorer.figure is None
    assert explorer.axes is None
    assert explorer.diagnostics()["backend"] == "not-created"
    assert "render_dispersion" in explorer.controls


def test_dispersion_interactive_toolbar_exposes_analytical_overlay_controls(
    monkeypatch, tmp_path
):
    import types

    from mmpp.fft.dispersion._interactive import widgets as toolbar_widgets
    from mmpp.fft.dispersion._interactive.state import DispersionExplorerState

    class FakeWidget:
        def __init__(self, *children, **kwargs):
            self.children = tuple(children)
            self.kwargs = kwargs
            self.value = kwargs.get("value")
            self.description = kwargs.get("description", "")
            self.options = kwargs.get("options", [])
            self._observers = []
            self._clicks = []

        def observe(self, callback, names=None):
            self._observers.append((callback, names))

        def on_click(self, callback):
            self._clicks.append(callback)

    class FakeTab(FakeWidget):
        def __init__(self, *children, **kwargs):
            super().__init__(*children, **kwargs)
            self.titles = {}

        def set_title(self, index, title):
            self.titles[index] = title

    fake_widgets = types.SimpleNamespace(
        FloatText=FakeWidget,
        Dropdown=FakeWidget,
        Checkbox=FakeWidget,
        HTML=FakeWidget,
        Output=FakeWidget,
        Text=FakeWidget,
        Button=FakeWidget,
        VBox=lambda children=(), **kwargs: FakeWidget(*children, **kwargs),
        HBox=lambda children=(), **kwargs: FakeWidget(*children, **kwargs),
        Tab=FakeTab,
    )
    monkeypatch.setattr(toolbar_widgets, "draw_dispersion_panel", lambda explorer: None)
    monkeypatch.setattr(toolbar_widgets, "refresh_output_widget", lambda explorer: None)
    monkeypatch.setattr(toolbar_widgets, "set_status", lambda *args, **kwargs: None)

    explorer = types.SimpleNamespace(
        result=types.SimpleNamespace(f_axis=[]),
        options={},
        state=DispersionExplorerState(
            analytical={
                "enabled": True,
                "model": "kalinikos",
                "sw_config": "BV",
                "n_modes": 2,
                "k_points": 512,
                "B": 0.09,
                "Ms": 900e3,
                "Aex": 12e-12,
                "d": 25e-9,
                "phi": 0.25,
                "D": 0.8e-3,
            }
        ),
        controls={},
        _presets_dir=tmp_path,
        collect_preset=lambda: {
            "schema_version": "test",
            "numpy_value": np.float32(1.5),
            "unsafe": "<script>",
        },
    )

    toolbar_widgets.build_toolbar(explorer, fake_widgets)

    assert callable(explorer.refresh_auxiliary_panels)
    assert explorer.controls["analytical_enabled"].value is True
    assert explorer.controls["analytical_sw_config"].value == "BV"
    assert explorer.controls["analytical_model"].value == "kalinikos"
    assert explorer.controls["analytical_n_modes"].value == 2.0
    assert explorer.controls["analytical_k_points"].value == 512.0
    assert explorer.controls["analytical_B"].value == "0.09"
    assert explorer.controls["analytical_Ms"].value == "900000.0"
    assert explorer.controls["analytical_Aex"].value == "1.2e-11"
    assert explorer.controls["analytical_d"].value == "2.5e-08"
    assert explorer.controls["analytical_phi"].value == "0.25"
    assert explorer.controls["analytical_D"].value == "0.0008"
    assert explorer.controls["tabs"].titles[2] == "Analytical"
    assert explorer.controls["mode_type"].value == "abs"
    assert "mode_show_dispersion" in explorer.controls
    assert explorer.controls["render_dispersion"].description == (
        "Render / refresh dispersion"
    )
    assert (
        "Render / refresh dispersion" in explorer.controls["output_placeholder"].value
    )
    display_tab = explorer.controls["tabs"].kwargs["children"][0]
    assert display_tab.children[0] is explorer.controls["render_dispersion"]
    assert explorer.controls["export_refresh"].description == "Refresh export snapshot"
    assert explorer.controls["analysis_refresh"].description == (
        "Refresh analysis summary"
    )
    assert explorer.controls["tabs"].titles[3] == "Modes"
    assert explorer.controls["tabs"].titles[4] == "Analysis"
    assert explorer.controls["tabs"].titles[5] == "Export"

    explorer.state.selected_k = np.float64(2.5e6)
    explorer.state.selected_f = np.float64(3.25e9)
    explorer.state.selected_power = np.float32(4.5)
    toolbar_widgets._export_snapshot(explorer)

    snapshot_html = explorer.controls["export_snapshot"].value
    assert "&lt;script&gt;" in snapshot_html
    assert '"numpy_value": 1.5' in snapshot_html
    assert '"power": 4.5' in snapshot_html

    toolbar_widgets._refresh_analysis_summary(explorer)
    analysis_html = explorer.controls["analysis_summary"].value
    assert "display_window_ghz" in analysis_html
    assert "has_complex_modes" in analysis_html


def test_dispersion_interactive_mode_extract_renders_mode_in_main_output(monkeypatch):
    import types

    from mmpp.fft.dispersion._interactive import callbacks
    from mmpp.fft.dispersion._interactive.state import DispersionExplorerState

    class Control:
        def __init__(self, value=""):
            self.value = value

    class FakeOutput:
        def __init__(self):
            self.cleared = False
            self.items = []

        def clear_output(self, wait=False):
            self.cleared = True

        def append_display_data(self, item):
            self.items.append(item)

    class FakePlot:
        def __init__(self):
            self.calls = []

        def imshow(self, **kwargs):
            self.calls.append(kwargs)
            return "mode-figure", "mode-axis"

    class FakeMode:
        def __init__(self):
            self.k_rad_um = 2.5
            self.f_ghz = 3.25
            self.z_layer = 1
            self.component = "z"
            self.plot = FakePlot()

    class FakeModes:
        def __init__(self):
            self.calls = []
            self.mode = FakeMode()

        def at(self, **kwargs):
            self.calls.append(kwargs)
            return self.mode

    output = FakeOutput()
    modes = FakeModes()
    explorer = types.SimpleNamespace(
        state=DispersionExplorerState(selected_k=2.5e6, selected_f=3.25e9),
        result=types.SimpleNamespace(
            component="perp",
            S_complex=np.ones((4, 4), dtype=np.complex128),
            modes=modes,
        ),
        controls={
            "mode_z_layer": Control(1),
            "mode_component": Control("z"),
            "mode_type": Control("phase"),
            "mode_info": Control(""),
            "output": output,
        },
        last_mode=None,
    )
    monkeypatch.setattr(callbacks, "set_status", lambda *args, **kwargs: None)

    callbacks.on_mode_extract(explorer)

    assert modes.calls == [
        {"k_rad_um": 2.5, "f_ghz": 3.25, "z_layer": 1, "component": "z"}
    ]
    assert explorer.last_mode is modes.mode
    assert output.cleared is True
    assert output.items == ["mode-figure"]
    assert modes.mode.plot.calls[0]["mode_type"] == "phase"
    assert modes.mode.plot.calls[0]["cmap"] == "hsv"


def test_dispersion_interactive_export_selection_uses_widget_state_for_mode_request():
    import types

    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=np.ones((n_k, n_f), dtype=np.complex128),
    )
    viewer = result.plot.interactive(show=False, modes=True)
    viewer._widget_engine = types.SimpleNamespace(
        state=types.SimpleNamespace(
            selected_k=np.float64(2.5e6),
            selected_f=np.float64(3.25e9),
            selected_power=np.float32(4.5),
        )
    )

    exported = viewer.export_selection()

    assert exported["selection"] == {
        "source": "widget",
        "k_rad_per_m": 2.5e6,
        "k_rad_um": 2.5,
        "f_hz": 3.25e9,
        "f_ghz": 3.25,
        "power": 4.5,
    }
    assert exported["mode_request"] == {
        "available": True,
        "k_rad_um": 2.5,
        "f_ghz": 3.25,
        "z_layer": 0,
        "component": "perp",
        "reason": "",
    }
    json.dumps(exported)


def test_dispersion_interactive_mode_at_selection_extracts_selected_mode():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    idx_k = n_k // 2 + 1
    idx_f = n_f // 2 + 1
    S_complex = np.zeros((n_k, n_f), dtype=np.complex128)
    S_complex[idx_k, idx_f] = 5.0 + 1.0j
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
    viewer = result.plot.interactive(show=False, modes=True)

    mode = viewer.mode_at_selection(
        k_rad_um=float(k_axis[idx_k]) / 1e6,
        f_ghz=float(f_axis[idx_f]) / 1e9,
        z_layer=2,
        component="z",
    )

    assert mode.mode_data == S_complex[idx_k, idx_f]
    assert mode.z_layer == 2
    assert mode.component == "z"


def test_dispersion_interactive_mode_request_reports_unavailable_without_complex_data():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    viewer = result.plot.interactive(show=False, modes=True)

    exported = viewer.export_selection(k_rad_um=1.0, f_ghz=2.0)

    assert exported["mode_request"]["available"] is False
    assert "S_complex" in exported["mode_request"]["reason"]
    with pytest.raises(ValueError, match="S_complex"):
        viewer.mode_at_selection(k_rad_um=1.0, f_ghz=2.0)


def test_dispersion_interactive_modes_panel_extracts_current_selection(
    monkeypatch, tmp_path
):
    import types

    from mmpp.fft.dispersion._interactive import widgets as toolbar_widgets
    from mmpp.fft.dispersion._interactive.callbacks import on_mode_extract
    from mmpp.fft.dispersion._interactive.state import DispersionExplorerState

    class FakeWidget:
        def __init__(self, *children, **kwargs):
            self.children = tuple(children)
            self.kwargs = kwargs
            self.value = kwargs.get("value")
            self.description = kwargs.get("description", "")
            self.options = kwargs.get("options", [])
            self._observers = []
            self._clicks = []

        def observe(self, callback, names=None):
            self._observers.append((callback, names))

        def on_click(self, callback):
            self._clicks.append(callback)

    class FakeTab(FakeWidget):
        def __init__(self, *children, **kwargs):
            super().__init__(*children, **kwargs)
            self.titles = {}

        def set_title(self, index, title):
            self.titles[index] = title

    fake_widgets = types.SimpleNamespace(
        FloatText=FakeWidget,
        Dropdown=FakeWidget,
        Checkbox=FakeWidget,
        HTML=FakeWidget,
        Output=FakeWidget,
        Text=FakeWidget,
        Button=FakeWidget,
        VBox=lambda children=(), **kwargs: FakeWidget(*children, **kwargs),
        HBox=lambda children=(), **kwargs: FakeWidget(*children, **kwargs),
        Tab=FakeTab,
    )
    monkeypatch.setattr(toolbar_widgets, "draw_dispersion_panel", lambda explorer: None)
    monkeypatch.setattr(toolbar_widgets, "refresh_output_widget", lambda explorer: None)
    monkeypatch.setattr(toolbar_widgets, "set_status", lambda *args, **kwargs: None)

    extracted = []

    class FakeResult:
        f_axis = []
        component = "perp"
        S_complex = object()

        class modes:
            @staticmethod
            def at(k_rad_um, f_ghz, *, z_layer=0, component=None):
                extracted.append(
                    {
                        "k_rad_um": k_rad_um,
                        "f_ghz": f_ghz,
                        "z_layer": z_layer,
                        "component": component,
                    }
                )
                return types.SimpleNamespace(
                    k_rad_um=k_rad_um,
                    f_ghz=f_ghz,
                    z_layer=z_layer,
                    component=component,
                    mode_data=[1, 2, 3],
                )

    explorer = types.SimpleNamespace(
        result=FakeResult(),
        options={"can_reconstruct_modes": True, "mode_components": ["z", "+"]},
        state=DispersionExplorerState(),
        controls={},
        _presets_dir=tmp_path,
        last_mode=None,
    )
    explorer.state.selected_k = 2.25e6
    explorer.state.selected_f = 4.5e9

    toolbar_widgets.build_toolbar(explorer, fake_widgets)
    explorer.controls["mode_component"].value = "z"
    explorer.controls["mode_z_layer"].value = 2

    on_mode_extract(explorer)

    assert explorer.controls["tabs"].titles[3] == "Modes"
    assert extracted == [
        {"k_rad_um": 2.25, "f_ghz": 4.5, "z_layer": 2, "component": "z"}
    ]
    assert explorer.last_mode.component == "z"
    assert "k=2.25 rad/um" in explorer.controls["mode_info"].value
    assert "f=4.5 GHz" in explorer.controls["mode_info"].value


def test_legacy_dispersion_modes_attaches_interface_to_reused_result(monkeypatch):
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
        S_complex=np.ones((n_k, n_f), dtype=np.complex128),
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr"))
    )

    class FakeInteractiveDispersionModes:
        def __init__(self, interface):
            self.interface = interface
            self._default_params = {}
            self.result = None

    import mmpp.fft.dispersion.modes as modes_module

    monkeypatch.setattr(
        modes_module,
        "InteractiveDispersionModes",
        FakeInteractiveDispersionModes,
    )

    modes = iface.dispersion_modes(result=result, lattice_constant_nm=512.0)

    assert modes.result is result
    assert result._interface is iface
    assert modes._default_params["lattice_nm"] == 512.0


def test_dispersion_plot_interactive_reports_progress_without_polluting_compute_kwargs():
    from types import SimpleNamespace

    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr")),
        dataset_name="m",
        slice_info=(slice(None, 100, None), Ellipsis),
    )
    calls = []
    events = []

    def fake_compute_1d(**kwargs):
        calls.append(dict(kwargs))
        callback = kwargs.pop("progress_callback")
        callback({"stage": "compute", "message": "fake compute started"})
        callback({"stage": "compute", "message": "fake compute finished"})
        return result

    iface.compute_1d = fake_compute_1d

    viewer = iface.plot.interactive(
        show=False,
        axis="x",
        fmax=4.0,
        progress=True,
        progress_callback=events.append,
    )

    assert viewer.result is result
    assert calls == [
        {
            "axis": "x",
            "store_complex": False,
            "progress_callback": calls[0]["progress_callback"],
        }
    ]
    assert callable(calls[0]["progress_callback"])
    stages = [event["stage"] for event in events]
    assert "prepare" in stages
    assert "compute" in stages
    assert "viewer" in stages
    assert any("slice" in event["message"] for event in events)
    assert any("time_steps=100" in event["message"] for event in events)
    assert events[0]["time_steps"] == 100


def test_dispersion_plot_interactive_finishes_progress_before_notebook_show(
    monkeypatch,
):
    from types import SimpleNamespace

    from mmpp.fft.dispersion._interactive_viewer import DispersionInteractiveViewer
    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    dx, dt = 5e-9, 2e-9
    k_axis, f_axis = _make_axes(n_k, n_f, dx=dx, dt=dt)
    result = DispersionResult1D(
        S=np.ones((n_k, n_f), dtype=np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=dt, dx=dx),
        dt=dt,
        dx=dx,
    )
    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path="/tmp/run.zarr"))
    )
    events = []
    show_positions = []

    def fake_compute_1d(**kwargs):
        kwargs.pop("progress_callback")({"stage": "compute", "message": "done"})
        return result

    def fake_show(self, *, toolbar="auto"):
        show_positions.append([event["stage"] for event in events])
        self.show_requested = True
        self.options["toolbar"] = toolbar
        return self

    iface.compute_1d = fake_compute_1d
    monkeypatch.setattr(DispersionInteractiveViewer, "show", fake_show)

    viewer = iface.plot.interactive(
        show=True,
        axis="x",
        progress=True,
        progress_callback=events.append,
    )

    assert viewer.result is result
    assert viewer.options["toolbar"] == "auto"
    assert show_positions
    assert "done" in show_positions[0]
    assert events[-1]["stage"] == "done"


def test_dispersion_progress_bar_does_not_advance_past_declared_stage_total():
    from mmpp.fft.dispersion.interface import _DispersionProgressReporter

    class FakeBar:
        def __init__(self):
            self.updates = []
            self.descriptions = []
            self.postfixes = []
            self.closed = False

        def set_description(self, value):
            self.descriptions.append(value)

        def set_postfix_str(self, value):
            self.postfixes.append(value)

        def update(self, value):
            self.updates.append(value)

        def close(self):
            self.closed = True

    bar = FakeBar()
    reporter = _DispersionProgressReporter(enabled=False, total=2)
    reporter._visible = True
    reporter._bar = bar
    reporter._use_print = False

    reporter.emit(stage="prepare", message="one")
    reporter.emit(stage="compute", message="two")
    reporter.emit(stage="done", message="three")
    reporter.close()

    assert reporter.count == 2
    assert bar.updates == [1, 1]
    assert bar.closed is True


def test_fft_view_time_slice_stride_changes_effective_dt():
    from types import SimpleNamespace

    from mmpp.fft._compute_loading import resolve_dt_from_metadata

    class Logger:
        debug = info = warning = lambda *args, **kwargs: None

    dt = resolve_dt_from_metadata(
        data_set=SimpleNamespace(attrs={"t_sampl": 2e-12}),
        job=SimpleNamespace(attrs={}),
        logger=Logger(),
        slice_info=(slice(None, None, 3), slice(None)),
    )
    assert dt == 6e-12


def test_materialized_fft_time_scale_changes_effective_dt():
    from types import SimpleNamespace

    from mmpp.fft._compute_loading import load_fft_input_data

    class Logger:
        debug = info = warning = lambda *args, **kwargs: None

    dataset = SimpleNamespace(shape=(8, 1, 2, 2, 3), attrs={"t_sampl": 2e-12})

    class Job:
        attrs = {}
        m = dataset

        def __getitem__(self, key):
            return dataset

    job = Job()

    class Pyzfn:
        def __new__(cls, path):
            return job

    data = np.zeros((4, 1, 2, 2, 3), dtype=np.float32)
    _, dt = load_fft_input_data(
        zarr_path="unused",
        dataset="m",
        z_layer=0,
        tmax=None,
        slice_info=None,
        pyzfn_available=True,
        pyzfn_cls=Pyzfn,
        psutil_module=None,
        logger=Logger(),
        preloaded_data=data,
        time_step_scale=2.0,
    )
    assert dt == 4e-12


def test_materialized_fft_views_have_content_sensitive_cache_identity():
    from mmpp.fft.spectrum.compute import fingerprint_array

    first = np.zeros((4, 1, 2, 3, 3), dtype=np.float32)
    second = first.copy()
    second[0, 0, 0, 0, 0] = 1
    assert fingerprint_array(first) != fingerprint_array(second)
    assert fingerprint_array(first) == fingerprint_array(first.copy())


def test_fft_compute_forwards_materialized_view(monkeypatch):
    import mmpp.fft.compute_fft as module

    expected = np.zeros((8, 2, 2, 3), dtype=np.float32)
    captured = {}

    def fake_load(**kwargs):
        captured.update(kwargs)
        return expected, 1e-12, object()

    monkeypatch.setattr(module, "load_fft_input_data_profiled", fake_load)
    monkeypatch.setattr(module, "normalize_z_layer_index", lambda **_: 0)
    monkeypatch.setattr(module, "log_input_load_metrics", lambda **_: None)
    monkeypatch.setattr(
        module.FFTCompute,
        "calculate_fft_method1",
        lambda *args, **kwargs: module.FFTComputeResult(
            frequencies=np.arange(2), spectrum=np.arange(2), metadata={}, config={}
        ),
    )
    module.FFTCompute().calculate_fft_data(
        "/tmp/source.zarr", "m", preloaded_data=expected, force=True
    )
    assert captured["preloaded_data"] is expected


def test_filter_pipeline_forwards_configured_preprocess_parameters(monkeypatch):
    import mmpp.fft.filters.pipeline as pipeline_module

    calls = []

    def fake_filter(data, name, **parameters):
        calls.append((name, parameters))
        return data

    monkeypatch.setattr(pipeline_module, "apply_preprocess_filter", fake_filter)
    pipeline_module.FilterPipeline().preprocess(
        np.arange(8.0),
        filters={
            "pre": {
                "high_pass": {"cutoff_fraction": 0.125},
                "band_pass": {"low_fraction": 0.05, "high_fraction": 0.2},
                "spectral_derivative": {"order": 2},
            }
        },
    )
    assert calls == [
        ("high_pass", {"cutoff_fraction": 0.125}),
        ("band_pass", {"low_fraction": 0.05, "high_fraction": 0.2}),
        ("spectral_derivative", {"order": 2, "spacing": 1.0}),
    ]


def test_single_selected_mode_component_is_restored_to_cartesian_slot():
    from types import SimpleNamespace

    pytest.importorskip("matplotlib")
    from mmpp.fft.modes import FMRModeAnalyzer

    analyzer = object.__new__(FMRModeAnalyzer)
    analyzer.frequencies = np.asarray([1.0])
    analyzer.modes_path = "modes/m/views/test/arr"
    analyzer.mode_group = "modes/m/views/test"
    analyzer.component_index = 0
    analyzer.dx = analyzer.dy = 1.0
    analyzer._mode_cache = SimpleNamespace(
        get=lambda *args: None, put=lambda *args: None
    )
    array = np.ones((1, 1, 2, 3, 1), dtype=np.complex64)
    analyzer.zarr_file = {analyzer.modes_path: array}

    mode = analyzer.get_mode(1.0, z_layer=0)
    assert np.all(mode.mode_array[..., 0] == 1)
    assert np.all(mode.mode_array[..., 1:] == 0)


def test_transmission_cache_distinguishes_materialized_view_identity():
    from types import SimpleNamespace

    from mmpp.fft.transmission.cache import TransmissionCache
    from mmpp.fft.transmission.compute import TransmissionConfig

    cache = TransmissionCache(SimpleNamespace(), dataset_name="m")
    config = TransmissionConfig(dataset_name="m")
    first = cache.generate_cache_key(
        config, slice_info=None, view_identity="float32:(4,):aaa;dt_scale=1"
    )
    second = cache.generate_cache_key(
        config, slice_info=None, view_identity="float32:(4,):bbb;dt_scale=2"
    )
    assert first != second


def test_band_pass_fractions_are_relative_to_nyquist():
    from mmpp.fft.filters.preprocess import band_pass

    n = 1024
    samples = np.arange(n, dtype=float)
    pass_tone = np.sin(2 * np.pi * 0.05 * samples)  # 0.1 Nyquist
    stop_tone = np.sin(2 * np.pi * 0.20 * samples)  # 0.4 Nyquist
    filtered_pass = band_pass(pass_tone, low_fraction=0.05, high_fraction=0.2)
    filtered_stop = band_pass(stop_tone, low_fraction=0.05, high_fraction=0.2)
    assert np.std(filtered_pass) > 5 * np.std(filtered_stop)


@pytest.mark.parametrize(
    "low,high",
    [(0.0, 0.5), (0.5, 0.5), (0.6, 0.5), (0.1, 1.1)],
)
def test_band_pass_rejects_invalid_fraction_ranges(low, high):
    from mmpp.fft.filters.preprocess import band_pass

    with pytest.raises(ValueError, match="fractions"):
        band_pass(np.arange(16.0), low_fraction=low, high_fraction=high)


def test_fft_engine_rejects_unknown_name_instead_of_silent_numpy_fallback():
    from mmpp.fft._compute_engines import compute_fft_data

    with pytest.raises(ValueError, match="Unsupported FFT engine"):
        compute_fft_data(
            data=np.arange(8.0),
            dt=1.0,
            engine="numpyy",
            zero_padding=False,
            nfft=None,
            scipy_available=False,
            pyfftw_available=False,
        )


def test_requested_unavailable_fft_engine_fails_explicitly():
    from mmpp.fft._compute_engines import determine_engine_name

    with pytest.raises(ImportError, match="scipy"):
        determine_engine_name(
            configured_engine="scipy",
            data_size=10,
            scipy_available=False,
            pyfftw_available=False,
        )


def test_spectrum_filter_chain_preserves_preprocess_parameters():
    from mmpp.fft.spectrum.filter_chain import SpectrumFilterChain

    captured = {}

    def spectrum_callable(**kwargs):
        captured.update(kwargs)
        return object()

    SpectrumFilterChain(
        spectrum_callable,
        {
            "pre": {
                "high_pass": {"cutoff_fraction": 0.125},
                "band_pass": {"low_fraction": 0.05, "high_fraction": 0.2},
            }
        },
    ).spectrum()
    assert captured["filter_type"] == {
        "high_pass": {"cutoff_fraction": 0.125},
        "band_pass": {"low_fraction": 0.05, "high_fraction": 0.2},
    }


def test_fft_save_failure_is_not_reported_as_success(monkeypatch):
    import mmpp.fft.compute_fft as module

    data = np.arange(8.0)
    monkeypatch.setattr(module, "normalize_z_layer_index", lambda **_: 0)
    monkeypatch.setattr(
        module,
        "load_fft_input_data_profiled",
        lambda **_: (data, 1.0, object()),
    )
    monkeypatch.setattr(module, "log_input_load_metrics", lambda **_: None)
    monkeypatch.setattr(
        module.FFTCompute,
        "calculate_fft_method1",
        lambda *args, **kwargs: module.FFTComputeResult(
            frequencies=np.arange(5.0),
            spectrum=np.arange(5.0),
            metadata={},
            config=module.FFTComputeConfig(),
        ),
    )
    monkeypatch.setattr(
        module.FFTComputeResult,
        "save_to_zarr",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read-only")),
    )

    with pytest.raises(RuntimeError, match="save=True failed"):
        module.FFTCompute().calculate_fft_data(
            "/tmp/source.zarr", "m", save=True, force=True
        )


def test_fft_cache_missing_requested_slice_metadata_is_not_a_match():
    from types import SimpleNamespace

    from mmpp.fft._compute_cache import verify_fft_parameters

    cached = SimpleNamespace(
        config=SimpleNamespace(
            window_function="hann",
            filter_type="remove_mean",
            fft_engine="numpy",
            scaling="raw",
            zero_padding=True,
            nfft=None,
        ),
        metadata={"z_layer": 0, "source_dataset": "m"},
    )
    assert not verify_fft_parameters(
        existing_result=cached,
        window="hann",
        filter_type="remove_mean",
        engine="numpy",
        scaling="raw",
        zero_padding=True,
        nfft=None,
        metadata_overrides={
            "z_layer": 0,
            "source_dataset": "m",
            "slice_identifier": "slice=0:10:None",
        },
    )


def test_dispersion_pipeline_executes_standard_parameterized_pre_filter():
    from mmpp.fft.dispersion.utils import apply_filter_pipeline

    n = 512
    samples = np.arange(n, dtype=float)
    low = np.sin(2 * np.pi * 0.01 * samples)
    high = 0.2 * np.sin(2 * np.pi * 0.20 * samples)
    signal = (low + high)[:, None, None, None]
    filtered = apply_filter_pipeline(
        signal,
        {"pre": {"high_pass": {"cutoff_fraction": 0.2}}},
        time_axis=0,
    )
    assert not np.allclose(filtered, signal)
    assert np.std(filtered) < np.std(signal)


def test_spectral_derivative_uses_physical_time_spacing():
    from mmpp.fft.filters.pipeline import FilterPipeline

    dt = 2.5e-12
    slope = 3.0e6
    time = np.arange(32, dtype=float) * dt
    signal = (slope * time)[:, None]

    differentiated = FilterPipeline().preprocess(
        signal,
        dt=dt,
        filters={"pre": {"spectral_derivative": {"order": 1}}},
    )

    assert np.allclose(differentiated, slope, rtol=1e-12, atol=1e-9)


def test_dispersion_spectral_derivative_uses_physical_time_spacing():
    from mmpp.fft.dispersion.utils import apply_filter_pipeline

    dt = 4.0e-12
    slope = 7.0e5
    time = np.arange(24, dtype=float) * dt
    signal = (slope * time)[:, None, None, None]

    differentiated = apply_filter_pipeline(
        signal,
        {"pre": {"spectral_derivative": {"order": 1}}},
        time_axis=0,
        dt=dt,
    )

    assert np.allclose(differentiated, slope, rtol=1e-12, atol=1e-9)


def test_dispersion_welch_rejects_unsupported_physical_scaling(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    path = _write_complex_wave_zarr(
        tmp_path / "welch_scaling.zarr",
        n_t=16,
        n_x=8,
    )
    analyzer = SpinWaveAnalyzer(
        path,
        config=DispersionConfig(time_window=None, space_window=None, detrend=None),
        tmax=None,
    )

    with pytest.raises(ValueError, match="supports only scaling='raw_power'"):
        analyzer.compute_dispersion_1d(
            axis="x",
            component="perp",
            scaling="psd",
            filters={"pre": {"welch_average": {"n_segments": 2}}},
        )


def test_dispersion_preloaded_single_component_is_normalized_before_1d_and_2d(
    tmp_path,
):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    n_t, n_y, n_x = 16, 6, 8
    path = _write_complex_xy_wave_zarr(
        tmp_path / "preloaded_scalar_dispersion.zarr",
        n_t=n_t,
        n_y=n_y,
        n_x=n_x,
        f_bin=2,
        kx_bin=2,
        ky_bin=1,
    )
    full = np.asarray(zarr.open(str(path), mode="r")["m"])
    selected_mx = full[..., 0]
    analyzer = SpinWaveAnalyzer(
        path,
        config=DispersionConfig(time_window=None, space_window=None, detrend=None),
        tmax=None,
        preloaded_data=selected_mx,
    )

    assert analyzer.M_data.shape == (n_t, 1, n_y, n_x, 1)
    result_1d = analyzer.compute_dispersion_1d(
        axis="x", component="auto", store_complex=False
    )
    result_2d = analyzer.compute_dispersion_2d(
        component="auto", time_window=None, detrend=None
    )
    assert result_1d.shape == (n_x, n_t)
    assert result_2d.shape == (n_x, n_y, n_t // 2 + 1)


def test_dispersion_filter_chain_compute_2d_does_not_forward_false_cache_flags():
    from mmpp.fft.dispersion.filter_chain import DispersionFilterChain

    class FakeInterface:
        def compute_2d(self, component=None, **kwargs):
            return component, kwargs

    chain = DispersionFilterChain(FakeInterface())
    assert chain.compute_2d(component="perp") == ("perp", {})
    with pytest.raises(NotImplementedError, match="not implemented"):
        chain.compute_2d(save=True)
    with pytest.raises(ValueError, match="has no effect"):
        chain.compute_2d(force=True)


def test_dispersion_2d_scaling_windows_and_filter_contract(tmp_path):
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    amplitude = 0.25
    path = _write_complex_xy_wave_zarr(
        tmp_path / "dispersion_2d_scaling.zarr",
        n_t=16,
        n_y=8,
        n_x=8,
        amplitude=amplitude,
        f_bin=2,
        kx_bin=2,
        ky_bin=1,
    )
    analyzer = SpinWaveAnalyzer(
        path,
        config=DispersionConfig(time_window=None, space_window=None, detrend=None),
        tmax=None,
    )

    scaled = analyzer.compute_dispersion_2d(
        component="perp",
        time_window="hann",
        space_window="hann",
        detrend=None,
        scaling="amplitude_squared",
        filters={"pre": {"remove_mean": True}},
    )
    assert np.isclose(scaled.S.max(), amplitude**2, rtol=1e-5)
    assert scaled.scaling == "amplitude_squared"
    assert scaled.scaling_factors["scale"] > 0
    assert any("Pre-filters: remove_mean" in note for note in scaled.notes)

    with pytest.raises(NotImplementedError, match="Post/live filters"):
        analyzer.compute_dispersion_2d(
            component="perp",
            filters={"post": {"normalize": True}},
        )


def test_circular_basis_is_unitary_and_uses_documented_algebraic_sign():
    from mmpp.fft.modes.vortex_optics import VortexOptics

    mx = np.array([1.0 + 2.0j, -0.5j])
    my = np.array([0.25 - 1.0j, 3.0])
    plus, minus = VortexOptics.to_circular_basis(mx, my)

    assert np.allclose(plus, (mx + 1j * my) / np.sqrt(2.0))
    assert np.allclose(minus, (mx - 1j * my) / np.sqrt(2.0))
    assert np.allclose(
        np.abs(plus) ** 2 + np.abs(minus) ** 2,
        np.abs(mx) ** 2 + np.abs(my) ** 2,
    )


def test_transmission_cpsd_uses_fixed_external_reference_spectrum():
    from mmpp.fft.transmission.compute import _apply_transmission_method

    spectrum = np.array([[[[2.0 + 0.0j], [3.0 + 0.0j]]]])
    reference = np.array([[[5.0 + 0.0j]]])
    metric = _apply_transmission_method(
        spectrum,
        component_weights=np.array([1.0]),
        method="cpsd",
        window_axis=2,
        reference_spectrum=reference,
    )

    assert metric.shape == (1, 1, 2)
    assert np.allclose(metric, [[[10.0, 15.0]]])
    with pytest.raises(ValueError, match="fixed reference spectrum"):
        _apply_transmission_method(
            spectrum,
            component_weights=np.array([1.0]),
            method="cpsd",
            window_axis=2,
        )


def test_transmission_power_metrics_use_squared_fft_magnitude():
    from mmpp.fft.transmission.compute import _apply_transmission_method

    spectrum = np.array([[[3.0 + 4.0j, 0.0j, 2.0j]]])
    metric = _apply_transmission_method(
        spectrum,
        component_weights=np.array([1.0, 0.0, 0.5]),
        method="power_ratio",
        window_axis=None,
    )
    assert np.allclose(metric, [[25.0 + 0.5 * 4.0]])

    circular_spectrum = np.array([[[1.0 + 0.0j, 0.0 + 1.0j]]])
    circular = _apply_transmission_method(
        circular_spectrum,
        component_weights=np.array([1.0, 1.0]),
        method="circular",
        window_axis=None,
    )
    assert np.allclose(circular, [[1.0]])


def test_transmission_reference_normalization_marks_zero_reference_undefined():
    from mmpp.fft.transmission.compute import _normalize_transmission_map

    power = np.array([[0.0, 2.0], [3.0, 6.0]])
    reference = np.array([0.0, 3.0])
    normalized, invalid = _normalize_transmission_map(power, reference, "reference")

    assert invalid == 1
    assert np.all(np.isnan(normalized[0]))
    assert np.allclose(normalized[1], [1.0, 2.0])


@pytest.mark.parametrize("component", ["x", "y", "z"])
def test_interactive_mode_resolver_preserves_singleton_component_identity(component):
    from mmpp.fft.modes._interactive.filters import resolve_mode_components

    singleton = np.full((3, 4, 1), 7.0 + 2.0j)
    resolved = resolve_mode_components(singleton, [component])
    assert list(resolved) == [component]
    assert np.array_equal(resolved[component], singleton[..., 0])


def test_interactive_mode_resolver_rejects_topological_basis_from_single_channel():
    from mmpp.fft.modes._interactive.filters import resolve_mode_components

    with pytest.raises(ValueError, match="require both mx and my"):
        resolve_mode_components(np.ones((3, 4, 1), dtype=complex), ["+"])


def test_interactive_holography_forwards_gamma_and_noise_threshold(monkeypatch):
    from mmpp.fft.modes._interactive.callbacks import _resolve_mode_viz
    from mmpp.fft.modes.vortex_optics import VortexOptics

    captured = {}

    def fake_holography(data, gamma=0.6, noise_threshold=1e-4, saturation=1.0):
        captured.update(gamma=gamma, noise_threshold=noise_threshold)
        return np.zeros(data.shape + (3,))

    monkeypatch.setattr(VortexOptics, "complex_holography", fake_holography)
    rendered, vmin, vmax = _resolve_mode_viz(
        np.ones((2, 3), dtype=complex),
        viz_type="phase",
        use_holography=True,
        holography_gamma=0.35,
        holography_noise_threshold=0.02,
    )
    assert rendered.shape == (2, 3, 3)
    assert (vmin, vmax) == (None, None)
    assert captured == {"gamma": 0.35, "noise_threshold": 0.02}


def test_interactive_show_rejects_unknown_options_before_loading_data():
    from mmpp.fft.modes.interactive import InteractiveSpectrum

    viewer = object.__new__(InteractiveSpectrum)
    with pytest.raises(TypeError, match="unknown_typo"):
        viewer.show(unknown_typo=True)


def test_topological_holography_preserves_arbitrary_input_shape(monkeypatch):
    import mmpp.fft.modes.vortex_optics as module

    monkeypatch.setattr(module, "_hsv_to_rgb", lambda hsv: hsv)
    data = np.ones((2, 3, 4), dtype=complex)
    static = module.VortexOptics.complex_holography(data)
    animated = module.TopologicalAnimator(data).get_hologram_frame(0.5)
    assert static.shape == (2, 3, 4, 3)
    assert animated.shape == (2, 3, 4, 3)


@pytest.mark.parametrize("gamma,noise", [(0.0, 1e-4), (0.6, -0.1)])
def test_topological_holography_rejects_invalid_scaling(gamma, noise):
    from mmpp.fft.modes.vortex_optics import VortexOptics

    with pytest.raises(ValueError):
        VortexOptics.complex_holography(
            np.ones((2, 2), dtype=complex),
            gamma=gamma,
            noise_threshold=noise,
        )


def test_phase_preview_time_is_undefined_for_dc_without_division_by_zero():
    from mmpp.fft.modes._interactive.callbacks import _phase_preview_time_ns

    assert np.isnan(_phase_preview_time_ns(0.0, 0.25))
    assert np.isclose(_phase_preview_time_ns(2.0, 0.25), 0.125)


def test_cylindrical_basis_uses_right_handed_increasing_y_axis():
    from mmpp.fft.modes.vortex_optics import VortexOptics

    ny, nx = 5, 7
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    y, x = np.indices((ny, nx))
    mx = x - cx
    my = y - cy
    rho, phi = VortexOptics.to_cylindrical_basis(mx, my, center=(cx, cy))

    assert np.allclose(rho, np.hypot(mx, my))
    assert np.allclose(phi, 0.0, atol=1e-14)


def test_fft_numeric_interfaces_import_without_plotting_dependencies():
    import mmpp.fft.core as fft_core
    import mmpp.fft.electromagnetic_analysis as electromagnetic
    import mmpp.fft.transmission.experimental as transmission_experimental
    import mmpp.fft.transmission.interface as transmission_interface
    import mmpp.fft.transmission.overlay_experimental as overlay_experimental
    from mmpp.fft.modes.analyzer.data_access import DataAccessMixin

    assert fft_core.FFT is not None
    assert electromagnetic.PoyntingVectorAnalysis is not None
    assert transmission_interface.FFTTransmissionInterface is not None
    assert transmission_experimental.overlay_transmission is not None
    assert overlay_experimental.overlay_experimental_transmission is not None
    assert DataAccessMixin is not None


def test_legacy_modes_init_exports_only_defined_names():
    import mmpp.fft.modes.init as legacy_init

    assert all(hasattr(legacy_init, name) for name in legacy_init.__all__)


def test_manual_dispersion_diagnostic_is_silent_on_import(capsys):
    import importlib

    import mmpp.fft.dispersion.test_dispersion_models as diagnostic

    importlib.reload(diagnostic)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_transmission_batch_cache_hash_preserves_input_job_order():
    from types import SimpleNamespace

    from mmpp.fft.transmission.batch import BatchTransmission
    from mmpp.fft.transmission.compute import TransmissionConfig

    first = SimpleNamespace(path="/tmp/job-a.zarr")
    second = SimpleNamespace(path="/tmp/job-b.zarr")
    batch = BatchTransmission([first, second], mmpp_ref=None)
    config = TransmissionConfig(metadata={"tag": "ordered"})
    forward = batch._generate_batch_cache_hash(config, "m", None, ["B0"])
    batch.results = [second, first]
    reverse = batch._generate_batch_cache_hash(config, "m", None, ["B0"])
    assert forward != reverse


def test_transmission_batch_result_persists_partial_errors(tmp_path):
    from mmpp.fft.transmission.batch import BatchTransmissionResult

    batch = BatchTransmissionResult(
        results=[object()],
        parameters={"B0": [0.1]},
        job_paths=["job-a.zarr"],
        errors=[{"index": 1, "path": "job-b.zarr", "error": "broken"}],
    )
    path = tmp_path / "batch.pkl"
    batch.save(path)
    loaded = BatchTransmissionResult.load(path)
    assert loaded.errors == batch.errors


def test_transmission_batch_cache_hash_changes_when_source_state_changes(tmp_path):
    from types import SimpleNamespace

    from mmpp.fft.transmission.batch import BatchTransmission
    from mmpp.fft.transmission.compute import TransmissionConfig

    source = tmp_path / "job.zarr"
    source.mkdir()
    chunk = source / "0.0.0"
    chunk.write_bytes(b"first")
    batch = BatchTransmission([SimpleNamespace(path=source)], mmpp_ref=None)
    config = TransmissionConfig()
    initial = batch._generate_batch_cache_hash(config, "m", None, [])
    chunk.write_bytes(b"second-payload")
    changed = batch._generate_batch_cache_hash(config, "m", None, [])
    assert changed != initial


def test_transmission_batch_detects_shifted_same_length_frequency_grid():
    from mmpp.fft.transmission.batch import _frequency_grids_match

    reference = np.array([1.0, 2.0, 3.0])
    shifted = np.array([1.5, 2.5, 3.5])
    assert not _frequency_grids_match(reference, shifted)


def test_transmission_batch_frequency_interpolation_preserves_unknown_support():
    from mmpp.fft.transmission.batch import _interpolate_frequency_series

    source_freq = np.array([1.0, 2.0, 3.0])
    source_values = np.array([10.0, 20.0, 30.0])
    target_freq = np.array([0.5, 1.5, 2.5, 3.5])

    aligned = _interpolate_frequency_series(source_freq, source_values, target_freq)

    assert np.isnan(aligned[[0, -1]]).all()
    assert np.allclose(aligned[1:3], [15.0, 25.0])


def test_transmission_batch_frequency_interpolation_handles_descending_grids():
    from mmpp.fft.transmission.batch import _interpolate_frequency_series

    aligned = _interpolate_frequency_series(
        np.array([3.0, 2.0, 1.0]),
        np.array([30.0, 20.0, 10.0]),
        np.array([2.5, 1.5]),
    )

    assert np.allclose(aligned, [25.0, 15.0])


def test_mode_extent_converts_view_geometry_from_metres_to_nanometres():
    from types import SimpleNamespace

    from mmpp.fft.modes import _mode_extent_nm

    geometry = SimpleNamespace(
        axes={
            "x": SimpleNamespace(min_m=100e-9, max_m=500e-9),
            "y": SimpleNamespace(min_m=20e-9, max_m=220e-9),
        }
    )

    extent = _mode_extent_nm(geometry, nx=4, ny=2, dx_nm=100.0, dy_nm=100.0)

    assert np.allclose(extent, (100.0, 500.0, 20.0, 220.0))


def test_mode_get_reuses_cached_actual_frequency_bin():
    from mmpp.fft.modes import FMRModeAnalyzer
    from mmpp.fft.modes.analyzer.cache import ModeCache

    class CountingModesArray:
        shape = (2, 1, 2, 3, 3)

        def __init__(self):
            self.reads = 0

        def __getitem__(self, key):
            self.reads += 1
            return np.ones((2, 3, 3), dtype=complex)

    modes_array = CountingModesArray()
    analyzer = object.__new__(FMRModeAnalyzer)
    analyzer.frequencies = np.array([1.0, 2.0])
    analyzer.modes_path = "modes/m/arr"
    analyzer.mode_group = "modes/m"
    analyzer.zarr_file = {analyzer.modes_path: modes_array}
    analyzer.component_index = None
    analyzer.view_geometry = None
    analyzer.dx = 5.0
    analyzer.dy = 7.0
    analyzer._mode_cache = ModeCache(maxsize=4)

    first = analyzer.get_mode(1.01)
    second = analyzer.get_mode(0.99)

    assert second is first
    assert modes_array.reads == 1


def test_mode_time_axis_selection_matches_view_and_nested_time_slice():
    from mmpp.fft.modes import _select_mode_time_axis, _uniform_mode_dt

    raw_time = np.arange(12, dtype=float) * 2e-12
    selected = _select_mode_time_axis(
        raw_time,
        total_samples=5,
        view_slice=(slice(1, 11, 2), slice(None)),
        time_slice=slice(1, 5, 2),
        expected_samples=2,
    )

    assert np.array_equal(selected, raw_time[1:11:2][1:5:2])
    assert np.isclose(_uniform_mode_dt(selected), 8e-12)


def test_mode_time_axis_selection_rejects_mismatched_materialized_metadata():
    from mmpp.fft.modes import _select_mode_time_axis

    selected = _select_mode_time_axis(
        np.arange(20, dtype=float),
        total_samples=5,
        view_slice=None,
        time_slice=slice(None),
        expected_samples=5,
    )

    assert selected is None


def test_mode_fft_rejects_nonuniform_time_axis():
    from mmpp.fft.modes import _uniform_mode_dt

    with pytest.raises(ValueError, match="uniformly sampled"):
        _uniform_mode_dt(np.array([0.0, 1e-12, 3e-12, 4e-12]))


def test_fft_dt_uses_full_selected_time_axis_and_rejects_irregular_sampling():
    from types import SimpleNamespace

    from mmpp.fft._compute_loading import resolve_dt_from_metadata

    class Logger:
        debug = info = warning = lambda *args, **kwargs: None

    uniform = np.arange(10, dtype=float) * 2e-12
    dt = resolve_dt_from_metadata(
        data_set=SimpleNamespace(attrs={"t": uniform}),
        job=SimpleNamespace(attrs={}),
        logger=Logger(),
        slice_info=(slice(1, 9, 2), slice(None)),
    )
    assert np.isclose(dt, 4e-12)

    irregular = uniform.copy()
    irregular[5:] += 0.4e-12
    with pytest.raises(ValueError, match="uniformly sampled"):
        resolve_dt_from_metadata(
            data_set=SimpleNamespace(attrs={"t": irregular}),
            job=SimpleNamespace(attrs={}),
            logger=Logger(),
        )


@pytest.mark.parametrize(
    "source_shape,component_index,expected_shape",
    [
        ((8, 2, 3, 4, 3), None, (8, 2, 3, 4, 3)),
        ((8, 3, 4, 3), None, (8, 1, 3, 4, 3)),
        ((8, 2, 3, 4), 1, (8, 2, 3, 4, 1)),
        ((8, 3, 4), 2, (8, 1, 3, 4, 1)),
        ((8, 1, 3, 4, 1), 2, (8, 1, 3, 4, 1)),
        ((8, 3, 4, 1), 0, (8, 1, 3, 4, 1)),
    ],
)
def test_mode_input_shape_preserves_selected_z_and_component_axes(
    source_shape, component_index, expected_shape
):
    from mmpp.fft.modes import _normalize_mode_input_shape

    normalized = _normalize_mode_input_shape(
        np.zeros(source_shape), component_index=component_index
    )

    assert normalized.shape == expected_shape


def test_mode_power_cache_candidates_never_fall_back_to_full_dataset_for_view():
    from mmpp.fft.modes import _mode_power_paths

    view_group = "modes/m/views/abc123"
    paths = _mode_power_paths(view_group, dataset_name="m", include_legacy=False)

    assert paths == [
        (f"{view_group}/power_sum", "power_sum"),
        (f"{view_group}/power_max", "power_max"),
    ]
    assert all(path.startswith(view_group) for path, _ in paths)


def test_mode_power_summaries_use_squared_fft_magnitude():
    from mmpp.fft.modes import _mode_power_summaries

    fft_result = np.array(
        [
            [[3.0 + 4.0j, 0.0j]],
            [[1.0j, 2.0 + 0.0j]],
        ]
    )
    power_max, power_sum = _mode_power_summaries(fft_result)

    assert np.allclose(power_max, [25.0, 4.0])
    assert np.allclose(power_sum, [25.0, 5.0])


def test_mode_power_cache_requires_explicit_squared_definition():
    from types import SimpleNamespace

    from mmpp.fft.modes import _mode_power_cache_is_squared

    assert _mode_power_cache_is_squared(
        SimpleNamespace(attrs={"power_definition": "abs_fft_squared"})
    )
    assert not _mode_power_cache_is_squared(SimpleNamespace(attrs={}))
    assert not _mode_power_cache_is_squared(
        SimpleNamespace(attrs={"power_definition": "abs_fft"})
    )


def test_mode_find_peaks_honors_component_and_explicit_threshold():
    from types import SimpleNamespace

    from mmpp.fft.modes import FMRModeAnalyzer

    analyzer = object.__new__(FMRModeAnalyzer)
    analyzer.config = SimpleNamespace(
        peak_threshold=0.1,
        peak_min_distance=1,
        spectrum_normalize=True,
        f_min=0.0,
        f_max=4.0,
    )
    frequencies = np.arange(5, dtype=float)
    spectrum = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 0.0],
            [0.0, 7.0],
            [0.0, 0.0],
        ]
    )

    first = analyzer.find_peaks(
        spectrum=spectrum, frequencies=frequencies, component=0, threshold=0.0
    )
    second = analyzer.find_peaks(
        spectrum=spectrum, frequencies=frequencies, component=1, threshold=0.0
    )
    suppressed = analyzer.find_peaks(
        spectrum=spectrum, frequencies=frequencies, component=1, threshold=1.1
    )

    assert [peak.freq for peak in first] == [1.0]
    assert [peak.freq for peak in second] == [3.0]
    assert suppressed == []


def test_mode_data_loader_resolves_only_exact_view_group():
    from mmpp.fft.modes.data_loader import ModeDataContext, ModeDataLoader

    view_group = "modes/m/views/exact"
    context = ModeDataContext(
        zarr_path="unused", dataset_name="m", mode_group=view_group
    )
    loader = ModeDataLoader(context)
    loader._zarr_file = {
        f"{view_group}/arr": np.zeros((3, 1, 2, 2, 3)),
        f"{view_group}/freqs": np.arange(3.0),
        f"{view_group}/power_sum": np.arange(3.0),
        "modes/m/arr": np.zeros((7, 1, 2, 2, 3)),
        "modes/m/freqs": np.arange(7.0),
        "modes/m/power_sum": np.arange(7.0),
    }

    modes_path, freqs_path, spectrum_path = loader._resolve_paths()

    assert modes_path == f"{view_group}/arr"
    assert freqs_path == f"{view_group}/freqs"
    assert spectrum_path == f"{view_group}/power_sum"


def test_mode_data_loader_refuses_to_invent_frequency_axis():
    from types import SimpleNamespace

    from mmpp.fft.modes.data_loader import ModeDataContext, ModeDataLoader

    group = "modes/m/views/exact"
    loader = ModeDataLoader(
        ModeDataContext(zarr_path="unused", dataset_name="m", mode_group=group)
    )
    loader._zarr_file = {
        f"{group}/arr": np.zeros((2, 1, 1, 1, 3)),
        f"{group}/freqs": np.arange(2.0),
        f"{group}/power_sum": np.arange(3.0),
        group: SimpleNamespace(attrs={"power_definition": "abs_fft_squared"}),
    }

    with pytest.raises(RuntimeError, match="No frequency axis matches"):
        loader.load_spectrum()


def test_interactive_spectrum_uses_explicit_hz_to_ghz_contract_for_low_frequency():
    from types import SimpleNamespace

    from mmpp.fft.modes._interactive.data import load_spectrum_data
    from mmpp.fft.spectrum.result import SpectrumResult

    result = SpectrumResult(
        frequencies=np.array([100e3, 200e3]),
        spectrum=np.array([1.0 + 0.0j, 2.0 + 0.0j]),
    )
    explorer = SimpleNamespace(
        spectrum_result=result,
        data_loader=None,
        analyzer=None,
        _component_label=None,
    )

    load_spectrum_data(explorer)

    assert np.allclose(explorer._raw_frequencies_ghz, [1e-4, 2e-4])


def test_interactive_spectrum_filters_reject_frequency_length_mismatch():
    from mmpp.fft.modes._interactive.filters import (
        SpectrumFilterState,
        apply_spectrum_filters,
    )

    with pytest.raises(ValueError, match="frequency axis has 3"):
        apply_spectrum_filters(
            np.arange(3.0),
            {"z": np.arange(2.0)},
            SpectrumFilterState(freq_min=0.0, freq_max=2.0),
        )


def test_interactive_spectrum_filters_reject_empty_frequency_window():
    from mmpp.fft.modes._interactive.filters import (
        SpectrumFilterState,
        apply_spectrum_filters,
    )

    with pytest.raises(ValueError, match="does not overlap available data"):
        apply_spectrum_filters(
            np.array([1.0, 2.0, 3.0]),
            {"z": np.array([2.0, 4.0, 2.0])},
            SpectrumFilterState(freq_min=10.0, freq_max=20.0),
        )


def test_modes_fluent_frequency_properties_and_default_are_in_ghz():
    from types import SimpleNamespace

    from mmpp.fft.modes.interface import FFTModeInterfaceNew
    from mmpp.fft.spectrum.result import SpectrumResult

    result = SpectrumResult(
        frequencies=np.array([1e9, 2e9, 3e9]),
        spectrum=np.array([1.0 + 0.0j, 4.0 + 0.0j, 2.0 + 0.0j]),
    )
    parent = SimpleNamespace(
        job_result=SimpleNamespace(path="unused"),
        _spectrum_impl=lambda **kwargs: result,
    )
    interface = FFTModeInterfaceNew(0, parent)
    interface._dataset_context = "m"

    assert np.allclose(interface.frequencies, [1.0, 2.0, 3.0])
    assert np.isclose(interface._default_mode_frequency(), 2.0)


def test_spectrum_modes_bridge_clones_and_preserves_materialized_view_context():
    from types import SimpleNamespace

    from mmpp.fft.spectrum.modes.bridge import SpectrumModes
    from mmpp.fft.spectrum.result import SpectrumResult

    class BaseInterface:
        _dataset_context = None
        _slice_context = None

        def _clone(self):
            return SimpleNamespace(
                _dataset_context=self._dataset_context,
                _slice_context=self._slice_context,
            )

    base = BaseInterface()
    source_fft = SimpleNamespace(modes=base)
    materialized = np.arange(6.0).reshape(2, 3)
    geometry = object()
    spectrum = SpectrumResult(
        frequencies=np.array([1e9]),
        spectrum=np.array([1.0 + 0.0j]),
        source_fft=source_fft,
        mode_context={
            "dset": "m_view",
            "slice_info": (slice(2, 8), Ellipsis),
            "preloaded_data": materialized,
            "time_step_scale": 3.0,
            "view_geometry": geometry,
        },
    )

    resolved = SpectrumModes(spectrum)._resolve_interface()

    assert resolved is not base
    assert base._dataset_context is None
    assert resolved._dataset_context == "m_view"
    assert resolved._slice_context == (slice(2, 8), Ellipsis)
    assert resolved._preloaded_context is materialized
    assert resolved._time_step_scale_context == 3.0
    assert resolved._geometry_context is geometry


def test_bulk_dispersion_aligns_shifted_same_length_k_grid_without_fake_zeros():
    from mmpp.fft.dispersion.bulk import _align_crosssection_to_k_grid

    aligned = _align_crosssection_to_k_grid(
        np.array([1.0, 2.0, 3.0]),
        np.array([10.0, 20.0, 30.0]),
        np.array([0.5, 1.5, 2.5, 3.5]),
    )

    assert np.isnan(aligned[[0, -1]]).all()
    assert np.allclose(aligned[1:3], [15.0, 25.0])


def test_bulk_dispersion_alignment_handles_descending_k_grid():
    from mmpp.fft.dispersion.bulk import _align_crosssection_to_k_grid

    aligned = _align_crosssection_to_k_grid(
        np.array([3.0, 2.0, 1.0]),
        np.array([30.0, 20.0, 10.0]),
        np.array([2.5, 1.5]),
    )

    assert np.allclose(aligned, [25.0, 15.0])


def test_bulk_dispersion_heatmap_rows_follow_sorted_parameter_values():
    from mmpp.fft.dispersion.bulk import _prepare_bulk_heatmap_matrix

    params = np.array([20.0, 0.0, 10.0])
    k_axes = [np.array([0.0, 1.0])] * 3
    crosssections = [
        np.array([20.0, 21.0]),
        np.array([0.0, 1.0]),
        np.array([10.0, 11.0]),
    ]

    matrix, k_ref, sorted_params, order = _prepare_bulk_heatmap_matrix(
        params, k_axes, crosssections
    )

    assert np.array_equal(sorted_params, [0.0, 10.0, 20.0])
    assert np.array_equal(order, [1, 2, 0])
    assert np.array_equal(k_ref, [0.0, 1.0])
    assert np.array_equal(matrix[:, 0], [0.0, 10.0, 20.0])


def test_bulk_dispersion_heatmap_preserves_nonuniform_parameter_coordinates():
    from mmpp.fft.dispersion.bulk import _prepare_bulk_heatmap_matrix

    params = np.array([10.0, 0.0, 1.0])
    axes = [np.array([0.0, 1.0])] * 3
    rows = [np.full(2, value) for value in params]

    matrix, _, sorted_params, _ = _prepare_bulk_heatmap_matrix(params, axes, rows)

    assert np.array_equal(sorted_params, [0.0, 1.0, 10.0])
    assert np.array_equal(matrix[:, 0], sorted_params)

    with pytest.raises(ValueError, match="unique and strictly increasing"):
        _prepare_bulk_heatmap_matrix(np.array([1.0, 1.0]), axes[:2], rows[:2])


def _bulk_result_for_serialization_tests():
    from mmpp.fft.dispersion.bulk import BulkMinimumFrequencyResult

    return BulkMinimumFrequencyResult(
        param_values=np.array([20.0, 10.0]),
        param_label="B [mT]",
        f_min_hz=np.array([2e9, 1e9]),
        k_star_rad_m=np.array([2e6, 1e6]),
        vg_at_min=np.array([2000.0, 1000.0]),
        f_at_k0_hz=np.array([2.2e9, 1.1e9]),
        crosssections_at_fmin=[np.array([2.0, 3.0]), np.array([1.0, 2.0])],
        crosssections_at_fk0=[np.array([4.0, 5.0]), np.array([3.0, 4.0])],
        branches_f=[np.array([2e9]), np.array([1e9])],
        branches_k=[np.array([2e6]), np.array([1e6])],
        k_axes=[np.array([0.0, 2e6]), np.array([0.0, 1e6])],
        errors={1: "synthetic failure"},
        meta={"axis": "x", "component": "perp"},
        analytical_f_min_hz=np.array([1.9e9, 0.9e9]),
        analytical_k_star_rad_m=np.array([1.9e6, 0.9e6]),
        analytical_f_k0_hz=np.array([2.1e9, 1.0e9]),
        analytical_model="latest",
        analytical_params={"Ms": 800e3},
        analytical_overlays=[
            {
                "label": "model A",
                "model": "kalinikos",
                "f_min_hz": np.array([1.8e9, 0.8e9]),
                "k_star_rad_m": np.array([1.8e6, 0.8e6]),
                "f_k0_hz": np.array([2.0e9, 0.9e9]),
                "params": {"Ku": 0.0},
            },
            {
                "label": "model B",
                "model": "kalinikos",
                "f_min_hz": np.array([1.9e9, 0.9e9]),
                "k_star_rad_m": np.array([1.9e6, 0.9e6]),
                "f_k0_hz": np.array([2.1e9, 1.0e9]),
                "params": {"Ku": 1.0},
            },
        ],
    )


def test_bulk_result_roundtrip_preserves_metadata_and_all_overlays(tmp_path):
    from mmpp.fft.dispersion.bulk import BulkMinimumFrequencyResult

    original = _bulk_result_for_serialization_tests()
    restored = BulkMinimumFrequencyResult.load(original.save(tmp_path / "bulk"))

    assert restored.meta == original.meta
    assert restored.errors == original.errors
    assert len(restored.analytical_overlays) == 2
    assert [item["label"] for item in restored.analytical_overlays] == [
        "model A",
        "model B",
    ]
    assert restored.analytical_overlays[1]["params"] == {"Ku": 1.0}
    assert np.array_equal(
        restored.analytical_overlays[0]["f_min_hz"],
        original.analytical_overlays[0]["f_min_hz"],
    )


def test_bulk_result_rejects_silently_truncated_per_point_sequences():
    result = _bulk_result_for_serialization_tests()
    fields = dict(result.__dict__)
    fields["k_axes"] = fields["k_axes"][:1]

    from mmpp.fft.dispersion.bulk import BulkMinimumFrequencyResult

    with pytest.raises(ValueError, match="one array per scan point"):
        BulkMinimumFrequencyResult(**fields)


def test_bulk_result_rejects_crosssection_axis_length_mismatch():
    result = _bulk_result_for_serialization_tests()
    fields = dict(result.__dict__)
    fields["crosssections_at_fmin"] = [
        np.array([1.0]),
        fields["crosssections_at_fmin"][1],
    ]

    from mmpp.fft.dispersion.bulk import BulkMinimumFrequencyResult

    with pytest.raises(ValueError, match="cross-section/k-axis length mismatch"):
        BulkMinimumFrequencyResult(**fields)


def test_bulk_result_roundtrip_accepts_numpy_analytical_parameters(tmp_path):
    from mmpp.fft.dispersion.bulk import BulkMinimumFrequencyResult

    original = _bulk_result_for_serialization_tests()
    original.analytical_params = {
        "Ms": np.float64(800e3),
        "sampled_k": np.array([0.0, 1e6]),
    }
    restored = BulkMinimumFrequencyResult.load(original.save(tmp_path / "numpy"))

    assert restored.analytical_params["Ms"] == np.float64(800e3)
    assert np.array_equal(restored.analytical_params["sampled_k"], np.array([0.0, 1e6]))


def test_bz_mask_does_not_project_out_of_range_replicas_to_edge_bins():
    from mmpp.fft.dispersion.modes.extraction import build_bz_k_mask

    k_axis = np.linspace(-2e6, 2e6, 9)
    mask = build_bz_k_mask(
        k_axis,
        k_0=0.0,
        lattice_constant=1e-6,
        n_bz=2,
    )

    assert np.array_equal(np.flatnonzero(mask), [4])


def test_bz_mask_explicit_empty_delta_window_fails_closed():
    from mmpp.fft.dispersion.modes.extraction import build_bz_k_mask

    with pytest.raises(ValueError, match="does not contain any sampled bin"):
        build_bz_k_mask(
            np.array([-1.0, 0.0, 1.0]),
            k_0=0.4,
            lattice_constant=2 * np.pi,
            n_bz=0,
            delta_k=0.1,
        )


def test_frequency_selection_explicit_empty_delta_window_fails_closed():
    from mmpp.fft.dispersion.modes.extraction import select_frequency_indices

    with pytest.raises(ValueError, match="does not contain any sampled"):
        select_frequency_indices(
            np.array([0.0, 1e9, 2e9]),
            f_0=1.4e9,
            delta_f=0.1e9,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k_direction": "sideways"}, "k_direction"),
        ({"n_bz": -1}, "n_bz"),
        ({"k_margin_bins": -1}, "k_margin_bins"),
        ({"delta_k": 0.0}, "delta_k"),
    ],
)
def test_bz_mask_rejects_invalid_selection_parameters(kwargs, message):
    from mmpp.fft.dispersion.modes.extraction import build_bz_k_mask

    base = {
        "k_axis": np.array([-1.0, 0.0, 1.0]),
        "k_0": 0.0,
        "lattice_constant": 2 * np.pi,
        "n_bz": 0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_bz_k_mask(**base)


def test_dispersion_result_rejects_axis_and_spectrum_shape_mismatches():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    with pytest.raises(ValueError, match="k_axis length"):
        DispersionResult1D(
            S=np.ones((3, 4)),
            k_axis=np.arange(2),
            f_axis=np.arange(4),
            axis="x",
            component="mz",
            config=DispersionConfig(),
        )

    with pytest.raises(ValueError, match="S_complex shape"):
        DispersionResult1D(
            S=np.ones((3, 4)),
            k_axis=np.arange(3),
            f_axis=np.arange(4),
            axis="x",
            component="mz",
            config=DispersionConfig(),
            S_complex=np.ones((2, 4), dtype=complex),
        )


def test_dispersion_result_canonicalizes_legacy_transposed_complex_spectrum():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    legacy = np.arange(12).reshape(4, 3).astype(complex)
    result = DispersionResult1D(
        S=np.ones((3, 4)),
        k_axis=np.arange(3),
        f_axis=np.arange(4),
        axis="x",
        component="mz",
        config=DispersionConfig(),
        S_complex=legacy,
    )

    assert result.S_complex.shape == (3, 4)
    assert np.array_equal(result.S_complex, legacy.T)


def test_select_orthogonal_slice_does_not_reuse_aggregate_folded_spectrum():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    local = np.stack([np.ones((3, 4)), np.full((3, 4), 2.0)])
    result = DispersionResult1D(
        S=np.mean(local, axis=0),
        k_axis=np.arange(3),
        f_axis=np.arange(4),
        axis="x",
        component="mz",
        config=DispersionConfig(),
        S_local=local,
        S_complex=local.astype(complex),
        orth_axis=np.array([0.0, 1.0]),
        S_folded=np.full((2, 4), 99.0),
        k_folded=np.array([-0.5, 0.5]),
        fold_period=1.0,
    )

    selected = result.select_orthogonal_slice(1)

    assert np.array_equal(selected.S, local[1])
    assert selected.S_folded is None
    assert selected.k_folded is None
    active, _, _ = selected.get_active_data()
    assert np.array_equal(active, local[1])


def test_dispersion_filtered_rejects_unknown_filter_instead_of_noop():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    result = DispersionResult1D(
        S=np.arange(12, dtype=float).reshape(3, 4),
        k_axis=np.arange(3),
        f_axis=np.arange(4),
        axis="x",
        component="mz",
        config=DispersionConfig(),
    )

    with pytest.raises(ValueError, match="Unknown live dispersion filter"):
        result.filtered(live={"gausian_morph": True})


def test_dispersion_filtered_applies_keyword_filter_and_preserves_raw_data():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    raw = np.arange(12, dtype=float).reshape(3, 4)
    result = DispersionResult1D(
        S=raw.copy(),
        k_axis=np.arange(3),
        f_axis=np.arange(4),
        axis="x",
        component="mz",
        config=DispersionConfig(),
    )

    filtered = result.filtered(normalize=True)

    assert np.isclose(np.max(filtered.S), 1.0)
    assert np.array_equal(result.S, raw)
    assert np.array_equal(filtered.S_raw, raw)


def test_remove_mean_and_static_does_not_reintroduce_dc_component():
    from mmpp.fft.filters.preprocess import remove_mean_and_static

    data = np.array([10.0, 12.0, 15.0, 19.0])[:, None]
    filtered = remove_mean_and_static(data)

    assert np.allclose(np.mean(filtered, axis=0), 0.0)


def test_common_filter_pipeline_rejects_unknown_filters_and_bad_stages():
    from mmpp.fft.filters.pipeline import normalize_filter_config

    with pytest.raises(ValueError, match="Unknown FFT filter"):
        normalize_filter_config({"gausian_smooth": True})
    with pytest.raises(ValueError, match="Unknown post FFT filter"):
        normalize_filter_config({"post": {"gausian_smooth": True}})
    with pytest.raises(TypeError, match="post filter stage"):
        normalize_filter_config({"post": ["normalize"]})


def test_postprocess_filter_failure_is_not_silently_ignored():
    from mmpp.fft.filters.postprocess import apply_postprocess_filters

    with pytest.raises(ValueError, match="gamma must be"):
        apply_postprocess_filters(
            np.arange(5, dtype=float),
            np.arange(5, dtype=float),
            {"gamma": {"gamma": -1.0}},
        )
    with pytest.raises(ValueError, match="Unknown baseline mode"):
        apply_postprocess_filters(
            np.arange(5, dtype=float),
            np.arange(5, dtype=float),
            {"baseline_correction": {"mode": "linar"}},
        )


def test_tracewise_preprocessing_does_not_truncate_integer_input():
    from mmpp.fft.filters.preprocess import _apply_tracewise

    data = np.array([[1, 2], [2, 4], [4, 8]], dtype=int)
    result = _apply_tracewise(data, lambda trace: trace / 2.0)

    assert np.issubdtype(result.dtype, np.floating)
    assert np.allclose(result[:, 0], [0.5, 1.0, 2.0])


def test_common_postprocess_rejects_wrong_frequency_axis_orientation():
    from mmpp.fft.filters.pipeline import FilterPipeline

    spectrum = np.ones((2, 5))
    with pytest.raises(ValueError, match="match spectrum axis 0"):
        FilterPipeline().postprocess(
            spectrum,
            np.arange(5),
            filters={"post": {"normalize": True}},
        )
    with pytest.raises(ValueError, match="stage must be"):
        FilterPipeline().postprocess(
            np.ones(5),
            np.arange(5),
            filters=None,
            stage="display",
        )


def test_dispersion_result_2d_validates_axis_shapes_and_slice_width():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult2D

    with pytest.raises(ValueError, match="does not match axis lengths"):
        DispersionResult2D(
            S=np.ones((3, 2, 4)),
            kx_axis=np.arange(2),
            ky_axis=np.arange(2),
            f_axis=np.arange(4),
            component="mz",
            config=DispersionConfig(),
        )

    result = DispersionResult2D(
        S=np.ones((3, 2, 4)),
        kx_axis=np.arange(3),
        ky_axis=np.arange(2),
        f_axis=np.arange(4),
        component="mz",
        config=DispersionConfig(),
    )
    with pytest.raises(ValueError, match="dk_max must be"):
        result.slice_1d("kx", dk_max=-1.0)


def test_dispersion_branch_smooth_flag_performs_real_smoothing():
    from mmpp.fft.dispersion.models import DispersionBranch

    k = np.linspace(0.0, 10.0, 21)
    true_slope = 3.0
    noise = 0.8 * (-1.0) ** np.arange(k.size)
    branch = DispersionBranch(
        k_path=k,
        f_values=true_slope * k + noise,
        amplitudes=np.ones_like(k),
    )

    raw = branch.compute_group_velocity(smooth=False)
    smoothed = branch.compute_group_velocity(smooth=True)
    expected = 2 * np.pi * true_slope

    assert not np.array_equal(raw, smoothed)
    assert np.mean(np.abs(smoothed - expected)) < np.mean(np.abs(raw - expected))


def test_dispersion_branch_rejects_nonmonotonic_or_mismatched_coordinates():
    from mmpp.fft.dispersion.models import DispersionBranch

    with pytest.raises(ValueError, match="matching lengths"):
        DispersionBranch(
            k_path=np.arange(3),
            f_values=np.arange(2),
            amplitudes=np.arange(3),
        )
    with pytest.raises(ValueError, match="strictly monotonic"):
        DispersionBranch(
            k_path=np.array([0.0, 1.0, 0.5]),
            f_values=np.arange(3),
            amplitudes=np.arange(3),
        )


def _lowest_frequency_test_result(raw, display=None):
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    raw = np.asarray(raw, dtype=float)
    shown = raw.copy() if display is None else np.asarray(display, dtype=float)
    return DispersionResult1D(
        S=shown,
        S_raw=raw,
        S_display=shown,
        k_axis=np.array([-1e6, 0.0, 1e6]),
        f_axis=np.array([0.0, 1e9, 2e9, 3e9]),
        axis="x",
        component="mz",
        config=DispersionConfig(),
    )


def test_find_lowest_uses_raw_spectrum_by_default():
    raw = np.zeros((3, 4))
    raw[1, 2] = 5.0
    raw[2, 1] = 4.0
    display = np.zeros_like(raw)
    display[1:, 3] = 10.0
    result = _lowest_frequency_test_result(raw, display)

    lowest = result.analyze.find_lowest_possible_frequency(
        smooth_sigma=None,
        min_snr=0.1,
    )

    assert lowest.f_min_hz == 1e9
    assert lowest.f_at_k0_hz == 2e9


def test_find_lowest_rejects_zero_spectrum_and_empty_snr_gate():
    zero = _lowest_frequency_test_result(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="no positive spectral power"):
        zero.analyze.find_lowest_possible_frequency(smooth_sigma=None)

    power = np.zeros((3, 4))
    power[0, 2] = 10.0
    power[1, 2] = 10.0
    power[2, 1] = 1.0
    weak_positive = _lowest_frequency_test_result(power)
    with pytest.raises(ValueError, match="passes the min_snr"):
        weak_positive.analyze.find_lowest_possible_frequency(
            smooth_sigma=None,
            min_snr=0.5,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"side": "right"}, "side must be"),
        ({"peak_method": "maximum"}, "peak_method"),
        ({"min_snr": -0.1}, "min_snr"),
        ({"smooth_sigma": -1.0}, "smooth_sigma"),
        ({"k_min_rad_um": 2.0, "k_max_rad_um": 1.0}, "k_max_rad_um"),
        ({"fmin_hz": "ten"}, "fmin_hz"),
    ],
)
def test_find_lowest_rejects_invalid_quantitative_options(kwargs, message):
    power = np.ones((3, 4))
    result = _lowest_frequency_test_result(power)
    with pytest.raises(ValueError, match=message):
        result.analyze.find_lowest_possible_frequency(**kwargs)


def test_branch_peak_detection_does_not_bypass_prominence_with_argmax():
    from mmpp.fft.dispersion._branch_linker import _find_peaks_column

    frequencies, amplitudes = _find_peaks_column(
        np.ones(9),
        np.arange(9, dtype=float),
        min_prominence_log=0.3,
        noise_floor=0.0,
    )

    assert frequencies.size == 0
    assert amplitudes.size == 0


def test_find_branches_rejects_zero_power_and_invalid_options():
    zero = _lowest_frequency_test_result(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="no positive spectral power"):
        zero.analyze.find_branches(min_branch_length=1)

    power = np.full((3, 4), 0.01)
    power[:, 2] = 10.0
    result = _lowest_frequency_test_result(power)
    invalid = [
        ({"n_branches": 0}, "n_branches"),
        ({"side": "right"}, "side must be"),
        ({"min_peak_distance": 0}, "min_peak_distance"),
        ({"max_df_ghz": 0.0}, "max_df_ghz"),
        ({"noise_floor_percentile": 101.0}, "noise_floor_percentile"),
        ({"min_quality": -0.1}, "min_quality"),
        ({"smooth_sigma": -1.0}, "smooth_sigma"),
    ]
    for kwargs, message in invalid:
        with pytest.raises(ValueError, match=message):
            result.analyze.find_branches(**kwargs)


def test_find_branches_sorts_descending_frequency_axis_with_spectrum():
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    ascending = np.full((3, 5), 0.01)
    ascending[:, 2] = 10.0
    descending = ascending[:, ::-1]
    result = DispersionResult1D(
        S=descending,
        k_axis=np.array([-1e6, 0.0, 1e6]),
        f_axis=np.array([4e9, 3e9, 2e9, 1e9, 0.0]),
        axis="x",
        component="mz",
        config=DispersionConfig(),
    )

    branches = result.analyze.find_branches(
        n_branches=1,
        min_peak_distance=1,
        min_branch_length=1,
        min_quality=0.0,
        smooth_sigma=None,
        noise_floor_percentile=0.0,
    )

    assert len(branches) == 1
    assert np.all(branches[0].f_hz == 2e9)


def test_branch_quality_smoothness_uses_physical_k_spacing():
    from mmpp.fft.dispersion._branch_linker import _branch_quality_metrics

    k = np.array([0.0, 1.0, 3.0, 6.0])
    metrics = _branch_quality_metrics(
        k,
        2.0 * k + 5.0,
        np.ones_like(k),
        reference_k_axis=k,
    )

    assert np.isclose(metrics["smoothness"], 1.0)


def test_branch_plot_unit_conversion_handles_cycles_per_meter_consistently():
    from mmpp.fft.dispersion._branch_linker import (
        TrackedBranch,
        _branch_plot_values,
    )

    branch = TrackedBranch(
        k=np.array([2 * np.pi, 4 * np.pi]),
        f_hz=np.array([1e9, 2e9]),
        amplitude=np.ones(2),
    )
    k_plot, f_plot, k_label, f_label = _branch_plot_values(
        branch,
        kscale="cycles_m",
        f_units="GHz",
    )

    assert np.allclose(k_plot, [1.0, 2.0])
    assert np.allclose(f_plot, [1.0, 2.0])
    assert "m" in k_label
    assert f_label == "f [GHz]"
    with pytest.raises(ValueError, match="kscale must be"):
        _branch_plot_values(branch, kscale="metres", f_units="GHz")


def test_interactive_dispersion_does_not_apply_hidden_default_k_crop():
    from types import SimpleNamespace

    from mmpp.fft.dispersion._interactive.rendering import _display_k_xlim

    explorer = SimpleNamespace(options={}, state=SimpleNamespace(kscale="rad_um"))
    assert _display_k_xlim(explorer, np.array([-50.0, 50.0])) is None

    explorer.options["k_xlim"] = (-25.0, 30.0)
    assert _display_k_xlim(explorer, np.array([-50.0, 50.0])) == (-25.0, 30.0)

    explorer.options["k_xlim"] = (5.0, -5.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        _display_k_xlim(explorer, np.array([-50.0, 50.0]))


def test_interactive_dispersion_rejects_unknown_k_scale():
    from mmpp.fft.dispersion._interactive.rendering import _scaled_k_axis

    with pytest.raises(ValueError, match="kscale must be"):
        _scaled_k_axis(np.arange(3), "micrometers")


def test_legacy_mode_peak_detection_preserves_frequency_axis_in_2d():
    from mmpp.fft.modes.utils.peak_detection import detect_peaks_simple

    frequencies = np.arange(7, dtype=float)
    trace = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0])
    spectrum = np.stack([trace, trace * 0.5], axis=1)

    peaks = detect_peaks_simple(
        spectrum,
        frequencies,
        threshold=0.1,
        min_distance=1,
    )

    assert [peak.idx for peak in peaks] == [2, 4]
    assert [peak.freq for peak in peaks] == [2.0, 4.0]


def test_simple_peak_fallback_honors_minimum_distance():
    from mmpp.fft.modes.utils.peak_detection import detect_peaks_simple

    spectrum = np.array([0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0])
    peaks = detect_peaks_simple(
        spectrum,
        np.arange(spectrum.size, dtype=float),
        threshold=0.0,
        min_distance=3,
    )

    assert [peak.idx for peak in peaks] == [1, 5]


def test_mode_peak_detection_rejects_ambiguous_or_invalid_data():
    from mmpp.fft.modes.models import Peak
    from mmpp.fft.modes.utils.peak_detection import detect_peaks_simple

    with pytest.raises(ValueError, match="Cannot identify"):
        detect_peaks_simple(np.ones((2, 3)), np.arange(5, dtype=float))
    with pytest.raises(ValueError, match="finite non-negative"):
        detect_peaks_simple(np.array([0.0, np.nan, 1.0]), np.arange(3, dtype=float))
    with pytest.raises(ValueError, match="finite and non-negative"):
        Peak(idx=0, freq=np.nan, amplitude=1.0)


def test_fmr_mode_crop_reports_actual_selected_cell_extent():
    from mmpp.fft.modes.models import FMRModeData

    mode = FMRModeData(
        frequency=2.0,
        mode_array=np.ones((4, 4, 3), dtype=complex),
        extent=(0.0, 40.0, 0.0, 40.0),
    )
    cropped = mode.crop_to_region((-5.0, 25.0), (15.0, 50.0))

    assert cropped.spatial_shape == (3, 3)
    assert cropped.extent == (0.0, 30.0, 10.0, 40.0)
    assert cropped.width_nm == 30.0
    assert cropped.height_nm == 30.0

    with pytest.raises(ValueError, match="does not overlap"):
        mode.crop_to_region((50.0, 60.0), (0.0, 10.0))


def test_fmr_mode_validates_geometry_grid_and_numpy_component_index():
    from mmpp.fft.modes.models import FMRModeData

    mode = FMRModeData(
        frequency=np.float64(1.0),
        mode_array=np.ones((2, 3, 3), dtype=complex),
        extent=np.array([0.0, 30.0, 0.0, 20.0]),
    )
    assert np.array_equal(mode.get_component(np.int64(1)), mode.mode_array[:, :, 1])

    with pytest.raises(ValueError, match="strictly increasing"):
        FMRModeData(1.0, np.ones((2, 2, 3)), extent=(0.0, 0.0, 0.0, 2.0))
    with pytest.raises(ValueError, match="positive integers"):
        mode.interpolate_to_grid((0, 3))
