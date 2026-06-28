"""
Fast unit tests for dispersion mode extraction helpers.

These tests are synthetic and do not require zarr/job infrastructure.
"""

import os
import sys
import types
import logging
import json

import numpy as np
import pytest
import zarr

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


def test_dispersion_interactive_viewer_can_save_load_preset_and_export_selection(tmp_path):
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
    assert {"tabs", "status_log", "preset_select", "output"}.issubset(
        viewer._controls
    )
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
    assert "DispersionInteractiveViewer: mode-ready" in viewer_html
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
    assert raw_branches.branches[0].quality_metrics["confidence"] == raw_branches.branches[0].quality


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
    phase = 2.0 * np.pi * (
        (f_bin * t / n_t) + (ky_bin * y / n_y) + (kx_bin * x / n_x)
    )
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

    no_window_config = DispersionConfig(time_window=None, space_window=None, detrend=None)
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
    expected_k_axis = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(effective_n_y, effective_dy))

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
    assert np.allclose(numpy_result.S_display, scipy_result.S_display, rtol=1e-6, atol=1e-9)
    assert numpy_result.S_complex is not None
    assert scipy_result.S_complex is not None
    complex_scale = max(float(np.abs(numpy_result.S_complex).max()), 1.0)
    complex_delta = float(np.max(np.abs(numpy_result.S_complex - scipy_result.S_complex)))
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
    complex_delta = float(np.max(np.abs(numpy_result.S_complex - pyfftw_result.S_complex)))
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

    result = analyzer.compute_dispersion_2d(component="perp", time_window=None, detrend=None)
    idx_kx, idx_ky, idx_f = np.unravel_index(int(np.argmax(result.S)), result.S.shape)
    slice_kx = result.slice_1d("kx", k_value=float(result.ky_axis[idx_ky]), dk_max=0.0)

    assert result.shape == (n_x, n_y, n_t // 2 + 1)
    assert np.allclose(result.kx_axis, np.fft.fftshift(2 * np.pi * np.fft.fftfreq(n_x, dx)))
    assert np.allclose(result.ky_axis, np.fft.fftshift(2 * np.pi * np.fft.fftfreq(n_y, dy)))
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

    from mmpp.fft.dispersion.modes import interactive as interactive_module
    from mmpp.fft.dispersion.modes.interactive import InteractiveDispersionModes
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


def test_interactive_dispersion_modes_close_cleans_display_animation_and_figure(
    monkeypatch,
):
    from types import SimpleNamespace

    from mmpp.fft.dispersion.modes import interactive as interactive_module
    from mmpp.fft.dispersion.modes.interactive import InteractiveDispersionModes
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
    modes = InteractiveDispersionModes(SimpleNamespace(compute_1d=lambda **_: result))

    display_updates = []
    stopped = []
    closed = []
    fig = object()

    modes._display_handle = SimpleNamespace(update=lambda value: display_updates.append(value))
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
    iface._tmax = 100
    calls = []

    def fake_compute_1d(**kwargs):
        calls.append((dict(kwargs), iface._tmax, iface._cache_dir))
        return result

    iface.compute_1d = fake_compute_1d

    viewer = iface.plot.interactive(
        show=False,
        axis="x",
        tmax=800,
        cache="/tmp/mmpp-dispersion",
        save=True,
        fmax=4.0,
    )

    assert viewer.result is result
    assert calls == [
        (
            {"axis": "x", "save": True, "store_complex": False, "disk_cache": True},
            800,
            "/tmp/mmpp-dispersion",
        )
    ]
    assert iface._tmax == 100
    assert iface._cache_dir is None


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

    viewer = iface.plot.interactive(show=False, axis="x", analitical="DE")

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

    iface.plot.interactive(show=False, axis="x")
    iface.plot.interactive(show=False, axis="x", store_complex=True)

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

    viewer = iface.interactive_analysis(show=False, axis="x", fmax=4.0)

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


def test_spin_wave_analyzer_accepts_nonuniform_time_axis_with_effective_dt(tmp_path):
    from types import SimpleNamespace

    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.interface import FFTDispersionInterface
    from mmpp.fft.dispersion.models import DispersionConfig

    zarr_path = tmp_path / "dispersion_nonuniform_time_axis.zarr"
    data = np.zeros((4, 1, 1, 4, 3), dtype=np.float32)
    root = zarr.open(str(zarr_path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root["m"].attrs["t"] = [0.0, 1.2e-12, 2.8e-12, 4.5e-12]
    root.attrs["t_sampl"] = 1e-12
    root.attrs["dx"] = 1e-9
    root.attrs["dy"] = 1e-9

    analyzer = SpinWaveAnalyzer(
        zarr_path,
        config=DispersionConfig(dt=9e-12),
        tmax=None,
    )

    assert np.isclose(analyzer.dt, 1.5e-12)
    assert np.isclose(analyzer.config.dt, 1.5e-12)
    assert np.allclose(analyzer.time_axis, [0.0, 1.2e-12, 2.8e-12, 4.5e-12])

    result = analyzer.compute_dispersion_1d(axis="x", component="mx")
    assert any("Non-uniform time axis" in note for note in result.notes)
    assert any("declared t_sampl=1e-12" in note for note in result.notes)

    iface = FFTDispersionInterface(
        SimpleNamespace(job_result=SimpleNamespace(path=str(zarr_path), name="run")),
        dataset_name="m",
    )
    viewer = iface.plot.interactive(show=False, axis="x", component="mx", disk_cache=False)
    assert viewer.state["show"] is False
    assert any("Non-uniform time axis" in note for note in viewer.state["result_notes"])


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
