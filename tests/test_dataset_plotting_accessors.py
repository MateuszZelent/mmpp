import types

import numpy as np

from mmpp.core.dataset import DatasetAwareWrapper
from mmpp.core.dataset_plotting_accessors import _DatasetMatplotlibPlotAccessor


class _DummyJob:
    def __init__(self, attrs=None):
        base = {"dx": 1e-9, "dy": 2e-9, "dz": 3e-9}
        if attrs is not None:
            base.update(dict(attrs))
        self.attrs = base
        self.name = "job"


def _wrapper(array: np.ndarray, attrs=None) -> DatasetAwareWrapper:
    return DatasetAwareWrapper(_DummyJob(attrs=attrs), "m", array.astype(np.float32))


def _quiver_scale(ax) -> float:
    for artist in getattr(ax, "collections", []):
        if artist.__class__.__name__ == "Quiver" and hasattr(artist, "scale"):
            return float(artist.scale)
    raise AssertionError("No quiver artist found on axis")


def test_mpl_accessor_injects_default_figsize_and_dpi():
    class _DummyParent:
        def __init__(self):
            self._dataset = types.SimpleNamespace(dataset_name="m")

        def _heatmap_impl(self, **kwargs):
            return kwargs

    accessor = _DatasetMatplotlibPlotAccessor(_DummyParent())
    kwargs = accessor.heatmap(component="mz")

    assert tuple(kwargs["figsize"]) == (8.0, 5.0)
    assert int(kwargs["dpi"]) == 100
    assert kwargs["component"] == "mz"


def test_downsample_block_mean_xy():
    arr = np.arange(1 * 1 * 4 * 4 * 1, dtype=np.float32).reshape(1, 1, 4, 4, 1)
    wrapped = _wrapper(arr)

    reduced = wrapped.downsample(":", 1, 2, 2, ":", strict=True).numpy(
        copy=False, squeeze=False
    )
    expected = np.array(
        [[[[[2.5], [4.5]], [[10.5], [12.5]]]]],
        dtype=np.float32,
    )

    assert reduced.shape == (1, 1, 2, 2, 1)
    assert np.allclose(reduced, expected)


def test_plot_mpl_extended_api():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    arr = np.random.default_rng(0).normal(size=(2, 1, 20, 30, 3)).astype(np.float32)
    frame = _wrapper(arr)[0, 0, :, :, :]

    ax_auto = frame.plot.mpl()
    ax_scalar = frame.plot.mpl.scalar(component="mz", figsize=(6, 3))
    ax_vector = frame.plot.mpl.vector(
        use_color=True,
        colorbar=True,
        colorbar_label="mz",
        vdims=("mx", "my"),
        scale=2.5,
    )
    ax_contour = frame.plot.mpl.contour(component="mx", figsize=(5, 3))
    ax_lightness = frame.plot.mpl.lightness(
        colorwheel=True,
        colorwheel_args={"loc": "upper left", "width": 1.2, "height": 1.2},
    )
    ax_lightness_alias = frame.plot.lightness(colorwheel=False)
    ax_scalar_um = frame.plot.mpl.scalar(component="mz", colorbar=False, multiplier=1e-6)
    ax_snapshot_um = frame.plot.snapshot(colorbar=False, multiplier=1e-6)

    fig_cb, ax_cb = plt.subplots()
    frame.plot.mpl.scalar(
        ax=ax_cb, component="mz", colorbar=True, colorbar_label="mz", multiplier=1e-9
    )
    frame.plot.mpl.vector(
        ax=ax_cb,
        vdims=("mx", "my"),
        use_color=True,
        colorbar=True,
        colorbar_label="oop",
        multiplier=1e-9,
    )

    assert hasattr(ax_auto, "imshow")
    assert hasattr(ax_scalar, "imshow")
    assert hasattr(ax_vector, "quiver")
    assert hasattr(ax_contour, "contour")
    assert hasattr(ax_lightness, "imshow")
    assert hasattr(ax_lightness_alias, "imshow")
    assert "um" in ax_scalar_um.get_xlabel()
    assert "um" in ax_snapshot_um.get_xlabel()
    assert "nm" in ax_cb.get_xlabel()
    assert len(fig_cb.axes) >= 3


def test_plot_mpl_default_dpi_and_figsize():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.close("all")

    arr = np.random.default_rng(99).normal(size=(2, 1, 18, 26, 3)).astype(np.float32)
    frame = _wrapper(arr)[0, 0, :, :, :]

    axes = [
        frame.plot.mpl(),
        frame.plot.mpl.scalar(colorbar=False),
        frame.plot.mpl.vector(use_color=False, colorbar=False),
        frame.plot.mpl.contour(component="mx", colorbar=False),
        frame.plot.mpl.lightness(colorwheel=False),
        frame.plot.snapshot(colorbar=False),
        frame.plot.heatmap(colorbar=False),
    ]
    fig = frame.plot.interactive(toolbar=False)
    anim = frame.plot.animate(save_path=None)
    anim._draw_was_started = True

    for ax in axes:
        assert np.isclose(ax.figure.get_dpi(), 100.0)
        assert np.allclose(ax.figure.get_size_inches(), [8.0, 5.0])
    assert np.isclose(fig.get_dpi(), 100.0)
    assert np.allclose(fig.get_size_inches(), [8.0, 5.0])
    assert np.isclose(anim._fig.get_dpi(), 100.0)
    assert np.allclose(anim._fig.get_size_inches(), [8.0, 5.0])

    ax_custom = frame.plot.snapshot(colorbar=False, figsize=(11, 4), dpi=150)
    fig_custom = frame.plot.interactive(toolbar=False, figsize=(9, 3), dpi=140)
    anim_custom = frame.plot.animate(save_path=None, figsize=(7, 2), dpi=130)
    anim_custom._draw_was_started = True

    assert np.allclose(ax_custom.figure.get_size_inches(), [11.0, 4.0])
    assert np.isclose(ax_custom.figure.get_dpi(), 150.0)
    assert np.allclose(fig_custom.get_size_inches(), [9.0, 3.0])
    assert np.isclose(fig_custom.get_dpi(), 140.0)
    assert np.allclose(anim_custom._fig.get_size_inches(), [7.0, 2.0])
    assert np.isclose(anim_custom._fig.get_dpi(), 130.0)

    plt.close("all")


def test_plot_mpl_vector_auto_color_warning_for_2comp():
    import matplotlib

    matplotlib.use("Agg", force=True)

    arr = np.random.default_rng(7).normal(size=(1, 1, 16, 20, 2)).astype(np.float32)
    frame = _wrapper(arr)[0, 0, :, :, :]

    import pytest

    with pytest.warns(RuntimeWarning, match="Automatic coloring is only supported"):
        ax = frame.plot.mpl.vector(use_color=True, colorbar=True)

    assert hasattr(ax, "quiver")


def test_plot_k3d_extended_api_with_stub(monkeypatch):
    class DummyObj:
        def __init__(self, name="", **kwargs):
            self.name = name
            self.kwargs = dict(kwargs)

    class DummyPlot:
        def __init__(self, name="plot"):
            self.name = name
            self.objects = []

        def __iadd__(self, obj):
            self.objects.append(obj)
            return self

        def __isub__(self, obj):
            if obj in self.objects:
                self.objects.remove(obj)
            return self

    fake_k3d = types.SimpleNamespace()
    fake_k3d.plot = lambda name="": DummyPlot(name=name)
    fake_k3d.voxels = lambda *a, **k: DummyObj(name=k.get("name", "voxels"), **k)
    fake_k3d.vectors = lambda *a, **k: DummyObj(name=k.get("name", "vectors"), **k)
    fake_k3d.points = lambda *a, **k: DummyObj(name=k.get("name", "points"), **k)
    fake_k3d.surface = lambda *a, **k: DummyObj(name=k.get("name", "surface"), **k)
    fake_k3d.colormaps = types.SimpleNamespace(
        matplotlib_color_maps=types.SimpleNamespace(Viridis=[0.0, 0.0, 0.0, 0.0])
    )
    monkeypatch.setitem(__import__("sys").modules, "k3d", fake_k3d)

    arr = np.random.default_rng(1).normal(size=(2, 4, 12, 16, 3)).astype(np.float32)
    wrapped = _wrapper(arr)
    frame = wrapped[0, 0, :, :, :]

    p_scalar = wrapped.plot.k3d.scalar(t=0, component="mz")
    p_nonzero = wrapped.plot.k3d.nonzero(t=0)
    p_vector = wrapped.plot.k3d.vector(t=0, quiver_density=4)
    p_heatmap = frame.plot.k3d.heatmap(component="mz")

    assert len(p_scalar.objects) == 1
    assert len(p_nonzero.objects) == 1
    assert len(p_vector.objects) >= 1
    assert len(p_heatmap.objects) == 1


def test_plot_k3d_scalar_thin_geometry_camera_and_bounds(monkeypatch):
    class DummyObj:
        def __init__(self, name="", **kwargs):
            self.name = name
            self.kwargs = dict(kwargs)

    class DummyPlot:
        def __init__(self, name="plot"):
            self.name = name
            self.objects = []
            self.camera_auto_fit = True
            self.camera = []

        def __iadd__(self, obj):
            self.objects.append(obj)
            return self

        def __isub__(self, obj):
            if obj in self.objects:
                self.objects.remove(obj)
            return self

    fake_k3d = types.SimpleNamespace()
    fake_k3d.plot = lambda name="": DummyPlot(name=name)
    fake_k3d.voxels = lambda *a, **k: DummyObj(name=k.get("name", "voxels"), **k)
    fake_k3d.vectors = lambda *a, **k: DummyObj(name=k.get("name", "vectors"), **k)
    fake_k3d.points = lambda *a, **k: DummyObj(name=k.get("name", "points"), **k)
    fake_k3d.surface = lambda *a, **k: DummyObj(name=k.get("name", "surface"), **k)
    fake_k3d.colormaps = types.SimpleNamespace(
        matplotlib_color_maps=types.SimpleNamespace(Viridis=[0.0, 0.0, 0.0, 0.0])
    )
    monkeypatch.setitem(__import__("sys").modules, "k3d", fake_k3d)

    arr = np.random.default_rng(12).normal(size=(1, 1, 8, 32, 3)).astype(np.float32)
    wrapped = _wrapper(arr)
    p_scalar = wrapped.plot.k3d.scalar(t=0, component="mz")

    assert len(p_scalar.objects) == 1
    vox = p_scalar.objects[0]
    assert "bounds" in vox.kwargs
    assert np.allclose(np.asarray(vox.kwargs["bounds"], dtype=np.float64), [0, 32, 0, 16, 0, 3])
    assert hasattr(p_scalar, "axes")
    assert len(p_scalar.axes) == 3
    assert getattr(p_scalar, "camera_auto_fit", None) is True
    assert getattr(p_scalar, "camera", None) == []


def test_plot_k3d_scalar_uses_slice_bounds_from_source_geometry(monkeypatch):
    class DummyObj:
        def __init__(self, name="", **kwargs):
            self.name = name
            self.kwargs = dict(kwargs)

    class DummyPlot:
        def __init__(self, name="plot"):
            self.name = name
            self.objects = []

        def __iadd__(self, obj):
            self.objects.append(obj)
            return self

        def __isub__(self, obj):
            if obj in self.objects:
                self.objects.remove(obj)
            return self

    fake_k3d = types.SimpleNamespace()
    fake_k3d.plot = lambda name="": DummyPlot(name=name)
    fake_k3d.voxels = lambda *a, **k: DummyObj(name=k.get("name", "voxels"), **k)
    fake_k3d.vectors = lambda *a, **k: DummyObj(name=k.get("name", "vectors"), **k)
    fake_k3d.points = lambda *a, **k: DummyObj(name=k.get("name", "points"), **k)
    fake_k3d.surface = lambda *a, **k: DummyObj(name=k.get("name", "surface"), **k)
    fake_k3d.colormaps = types.SimpleNamespace(
        matplotlib_color_maps=types.SimpleNamespace(Viridis=[0.0, 0.0, 0.0, 0.0])
    )
    monkeypatch.setitem(__import__("sys").modules, "k3d", fake_k3d)

    attrs = {"dx": 1.0, "dy": 2.0, "dz": 3.0, "xmin": 10.0, "ymin": 20.0, "zmin": 30.0}
    arr = np.ones((2, 4, 8, 10, 1), dtype=np.float32)
    sliced = _wrapper(arr, attrs=attrs)[0, 1:3, 2:6, 4:9, 0]
    p = sliced.plot.k3d.scalar()

    vox = p.objects[0]
    bounds = np.asarray(vox.kwargs["bounds"], dtype=np.float64)
    assert np.allclose(bounds, [14.0, 19.0, 24.0, 32.0, 33.0, 39.0])


def test_plot_hv_extended_api_with_stub(monkeypatch):
    class _Dimension:
        def __init__(self, name, values=None):
            self.name = name
            self.values = values

    class _Obj:
        def __init__(self, kind, payload=None, kdims=None, vdims=None):
            self.kind = kind
            self.payload = payload
            self.kdims = kdims or []
            self.vdims = vdims or []
            self.opts_kwargs = {}

        def opts(self, **kwargs):
            self.opts_kwargs.update(kwargs)
            return self

    class _DynamicMap(_Obj):
        def __init__(self, fn, kdims=None):
            super().__init__("DynamicMap", payload=fn, kdims=kdims)
            self.fn = fn

    def _extension(*args, **kwargs):
        _extension._loaded = True
        return None

    _extension._loaded = False
    fake_hv = types.SimpleNamespace(
        extension=_extension,
        Dimension=_Dimension,
        DynamicMap=_DynamicMap,
        Image=lambda *a, **k: _Obj("Image", payload=a, kdims=k.get("kdims"), vdims=k.get("vdims")),
        VectorField=lambda *a, **k: _Obj("VectorField", payload=a, kdims=k.get("kdims"), vdims=k.get("vdims")),
    )
    monkeypatch.setitem(__import__("sys").modules, "holoviews", fake_hv)

    arr = np.random.default_rng(2).normal(size=(3, 2, 18, 22, 3)).astype(np.float32)
    wrapped = _wrapper(arr)

    hv_scalar = wrapped.plot.hv.scalar(component="mz", dynamic=True)
    hv_vector = wrapped.plot.hv.vector(vdims=("mx", "my"), dynamic=True)
    hv_contour = wrapped.plot.hv.contour(component="mx", dynamic=False)

    assert getattr(hv_scalar, "kind", "") == "DynamicMap"
    assert getattr(hv_vector, "kind", "") == "DynamicMap"
    assert getattr(hv_contour, "kind", "") in {"Image", "DynamicMap"}


def test_plot_pyvista_extended_api_with_stub(monkeypatch):
    class _ImageData:
        def __init__(self, dimensions):
            self.dimensions = dimensions
            self.origin = (0, 0, 0)
            self.spacing = (1, 1, 1)
            self.cell_data = {}

    class _Glyph:
        pass

    class _PolyData:
        def __init__(self, points):
            self.points = points
            self.data = {}

        def __setitem__(self, key, value):
            self.data[key] = value

        def glyph(self, orient=None, scale=None, factor=1.0):
            return _Glyph()

    class _Plotter:
        def __init__(self):
            self.calls = []

        def add_volume(self, grid, scalars=None, **kwargs):
            self.calls.append(("add_volume", scalars, kwargs))
            return object()

        def add_mesh(self, mesh, scalars=None, **kwargs):
            self.calls.append(("add_mesh", scalars, kwargs))
            return object()

        def add_points(self, points, **kwargs):
            self.calls.append(("add_points", None, kwargs))
            return object()

        def show(self):
            self.calls.append(("show", None, {}))
            return None

    fake_pv = types.SimpleNamespace(
        ImageData=_ImageData,
        PolyData=_PolyData,
        Plotter=_Plotter,
    )
    monkeypatch.setitem(__import__("sys").modules, "pyvista", fake_pv)

    arr = np.random.default_rng(3).normal(size=(2, 4, 10, 12, 3)).astype(np.float32)
    wrapped = _wrapper(arr)

    p1 = wrapped.plot.pyvista.scalar(t=0, component="mz")
    p2 = wrapped.plot.pyvista.nonzero(t=0, threshold=0.1)
    p3 = wrapped.plot.pyvista.vector(t=0, vdims=("mx", "my", "mz"), quiver_density=4)

    assert any(call[0] == "add_volume" for call in p1.calls)
    assert any(call[0] == "add_volume" for call in p2.calls)
    assert any(call[0] == "add_mesh" for call in p3.calls)


def test_snapshot_quiver_scale_tracks_axis_multiplier():
    import matplotlib

    matplotlib.use("Agg", force=True)

    arr = np.random.default_rng(42).normal(size=(1, 1, 20, 28, 3)).astype(np.float32)
    frame = _wrapper(arr)[0, 0, :, :, :]

    ax_nm = frame.plot.snapshot(colorbar=False, multiplier=1e-9)
    ax_um = frame.plot.snapshot(colorbar=False, multiplier=1e-6)

    scale_nm = _quiver_scale(ax_nm)
    scale_um = _quiver_scale(ax_um)
    assert np.isclose(scale_um / scale_nm, 1e3, rtol=1e-6)


def test_mpl_vector_quiver_scale_tracks_axis_multiplier_and_respects_override():
    import matplotlib

    matplotlib.use("Agg", force=True)

    arr = np.random.default_rng(123).normal(size=(1, 1, 24, 24, 3)).astype(np.float32)
    frame = _wrapper(arr)[0, 0, :, :, :]

    ax_nm = frame.plot.mpl.vector(use_color=False, colorbar=False, multiplier=1e-9)
    ax_um = frame.plot.mpl.vector(use_color=False, colorbar=False, multiplier=1e-6)
    ax_fixed = frame.plot.mpl.vector(
        use_color=False,
        colorbar=False,
        multiplier=1e-6,
        scale=2.5,
    )

    scale_nm = _quiver_scale(ax_nm)
    scale_um = _quiver_scale(ax_um)
    scale_fixed = _quiver_scale(ax_fixed)

    assert np.isclose(scale_um / scale_nm, 1e3, rtol=1e-6)
    assert np.isclose(scale_fixed, 2.5, rtol=0.0, atol=0.0)
