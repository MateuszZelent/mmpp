import types

import numpy as np

from mmpp.core.dataset import DatasetAwareWrapper


class _DummyJob:
    def __init__(self):
        self.attrs = {"dx": 1e-9, "dy": 2e-9, "dz": 3e-9}
        self.name = "job"


def _wrapper(array: np.ndarray) -> DatasetAwareWrapper:
    return DatasetAwareWrapper(_DummyJob(), "m", array.astype(np.float32))


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
        def __init__(self, name=""):
            self.name = name

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
    fake_k3d.voxels = lambda *a, **k: DummyObj(name=k.get("name", "voxels"))
    fake_k3d.vectors = lambda *a, **k: DummyObj(name=k.get("name", "vectors"))
    fake_k3d.points = lambda *a, **k: DummyObj(name=k.get("name", "points"))
    fake_k3d.surface = lambda *a, **k: DummyObj(name=k.get("name", "surface"))
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
