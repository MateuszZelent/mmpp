"""Facade class combining dataset plotting backends."""

from __future__ import annotations

from .dataset_plotting_accessors import (
    _DatasetHVPlotAccessor,
    _DatasetK3DPlotAccessor,
    _DatasetMatplotlibPlotAccessor,
    _DatasetPyVistaPlotAccessor,
)
from .dataset_plotting_core import DatasetPlotCoreMixin
from .dataset_plotting_hv import DatasetPlotHVMixin
from .dataset_plotting_k3d import DatasetPlotK3DMixin
from .dataset_plotting_mpl import DatasetPlotMplMixin
from .dataset_plotting_pyvista import DatasetPlotPyVistaMixin

class DatasetPlotAccessor(
    DatasetPlotCoreMixin,
    DatasetPlotMplMixin,
    DatasetPlotK3DMixin,
    DatasetPlotHVMixin,
    DatasetPlotPyVistaMixin,
):
    def __init__(self, dataset_wrapper: "DatasetAwareWrapper"):
        self._dataset = dataset_wrapper
        self._mpl = None
        self._k3d = None
        self._hv = None
        self._pyvista = None

    @property
    def mpl(self):
        if self._mpl is None:
            self._mpl = _DatasetMatplotlibPlotAccessor(self)
        return self._mpl

    @property
    def k3d(self):
        if self._k3d is None:
            self._k3d = _DatasetK3DPlotAccessor(self)
        return self._k3d

    @property
    def hv(self):
        if self._hv is None:
            self._hv = _DatasetHVPlotAccessor(self)
        return self._hv

    @property
    def pyvista(self):
        if self._pyvista is None:
            self._pyvista = _DatasetPyVistaPlotAccessor(self)
        return self._pyvista

    def snapshot(self, **kwargs):
        """Convenience alias for ``plot.mpl.snapshot(...)``."""
        return self.mpl.snapshot(**kwargs)

    def scalar(self, **kwargs):
        """Convenience alias for ``plot.mpl.scalar(...)``."""
        return self.mpl.scalar(**kwargs)

    def vector(self, **kwargs):
        """Convenience alias for ``plot.mpl.vector(...)``."""
        return self.mpl.vector(**kwargs)

    def contour(self, **kwargs):
        """Convenience alias for ``plot.mpl.contour(...)``."""
        return self.mpl.contour(**kwargs)

    def lightness(self, **kwargs):
        """Convenience alias for ``plot.mpl.lightness(...)``."""
        return self.mpl.lightness(**kwargs)

    def magnetization(self, **kwargs):
        """Convenience alias for ``plot.mpl.magnetization(...)``."""
        return self.mpl.magnetization(**kwargs)

    def heatmap(self, **kwargs):
        """Convenience alias for ``plot.mpl.heatmap(...)``."""
        return self.mpl.heatmap(**kwargs)

    def heamtp(self, **kwargs):
        """Compatibility alias for ``heatmap``."""
        return self.mpl.heatmap(**kwargs)

    def __repr__(self):
        dset = self._dataset.dataset_name
        return (
            f"<DatasetPlotAccessor('{dset}'): .snapshot(), .scalar(), .vector(), .magnetization(), "
            ".contour(), .lightness(), .heatmap(), .interactive(), .animate(), "
            ".mpl, .k3d, .hv, .pyvista>"
        )

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        dset = self._dataset.dataset_name
        return plot_accessor_html(f"DatasetPlotAccessor — {dset}", [
            (".snapshot(z=0, t=-1, repeat=1, figsize=(8, 5), dpi=100)",
             "HSL colour-wheel snapshot (vector) or heatmap (scalar)",
             "z: z-slice, t: timestep, repeat: tile count, zero: reference timestep for difference, "
             "cmap: colormap for scalar, component: int to force scalar, figsize: figure size, "
             "dpi: figure resolution."),
            (".scalar(**kw)",
             "Scalar component heatmap (delegates to .mpl.scalar)",
             "component: 'x'|'y'|'z'|'norm'|int, cmap, vmin/vmax, colorbar."),
            (".vector(**kw)",
             "Vector field quiver plot (delegates to .mpl.vector)",
             "step: arrow spacing, scale: arrow size, color, alpha."),
            (".magnetization(**kw)",
             "Micromagnetic 2-D view (delegates to .mpl.magnetization)",
             "scalar_component='mz', vector_vdims=('mx','my'), filter_field='norm', cell_grid."),
            (".contour(**kw)",
             "Contour plot of scalar component",
             "component, levels, cmap, filled."),
            (".lightness(**kw)",
             "Lightness-based visualisation of out-of-plane component",
             "Renders mz as lightness on HSL colour wheel."),
            (".heatmap(**kw)",
             "2-D heatmap of selected component over time",
             "component, cmap, vmin/vmax, aspect."),
            (".interactive(remove_static=True, static_reference=0, **kw)",
             "Notebook-friendly time browser with slider, play button, and optional static subtraction",
             "mode='snapshot'|'heatmap', component, z, fps, toolbar, remove_static subtracts m[static_reference]."),
            (".k3d.magnetization(**kw)",
             "Micromagnetic 3-D view with voxel colouring and vectors",
             "style='hsl'|'mz'|'norm', show_vectors, voxel_opacity, quiver_density, "
             "color_field/filter_field can be (wrapper, 'mz')."),
            (".k3d.stack(**kw)",
             "Overlay multiple physical slices on one k3d scene",
             "axis, positions, mode='magnetization'|'vector'|'scalar', slice_kwargs."),
        ], footer=(
            "Backends: <code style='color:#bae6fd;'>.mpl</code>, "
            "<code style='color:#bae6fd;'>.k3d</code>, "
            "<code style='color:#bae6fd;'>.hv</code>, "
            "<code style='color:#bae6fd;'>.pyvista</code>  ·  "
            "All methods accept <code style='color:#bae6fd;'>**kwargs</code> forwarded to the backend."
        ))

__all__ = ["DatasetPlotAccessor"]
