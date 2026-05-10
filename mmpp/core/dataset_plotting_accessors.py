"""Backend namespace accessors for dataset plotting."""

from __future__ import annotations

import uuid

from mmpp._repr_helpers import api_help_html, html_tabs, plot_accessor_html


def _tabbed_backend_help(obj, *, title: str, prefix: str, methods, overview: str) -> str:
    api = api_help_html(
        obj,
        title=f"{title} API help",
        prefix=prefix,
        methods=methods,
        subtitle="Live public plotting API with signatures and generated examples.",
        chrome=False,
    )
    return (
        "<div style='font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;"
        "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
        "background:#0f172a;color:#e2e8f0;'>"
        + html_tabs(
            [("Overview", overview), ("API", api)],
            uid=f"mmpp-dataset-plot-{uuid.uuid4().hex}",
        )
        + "</div>"
    )


class _DatasetMatplotlibPlotAccessor:
    """Matplotlib backend namespace for dataset-aware plotting."""

    _DEFAULT_MPL_FIGSIZE = (8.0, 5.0)
    _DEFAULT_MPL_DPI = 100

    def __init__(self, parent: "DatasetPlotAccessor"):
        self._parent = parent

    @classmethod
    def _with_mpl_defaults(cls, kwargs: dict) -> dict:
        merged = dict(kwargs)
        merged.setdefault("figsize", cls._DEFAULT_MPL_FIGSIZE)
        merged.setdefault("dpi", cls._DEFAULT_MPL_DPI)
        return merged

    def __call__(self, **kwargs):
        return self._parent._mpl_auto_impl(**self._with_mpl_defaults(kwargs))

    def scalar(self, **kwargs):
        return self._parent._mpl_scalar_impl(**self._with_mpl_defaults(kwargs))

    def vector(self, **kwargs):
        return self._parent._mpl_vector_impl(**self._with_mpl_defaults(kwargs))

    def contour(self, **kwargs):
        return self._parent._mpl_contour_impl(**self._with_mpl_defaults(kwargs))

    def lightness(self, **kwargs):
        return self._parent._mpl_lightness_impl(**self._with_mpl_defaults(kwargs))

    def magnetization(self, **kwargs):
        return self._parent._mpl_magnetization_impl(**self._with_mpl_defaults(kwargs))

    def snapshot(self, **kwargs):
        return self._parent._snapshot_impl(**self._with_mpl_defaults(kwargs))

    def heatmap(self, **kwargs):
        return self._parent._heatmap_impl(**self._with_mpl_defaults(kwargs))

    def heamtp(self, **kwargs):
        """Compatibility alias for a common typo: ``heatmap``."""
        return self.heatmap(**kwargs)

    def __repr__(self):
        dset = self._parent._dataset.dataset_name
        return f"<DatasetMplPlotAccessor('{dset}')>"

    def _repr_html_(self) -> str:
        overview = plot_accessor_html(
            "Matplotlib Plot Backend",
            [
                (
                    ".snapshot(z=0, t=-1, figsize=(8, 5), dpi=100)",
                    "HSL colour-wheel snapshot (vector) or heatmap (scalar)",
                    "z, t, repeat, zero, cmap, component, figsize, dpi.",
                ),
                (
                    ".scalar(**kw)",
                    "Scalar component heatmap",
                    "component, cmap, vmin/vmax, colorbar.",
                ),
                (
                    ".vector(**kw)",
                    "Vector field quiver plot",
                    "step, scale, color, alpha.",
                ),
                (
                    ".magnetization(**kw)",
                    "Micromagnetic 2-D view (scalar + quiver)",
                    "scalar_component='mz', vector_vdims=('mx','my'), filter_field='norm', cell_grid.",
                ),
                (
                    ".contour(**kw)",
                    "Contour plot of scalar component",
                    "component, levels, cmap, filled.",
                ),
                (
                    ".lightness(**kw)",
                    "Lightness-based mz visualisation",
                    "Renders mz as lightness.",
                ),
                (
                    ".heatmap(**kw)",
                    "2-D component heatmap over time",
                    "component, cmap, vmin/vmax, aspect.",
                ),
            ],
        )
        return _tabbed_backend_help(
            self,
            title="Matplotlib Plot Backend",
            prefix="job[0].m.plt.mpl",
            methods=[
                "snapshot",
                "scalar",
                "vector",
                "contour",
                "lightness",
                "magnetization",
                "heatmap",
                "heamtp",
            ],
            overview=overview,
        )

class _DatasetK3DPlotAccessor:
    """K3D backend namespace for dataset-aware plotting."""

    @staticmethod
    def _with_k3d_defaults(kwargs: dict) -> dict:
        merged = dict(kwargs)
        # Compatibility shim: some in-memory/stale class versions require explicit multiplier.
        merged.setdefault("multiplier", None)
        return merged

    def __init__(self, parent: "DatasetPlotAccessor"):
        self._parent = parent

    def __call__(self, **kwargs):
        return self.scalar(**kwargs)

    def scalar(self, **kwargs):
        return self._parent._k3d_scalar_impl(**self._with_k3d_defaults(kwargs))

    def vector(self, **kwargs):
        return self._parent._k3d_vector_impl(**self._with_k3d_defaults(kwargs))

    def nonzero(self, **kwargs):
        return self._parent._k3d_nonzero_impl(**self._with_k3d_defaults(kwargs))

    def heatmap(self, **kwargs):
        return self._parent._k3d_heatmap_impl(**kwargs)

    def voxels_vectors(self, **kwargs):
        """3-D voxels coloured by scalar with arrows overlaid inside."""
        return self._parent._k3d_voxels_vectors_impl(**self._with_k3d_defaults(kwargs))

    def magnetization(self, **kwargs):
        """Micromagnetic 3-D view: voxel colouring + optional vector overlay."""
        return self._parent._k3d_magnetization_impl(**self._with_k3d_defaults(kwargs))

    def stack(self, **kwargs):
        """Overlay multiple physical slices on one k3d plot."""
        return self._parent._k3d_stack_impl(**self._with_k3d_defaults(kwargs))

    def __repr__(self):
        dset = self._parent._dataset.dataset_name
        return f"<DatasetK3DPlotAccessor('{dset}')>"

    def _repr_html_(self) -> str:
        overview = plot_accessor_html(
            "K3D Plot Backend",
            [
                (
                    ".scalar(**kw)",
                    "3-D scalar voxel plot",
                    "component, cmap, opacity, hide_zeros, grid_from_centers.",
                ),
                (".vector(**kw)", "3-D vector field (arrows)", "step, scale, color."),
                (
                    ".voxels_vectors(**kw)",
                    "Voxels + arrows - combined 3-D view",
                    "scalar_component, cmap, voxel_opacity, quiver_density, vector_scale.",
                ),
                (
                    ".magnetization(**kw)",
                    "Micromagnetic default 3-D view",
                    "style='hsl'|'mz'|'norm', show_vectors, voxel_opacity, quiver_density, "
                    "color_field/filter_field can be (wrapper, 'mz').",
                ),
                (
                    ".stack(**kw)",
                    "Overlay multiple physical slices on one scene",
                    "axis, positions, mode='magnetization'|'vector'|'scalar', slice_kwargs.",
                ),
                (".nonzero(**kw)", "Plot non-zero voxels", "threshold, color."),
                (".heatmap(**kw)", "3-D heatmap", "component, cmap."),
            ],
            accent="#059669",
            title_color="#34d399",
        )
        return _tabbed_backend_help(
            self,
            title="K3D Plot Backend",
            prefix="job[0].m.plt.k3d",
            methods=[
                "scalar",
                "vector",
                "nonzero",
                "heatmap",
                "voxels_vectors",
                "magnetization",
                "stack",
            ],
            overview=overview,
        )

class _DatasetHVPlotAccessor:
    """Holoviews backend namespace for dataset-aware plotting."""

    def __init__(self, parent: "DatasetPlotAccessor"):
        self._parent = parent

    def __call__(self, **kwargs):
        return self.scalar(**kwargs)

    def scalar(self, **kwargs):
        return self._parent._hv_scalar_impl(**kwargs)

    def vector(self, **kwargs):
        return self._parent._hv_vector_impl(**kwargs)

    def contour(self, **kwargs):
        return self._parent._hv_contour_impl(**kwargs)

    def __repr__(self):
        dset = self._parent._dataset.dataset_name
        return f"<DatasetHVPlotAccessor('{dset}')>"

    def _repr_html_(self) -> str:
        overview = plot_accessor_html(
            "Holoviews Plot Backend",
            [
                (".scalar(**kw)", "Interactive scalar heatmap", "component, cmap."),
                (".vector(**kw)", "Interactive vector field", "step, scale."),
                (".contour(**kw)", "Interactive contour plot", "component, levels."),
            ],
            accent="#7c3aed",
            title_color="#a78bfa",
        )
        return _tabbed_backend_help(
            self,
            title="Holoviews Plot Backend",
            prefix="job[0].m.plt.hv",
            methods=["scalar", "vector", "contour"],
            overview=overview,
        )

class _DatasetPyVistaPlotAccessor:
    """PyVista backend namespace for dataset-aware plotting."""

    def __init__(self, parent: "DatasetPlotAccessor"):
        self._parent = parent

    def __call__(self, **kwargs):
        return self.scalar(**kwargs)

    def scalar(self, **kwargs):
        return self._parent._pyvista_scalar_impl(**kwargs)

    def vector(self, **kwargs):
        return self._parent._pyvista_vector_impl(**kwargs)

    def nonzero(self, **kwargs):
        return self._parent._pyvista_nonzero_impl(**kwargs)

    def __repr__(self):
        dset = self._parent._dataset.dataset_name
        return f"<DatasetPyVistaPlotAccessor('{dset}')>"

    def _repr_html_(self) -> str:
        overview = plot_accessor_html(
            "PyVista Plot Backend",
            [
                (
                    ".scalar(**kw)",
                    "3-D scalar volume rendering",
                    "component, cmap, opacity.",
                ),
                (".vector(**kw)", "3-D vector glyphs", "step, scale, color."),
                (".nonzero(**kw)", "Plot non-zero cells", "threshold, color."),
            ],
            accent="#b91c1c",
            title_color="#fca5a5",
        )
        return _tabbed_backend_help(
            self,
            title="PyVista Plot Backend",
            prefix="job[0].m.plt.pyvista",
            methods=["scalar", "vector", "nonzero"],
            overview=overview,
        )
