"""Backend namespace accessors for dataset plotting."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from mmpp._repr_helpers import (
    NODE_COLOR_ADVANCED,
    NODE_COLOR_ANALYSIS,
    NODE_COLOR_COMPUTE,
    NODE_COLOR_PLOT,
    api_help_html,
    examples_section_html,
    metrics_section_html,
    node_card_html,
)

if TYPE_CHECKING:
    from .dataset_plotting import DatasetPlotAccessor


def _backend_methods_section(methods) -> str:
    rows = "".join(
        "<tr style='border-top:1px solid rgba(98,114,164,0.2);'>"
        f"<td title='{tip}' style='padding:5px 8px;font-family:monospace;color:#8be9fd;"
        f"font-size:0.9em;white-space:nowrap;vertical-align:top;'>{sig}</td>"
        f"<td style='padding:5px 8px;color:#f8f8f2;font-size:0.88em;'>{desc}</td>"
        "</tr>"
        for sig, desc, tip in methods
    )
    return (
        "<div style='background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%);"
        "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(98,114,164,0.35);'>"
        "<b style='color:#bd93f9;'>Backend methods</b><br>"
        "<table style='width:100%;border-collapse:collapse;margin-top:6px;'>"
        "<thead><tr style='text-align:left;background:rgba(68,71,90,0.4);'>"
        "<th style='padding:4px 8px;color:#f8f8f2;'>Method</th>"
        "<th style='padding:4px 8px;color:#f8f8f2;'>Description</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<div style='margin-top:8px;font-size:0.8em;color:#6272a4;'>"
        "Hover the method signature cell for parameter hints."
        "</div></div>"
    )


def _tabbed_backend_help(
    obj,
    *,
    title: str,
    prefix: str,
    methods,
    overview_methods,
    summary: str,
    badge: tuple[str, str] | None = ("ready", "#50fa7b"),
    example_lines: list[str] | None = None,
) -> str:
    api = api_help_html(
        obj,
        title=f"{title} API help",
        prefix=prefix,
        methods=methods,
        subtitle="Live public plotting API with signatures and generated examples.",
        chrome=False,
    )
    sections = [
        metrics_section_html(
            [
                ("prefix", prefix, NODE_COLOR_COMPUTE),
                ("backend", title.split(" ", 1)[0], NODE_COLOR_ANALYSIS),
                ("default", "__call__(**kwargs)", NODE_COLOR_PLOT),
            ]
        ),
        _backend_methods_section(overview_methods),
    ]
    if example_lines:
        sections.append(examples_section_html("\n".join(example_lines)))
    return node_card_html(
        title,
        icon="🎨",
        subtitle=summary,
        badge=badge,
        sections=sections,
        api=api,
        uid=f"mmpp-dataset-plot-{uuid.uuid4().hex}",
    )


class _DatasetMatplotlibPlotAccessor:
    """Matplotlib backend namespace for dataset-aware plotting."""

    _DEFAULT_MPL_FIGSIZE = (8.0, 5.0)
    _DEFAULT_MPL_DPI = 100

    def __init__(self, parent: DatasetPlotAccessor):
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
        overview_methods = [
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
            (".vector(**kw)", "Vector field quiver plot", "step, scale, color, alpha."),
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
        ]
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
            overview_methods=overview_methods,
            summary="2-D plotting backend with publication-style Matplotlib defaults for snapshots, scalar maps, quivers and mixed magnetization views.",
            example_lines=[
                "mpl = job[0].m.plot.mpl",
                "mpl.snapshot(z=0, t=-1)",
                "mpl.magnetization(scalar_component='mz', step=3)",
            ],
        )


class _DatasetK3DPlotAccessor:
    """K3D backend namespace for dataset-aware plotting."""

    @staticmethod
    def _with_k3d_defaults(kwargs: dict) -> dict:
        merged = dict(kwargs)
        # Compatibility shim: some in-memory/stale class versions require explicit multiplier.
        merged.setdefault("multiplier", None)
        return merged

    def __init__(self, parent: DatasetPlotAccessor):
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
        overview_methods = [
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
                "style='hsl'|'mz'|'norm', show_vectors, voxel_opacity, quiver_density, color_field/filter_field can be (wrapper, 'mz').",
            ),
            (
                ".stack(**kw)",
                "Overlay multiple physical slices on one scene",
                "axis, positions, mode='magnetization'|'vector'|'scalar', slice_kwargs.",
            ),
            (".nonzero(**kw)", "Plot non-zero voxels", "threshold, color."),
            (".heatmap(**kw)", "3-D heatmap", "component, cmap."),
        ]
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
            overview_methods=overview_methods,
            summary="Interactive 3-D backend for voxel, vector and magnetization rendering, including combined volume and arrow views.",
            badge=("3D", NODE_COLOR_ANALYSIS),
            example_lines=[
                "k3d = job[0].m.plot.k3d",
                "k3d.magnetization(style='hsl', show_vectors=True)",
                "k3d.stack(axis='z', positions=[0.0, 10e-9])",
            ],
        )


class _DatasetHVPlotAccessor:
    """Holoviews backend namespace for dataset-aware plotting."""

    def __init__(self, parent: DatasetPlotAccessor):
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
        overview_methods = [
            (".scalar(**kw)", "Interactive scalar heatmap", "component, cmap."),
            (".vector(**kw)", "Interactive vector field", "step, scale."),
            (".contour(**kw)", "Interactive contour plot", "component, levels."),
        ]
        return _tabbed_backend_help(
            self,
            title="Holoviews Plot Backend",
            prefix="job[0].m.plt.hv",
            methods=["scalar", "vector", "contour"],
            overview_methods=overview_methods,
            summary="Interactive backend for lightweight exploratory scalar, vector and contour views.",
            badge=("interactive", NODE_COLOR_PLOT),
            example_lines=[
                "hv = job[0].m.plot.hv",
                "hv.scalar(component='mz')",
                "hv.vector(step=4)",
            ],
        )


class _DatasetPyVistaPlotAccessor:
    """PyVista backend namespace for dataset-aware plotting."""

    def __init__(self, parent: DatasetPlotAccessor):
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
        overview_methods = [
            (
                ".scalar(**kw)",
                "3-D scalar volume rendering",
                "component, cmap, opacity.",
            ),
            (".vector(**kw)", "3-D vector glyphs", "step, scale, color."),
            (".nonzero(**kw)", "Plot non-zero cells", "threshold, color."),
        ]
        return _tabbed_backend_help(
            self,
            title="PyVista Plot Backend",
            prefix="job[0].m.plt.pyvista",
            methods=["scalar", "vector", "nonzero"],
            overview_methods=overview_methods,
            summary="PyVista backend for 3-D scalar volumes, vector glyphs and sparse non-zero region inspection.",
            badge=("3D", NODE_COLOR_ADVANCED),
            example_lines=[
                "pv = job[0].m.plot.pyvista",
                "pv.scalar(component='mz', opacity=0.35)",
                "pv.vector(step=3)",
            ],
        )
