"""PyVista mixin for dataset plotting."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np

class DatasetPlotPyVistaMixin:
    @staticmethod
    def _pyvista_import():
        try:
            import pyvista as pv
        except Exception as exc:
            raise ImportError(
                "pyvista is required for plot.pyvista.*(). Install with: pip install pyvista"
            ) from exc
        return pv

    def _pyvista_scalar_impl(
        self,
        *,
        plotter=None,
        t: int = -1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        multiplier: Optional[float] = None,
        cmap: str = "viridis",
        opacity: Union[str, float, list[float]] = "linear",
        show: bool = False,
        name: str = "value",
        **kwargs,
    ):
        pv = self._pyvista_import()
        volume = self._extract_volume(t=t, zero=zero)
        scalar = self._component_volume(volume, component, default="norm")
        if scalar.ndim != 3:
            raise ValueError(
                "pyvista.scalar expects a 3d scalar volume. "
                f"Got shape {scalar.shape}."
            )

        nz, ny, nx = scalar.shape
        attrs = getattr(self._dataset.job_result, "attrs", {})
        if hasattr(attrs, "get"):
            dx = float(attrs.get("dx", 1e-9))
            dy = float(attrs.get("dy", 1e-9))
            dz = float(attrs.get("dz", 1e-9))
        else:
            dx = dy = dz = 1e-9
        m = 1.0 if multiplier is None else float(multiplier)
        spacing = (dx / m, dy / m, dz / m)

        grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
        grid.origin = (0.0, 0.0, 0.0)
        grid.spacing = spacing
        values = np.transpose(scalar, (2, 1, 0)).ravel(order="F")
        try:
            grid.cell_data[name] = values
        except Exception:
            setattr(grid, "cell_data", {name: values})

        p = plotter if plotter is not None else pv.Plotter()
        add_kwargs = dict(kwargs)
        add_kwargs.setdefault("cmap", cmap)
        add_kwargs.setdefault("opacity", opacity)
        p.add_volume(grid, scalars=name, **add_kwargs)
        if show and plotter is None:
            p.show()
        return p

    def _pyvista_nonzero_impl(
        self,
        *,
        plotter=None,
        t: int = -1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        threshold: float = 0.0,
        multiplier: Optional[float] = None,
        show: bool = False,
        **kwargs,
    ):
        pv = self._pyvista_import()
        volume = self._extract_volume(t=t, zero=zero)
        scalar = self._component_volume(volume, component, default="norm")
        if scalar.ndim != 3:
            raise ValueError(
                "pyvista.nonzero expects a 3d scalar volume. "
                f"Got shape {scalar.shape}."
            )

        mask = (np.abs(scalar) > float(threshold)).astype(np.float32)
        nz, ny, nx = mask.shape
        attrs = getattr(self._dataset.job_result, "attrs", {})
        if hasattr(attrs, "get"):
            dx = float(attrs.get("dx", 1e-9))
            dy = float(attrs.get("dy", 1e-9))
            dz = float(attrs.get("dz", 1e-9))
        else:
            dx = dy = dz = 1e-9
        m = 1.0 if multiplier is None else float(multiplier)

        grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
        grid.origin = (0.0, 0.0, 0.0)
        grid.spacing = (dx / m, dy / m, dz / m)
        values = np.transpose(mask, (2, 1, 0)).ravel(order="F")
        try:
            grid.cell_data["nonzero"] = values
        except Exception:
            setattr(grid, "cell_data", {"nonzero": values})

        p = plotter if plotter is not None else pv.Plotter()
        add_kwargs = dict(kwargs)
        add_kwargs.setdefault("opacity", [0.0, 0.25, 1.0])
        add_kwargs.setdefault("cmap", "viridis")
        p.add_volume(grid, scalars="nonzero", **add_kwargs)
        if show and plotter is None:
            p.show()
        return p

    def _pyvista_vector_impl(
        self,
        *,
        plotter=None,
        t: int = -1,
        zero: Optional[int] = None,
        vdims: Optional[
            tuple[
                Optional[Union[int, str]],
                Optional[Union[int, str]],
                Optional[Union[int, str]],
            ]
        ] = None,
        vdim_mapping: Optional[dict[Any, Any]] = None,
        multiplier: Optional[float] = None,
        quiver_density: int = 8,
        vector_scale: float = 1.0,
        min_magnitude: float = 0.0,
        points: bool = True,
        point_size: float = 5.0,
        point_color: str = "#4c72b0",
        use_color: bool = True,
        cmap: str = "viridis",
        color: str = "#dd8452",
        show: bool = False,
        **kwargs,
    ):
        pv = self._pyvista_import()
        volume = self._extract_volume(t=t, zero=zero)
        if volume.ndim != 4 or volume.shape[-1] < 2:
            raise ValueError(
                "pyvista.vector expects a vector volume with shape (z, y, x, c>=2), "
                f"got {volume.shape}"
            )

        src_n_comp = int(volume.shape[-1])
        comp_mapping = self._resolve_vdim_mapping(src_n_comp, vdim_mapping)
        vec = np.asarray(volume, dtype=np.float32)
        if vec.shape[-1] < 3:
            padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
            padded[..., : vec.shape[-1]] = vec
            vec = padded

        if vdims is None:
            ix = comp_mapping.get("x", 0 if src_n_comp >= 1 else None)
            iy = comp_mapping.get("y", 1 if src_n_comp >= 2 else None)
            iz = comp_mapping.get("z", 2 if src_n_comp >= 3 else None)
        else:
            ix = self._resolve_component_index(
                vdims[0],
                src_n_comp,
                mapping=comp_mapping,
                allow_none=True,
            )
            iy = self._resolve_component_index(
                vdims[1],
                src_n_comp,
                mapping=comp_mapping,
                allow_none=True,
            )
            iz = self._resolve_component_index(
                vdims[2],
                src_n_comp,
                mapping=comp_mapping,
                allow_none=True,
            )
            if ix is None and iy is None and iz is None:
                raise ValueError(f"At least one element in {vdims=} must be not None")

        nz, ny, nx, _ = vec.shape
        dens = max(int(quiver_density), 1)
        stepx = max(int(nx / dens), 1)
        stepy = max(int(ny / dens), 1)
        stepz = max(int(nz / dens), 1)
        z_idx = np.arange(0, nz, stepz, dtype=np.float32)
        y_idx = np.arange(0, ny, stepy, dtype=np.float32)
        x_idx = np.arange(0, nx, stepx, dtype=np.float32)
        zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing="ij")

        u = (
            np.asarray(vec[::stepz, ::stepy, ::stepx, ix], dtype=np.float32)
            if ix is not None
            else np.zeros_like(xx, dtype=np.float32)
        )
        v = (
            np.asarray(vec[::stepz, ::stepy, ::stepx, iy], dtype=np.float32)
            if iy is not None
            else np.zeros_like(xx, dtype=np.float32)
        )
        w = (
            np.asarray(vec[::stepz, ::stepy, ::stepx, iz], dtype=np.float32)
            if iz is not None
            else np.zeros_like(xx, dtype=np.float32)
        )

        attrs = getattr(self._dataset.job_result, "attrs", {})
        if hasattr(attrs, "get"):
            dx = float(attrs.get("dx", 1e-9))
            dy = float(attrs.get("dy", 1e-9))
            dz = float(attrs.get("dz", 1e-9))
        else:
            dx = dy = dz = 1e-9
        m = 1.0 if multiplier is None else float(multiplier)

        points_arr = np.stack(
            [xx * (dx / m), yy * (dy / m), zz * (dz / m)],
            axis=-1,
        ).reshape(-1, 3)
        vectors_arr = np.stack([u, v, w], axis=-1).reshape(-1, 3)
        magnitude = np.linalg.norm(vectors_arr, axis=1)
        valid = np.isfinite(magnitude) & (magnitude >= float(min_magnitude))
        points_arr = np.asarray(points_arr[valid], dtype=np.float32)
        vectors_arr = np.asarray(vectors_arr[valid], dtype=np.float32)
        magnitude = np.asarray(magnitude[valid], dtype=np.float32)

        if len(points_arr) == 0:
            raise ValueError("No vectors left after filtering/downsampling.")

        pdata = pv.PolyData(points_arr)
        pdata["vectors"] = vectors_arr
        pdata["magnitude"] = magnitude
        glyph = pdata.glyph(
            orient="vectors",
            scale="magnitude",
            factor=float(vector_scale),
        )

        p = plotter if plotter is not None else pv.Plotter()
        add_kwargs = dict(kwargs)
        if use_color:
            add_kwargs.setdefault("cmap", cmap)
            p.add_mesh(glyph, scalars="magnitude", **add_kwargs)
        else:
            add_kwargs.setdefault("color", color)
            p.add_mesh(glyph, **add_kwargs)

        if points:
            try:
                p.add_points(
                    points_arr,
                    color=point_color,
                    point_size=float(point_size),
                    render_points_as_spheres=True,
                )
            except Exception:
                pass

        if show and plotter is None:
            p.show()
        return p

