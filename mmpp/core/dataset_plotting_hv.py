"""Holoviews mixin for dataset plotting."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np

class DatasetPlotHVMixin:
    @staticmethod
    def _hv_apply_opts(obj, **kwargs):
        opts_fn = getattr(obj, "opts", None)
        if callable(opts_fn):
            try:
                return opts_fn(**kwargs)
            except Exception:
                return obj
        return obj

    @staticmethod
    def _hv_import():
        try:
            import holoviews as hv
        except Exception as exc:
            raise ImportError(
                "holoviews is required for plot.hv.*(). Install with: pip install holoviews"
            ) from exc

        loaded = bool(getattr(getattr(hv, "extension", None), "_loaded", False))
        if not loaded:
            try:
                hv.extension("bokeh", logo=False)
            except Exception:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hv.extension("bokeh")
        return hv

    def _hv_dynamic_axes(self, arr: np.ndarray) -> tuple[Optional[list[int]], Optional[list[int]]]:
        if arr.ndim == 5:
            return list(range(int(arr.shape[0]))), list(range(int(arr.shape[1])))
        if arr.ndim == 4:
            if arr.shape[-1] <= 4:
                return list(range(int(arr.shape[0]))), None
            return list(range(int(arr.shape[0]))), list(range(int(arr.shape[1])))
        if arr.ndim == 3 and arr.shape[-1] > 4:
            return None, list(range(int(arr.shape[0])))
        return None, None

    def _hv_scalar_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        multiplier: Optional[float] = None,
        repeat: int = 1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        filter_field: Any = None,
        cmap: str = "viridis",
        colorbar: bool = True,
        clim: Optional[tuple[float, float]] = None,
        dynamic: bool = True,
    ):
        hv = self._hv_import()
        arr = np.asarray(self._dataset.numpy(copy=False, squeeze=False), dtype=np.float32)
        frame_values, z_values = self._hv_dynamic_axes(arr)

        x_name, y_name = self._resolve_axis_names()

        def _build(frame_index: Optional[int] = None, z_index: Optional[int] = None):
            t_local = t if frame_index is None else int(frame_index)
            z_local = z if z_index is None else int(z_index)
            frame = self._extract_frame(z=z_local, t=t_local, zero=zero)
            image = self._component_image(frame, component, default="norm")
            image = np.tile(image, (max(int(repeat), 1), max(int(repeat), 1))).astype(
                np.float32,
                copy=False,
            )
            if filter_field is not None:
                mask = self._coerce_mask(
                    filter_field,
                    image.shape,
                    t=t_local,
                    z=z_local,
                    zero=zero,
                )
                image = np.where(mask, image, np.nan)

            _, _, extent, _, unit = self._resolve_plot_geometry(
                image.shape,
                multiplier=multiplier,
            )
            x = np.linspace(extent[0], extent[1], image.shape[1], endpoint=False)
            y = np.linspace(extent[2], extent[3], image.shape[0], endpoint=False)
            im = hv.Image(
                (x, y, image),
                kdims=[f"{x_name} ({unit})", f"{y_name} ({unit})"],
                vdims=["value"],
            )
            opts = {"cmap": cmap, "colorbar": bool(colorbar), "aspect": "equal"}
            if clim is not None:
                opts["clim"] = clim
            return self._hv_apply_opts(im, **opts)

        if not dynamic:
            return _build()

        kdims = []
        if frame_values is not None:
            kdims.append(hv.Dimension("frame", values=frame_values))
        if z_values is not None:
            kdims.append(hv.Dimension("z", values=z_values))
        if not kdims:
            return _build()

        def _dynamic(*args):
            frame_idx = None
            z_idx = None
            if frame_values is not None and len(args) >= 1:
                frame_idx = int(args[0])
            if z_values is not None:
                z_arg_pos = 1 if frame_values is not None else 0
                if len(args) > z_arg_pos:
                    z_idx = int(args[z_arg_pos])
            return _build(frame_idx, z_idx)

        return hv.DynamicMap(_dynamic, kdims=kdims)

    def _hv_vector_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        multiplier: Optional[float] = None,
        repeat: int = 1,
        zero: Optional[int] = None,
        filter_field: Any = None,
        vdims: Optional[tuple[Optional[Union[int, str]], Optional[Union[int, str]]]] = None,
        vdim_mapping: Optional[dict[Any, Any]] = None,
        color_field: Optional[Union[int, str, np.ndarray]] = None,
        cmap: str = "viridis",
        use_color: bool = True,
        colorbar: bool = True,
        quiver_density: int = 20,
        dynamic: bool = True,
    ):
        hv = self._hv_import()
        arr = np.asarray(self._dataset.numpy(copy=False, squeeze=False), dtype=np.float32)
        frame_values, z_values = self._hv_dynamic_axes(arr)
        x_name, y_name = self._resolve_axis_names()

        def _build(frame_index: Optional[int] = None, z_index: Optional[int] = None):
            t_local = t if frame_index is None else int(frame_index)
            z_local = z if z_index is None else int(z_index)
            frame = self._extract_frame(z=z_local, t=t_local, zero=zero)
            if frame.ndim != 3 or frame.shape[-1] < 2:
                raise ValueError(
                    f"hv.vector expects vector frame with shape (y, x, c>=2), got {frame.shape}"
                )

            src_n_comp = int(frame.shape[-1])
            comp_mapping = self._resolve_vdim_mapping(src_n_comp, vdim_mapping)
            vec = np.asarray(frame, dtype=np.float32)
            if vec.shape[-1] < 3:
                padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
                padded[..., : vec.shape[-1]] = vec
                vec = padded
            vec = np.tile(vec, (max(int(repeat), 1), max(int(repeat), 1), 1))

            if vdims is None:
                ix = comp_mapping.get("x", 0 if src_n_comp >= 1 else None)
                iy = comp_mapping.get("y", 1 if src_n_comp >= 2 else None)
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
                if ix is None and iy is None:
                    raise ValueError(f"At least one element in {vdims=} must not be None")

            u = (
                np.asarray(vec[:, :, ix], dtype=np.float32)
                if ix is not None
                else np.zeros(vec.shape[:2], dtype=np.float32)
            )
            v = (
                np.asarray(vec[:, :, iy], dtype=np.float32)
                if iy is not None
                else np.zeros(vec.shape[:2], dtype=np.float32)
            )

            if filter_field is not None:
                mask = self._coerce_mask(
                    filter_field,
                    u.shape,
                    t=t_local,
                    z=z_local,
                    zero=zero,
                )
            else:
                mask = np.ones_like(u, dtype=bool)

            dens = max(int(quiver_density), 1)
            stepx = max(int(u.shape[1] / dens), 1)
            stepy = max(int(u.shape[0] / dens), 1)

            u_ds = np.asarray(u[::stepy, ::stepx], dtype=np.float32)
            v_ds = np.asarray(v[::stepy, ::stepx], dtype=np.float32)
            mask_ds = np.asarray(mask[::stepy, ::stepx], dtype=bool)
            u_ds = np.where(mask_ds, u_ds, np.nan)
            v_ds = np.where(mask_ds, v_ds, np.nan)

            dx_u, dy_u, _, _, unit = self._resolve_plot_geometry(
                u.shape,
                multiplier=multiplier,
            )
            xs, ys = np.meshgrid(
                np.arange(0, u.shape[1], stepx, dtype=np.float32) * dx_u,
                np.arange(0, u.shape[0], stepy, dtype=np.float32) * dy_u,
            )
            angles = np.arctan2(v_ds, u_ds)
            mag = np.sqrt(u_ds**2 + v_ds**2)

            valid = np.isfinite(angles) & np.isfinite(mag)
            data = (
                xs[valid].ravel(),
                ys[valid].ravel(),
                angles[valid].ravel(),
                mag[valid].ravel(),
            )
            vf = hv.VectorField(
                data,
                kdims=[f"{x_name} ({unit})", f"{y_name} ({unit})", "angle"],
                vdims=["magnitude"],
            )

            opts = {"magnitude": "magnitude"}
            if use_color:
                opts["color"] = "magnitude"
                opts["cmap"] = cmap
                opts["colorbar"] = bool(colorbar)
            return self._hv_apply_opts(vf, **opts)

        if not dynamic:
            return _build()

        kdims = []
        if frame_values is not None:
            kdims.append(hv.Dimension("frame", values=frame_values))
        if z_values is not None:
            kdims.append(hv.Dimension("z", values=z_values))
        if not kdims:
            return _build()

        def _dynamic(*args):
            frame_idx = None
            z_idx = None
            if frame_values is not None and len(args) >= 1:
                frame_idx = int(args[0])
            if z_values is not None:
                z_arg_pos = 1 if frame_values is not None else 0
                if len(args) > z_arg_pos:
                    z_idx = int(args[z_arg_pos])
            return _build(frame_idx, z_idx)

        return hv.DynamicMap(_dynamic, kdims=kdims)

    def _hv_contour_impl(self, **kwargs):
        self._hv_import()
        levels = int(kwargs.pop("levels", 12))
        scalar = self._hv_scalar_impl(**kwargs)
        try:
            from holoviews.operation import contours as hv_contours
        except Exception:
            return scalar
        try:
            return hv_contours(scalar, levels=max(levels, 2))
        except Exception:
            return scalar
