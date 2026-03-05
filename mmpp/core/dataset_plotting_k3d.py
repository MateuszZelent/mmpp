"""K3D mixin for dataset plotting."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np

class DatasetPlotK3DMixin:
    @staticmethod
    def _k3d_dataset_attrs(dataset_obj) -> Any:
        return getattr(getattr(dataset_obj, "job_result", None), "attrs", {})

    @staticmethod
    def _k3d_source_shape(dataset_obj) -> Optional[tuple[int, ...]]:
        source = getattr(dataset_obj, "zarr_array", None)
        shape = getattr(source, "shape", None)
        if shape is None:
            shape = getattr(dataset_obj, "shape", None)
        if shape is None:
            return None
        return tuple(int(v) for v in shape)

    @staticmethod
    def _k3d_source_spatial_axes(
        source_shape: Optional[tuple[int, ...]],
    ) -> Optional[dict[str, int]]:
        if source_shape is None:
            return None

        ndim = len(source_shape)
        if ndim == 5:
            return {"x": 3, "y": 2, "z": 1}
        if ndim == 4:
            if int(source_shape[-1]) <= 4:
                return {"x": 2, "y": 1, "z": 0}
            return {"x": 3, "y": 2, "z": 1}
        if ndim == 3:
            return {"x": 2, "y": 1, "z": 0}
        return None

    def _k3d_source_spatial_shape(
        self,
        dataset_obj=None,
    ) -> Optional[tuple[int, int, int]]:
        obj = self._dataset if dataset_obj is None else dataset_obj
        source_shape = self._k3d_source_shape(obj)
        spatial_axes = self._k3d_source_spatial_axes(source_shape)
        if source_shape is None or spatial_axes is None:
            return None
        return (
            int(source_shape[spatial_axes["z"]]),
            int(source_shape[spatial_axes["y"]]),
            int(source_shape[spatial_axes["x"]]),
        )

    @staticmethod
    def _k3d_attr_triplet(
        attrs: Any,
        *,
        key: str,
    ) -> Optional[tuple[float, float, float]]:
        if not hasattr(attrs, "get"):
            return None
        raw = attrs.get(key, None)
        if raw is None:
            return None
        try:
            seq = tuple(float(v) for v in raw)
        except Exception:
            return None
        if len(seq) < 3:
            return None
        return (float(seq[0]), float(seq[1]), float(seq[2]))

    @classmethod
    def _k3d_axis_min_and_cell(
        cls,
        *,
        attrs: Any,
        axis: str,
        total_n: int,
        default_cell_m: float,
    ) -> tuple[float, float]:
        axis_pos = {"x": 0, "y": 1, "z": 2}[axis]
        pmin_triplet = cls._k3d_attr_triplet(attrs, key="pmin")
        pmax_triplet = cls._k3d_attr_triplet(attrs, key="pmax")

        min_key = f"{axis}min"
        max_key = f"{axis}max"
        base_key = f"{axis}base"

        min_value = None
        max_value = None
        if pmin_triplet is not None:
            min_value = float(pmin_triplet[axis_pos])
        elif hasattr(attrs, "get"):
            raw = attrs.get(min_key, None)
            if raw is not None:
                min_value = float(raw)

        if pmax_triplet is not None:
            max_value = float(pmax_triplet[axis_pos])
        elif hasattr(attrs, "get"):
            raw = attrs.get(max_key, None)
            if raw is not None:
                max_value = float(raw)

        cell_m = float(default_cell_m)
        if min_value is not None and max_value is not None and int(total_n) > 0:
            span = float(max_value) - float(min_value)
            if np.isfinite(span) and span > 0.0:
                cell_m = span / float(total_n)

        if min_value is None and hasattr(attrs, "get"):
            raw = attrs.get(base_key, None)
            if raw is not None:
                min_value = float(raw) - 0.5 * cell_m

        if min_value is None and max_value is not None and int(total_n) > 0:
            min_value = float(max_value) - float(total_n) * cell_m

        if min_value is None:
            min_value = 0.0

        return float(min_value), float(cell_m)

    @staticmethod
    def _k3d_axis_selection_geometry(
        *,
        total_n: int,
        token: Any,
        axis_min_m: float,
        cell_m: float,
        fallback_count: int,
    ) -> tuple[float, float, float]:
        count_target = max(int(fallback_count), 1)
        if token is None:
            pmin = float(axis_min_m)
            pmax = pmin + float(count_target) * float(cell_m)
            return pmin, pmax, float(cell_m)

        if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
            idx = int(token)
            if idx < 0:
                idx += int(total_n)
            idx = int(np.clip(idx, 0, max(int(total_n) - 1, 0)))
            pmin = float(axis_min_m) + float(idx) * float(cell_m)
            pmax = pmin + float(cell_m)
            return pmin, pmax, float(cell_m)

        if isinstance(token, slice):
            start, stop, step = token.indices(int(total_n))
            indices = list(range(start, stop, step))
            if not indices:
                pmin = float(axis_min_m)
                pmax = pmin + float(count_target) * float(cell_m)
                return pmin, pmax, float(cell_m)

            lo = min(indices)
            hi = max(indices)
            pmin = float(axis_min_m) + float(lo) * float(cell_m)
            pmax = float(axis_min_m) + float(hi + 1) * float(cell_m)
            cell_eff = (pmax - pmin) / float(count_target)
            return pmin, pmax, float(cell_eff)

        pmin = float(axis_min_m)
        pmax = pmin + float(count_target) * float(cell_m)
        return pmin, pmax, float(cell_m)

    def _k3d_resolve_geometry(
        self,
        shape_zyx: tuple[int, int, int],
        *,
        multiplier: Optional[float] = None,
        dataset_obj=None,
        include_slice: bool = True,
    ) -> tuple[list[float], list[str], float, tuple[float, float, float], tuple[float, float, float]]:
        """Resolve k3d bounds/labels for a volume with shape (z, y, x)."""
        obj = self._dataset if dataset_obj is None else dataset_obj
        attrs = self._k3d_dataset_attrs(obj)
        if hasattr(attrs, "get"):
            dx_m = float(attrs.get("dx", 1e-9))
            dy_m = float(attrs.get("dy", 1e-9))
            dz_m = float(attrs.get("dz", 1e-9))
            x_name = str(attrs.get("x_name", "x"))
            y_name = str(attrs.get("y_name", "y"))
            z_name = str(attrs.get("z_name", "z"))
        else:
            dx_m = dy_m = dz_m = 1e-9
            x_name, y_name, z_name = "x", "y", "z"

        nz, ny, nx = (
            max(int(shape_zyx[0]), 1),
            max(int(shape_zyx[1]), 1),
            max(int(shape_zyx[2]), 1),
        )
        source_shape = self._k3d_source_shape(obj)
        spatial_axes = self._k3d_source_spatial_axes(source_shape)
        slice_info = getattr(obj, "slice_info", None) if bool(include_slice) else None

        if source_shape is not None and spatial_axes is not None and not bool(include_slice):
            nz = max(int(source_shape[spatial_axes["z"]]), 1)
            ny = max(int(source_shape[spatial_axes["y"]]), 1)
            nx = max(int(source_shape[spatial_axes["x"]]), 1)

        total_nx = int(source_shape[spatial_axes["x"]]) if source_shape is not None and spatial_axes is not None else int(nx)
        total_ny = int(source_shape[spatial_axes["y"]]) if source_shape is not None and spatial_axes is not None else int(ny)
        total_nz = int(source_shape[spatial_axes["z"]]) if source_shape is not None and spatial_axes is not None else int(nz)

        base_x_min_m, base_dx_m = self._k3d_axis_min_and_cell(
            attrs=attrs,
            axis="x",
            total_n=total_nx,
            default_cell_m=dx_m,
        )
        base_y_min_m, base_dy_m = self._k3d_axis_min_and_cell(
            attrs=attrs,
            axis="y",
            total_n=total_ny,
            default_cell_m=dy_m,
        )
        base_z_min_m, base_dz_m = self._k3d_axis_min_and_cell(
            attrs=attrs,
            axis="z",
            total_n=total_nz,
            default_cell_m=dz_m,
        )

        x_token = None
        y_token = None
        z_token = None
        if (
            slice_info is not None
            and isinstance(slice_info, tuple)
            and source_shape is not None
            and spatial_axes is not None
            and len(slice_info) >= len(source_shape)
        ):
            x_token = slice_info[spatial_axes["x"]]
            y_token = slice_info[spatial_axes["y"]]
            z_token = slice_info[spatial_axes["z"]]

        x_min_m, x_max_m, dx_eff_m = self._k3d_axis_selection_geometry(
            total_n=total_nx,
            token=x_token,
            axis_min_m=base_x_min_m,
            cell_m=base_dx_m,
            fallback_count=nx,
        )
        y_min_m, y_max_m, dy_eff_m = self._k3d_axis_selection_geometry(
            total_n=total_ny,
            token=y_token,
            axis_min_m=base_y_min_m,
            cell_m=base_dy_m,
            fallback_count=ny,
        )
        z_min_m, z_max_m, dz_eff_m = self._k3d_axis_selection_geometry(
            total_n=total_nz,
            token=z_token,
            axis_min_m=base_z_min_m,
            cell_m=base_dz_m,
            fallback_count=nz,
        )

        size_x_m = float(x_max_m - x_min_m)
        size_y_m = float(y_max_m - y_min_m)
        size_z_m = float(z_max_m - z_min_m)

        if multiplier is None:
            m = self._auto_si_multiplier((size_x_m, size_y_m, size_z_m))
        else:
            m = float(multiplier)
            if m <= 0.0:
                raise ValueError(f"multiplier must be > 0, got {m}")

        x_min_u = float(x_min_m / m)
        x_max_u = float(x_max_m / m)
        y_min_u = float(y_min_m / m)
        y_max_u = float(y_max_m / m)
        z_min_u = float(z_min_m / m)
        z_max_u = float(z_max_m / m)

        extent_x = float(x_max_u - x_min_u)
        extent_y = float(y_max_u - y_min_u)
        extent_z = float(z_max_u - z_min_u)

        extent_max = max(abs(extent_x), abs(extent_y), abs(extent_z), 1.0)
        eps = extent_max * 1e-6
        if extent_x <= 0.0:
            x_max_u = x_min_u + eps
            extent_x = eps
        if extent_y <= 0.0:
            y_max_u = y_min_u + eps
            extent_y = eps
        if extent_z <= 0.0:
            z_max_u = z_min_u + eps
            extent_z = eps

        bounds = [x_min_u, x_max_u, y_min_u, y_max_u, z_min_u, z_max_u]
        unit_label = self._unit_label_from_multiplier(m)
        axes = [
            rf"{x_name}\,(\text{{{unit_label}}})",
            rf"{y_name}\,(\text{{{unit_label}}})",
            rf"{z_name}\,(\text{{{unit_label}}})",
        ]
        cell = (float(dx_eff_m / m), float(dy_eff_m / m), float(dz_eff_m / m))
        extents = (float(extent_x), float(extent_y), float(extent_z))
        return bounds, axes, float(m), extents, cell

    @staticmethod
    def _k3d_set_axes(plot_obj, axes: list[str]) -> None:
        try:
            plot_obj.axes = list(axes)
        except Exception:
            pass

    def _k3d_prepare_interactive_plot(
        self,
        plot_obj,
        *,
        multiplier: Optional[float],
        interactive_field: Any,
    ) -> None:
        self._k3d_clear_plot(plot_obj)
        if interactive_field is None:
            return

        try:
            plot_obj.camera_auto_fit = False
        except Exception:
            pass

        try:
            objects = list(getattr(plot_obj, "objects", []))
        except Exception:
            objects = []
        if any(getattr(obj, "name", None) == "total_region" for obj in objects):
            return

        try:
            import k3d
        except Exception:
            return

        total_shape = self._k3d_source_spatial_shape(interactive_field)
        if total_shape is None:
            return

        bounds, axes, _, _, _ = self._k3d_resolve_geometry(
            total_shape,
            multiplier=multiplier,
            dataset_obj=interactive_field,
            include_slice=False,
        )
        try:
            plot_obj += k3d.voxels(
                np.ones((1, 1, 1), dtype=np.uint8),
                color_map=0x4C72B0,
                bounds=bounds,
                outlines=False,
                name="total_region",
                opacity=0.025,
            )
        except Exception:
            return

        self._k3d_set_axes(plot_obj, axes)

    def _k3d_scalar_impl(
        self,
        *,
        plot=None,
        t: int = -1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        filter_field: Any = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        multiplier: Optional[float] = None,
        interactive_field: Any = None,
        interactive: bool = False,
        **kwargs,
    ):
        try:
            import k3d
        except Exception as exc:
            raise ImportError(
                "k3d is required for plot.k3d.scalar(). Install with: pip install k3d"
            ) from exc

        volume = self._extract_volume(t=t, zero=zero)
        scalar = self._component_volume(volume, component, default="norm")
        if scalar.ndim != 3:
            raise ValueError(
                "k3d.scalar expects a 3d scalar volume. "
                f"Got shape {scalar.shape}."
            )

        visible = self._coerce_mask(filter_field, scalar.shape)
        voxels = self._normalise_to_uint8(
            scalar,
            vmin=vmin,
            vmax=vmax,
            visible_mask=visible,
        )
        bounds, axes, _, _, _ = self._k3d_resolve_geometry(
            tuple(int(v) for v in scalar.shape),
            multiplier=multiplier,
        )

        _created_internally = plot is None
        plot_obj = plot if plot is not None else k3d.plot(name=f"{self._dataset.dataset_name} scalar")
        if interactive or interactive_field is not None:
            self._k3d_prepare_interactive_plot(
                plot_obj,
                multiplier=multiplier,
                interactive_field=interactive_field,
            )

        cmap_int = self._k3d_colormap_int(cmap)
        try:
            plot_obj += k3d.voxels(
                voxels,
                color_map=cmap_int,
                bounds=bounds,
                outlines=False,
                **kwargs,
            )
        except Exception:
            plot_obj += k3d.voxels(voxels, color_map=cmap_int, bounds=bounds, **kwargs)

        self._k3d_set_axes(plot_obj, axes)

        if _created_internally:
            plot_obj.display()
        return plot_obj

    def _k3d_nonzero_impl(
        self,
        *,
        plot=None,
        t: int = -1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        threshold: float = 0.0,
        color: int = 0x4C72B0,
        multiplier: Optional[float] = None,
        interactive_field: Any = None,
        interactive: bool = False,
        **kwargs,
    ):
        try:
            import k3d
        except Exception as exc:
            raise ImportError(
                "k3d is required for plot.k3d.nonzero(). Install with: pip install k3d"
            ) from exc

        volume = self._extract_volume(t=t, zero=zero)
        scalar = self._component_volume(volume, component, default="norm")
        if scalar.ndim != 3:
            raise ValueError(
                "k3d.nonzero expects a 3d scalar volume. "
                f"Got shape {scalar.shape}."
            )

        voxels = np.where(np.abs(scalar) > float(threshold), 1, 0).astype(np.uint8)
        bounds, axes, _, _, _ = self._k3d_resolve_geometry(
            tuple(int(v) for v in scalar.shape),
            multiplier=multiplier,
        )

        _created_internally = plot is None
        plot_obj = plot if plot is not None else k3d.plot(
            name=f"{self._dataset.dataset_name} nonzero"
        )
        if interactive or interactive_field is not None:
            self._k3d_prepare_interactive_plot(
                plot_obj,
                multiplier=multiplier,
                interactive_field=interactive_field,
            )

        try:
            plot_obj += k3d.voxels(
                voxels,
                color_map=int(color),
                bounds=bounds,
                outlines=False,
                **kwargs,
            )
        except Exception:
            plot_obj += k3d.voxels(voxels, color_map=int(color), bounds=bounds, **kwargs)

        self._k3d_set_axes(plot_obj, axes)
        if _created_internally:
            plot_obj.display()
        return plot_obj

    def _k3d_vector_impl(
        self,
        *,
        plot=None,
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
        color_field: Any = None,
        cmap: str = "viridis",
        head_size: float = 1.0,
        points: bool = True,
        point_size: Optional[float] = None,
        vector_multiplier: Optional[float] = None,
        vector_scale: float = 1.0,
        multiplier: Optional[float] = None,
        quiver_density: Optional[int] = None,
        min_magnitude: float = 0.0,
        color: int = 0xDD8452,
        interactive_field: Any = None,
        interactive: bool = False,
        **kwargs,
    ):
        try:
            import k3d
        except Exception as exc:
            raise ImportError(
                "k3d is required for plot.k3d.vector(). Install with: pip install k3d"
            ) from exc

        volume = self._extract_volume(t=t, zero=zero)
        if volume.ndim != 4 or volume.shape[-1] < 2:
            raise ValueError(
                "k3d.vector expects a vector volume with shape (z, y, x, c>=2), "
                f"got {volume.shape}"
            )

        src_n_comp = int(volume.shape[-1])
        comp_mapping = self._resolve_vdim_mapping(src_n_comp, vdim_mapping)
        vec = np.asarray(volume, dtype=np.float32)
        if vec.shape[-1] < 3:
            padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
            padded[..., : vec.shape[-1]] = vec
            vec = padded

        nz, ny, nx, _ = vec.shape
        if quiver_density is not None:
            dens = max(int(quiver_density), 1)
            stepx = max(int(nx / dens), 1)
            stepy = max(int(ny / dens), 1)
            stepz = max(int(nz / dens), 1)
        else:
            stepx = stepy = stepz = 1
        z_idx = np.arange(0, nz, stepz, dtype=np.float32)
        y_idx = np.arange(0, ny, stepy, dtype=np.float32)
        x_idx = np.arange(0, nx, stepx, dtype=np.float32)

        if vdims is None:
            ix = comp_mapping.get("x", 0 if src_n_comp >= 1 else None)
            iy = comp_mapping.get("y", 1 if src_n_comp >= 2 else None)
            iz = comp_mapping.get("z", 2 if src_n_comp >= 3 else None)
        else:
            if len(vdims) != 3:
                raise ValueError(f"{vdims=} must contain exactly 3 elements")
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
                raise ValueError(f"At least one entry in {vdims=} must be not None")

        bounds, axes, _, _, (dx_u, dy_u, dz_u) = self._k3d_resolve_geometry(
            tuple(int(v) for v in vec.shape[:3]),
            multiplier=multiplier,
        )

        u = (
            np.asarray(vec[::stepz, ::stepy, ::stepx, ix], dtype=np.float32)
            if ix is not None
            else np.zeros((len(z_idx), len(y_idx), len(x_idx)), dtype=np.float32)
        )
        v = (
            np.asarray(vec[::stepz, ::stepy, ::stepx, iy], dtype=np.float32)
            if iy is not None
            else np.zeros((len(z_idx), len(y_idx), len(x_idx)), dtype=np.float32)
        )
        w = (
            np.asarray(vec[::stepz, ::stepy, ::stepx, iz], dtype=np.float32)
            if iz is not None
            else np.zeros((len(z_idx), len(y_idx), len(x_idx)), dtype=np.float32)
        )

        magnitude = np.sqrt(u**2 + v**2 + w**2)
        visible = np.isfinite(magnitude) & (magnitude >= float(min_magnitude))

        zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing="ij")

        origins = np.stack(
            [
                float(bounds[0]) + (xx + 0.5) * float(dx_u),
                float(bounds[2]) + (yy + 0.5) * float(dy_u),
                float(bounds[4]) + (zz + 0.5) * float(dz_u),
            ],
            axis=-1,
        ).reshape(-1, 3)
        vectors = np.stack([u, v, w], axis=-1).reshape(-1, 3)
        visible_flat = visible.reshape(-1)

        if vector_multiplier is None:
            cell_min = max(min(abs(dx_u), abs(dy_u), abs(dz_u)), 1e-12)
            # Max absolute component value (like discretisedfield) — not the norm,
            # which would make arrows look proportionally smaller.
            vmax = float(np.nanmax(np.abs(vectors))) if vectors.size > 0 else 1.0
            vector_multiplier = vmax / max(cell_min, 1e-12)
            if not np.isfinite(vector_multiplier) or vector_multiplier <= 0:
                vector_multiplier = 1.0

        vectors = vectors / float(vector_multiplier)
        vectors = vectors * float(vector_scale)
        origins = np.asarray(origins[visible_flat], dtype=np.float32)
        vectors = np.asarray(vectors[visible_flat], dtype=np.float32)

        if origins.size == 0:
            raise ValueError("No vectors left after filtering/downsampling.")

        _created_internally = plot is None
        plot_obj = plot if plot is not None else k3d.plot(
            name=f"{self._dataset.dataset_name} vector"
        )
        if interactive or interactive_field is not None:
            self._k3d_prepare_interactive_plot(
                plot_obj,
                multiplier=multiplier,
                interactive_field=interactive_field,
            )

        vector_kwargs = dict(kwargs)
        colors = None
        if color_field is not None:
            if isinstance(color_field, (int, np.integer, str)):
                if isinstance(color_field, str):
                    idx = self._resolve_component_index(
                        color_field,
                        src_n_comp,
                        mapping=comp_mapping,
                        allow_none=False,
                    )
                    color_volume = np.asarray(volume[..., idx], dtype=np.float32)
                else:
                    color_volume = self._component_volume(volume, color_field, default="norm")
            else:
                color_volume = np.asarray(color_field, dtype=np.float32)
                color_volume = np.squeeze(color_volume)
                if color_volume.shape != vec.shape[:-1]:
                    color_volume = np.broadcast_to(color_volume, vec.shape[:-1])

            c_sampled = np.asarray(
                color_volume[::stepz, ::stepy, ::stepx],
                dtype=np.float32,
            ).reshape(-1)
            c_sampled = c_sampled[visible_flat]
            c_uint8 = self._normalise_to_uint8(c_sampled, vmin=None, vmax=None)
            cmap_int = self._k3d_colormap_int(cmap)
            colors = []
            for value in c_uint8:
                idx = int(np.clip(value, 0, len(cmap_int) - 1))
                colors.append(2 * (cmap_int[idx],))
            vector_kwargs["colors"] = colors
        else:
            vector_kwargs["color"] = int(color)

        try:
            plot_obj += k3d.vectors(
                origins - 0.5 * vectors,
                vectors,
                head_size=float(head_size),
                **vector_kwargs,
            )
        except Exception:
            plot_obj += k3d.vectors(
                origins=origins - 0.5 * vectors,
                vectors=vectors,
                head_size=float(head_size),
                **vector_kwargs,
            )

        if points:
            if point_size is None:
                point_size = max(min(abs(dx_u), abs(dy_u), abs(dz_u)) / 4.0, 1e-6)
            try:
                plot_obj += k3d.points(
                    origins,
                    color=0x4C72B0,
                    point_size=float(point_size),
                )
            except Exception:
                pass

        self._k3d_set_axes(plot_obj, axes)

        if _created_internally:
            plot_obj.display()
        return plot_obj

    def _k3d_heatmap_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        repeat: int = 1,
        zero: Optional[int] = None,
        component: Optional[Union[int, str]] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_vectors: bool = False,
        quiver_density: int = 20,
        vector_scale: float = 1.0,
        vector_color: int = 0x00D1FF,
        vector_head_size: float = 0.8,
        vector_line_width: float = 0.015,
        height_scale: float = 0.0,
    ):
        try:
            import k3d
        except Exception as exc:
            raise ImportError(
                "k3d is required for plot.k3d.heatmap(). Install with: pip install k3d"
            ) from exc

        frame = self._extract_frame(z=z, t=t, zero=zero)
        image = self._component_image(frame, component, default="norm")
        image = np.tile(image, (max(int(repeat), 1), max(int(repeat), 1))).astype(
            np.float32,
            copy=False,
        )
        colormap = self._k3d_colormap(cmap)
        color_range = self._k3d_color_range(image, vmin=vmin, vmax=vmax)

        if float(height_scale) != 0.0:
            surface = (image * float(height_scale)).astype(np.float32, copy=False)
        else:
            surface = np.zeros_like(image, dtype=np.float32)

        plot = k3d.plot(name=f"{self._dataset.dataset_name} heatmap")
        surface_kwargs = {"attribute": image, "color_range": color_range}
        if colormap is not None:
            surface_kwargs["color_map"] = colormap

        try:
            plot += k3d.surface(surface, **surface_kwargs)
        except Exception:
            # Fallback for older/newer k3d variants with limited kwargs.
            fallback_kwargs = {}
            if colormap is not None:
                fallback_kwargs["color_map"] = colormap
            try:
                plot += k3d.surface(surface, **fallback_kwargs)
            except Exception:
                plot += k3d.surface(image.astype(np.float32, copy=False))

        if show_vectors and frame.ndim == 3 and frame.shape[-1] >= 2:
            vec = np.asarray(frame, dtype=np.float32)
            if vec.shape[-1] < 3:
                padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
                padded[..., : vec.shape[-1]] = vec
                vec = padded
            vec = np.tile(vec, (max(int(repeat), 1), max(int(repeat), 1), 1))
            u = vec[:, :, 0]
            v = vec[:, :, 1]
            stepx = max(int(u.shape[1] / max(int(quiver_density), 1)), 1)
            stepy = max(int(u.shape[0] / max(int(quiver_density), 1)), 1)
            grid_x, grid_y = np.meshgrid(
                np.arange(0, u.shape[1], stepx, dtype=np.float32),
                np.arange(0, u.shape[0], stepy, dtype=np.float32),
            )
            origins = np.stack(
                [grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size, dtype=np.float32)],
                axis=1,
            ).astype(np.float32)
            vectors = np.stack(
                [
                    u[::stepy, ::stepx].ravel(),
                    v[::stepy, ::stepx].ravel(),
                    np.zeros(grid_x.size, dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32)
            vectors *= float(vector_scale)
            try:
                plot += k3d.vectors(
                    origins,
                    vectors,
                    color=int(vector_color),
                    head_size=float(vector_head_size),
                    line_width=float(vector_line_width),
                )
            except Exception:
                try:
                    plot += k3d.vectors(
                        origins=origins,
                        vectors=vectors,
                        color=int(vector_color),
                        head_size=float(vector_head_size),
                        line_width=float(vector_line_width),
                    )
                except Exception:
                    pass

        return plot

    def _k3d_voxels_vectors_impl(
        self,
        *,
        plot=None,
        t: int = -1,
        zero: Optional[int] = None,
        scalar_component: Optional[Union[int, str]] = None,
        vdims: Optional[
            tuple[
                Optional[Union[int, str]],
                Optional[Union[int, str]],
                Optional[Union[int, str]],
            ]
        ] = None,
        vdim_mapping: Optional[dict[Any, Any]] = None,
        color_field: Any = None,
        cmap: str = "cividis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        filter_field: Any = None,
        head_size: float = 2.0,
        points: bool = False,
        point_size: Optional[float] = None,
        vector_multiplier: Optional[float] = None,
        vector_scale: float = 1.0,
        quiver_density: Optional[int] = None,
        min_magnitude: float = 0.0,
        vector_color: int = 0xFFFFFF,
        color_vectors_by_scalar: bool = True,
        voxel_opacity: float = 0.6,
        multiplier: Optional[float] = None,
        **kwargs,
    ):
        """Combined voxel + vector k3d plot.

        Renders scalar voxels (coloured by ``scalar_component`` or |m| norm)
        with vector arrows overlaid inside each voxel.  The arrows are coloured
        by the same scalar as the voxels when ``color_vectors_by_scalar=True``
        so that the two layers form a coherent, publication-quality 3-D plot.

        Parameters
        ----------
        scalar_component:
            Which component to use for the voxel colour.  ``None`` → |m| norm.
        vdims:
            Triplet ``(x_comp, y_comp, z_comp)`` selecting which data components
            map to plot-space (x, y, z) arrows.  ``None`` → auto-detect.
        color_field:
            Override the voxel-synchronised colouring with a custom scalar
            volume (array or component name).
        cmap:
            Matplotlib colormap name, applied to both voxels and arrows.
        vmin / vmax:
            Explicit data range for the scalar normalisation.
        filter_field:
            Boolean / scalar mask — zero entries hide the corresponding voxels
            *and* their arrows.
        head_size:
            Arrow head size (k3d.vectors parameter). Default ``2.0`` is larger
            than the standalone ``vector()`` default for better visibility
            against dense voxels.
        points:
            Draw a dot at each arrow origin.  Default ``False`` for cleaner
            appearance when voxels are present.
        vector_multiplier:
            Manual override for arrow scaling.  ``None`` → auto (max absolute
            component value / min cell dimension), matching discretisedfield.
        vector_scale:
            Multiplicative post-processing scale applied on top of
            ``vector_multiplier``.
        quiver_density:
            Number of arrows per axis (integer).  ``None`` → render arrows for
            **every** non-masked cell (like discretisedfield behaviour).
        min_magnitude:
            Cells with |m| < min_magnitude are skipped for arrows.
        vector_color:
            Uniform arrow colour used when ``color_vectors_by_scalar=False``
            and ``color_field=None``.  Default white (``0xFFFFFF``).
        color_vectors_by_scalar:
            If ``True`` (default) arrows are coloured by the same scalar as
            the voxels.  Set to ``False`` for a uniform ``vector_color``.
        voxel_opacity:
            Global opacity of the voxel layer.  ``0.6`` lets you see
            the arrows through the voxels.
        multiplier:
            Physical unit multiplier (e.g. ``1e-9`` for nm).  ``None`` → auto.
        """
        try:
            import k3d
        except Exception as exc:
            raise ImportError(
                "k3d is required for plot.k3d.voxels_vectors(). "
                "Install with: pip install k3d"
            ) from exc

        volume = self._extract_volume(t=t, zero=zero)

        # ── scalar for voxel colouring ───────────────────────────────────────
        scalar = self._component_volume(volume, scalar_component, default="norm")
        if scalar.ndim != 3:
            raise ValueError(
                "voxels_vectors expects a 3-d scalar volume; "
                f"got shape {scalar.shape}."
            )

        visible_mask = self._coerce_mask(filter_field, scalar.shape)
        voxels_arr = self._normalise_to_uint8(
            scalar, vmin=vmin, vmax=vmax, visible_mask=visible_mask
        )

        bounds, axes, _m, _extents, (dx_u, dy_u, dz_u) = self._k3d_resolve_geometry(
            tuple(int(v) for v in scalar.shape),
            multiplier=multiplier,
        )

        # ── vector components ────────────────────────────────────────────────
        if volume.ndim != 4 or volume.shape[-1] < 2:
            raise ValueError(
                "voxels_vectors expects a vector volume (z, y, x, c>=2); "
                f"got {volume.shape}."
            )

        src_n_comp = int(volume.shape[-1])
        comp_mapping = self._resolve_vdim_mapping(src_n_comp, vdim_mapping)
        vec = np.asarray(volume, dtype=np.float32)
        if vec.shape[-1] < 3:
            padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
            padded[..., : vec.shape[-1]] = vec
            vec = padded

        nz, ny, nx, _ = vec.shape
        if quiver_density is not None:
            dens = max(int(quiver_density), 1)
            stepx = max(int(nx / dens), 1)
            stepy = max(int(ny / dens), 1)
            stepz = max(int(nz / dens), 1)
        else:
            stepx = stepy = stepz = 1

        z_idx = np.arange(0, nz, stepz, dtype=np.float32)
        y_idx = np.arange(0, ny, stepy, dtype=np.float32)
        x_idx = np.arange(0, nx, stepx, dtype=np.float32)

        if vdims is None:
            ix = comp_mapping.get("x", 0 if src_n_comp >= 1 else None)
            iy = comp_mapping.get("y", 1 if src_n_comp >= 2 else None)
            iz = comp_mapping.get("z", 2 if src_n_comp >= 3 else None)
        else:
            if len(vdims) != 3:
                raise ValueError(f"{vdims=} must contain exactly 3 elements")
            ix = self._resolve_component_index(
                vdims[0], src_n_comp, mapping=comp_mapping, allow_none=True
            )
            iy = self._resolve_component_index(
                vdims[1], src_n_comp, mapping=comp_mapping, allow_none=True
            )
            iz = self._resolve_component_index(
                vdims[2], src_n_comp, mapping=comp_mapping, allow_none=True
            )
            if ix is None and iy is None and iz is None:
                raise ValueError(f"At least one entry in {vdims=} must be not None")

        u = (
            np.asarray(vec[::stepz, ::stepy, ::stepx, ix], dtype=np.float32)
            if ix is not None
            else np.zeros((len(z_idx), len(y_idx), len(x_idx)), dtype=np.float32)
        )
        v_comp = (
            np.asarray(vec[::stepz, ::stepy, ::stepx, iy], dtype=np.float32)
            if iy is not None
            else np.zeros((len(z_idx), len(y_idx), len(x_idx)), dtype=np.float32)
        )
        w = (
            np.asarray(vec[::stepz, ::stepy, ::stepx, iz], dtype=np.float32)
            if iz is not None
            else np.zeros((len(z_idx), len(y_idx), len(x_idx)), dtype=np.float32)
        )

        magnitude = np.sqrt(u**2 + v_comp**2 + w**2)
        # hide cells below min_magnitude AND cells hidden by filter_field
        voxels_sampled = voxels_arr[::stepz, ::stepy, ::stepx]
        visible_vec = (
            np.isfinite(magnitude)
            & (magnitude >= float(min_magnitude))
            & (voxels_sampled > 0)
        )

        zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing="ij")
        origins_all = np.stack(
            [
                float(bounds[0]) + (xx + 0.5) * float(dx_u),
                float(bounds[2]) + (yy + 0.5) * float(dy_u),
                float(bounds[4]) + (zz + 0.5) * float(dz_u),
            ],
            axis=-1,
        ).reshape(-1, 3)
        vectors_all = np.stack([u, v_comp, w], axis=-1).reshape(-1, 3)
        visible_flat = visible_vec.reshape(-1)

        # ── auto-scale arrows: max |component| / min cell (matches discretisedfield) ──
        if vector_multiplier is None:
            cell_min = max(min(abs(dx_u), abs(dy_u), abs(dz_u)), 1e-12)
            vm_val = float(np.nanmax(np.abs(vectors_all))) if vectors_all.size > 0 else 1.0
            vector_multiplier = vm_val / max(cell_min, 1e-12)
            if not np.isfinite(vector_multiplier) or vector_multiplier <= 0:
                vector_multiplier = 1.0

        vectors_scaled = (vectors_all / float(vector_multiplier)) * float(vector_scale)

        origins_f = np.asarray(origins_all[visible_flat], dtype=np.float32)
        vectors_f = np.asarray(vectors_scaled[visible_flat], dtype=np.float32)

        if origins_f.size == 0:
            raise ValueError(
                "No vectors left after applying filter_field / min_magnitude. "
                "Try reducing min_magnitude or removing the filter."
            )

        # ── build plot ───────────────────────────────────────────────────────
        _created_internally = plot is None
        plot_obj = plot if plot is not None else k3d.plot(
            name=f"{self._dataset.dataset_name} voxels+vectors"
        )

        cmap_int = self._k3d_colormap_int(cmap)

        # Layer 1 — coloured voxels
        vox_kw: dict = {"outlines": False}
        vox_kw.update(kwargs)
        if float(voxel_opacity) < 1.0:
            vox_kw["opacity"] = float(voxel_opacity)
        try:
            plot_obj += k3d.voxels(
                voxels_arr, color_map=cmap_int, bounds=bounds, **vox_kw
            )
        except Exception:
            plot_obj += k3d.voxels(voxels_arr, color_map=cmap_int, bounds=bounds)

        # Layer 2 — arrows
        arrow_kw: dict = {}
        if color_vectors_by_scalar and color_field is None:
            # Colour arrows by the same scalar that drives voxel colours.
            c_flat = self._normalise_to_uint8(
                scalar[::stepz, ::stepy, ::stepx].reshape(-1)[visible_flat],
                vmin=vmin,
                vmax=vmax,
            )
            colors = [
                2 * (cmap_int[int(np.clip(c, 0, len(cmap_int) - 1))],)
                for c in c_flat
            ]
            arrow_kw["colors"] = colors

        elif color_field is not None:
            if isinstance(color_field, str):
                cf_idx = self._resolve_component_index(
                    color_field, src_n_comp, mapping=comp_mapping, allow_none=False
                )
                cf_vol = np.asarray(volume[..., cf_idx], dtype=np.float32)
            else:
                cf_vol = np.asarray(color_field, dtype=np.float32)
            c_flat = self._normalise_to_uint8(
                cf_vol[::stepz, ::stepy, ::stepx].reshape(-1)[visible_flat]
            )
            colors = [
                2 * (cmap_int[int(np.clip(c, 0, len(cmap_int) - 1))],)
                for c in c_flat
            ]
            arrow_kw["colors"] = colors
        else:
            arrow_kw["color"] = int(vector_color)

        try:
            plot_obj += k3d.vectors(
                origins_f - 0.5 * vectors_f,
                vectors_f,
                head_size=float(head_size),
                **arrow_kw,
            )
        except Exception:
            plot_obj += k3d.vectors(
                origins=origins_f - 0.5 * vectors_f,
                vectors=vectors_f,
                head_size=float(head_size),
                **arrow_kw,
            )

        if points:
            if point_size is None:
                point_size = max(min(abs(dx_u), abs(dy_u), abs(dz_u)) / 4.0, 1e-6)
            try:
                plot_obj += k3d.points(
                    origins_f, color=0x4C72B0, point_size=float(point_size)
                )
            except Exception:
                pass

        self._k3d_set_axes(plot_obj, axes)

        if _created_internally:
            plot_obj.display()
        return plot_obj
