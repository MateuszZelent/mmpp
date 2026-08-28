"""K3D mixin for dataset plotting."""

from __future__ import annotations

import colorsys
from typing import Any

import numpy as np

from .dataset_geometry import (
    _dataset_attrs,
    resolve_dataset_geometry,
    source_spatial_axes,
)
from .dataset_plotting_core import DatasetPlotCoreMixin


class DatasetPlotK3DMixin(DatasetPlotCoreMixin):
    @staticmethod
    def _k3d_dataset_attrs(dataset_obj) -> Any:
        return _dataset_attrs(dataset_obj)

    @staticmethod
    def _k3d_source_shape(dataset_obj) -> tuple[int, ...] | None:
        source = getattr(dataset_obj, "zarr_array", None)
        shape = getattr(source, "shape", None)
        if shape is None:
            shape = getattr(dataset_obj, "shape", None)
        if shape is None:
            return None
        return tuple(int(v) for v in shape)

    @staticmethod
    def _k3d_source_spatial_axes(
        source_shape: tuple[int, ...] | None,
        attrs: Any = None,
    ) -> dict[str, int | None] | None:
        return source_spatial_axes(source_shape, attrs)

    def _k3d_source_spatial_shape(
        self,
        dataset_obj=None,
    ) -> tuple[int, int, int] | None:
        obj = self._dataset if dataset_obj is None else dataset_obj
        source_shape = self._k3d_source_shape(obj)
        spatial_axes = self._k3d_source_spatial_axes(
            source_shape, self._k3d_dataset_attrs(obj)
        )
        if source_shape is None or spatial_axes is None:
            return None
        return (
            int(source_shape[spatial_axes["z"]])
            if spatial_axes["z"] is not None
            else 1,
            int(source_shape[spatial_axes["y"]])
            if spatial_axes["y"] is not None
            else 1,
            int(source_shape[spatial_axes["x"]])
            if spatial_axes["x"] is not None
            else 1,
        )

    @staticmethod
    def _k3d_attr_triplet(
        attrs: Any,
        *,
        key: str,
    ) -> tuple[float, float, float] | None:
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
        shape_zyx: tuple[int, ...],
        *,
        multiplier: float | None = None,
        dataset_obj=None,
        include_slice: bool = True,
    ) -> tuple[
        list[float],
        list[str],
        float,
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Resolve k3d bounds/labels for a volume with shape (z, y, x)."""
        obj = self._dataset if dataset_obj is None else dataset_obj
        geometry = resolve_dataset_geometry(obj, include_slice=include_slice)
        if geometry.axes:
            x_min_m, x_max_m, y_min_m, y_max_m, z_min_m, z_max_m = (
                geometry.bounds_xyz_m()
            )
            dx_eff_m, dy_eff_m, dz_eff_m = geometry.cell_xyz_m()
            x_name, y_name, z_name = geometry.axis_names_xyz()
        else:
            attrs = self._k3d_dataset_attrs(obj)
            if hasattr(attrs, "get"):
                dx_eff_m = float(attrs.get("dx", 1e-9))
                dy_eff_m = float(attrs.get("dy", 1e-9))
                dz_eff_m = float(attrs.get("dz", 1e-9))
                x_name = str(attrs.get("x_name", "x"))
                y_name = str(attrs.get("y_name", "y"))
                z_name = str(attrs.get("z_name", "z"))
            else:
                dx_eff_m = dy_eff_m = dz_eff_m = 1e-9
                x_name, y_name, z_name = "x", "y", "z"
            nz, ny, nx = (
                max(int(shape_zyx[0]), 1),
                max(int(shape_zyx[1]), 1),
                max(int(shape_zyx[2]), 1),
            )
            x_min_m, x_max_m = 0.0, float(nx) * float(dx_eff_m)
            y_min_m, y_max_m = 0.0, float(ny) * float(dy_eff_m)
            z_min_m, z_max_m = 0.0, float(nz) * float(dz_eff_m)

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

        # For degenerate (zero-size) axes use the actual voxel cell size,
        # not an arbitrary 1-unit fallback.
        cell_x_u = float(dx_eff_m / m)
        cell_y_u = float(dy_eff_m / m)
        cell_z_u = float(dz_eff_m / m)
        if extent_x <= 0.0:
            x_max_u = x_min_u + max(cell_x_u, 1e-12)
            extent_x = x_max_u - x_min_u
        if extent_y <= 0.0:
            y_max_u = y_min_u + max(cell_y_u, 1e-12)
            extent_y = y_max_u - y_min_u
        if extent_z <= 0.0:
            z_max_u = z_min_u + max(cell_z_u, 1e-12)
            extent_z = z_max_u - z_min_u

        bounds = [x_min_u, x_max_u, y_min_u, y_max_u, z_min_u, z_max_u]
        unit_label = self._unit_label_from_multiplier(m)
        axes = [
            rf"{x_name}\,(\text{{{unit_label}}})",
            rf"{y_name}\,(\text{{{unit_label}}})",
            rf"{z_name}\,(\text{{{unit_label}}})",
        ]
        cell = (cell_x_u, cell_y_u, cell_z_u)
        extents = (float(extent_x), float(extent_y), float(extent_z))
        return bounds, axes, float(m), extents, cell

    @staticmethod
    def _k3d_set_axes(plot_obj, axes: list[str]) -> None:
        try:
            plot_obj.axes = list(axes)
        except Exception:
            pass

    @staticmethod
    def _k3d_nonzero_voxels(
        scalar: np.ndarray,
        *,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """Build the binary voxel mask used by ``plot.k3d.nonzero``.

        This mirrors the notebook reconstruction of
        ``discretisedfield.Field.norm.k3d.nonzero()``. The only difference is
        that mmpp scalar volumes already arrive in ``(z, y, x)`` order, so no
        additional axis swap is required before passing them to
        ``k3d.voxels``.
        """
        arr = np.asarray(scalar, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected a 3d scalar volume, got shape {arr.shape}.")

        voxels = np.ones(arr.shape, dtype=np.uint8)
        hidden = ~np.isfinite(arr)
        if float(threshold) > 0.0:
            hidden |= np.abs(arr) <= float(threshold)
        else:
            hidden |= arr == 0
        voxels[hidden] = 0
        return voxels

    def _k3d_expand_singleton_bounds_to_source(
        self,
        bounds: list[float],
        shape_zyx: tuple[int, ...],
        *,
        multiplier: float,
        dataset_obj=None,
    ) -> bool:
        """Expand single-cell spatial slices to the full source thickness.

        Thin one-cell volumes are technically correct but produce tiny slabs and
        poor camera/grid defaults in K3D. For visualisation we expand only the
        singleton spatial axis/axes to the full source extent while keeping the
        in-plane bounds unchanged.
        """
        obj = self._dataset if dataset_obj is None else dataset_obj
        source_geometry = resolve_dataset_geometry(obj, include_slice=False)
        if not source_geometry.axes:
            return False

        current_sizes = {
            "z": max(int(shape_zyx[0]), 1),
            "y": max(int(shape_zyx[1]), 1),
            "x": max(int(shape_zyx[2]), 1),
        }
        bound_index = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}

        expanded = False
        for axis in ("x", "y", "z"):
            source_axis = source_geometry.axes.get(axis)
            if source_axis is None:
                continue
            if current_sizes[axis] != 1 or int(source_axis.size) <= 1:
                continue

            lo_idx, hi_idx = bound_index[axis]
            bounds[lo_idx] = float(source_axis.min_m / multiplier)
            bounds[hi_idx] = float(source_axis.max_m / multiplier)
            expanded = True

        return expanded

    @staticmethod
    def _k3d_set_exact_grid(plot_obj, bounds: list[float]) -> None:
        """Disable K3D grid auto-fit and set grid to exact data bounds.

        K3D's default ``gridAutoFit`` rounds the scene bounding box to
        *nice* intervals using ``Math.ceil`` / ``Math.floor``, which can
        visually extend grid planes and axis labels beyond the actual data
        extent (e.g.  Z max 3.5 nm → grid label 4.0 nm).
        """
        try:
            plot_obj.grid_auto_fit = False
        except Exception:
            pass
        try:
            plot_obj.grid = list(bounds)
        except Exception:
            pass

    @staticmethod
    def _k3d_snap_scene_bounds(plot_obj, bounds: list[float]) -> None:
        """Add invisible anchors to snap K3D's auto-fit grid to integer ticks.

        K3D's ``gridAutoFit`` computes the grid from the scene bounding box.
        When data bounds don't align with major tick marks (e.g. Z max = 3.5
        instead of 4.0), the grid shows a partial cell.  This method adds
        invisible zero-opacity points at integer-snapped positions to expand
        the scene bounding box to clean grid boundaries.
        """
        import math

        try:
            import k3d
        except ImportError:
            return

        snapped = [
            math.floor(bounds[0]),  # x_min
            math.ceil(bounds[1]),  # x_max
            math.floor(bounds[2]),  # y_min
            math.ceil(bounds[3]),  # y_max
            math.floor(bounds[4]),  # z_min
            math.ceil(bounds[5]),  # z_max
        ]

        # Only add anchors if snapping actually changed any bound
        if snapped == [float(b) for b in bounds]:
            return

        anchor = np.array(
            [
                [snapped[0], snapped[2], snapped[4]],
                [snapped[1], snapped[3], snapped[5]],
            ],
            dtype=np.float32,
        )
        try:
            plot_obj += k3d.points(
                anchor,
                point_size=0.0,
                opacity=0.0,
                color=0x000000,
            )
        except Exception:
            pass

    @staticmethod
    def _k3d_apply_thin_slice_scene_defaults(
        plot_obj,
        bounds: list[float],
        shape_zyx: tuple[int, ...] = (1, 1, 1),
        *,
        expand_fraction: float = 0.1,
    ) -> None:
        """Adjust scene for thin (single-cell) spatial slices.

        Instead of disabling ``grid_auto_fit`` (which breaks grid rendering
        in K3D ≤ 2.14), we expand the thin dimension's bounds to a visible
        fraction of the largest in-plane extent.  This makes thin-film data
        visible in 3D without losing the grid.
        """
        nz, ny, nx = (max(int(v), 1) for v in shape_zyx)
        extent_x = abs(bounds[1] - bounds[0])
        extent_y = abs(bounds[3] - bounds[2])
        extent_z = abs(bounds[5] - bounds[4])
        in_plane = max(extent_x, extent_y, extent_z)
        target = max(in_plane * expand_fraction, 1e-12)

        if nz == 1 and extent_z < target:
            centre = (bounds[4] + bounds[5]) / 2.0
            bounds[4] = centre - target / 2.0
            bounds[5] = centre + target / 2.0
        if ny == 1 and extent_y < target:
            centre = (bounds[2] + bounds[3]) / 2.0
            bounds[2] = centre - target / 2.0
            bounds[3] = centre + target / 2.0
        if nx == 1 and extent_x < target:
            centre = (bounds[0] + bounds[1]) / 2.0
            bounds[0] = centre - target / 2.0
            bounds[1] = centre + target / 2.0

        try:
            plot_obj.camera_up_axis = "z"
        except Exception:
            pass

    @staticmethod
    def _k3d_center_grid_bounds(
        bounds: list[float],
        shape_zyx: tuple[int, ...],
        cell_xyz: tuple[float, float, float],
    ) -> list[float]:
        """Inset the display grid to voxel centres without changing voxel bounds."""
        nz, ny, nx = (max(int(v), 1) for v in shape_zyx)
        dx_u, dy_u, dz_u = (float(v) for v in cell_xyz)
        grid = [float(v) for v in bounds]
        if nx > 1:
            grid[0] += 0.5 * dx_u
            grid[1] -= 0.5 * dx_u
        if ny > 1:
            grid[2] += 0.5 * dy_u
            grid[3] -= 0.5 * dy_u
        if nz > 1:
            grid[4] += 0.5 * dz_u
            grid[5] -= 0.5 * dz_u
        return grid

    def _k3d_default_stack_axis(self, dataset_obj=None) -> str:
        obj = self._dataset if dataset_obj is None else dataset_obj
        geometry = resolve_dataset_geometry(obj)
        if geometry.axes:
            candidates: list[tuple[int, int, str]] = []
            axis_order = {"z": 0, "y": 1, "x": 2}
            for axis in ("z", "y", "x"):
                axis_geom = geometry.axes.get(axis)
                if axis_geom is None:
                    continue
                size = int(axis_geom.size)
                if size > 1:
                    candidates.append((size, axis_order[axis], axis))
            if candidates:
                candidates.sort()
                return candidates[0][2]
            for axis in ("z", "y", "x"):
                if axis in geometry.axes:
                    return axis
        return "z"

    def _k3d_default_stack_positions(
        self,
        axis: str,
        *,
        dataset_obj=None,
        max_slices: int = 8,
    ) -> list[float]:
        obj = self._dataset if dataset_obj is None else dataset_obj
        geometry = resolve_dataset_geometry(obj)
        axis_geom = geometry.axes.get(axis) if geometry.axes else None
        if axis_geom is None or int(axis_geom.size) <= 0:
            raise ValueError(
                f"Cannot infer default stack positions for axis {axis!r}; "
                "provide positions explicitly."
            )

        size = int(axis_geom.size)
        centers = float(axis_geom.min_m) + (np.arange(size, dtype=float) + 0.5) * float(
            axis_geom.cell_m
        )

        limit = max(int(max_slices), 1)
        if size > limit:
            sample_idx = np.rint(np.linspace(0, size - 1, limit)).astype(int)
            sample_idx = np.unique(np.clip(sample_idx, 0, size - 1))
            centers = centers[sample_idx]

        return [float(v) for v in centers]

    @staticmethod
    def _k3d_hls_palette(
        *,
        hue_bins: int = 24,
        lightness_bins: int = 5,
        saturation_bins: int = 2,
    ) -> list[int]:
        total = int(hue_bins) * int(lightness_bins) * int(saturation_bins)
        if total < 1 or total > 255:
            raise ValueError(
                "HLS palette must contain between 1 and 255 visible entries, "
                f"got {total}"
            )

        palette = [0x000000]
        hue_bins = int(hue_bins)
        lightness_bins = int(lightness_bins)
        saturation_bins = int(saturation_bins)
        for sat_idx in range(saturation_bins):
            if saturation_bins <= 1:
                sat = 1.0
            else:
                sat = 0.35 + 0.65 * (float(sat_idx) / float(saturation_bins - 1))
            for light_idx in range(lightness_bins):
                if lightness_bins <= 1:
                    light = 0.5
                else:
                    light = 0.15 + 0.70 * (float(light_idx) / float(lightness_bins - 1))
                for hue_idx in range(hue_bins):
                    hue = (float(hue_idx) + 0.5) / float(hue_bins)
                    rgb = colorsys.hls_to_rgb(hue, light, sat)
                    palette.append(
                        (int(np.clip(rgb[0], 0.0, 1.0) * 255.0) << 16)
                        | (int(np.clip(rgb[1], 0.0, 1.0) * 255.0) << 8)
                        | int(np.clip(rgb[2], 0.0, 1.0) * 255.0)
                    )
        return palette

    def _k3d_magnetization_voxels_hls(
        self,
        volume: np.ndarray,
        *,
        visible_mask: np.ndarray | None = None,
        vdim_mapping: dict[Any, Any] | None = None,
        hue_bins: int = 24,
        lightness_bins: int = 5,
        saturation_bins: int = 2,
    ) -> tuple[np.ndarray, list[int]]:
        vec = np.asarray(volume, dtype=np.float32)
        if vec.ndim != 4 or vec.shape[-1] < 2:
            raise ValueError(
                "HLS magnetization colouring expects a vector volume (z, y, x, c>=2), "
                f"got {vec.shape}."
            )

        src_n_comp = int(vec.shape[-1])
        comp_mapping = self._resolve_vdim_mapping(src_n_comp, vdim_mapping)
        mx_idx = self._resolve_component_index(
            "x", src_n_comp, mapping=comp_mapping, allow_none=False
        )
        my_idx = self._resolve_component_index(
            "y", src_n_comp, mapping=comp_mapping, allow_none=False
        )
        mz_idx = comp_mapping.get("z", None)

        mx = np.asarray(vec[..., mx_idx], dtype=np.float32)
        my = np.asarray(vec[..., my_idx], dtype=np.float32)
        mz = (
            np.asarray(vec[..., mz_idx], dtype=np.float32)
            if mz_idx is not None and int(mz_idx) < src_n_comp
            else np.zeros(vec.shape[:-1], dtype=np.float32)
        )

        hue = np.mod(np.arctan2(my, mx) / (2.0 * np.pi), 1.0)
        lightness = np.clip(0.5 * (np.clip(mz, -1.0, 1.0) + 1.0), 0.0, 1.0)
        saturation = np.clip(np.sqrt(mx**2 + my**2), 0.0, 1.0)
        magnitude = np.sqrt(mx**2 + my**2 + mz**2)

        valid = np.isfinite(hue) & np.isfinite(lightness) & np.isfinite(saturation)
        valid &= np.isfinite(magnitude) & (magnitude > 0.0)
        if visible_mask is not None:
            mask = np.asarray(visible_mask, dtype=bool)
            if mask.shape != magnitude.shape:
                mask = np.broadcast_to(mask, magnitude.shape)
            valid &= mask

        h_bins = max(int(hue_bins), 1)
        l_bins = max(int(lightness_bins), 1)
        s_bins = max(int(saturation_bins), 1)
        total = h_bins * l_bins * s_bins
        if total > 255:
            raise ValueError(
                "Too many HLS quantisation bins for k3d.voxels (max 255 visible colours): "
                f"{h_bins}*{l_bins}*{s_bins}={total}"
            )

        hue_q = np.floor(np.clip(hue, 0.0, 1.0 - 1e-7) * float(h_bins)).astype(np.int32)
        light_q = np.floor(np.clip(lightness, 0.0, 1.0 - 1e-7) * float(l_bins)).astype(
            np.int32
        )
        sat_q = np.floor(np.clip(saturation, 0.0, 1.0 - 1e-7) * float(s_bins)).astype(
            np.int32
        )

        voxels = np.zeros(vec.shape[:-1], dtype=np.uint8)
        encoded = 1 + hue_q + h_bins * (light_q + l_bins * sat_q)
        voxels[valid] = np.asarray(encoded[valid], dtype=np.uint8)

        return voxels, self._k3d_hls_palette(
            hue_bins=h_bins,
            lightness_bins=l_bins,
            saturation_bins=s_bins,
        )

    def _k3d_prepare_interactive_plot(
        self,
        plot_obj,
        *,
        multiplier: float | None,
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
        zero: int | None = None,
        component: int | str | None = None,
        filter_field: Any = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        multiplier: float | None = None,
        interactive_field: Any = None,
        interactive: bool = False,
        hide_zeros: bool = False,
        grid_from_centers: bool = False,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        zlim: tuple | None = None,
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
                f"k3d.scalar expects a 3d scalar volume. Got shape {scalar.shape}."
            )

        visible = self._coerce_mask(filter_field, scalar.shape, t=t, z=0, zero=zero)
        if bool(hide_zeros):
            visible = (
                np.asarray(visible, dtype=bool) & np.isfinite(scalar) & (scalar != 0)
            )
        voxels = self._normalise_to_uint8(
            scalar,
            vmin=vmin,
            vmax=vmax,
            visible_mask=visible,
        )
        shape_zyx = tuple(int(v) for v in scalar.shape)
        bounds, axes, m, _, cell_xyz = self._k3d_resolve_geometry(
            shape_zyx,
            multiplier=multiplier,
        )
        _created_internally = plot is None
        thin_scene = any(int(v) == 1 for v in shape_zyx)
        if _created_internally:
            expanded = self._k3d_expand_singleton_bounds_to_source(
                bounds,
                shape_zyx,
                multiplier=m,
            )
            thin_scene = bool(thin_scene or expanded)

        # Manual axis limit overrides (values in metres, converted to plot units)
        if xlim is not None:
            bounds[0], bounds[1] = float(xlim[0]) / m, float(xlim[1]) / m
        if ylim is not None:
            bounds[2], bounds[3] = float(ylim[0]) / m, float(ylim[1]) / m
        if zlim is not None:
            bounds[4], bounds[5] = float(zlim[0]) / m, float(zlim[1]) / m

        plot_obj = (
            plot
            if plot is not None
            else k3d.plot(name=f"{self._dataset.dataset_name} scalar")
        )
        if interactive or interactive_field is not None:
            self._k3d_prepare_interactive_plot(
                plot_obj,
                multiplier=multiplier,
                interactive_field=interactive_field,
            )

        voxel_kwargs = dict(kwargs)
        voxel_kwargs.setdefault("outlines", bool(thin_scene))

        cmap_int = self._k3d_colormap_int(cmap)
        try:
            plot_obj += k3d.voxels(
                voxels,
                color_map=cmap_int,
                bounds=bounds,
                **voxel_kwargs,
            )
        except Exception:
            plot_obj += k3d.voxels(
                voxels, color_map=cmap_int, bounds=bounds, **voxel_kwargs
            )

        self._k3d_set_axes(plot_obj, axes)
        if thin_scene:
            self._k3d_apply_thin_slice_scene_defaults(plot_obj, bounds, shape_zyx)
        elif _created_internally:
            self._k3d_snap_scene_bounds(plot_obj, bounds)
        if bool(grid_from_centers):
            grid_bounds = self._k3d_center_grid_bounds(bounds, shape_zyx, cell_xyz)
            self._k3d_set_exact_grid(plot_obj, grid_bounds)

        if _created_internally and hasattr(plot_obj, "display"):
            plot_obj.display()
            return None
        return plot_obj

    def _k3d_nonzero_impl(
        self,
        *,
        plot=None,
        t: int = -1,
        zero: int | None = None,
        component: int | str | None = None,
        threshold: float = 0.0,
        color: int = 0x4C72B0,
        multiplier: float | None = None,
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
                f"k3d.nonzero expects a 3d scalar volume. Got shape {scalar.shape}."
            )

        voxels = self._k3d_nonzero_voxels(scalar, threshold=threshold)
        bounds, axes, m, _, _ = self._k3d_resolve_geometry(
            tuple(int(v) for v in scalar.shape),
            multiplier=multiplier,
        )
        _created_internally = plot is None
        thin_scene = any(int(v) == 1 for v in scalar.shape)
        if _created_internally:
            expanded = self._k3d_expand_singleton_bounds_to_source(
                bounds,
                tuple(int(v) for v in scalar.shape),
                multiplier=m,
            )
            thin_scene = bool(thin_scene or expanded)
        plot_obj = (
            plot
            if plot is not None
            else k3d.plot(name=f"{self._dataset.dataset_name} nonzero")
        )
        if interactive or interactive_field is not None:
            self._k3d_prepare_interactive_plot(
                plot_obj,
                multiplier=multiplier,
                interactive_field=interactive_field,
            )

        voxel_kwargs = dict(kwargs)
        voxel_kwargs.setdefault("outlines", bool(thin_scene))

        try:
            plot_obj += k3d.voxels(
                voxels,
                color_map=int(color),
                bounds=bounds,
                **voxel_kwargs,
            )
        except Exception:
            plot_obj += k3d.voxels(
                voxels, color_map=int(color), bounds=bounds, **voxel_kwargs
            )

        self._k3d_set_axes(plot_obj, axes)
        if thin_scene:
            self._k3d_apply_thin_slice_scene_defaults(
                plot_obj,
                bounds,
                tuple(int(v) for v in scalar.shape),
            )
        elif _created_internally:
            self._k3d_snap_scene_bounds(plot_obj, bounds)
        if _created_internally and hasattr(plot_obj, "display"):
            plot_obj.display()
            return None
        return plot_obj

    def _k3d_vector_impl(
        self,
        *,
        plot=None,
        t: int = -1,
        zero: int | None = None,
        vdims: tuple[int | str | None, int | str | None, int | str | None]
        | None = None,
        vdim_mapping: dict[Any, Any] | None = None,
        color_field: Any = None,
        cmap: str = "viridis",
        head_size: float = 1.0,
        points: bool = True,
        point_size: float | None = None,
        vector_multiplier: float | None = None,
        vector_scale: float = 1.0,
        multiplier: float | None = None,
        quiver_density: int | None = None,
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

        origins: Any = np.stack(
            [
                float(bounds[0]) + (xx + 0.5) * float(dx_u),
                float(bounds[2]) + (yy + 0.5) * float(dy_u),
                float(bounds[4]) + (zz + 0.5) * float(dz_u),
            ],
            axis=-1,
        ).reshape(-1, 3)
        vectors: Any = np.stack([u, v, w], axis=-1).reshape(-1, 3)
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
        plot_obj = (
            plot
            if plot is not None
            else k3d.plot(name=f"{self._dataset.dataset_name} vector")
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
                    color_volume = self._component_volume(
                        volume, color_field, default="norm"
                    )
            else:
                color_volume = self._coerce_scalar_field(
                    color_field,
                    vec.shape[:-1],
                    t=t,
                    z=0,
                    zero=zero,
                    component=None,
                    default="norm",
                )

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

        if _created_internally and hasattr(plot_obj, "display"):
            plot_obj.display()
            return None
        return plot_obj

    def _k3d_magnetization_impl(
        self,
        *,
        plot=None,
        t: int = -1,
        zero: int | None = None,
        style: int | str | None = "hsl",
        show_vectors: bool = True,
        filter_field: Any = None,
        vdims: tuple[int | str | None, int | str | None, int | str | None]
        | None = None,
        vdim_mapping: dict[Any, Any] | None = None,
        color_field: Any = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        hue_bins: int = 24,
        lightness_bins: int = 5,
        saturation_bins: int = 2,
        head_size: float = 1.8,
        points: bool = False,
        point_size: float | None = None,
        vector_multiplier: float | None = None,
        vector_scale: float = 1.0,
        multiplier: float | None = None,
        quiver_density: int | None = None,
        min_magnitude: float = 1e-12,
        vector_color: int = 0xFFFFFF,
        voxel_opacity: float = 0.35,
        interactive_field: Any = None,
        interactive: bool = False,
        **kwargs,
    ):
        style_key = "hsl" if style is None else str(style).strip().lower()
        if style_key in {"hsl", "hls", "orientation"}:
            try:
                import k3d
            except Exception as exc:
                raise ImportError(
                    "k3d is required for plot.k3d.magnetization(). Install with: pip install k3d"
                ) from exc

            volume = self._extract_volume(t=t, zero=zero)
            if volume.ndim != 4 or volume.shape[-1] < 2:
                raise ValueError(
                    "k3d.magnetization(style='hsl') expects a vector volume "
                    f"(z, y, x, c>=2), got {volume.shape}"
                )

            visible_mask = self._coerce_mask(
                filter_field,
                volume.shape[:3],
                t=t,
                z=0,
                zero=zero,
            )
            voxels_arr, color_map = self._k3d_magnetization_voxels_hls(
                volume,
                visible_mask=visible_mask,
                vdim_mapping=vdim_mapping,
                hue_bins=int(hue_bins),
                lightness_bins=int(lightness_bins),
                saturation_bins=int(saturation_bins),
            )
            bounds, axes, _, _, (dx_u, dy_u, dz_u) = self._k3d_resolve_geometry(
                tuple(int(v) for v in volume.shape[:3]),
                multiplier=multiplier,
            )

            _created_internally = plot is None
            plot_obj = (
                plot
                if plot is not None
                else k3d.plot(name=f"{self._dataset.dataset_name} magnetization")
            )
            if interactive or interactive_field is not None:
                self._k3d_prepare_interactive_plot(
                    plot_obj,
                    multiplier=multiplier,
                    interactive_field=interactive_field,
                )

            vox_kw: dict[str, Any] = dict(kwargs)
            vox_kw.setdefault("outlines", False)
            if float(voxel_opacity) < 1.0:
                vox_kw["opacity"] = float(voxel_opacity)
            try:
                plot_obj += k3d.voxels(
                    voxels_arr,
                    color_map=color_map,
                    bounds=bounds,
                    **vox_kw,
                )
            except Exception:
                plot_obj += k3d.voxels(voxels_arr, color_map=color_map, bounds=bounds)

            if show_vectors:
                vec = np.asarray(volume, dtype=np.float32)
                src_n_comp = int(vec.shape[-1])
                comp_mapping = self._resolve_vdim_mapping(src_n_comp, vdim_mapping)
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
                        raise ValueError(
                            f"At least one entry in {vdims=} must be not None"
                        )

                u = (
                    np.asarray(vec[::stepz, ::stepy, ::stepx, ix], dtype=np.float32)
                    if ix is not None
                    else np.zeros(
                        (len(z_idx), len(y_idx), len(x_idx)), dtype=np.float32
                    )
                )
                v = (
                    np.asarray(vec[::stepz, ::stepy, ::stepx, iy], dtype=np.float32)
                    if iy is not None
                    else np.zeros(
                        (len(z_idx), len(y_idx), len(x_idx)), dtype=np.float32
                    )
                )
                w = (
                    np.asarray(vec[::stepz, ::stepy, ::stepx, iz], dtype=np.float32)
                    if iz is not None
                    else np.zeros(
                        (len(z_idx), len(y_idx), len(x_idx)), dtype=np.float32
                    )
                )

                magnitude = np.sqrt(u**2 + v**2 + w**2)
                sampled_voxels = voxels_arr[::stepz, ::stepy, ::stepx]
                visible = (
                    np.isfinite(magnitude)
                    & (magnitude >= float(min_magnitude))
                    & (sampled_voxels > 0)
                )

                zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing="ij")
                origins: Any = np.stack(
                    [
                        float(bounds[0]) + (xx + 0.5) * float(dx_u),
                        float(bounds[2]) + (yy + 0.5) * float(dy_u),
                        float(bounds[4]) + (zz + 0.5) * float(dz_u),
                    ],
                    axis=-1,
                ).reshape(-1, 3)
                vectors: Any = np.stack([u, v, w], axis=-1).reshape(-1, 3)
                visible_flat = visible.reshape(-1)

                if vector_multiplier is None:
                    cell_min = max(min(abs(dx_u), abs(dy_u), abs(dz_u)), 1e-12)
                    vmax_vec = (
                        float(np.nanmax(np.abs(vectors))) if vectors.size > 0 else 1.0
                    )
                    vector_multiplier = vmax_vec / max(cell_min, 1e-12)
                    if not np.isfinite(vector_multiplier) or vector_multiplier <= 0:
                        vector_multiplier = 1.0

                vectors = (vectors / float(vector_multiplier)) * float(vector_scale)
                origins = np.asarray(origins[visible_flat], dtype=np.float32)
                vectors = np.asarray(vectors[visible_flat], dtype=np.float32)

                if origins.size > 0:
                    vector_kwargs: dict[str, Any] = {}
                    if color_field is None:
                        sampled_indices = sampled_voxels.reshape(-1)[visible_flat]
                        colors = [
                            2 * (color_map[int(np.clip(index, 0, len(color_map) - 1))],)
                            for index in sampled_indices
                        ]
                        vector_kwargs["colors"] = colors
                    else:
                        if isinstance(color_field, (int, np.integer, str)):
                            cf_idx = self._resolve_component_index(
                                color_field,
                                src_n_comp,
                                mapping=comp_mapping,
                                allow_none=False,
                            )
                            cf_vol = np.asarray(volume[..., cf_idx], dtype=np.float32)
                        else:
                            cf_vol = self._coerce_scalar_field(
                                color_field,
                                vec.shape[:-1],
                                t=t,
                                z=0,
                                zero=zero,
                                component=None,
                                default="norm",
                            )
                        c_uint8 = self._normalise_to_uint8(
                            cf_vol[::stepz, ::stepy, ::stepx].reshape(-1)[visible_flat],
                            vmin=vmin,
                            vmax=vmax,
                        )
                        cmap_int = self._k3d_colormap_int(cmap)
                        vector_kwargs["colors"] = [
                            2 * (cmap_int[int(np.clip(value, 0, len(cmap_int) - 1))],)
                            for value in c_uint8
                        ]

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
                            point_size = max(
                                min(abs(dx_u), abs(dy_u), abs(dz_u)) / 4.0,
                                1e-6,
                            )
                        try:
                            plot_obj += k3d.points(
                                origins,
                                color=int(vector_color),
                                point_size=float(point_size),
                            )
                        except Exception:
                            pass

            self._k3d_set_axes(plot_obj, axes)
            if _created_internally and hasattr(plot_obj, "display"):
                plot_obj.display()
                return None
            return plot_obj

        scalar_component = style if style is not None else "mz"
        if show_vectors:
            return self._k3d_voxels_vectors_impl(
                plot=plot,
                t=t,
                zero=zero,
                scalar_component=scalar_component,
                vdims=vdims,
                vdim_mapping=vdim_mapping,
                color_field=color_field,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                filter_field=filter_field,
                head_size=head_size,
                points=points,
                point_size=point_size,
                vector_multiplier=vector_multiplier,
                vector_scale=vector_scale,
                quiver_density=quiver_density,
                min_magnitude=min_magnitude,
                vector_color=vector_color,
                voxel_opacity=voxel_opacity,
                multiplier=multiplier,
                **kwargs,
            )
        return self._k3d_scalar_impl(
            plot=plot,
            t=t,
            zero=zero,
            component=scalar_component,
            filter_field=filter_field,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            multiplier=multiplier,
            interactive_field=interactive_field,
            interactive=interactive,
            **kwargs,
        )

    def _k3d_stack_impl(
        self,
        *,
        axis: str | None = None,
        positions: Any = None,
        mode: str = "magnetization",
        plot=None,
        slice_thickness: float | None = None,
        slice_kwargs: list[dict[str, Any]] | None = None,
        display: bool | None = None,
        **kwargs,
    ):
        try:
            import k3d
        except Exception as exc:
            raise ImportError(
                "k3d is required for plot.k3d.stack(). Install with: pip install k3d"
            ) from exc

        geometry = resolve_dataset_geometry(self._dataset)
        if axis is None:
            axis_key = self._k3d_default_stack_axis(self._dataset)
        elif geometry.axes:
            axis_key = geometry.canonical_axis(str(axis))
        else:
            axis_key = str(axis).strip().lower()

        if positions is None:
            positions = self._k3d_default_stack_positions(
                axis_key, dataset_obj=self._dataset
            )

        if not isinstance(positions, (list, tuple, np.ndarray)):
            positions = [positions]
        positions = list(positions)
        if not positions:
            raise ValueError("positions must contain at least one selection")

        if slice_kwargs is not None and len(slice_kwargs) != len(positions):
            raise ValueError(
                f"slice_kwargs length {len(slice_kwargs)} must match positions length {len(positions)}"
            )

        mode_key = str(mode).strip().lower()
        valid_modes = {"magnetization", "vector", "scalar", "voxels_vectors", "nonzero"}
        if mode_key not in valid_modes:
            raise ValueError(
                f"Unsupported stack mode {mode!r}. Use one of {sorted(valid_modes)}."
            )

        created_internally = plot is None
        plot_obj = (
            plot
            if plot is not None
            else k3d.plot(name=f"{self._dataset.dataset_name} {mode_key} stack")
        )

        for index, position in enumerate(positions):
            if isinstance(position, (tuple, list)) and len(position) == 2:
                selection = (float(position[0]), float(position[1]))
            else:
                center = float(position)
                if slice_thickness is None:
                    selection = (center, center)
                else:
                    half = 0.5 * float(slice_thickness)
                    selection = (center - half, center + half)

            view = self._dataset.sel(**{axis_key: selection})
            render_kwargs = dict(kwargs)
            if slice_kwargs is not None:
                render_kwargs.update(dict(slice_kwargs[index]))
            render_kwargs["plot"] = plot_obj

            if mode_key == "magnetization":
                view.plot.k3d.magnetization(**render_kwargs)
            elif mode_key == "vector":
                view.plot.k3d.vector(**render_kwargs)
            elif mode_key == "scalar":
                view.plot.k3d.scalar(**render_kwargs)
            elif mode_key == "voxels_vectors":
                view.plot.k3d.voxels_vectors(**render_kwargs)
            elif mode_key == "nonzero":
                view.plot.k3d.nonzero(**render_kwargs)

        if display is None:
            display = created_internally
        if bool(display) and hasattr(plot_obj, "display"):
            plot_obj.display()
            return None
        return plot_obj

    def _k3d_heatmap_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        repeat: int = 1,
        zero: int | None = None,
        component: int | str | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
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
                [
                    grid_x.ravel(),
                    grid_y.ravel(),
                    np.zeros(grid_x.size, dtype=np.float32),
                ],
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
        zero: int | None = None,
        scalar_component: int | str | None = None,
        vdims: tuple[int | str | None, int | str | None, int | str | None]
        | None = None,
        vdim_mapping: dict[Any, Any] | None = None,
        color_field: Any = None,
        cmap: str = "cividis",
        vmin: float | None = None,
        vmax: float | None = None,
        filter_field: Any = None,
        head_size: float = 2.0,
        points: bool = False,
        point_size: float | None = None,
        vector_multiplier: float | None = None,
        vector_scale: float = 1.0,
        quiver_density: int | None = None,
        min_magnitude: float = 0.0,
        vector_color: int = 0xFFFFFF,
        color_vectors_by_scalar: bool = True,
        voxel_opacity: float = 0.6,
        multiplier: float | None = None,
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
                f"voxels_vectors expects a 3-d scalar volume; got shape {scalar.shape}."
            )

        visible_mask = self._coerce_mask(
            filter_field,
            scalar.shape,
            t=t,
            z=0,
            zero=zero,
        )
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
            vm_val = (
                float(np.nanmax(np.abs(vectors_all))) if vectors_all.size > 0 else 1.0
            )
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
        plot_obj = (
            plot
            if plot is not None
            else k3d.plot(name=f"{self._dataset.dataset_name} voxels+vectors")
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
                2 * (cmap_int[int(np.clip(c, 0, len(cmap_int) - 1))],) for c in c_flat
            ]
            arrow_kw["colors"] = colors

        elif color_field is not None:
            if isinstance(color_field, (int, np.integer, str)):
                cf_idx = self._resolve_component_index(
                    color_field, src_n_comp, mapping=comp_mapping, allow_none=False
                )
                cf_vol = np.asarray(volume[..., cf_idx], dtype=np.float32)
            else:
                cf_vol = self._coerce_scalar_field(
                    color_field,
                    vec.shape[:-1],
                    t=t,
                    z=0,
                    zero=zero,
                    component=None,
                    default="norm",
                )
            c_flat = self._normalise_to_uint8(
                cf_vol[::stepz, ::stepy, ::stepx].reshape(-1)[visible_flat]
            )
            colors = [
                2 * (cmap_int[int(np.clip(c, 0, len(cmap_int) - 1))],) for c in c_flat
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

        if _created_internally and hasattr(plot_obj, "display"):
            plot_obj.display()
            return None
        return plot_obj
