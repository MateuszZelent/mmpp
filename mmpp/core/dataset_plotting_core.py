"""Shared helpers for dataset plotting backends."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset_geometry import _dataset_attrs, resolve_dataset_geometry


class DatasetPlotCoreMixin:
    _dataset: Any
    _SI_PREFIX_BY_EXP = {
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "u",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
    }

    @staticmethod
    def _normalize_index(index: int, size: int) -> int:
        idx = int(index)
        if idx < 0:
            idx = size + idx
        return int(np.clip(idx, 0, max(size - 1, 0)))

    def _resolve_dx_dy_nm(self) -> tuple[float, float]:
        attrs = _dataset_attrs(self._dataset)
        if hasattr(attrs, "get"):
            dx = float(attrs.get("dx", 1e-9))
            dy = float(attrs.get("dy", 1e-9))
        else:
            dx = 1e-9
            dy = 1e-9
        return dx * 1e9, dy * 1e9

    def _resolve_dx_dy_m(self) -> tuple[float, float]:
        attrs = _dataset_attrs(self._dataset)
        if hasattr(attrs, "get"):
            dx = float(attrs.get("dx", 1e-9))
            dy = float(attrs.get("dy", 1e-9))
        else:
            dx = 1e-9
            dy = 1e-9
        return dx, dy

    def _resolve_axis_names(self) -> tuple[str, str]:
        attrs = _dataset_attrs(self._dataset)
        if hasattr(attrs, "get"):
            x_name = str(attrs.get("x_name", "x"))
            y_name = str(attrs.get("y_name", "y"))
        else:
            x_name, y_name = "x", "y"
        return x_name, y_name

    @classmethod
    def _auto_si_multiplier(cls, lengths_m: tuple[float, ...]) -> float:
        finite = [
            abs(float(value))
            for value in lengths_m
            if np.isfinite(value) and abs(float(value)) > 0.0
        ]
        if not finite:
            return 1.0

        vmax = max(finite)
        exp3 = int(np.floor(np.log10(vmax) / 3.0) * 3)
        exp3 = int(np.clip(exp3, -15, 12))
        return float(10.0**exp3)

    @classmethod
    def _unit_label_from_multiplier(cls, multiplier: float) -> str:
        if multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")

        exp = np.log10(multiplier)
        exp_round = int(round(exp))
        if abs(exp - exp_round) < 1e-10 and exp_round in cls._SI_PREFIX_BY_EXP:
            return f"{cls._SI_PREFIX_BY_EXP[exp_round]}m"
        if np.isclose(multiplier, 1.0):
            return "m"
        return f"{multiplier:g} m"

    def _resolve_plot_geometry(
        self,
        shape_xy: tuple[int, ...],
        *,
        multiplier: float | None = None,
    ) -> tuple[float, float, tuple[float, float, float, float], float, str]:
        ny, nx = int(shape_xy[0]), int(shape_xy[1])
        geometry = resolve_dataset_geometry(self._dataset, include_slice=True)

        if geometry.axes:
            x_min_m = float(geometry.axes["x"].min_m)
            y_min_m = float(geometry.axes["y"].min_m)
            dx_m = float(geometry.axes["x"].cell_m)
            dy_m = float(geometry.axes["y"].cell_m)
        else:
            dx_m, dy_m = self._resolve_dx_dy_m()
            x_min_m = 0.0
            y_min_m = 0.0

        size_x = float(nx) * dx_m
        size_y = float(ny) * dy_m

        if multiplier is None:
            if geometry.axes:
                m = self._auto_si_multiplier(
                    (
                        float(geometry.axes["x"].extent_m),
                        float(geometry.axes["y"].extent_m),
                    )
                )
            else:
                m = self._auto_si_multiplier((size_x, size_y))
        else:
            m = float(multiplier)
            if m <= 0:
                raise ValueError(f"multiplier must be > 0, got {m}")

        dx_u = dx_m / m
        dy_u = dy_m / m
        x_min_u = float(x_min_m / m)
        y_min_u = float(y_min_m / m)
        extent = (
            x_min_u,
            x_min_u + float(nx) * dx_u,
            y_min_u,
            y_min_u + float(ny) * dy_u,
        )
        unit_label = self._unit_label_from_multiplier(m)
        return dx_u, dy_u, extent, m, unit_label

    def _set_axis_labels(self, ax, unit_label: str) -> None:
        x_name, y_name = self._resolve_axis_names()
        ax.set_xlabel(f"{x_name} ({unit_label})")
        ax.set_ylabel(f"{y_name} ({unit_label})")

    def _extract_frame(
        self,
        *,
        z: int = 0,
        t: int = -1,
        zero: int | None = None,
        dataset_obj=None,
    ) -> np.ndarray:
        dataset = self._dataset if dataset_obj is None else dataset_obj
        data = dataset.numpy(copy=False, squeeze=False)
        arr = np.asarray(data, dtype=np.float32)
        ndim = arr.ndim

        if ndim == 5:
            t_idx = self._normalize_index(t, arr.shape[0])
            z_idx = self._normalize_index(z, arr.shape[1])
            frame = np.asarray(arr[t_idx, z_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                frame = frame - np.asarray(arr[zref_idx, z_idx], dtype=np.float32)
            return frame

        if ndim == 4:
            t_idx = self._normalize_index(t, arr.shape[0])
            frame = np.asarray(arr[t_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                frame = frame - np.asarray(arr[zref_idx], dtype=np.float32)
            return frame

        if ndim == 3:
            if arr.shape[-1] <= 3:
                return arr
            z_idx = self._normalize_index(z, arr.shape[0])
            return np.asarray(arr[z_idx], dtype=np.float32)

        if ndim == 2:
            return arr

        raise ValueError(
            f"Dataset '{self._dataset.dataset_name}' has unsupported shape {arr.shape} for plotting"
        )

    def _extract_sequence(
        self,
        *,
        z: int = 0,
        zero: int | None = None,
        dataset_obj=None,
    ) -> np.ndarray:
        dataset = self._dataset if dataset_obj is None else dataset_obj
        data = dataset.numpy(copy=False, squeeze=False)
        arr = np.asarray(data, dtype=np.float32)

        if arr.ndim == 5:
            z_idx = self._normalize_index(z, arr.shape[1])
            seq = np.asarray(arr[:, z_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                seq = seq - np.asarray(arr[zref_idx, z_idx], dtype=np.float32)
            return seq

        if arr.ndim == 4:
            # Typical magnetization after slicing: (t, y, x, c)
            if arr.shape[-1] <= 4:
                seq = np.asarray(arr, dtype=np.float32)
                if zero is not None:
                    zref_idx = self._normalize_index(zero, arr.shape[0])
                    seq = seq - np.asarray(arr[zref_idx], dtype=np.float32)
                return seq

            # Scalar-with-z volume over time: (t, z, y, x)
            z_idx = self._normalize_index(z, arr.shape[1])
            seq = np.asarray(arr[:, z_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                seq = seq - np.asarray(arr[zref_idx, z_idx], dtype=np.float32)
            return seq

        # No explicit time axis: wrap as single-frame sequence.
        frame = self._extract_frame(z=z, t=-1, zero=None, dataset_obj=dataset)
        if zero is not None:
            frame = frame - frame
        return frame[np.newaxis, ...]

    def _extract_volume(
        self,
        *,
        t: int = -1,
        zero: int | None = None,
        dataset_obj=None,
    ) -> np.ndarray:
        """Extract 3d (scalar) or 4d (vector) volume for volumetric plotting."""
        dataset = self._dataset if dataset_obj is None else dataset_obj
        data = dataset.numpy(copy=False, squeeze=False)
        arr = np.asarray(data, dtype=np.float32)

        if arr.ndim == 5:
            t_idx = self._normalize_index(t, arr.shape[0])
            volume = np.asarray(arr[t_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                volume = volume - np.asarray(arr[zref_idx], dtype=np.float32)
            return volume

        if arr.ndim == 4:
            # Heuristic:
            # - last axis <= 4 -> vector volume (z, y, x, c)
            # - otherwise first axis is treated as time in scalar volume (t, z, y, x)
            if arr.shape[-1] <= 4:
                return np.asarray(arr, dtype=np.float32)
            t_idx = self._normalize_index(t, arr.shape[0])
            volume = np.asarray(arr[t_idx], dtype=np.float32)
            if zero is not None:
                zref_idx = self._normalize_index(zero, arr.shape[0])
                volume = volume - np.asarray(arr[zref_idx], dtype=np.float32)
            return volume

        if arr.ndim == 3:
            return np.asarray(arr, dtype=np.float32)

        raise ValueError(
            f"Dataset '{self._dataset.dataset_name}' has unsupported shape {arr.shape} for volumetric plotting"
        )

    @staticmethod
    def _component_volume(
        volume: np.ndarray,
        component: int | str | np.integer | None,
        *,
        default: str = "norm",
    ) -> np.ndarray:
        """Select scalar component from a volumetric scalar/vector array."""
        arr = np.asarray(volume, dtype=np.float32)

        if arr.ndim == 3:
            return arr

        if arr.ndim != 4:
            raise ValueError(f"Volume must be 3D or 4D, got shape {arr.shape}")

        n_comp = int(arr.shape[-1])
        if n_comp < 1:
            raise ValueError("Vector volume has no components")

        comp = default if component is None else component

        if isinstance(comp, (int, np.integer)) and not isinstance(comp, bool):
            idx = int(comp)
            if idx < 0 or idx >= n_comp:
                raise IndexError(
                    f"component={idx} is out of range for volume with {n_comp} components"
                )
            return np.asarray(arr[..., idx], dtype=np.float32)

        if isinstance(comp, str):
            key = comp.strip().lower()
            mapping = {"x": 0, "mx": 0, "y": 1, "my": 1, "z": 2, "mz": 2}
            if key in mapping:
                idx = mapping[key]
                if idx >= n_comp:
                    raise IndexError(
                        f"component='{comp}' requires component index {idx}, "
                        f"but volume has only {n_comp} components"
                    )
                return np.asarray(arr[..., idx], dtype=np.float32)
            if key in {"norm", "magnitude", "|m|"}:
                return np.linalg.norm(arr[..., : min(3, n_comp)], axis=-1).astype(
                    np.float32,
                    copy=False,
                )

        raise ValueError(
            f"Unsupported volumetric component selector: {component!r}. "
            "Use int, x/y/z, mx/my/mz, norm/magnitude."
        )

    @staticmethod
    def _unwrap_field_like_component(
        field_like: Any,
        component: int | str | None = None,
    ) -> tuple[Any, int | str | None]:
        """Unpack compact field spec ``(field_like, component)`` if provided."""
        source = field_like
        comp = component

        if (
            comp is None
            and isinstance(field_like, tuple)
            and len(field_like) == 2
            and isinstance(field_like[1], (int, np.integer, str))
        ):
            source, comp = field_like

        return source, comp

    @staticmethod
    def _spatial_axes_for_ndim(dataset_obj: Any, spatial_ndim: int):
        geometry = getattr(dataset_obj, "geometry", None)
        if geometry is None or not getattr(geometry, "axes", None):
            return None
        if int(spatial_ndim) == 2:
            return (geometry.axes["y"], geometry.axes["x"])
        if int(spatial_ndim) == 3:
            return (geometry.axes["z"], geometry.axes["y"], geometry.axes["x"])
        return None

    @staticmethod
    def _split_target_shape_and_repeat(
        target_shape: tuple[int, ...],
        base_shape: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if len(target_shape) != len(base_shape):
            return tuple(int(v) for v in target_shape), (1,) * len(target_shape)

        repeat: list[int] = []
        for target, base in zip(target_shape, base_shape, strict=False):
            target_i = int(target)
            base_i = int(base)
            if base_i <= 0 or target_i <= 0 or target_i % base_i != 0:
                return tuple(int(v) for v in target_shape), (1,) * len(target_shape)
            repeat.append(target_i // base_i)

        return tuple(int(v) for v in base_shape), tuple(int(v) for v in repeat)

    @staticmethod
    def _resample_scalar_by_geometry(
        scalar: np.ndarray,
        *,
        source_axes,
        target_axes,
        target_shape: tuple[int, ...],
    ) -> np.ndarray:
        arr = np.asarray(scalar, dtype=np.float32)
        expected = tuple(int(ax.size) for ax in source_axes)
        if arr.shape != expected:
            raise ValueError(
                f"Source scalar shape {arr.shape} does not match source geometry {expected}"
            )

        target_base_shape = tuple(int(ax.size) for ax in target_axes)
        effective_shape, repeats = DatasetPlotCoreMixin._split_target_shape_and_repeat(
            target_shape,
            target_base_shape,
        )

        resampled = arr
        for axis, (target_axis, source_axis) in enumerate(
            zip(target_axes, source_axes, strict=False)
        ):
            count = int(effective_shape[axis])
            centers = float(target_axis.min_m) + (
                np.arange(count, dtype=np.float32) + 0.5
            ) * float(target_axis.cell_m)
            source_cell = max(float(source_axis.cell_m), 1e-30)
            indices = np.floor(
                (centers - float(source_axis.min_m)) / source_cell
            ).astype(int)
            indices = np.clip(indices, 0, max(int(source_axis.size) - 1, 0))
            resampled = np.take(resampled, indices, axis=axis)

        if any(int(rep) > 1 for rep in repeats):
            resampled = np.tile(resampled, repeats)
        return np.asarray(resampled, dtype=np.float32)

    def _coerce_scalar_field(
        self,
        field_like: Any,
        target_shape: tuple[int, ...],
        *,
        t: int = -1,
        z: int = 0,
        zero: int | None = None,
        component: int | str | None = None,
        default: str = "norm",
    ) -> np.ndarray:
        field_like, component = self._unwrap_field_like_component(field_like, component)
        spatial_ndim = len(target_shape)
        if spatial_ndim not in {2, 3}:
            raise ValueError(
                f"Only 2D/3D scalar coercion is supported, got {target_shape}"
            )

        source_axes = None
        if hasattr(field_like, "numpy"):
            if spatial_ndim == 2:
                source = self._extract_frame(
                    z=z,
                    t=t,
                    zero=zero,
                    dataset_obj=field_like,
                )
                scalar = self._component_image(source, component, default=default)
            else:
                source = self._extract_volume(
                    t=t,
                    zero=zero,
                    dataset_obj=field_like,
                )
                scalar = self._component_volume(source, component, default=default)
            source_axes = self._spatial_axes_for_ndim(field_like, spatial_ndim)
        else:
            scalar = np.asarray(field_like, dtype=np.float32)
            scalar = np.squeeze(scalar)
            if scalar.ndim == spatial_ndim + 1 and scalar.shape[-1] <= 4:
                if spatial_ndim == 2:
                    scalar = self._component_image(scalar, component, default=default)
                else:
                    scalar = self._component_volume(scalar, component, default=default)

        scalar = np.asarray(scalar, dtype=np.float32)
        target_shape = tuple(int(v) for v in target_shape)
        if scalar.shape == target_shape:
            return scalar

        target_axes = self._spatial_axes_for_ndim(self._dataset, spatial_ndim)
        if source_axes is not None and target_axes is not None:
            return self._resample_scalar_by_geometry(
                scalar,
                source_axes=source_axes,
                target_axes=target_axes,
                target_shape=target_shape,
            )

        try:
            return np.asarray(np.broadcast_to(scalar, target_shape), dtype=np.float32)
        except ValueError as exc:
            raise ValueError(
                f"Field shape {scalar.shape} is not compatible with target shape {target_shape}"
            ) from exc

    def _coerce_mask(
        self,
        mask_like: Any,
        target_shape: tuple[int, ...],
        *,
        t: int = -1,
        z: int = 0,
        zero: int | None = None,
    ) -> np.ndarray:
        """Convert mask-like input to boolean array broadcastable to target shape."""
        if mask_like is None:
            return np.ones(target_shape, dtype=bool)

        mask_like, component = self._unwrap_field_like_component(mask_like, None)

        if hasattr(mask_like, "numpy"):
            mask = self._coerce_scalar_field(
                mask_like,
                target_shape,
                t=t,
                z=z,
                zero=zero,
                component=component,
                default="norm",
            )
            return np.asarray(mask != 0, dtype=bool)

        raw = np.asarray(mask_like, dtype=np.float32)
        raw = np.squeeze(raw)
        if raw.ndim == len(target_shape) + 1 and raw.shape[-1] <= 4:
            raw = np.linalg.norm(raw[..., : min(3, raw.shape[-1])], axis=-1)

        if raw.shape != target_shape:
            try:
                raw = np.broadcast_to(raw, target_shape)
            except ValueError as exc:
                raise ValueError(
                    f"Mask shape {raw.shape} is not broadcastable to target shape {target_shape}"
                ) from exc

        return np.asarray(raw != 0, dtype=bool)

    def _resolve_dxyz_nm(self) -> tuple[float, float, float]:
        attrs = _dataset_attrs(self._dataset)
        if hasattr(attrs, "get"):
            dx = float(attrs.get("dx", 1e-9))
            dy = float(attrs.get("dy", 1e-9))
            dz = float(attrs.get("dz", 1e-9))
        else:
            dx = dy = dz = 1e-9
        return dx * 1e9, dy * 1e9, dz * 1e9

    @staticmethod
    def _normalise_to_uint8(
        values: np.ndarray,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        visible_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if vmin is None:
            lo = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
        else:
            lo = float(vmin)
        if vmax is None:
            hi = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
        else:
            hi = float(vmax)

        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = 1.0
        if hi <= lo:
            hi = lo + 1e-12

        scaled = (arr - lo) / (hi - lo)
        scaled = np.clip(scaled, 0.0, 1.0)
        out = (scaled * 254.0 + 1.0).astype(np.uint8)
        out[~np.isfinite(arr)] = 0

        if visible_mask is not None:
            mask = np.asarray(visible_mask, dtype=bool)
            if mask.shape != out.shape:
                mask = np.broadcast_to(mask, out.shape)
            out[~mask] = 0

        return out

    @staticmethod
    def _k3d_colormap_int(cmap_name: str) -> list[int]:
        import matplotlib.colors as mpl_colors
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(cmap_name or "viridis")
        cmap_int: list[int] = []
        for i in range(int(getattr(cmap, "N", 256))):
            rgb = cmap(i)[:3]
            cmap_int.append(int(mpl_colors.rgb2hex(rgb)[1:], 16))
        return cmap_int

    @staticmethod
    def _component_image(
        frame: np.ndarray,
        component: int | str | None,
        *,
        default: str = "norm",
    ) -> np.ndarray:
        if frame.ndim == 2:
            return np.asarray(frame, dtype=np.float32)

        if frame.ndim < 2:
            raise ValueError(f"Frame must be at least 2D, got shape {frame.shape}")

        if frame.ndim > 3:
            image = np.asarray(frame, dtype=np.float32)
            while image.ndim > 2:
                image = image[..., 0]
            return image

        n_comp = int(frame.shape[-1])
        if n_comp < 1:
            raise ValueError("Vector frame has no components")

        comp = default if component is None else component

        if isinstance(comp, (int, np.integer)) and not isinstance(comp, bool):
            idx = int(comp)
            if idx < 0 or idx >= n_comp:
                raise IndexError(
                    f"component={idx} is out of range for frame with {n_comp} components"
                )
            return np.asarray(frame[..., idx], dtype=np.float32)

        if isinstance(comp, str):
            key = comp.strip().lower()
            mapping = {"x": 0, "mx": 0, "y": 1, "my": 1, "z": 2, "mz": 2}
            if key in mapping:
                idx = mapping[key]
                if idx >= n_comp:
                    raise IndexError(
                        f"component='{comp}' requires component index {idx}, "
                        f"but frame has only {n_comp} components"
                    )
                return np.asarray(frame[..., idx], dtype=np.float32)
            if key in {"norm", "magnitude", "|m|", "snapshot"}:
                return np.linalg.norm(frame[..., : min(3, n_comp)], axis=-1).astype(
                    np.float32,
                    copy=False,
                )

        raise ValueError(
            f"Unsupported component selector: {component!r}. "
            "Use int, x/y/z, mx/my/mz, norm/magnitude."
        )

    def _resolve_vdim_mapping(
        self,
        n_comp: int,
        vdim_mapping: dict[Any, Any] | None = None,
    ) -> dict[str, int]:
        """Resolve mapping from symbolic component names to component indices."""
        raw = None
        if vdim_mapping is not None:
            raw = dict(vdim_mapping)
        else:
            attrs = getattr(self._dataset.job_result, "attrs", {})
            if hasattr(attrs, "get"):
                candidate = attrs.get("vdim_mapping", None)
                if isinstance(candidate, dict):
                    raw = dict(candidate)

        mapping: dict[str, int] = {}
        if raw:
            for key, value in raw.items():
                name = str(key).strip().lower()
                if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                    idx = int(value)
                else:
                    continue
                if 0 <= idx < int(n_comp):
                    mapping[name] = idx
                    if name.startswith("m"):
                        mapping[name[1:]] = idx

        # Defaults/fallbacks
        if n_comp >= 1:
            mapping.setdefault("x", 0)
            mapping.setdefault("mx", 0)
        if n_comp >= 2:
            mapping.setdefault("y", 1)
            mapping.setdefault("my", 1)
        if n_comp >= 3:
            mapping.setdefault("z", 2)
            mapping.setdefault("mz", 2)

        return mapping

    @staticmethod
    def _resolve_component_index(
        token: int | str | np.integer | None,
        n_comp: int,
        *,
        mapping: dict[str, int] | None = None,
        allow_none: bool = True,
    ) -> int | None:
        if token is None:
            if allow_none:
                return None
            raise ValueError("Component token cannot be None in this context")

        if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
            idx = int(token)
            if idx < 0 or idx >= n_comp:
                raise IndexError(
                    f"component index {idx} is out of range for {n_comp} components"
                )
            return idx

        if isinstance(token, str):
            key = token.strip().lower()
            local_map = {} if mapping is None else dict(mapping)
            if key in local_map:
                idx = local_map[key]
                if idx >= n_comp:
                    raise IndexError(
                        f"component '{token}' requires index {idx}, "
                        f"but only {n_comp} components are available"
                    )
                return idx
            raise ValueError(
                f"Unsupported component label {token!r}. Use x/y/z or mx/my/mz or int."
            )

        raise TypeError(
            f"Unsupported component token type {type(token).__name__}; use int/str/None"
        )

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        value = str(mode).strip().lower()
        if value in {"snapshot", "vector", "quiver"}:
            return "snapshot"
        if value in {"heatmap", "scalar", "mpl_heatmap"}:
            return "heatmap"
        raise ValueError(
            f"Unsupported render mode: {mode!r}. Use 'snapshot' or 'heatmap'."
        )

    @staticmethod
    def _k3d_colormap(cmap_name: str) -> list[float] | None:
        """Build K3D-compatible colormap from k3d built-ins or matplotlib."""
        name = str(cmap_name or "viridis").strip()
        if not name:
            name = "viridis"

        try:
            import k3d

            mpl_maps = getattr(
                getattr(k3d, "colormaps", None), "matplotlib_color_maps", None
            )
            if mpl_maps is not None:
                candidates = [
                    name,
                    name.capitalize(),
                    name.title().replace("_", ""),
                    name.replace("_", "").capitalize(),
                ]
                for candidate in candidates:
                    if hasattr(mpl_maps, candidate):
                        value = getattr(mpl_maps, candidate)
                        if isinstance(value, list) and value:
                            return [float(v) for v in value]
        except Exception:
            pass

        try:
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap(name)
            samples = 256
            data: list[float] = []
            denom = max(samples - 1, 1)
            for i in range(samples):
                x = float(i) / float(denom)
                r, g, b, _ = cmap(x)
                data.extend([x, float(r), float(g), float(b)])
            return data
        except Exception:
            return None

    @staticmethod
    def _k3d_color_range(
        image: np.ndarray, vmin: float | None, vmax: float | None
    ) -> list[float]:
        if vmin is not None and vmax is not None:
            lo = float(vmin)
            hi = float(vmax)
        else:
            lo = float(np.nanmin(image)) if vmin is None else float(vmin)
            hi = float(np.nanmax(image)) if vmax is None else float(vmax)
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = 1.0
        if hi <= lo:
            hi = lo + 1e-12
        return [lo, hi]

    @staticmethod
    def _k3d_clear_plot(plot, keep_names: tuple[str, ...] = ("total_region",)) -> None:
        try:
            objects = list(getattr(plot, "objects", []))
        except Exception:
            return

        for obj in objects:
            name = getattr(obj, "name", None)
            if name in keep_names:
                continue
            try:
                plot -= obj
            except Exception:
                continue
