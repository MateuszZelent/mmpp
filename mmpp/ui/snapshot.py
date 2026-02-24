"""Shared snapshot loading and rendering helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from ..pyzfn.snapshot import _create_quiver_grid_and_scale, _vector_field_to_rgb


def vector_field_to_rgb(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert vector-field components to RGB + alpha arrays."""
    return _vector_field_to_rgb(u, v, w)


def create_quiver_grid_and_scale(
    *,
    ax,
    alphas: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
) -> None:
    """Render decimated quiver overlay for a vector field."""
    _create_quiver_grid_and_scale(ax=ax, alphas=alphas, u=u, v=v, dx=dx, dy=dy)


class SnapshotCache:
    """Lazy-loading LRU cache for magnetization snapshots."""

    def __init__(
        self,
        job_result,
        *,
        dset: str = "m",
        slice_info: Any | None = None,
        max_cached: int = 50,
        frame_keys: dict[int, str] | None = None,
        zarr_group=None,
    ):
        self._job = job_result
        self._dset = dset
        self._slice_info = slice_info
        self._max = int(max(1, max_cached))
        self._lru: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._frame_keys = frame_keys
        self._zarr_group = zarr_group

    def clear(self) -> None:
        self._lru.clear()

    def _load_raw_frame(
        self,
        frame_idx: int,
        *,
        z_layer: int | str = 0,
        roi: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        if self._frame_keys is not None and self._zarr_group is not None:
            key = self._frame_keys.get(int(frame_idx))
            if key is None:
                available = sorted(self._frame_keys.keys())
                closest = min(available, key=lambda value: abs(value - int(frame_idx)))
                key = self._frame_keys[closest]

            arr = self._zarr_group[key]
            data = np.asarray(arr, dtype=float)
            while data.ndim > 4 and data.shape[0] == 1:
                data = data[0]

            if data.ndim == 4:
                z_count = data.shape[0]
                if z_layer == "all":
                    frame = data.mean(axis=0)
                else:
                    z_idx = int(np.clip(int(z_layer), 0, z_count - 1))
                    frame = data[z_idx]
            elif data.ndim == 3:
                frame = data
            else:
                raise ValueError(f"Unexpected shape {tuple(data.shape)} for key '{key}'")

            if roi is not None:
                x0, x1, y0, y1 = roi
                frame = frame[y0:y1, x0:x1, :]
            if frame.size == 0:
                raise ValueError(f"Snapshot ROI produced empty frame for key '{key}'")
            return frame

        dset_obj = self._job.get_raw(self._dset)
        data_obj = dset_obj[self._slice_info] if self._slice_info is not None else dset_obj
        shape = tuple(getattr(data_obj, "shape", ()))

        if len(shape) == 5:
            t_count, z_count, _, _, c_count = shape
            if c_count < 1:
                raise ValueError(f"Dataset '{self._dset}' has no vector components")

            t_idx = int(np.clip(int(frame_idx), 0, t_count - 1))
            n_comp = max(1, min(3, int(c_count)))
            if z_layer == "all":
                frame = np.asarray(data_obj[t_idx, :, :, :, :n_comp], dtype=float).mean(axis=0)
            else:
                z_idx = int(np.clip(int(z_layer), 0, z_count - 1))
                frame = np.asarray(data_obj[t_idx, z_idx, :, :, :n_comp], dtype=float)
        elif len(shape) == 4:
            t_count, _, _, c_count = shape
            if c_count < 1:
                raise ValueError(f"Dataset '{self._dset}' has no vector components")

            t_idx = int(np.clip(int(frame_idx), 0, t_count - 1))
            n_comp = max(1, min(3, int(c_count)))
            frame = np.asarray(data_obj[t_idx, :, :, :n_comp], dtype=float)
        else:
            raise ValueError(
                f"Dataset '{self._dset}' must be 4D/5D magnetization for snapshots"
            )

        if frame.shape[-1] < 3:
            padded = np.zeros(frame.shape[:-1] + (3,), dtype=float)
            padded[..., : frame.shape[-1]] = frame
            frame = padded

        if roi is not None:
            x0, x1, y0, y1 = roi
            frame = frame[y0:y1, x0:x1, :]

        if frame.size == 0:
            raise ValueError("Snapshot ROI produced empty frame")

        return frame

    def get_frame(
        self,
        frame_idx: int,
        *,
        component: str,
        z_layer: int | str,
        roi: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        key = (int(frame_idx), str(component), str(z_layer), tuple(roi) if roi else None)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]

        frame = self._load_raw_frame(frame_idx, z_layer=z_layer, roi=roi)
        self._lru[key] = frame
        if len(self._lru) > self._max:
            self._lru.popitem(last=False)
        return frame


def _component_image(frame: np.ndarray, component: str) -> np.ndarray:
    comp = str(component).lower()
    if comp in {"x", "mx"}:
        return np.asarray(frame[:, :, 0], dtype=float)
    if comp in {"y", "my"}:
        return np.asarray(frame[:, :, 1], dtype=float)
    if comp in {"z", "mz"}:
        return np.asarray(frame[:, :, 2], dtype=float)
    if comp in {"norm", "magnitude", "|m|", "snapshot"}:
        return np.linalg.norm(frame, axis=-1)
    return np.linalg.norm(frame, axis=-1)


def render_snapshot(
    ax,
    frame: np.ndarray,
    *,
    component: str = "snapshot",
    field_value: float | None = None,
    field_unit: str = "input",
    dx: float = 1.0,
    dy: float = 1.0,
    cmap: str = "viridis",
) -> None:
    """Render a snapshot panel for selected component."""
    ax.clear()
    comp = str(component).lower()
    x_size = int(frame.shape[1])
    y_size = int(frame.shape[0])

    x_nm = float(x_size) * float(dx) * 1e9
    y_nm = float(y_size) * float(dy) * 1e9
    max_nm = max(x_nm, y_nm)

    if max_nm >= 1000.0:
        axis_unit = "um"
        scale = 1e6
        qdx = float(dx) * 1e-3
        qdy = float(dy) * 1e-3
    else:
        axis_unit = "nm"
        scale = 1e9
        qdx = float(dx)
        qdy = float(dy)

    extent_x = float(x_size) * float(dx) * scale
    extent_y = float(y_size) * float(dy) * scale

    if comp == "snapshot":
        u, v, w = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        rgb, alphas = vector_field_to_rgb(u, v, w)
        if axis_unit == "um":
            quiver_u = np.asarray(u, dtype=float) * 1e-3
            quiver_v = np.asarray(v, dtype=float) * 1e-3
        else:
            quiver_u = u
            quiver_v = v

        create_quiver_grid_and_scale(
            ax=ax,
            alphas=alphas,
            u=quiver_u,
            v=quiver_v,
            dx=qdx,
            dy=qdy,
        )
        ax.imshow(
            rgb,
            interpolation="none",
            origin="lower",
            aspect="equal",
            extent=(0, extent_x, 0, extent_y),
        )
    else:
        image = _component_image(frame, comp)
        ax.imshow(
            image,
            cmap=cmap,
            interpolation="none",
            origin="lower",
            aspect="equal",
            extent=(0, extent_x, 0, extent_y),
        )

    if field_value is not None and np.isfinite(float(field_value)):
        field_unit_text = str(field_unit or "").strip()
        suffix = f" {field_unit_text}" if field_unit_text else ""
        ax.set_title(f"Field: {float(field_value):.5g}{suffix}")
    else:
        ax.set_title(f"Component: {component}")

    ax.set_xlabel(rf"$x$ ({axis_unit})")
    ax.set_ylabel(rf"$y$ ({axis_unit})")


__all__ = [
    "SnapshotCache",
    "vector_field_to_rgb",
    "create_quiver_grid_and_scale",
    "render_snapshot",
]
