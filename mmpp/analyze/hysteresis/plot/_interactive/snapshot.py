"""Snapshot loading and rendering helpers for hysteresis explorer."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from .....ui.snapshot import (
    create_quiver_grid_and_scale as _create_quiver_grid_and_scale,
)
from .....ui.snapshot import (
    vector_field_to_rgb as _vector_field_to_rgb,
)


class SnapshotCache:
    """Lazy-loading LRU cache for magnetization snapshots.

    Supports two modes:
    * Standard: loads frames from a single 4D/5D dataset (``dset``) via
      ``job_result.get_raw(dset)``.
    * zarr_keys: each frame is a *separate* zarr array in ``zarr_group``,
      identified by ``frame_keys[frame_idx]``.  The last time-step of each
      array is used (steady-state snapshot).
    """

    def __init__(
        self,
        job_result,
        *,
        dset: str = "m",
        slice_info: Any | None = None,
        max_cached: int = 50,
        # zarr_keys mode ──────────────────────────────────────────
        frame_keys: dict[int, str] | None = None,
        zarr_group=None,
    ):
        self._job = job_result
        self._dset = dset
        self._slice_info = slice_info
        self._max = int(max(1, max_cached))
        self._lru: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        # zarr_keys mode
        self._frame_keys: dict[int, str] | None = frame_keys
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
        # ── zarr_keys mode: each frame is a separate zarr array ───────────
        if self._frame_keys is not None and self._zarr_group is not None:
            # frame_idx is the candidate index stored in result.frame_index
            key = self._frame_keys.get(int(frame_idx))
            if key is None:
                # fallback: nearest available key
                available = sorted(self._frame_keys.keys())
                closest = min(available, key=lambda k: abs(k - frame_idx))
                key = self._frame_keys[closest]
            arr = self._zarr_group[key]
            data = np.asarray(arr, dtype=float)
            # shape: (batch, z, y, x, c)  or simpler variants
            # squeeze leading size-1 batch dims
            while data.ndim > 4 and data.shape[0] == 1:
                data = data[0]
            # now expecting (z, y, x, c) or (t, y, x, c) or (y, x, c)
            shape = data.shape
            ndim = len(shape)
            if ndim == 4:  # (z_or_t, y, x, c)
                z_count = shape[0]
                if z_layer == "all":
                    frame = data.mean(axis=0)
                else:
                    z_idx = int(np.clip(int(z_layer), 0, z_count - 1))
                    frame = data[z_idx]  # (y, x, c)
            elif ndim == 3:  # (y, x, c)
                frame = data
            else:
                raise ValueError(f"Unexpected shape {shape} for zarr key '{key}'")
            if roi is not None:
                x0, x1, y0, y1 = roi
                frame = frame[y0:y1, x0:x1, :]
            if frame.size == 0:
                raise ValueError(f"Snapshot ROI produced empty frame for key '{key}'")
            return frame

        # ── standard mode: single dataset indexed by frame_idx ────────────
        dset_obj = self._job.get_raw(self._dset)
        data_obj = (
            dset_obj[self._slice_info] if self._slice_info is not None else dset_obj
        )
        shape = tuple(getattr(data_obj, "shape", ()))
        ndim = len(shape)

        if ndim == 5:
            # (t, z, y, x, c)
            t_count, z_count, _, _, c_count = shape
            if c_count < 1:
                raise ValueError(f"Dataset '{self._dset}' has no vector components")
            t_idx = int(np.clip(frame_idx, 0, t_count - 1))
            if z_layer == "all":
                frame = np.asarray(
                    data_obj[t_idx, :, :, :, : max(1, min(3, c_count))],
                    dtype=float,
                ).mean(axis=0)
            else:
                z_idx = int(z_layer)
                z_idx = int(np.clip(z_idx, 0, z_count - 1))
                frame = np.asarray(
                    data_obj[t_idx, z_idx, :, :, : max(1, min(3, c_count))],
                    dtype=float,
                )
        elif ndim == 4:
            # (t, y, x, c)
            t_count, _, _, c_count = shape
            if c_count < 1:
                raise ValueError(f"Dataset '{self._dset}' has no vector components")
            t_idx = int(np.clip(frame_idx, 0, t_count - 1))
            frame = np.asarray(
                data_obj[t_idx, :, :, : max(1, min(3, c_count))],
                dtype=float,
            )
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
        key = (
            int(frame_idx),
            str(component),
            str(z_layer),
            tuple(roi) if roi else None,
        )
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
    """Render snapshot panel for selected component."""
    ax.clear()
    comp = str(component).lower()
    x_size = int(frame.shape[1])
    y_size = int(frame.shape[0])
    x_nm = float(x_size) * float(dx) * 1e9
    y_nm = float(y_size) * float(dy) * 1e9
    max_nm = max(x_nm, y_nm)

    if max_nm >= 1000.0:
        # Use micrometers for large domains.
        axis_unit = "um"
        scale = 1e6
        dx_plot = float(dx)
        dy_plot = float(dy)
        extent_x = float(x_size) * dx_plot * scale
        extent_y = float(y_size) * dy_plot * scale
        # Quiver helper expects *nm* conversion internally, so pre-scale dx/dy.
        qdx = float(dx) * 1e-3
        qdy = float(dy) * 1e-3
    else:
        axis_unit = "nm"
        scale = 1e9
        dx_plot = float(dx)
        dy_plot = float(dy)
        extent_x = float(x_size) * dx_plot * scale
        extent_y = float(y_size) * dy_plot * scale
        qdx = float(dx)
        qdy = float(dy)

    if comp == "snapshot":
        u, v, w = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        rgb, alphas = _vector_field_to_rgb(u, v, w)
        if axis_unit == "um":
            quiver_u = np.asarray(u, dtype=float) * 1e-3
            quiver_v = np.asarray(v, dtype=float) * 1e-3
        else:
            quiver_u = u
            quiver_v = v
        _create_quiver_grid_and_scale(
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
        unit_suffix = f" {field_unit_text}" if field_unit_text else ""
        ax.set_title(f"Field: {float(field_value):.5g}{unit_suffix}")
    else:
        ax.set_title(f"Component: {component}")

    ax.set_xlabel(rf"$x$ ({axis_unit})")
    ax.set_ylabel(rf"$y$ ({axis_unit})")
