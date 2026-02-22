"""Snapshot loading and rendering helpers for hysteresis explorer."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from .....pyzfn.snapshot import _create_quiver_grid_and_scale, _vector_field_to_rgb


class SnapshotCache:
    """Lazy-loading LRU cache for magnetization snapshots."""

    def __init__(
        self,
        job_result,
        *,
        dset: str = "m",
        slice_info: Any | None = None,
        max_cached: int = 50,
    ):
        self._job = job_result
        self._dset = dset
        self._slice_info = slice_info
        self._max = int(max(1, max_cached))
        self._lru: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()

    def clear(self) -> None:
        self._lru.clear()

    def _load_raw_frame(
        self,
        frame_idx: int,
        *,
        z_layer: int | str = 0,
        roi: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        dset_obj = self._job.get_raw(self._dset)
        data_obj = dset_obj[self._slice_info] if self._slice_info is not None else dset_obj
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
    dx: float = 1.0,
    dy: float = 1.0,
    cmap: str = "viridis",
) -> None:
    """Render snapshot panel for selected component."""
    ax.clear()
    comp = str(component).lower()

    if comp == "snapshot":
        u, v, w = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        rgb, alphas = _vector_field_to_rgb(u, v, w)
        _create_quiver_grid_and_scale(ax=ax, alphas=alphas, u=u, v=v, dx=dx, dy=dy)
        ax.imshow(
            rgb,
            interpolation="none",
            origin="lower",
            aspect="equal",
            extent=(0, rgb.shape[1] * dx * 1e9, 0, rgb.shape[0] * dy * 1e9),
        )
        ax.set_title("Snapshot (RGB + quiver)")
    else:
        image = _component_image(frame, comp)
        ax.imshow(
            image,
            cmap=cmap,
            interpolation="none",
            origin="lower",
            aspect="equal",
            extent=(0, image.shape[1] * dx * 1e9, 0, image.shape[0] * dy * 1e9),
        )
        ax.set_title(f"Component: {component}")

    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
