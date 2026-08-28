"""Material-mask inference shared by mode computation and visualization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import numpy as np


def infer_material_mask(
    magnetization: np.ndarray,
    *,
    relative_tolerance: float = 1e-12,
) -> np.ndarray:
    """Infer a ``(z, y, x)`` magnetic-material mask from canonical ``m`` data.

    The input must be ``(t,z,y,x,c)``. A cell is active when any sampled time
    or component contains a finite magnitude above a scale-relative numerical
    floor. The reduction is deliberately based on source magnetization, not on
    one eigenmode: physical mode nodes are allowed to have zero amplitude.
    """
    arr = np.asarray(magnetization)
    if arr.ndim != 5:
        raise ValueError(
            f"material-mask inference expects (t,z,y,x,c), got {arr.shape}"
        )
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")

    finite_magnitude = np.where(np.isfinite(arr), np.abs(arr), 0.0)
    activity = np.max(finite_magnitude, axis=(0, 4))
    scale = float(np.max(activity)) if activity.size else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros(activity.shape, dtype=bool)
    floor = max(scale * tolerance, np.finfo(float).tiny)
    return np.asarray(activity > floor, dtype=bool)


def coerce_material_mask(
    candidate: Any,
    target_shape: tuple[int, int, int],
) -> np.ndarray | None:
    """Coerce a geometry/material candidate to an exact ``(z,y,x)`` mask.

    Only shape-preserving or singleton reductions are accepted. The helper
    refuses interpolation because categorical geometry must not be blurred.
    """
    target = tuple(int(v) for v in target_shape)
    if len(target) != 3 or any(v < 1 for v in target):
        raise ValueError(f"target_shape must be positive (z,y,x), got {target}")
    arr = np.asarray(candidate)
    if arr.size == 0:
        return None
    arr = np.where(np.isfinite(arr), arr, 0)

    if arr.shape == target:
        return np.asarray(arr != 0, dtype=bool)
    if target[0] == 1 and arr.shape == target[1:]:
        return np.asarray(arr != 0, dtype=bool)[None, ...]

    # Accept extra singleton/time/component axes only when an ordered triplet
    # exactly matches the target. All other axes are reduced by logical any.
    for start in range(max(arr.ndim - 2, 0)):
        axes = (start, start + 1, start + 2)
        if tuple(int(arr.shape[a]) for a in axes) != target:
            continue
        active = np.asarray(arr != 0, dtype=bool)
        reduce_axes = tuple(axis for axis in range(arr.ndim) if axis not in axes)
        if reduce_axes:
            active = np.any(active, axis=reduce_axes)
        if active.shape == target:
            return np.asarray(active, dtype=bool)
    return None


def resolve_material_mask(
    magnetization: np.ndarray,
    *,
    geometry_candidates: Iterable[Any] = (),
    relative_tolerance: float = 1e-12,
) -> tuple[np.ndarray, str]:
    """Resolve mask with geometry priority and magnetization fallback."""
    arr = np.asarray(magnetization)
    if arr.ndim != 5:
        raise ValueError(f"expected canonical magnetization, got {arr.shape}")
    target = cast(tuple[int, int, int], tuple(int(v) for v in arr.shape[1:4]))
    for name, candidate in geometry_candidates:
        mask = coerce_material_mask(candidate, target)
        if mask is not None and np.any(mask):
            return mask, str(name)
    return (
        infer_material_mask(arr, relative_tolerance=relative_tolerance),
        "magnetization_nonzero",
    )


def masked_spatial(values: Any, material_mask: np.ndarray | None) -> Any:
    """Return a masked view whose non-material cells are transparent."""
    if material_mask is None:
        return values
    arr = np.asanyarray(values)
    mask = np.asarray(material_mask, dtype=bool)
    if arr.shape[:2] != mask.shape:
        raise ValueError(
            f"material mask shape {mask.shape} does not match spatial data {arr.shape}"
        )
    outside = ~mask
    if arr.ndim > 2:
        outside = np.broadcast_to(outside[(...,) + (None,) * (arr.ndim - 2)], arr.shape)
    return np.ma.array(arr, mask=outside, copy=False)


__all__ = [
    "coerce_material_mask",
    "infer_material_mask",
    "masked_spatial",
    "resolve_material_mask",
]
