"""Shared magnetization-field helpers for soliton analysis."""

from __future__ import annotations

import numpy as np


def _resolve_index(index: int, size: int, name: str) -> int:
    """Resolve a possibly negative index and raise a consistent error."""
    resolved = int(index)
    if resolved < 0:
        resolved += int(size)
    if resolved < 0 or resolved >= int(size):
        raise IndexError(f"{name} index {index} out of bounds for size {size}")
    return resolved


def select_snapshot(
    m: np.ndarray,
    frame: int = 0,
    z_layer: int = -1,
) -> np.ndarray:
    """Select one ``(Ny, Nx, 3)`` snapshot from a magnetization field.

    Supported inputs are a single snapshot ``(Ny, Nx, 3)``, a time series
    ``(Nt, Ny, Nx, 3)``, or a layered time series ``(Nt, Nz, Ny, Nx, 3)``.
    ``frame`` and ``z_layer`` accept Python-style negative indices where they
    apply.  The returned array is a view when the input supports it.
    """
    arr = np.asarray(m, dtype=float)

    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr

    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr[_resolve_index(frame, arr.shape[0], "frame")]

    if arr.ndim == 5 and arr.shape[-1] == 3:
        frame_idx = _resolve_index(frame, arr.shape[0], "frame")
        z_idx = _resolve_index(z_layer, arr.shape[1], "z_layer")
        return arr[frame_idx, z_idx]

    raise ValueError(
        "Unsupported magnetization shape. Expected (Ny,Nx,3), (Nt,Ny,Nx,3), "
        "or (Nt,Nz,Ny,Nx,3)."
    )


def valid_magnetization_mask(
    m: np.ndarray,
    *,
    atol: float = 1e-30,
) -> np.ndarray:
    """Return a mask for finite, non-zero magnetization vectors.

    The mask has the input shape with its trailing vector axis removed.  It
    therefore works for snapshots as well as time/layer series and can be
    reduced or indexed by callers according to their workflow.
    """
    arr = np.asarray(m, dtype=float)
    if arr.ndim < 3 or arr.shape[-1] < 3:
        raise ValueError(
            "Expected magnetization with trailing vector axis of size >= 3"
        )

    vectors = arr[..., :3]
    finite = np.isfinite(vectors).all(axis=-1)
    norm = np.linalg.norm(np.where(finite[..., None], vectors, 0.0), axis=-1)
    return np.asarray(finite & (norm > float(atol)), dtype=bool)


__all__ = ["select_snapshot", "valid_magnetization_mask"]
