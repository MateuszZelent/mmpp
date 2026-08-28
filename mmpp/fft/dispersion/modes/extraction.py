"""
Shared mode-extraction utilities for dispersion-mode reconstruction.

The interactive UI (InteractiveDispersionModes) and the folding helpers
(BrillouinZoneFolding) both need to reconstruct spatial mode profiles from
pre-computed dispersion FFT data.

Key requirement:
- Mode reconstruction is phase-sensitive and therefore requires
  ``DispersionResult1D.S_complex``. Reconstructing from power ``S`` is not
  physically meaningful (phase is lost).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..models import DispersionResult1D

logger = logging.getLogger(__name__)

KDirection = Literal["both", "positive", "negative"]
ReduceMode = Literal["mean", "sum"]


def canonicalize_s_complex(
    S_complex: np.ndarray,
    *,
    k_axis: np.ndarray,
    f_axis: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Return ``S_complex`` in canonical shape.

    Canonical shapes:
    - (Nk, Nf) when the orthogonal dimension was averaged away
    - (N_orth, Nk, Nf) when orthogonal spectra were preserved

    Notes:
    - Some cached/legacy code may store transposed axes; we detect and fix the
      most common variants by matching axis lengths to ``k_axis``/``f_axis``.
    """

    arr = np.asarray(S_complex)
    n_k = int(len(k_axis))
    n_f = int(len(f_axis))

    if arr.ndim == 2:
        if arr.shape == (n_k, n_f):
            return arr, False
        if arr.shape == (n_f, n_k):
            logger.warning("S_complex stored as (Nf, Nk); transposing to (Nk, Nf)")
            return arr.T, False
        raise ValueError(
            f"Unexpected 2D S_complex shape {arr.shape}; expected (Nk,Nf)=({n_k},{n_f})"
        )

    if arr.ndim != 3:
        raise ValueError(f"Unsupported S_complex rank={arr.ndim}; expected 2D or 3D")

    # Move frequency axis to last (axis=2) by matching length to Nf.
    freq_candidates = [ax for ax, size in enumerate(arr.shape) if size == n_f]
    if not freq_candidates:
        raise ValueError(
            f"Cannot locate frequency axis in S_complex shape {arr.shape}; "
            f"no axis matches Nf={n_f}"
        )
    freq_axis = 2 if 2 in freq_candidates else freq_candidates[0]
    arr = np.moveaxis(arr, freq_axis, 2)

    # Move k axis to axis=1 by matching length to Nk among the remaining axes.
    k_candidates = [ax for ax in (0, 1) if arr.shape[ax] == n_k]
    if not k_candidates:
        raise ValueError(
            f"Cannot locate k-axis in S_complex shape {arr.shape}; "
            f"no axis matches Nk={n_k}"
        )
    k_axis_idx = 1 if 1 in k_candidates else k_candidates[0]
    if k_axis_idx != 1:
        logger.warning("S_complex stored with k-axis=%d; moving to axis=1", k_axis_idx)
        arr = np.moveaxis(arr, k_axis_idx, 1)

    return arr, True


def _axis_is_shifted(axis_values: np.ndarray) -> bool:
    """Heuristic: fftshifted axes are monotonic (unshifted have a jump)."""
    axis = np.asarray(axis_values)
    if axis.ndim != 1 or axis.size < 3:
        return False
    diffs = np.diff(axis)
    return bool(np.all(diffs > 0) or np.all(diffs < 0))


def select_frequency_indices(
    f_axis: np.ndarray,
    *,
    f_0: float,
    f_margin_bins: int = 0,
    delta_f: float | None = None,
) -> np.ndarray:
    """Select frequency-bin indices around f0 using either delta_f (Hz) or bin margin."""
    f_axis = np.asarray(f_axis, dtype=float)
    if f_axis.ndim != 1 or f_axis.size == 0:
        raise ValueError("f_axis must be 1D and non-empty")
    if not np.all(np.isfinite(f_axis)):
        raise ValueError("f_axis must contain only finite values")
    f_target = float(f_0)
    if not np.isfinite(f_target):
        raise ValueError("f_0 must be finite")
    if (
        isinstance(f_margin_bins, (bool, np.bool_))
        or int(f_margin_bins) != f_margin_bins
    ):
        raise ValueError("f_margin_bins must be a non-negative integer")
    margin = int(f_margin_bins)
    if margin < 0:
        raise ValueError("f_margin_bins must be a non-negative integer")

    if delta_f is not None:
        width = float(delta_f)
        if not np.isfinite(width) or width <= 0:
            raise ValueError("delta_f must be finite and positive when provided")
        mask = np.abs(f_axis - f_target) <= width
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise ValueError(
                "delta_f window does not contain any sampled frequency bin"
            )
        return idx.astype(int, copy=False)

    idx0 = int(np.argmin(np.abs(f_axis - f_target)))
    start = max(0, idx0 - margin)
    stop = min(f_axis.size, idx0 + margin + 1)
    return np.arange(start, stop, dtype=int)


def build_bz_k_mask(
    k_axis: np.ndarray,
    *,
    k_0: float,
    lattice_constant: float,
    n_bz: int,
    k_direction: KDirection = "both",
    k_margin_bins: int = 0,
    delta_k: float | None = None,
) -> np.ndarray:
    """Build k-mask selecting k0 +/- n*G replicas (optionally with neighborhood width)."""

    k_axis = np.asarray(k_axis, dtype=float)
    if k_axis.ndim != 1 or k_axis.size == 0:
        raise ValueError("k_axis must be 1D and non-empty")
    if not np.all(np.isfinite(k_axis)):
        raise ValueError("k_axis must contain only finite values")
    k_center = float(k_0)
    if not np.isfinite(k_center):
        raise ValueError("k_0 must be finite")

    a = float(lattice_constant)
    if not np.isfinite(a) or a <= 0:
        raise ValueError(f"lattice_constant must be positive, got {lattice_constant!r}")

    if isinstance(n_bz, (bool, np.bool_)) or int(n_bz) != n_bz or int(n_bz) < 0:
        raise ValueError("n_bz must be a non-negative integer")
    n_bz_val = int(n_bz)
    direction = str(k_direction).lower()
    if direction not in {"both", "positive", "negative"}:
        raise ValueError("k_direction must be 'both', 'positive', or 'negative'")

    if (
        isinstance(k_margin_bins, (bool, np.bool_))
        or int(k_margin_bins) != k_margin_bins
        or int(k_margin_bins) < 0
    ):
        raise ValueError("k_margin_bins must be a non-negative integer")
    margin = int(k_margin_bins)
    use_delta = delta_k is not None
    if use_delta:
        assert delta_k is not None
        width = float(delta_k)
        if not np.isfinite(width) or width <= 0:
            raise ValueError("delta_k must be finite and positive when provided")

    G = 2.0 * np.pi / a
    replicas = k_center + np.arange(-n_bz_val, n_bz_val + 1, dtype=float) * G
    k_min = float(np.min(k_axis))
    k_max = float(np.max(k_axis))

    mask: Any = np.zeros(k_axis.size, dtype=bool)
    n_targets = 0
    for k_target in replicas:
        if direction == "positive" and k_target < 0:
            continue
        if direction == "negative" and k_target > 0:
            continue
        n_targets += 1

        if use_delta:
            mask |= np.abs(k_axis - k_target) <= width
            continue

        # A non-sampled reciprocal-lattice replica must not be projected onto
        # a boundary bin: doing so fabricates spectral weight at the edge.
        if k_target < k_min or k_target > k_max:
            continue

        idx_k = int(np.argmin(np.abs(k_axis - k_target)))
        lo = max(0, idx_k - margin)
        hi = min(k_axis.size, idx_k + margin + 1)
        mask[lo:hi] = True

    if use_delta and margin > 0 and np.any(mask):
        # Bin-dilate the selection by +/- margin bins.
        kernel = np.ones(2 * margin + 1, dtype=int)
        mask = np.convolve(mask.astype(int), kernel, mode="same") > 0

    if not np.any(mask):
        if n_targets == 0:
            raise ValueError(
                "k-mask is empty: k_direction excluded all BZ replicas. "
                "Try k_direction='both' or increase n_bz."
            )
        raise ValueError(
            "k selection does not contain any sampled bin; the requested BZ "
            "replicas/window lie outside the available k-axis"
        )

    return mask


def extract_mode_2d(
    result: DispersionResult1D,
    *,
    k_0: float,
    f_0: float,
    lattice_constant: float,
    n_bz: int,
    k_direction: KDirection = "both",
    k_margin_bins: int = 0,
    f_margin_bins: int = 0,
    neighbor_reduce: ReduceMode = "mean",
    delta_k: float | None = None,
    delta_f: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract complex 2D spatial mode m(x,y) from a dispersion result."""

    if result.S_complex is None:
        raise ValueError(
            "Mode visualization requires complex spectrum S_complex. "
            "Recompute dispersion with caching enabled and avg_over_orthogonal=False."
        )

    k_axis = np.asarray(result.k_axis)
    f_axis = np.asarray(result.f_axis)
    axis = str(getattr(result, "axis", "x")).lower()
    if axis not in {"x", "y"}:
        raise ValueError("Dispersion propagation axis must be 'x' or 'y'")

    S_complex, has_orth = canonicalize_s_complex(
        result.S_complex, k_axis=k_axis, f_axis=f_axis
    )

    # Frequency neighborhood selection.
    f_indices = select_frequency_indices(
        f_axis,
        f_0=float(f_0),
        f_margin_bins=int(f_margin_bins),
        delta_f=delta_f,
    )
    reducer = str(neighbor_reduce).lower()
    if reducer not in {"mean", "sum"}:
        raise ValueError("neighbor_reduce must be 'mean' or 'sum'")

    # BZ replica mask in k.
    k_mask = build_bz_k_mask(
        k_axis,
        k_0=float(k_0),
        lattice_constant=float(lattice_constant),
        n_bz=int(n_bz),
        k_direction=k_direction,
        k_margin_bins=int(k_margin_bins),
        delta_k=delta_k,
    )

    # Reduce over selected f bins (complex, phase-preserving).
    if has_orth:
        S_sel = S_complex[:, :, f_indices]  # (N_orth, Nk, Nf_sel)
        if reducer == "sum":
            S_k = np.sum(S_sel, axis=2)
        else:
            S_k = np.mean(S_sel, axis=2)
        S_k = np.array(S_k, copy=True)
        S_k[:, ~k_mask] = 0
    else:
        S_sel = S_complex[:, f_indices]  # (Nk, Nf_sel)
        if reducer == "sum":
            S_k = np.sum(S_sel, axis=1)
        else:
            S_k = np.mean(S_sel, axis=1)
        S_k = np.array(S_k, copy=True)
        S_k[~k_mask] = 0

    # IFFT over k -> propagation axis.
    if _axis_is_shifted(k_axis):
        if has_orth:
            S_k = np.fft.ifftshift(S_k, axes=1)
        else:
            S_k = np.fft.ifftshift(S_k)

    if has_orth:
        M_mode = np.fft.ifft(S_k, axis=1)  # (N_orth, N_prop)
    else:
        M_mode = np.fft.ifft(S_k)  # (N_prop,)
        M_mode = M_mode[np.newaxis, :]  # (1, N_prop)

    # ``S_complex`` is stored in the public k-axis convention already.  The
    # inverse transform above is therefore the complete reconstruction;
    # applying ``flipx`` again would reverse and roll the physical profile.

    # Axes in real space.
    n_prop = int(M_mode.shape[1])
    dx = float(getattr(result, "dx", 0.0) or 0.0)
    if dx > 0:
        prop_axis: Any = np.arange(n_prop, dtype=float) * dx
    else:
        if k_axis.size > 1:
            dk = float(np.abs(k_axis[1] - k_axis[0]))
            L = 2.0 * np.pi / dk if dk > 0 else float(n_prop)
        else:
            L = float(n_prop)
        prop_axis = np.linspace(0.0, L, n_prop, endpoint=False)

    if has_orth and getattr(result, "orth_axis", None) is not None:
        orth_axis = np.asarray(result.orth_axis, dtype=float)
        if orth_axis.ndim != 1 or orth_axis.size != M_mode.shape[0]:
            raise ValueError(
                "orth_axis length must match the preserved orthogonal dimension"
            )
        if not np.all(np.isfinite(orth_axis)):
            raise ValueError("orth_axis must contain only finite values")
    else:
        orth_axis = np.arange(int(M_mode.shape[0]), dtype=float) * (
            dx if dx > 0 else 1.0
        )

    # Assign to x/y arrays and return mode_2d shaped as (Ny, Nx).
    if axis == "x":
        x_axis = prop_axis
        y_axis = orth_axis
        mode_2d = M_mode  # (Ny, Nx)
    else:
        x_axis = orth_axis
        y_axis = prop_axis
        mode_2d = M_mode.T  # -> (Ny, Nx)

    info = {
        "axis": axis,
        "has_orth": bool(has_orth),
        "n_bz": int(n_bz),
        "k_direction": str(k_direction),
        "k_margin_bins": int(k_margin_bins),
        "f_margin_bins": int(f_margin_bins),
        "neighbor_reduce": reducer,
        "delta_k": None if delta_k is None else float(delta_k),
        "delta_f": None if delta_f is None else float(delta_f),
        "k_bins_selected": int(np.sum(k_mask)),
        "f_bins_selected": int(f_indices.size),
        "k_0": float(k_0),
        "f_0": float(f_0),
    }

    return x_axis, y_axis, mode_2d, info


def extract_mode_profile_1d(
    result: DispersionResult1D,
    *,
    k_0: float,
    f_0: float,
    lattice_constant: float,
    n_bz: int,
    k_direction: KDirection = "both",
    k_margin_bins: int = 0,
    f_margin_bins: int = 0,
    neighbor_reduce: ReduceMode = "mean",
    orth_reduce: ReduceMode = "mean",
    delta_k: float | None = None,
    delta_f: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract a 1D complex profile along the propagation axis.

    This is primarily used for 1D animations (amplitude/phase vs position).
    For results with an orthogonal dimension, this function collapses it using
    ``orth_reduce`` after reconstructing the 2D field.
    """

    x_axis, y_axis, mode_2d, info = extract_mode_2d(
        result,
        k_0=float(k_0),
        f_0=float(f_0),
        lattice_constant=float(lattice_constant),
        n_bz=int(n_bz),
        k_direction=k_direction,
        k_margin_bins=int(k_margin_bins),
        f_margin_bins=int(f_margin_bins),
        neighbor_reduce=neighbor_reduce,
        delta_k=delta_k,
        delta_f=delta_f,
    )

    reducer = str(orth_reduce).lower()
    if reducer not in {"mean", "sum"}:
        raise ValueError("orth_reduce must be 'mean' or 'sum'")

    axis = str(info.get("axis", "x"))
    if axis == "x":
        # Propagation along x -> collapse over y (rows).
        if reducer == "sum":
            profile = np.sum(mode_2d, axis=0)
        else:
            profile = np.mean(mode_2d, axis=0)
        prop_axis = x_axis
    else:
        # Propagation along y -> collapse over x (columns).
        if reducer == "sum":
            profile = np.sum(mode_2d, axis=1)
        else:
            profile = np.mean(mode_2d, axis=1)
        prop_axis = y_axis

    info = dict(info)
    info["orth_reduce"] = reducer
    return prop_axis, profile, info
