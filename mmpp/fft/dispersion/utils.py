"""
Utility functions for spin-wave dispersion analysis.

Contains low-level functions for FFT, k-space operations, windowing, 
and mathematical operations used in dispersion calculations.
"""

from __future__ import annotations
from typing import Tuple, List, Optional, Sequence
import logging
import numpy as np


logger = logging.getLogger(__name__)


def fftfreq_axis(n: int, d: float, shift: bool = True) -> np.ndarray:
    """
    Frequency axis (Hz) for FFT length n and sample spacing d.
    
    Parameters
    ----------
    n : int
        FFT length
    d : float  
        Sample spacing (time step) [s]
    shift : bool
        If True, returns fftshifted (centered) axis
        
    Returns
    -------
    np.ndarray
        Frequency axis [Hz]
    """
    f = np.fft.fftfreq(n, d)
    return np.fft.fftshift(f) if shift else f


def k_axis_from_grid(n: int, d: float, shift: bool = True) -> np.ndarray:
    """
    Wavevector axis k (rad/m) for FFT length n and grid spacing d [m].
    
    Parameters
    ----------
    n : int
        FFT length (number of grid points)
    d : float
        Grid spacing [m]  
    shift : bool
        If True, returns fftshifted (centered) axis
        
    Returns
    -------
    np.ndarray
        Wavevector axis k [rad/m], range approximately [-π/d, π/d)
    """
    k = 2.0 * np.pi * np.fft.fftfreq(n, d)
    return np.fft.fftshift(k) if shift else k


def fold_k_to_bz(k: np.ndarray, a: float) -> np.ndarray:
    """
    Fold wavevector(s) k [rad/m] to first Brillouin zone (-π/a, π/a].
    
    Parameters
    ----------
    k : np.ndarray
        Wavevector(s) [rad/m]
    a : float
        Real-space period [m] defining BZ size
        
    Returns
    -------
    np.ndarray
        Folded wavevectors in first BZ
    """
    width = 2.0 * np.pi / a
    # Map to (-π/a, π/a]
    k_fold = (k + np.pi / a) % width - np.pi / a
    return k_fold


def fold_spectrum_1d(
    Skf: np.ndarray, 
    k: np.ndarray, 
    a: float, 
    agg: str = "sum"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fold a 1D dispersion S(k,f) into first BZ defined by period a.
    
    Parameters
    ----------
    Skf : np.ndarray
        Dispersion spectrum with shape (Nk, Nf)
    k : np.ndarray  
        Wavevector axis (Nk,) [rad/m]
    a : float
        Real-space period [m]
    agg : {'sum', 'max'}
        Aggregation method for aliased k bins
        
    Returns
    -------
    k_fold_sorted : np.ndarray
        Unique folded k values, sorted
    Skf_folded : np.ndarray  
        Folded spectrum (Nk_fold, Nf)
    """
    k_fold = fold_k_to_bz(k, a)
    
    # Group by unique folded k values (with tolerance)
    dk = np.median(np.diff(np.sort(k))) if len(k) > 1 else 1.0
    tol = dk * 0.25
    
    # Sort by folded k
    order = np.argsort(k_fold)
    kf_sorted = k_fold[order]
    Skf_sorted = Skf[order]
    
    # Build index groups for identical k values
    groups: List[np.ndarray] = []
    current = [0]
    for i in range(1, len(kf_sorted)):
        if abs(kf_sorted[i] - kf_sorted[current[-1]]) <= tol:
            current.append(i)
        else:
            groups.append(np.array(current, dtype=int))
            current = [i]
    groups.append(np.array(current, dtype=int))

    # Aggregate each group
    k_fold_unique = np.array([np.mean(kf_sorted[g]) for g in groups])
    
    if agg == "max":
        Skf_folded = np.stack([np.nanmax(Skf_sorted[g], axis=0) for g in groups], axis=0)
    else:  # sum
        Skf_folded = np.stack([np.nansum(Skf_sorted[g], axis=0) for g in groups], axis=0)

    # Sort by k
    srt = np.argsort(k_fold_unique)
    return k_fold_unique[srt], Skf_folded[srt, :]


def hann_window(n: int) -> np.ndarray:
    """
    Hann window (periodic) of length n.

    Parameters
    ----------
    n : int
        Window length

    Returns
    -------
    np.ndarray
        Hann window values
    """
    if n <= 1:
        return np.ones(n, dtype=float)
    idx = np.arange(n, dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * idx / (n - 1))


def apply_window_1d(
    x: np.ndarray, 
    axis: int, 
    window: Optional[str]
) -> np.ndarray:
    """
    Apply window function along specified axis.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    axis : int
        Axis along which to apply window
    window : Optional[str]
        Window type: 'hann' or None
        
    Returns
    -------
    np.ndarray
        Windowed array
    """
    if window is None:
        return x
        
    n = x.shape[axis]
    if window == "hann":
        w = hann_window(n)
    else:
        raise ValueError(f"Unknown window '{window}'")
        
    # Reshape for broadcasting
    shape = [1] * x.ndim
    shape[axis] = n
    return x * w.reshape(shape)



def apply_filter_pipeline(
    x: np.ndarray,
    filters: Optional[dict[str, bool]],
    *,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (2, 3),
) -> np.ndarray:
    """Apply canonical dispersion filters to magnetization data."""
    if not filters:
        return x

    result = x
    copied = False

    def ensure_copy() -> None:
        nonlocal result, copied
        if not copied:
            result = np.array(result, copy=True)
            copied = True

    applied: list[str] = []

    if filters.get("remove_static"):
        ensure_copy()
        first = np.take(result, indices=0, axis=time_axis)
        expanded = np.expand_dims(first, axis=time_axis)
        result -= expanded
        applied.append("remove_static")

    if filters.get("remove_average"):
        ensure_copy()
        mean = np.mean(result, axis=time_axis, keepdims=True)
        result -= mean
        applied.append("remove_average")

    if filters.get("hann_time"):
        ensure_copy()
        result = apply_window_1d(result, axis=time_axis, window="hann")
        applied.append("hann_time")

    if filters.get("hann_space") and spatial_axes:
        ensure_copy()
        ndims = result.ndim
        for axis in spatial_axes:
            if 0 <= axis < ndims and result.shape[axis] > 1:
                result = apply_window_1d(result, axis=axis, window="hann")
        applied.append("hann_space")

    if applied:
        logger.info("Dispersion filters applied: %s", ", ".join(applied))

    return result


def detrend_time_series(
    Mt: np.ndarray, 
    axis: int = 0, 
    method: str = "mean"
) -> np.ndarray:
    """
    Detrend time series along specified axis.
    
    Parameters
    ----------
    Mt : np.ndarray
        Time series data
    axis : int
        Time axis
    method : {'mean', 'initial'}
        Detrending method:
        - 'mean': Remove time average
        - 'initial': Remove initial value
        
    Returns
    -------
    np.ndarray
        Detrended data
    """
    if method == "mean":
        mean = np.mean(Mt, axis=axis, keepdims=True)
        return Mt - mean
    elif method == "initial":
        init = np.take(Mt, indices=0, axis=axis)
        # Reshape for broadcasting
        slicer = [slice(None)] * Mt.ndim
        slicer[axis] = slice(0, 1)
        init = init.reshape([Mt.shape[i] if i != axis else 1 for i in range(Mt.ndim)])
        return Mt - init
    else:
        return Mt


def find_peaks_1d(
    y: np.ndarray, 
    min_prominence: float = 0.0
) -> np.ndarray:
    """
    Simple peak finder for 1D arrays.
    
    Parameters
    ----------
    y : np.ndarray
        1D signal
    min_prominence : float
        Minimum peak prominence to keep
        
    Returns
    -------
    np.ndarray
        Indices of detected peaks
    """
    if y.size < 3:
        return np.array([], dtype=int)
        
    dy = np.diff(y)
    # Maxima where derivative changes from + to -
    cand = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0] + 1
    
    if min_prominence <= 0:
        return cand
        
    # Filter by prominence
    keep = []
    for i in cand:
        left_max = np.max(y[:i]) if i > 0 else y[i]
        right_max = np.max(y[i+1:]) if i < y.size - 1 else y[i]
        base = max(min(left_max, right_max), 0.0)
        prominence = y[i] - base
        if prominence >= min_prominence:
            keep.append(i)
    
    return np.array(keep, dtype=int)


def group_velocity_1d(
    k_axis: np.ndarray,
    f_branch: np.ndarray, 
    angular: bool = True
) -> np.ndarray:
    """
    Estimate group velocity from dispersion branch.
    
    Parameters
    ----------
    k_axis : np.ndarray
        Wave vector values [rad/m]
    f_branch : np.ndarray
        Branch frequencies [Hz]
    angular : bool
        If True, return v_g = dω/dk [m/s] 
        If False, return df/dk [Hz⋅m]
        
    Returns
    -------
    np.ndarray
        Group velocity values
    """
    dk = np.gradient(k_axis)
    df = np.gradient(f_branch)
    
    vg = df / dk  # Hz⋅m
    
    if angular:
        vg *= 2 * np.pi  # Convert to rad/s per (rad/m) = m/s
        
    return vg


def normalize_magnetization_components(M: np.ndarray) -> np.ndarray:
    """
    Ensure magnetization array has proper shape and component ordering.
    
    Parameters
    ----------
    M : np.ndarray
        Magnetization array, expected shapes:
        - (T, Z, Y, X, 3)  - full 3-component vector
        - (T, Y, X, 3)     - 2D with 3 components
        - (T, X, 3)        - 1D with 3 components
        - (T, Z, Y, X)     - single component pre-selected
        - (T, Y, X)        - 2D single component
        - (T, X)           - 1D single component
        
    Returns
    -------
    np.ndarray
        Normalized array with shape (T, Z, Y, X, C) where C is 1 or 3
    """
    # Case 1: Full 3-component data
    if M.ndim == 5:
        # (T, Z, Y, X, 3)
        if M.shape[-1] != 3:
            raise ValueError(f"5D array must have last axis=3 (mx,my,mz), got {M.shape[-1]}")
        return M
    elif M.ndim == 4:
        # Could be (T, Y, X, 3) or (T, Z, Y, X) single component
        if M.shape[-1] == 3:
            # (T, Y, X, 3) -> (T, 1, Y, X, 3)
            T, Y, X, C = M.shape
            return M.reshape(T, 1, Y, X, C)
        else:
            # (T, Z, Y, X) single component -> (T, Z, Y, X, 1)
            return M[..., np.newaxis]
    elif M.ndim == 3:
        # Could be (T, X, 3) or (T, Y, X) single component
        if M.shape[-1] == 3:
            # (T, X, 3) -> (T, 1, 1, X, 3)
            T, X, C = M.shape
            return M.reshape(T, 1, 1, X, C)
        else:
            # (T, Y, X) single component -> (T, 1, Y, X, 1)
            T, Y, X = M.shape
            return M.reshape(T, 1, Y, X, 1)
    elif M.ndim == 2:
        # (T, X) single component -> (T, 1, 1, X, 1)
        T, X = M.shape
        return M.reshape(T, 1, 1, X, 1)
    else:
        raise ValueError(
            f"M must have 2-5 dimensions, got {M.ndim}. "
            f"Expected shapes: (T,Z,Y,X,3), (T,Y,X,3), (T,X,3), "
            f"or single-component (T,Z,Y,X), (T,Y,X), (T,X)"
        )
        
    return M


def extract_magnetization_component(
    M: np.ndarray, 
    component: str
) -> np.ndarray:
    """
    Extract specified magnetization component(s).
    
    Parameters
    ----------
    M : np.ndarray
        Magnetization array (..., C) where C is 1 (single component) or 3 (mx, my, mz)
    component : str
        Component to extract:
        - 'perp': mx + i*my (complex transverse)
        - 'mx', 'my', 'mz': individual components
        - 'sum': rough sum of all components
        - None or 'auto': use data as-is if already single component
        
    Returns
    -------
    np.ndarray
        Selected component(s), complex dtype
    """
    # If M only has 1 component (already selected via slicing), return it as-is
    if M.shape[-1] == 1:
        if component is None or component == "auto":
            # Already single component, just return it
            return M[..., 0].astype(np.complex128)
        else:
            # User specified component but data already has only 1 component
            # This is fine - just use what we have
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Magnetization data already has single component (shape[-1]=1). "
                f"Ignoring component='{component}' parameter and using existing data."
            )
            return M[..., 0].astype(np.complex128)
    
    # Standard case: M has 3 components
    if M.shape[-1] != 3:
        raise ValueError(
            f"Magnetization array must have last axis = 1 (single component) or 3 (mx,my,mz). "
            f"Got shape[-1] = {M.shape[-1]}"
        )
    
    mx = M[..., 0]
    my = M[..., 1] 
    mz = M[..., 2]

    if component == "perp" or component is None:
        return (mx + 1j * my).astype(np.complex128)
    elif component == "mx":
        return mx.astype(np.complex128)
    elif component == "my":
        return my.astype(np.complex128)
    elif component == "mz":
        return mz.astype(np.complex128)
    elif component == "sum":
        return ((mx + 1j * my) + mz).astype(np.complex128)
    else:
        raise ValueError(f"Unknown component '{component}'. Use 'perp', 'mx', 'my', 'mz', or 'sum'.")


def validate_grid_parameters(
    dt: float,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    dz: Optional[float] = None
) -> None:
    """
    Validate grid spacing parameters.
    
    Parameters
    ----------
    dt : float
        Time step [s]
    dx, dy, dz : Optional[float]
        Spatial grid spacings [m]
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    if dt <= 0:
        raise ValueError("Time step dt must be positive")
        
    for name, val in [("dx", dx), ("dy", dy), ("dz", dz)]:
        if val is not None and val <= 0:
            raise ValueError(f"Grid spacing {name} must be positive, got {val}")


def get_frequency_band_mask(
    f_axis: np.ndarray,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None
) -> np.ndarray:
    """
    Create boolean mask for frequency band selection.
    
    Parameters
    ----------
    f_axis : np.ndarray
        Frequency axis [Hz]
    f_min, f_max : Optional[float]
        Frequency band limits [Hz]
        
    Returns
    -------
    np.ndarray
        Boolean mask for frequency selection
    """
    mask = np.ones(len(f_axis), dtype=bool)
    
    if f_min is not None:
        mask &= (f_axis >= f_min)
    if f_max is not None:
        mask &= (f_axis <= f_max)
        
    return mask