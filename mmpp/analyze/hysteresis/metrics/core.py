"""Core physical metrics for hysteresis loops."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..compute import find_zero_crossings, interpolate_at_x, numerical_derivative


@dataclass
class CoerciveFieldResult:
    """Coercive-field summary."""

    hc_minus: float
    hc_plus: float
    mean: float
    asymmetry: float
    unit: str = "input"
    branch_status: str = "unavailable"


@dataclass
class RemanenceResult:
    """Remanence summary at zero field."""

    mr_minus: float
    mr_plus: float
    mean: float


@dataclass
class SaturationResult:
    """Saturation points inferred from dM/dB thresholding."""

    ms_positive: float
    ms_negative: float
    hs_positive: float
    hs_negative: float
    ms_mean: float
    status: str = "unavailable"


@dataclass
class SusceptibilityResult:
    """Maximum susceptibility details."""

    chi_max: float
    field_at_max: float


def _nanmean_abs(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr)))


def _major_branches(branches) -> list:
    major = [b for b in branches if bool(getattr(b, "is_major", False))]
    return major if major else list(branches)


def compute_coercive_field(
    field: np.ndarray,
    magnetization: np.ndarray,
    branches,
    *,
    unit: str = "input",
) -> CoerciveFieldResult:
    """Compute branch-resolved coercive fields from M=0 crossings."""
    field_arr = np.asarray(field, dtype=float)
    mag_arr = np.asarray(magnetization, dtype=float)

    ascending_candidates: list[float] = []
    descending_candidates: list[float] = []

    for branch in _major_branches(branches):
        bx = field_arr[branch.slice]
        by = mag_arr[branch.slice]
        crossings = find_zero_crossings(bx, by)
        destination = (
            ascending_candidates
            if str(getattr(branch, "name", "")).lower() == "ascending"
            else descending_candidates
        )
        destination.extend(float(value) for value in crossings)

    # Hc+ is the M=0 crossing on the increasing-field branch and Hc- is the
    # crossing on the decreasing-field branch. Exchange bias may put both at
    # the same sign of H, so field sign cannot identify the branch.
    hc_plus = (
        float(np.mean(ascending_candidates)) if ascending_candidates else float("nan")
    )
    hc_minus = (
        float(np.mean(descending_candidates)) if descending_candidates else float("nan")
    )

    mean_val = _nanmean_abs([hc_minus, hc_plus])
    asym = float(np.abs(hc_plus) - np.abs(hc_minus))

    return CoerciveFieldResult(
        hc_minus=hc_minus,
        hc_plus=hc_plus,
        mean=mean_val,
        asymmetry=asym,
        unit=unit,
        branch_status=(
            "complete"
            if np.isfinite(hc_plus) and np.isfinite(hc_minus)
            else "incomplete_major_loop"
        ),
    )


def compute_remanence(
    field: np.ndarray,
    magnetization: np.ndarray,
    branches,
) -> RemanenceResult:
    """Compute remanence values at B=0 for major branches."""
    field_arr = np.asarray(field, dtype=float)
    mag_arr = np.asarray(magnetization, dtype=float)

    rem_values: list[float] = []
    for branch in _major_branches(branches):
        interp = interpolate_at_x(field_arr[branch.slice], mag_arr[branch.slice], 0.0)
        if np.isfinite(interp):
            rem_values.append(float(interp))

    if not rem_values:
        return RemanenceResult(float("nan"), float("nan"), float("nan"))

    rem_arr = np.asarray(rem_values, dtype=float)
    mr_plus = float(np.nanmax(rem_arr))
    mr_minus = float(np.nanmin(rem_arr))
    mean_val = _nanmean_abs([mr_minus, mr_plus])

    return RemanenceResult(mr_minus=mr_minus, mr_plus=mr_plus, mean=mean_val)


def _saturation_mask_from_derivative(
    derivative: np.ndarray,
    *,
    threshold: float,
    window: int,
) -> np.ndarray:
    mask = np.abs(np.asarray(derivative, dtype=float)) < float(threshold)
    n = len(mask)
    w = int(
        min(window, n)
    )  # clamp: konwolucja mode='same' zwraca max(M,N) jeśli kernel > dane
    if w <= 1:
        return mask

    kernel = np.ones(w, dtype=int)
    hits = np.convolve(mask.astype(int), kernel, mode="same")
    return (hits >= w)[:n]  # przytnij na wypadek gdyby numpy zwróciło za dużo


def compute_saturation_points(
    field: np.ndarray,
    magnetization: np.ndarray,
    *,
    threshold: float,
    window: int,
) -> SaturationResult:
    """Estimate saturation moments and fields."""
    field_arr = np.asarray(field, dtype=float)
    mag_arr = np.asarray(magnetization, dtype=float)
    derivative = numerical_derivative(field_arr, mag_arr)
    sat_mask = _saturation_mask_from_derivative(
        derivative,
        threshold=threshold,
        window=window,
    )

    def _pick(sign: int) -> tuple[float, float]:
        if sign > 0:
            idx = np.where((field_arr >= 0) & sat_mask)[0]
            if idx.size == 0:
                return float("nan"), float("nan")
            fields = field_arr[idx]
            cutoff = float(np.nanpercentile(fields, 80))
            selected = idx[fields >= cutoff]
        else:
            idx = np.where((field_arr <= 0) & sat_mask)[0]
            if idx.size == 0:
                return float("nan"), float("nan")
            fields = field_arr[idx]
            cutoff = float(np.nanpercentile(fields, 20))
            selected = idx[fields <= cutoff]

        if selected.size == 0:
            selected = idx

        ms = float(np.nanmean(mag_arr[selected]))
        hs = float(np.nanmean(field_arr[selected]))
        return ms, hs

    ms_pos, hs_pos = _pick(+1)
    ms_neg, hs_neg = _pick(-1)
    ms_mean = _nanmean_abs([ms_pos, ms_neg])

    status = (
        "detected"
        if np.isfinite(ms_pos) and np.isfinite(ms_neg)
        else "plateau_not_detected"
    )
    return SaturationResult(
        ms_positive=ms_pos,
        ms_negative=ms_neg,
        hs_positive=hs_pos,
        hs_negative=hs_neg,
        ms_mean=ms_mean,
        status=status,
    )


def compute_loop_area(field: np.ndarray, magnetization: np.ndarray) -> float:
    """Numerical loop area integral A = ∮ M dB."""
    field_arr = np.asarray(field, dtype=float)
    mag_arr = np.asarray(magnetization, dtype=float)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0 compatibility
        integrate = getattr(np, "trapz", None)
    if integrate is None:  # pragma: no cover - NumPy >= 1.20 always has trapz
        raise RuntimeError("NumPy has no trapezoidal integration function")
    area = float(integrate(mag_arr, field_arr))
    return float(np.abs(area))


def compute_squareness(
    remanence: RemanenceResult, saturation: SaturationResult
) -> float:
    """Compute squareness S = Mr / Ms."""
    mr = float(remanence.mean)
    ms = float(saturation.ms_mean)
    if (
        saturation.status != "detected"
        or not np.isfinite(mr)
        or not np.isfinite(ms)
        or ms == 0.0
    ):
        return float("nan")
    return float(mr / ms)


def compute_max_susceptibility(
    field: np.ndarray,
    magnetization: np.ndarray,
) -> SusceptibilityResult:
    """Compute max |dM/dB| and location."""
    field_arr = np.asarray(field, dtype=float)
    derivative = numerical_derivative(field_arr, np.asarray(magnetization, dtype=float))
    if derivative.size == 0:
        return SusceptibilityResult(float("nan"), float("nan"))

    idx = int(np.nanargmax(np.abs(derivative)))
    return SusceptibilityResult(
        chi_max=float(np.abs(derivative[idx])),
        field_at_max=float(field_arr[idx]),
    )


def compute_exchange_bias(coercive_field: CoerciveFieldResult) -> float:
    """Compute exchange bias field H_EB = (Hc+ + Hc-) / 2."""
    if not np.isfinite(coercive_field.hc_plus) or not np.isfinite(
        coercive_field.hc_minus
    ):
        return float("nan")
    return float((coercive_field.hc_plus + coercive_field.hc_minus) / 2.0)
