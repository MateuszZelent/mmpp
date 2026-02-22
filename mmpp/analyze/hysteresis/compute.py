"""Core computations for hysteresis processing and validation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .result import Branch


def validate_hysteresis_data(
    field: np.ndarray,
    magnetization: np.ndarray,
    *,
    require_non_monotonic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate input arrays and return finite 1D views.

    Raises
    ------
    ValueError
        If input does not represent a meaningful hysteresis loop.
    """
    field_arr = np.asarray(field, dtype=float).reshape(-1)
    magnetization_arr = np.asarray(magnetization, dtype=float).reshape(-1)

    errors: list[str] = []
    if field_arr.size != magnetization_arr.size:
        errors.append(
            f"Size mismatch: field({field_arr.size}) vs M({magnetization_arr.size})"
        )

    if field_arr.size < 10:
        errors.append(
            f"Too few points ({field_arr.size}), need >=10 for meaningful analysis"
        )

    finite_mask = np.isfinite(field_arr) & np.isfinite(magnetization_arr)
    finite_count = int(np.count_nonzero(finite_mask))
    if finite_count < 10:
        errors.append(
            "Too few finite points after NaN/Inf removal "
            f"({finite_count}), need >=10"
        )

    if require_non_monotonic and finite_count > 1:
        field_finite = field_arr[finite_mask]
        dfield = np.diff(field_finite)
        if np.all(dfield >= 0) or np.all(dfield <= 0):
            errors.append("Field is monotonic - this is not a hysteresis loop")

    if errors:
        bullet = "\n".join(f"  - {msg}" for msg in errors)
        raise ValueError(f"Data validation failed:\n{bullet}")

    return field_arr[finite_mask], magnetization_arr[finite_mask]


def _fill_zero_signs(sign: np.ndarray) -> np.ndarray:
    """Fill zero entries in sign array using nearest non-zero neighbors."""
    sign_out = np.asarray(sign, dtype=float).copy()
    if sign_out.size == 0:
        return sign_out

    # Forward fill
    for idx in range(1, sign_out.size):
        if sign_out[idx] == 0 and sign_out[idx - 1] != 0:
            sign_out[idx] = sign_out[idx - 1]

    # Backward fill
    for idx in range(sign_out.size - 2, -1, -1):
        if sign_out[idx] == 0 and sign_out[idx + 1] != 0:
            sign_out[idx] = sign_out[idx + 1]

    # Any remaining zeros -> ascending default
    sign_out[sign_out == 0] = 1.0
    return sign_out


def segment_branches(field: np.ndarray, slope_tolerance: float = 1e-15) -> list[Branch]:
    """Segment loop into monotonic ascending/descending branches.

    Returns
    -------
    list[Branch]
        Branches with cycle ids and major/minor flags.
    """
    field_arr = np.asarray(field, dtype=float).reshape(-1)
    n = int(field_arr.size)
    if n <= 1:
        return [Branch(name="ascending", start=0, stop=n, cycle_id=0, is_major=True)]

    dfield = np.diff(field_arr)
    sign = np.sign(dfield)
    sign[np.abs(dfield) <= slope_tolerance] = 0.0
    sign = _fill_zero_signs(sign)

    change_idx = np.where(sign[1:] != sign[:-1])[0]
    starts = [0] + [int(idx + 1) for idx in change_idx]
    stops = [int(idx + 1) for idx in change_idx] + [n]

    branches: list[Branch] = []
    cycle_id = 0
    prev_name: str | None = None

    for start, stop in zip(starts, stops):
        stop = max(stop, start + 1)
        local_diff = np.diff(field_arr[start:stop])
        trend = float(np.nanmedian(local_diff)) if local_diff.size else 0.0
        name = "ascending" if trend >= 0 else "descending"

        if prev_name == "descending" and name == "ascending":
            cycle_id += 1
        prev_name = name

        branches.append(
            Branch(
                name=name,
                start=int(start),
                stop=int(stop),
                cycle_id=int(cycle_id),
                is_major=False,
            )
        )

    # major cycle = cycle with maximum field span
    by_cycle: dict[int, list[Branch]] = defaultdict(list)
    for branch in branches:
        by_cycle[branch.cycle_id].append(branch)

    cycle_spans: dict[int, float] = {}
    cycle_points: dict[int, int] = {}
    for cid, segs in by_cycle.items():
        c_start = min(seg.start for seg in segs)
        c_stop = max(seg.stop for seg in segs)
        cycle_field = field_arr[c_start:c_stop]
        cycle_spans[cid] = float(np.nanmax(cycle_field) - np.nanmin(cycle_field))
        cycle_points[cid] = int(c_stop - c_start)

    # Tie-breaking: max field span -> max points -> earliest cycle id.
    major_cycle = (
        max(
            cycle_spans,
            key=lambda cid: (cycle_spans[cid], cycle_points[cid], -cid),
        )
        if cycle_spans
        else 0
    )

    for branch in branches:
        branch.is_major = branch.cycle_id == major_cycle

    return branches


def find_zero_crossings(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Find x-values where y crosses zero using linear interpolation."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    out: list[float] = []

    if x_arr.size != y_arr.size or x_arr.size < 2:
        return out

    for i in range(x_arr.size - 1):
        x0, x1 = float(x_arr[i]), float(x_arr[i + 1])
        y0, y1 = float(y_arr[i]), float(y_arr[i + 1])

        if not np.isfinite(y0) or not np.isfinite(y1):
            continue

        if y0 == 0.0:
            out.append(x0)
            continue

        if y0 * y1 < 0.0:
            denom = (y1 - y0)
            if denom == 0.0:
                continue
            alpha = -y0 / denom
            out.append(x0 + alpha * (x1 - x0))

    return out


def interpolate_at_x(x: np.ndarray, y: np.ndarray, target_x: float = 0.0) -> float:
    """Interpolate y at target_x using adjacent samples, return NaN if unavailable."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size != y_arr.size or x_arr.size == 0:
        return float("nan")

    exact_idx = np.where(np.isclose(x_arr, float(target_x)))[0]
    if exact_idx.size:
        return float(y_arr[int(exact_idx[0])])

    for i in range(x_arr.size - 1):
        x0, x1 = float(x_arr[i]), float(x_arr[i + 1])
        y0, y1 = float(y_arr[i]), float(y_arr[i + 1])
        if (x0 <= target_x <= x1) or (x1 <= target_x <= x0):
            denom = (x1 - x0)
            if denom == 0:
                return float("nan")
            alpha = (target_x - x0) / denom
            return float(y0 + alpha * (y1 - y0))

    return float("nan")


def numerical_derivative(field: np.ndarray, magnetization: np.ndarray) -> np.ndarray:
    """Compute robust dM/dB derivative."""
    field_arr = np.asarray(field, dtype=float).reshape(-1)
    mag_arr = np.asarray(magnetization, dtype=float).reshape(-1)

    if field_arr.size < 2:
        return np.zeros_like(field_arr)

    # Avoid repeated/near-repeated coordinates causing gradient singularities.
    # Use a scale-aware epsilon instead of a fixed machine-level value.
    span = float(np.nanmax(field_arr) - np.nanmin(field_arr))
    eps = max(1e-12, span * 1e-9)
    adjusted_field = field_arr.copy()
    repeated = np.where(np.abs(np.diff(adjusted_field)) <= eps)[0]
    for idx in repeated:
        adjusted_field[idx + 1] += eps

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        derivative = np.gradient(mag_arr, adjusted_field)
    return np.nan_to_num(derivative, nan=0.0, posinf=0.0, neginf=0.0)
