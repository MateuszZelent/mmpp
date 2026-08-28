"""Frequency-unit helpers for the interactive dispersion viewer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def frequency_units_to_ghz_factor(f_units: Any = "GHz") -> float:
    """Return multiplier converting values in *f_units* to GHz."""
    units = str(f_units or "GHz").strip().lower()
    if units in {"ghz", "gigahertz"}:
        return 1.0
    if units in {"hz", "hertz"}:
        return 1e-9
    raise ValueError("f_units must be 'GHz' or 'Hz'")


def frequency_axis_limits_ghz(
    f_axis: Iterable[Any] | None,
    *,
    positive_frequencies: bool = True,
) -> tuple[float, float]:
    """Return display defaults from an Hz frequency axis, expressed in GHz."""
    if f_axis is None:
        return 0.0, 1.0

    values: list[float] = []
    for value in f_axis:
        frequency_hz = float(value)
        if positive_frequencies and frequency_hz < 0.0:
            continue
        values.append(frequency_hz / 1e9)

    if not values:
        return 0.0, 1.0
    return min(values), max(values)


def normalize_frequency_limit_to_ghz(
    value: Any,
    *,
    default_ghz: float,
    f_units: Any = "GHz",
) -> float:
    """Normalize one optional frequency limit to GHz."""
    if value is None:
        return float(default_ghz)
    return float(value) * frequency_units_to_ghz_factor(f_units)


def normalize_frequency_window_ghz(
    options: dict[str, Any],
    f_axis: Iterable[Any] | None,
) -> tuple[float, float]:
    """Normalize interactive ``fmin``/``fmax`` options to GHz state values."""
    positive = bool(options.get("positive_frequencies", True))
    default_fmin, default_fmax = frequency_axis_limits_ghz(
        f_axis,
        positive_frequencies=positive,
    )
    f_units = options.get("f_units", "GHz")
    fmin = normalize_frequency_limit_to_ghz(
        options.get("fmin"),
        default_ghz=default_fmin,
        f_units=f_units,
    )
    fmax = normalize_frequency_limit_to_ghz(
        options.get("fmax"),
        default_ghz=default_fmax or 1.0,
        f_units=f_units,
    )
    return fmin, fmax
