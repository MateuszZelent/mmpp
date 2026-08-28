"""Shared registry metadata for generic skyrmion analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SkyrmionAnalysisSpec:
    """One registry entry shared by single-result and batch dispatchers."""

    name: str
    aliases: tuple[str, ...]
    single_handler: str
    batch_handler: str
    default_value_column: str
    default_unit: str
    description: str


ANALYSIS_SPECS = {
    "size": SkyrmionAnalysisSpec(
        name="size",
        aliases=(),
        single_handler="_analyze_size",
        batch_handler="_analyze_size",
        default_value_column="diameter_nm",
        default_unit="nm",
        description=("Skyrmion radius, diameter, profile model, and fit diagnostics"),
    ),
    "charge": SkyrmionAnalysisSpec(
        name="charge",
        aliases=("topological_charge", "topology", "q"),
        single_handler="_analyze_charge",
        batch_handler="_analyze_charge",
        default_value_column="Q",
        default_unit="1",
        description="Integrated topological charge Q and topology diagnostics",
    ),
}
_ANALYSIS_ALIASES = {
    alias: spec.name
    for spec in ANALYSIS_SPECS.values()
    for alias in (spec.name, *spec.aliases)
}
SIZE_METRIC_UNITS = {
    "radius_m": "m",
    "diameter_m": "m",
    "radius_nm": "nm",
    "diameter_nm": "nm",
    "wall_width_m": "m",
    "scale_m": "m",
    "sigma_m": "m",
}


def normalize_analysis(observable: str) -> str:
    """Resolve a public observable name to its canonical registry key."""
    normalized = str(observable).strip().casefold().replace("-", "_")
    canonical = _ANALYSIS_ALIASES.get(normalized)
    if canonical is None:
        available = ", ".join(repr(name) for name in ANALYSIS_SPECS)
        raise ValueError(
            f"Unknown skyrmion analysis {observable!r}. "
            f"Available analyses: {available}."
        )
    return canonical


def get_analysis_spec(observable: str) -> SkyrmionAnalysisSpec:
    """Return the registry specification for a name or accepted alias."""
    return ANALYSIS_SPECS[normalize_analysis(observable)]


def analysis_catalog_rows() -> list[dict[str, Any]]:
    """Return serializable rows describing registered analyses and aliases."""
    rows: list[dict[str, Any]] = []
    for spec in ANALYSIS_SPECS.values():
        rows.append(
            {
                "analysis": spec.name,
                "aliases": spec.aliases,
                "default_value_column": spec.default_value_column,
                "default_unit": spec.default_unit,
                "description": spec.description,
            }
        )
    return rows


__all__ = [
    "ANALYSIS_SPECS",
    "SIZE_METRIC_UNITS",
    "SkyrmionAnalysisSpec",
    "analysis_catalog_rows",
    "get_analysis_spec",
    "normalize_analysis",
]
