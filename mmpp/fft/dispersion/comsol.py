"""Utilities for loading COMSOL dispersion data."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ComsolDispersionData:
    """Container for columns selected from a COMSOL export file."""

    k_values: np.ndarray
    f_values: np.ndarray
    columns: dict[int, np.ndarray]
    metadata: list[str]


def _iter_numeric_lines(
    path: Path,
    comment_markers: Sequence[str],
    skip_blank: bool,
) -> tuple[list[str], list[str]]:
    markers = tuple(comment_markers)
    metadata: list[str] = []
    numeric_lines: list[str] = []

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                if skip_blank:
                    continue
                numeric_lines.append(raw)
                continue
            if stripped.startswith(markers):
                metadata.append(stripped)
                continue
            numeric_lines.append(raw)

    return metadata, numeric_lines


def _tokenise_lines(lines: Iterable[str]) -> list[list[str]]:
    matrix: list[list[str]] = []
    for line in lines:
        tokens = line.strip().split()
        if tokens:
            matrix.append(tokens)
    return matrix


def _parse_token(token: str, *, return_complex: bool = False) -> float | complex:
    token = token.strip()
    if not token or token.lower() in {"nan", "na"}:
        return float("nan")

    token = token.replace(",", "")

    # COMSOL uses trailing 'i' for imaginary component
    if "i" in token and token.lower() not in {"inf", "-inf"}:
        candidate = token.replace("i", "j")
        try:
            value = complex(candidate)
            return value if return_complex else value.real
        except ValueError:
            pass

    try:
        return float(token)
    except ValueError:
        # Retry with lowercase exponent
        try:
            return float(token.replace("E", "e"))
        except ValueError:
            logger.debug("Unable to parse COMSOL token '%s'; returning NaN", token)
            return float("nan")


def _column_to_array(
    matrix: list[list[str]], idx: int, *, return_complex: bool = False
) -> np.ndarray:
    values: list[float | complex] = []
    for row in matrix:
        if idx >= len(row):
            values.append(float("nan"))
        else:
            values.append(_parse_token(row[idx], return_complex=return_complex))
    dtype = complex if return_complex else float
    return np.asarray(values, dtype=dtype)


def read_data_from_comsol(
    file_path: str | Path,
    *,
    delimiter: str | None = None,  # retained for API compatibility (unused)
    comment_markers: Sequence[str] = ("#", "%"),
    skip_blank: bool = True,
    k_col: int = 0,
    f_col: int = 1,
    extra_cols: Sequence[int] | None = None,
    extra_as_complex: Sequence[int] | None = None,
) -> ComsolDispersionData:
    """Read a COMSOL export file with header metadata and numeric columns.

    Parameters
    ----------
    file_path : str or Path
        Path to COMSOL text file.
    delimiter : str, optional
        Kept for backwards compatibility but ignored; whitespace splitting is used.
    comment_markers : sequence of str, optional
        Strings that tag metadata/header rows to skip as comments.
    skip_blank : bool, default True
        If True, blank lines are ignored before parsing numeric data.
    k_col, f_col : int, optional
        Column indices (0-based) containing k-vector and frequency data.
    extra_cols : sequence of int, optional
        Additional column indices to load into the ``columns`` dictionary.
    extra_as_complex : sequence of int, optional
        Subset of ``extra_cols`` that should be stored as complex values.

    Returns
    -------
    ComsolDispersionData
        Parsed data including metadata lines.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"COMSOL data file not found: {path}")

    metadata, numeric_lines = _iter_numeric_lines(path, comment_markers, skip_blank)
    if not numeric_lines:
        raise ValueError(f"No numeric data found in COMSOL file: {path}")

    matrix = _tokenise_lines(numeric_lines)
    if not matrix:
        raise ValueError(f"Unable to parse numeric data from COMSOL file: {path}")

    max_cols = max(len(row) for row in matrix)
    for idx in (k_col, f_col, *(extra_cols or [])):
        if idx < 0 or idx >= max_cols:
            raise IndexError(f"Requested column {idx} outside range 0..{max_cols - 1}")

    k_values = _column_to_array(matrix, k_col, return_complex=False)
    f_values = _column_to_array(matrix, f_col, return_complex=False)

    extras: dict[int, np.ndarray] = {}
    extra_as_complex = tuple(extra_as_complex or [])
    if extra_cols:
        for idx in extra_cols:
            extras[idx] = _column_to_array(
                matrix,
                idx,
                return_complex=idx in extra_as_complex,
            )

    logger.info(
        "Loaded %d COMSOL dispersion rows from %s (k_col=%d, f_col=%d)",
        len(k_values),
        path,
        k_col,
        f_col,
    )

    for label, array in (("k", k_values), ("f", f_values)):
        nan_count = int(np.isnan(np.asarray(array)).sum())
        if nan_count:
            logger.warning(
                "COMSOL column '%s' contains %d NaN entries", label, nan_count
            )

    return ComsolDispersionData(
        k_values=np.asarray(k_values, dtype=float),
        f_values=np.asarray(f_values, dtype=float),
        columns=extras,
        metadata=metadata,
    )
