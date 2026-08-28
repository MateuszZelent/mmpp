"""Backward-compatible shim for spectrum filter utilities.

The implementation moved to :mod:`mmpp.fft.filters` to avoid duplication
between modes, interactive explorers and dispersion modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..filters import (
    LIVE_FILTER_KEYS,
    POSTPROCESS_FILTER_KEYS,
    PREPROCESS_FILTER_KEYS,
    FilterPipeline,
    classify_filter_execution,
    normalize_filter_config,
    split_filter_stages,
)

# Cache schema version - bump when cached results are no longer compatible
SPECTRUM_CACHE_SCHEMA_VERSION = 1


def apply_spectrum_filters(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    filters: dict[str, Any] | None,
    stage: str = "post",
) -> np.ndarray:
    """Apply normalized filters to spectrum data.

    Parameters are preserved for backward compatibility with existing code that
    imports from ``mmpp.fft.modes.filter_utils``.
    """
    if filters is None:
        return np.asarray(spectrum)

    pipeline = FilterPipeline()
    if stage == "pre":
        return pipeline.preprocess(np.asarray(spectrum), filters=filters)
    if stage == "live":
        return pipeline.live(
            np.asarray(spectrum), np.asarray(frequencies), filters=filters
        )
    return pipeline.postprocess(
        np.asarray(spectrum), np.asarray(frequencies), filters=filters, stage="post"
    )


__all__ = [
    "SPECTRUM_CACHE_SCHEMA_VERSION",
    "PREPROCESS_FILTER_KEYS",
    "POSTPROCESS_FILTER_KEYS",
    "LIVE_FILTER_KEYS",
    "normalize_filter_config",
    "split_filter_stages",
    "classify_filter_execution",
    "apply_spectrum_filters",
]
