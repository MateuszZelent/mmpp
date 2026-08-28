"""Shared FFT filter infrastructure.

This package centralizes preprocessing and postprocessing filters used by FFT
modules (spectrum, modes, dispersion).
"""

from .config import FilterConfig, PostprocessConfig, PreprocessConfig
from .pipeline import (
    LIVE_FILTER_KEYS,
    POSTPROCESS_FILTER_KEYS,
    PREPROCESS_FILTER_KEYS,
    FilterPipeline,
    classify_filter_execution,
    normalize_filter_config,
    split_filter_stages,
)

__all__ = [
    "FilterConfig",
    "PreprocessConfig",
    "PostprocessConfig",
    "FilterPipeline",
    "PREPROCESS_FILTER_KEYS",
    "POSTPROCESS_FILTER_KEYS",
    "LIVE_FILTER_KEYS",
    "normalize_filter_config",
    "split_filter_stages",
    "classify_filter_execution",
]
