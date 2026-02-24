"""Shared infrastructure helpers reused across analysis modules."""

from .cache import DEFAULT_MANIFEST_ATTR, InMemoryResultCache, build_cache_key
from .registry import (
    get_registered_metric,
    get_registry,
    iter_registered_metrics,
    register_metric,
)
from .repr_html import make_simple_card

__all__ = [
    "DEFAULT_MANIFEST_ATTR",
    "InMemoryResultCache",
    "build_cache_key",
    "register_metric",
    "get_registered_metric",
    "iter_registered_metrics",
    "get_registry",
    "make_simple_card",
]
