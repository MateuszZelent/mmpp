"""Hysteresis metric registry backed by shared plugin registry."""

from __future__ import annotations

from collections.abc import Callable

from ...._shared.registry import (
    get_registered_metric as _get_registered_metric,
    get_registry,
    iter_registered_metrics as _iter_registered_metrics,
    register_metric as _register_metric,
)

_NAMESPACE = "hysteresis.metrics"
_METRIC_REGISTRY = get_registry(_NAMESPACE)


def register_metric(name: str):
    """Register custom metric callable under hysteresis namespace."""
    return _register_metric(name, namespace=_NAMESPACE)


def get_registered_metric(name: str) -> Callable | None:
    """Return registered metric callable by name."""
    return _get_registered_metric(name, namespace=_NAMESPACE)


def iter_registered_metrics() -> list[tuple[str, Callable]]:
    """Return sorted (name, callable) registry snapshot."""
    return _iter_registered_metrics(namespace=_NAMESPACE)


__all__ = [
    "register_metric",
    "get_registered_metric",
    "iter_registered_metrics",
    "_METRIC_REGISTRY",
]
