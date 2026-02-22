"""Plugin registry for user-defined hysteresis metrics."""

from __future__ import annotations

import logging
from collections.abc import Callable

_LOG = logging.getLogger(__name__)
_METRIC_REGISTRY: dict[str, Callable] = {}


def register_metric(name: str):
    """Register a custom metric callable under a public name."""

    metric_name = str(name).strip()
    if not metric_name:
        raise ValueError("Metric name must be non-empty")

    def decorator(func: Callable):
        if metric_name in _METRIC_REGISTRY:
            _LOG.info("Replacing existing hysteresis metric registration: %s", metric_name)
        _METRIC_REGISTRY[metric_name] = func
        return func

    return decorator


def get_registered_metric(name: str) -> Callable | None:
    """Return registered metric callable by name."""
    return _METRIC_REGISTRY.get(str(name))


def iter_registered_metrics() -> list[tuple[str, Callable]]:
    """Return sorted (name, callable) metric registry snapshot."""
    return sorted(_METRIC_REGISTRY.items(), key=lambda item: item[0])


__all__ = [
    "register_metric",
    "get_registered_metric",
    "iter_registered_metrics",
    "_METRIC_REGISTRY",
]
