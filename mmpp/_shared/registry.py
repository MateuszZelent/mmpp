"""Shared plugin registry used by analysis modules."""

from __future__ import annotations

import logging
from collections.abc import Callable

_LOG = logging.getLogger(__name__)
_REGISTRIES: dict[str, dict[str, Callable]] = {}


def _normalize_namespace(namespace: str | None) -> str:
    ns = str(namespace or "default").strip()
    return ns if ns else "default"


def get_registry(namespace: str | None = None) -> dict[str, Callable]:
    """Return mutable registry storage for a namespace."""
    ns = _normalize_namespace(namespace)
    return _REGISTRIES.setdefault(ns, {})


def register_metric(name: str, *, namespace: str | None = None):
    """Decorator registering a callable metric under a public name."""
    metric_name = str(name).strip()
    if not metric_name:
        raise ValueError("Metric name must be non-empty")

    registry = get_registry(namespace)
    ns = _normalize_namespace(namespace)

    def decorator(func: Callable):
        if metric_name in registry:
            _LOG.info(
                "Replacing existing metric registration: namespace=%s name=%s",
                ns,
                metric_name,
            )
        registry[metric_name] = func
        return func

    return decorator


def get_registered_metric(name: str, *, namespace: str | None = None) -> Callable | None:
    """Return registered metric callable by name."""
    return get_registry(namespace).get(str(name))


def iter_registered_metrics(
    *,
    namespace: str | None = None,
) -> list[tuple[str, Callable]]:
    """Return sorted registry snapshot for a namespace."""
    registry = get_registry(namespace)
    return sorted(registry.items(), key=lambda item: item[0])


__all__ = [
    "get_registry",
    "register_metric",
    "get_registered_metric",
    "iter_registered_metrics",
]
