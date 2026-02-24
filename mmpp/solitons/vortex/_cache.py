"""Vortex cache wrappers built on shared cache infrastructure."""

from __future__ import annotations

from ..._shared.cache import InMemoryResultCache as _SharedInMemoryResultCache
from ..._shared.cache import build_cache_key as _shared_build_cache_key

MODULE_VERSION = "0.2.0"
_MANIFEST_ATTR = "_mmpp_solitons_cache_manifest"


def build_cache_key(
    method: str,
    *,
    config_payload: dict,
    namespace: str,
) -> tuple[str, str]:
    """Build deterministic cache key for vortex module payloads."""
    return _shared_build_cache_key(
        namespace=str(namespace),
        method=str(method),
        config_payload=config_payload,
        module_version=MODULE_VERSION,
    )


class InMemoryResultCache(_SharedInMemoryResultCache):
    """Backward-compatible cache class using legacy manifest attr name."""

    def __init__(self, job_result=None, *, namespace: str | None = None):
        super().__init__(
            job_result,
            namespace="default" if namespace is None else str(namespace),
            manifest_attr=_MANIFEST_ATTR,
        )


__all__ = ["InMemoryResultCache", "build_cache_key", "MODULE_VERSION"]
