"""Shared cache key and in-memory cache helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

try:  # avoid import-cycle issues during package bootstrap
    from mmpp import __version__ as _MMPP_VERSION
except Exception:  # pragma: no cover - safe fallback during bootstrap
    _MMPP_VERSION = "unknown"

DEFAULT_MANIFEST_ATTR = "_mmpp_cache_manifest"


def _normalize_payload(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _normalize_payload(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {
            str(k): _normalize_payload(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "y_axis"):
        return {"y_axis": str(value.y_axis)}
    return str(value)


def build_cache_key(
    namespace: str,
    method: str,
    *,
    config_payload: dict[str, Any],
    module_version: str,
) -> tuple[str, str]:
    """Build deterministic cache key and normalized payload JSON."""
    payload = {
        "mmpp_version": _MMPP_VERSION,
        "module_version": str(module_version),
        **_normalize_payload(config_payload),
    }
    config_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:12]
    key = f"{namespace}_{method}_{digest}"
    return key, config_json


class InMemoryResultCache:
    """Process-local cache that never modifies simulation metadata."""

    def __init__(
        self,
        job_result=None,
        *,
        namespace: str = "default",
        manifest_attr: str = DEFAULT_MANIFEST_ATTR,
    ):
        self._store: dict[str, tuple[Any, str]] = {}
        self._job = job_result
        self._namespace = str(namespace)
        self._manifest_attr = str(manifest_attr)

    def has(self, key: str, config_json: str) -> bool:
        entry = self._store.get(str(key))
        if entry is None:
            return False

        _, stored_json = entry
        if stored_json != config_json:
            return False

        return True

    def get(self, key: str) -> Any:
        return self._store[str(key)][0]

    def put(self, key: str, data: Any, config_json: str) -> None:
        self._store[str(key)] = (data, config_json)

    def invalidate(self) -> None:
        self._store.clear()


__all__ = [
    "DEFAULT_MANIFEST_ATTR",
    "InMemoryResultCache",
    "build_cache_key",
]
