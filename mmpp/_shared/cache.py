"""Shared cache key and in-memory cache helpers."""

from __future__ import annotations

import hashlib
import json
import time
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
    """Process-local cache with optional manifest persisted in zarr attrs."""

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

    @staticmethod
    def _config_sha(config_json: str) -> str:
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def _read_manifest(self) -> dict[str, Any]:
        if self._job is None:
            return {}

        try:
            raw = self._job.z.attrs.get(self._manifest_attr, {})
        except Exception:
            return {}

        return raw if isinstance(raw, dict) else {}

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        if self._job is None:
            return
        try:
            self._job.z.attrs[self._manifest_attr] = manifest
        except Exception:
            # best effort only, in-memory cache still works in read-only stores
            return

    def _manifest_has(self, key: str, config_json: str) -> bool:
        manifest = self._read_manifest()
        namespace_data = manifest.get(self._namespace, {})
        entry = namespace_data.get(key)
        if not isinstance(entry, dict):
            return False
        return entry.get("config_sha256") == self._config_sha(config_json)

    def _manifest_put(self, key: str, config_json: str) -> None:
        manifest = self._read_manifest()
        namespace_data = manifest.get(self._namespace, {})
        namespace_data[key] = {
            "config_sha256": self._config_sha(config_json),
            "updated_unix_s": float(time.time()),
        }
        manifest[self._namespace] = namespace_data
        self._write_manifest(manifest)

    def has(self, key: str, config_json: str) -> bool:
        entry = self._store.get(str(key))
        if entry is None:
            return False

        _, stored_json = entry
        if stored_json != config_json:
            return False

        if self._job is None:
            return True
        return self._manifest_has(str(key), config_json)

    def get(self, key: str) -> Any:
        return self._store[str(key)][0]

    def put(self, key: str, data: Any, config_json: str) -> None:
        self._store[str(key)] = (data, config_json)
        self._manifest_put(str(key), config_json)

    def invalidate(self) -> None:
        self._store.clear()
        if self._job is None:
            return

        manifest = self._read_manifest()
        if self._namespace in manifest:
            del manifest[self._namespace]
            self._write_manifest(manifest)


__all__ = [
    "DEFAULT_MANIFEST_ATTR",
    "InMemoryResultCache",
    "build_cache_key",
]
