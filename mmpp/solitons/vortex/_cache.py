"""Lightweight cache helpers for vortex analysis results."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any

from mmpp import __version__ as MMPP_VERSION

MODULE_VERSION = "0.1.0"


def _normalize_payload(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _normalize_payload(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _normalize_payload(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "y_axis"):
        return {"y_axis": str(value.y_axis)}
    return str(value)


def build_cache_key(
    method: str,
    *,
    config_payload: dict[str, Any],
    namespace: str,
) -> tuple[str, str]:
    """Build deterministic cache key and JSON payload."""
    payload = {
        "mmpp_version": MMPP_VERSION,
        "module_version": MODULE_VERSION,
        **_normalize_payload(config_payload),
    }
    config_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:12]
    key = f"{namespace}:{method}_{digest}"
    return key, config_json


class InMemoryResultCache:
    """Process-local cache with optional zarr-backed config metadata manifest."""

    _MANIFEST_ATTR = "_mmpp_solitons_cache_manifest"

    def __init__(self, job_result=None, *, namespace: str | None = None):
        self._store: dict[str, tuple[Any, str]] = {}
        self._job = job_result
        self._namespace = namespace or "default"

    def _config_sha(self, config_json: str) -> str:
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def _read_manifest(self) -> dict[str, Any]:
        if self._job is None:
            return {}
        try:
            raw = self._job.z.attrs.get(self._MANIFEST_ATTR, {})
            if isinstance(raw, dict):
                return raw
        except Exception:
            pass
        return {}

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        if self._job is None:
            return
        try:
            self._job.z.attrs[self._MANIFEST_ATTR] = manifest
        except Exception:
            # Best-effort persistence. In read-only stores we still use in-memory cache.
            return

    def _manifest_has(self, key: str, config_json: str) -> bool:
        manifest = self._read_manifest()
        by_ns = manifest.get(self._namespace, {})
        entry = by_ns.get(key)
        if not isinstance(entry, dict):
            return False
        return entry.get("config_sha256") == self._config_sha(config_json)

    def _manifest_put(self, key: str, config_json: str) -> None:
        manifest = self._read_manifest()
        by_ns = manifest.get(self._namespace, {})
        by_ns[key] = {
            "config_sha256": self._config_sha(config_json),
            "updated_unix_s": float(time.time()),
        }
        manifest[self._namespace] = by_ns
        self._write_manifest(manifest)

    def has(self, key: str, config_json: str) -> bool:
        item = self._store.get(key)
        if item is None:
            return False
        _, stored_json = item
        if stored_json != config_json:
            return False
        return self._manifest_has(key, config_json) if self._job is not None else True

    def get(self, key: str) -> Any:
        return self._store[key][0]

    def put(self, key: str, data: Any, config_json: str) -> None:
        self._store[key] = (data, config_json)
        self._manifest_put(key, config_json)

    def invalidate(self) -> None:
        self._store.clear()
        if self._job is None:
            return

        manifest = self._read_manifest()
        if self._namespace in manifest:
            del manifest[self._namespace]
            self._write_manifest(manifest)


__all__ = ["InMemoryResultCache", "build_cache_key", "MODULE_VERSION"]
